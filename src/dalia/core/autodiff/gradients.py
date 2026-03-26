# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Analytical gradient routines for the INLA objective.

The INLA objective f(theta) decomposes into independently differentiable
terms.  By the envelope theorem, x* satisfies Q_c x* = r, so the
implicit dependence of x* on theta contributes nothing to the total
derivative and drops out.  The gradient then splits into three phases:

    df/d(theta_k) = Phase_A + Phase_B + Phase_C + prior/likelihood terms

**Phase A** — Selected inversion gradient for -1/2 log|Q_c|:
    -1/2 tr(Q_c^{-1} dQ_c/d(theta_k))
    Implemented as a fused backward sweep of selected inversion + trace
    accumulation.  See :func:`selected_inversion_grads_jax` (full-factor)
    and :func:`cholesky_carries.selected_inversion_grads_from_carries_jax`
    (carry-based).

**Phase B** — Quadratic form gradient for -1/2 x*^T Q_p x*:
    -1/2 x*^T (dQ_p/d(theta_k)) x*
    Since x* is treated as constant at the mode, differentiation passes
    through directly to the BT blocks of Q_p.  See :func:`_compute_grad_quad`.

**Phase C** — Prior log-determinant gradient for +1/2 log|Q_p|:
    +1/2 tr(Q_p^{-1} dQ_p/d(theta_k))
    Same structure as Phase A but for the prior precision Q_p (BT, no
    arrowhead).  See :func:`bt_logdet_grad`.

Functions
---------
bt_logdet_grad
    Phase C: analytical gradient of log|Q_p| via BT selected inversion.
_spatial_traces
    Helper: compute tr(Z_block @ spatial_matrix) for all blocks.
_compute_grad_logdet_cond
    Phase A (full-factor): gradient of log|Q_c| from stored SI entries.
_compute_grad_quad
    Phase B: gradient of the quadratic form x*^T Q_p x*.
selected_inversion_grads_jax
    Phase A (full-factor): fused SI + gradient accumulation.
"""

import jax.numpy as jnp
from jax import lax

from dalia.core.autodiff.spatial_precompute import (
    _jax_cholesky,
    precompute_spatial_components,
    _reconstruct_diag_block,
    _reconstruct_lower_block,
)

import jax
import jax.scipy.linalg


def bt_logdet_grad(
    theta_st, spatial_matrices, temporal_matrices, manifold, nt, ns, n_theta_st, dtype
):
    """Phase C: analytical gradient of log|Q_p| via BT selected inversion.

    Computes d(log|Q_p|)/d(theta_st) for the prior precision Q_p, which
    has block-tridiagonal (BT) structure (no arrowhead).  This replaces
    ``jax.grad(logdet_Q_st_scan)`` with a hand-derived computation that
    avoids the AD tape entirely.

    The algorithm mirrors Phase A but is simpler (no arrowhead):

    1. **Forward BT Cholesky** storing incoming Schur carries
       S_i = L_B_i L_B_i^T (n blocks of (ns, ns)).
    2. **Backward BT selected inversion** reconstructing L_D_i and
       L_B_i from stored carries at each step, computing the SI
       entries Z_D_i and Z_B_i, and immediately accumulating::

           d(log|Q_p|)/d(theta_k) += tr(Z_D_i dD_i^p/d(theta_k))
                                   + 2 tr(Z_B_i^T dB_i^p/d(theta_k))

       Only one Z_D block is live at a time.

    Parameters
    ----------
    theta_st : (n_theta_st,)
    spatial_matrices, temporal_matrices : dict
    manifold : str
    nt, ns, n_theta_st : int
    dtype : jnp.dtype

    Returns
    -------
    grad_logdet_st : (n_theta_st,)
        d(logdet Q_st) / d(theta_st)
    """
    sc = precompute_spatial_components(theta_st, spatial_matrices, temporal_matrices, manifold)
    jac_sc = jax.jacfwd(precompute_spatial_components)(
        theta_st, spatial_matrices, temporal_matrices, manifold
    )

    eye_ns = jnp.eye(ns, dtype=dtype)
    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    # Pad subdiagonal to length nt
    m0_s_pad = jnp.concatenate([m0_s, jnp.zeros(1, dtype=dtype)])
    m1_s_pad = jnp.concatenate([m1_s, jnp.zeros(1, dtype=dtype)])
    m2_s_pad = jnp.concatenate([m2_s, jnp.zeros(1, dtype=dtype)])

    sc_pad = {**sc,
              'm0_subdiag': m0_s_pad,
              'm1_subdiag': m1_s_pad,
              'm2_subdiag': m2_s_pad}

    # --- 1. Forward BT Cholesky scan, store incoming Schur as outputs ---
    def fwd_body(schur, i):
        q_diag = _reconstruct_diag_block(sc_pad, i) - schur
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_lower_block(sc_pad, i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        new_schur = L_lower_i @ L_lower_i.T
        new_schur = jnp.where(i < nt - 1, new_schur, jnp.zeros_like(new_schur))
        return new_schur, schur  # output = incoming Schur for reconstruction

    init_schur = jnp.zeros((ns, ns), dtype=dtype)
    _, stored_schurs = lax.scan(fwd_body, init_schur, jnp.arange(nt))
    # stored_schurs[i] = Schur complement subtracted from Q_st_diag[i]

    # --- Helper: reconstruct L_diag[i] and L_lower[i] from stored carry ---
    def _reconstruct_L(i):
        q_diag = _reconstruct_diag_block(sc_pad, i) - stored_schurs[i]
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_lower_block(sc_pad, i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        return L_i, L_lower_i

    # --- Trace target matrices ---
    base_mats = [sc['q3s'], sc['q2s'], sc['q1s']]
    jac_mats = []
    for k in range(n_theta_st):
        jac_mats.extend([
            jac_sc['q3s'][..., k], jac_sc['q2s'][..., k], jac_sc['q1s'][..., k],
        ])
    all_mats = jnp.stack(base_mats + jac_mats, axis=0)
    n_mats = all_mats.shape[0]

    # --- 2. Last block (i = nt-1) ---
    L_last, _ = _reconstruct_L(nt - 1)
    L_inv_last = jax.scipy.linalg.solve_triangular(L_last, eye_ns, lower=True)
    sd_last = L_inv_last.T @ L_inv_last

    tr_d_last = jnp.einsum('ij,sji->s', sd_last, all_mats)
    mw_last = jnp.tile(
        jnp.array([m0_d[nt - 1], m1_d[nt - 1], m2_d[nt - 1]], dtype=dtype),
        1 + n_theta_st)
    acc_d = mw_last * tr_d_last
    acc_l = jnp.zeros(n_mats, dtype=dtype)

    # --- 3. Backward BT selected inversion + trace accumulation ---
    def bwd_body(i_rev, carry):
        sd_prev, acc_d_, acc_l_ = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i = _reconstruct_L(i)
        L_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_ns, lower=True)

        sl_i = -sd_prev @ L_lower_i @ L_inv_i
        sd_i = (L_inv_i.T - sl_i.T @ L_lower_i) @ L_inv_i

        tr_d = jnp.einsum('ij,sji->s', sd_i, all_mats)
        mw_d = jnp.tile(
            jnp.array([m0_d[i], m1_d[i], m2_d[i]], dtype=dtype), 1 + n_theta_st)
        acc_d_ = acc_d_ + mw_d * tr_d

        tr_l = jnp.einsum('ij,sji->s', sl_i, all_mats)
        mw_l = jnp.tile(
            jnp.array([m0_s[i], m1_s[i], m2_s[i]], dtype=dtype), 1 + n_theta_st)
        acc_l_ = acc_l_ + mw_l * tr_l

        return (sd_i, acc_d_, acc_l_)

    _, acc_d, acc_l = lax.fori_loop(
        0, nt - 1, bwd_body, (sd_last, acc_d, acc_l))

    # Factor of 2: lower block S_l[t] appears twice (lower + upper transpose)
    acc_l = 2.0 * acc_l

    # --- 4. Assemble gradient ---
    jac_scale = jac_sc['scale']
    jac_exp_gt = jac_sc['exp_gt']

    weighted_diag = acc_d[0] + exp_gt * acc_d[1] + exp_gt**2 * acc_d[2]
    weighted_lower = acc_l[0] + exp_gt * acc_l[1] + exp_gt**2 * acc_l[2]

    grad_st = jnp.zeros(n_theta_st, dtype=dtype)
    for k in range(n_theta_st):
        term_scale = jac_scale[k] * (weighted_diag + weighted_lower)

        off = 3 + 3 * k
        term_spatial_diag = scale * (
            acc_d[off] + exp_gt * acc_d[off + 1] + exp_gt**2 * acc_d[off + 2])
        term_spatial_lower = scale * (
            acc_l[off] + exp_gt * acc_l[off + 1] + exp_gt**2 * acc_l[off + 2])

        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            acc_d[1] + 2.0 * exp_gt * acc_d[2])
        term_exp_gt_lower = scale * jac_exp_gt[k] * (
            acc_l[1] + 2.0 * exp_gt * acc_l[2])

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    return grad_st


def _spatial_traces(S_diag, S_lower, sc, nt):
    """Compute traces tr(Sigma_block @ spatial_matrix) for all blocks.

    For diagonal blocks:
        tr_diag[i, j] = tr(S_diag[i] @ Sj)  for Sj in {q3s, q2s, q1s}

    For lower-diagonal blocks:
        tr_lower[i, j] = tr(S_lower[i] @ Sj)

    Parameters
    ----------
    S_diag : (nt, ns, ns)
    S_lower : (nt-1, ns, ns)
    sc : dict from precompute_spatial_components
    nt : int

    Returns
    -------
    tr_diag : (nt, 3)  traces with [q3s, q2s, q1s]
    tr_lower : (nt-1, 3)
    """
    spatial_mats = jnp.stack([sc['q3s'], sc['q2s'], sc['q1s']], axis=0)  # (3, ns, ns)

    # tr(A @ B) = sum(A * B^T) = einsum('ij,ji->')
    # Vectorized over blocks and spatial matrices:
    # S_diag: (nt, ns, ns), spatial_mats: (3, ns, ns)
    tr_diag = jnp.einsum('bij,sji->bs', S_diag, spatial_mats)  # (nt, 3)
    tr_lower = jnp.einsum('bij,sji->bs', S_lower, spatial_mats)  # (nt-1, 3)

    return tr_diag, tr_lower


def _compute_grad_logdet_cond(
    S_diag, S_lower, S_arrow, S_tip,
    sc, jac_sc,
    nt, ns, n_fe, n_theta_st,
    likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip,
):
    """Phase A (full-factor): gradient of log|Q_c| from stored SI entries.

    Given the full selected inverse entries (Z_D, Z_B, Z_C, Z_T) from
    :func:`pobtasi_jax`, computes the log-determinant gradients via::

        d(log|Q_c|)/d(theta_k) = sum_i tr(Z_D_i dD_i/dtheta_k)
                                + 2 sum_i tr(Z_B_i^T dB_i/dtheta_k)
                                + 2 sum_i tr(Z_C_i^T dC_i/dtheta_k)
                                + tr(Z_T dT/dtheta_k)

    For spatio-temporal hyperparameters theta_st, only the prior part of
    Q_c contributes (the A^T A term is independent of theta_st).
    For the likelihood precision theta_lik, dQ_c/d(theta_lik) = tau * A^T A.

    Parameters
    ----------
    S_diag, S_lower, S_arrow, S_tip : Sigma factors from selected inversion.
    sc : dict from precompute_spatial_components.
    jac_sc : dict of Jacobians of sc w.r.t. theta_st (each value has leading dim n_theta_st).
    nt, ns, n_fe, n_theta_st : int
    likelihood_prec : scalar
    ata_* : sparse COO data for AtA blocks.
    ata_tip : (n_fe, n_fe) dense AtA tip.

    Returns
    -------
    grad_st : (n_theta_st,) gradient w.r.t. theta_st
    grad_lik : scalar gradient w.r.t. theta_lik
    """
    # --- Gradient w.r.t. theta_st ---
    # Q_st_diag[i] = scale * (m0_d[i]*q3s + exp_gt*m1_d[i]*q2s + exp_gt^2*m2_d[i]*q1s)
    # dQ_st_diag[i]/d(theta_k) = d_scale_k * (...) + scale * (m0_d[i]*d_q3s_k + d_exp_gt_k*m1_d[i]*q2s + ...)
    # tr(Sigma_diag[i] @ dQ_st_diag[i]/d(theta_k)) can be reduced to sums of precomputed spatial traces.

    tr_diag, tr_lower = _spatial_traces(S_diag, S_lower, sc, nt)
    # tr_diag[:, 0] = tr(S_diag[i] @ q3s), [:, 1] = ... @ q2s, [:, 2] = ... @ q1s

    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    # Temporal coefficients for diagonal blocks (nt,)
    coeff_q3s_diag = m0_d                          # coefficient of q3s in diag block
    coeff_q2s_diag = exp_gt * m1_d                  # coefficient of q2s
    coeff_q1s_diag = exp_gt**2 * m2_d               # coefficient of q1s

    # For lower blocks (nt-1,)
    coeff_q3s_lower = m0_s
    coeff_q2s_lower = exp_gt * m1_s
    coeff_q1s_lower = exp_gt**2 * m2_s

    # Weighted spatial traces: sum over blocks of temporal_coeff[i] * tr_spatial[i, j]
    # This gives tr(Sigma @ scale * temporal_term_j) for each spatial matrix j
    # diag_contrib[j] = sum_i coeff_j_diag[i] * tr_diag[i, j]
    weighted_diag = (
        jnp.sum(coeff_q3s_diag * tr_diag[:, 0])
        + jnp.sum(coeff_q2s_diag * tr_diag[:, 1])
        + jnp.sum(coeff_q1s_diag * tr_diag[:, 2])
    )
    # Factor of 2 on lower: each S_l[t] counts for both lower and upper blocks
    weighted_lower = 2.0 * (
        jnp.sum(coeff_q3s_lower * tr_lower[:, 0])
        + jnp.sum(coeff_q2s_lower * tr_lower[:, 1])
        + jnp.sum(coeff_q1s_lower * tr_lower[:, 2])
    )

    # Jacobians of spatial components w.r.t. theta_st.
    # jacfwd puts the input dimension last: q3s has shape (ns, ns, n_theta_st), etc.
    jac_q3s = jac_sc['q3s']   # (ns, ns, n_theta_st)
    jac_q2s = jac_sc['q2s']
    jac_q1s = jac_sc['q1s']
    jac_scale = jac_sc['scale']   # (n_theta_st,)
    jac_exp_gt = jac_sc['exp_gt']  # (n_theta_st,)

    grad_st = jnp.zeros(n_theta_st, dtype=S_diag.dtype)

    for k in range(n_theta_st):
        dq3s_k = jac_q3s[..., k]  # (ns, ns)
        dq2s_k = jac_q2s[..., k]
        dq1s_k = jac_q1s[..., k]

        # Term 1: d_scale[k] * (original sum of traces)
        term_scale = jac_scale[k] * (weighted_diag + weighted_lower)

        # Term 2: scale * traces with Jacobian spatial matrices
        tr_S_dq3s = jnp.einsum('bij,ji->b', S_diag, dq3s_k)  # (nt,)
        tr_S_dq2s = jnp.einsum('bij,ji->b', S_diag, dq2s_k)
        tr_S_dq1s = jnp.einsum('bij,ji->b', S_diag, dq1s_k)

        tr_Sl_dq3s = jnp.einsum('bij,ji->b', S_lower, dq3s_k)  # (nt-1,)
        tr_Sl_dq2s = jnp.einsum('bij,ji->b', S_lower, dq2s_k)
        tr_Sl_dq1s = jnp.einsum('bij,ji->b', S_lower, dq1s_k)

        term_spatial_diag = scale * (
            jnp.sum(m0_d * tr_S_dq3s)
            + exp_gt * jnp.sum(m1_d * tr_S_dq2s)
            + exp_gt**2 * jnp.sum(m2_d * tr_S_dq1s)
        )
        term_spatial_lower = 2.0 * scale * (
            jnp.sum(m0_s * tr_Sl_dq3s)
            + exp_gt * jnp.sum(m1_s * tr_Sl_dq2s)
            + exp_gt**2 * jnp.sum(m2_s * tr_Sl_dq1s)
        )

        # Term 3: exp_gt derivative contributions
        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            jnp.sum(m1_d * tr_diag[:, 1])
            + 2.0 * exp_gt * jnp.sum(m2_d * tr_diag[:, 2])
        )
        term_exp_gt_lower = 2.0 * scale * jac_exp_gt[k] * (
            jnp.sum(m1_s * tr_lower[:, 1])
            + 2.0 * exp_gt * jnp.sum(m2_s * tr_lower[:, 2])
        )

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    # --- Gradient w.r.t. theta_lik ---
    # dQ_cond/d(theta_lik) = lik_prec * AtA (since Q_cond = Q_st + lik_prec * AtA)
    # tr(Sigma @ lik_prec * AtA) = lik_prec * tr(Sigma @ AtA)
    # Compute sparse trace using COO data:
    # tr(Sigma_diag[i] @ AtA_diag[i]) = sum_j Sigma_diag[i, rows[i,j], cols[i,j]] * vals[i,j]
    n_blocks_diag = ata_diag_rows.shape[0]
    block_idx_d = jnp.arange(n_blocks_diag)[:, None]
    sparse_tr_diag = jnp.sum(S_diag[block_idx_d, ata_diag_rows, ata_diag_cols] * ata_diag_vals)

    n_blocks_lower = ata_lower_rows.shape[0]
    block_idx_l = jnp.arange(n_blocks_lower)[:, None]
    sparse_tr_lower = jnp.sum(S_lower[block_idx_l, ata_lower_rows, ata_lower_cols] * ata_lower_vals)

    n_blocks_arrow = ata_arrow_rows.shape[0]
    block_idx_a = jnp.arange(n_blocks_arrow)[:, None]
    sparse_tr_arrow = jnp.sum(S_arrow[block_idx_a, ata_arrow_rows, ata_arrow_cols] * ata_arrow_vals)

    sparse_tr_tip = jnp.sum(S_tip * ata_tip)

    # Lower blocks contribute twice (symmetry: tr(Sigma @ Q) counts both off-diag blocks)
    grad_lik = likelihood_prec * (sparse_tr_diag + 2.0 * sparse_tr_lower + 2.0 * sparse_tr_arrow + sparse_tr_tip)

    return grad_st, grad_lik


def _compute_grad_quad(
    x, sc, jac_sc,
    nt, ns, n_fe, n_theta_st,
    rhs, likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip,
):
    """Phase B: gradient of the quadratic form x*^T Q_p x*.

    The quadratic form x*^T Q_p x* expands via the BT block structure
    of Q_p as::

        x*^T Q_p x* = sum_i x*_i^T D_i^p x*_i
                     + 2 sum_i x*_i^T B_i^{p,T} x*_{i+1}

    By the envelope theorem, x* is treated as fixed at the posterior
    mode, so differentiation passes directly to the blocks::

        d(x*^T Q_p x*)/d(theta_k) = sum_i x*_i^T (dD_i^p/dtheta_k) x*_i
                                   + 2 sum_i x*_i^T (dB_i^p/dtheta_k)^T x*_{i+1}

    For theta_lik, the derivative of the full quadratic form
    r^T Q_c^{-1} r also includes the A^T A contribution.

    Parameters
    ----------
    x : (nt*ns + n_fe,) solution vector
    sc, jac_sc : spatial components and their Jacobians
    nt, ns, n_fe, n_theta_st : int
    rhs : (nt*ns + n_fe,) right-hand side vector
    likelihood_prec : scalar
    ata_* : sparse COO data

    Returns
    -------
    grad_st : (n_theta_st,) gradient w.r.t. theta_st
    grad_lik : scalar gradient w.r.t. theta_lik
    """
    x_st = x[:nt * ns].reshape(nt, ns)  # (nt, ns)

    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    # Precompute x^T S_j x for each spatial matrix, for each block
    spatial_mats = jnp.stack([sc['q3s'], sc['q2s'], sc['q1s']], axis=0)  # (3, ns, ns)
    # x_st[i]^T @ S_j @ x_st[i] = einsum('j,jk,k->', x_st[i], S_j, x_st[i])
    # Vectorized: (nt, 3)
    xSx_diag = jnp.einsum('bi,sij,bj->bs', x_st, spatial_mats, x_st)  # (nt, 3)

    # Lower blocks: x_st[i+1]^T @ S_j @ x_st[i]
    xSx_lower = jnp.einsum('bi,sij,bj->bs', x_st[1:], spatial_mats, x_st[:-1])  # (nt-1, 3)

    # jacfwd puts input dim last: q3s is (ns, ns, n_theta_st)
    jac_q3s = jac_sc['q3s']   # (ns, ns, n_theta_st)
    jac_q2s = jac_sc['q2s']
    jac_q1s = jac_sc['q1s']
    jac_scale = jac_sc['scale']   # (n_theta_st,)
    jac_exp_gt = jac_sc['exp_gt']  # (n_theta_st,)

    coeff_q3s_diag = m0_d
    coeff_q2s_diag = exp_gt * m1_d
    coeff_q1s_diag = exp_gt**2 * m2_d
    coeff_q3s_lower = m0_s
    coeff_q2s_lower = exp_gt * m1_s
    coeff_q1s_lower = exp_gt**2 * m2_s

    weighted_diag = (
        jnp.sum(coeff_q3s_diag * xSx_diag[:, 0])
        + jnp.sum(coeff_q2s_diag * xSx_diag[:, 1])
        + jnp.sum(coeff_q1s_diag * xSx_diag[:, 2])
    )
    weighted_lower = (
        jnp.sum(coeff_q3s_lower * xSx_lower[:, 0])
        + jnp.sum(coeff_q2s_lower * xSx_lower[:, 1])
        + jnp.sum(coeff_q1s_lower * xSx_lower[:, 2])
    )

    grad_st = jnp.zeros(n_theta_st, dtype=x.dtype)

    for k in range(n_theta_st):
        dq3s_k = jac_q3s[..., k]  # (ns, ns)
        dq2s_k = jac_q2s[..., k]
        dq1s_k = jac_q1s[..., k]

        # Term 1: d_scale[k] contribution
        term_scale = jac_scale[k] * (weighted_diag + 2.0 * weighted_lower)

        # Term 2: Jacobians of spatial matrices
        xDqx_diag_q3 = jnp.einsum('bi,ij,bj->b', x_st, dq3s_k, x_st)  # (nt,)
        xDqx_diag_q2 = jnp.einsum('bi,ij,bj->b', x_st, dq2s_k, x_st)
        xDqx_diag_q1 = jnp.einsum('bi,ij,bj->b', x_st, dq1s_k, x_st)

        xDqx_lower_q3 = jnp.einsum('bi,ij,bj->b', x_st[1:], dq3s_k, x_st[:-1])  # (nt-1,)
        xDqx_lower_q2 = jnp.einsum('bi,ij,bj->b', x_st[1:], dq2s_k, x_st[:-1])
        xDqx_lower_q1 = jnp.einsum('bi,ij,bj->b', x_st[1:], dq1s_k, x_st[:-1])

        term_spatial_diag = scale * (
            jnp.sum(m0_d * xDqx_diag_q3)
            + exp_gt * jnp.sum(m1_d * xDqx_diag_q2)
            + exp_gt**2 * jnp.sum(m2_d * xDqx_diag_q1)
        )
        term_spatial_lower = 2.0 * scale * (
            jnp.sum(m0_s * xDqx_lower_q3)
            + exp_gt * jnp.sum(m1_s * xDqx_lower_q2)
            + exp_gt**2 * jnp.sum(m2_s * xDqx_lower_q1)
        )

        # Term 3: exp_gt derivative
        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            jnp.sum(m1_d * xSx_diag[:, 1])
            + 2.0 * exp_gt * jnp.sum(m2_d * xSx_diag[:, 2])
        )
        term_exp_gt_lower = 2.0 * scale * jac_exp_gt[k] * (
            jnp.sum(m1_s * xSx_lower[:, 1])
            + 2.0 * exp_gt * jnp.sum(m2_s * xSx_lower[:, 2])
        )

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    # Negate: total derivative is -x^T (dQ_st/dtheta_st) x
    grad_st = -grad_st

    # --- Gradient w.r.t. theta_lik ---
    # d(quad)/d(theta_lik) = 2*x^T*rhs - lik_prec * x^T*AtA*x
    # Compute x^T AtA x using sparse COO data.
    x_fe = x[nt * ns:]

    n_blocks_diag = ata_diag_rows.shape[0]
    block_idx_d = jnp.arange(n_blocks_diag)[:, None]
    xAtAx_diag = jnp.sum(x_st[block_idx_d, ata_diag_rows] * ata_diag_vals * x_st[block_idx_d, ata_diag_cols])

    n_blocks_lower = ata_lower_rows.shape[0]
    block_idx_l = jnp.arange(n_blocks_lower)[:, None]
    xAtAx_lower = jnp.sum(x_st[1:][block_idx_l, ata_lower_rows] * ata_lower_vals * x_st[:-1][block_idx_l, ata_lower_cols])

    n_blocks_arrow = ata_arrow_rows.shape[0]
    block_idx_a = jnp.arange(n_blocks_arrow)[:, None]
    xAtAx_arrow = jnp.sum(x_fe[ata_arrow_rows] * ata_arrow_vals * x_st[block_idx_a, ata_arrow_cols])

    xAtAx_tip = x_fe @ ata_tip @ x_fe

    xAtAx = xAtAx_diag + 2.0 * xAtAx_lower + 2.0 * xAtAx_arrow + xAtAx_tip

    grad_lik = 2.0 * jnp.dot(x, rhs) - likelihood_prec * xAtAx

    return grad_st, grad_lik


def selected_inversion_grads_jax(
    L_diag, L_lower, L_arrow, L_tip,
    sc, jac_sc,
    nt, ns, n_fe, n_theta_st,
    likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip,
):
    """Phase A (full-factor, fused): SI + log-determinant gradient in one sweep.

    Fuses the selected inversion (:func:`pobtasi_jax`) with the gradient
    trace accumulation (:func:`_compute_grad_logdet_cond`) into a single
    backward sweep.  At each block i, the SI entries Z_D_i, Z_B_i, Z_C_i
    are computed, the gradient contributions tr(Z @ dQ/dtheta) are
    accumulated, and the Z entries are discarded.  Only one Z_D block is
    live at a time, reducing peak memory from
    ``L (64 GiB) + full Z (64 GiB) = 128 GiB`` to
    ``L (64 GiB) + one Z block (~128 MiB) = ~64 GiB``.

    This version takes pre-computed L factor blocks.  For the carry-based
    variant that also reconstructs L on the fly, see
    :func:`cholesky_carries.selected_inversion_grads_from_carries_jax`.

    Parameters
    ----------
    L_diag : (nt, ns, ns)
    L_lower : (nt-1, ns, ns)
    L_arrow : (nt, n_fe, ns)
    L_tip : (n_fe, n_fe)
    sc : dict from :func:`precompute_spatial_components`
    jac_sc : dict of Jacobians (from ``jax.jacfwd``, input dim last)
    nt, ns, n_fe, n_theta_st : int
    likelihood_prec : scalar
    ata_* : sparse COO data for AtA blocks
    ata_tip : (n_fe, n_fe)

    Returns
    -------
    grad_st : (n_theta_st,)
        d(logdet Q_cond) / d(theta_st)
    grad_lik : scalar
        d(logdet Q_cond) / d(theta_lik)
    """
    dtype = L_diag.dtype
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    # Build trace-target matrix stack: (n_mats, ns, ns)
    # Layout: [q3s, q2s, q1s, dq3s_0, dq2s_0, dq1s_0, dq3s_1, ..., dq1s_2]
    base_mats = [sc['q3s'], sc['q2s'], sc['q1s']]
    jac_mats = []
    for k in range(n_theta_st):
        jac_mats.extend([
            jac_sc['q3s'][..., k],
            jac_sc['q2s'][..., k],
            jac_sc['q1s'][..., k],
        ])
    all_mats = jnp.stack(base_mats + jac_mats, axis=0)  # (3 + 3*n_theta_st, ns, ns)
    n_mats = all_mats.shape[0]

    # --- S_tip ---
    L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, eye_nfe, lower=True)
    S_tip = L_tip_inv.T @ L_tip_inv

    # --- Last block (i = nt-1) ---
    L_blk_inv = jax.scipy.linalg.solve_triangular(
        L_diag[nt - 1], eye_ns, lower=True
    )
    sa_last = -S_tip @ L_arrow[nt - 1] @ L_blk_inv
    sd_last = (L_blk_inv.T - sa_last.T @ L_arrow[nt - 1]) @ L_blk_inv

    # Accumulate traces from last diagonal block
    traces_d = jnp.einsum('ij,sji->s', sd_last, all_mats)
    m_wt = jnp.tile(jnp.array([m0_d[nt - 1], m1_d[nt - 1], m2_d[nt - 1]], dtype=dtype),
                     1 + n_theta_st)
    acc_d = m_wt * traces_d

    # No lower block for last position
    acc_l = jnp.zeros(n_mats, dtype=dtype)

    # Sparse traces from last block
    sp_d = jnp.sum(sd_last[ata_diag_rows[nt - 1], ata_diag_cols[nt - 1]]
                   * ata_diag_vals[nt - 1])
    sp_l = jnp.array(0.0, dtype=dtype)
    sp_a = jnp.sum(sa_last[ata_arrow_rows[nt - 1], ata_arrow_cols[nt - 1]]
                   * ata_arrow_vals[nt - 1])

    # --- Backward loop ---
    def body_fn(i_rev, carry):
        sd_prev, sa_prev, acc_d_, acc_l_, sp_d_, sp_l_, sp_a_ = carry
        i = nt - 2 - i_rev

        Li = L_diag[i]
        L_blk_inv_i = jax.scipy.linalg.solve_triangular(Li, eye_ns, lower=True)

        sl_i = (-sd_prev @ L_lower[i] - sa_prev.T @ L_arrow[i]) @ L_blk_inv_i
        sa_i = (-sa_prev @ L_lower[i] - S_tip @ L_arrow[i]) @ L_blk_inv_i
        sd_i = (L_blk_inv_i.T - sl_i.T @ L_lower[i] - sa_i.T @ L_arrow[i]) @ L_blk_inv_i

        # Diagonal traces
        tr_d = jnp.einsum('ij,sji->s', sd_i, all_mats)
        mw_d = jnp.tile(jnp.array([m0_d[i], m1_d[i], m2_d[i]], dtype=dtype),
                         1 + n_theta_st)
        acc_d_ = acc_d_ + mw_d * tr_d

        # Lower traces
        tr_l = jnp.einsum('ij,sji->s', sl_i, all_mats)
        mw_l = jnp.tile(jnp.array([m0_s[i], m1_s[i], m2_s[i]], dtype=dtype),
                         1 + n_theta_st)
        acc_l_ = acc_l_ + mw_l * tr_l

        # Sparse COO traces
        sp_d_ = sp_d_ + jnp.sum(
            sd_i[ata_diag_rows[i], ata_diag_cols[i]] * ata_diag_vals[i])
        sp_l_ = sp_l_ + jnp.sum(
            sl_i[ata_lower_rows[i], ata_lower_cols[i]] * ata_lower_vals[i])
        sp_a_ = sp_a_ + jnp.sum(
            sa_i[ata_arrow_rows[i], ata_arrow_cols[i]] * ata_arrow_vals[i])

        return (sd_i, sa_i, acc_d_, acc_l_, sp_d_, sp_l_, sp_a_)

    init_carry = (sd_last, sa_last, acc_d, acc_l, sp_d, sp_l, sp_a)
    _, _, acc_d, acc_l, sp_d, sp_l, sp_a = lax.fori_loop(
        0, nt - 1, body_fn, init_carry
    )

    # Factor of 2: lower block S_l[t] appears twice (lower + upper transpose)
    acc_l = 2.0 * acc_l

    # --- Assemble grad_st ---
    jac_scale = jac_sc['scale']    # (n_theta_st,)
    jac_exp_gt = jac_sc['exp_gt']  # (n_theta_st,)

    weighted_diag = acc_d[0] + exp_gt * acc_d[1] + exp_gt**2 * acc_d[2]
    weighted_lower = acc_l[0] + exp_gt * acc_l[1] + exp_gt**2 * acc_l[2]

    grad_st = jnp.zeros(n_theta_st, dtype=dtype)
    for k in range(n_theta_st):
        term_scale = jac_scale[k] * (weighted_diag + weighted_lower)

        off = 3 + 3 * k
        term_spatial_diag = scale * (
            acc_d[off] + exp_gt * acc_d[off + 1] + exp_gt**2 * acc_d[off + 2])
        term_spatial_lower = scale * (
            acc_l[off] + exp_gt * acc_l[off + 1] + exp_gt**2 * acc_l[off + 2])

        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            acc_d[1] + 2.0 * exp_gt * acc_d[2])
        term_exp_gt_lower = scale * jac_exp_gt[k] * (
            acc_l[1] + 2.0 * exp_gt * acc_l[2])

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    # --- Assemble grad_lik ---
    sp_tip = jnp.sum(S_tip * ata_tip)
    grad_lik = likelihood_prec * (sp_d + 2.0 * sp_l + 2.0 * sp_a + sp_tip)

    return grad_st, grad_lik
