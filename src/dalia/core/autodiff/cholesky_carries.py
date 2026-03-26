# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Carry-based BTA Cholesky factorization, forward/backward substitution, and
selected-inversion gradient accumulation.

This module implements the fused forward pass and the carry-based backward pass
for block-tridiagonal arrowhead (BTA) precision matrices.  The key idea is to
**fuse** the Cholesky factorization with the forward substitution in a single
sequential sweep so that per-block Cholesky factors (L_D, L_B, L_C) are used
as temporaries and never stored.  Only the compact Schur complement carries
S_i = L_B_i @ L_B_i^T are retained, halving storage relative to keeping the
full factor (~n dense b x b blocks instead of ~2n).

During the backward pass (back-substitution and selected inversion), the
Cholesky factors are **reconstructed on the fly** from the stored carries and
the original input blocks: given S_{i-1} and D_i, one Cholesky factorization
and two triangular solves recover L_D_i, L_B_i, and L_C_i.  This trades
one extra Cholesky per block for O(b^2) peak working memory per block.

The gradient of log|Q_c| is computed via a fused selected inversion + trace
accumulation sweep (Phase A of the analytical gradient decomposition), which
runs backward from block n to 1 reconstructing L factors, computing the
selected inverse entries Z_D, Z_B, Z_C, and immediately accumulating the
per-block gradient contributions tr(Z_block @ dQ/dtheta) before discarding
each Z block.  This keeps only two Z blocks live at any time.

Functions
---------
lazy_bta_cholesky_carries
    BTA Cholesky storing Schur carries (no forward substitution).
fused_cholesky_fwd_sub
    Fused BTA Cholesky + forward substitution in a single lax.scan.
backward_sub_from_carries
    Backward substitution L^T x = z with on-the-fly L reconstruction.
selected_inversion_grads_from_carries_jax
    Fused selected inversion + log-determinant gradient accumulation
    (Phase A) with on-the-fly L reconstruction.
"""

import jax
import jax.numpy as jnp
from jax import lax

from dalia.core.autodiff.spatial_precompute import (
    _jax_cholesky,
    _reconstruct_diag_block,
    _reconstruct_lower_block,
)


def lazy_bta_cholesky_carries(
    spatial_comp, nt, ns, n_fe, fe_prec, likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip, dtype,
):
    """BTA Cholesky factorization storing only Schur complement carries.

    Factorizes the conditional precision Q_c = L L^T where Q_c has
    block-tridiagonal arrowhead (BTA) structure::

        Q_c = | D_1   B_1^T         C_1^T |
              | B_1   D_2   B_2^T   C_2^T |
              |       ...   ...     ...   |
              | C_1   C_2   ...     T     |

    Instead of storing the full Cholesky factor (L_D, L_B, L_C for all
    n blocks), only the Schur complement carries entering each step are
    retained.  At step i, the carry S_i = L_B_i @ L_B_i^T propagates
    the coupling from block i to block i+1 via the Schur complement
    update D_{i+1} - S_i.  This halves storage (~n blocks instead of
    ~2n).

    The L factors can be reconstructed on-the-fly from stored carries
    during the backward pass (see :func:`backward_sub_from_carries` and
    :func:`selected_inversion_grads_from_carries_jax`).

    Parameters
    ----------
    spatial_comp : dict
        Output of :func:`precompute_spatial_components`.
    nt, ns, n_fe : int
    fe_prec : float
    likelihood_prec : scalar
    ata_diag_rows, ata_diag_cols, ata_diag_vals : jnp.ndarray
        Sparse COO for diagonal AtA blocks, shape (nt, max_nnz).
    ata_lower_rows, ata_lower_cols, ata_lower_vals : jnp.ndarray
        Sparse COO for lower-diagonal AtA blocks, shape (nt-1, max_nnz).
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals : jnp.ndarray
        Sparse COO for arrow AtA blocks, shape (nt, max_nnz).
    ata_tip : jnp.ndarray
        Arrow tip AtA block, shape (n_fe, n_fe).
    dtype : jnp.dtype

    Returns
    -------
    stored_cond_schurs : (nt, ns, ns)
        Schur complement entering each step (subtracted from diagonal).
    stored_arrow_schurs : (nt, n_fe, ns)
        Arrow Schur complement entering each step.
    L_tip : (n_fe, n_fe)
        Arrow tip Cholesky factor.
    logdet_Q_cond : scalar
    """
    eps = jnp.finfo(dtype).eps
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    m0_sub_pad = jnp.concatenate([spatial_comp['m0_subdiag'], jnp.zeros(1, dtype=dtype)])
    m1_sub_pad = jnp.concatenate([spatial_comp['m1_subdiag'], jnp.zeros(1, dtype=dtype)])
    m2_sub_pad = jnp.concatenate([spatial_comp['m2_subdiag'], jnp.zeros(1, dtype=dtype)])

    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)
    ], axis=0)

    sc_padded = {**spatial_comp,
                 'm0_subdiag': m0_sub_pad,
                 'm1_subdiag': m1_sub_pad,
                 'm2_subdiag': m2_sub_pad}

    def scan_body(carry, inputs):
        (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond) = carry
        i, d_rows, d_cols, d_vals, l_rows, l_cols, l_vals, a_rows, a_cols, a_vals = inputs

        saved = (cond_schur, arrow_schur)

        q_cond_diag_i = _reconstruct_diag_block(sc_padded, i)
        q_cond_diag_i = q_cond_diag_i.at[d_rows, d_cols].add(likelihood_prec * d_vals)
        q_cond_diag_i = q_cond_diag_i + eps_reg * eye_ns - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        q_cond_lower_i = _reconstruct_lower_block(sc_padded, i)
        q_cond_lower_i = q_cond_lower_i.at[l_rows, l_cols].add(likelihood_prec * l_vals)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True
        ).T

        q_arrow_i = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow_i = q_arrow_i.at[a_rows, a_cols].add(likelihood_prec * a_vals)
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True
        ).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(i < nt - 1, new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(i < nt - 1, new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        new_carry = (new_cond_schur, new_arrow_tip_acc, new_arrow_schur, logdet_cond)
        return new_carry, saved

    init_carry = (
        jnp.zeros((ns, ns), dtype=dtype),
        fe_prec * eye_nfe + likelihood_prec * ata_tip + eps_reg * eye_nfe,
        jnp.zeros((n_fe, ns), dtype=dtype),
        jnp.array(0.0, dtype=dtype),
    )

    scan_inputs = (
        jnp.arange(nt),
        ata_diag_rows,
        ata_diag_cols,
        ata_diag_vals,
        ata_lower_rows_pad,
        ata_lower_cols_pad,
        ata_lower_vals_pad,
        ata_arrow_rows,
        ata_arrow_cols,
        ata_arrow_vals,
    )

    (_, arrow_tip_final, _, logdet_cond), \
        (stored_cond_schurs, stored_arrow_schurs) = lax.scan(
            scan_body, init_carry, scan_inputs
        )

    L_tip = _jax_cholesky(arrow_tip_final)
    tip_diag = jnp.diag(L_tip)
    safe_tip_diag = jnp.maximum(tip_diag, eps)
    logdet_Q_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_tip_diag))

    return stored_cond_schurs, stored_arrow_schurs, L_tip, logdet_Q_cond


def fused_cholesky_fwd_sub(
    spatial_comp, nt, ns, n_fe, fe_prec, likelihood_prec,
    rhs_st, rhs_fe,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip, dtype,
):
    """Fused BTA Cholesky + forward substitution in a single sweep.

    Combines the Cholesky factorization of Q_c with the forward
    substitution L z = r into one ``lax.scan`` over temporal blocks.
    At each step i the per-block factors L_D_i, L_B_i, L_C_i are
    computed, used for both the Schur complement update and the
    forward-substitution step, and then discarded.

    The fused pass computes::

        L_D_i = chol(D_i - S_{i-1})              # diagonal factor
        L_B_i = B_i @ L_D_i^{-T}                 # sub-diagonal factor
        L_C_i = (C_i - C^L_{i-1} L_B_{i-1}^T) @ L_D_i^{-T}   # arrow factor
        S_i   = L_B_i @ L_B_i^T                  # Schur carry -> next step
        z_i   = L_D_i^{-1} (r_i - L_B_{i-1} z_{i-1})          # fwd sub

    Only the Schur carries {S_i} and forward-sub vectors {z_i} are
    retained.  The carries are needed for L reconstruction during the
    backward pass; the z_i are needed for back-substitution.

    This is the univariate (single-variable) version.  For coregional
    models with k variables, see
    :func:`coregional_solvers.fused_cholesky_fwd_sub_coregional`.

    Parameters
    ----------
    spatial_comp : dict
        Output of :func:`precompute_spatial_components`.  Contains
        precomputed spatial FEM matrices and temporal coefficients
        from which each Q_p block is reconstructed on the fly.
    nt : int
        Number of temporal blocks (n in the BTA structure).
    ns : int
        Spatial block size (b in the BTA structure).
    n_fe : int
        Number of fixed-effect parameters (arrowhead tip size t).
    fe_prec : float
        Fixed-effects prior precision.
    likelihood_prec : scalar
        Observation precision tau = exp(theta_lik).
    rhs_st : (nt, ns)
        Spatio-temporal portion of the right-hand side r = tau A^T y.
    rhs_fe : (n_fe,)
        Fixed-effects portion of the right-hand side.
    ata_diag_rows, ata_diag_cols, ata_diag_vals : jnp.ndarray
        Sparse COO for diagonal A^T A blocks, shape (nt, max_nnz).
    ata_lower_rows, ata_lower_cols, ata_lower_vals : jnp.ndarray
        Sparse COO for lower-diagonal A^T A blocks, shape (nt-1, max_nnz).
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals : jnp.ndarray
        Sparse COO for arrow A^T A blocks, shape (nt, max_nnz).
    ata_tip : jnp.ndarray
        Arrow tip A^T A block, shape (n_fe, n_fe).
    dtype : jnp.dtype

    Returns
    -------
    stored_cond_schurs : (nt, ns, ns)
        Schur complement carries entering each step.
    stored_arrow_schurs : (nt, n_fe, ns)
        Arrow Schur complement entering each step.
    y_st : (nt, ns)
        Forward substitution result z_i for spatio-temporal blocks.
    L_tip : (n_fe, n_fe)
        Cholesky factor of the updated arrowhead tip.
    arrow_rhs_acc : (n_fe,)
        Modified arrow RHS: r_fe - sum_i L_C_i @ z_i.
    logdet_Q_cond : scalar
        log|Q_c| = 2 sum_i log diag(L_D_i) + 2 log diag(L_T).
    """
    eps = jnp.finfo(dtype).eps
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    m0_sub_pad = jnp.concatenate([spatial_comp['m0_subdiag'], jnp.zeros(1, dtype=dtype)])
    m1_sub_pad = jnp.concatenate([spatial_comp['m1_subdiag'], jnp.zeros(1, dtype=dtype)])
    m2_sub_pad = jnp.concatenate([spatial_comp['m2_subdiag'], jnp.zeros(1, dtype=dtype)])

    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)
    ], axis=0)

    sc_padded = {**spatial_comp,
                 'm0_subdiag': m0_sub_pad,
                 'm1_subdiag': m1_sub_pad,
                 'm2_subdiag': m2_sub_pad}

    def scan_body(carry, inputs):
        (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
         prev_lower_y, arrow_rhs_acc) = carry
        (i, rhs_i, d_rows, d_cols, d_vals,
         l_rows, l_cols, l_vals, a_rows, a_cols, a_vals) = inputs

        saved_carries = (cond_schur, arrow_schur)

        # --- Cholesky step ---
        q_cond_diag_i = _reconstruct_diag_block(sc_padded, i)
        q_cond_diag_i = q_cond_diag_i.at[d_rows, d_cols].add(likelihood_prec * d_vals)
        q_cond_diag_i = q_cond_diag_i + eps_reg * eye_ns - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        q_cond_lower_i = _reconstruct_lower_block(sc_padded, i)
        q_cond_lower_i = q_cond_lower_i.at[l_rows, l_cols].add(likelihood_prec * l_vals)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True
        ).T

        q_arrow_i = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow_i = q_arrow_i.at[a_rows, a_cols].add(likelihood_prec * a_vals)
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True
        ).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(i < nt - 1, new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(i < nt - 1, new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        # --- Forward substitution step ---
        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)

        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(i < nt - 1, new_prev_lower_y,
                                     jnp.zeros_like(new_prev_lower_y))

        new_arrow_rhs_acc = arrow_rhs_acc - L_arrow_i @ y_i

        new_carry = (new_cond_schur, new_arrow_tip_acc, new_arrow_schur, logdet_cond,
                     new_prev_lower_y, new_arrow_rhs_acc)
        return new_carry, (saved_carries, y_i)

    init_carry = (
        jnp.zeros((ns, ns), dtype=dtype),
        fe_prec * eye_nfe + likelihood_prec * ata_tip + eps_reg * eye_nfe,
        jnp.zeros((n_fe, ns), dtype=dtype),
        jnp.array(0.0, dtype=dtype),
        jnp.zeros(ns, dtype=dtype),
        rhs_fe,
    )

    scan_inputs = (
        jnp.arange(nt),
        rhs_st,
        ata_diag_rows,
        ata_diag_cols,
        ata_diag_vals,
        ata_lower_rows_pad,
        ata_lower_cols_pad,
        ata_lower_vals_pad,
        ata_arrow_rows,
        ata_arrow_cols,
        ata_arrow_vals,
    )

    (_, arrow_tip_final, _, logdet_cond, _, arrow_rhs_acc), \
        ((stored_cond_schurs, stored_arrow_schurs), y_st) = lax.scan(
            scan_body, init_carry, scan_inputs
        )

    L_tip = _jax_cholesky(arrow_tip_final)
    tip_diag = jnp.diag(L_tip)
    safe_tip_diag = jnp.maximum(tip_diag, eps)
    logdet_Q_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_tip_diag))

    return stored_cond_schurs, stored_arrow_schurs, y_st, L_tip, arrow_rhs_acc, logdet_Q_cond


def backward_sub_from_carries(
    stored_cond_schurs, stored_arrow_schurs, L_tip,
    y_st, arrow_rhs_acc,
    sc, likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    nt, ns, n_fe, dtype,
):
    """Backward substitution L^T x* = z with on-the-fly L reconstruction.

    Given the forward-substitution result z from
    :func:`fused_cholesky_fwd_sub`, solves L^T x* = z to obtain the
    posterior mode x* = Q_c^{-1} r.  The Cholesky factors are not
    stored; instead, at each block i they are reconstructed from the
    stored Schur carry S_{i-1} and the original input blocks::

        L_D_i = chol(D_i + tau * AtA_diag_i - S_{i-1})
        L_B_i = (B_i + tau * AtA_lower_i) @ L_D_i^{-T}
        L_C_i = (tau * AtA_arrow_i - arrow_schur_i) @ L_D_i^{-T}

    The backward sweep proceeds from block n to 1::

        x*_T = L_T^{-T} z_T
        x*_n = L_D_n^{-T} (z_n - L_C_n^T x*_T)
        x*_i = L_D_i^{-T} (z_i - L_B_i^T x*_{i+1} - L_C_i^T x*_T)

    Also computes ||z||^2 = r^T Q_c^{-1} r (the quadratic form needed
    for the INLA objective).

    Parameters
    ----------
    stored_cond_schurs : (nt, ns, ns)
        Schur carries from the forward pass.
    stored_arrow_schurs : (nt, n_fe, ns)
        Arrow Schur carries from the forward pass.
    L_tip : (n_fe, n_fe)
        Cholesky factor of the arrowhead tip.
    y_st : (nt, ns)
        Forward-substitution vectors z_i.
    arrow_rhs_acc : (n_fe,)
        Modified arrow RHS: r_fe - sum_i L_C_i z_i.
    sc : dict
        Output of :func:`precompute_spatial_components` (unpadded).
    likelihood_prec : scalar
        Observation precision tau.
    ata_* : sparse COO data
        Sparse representation of A^T A blocks.
    nt, ns, n_fe : int
    dtype : jnp.dtype

    Returns
    -------
    x : (nt * ns + n_fe,)
        Posterior mode x* = Q_c^{-1} r.
    quad : scalar
        Quadratic form ||z||^2 = r^T Q_c^{-1} r.
    """
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)

    sc_padded = {**sc,
                 'm0_subdiag': jnp.concatenate([sc['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                 'm1_subdiag': jnp.concatenate([sc['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                 'm2_subdiag': jnp.concatenate([sc['m2_subdiag'], jnp.zeros(1, dtype=dtype)])}

    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)], axis=0)

    def reconstruct_L(i):
        q_diag = _reconstruct_diag_block(sc_padded, i)
        q_diag = q_diag.at[ata_diag_rows[i], ata_diag_cols[i]].add(
            likelihood_prec * ata_diag_vals[i])
        q_diag = q_diag + eps_reg * eye_ns - stored_cond_schurs[i]
        L_i = _jax_cholesky(q_diag)

        q_lower = _reconstruct_lower_block(sc_padded, i)
        q_lower = q_lower.at[ata_lower_rows_pad[i], ata_lower_cols_pad[i]].add(
            likelihood_prec * ata_lower_vals_pad[i])
        L_lower_i = jax.scipy.linalg.solve_triangular(L_i, q_lower.T, lower=True).T

        q_arrow = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow = q_arrow.at[ata_arrow_rows[i], ata_arrow_cols[i]].add(
            likelihood_prec * ata_arrow_vals[i])
        q_arrow = q_arrow - stored_arrow_schurs[i]
        L_arrow_i = jax.scipy.linalg.solve_triangular(L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    # Forward sub on tip + quadratic form
    y_fe = jax.scipy.linalg.solve_triangular(L_tip, arrow_rhs_acc, lower=True)
    quad = jnp.sum(y_st ** 2) + jnp.sum(y_fe ** 2)

    # Backward sub: L^T x = y
    x_fe = jax.scipy.linalg.solve_triangular(L_tip.T, y_fe, lower=False)

    L_last, _, L_arrow_last = reconstruct_L(nt - 1)
    x_last = jax.scipy.linalg.solve_triangular(
        L_last.T, y_st[nt - 1] - L_arrow_last.T @ x_fe, lower=False
    )

    x_st = jnp.zeros((nt, ns), dtype=dtype)
    x_st = x_st.at[nt - 1].set(x_last)

    def body_fn(i_rev, carry):
        x_st_, x_next = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i, L_arrow_i = reconstruct_L(i)
        x_i = jax.scipy.linalg.solve_triangular(
            L_i.T,
            y_st[i] - L_lower_i.T @ x_next - L_arrow_i.T @ x_fe,
            lower=False,
        )
        x_st_ = x_st_.at[i].set(x_i)
        return (x_st_, x_i)

    x_st, _ = lax.fori_loop(0, nt - 1, body_fn, (x_st, x_last))

    x = jnp.concatenate([x_st.reshape(-1), x_fe])
    return x, quad


def selected_inversion_grads_from_carries_jax(
    stored_cond_schurs, stored_arrow_schurs, L_tip,
    sc, jac_sc,
    nt, ns, n_fe, n_theta_st,
    likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip, dtype,
):
    """Phase A: fused selected inversion + log-determinant gradient from carries.

    Computes the gradient of -1/2 log|Q_c| with respect to the
    hyperparameters theta via the identity::

        d(log|Q|)/d(theta_k) = tr(Q^{-1} dQ/d(theta_k))

    Rather than forming the full inverse Q^{-1}, we compute only the
    *selected inverse* entries Z_D_i, Z_B_i, Z_C_i (the entries of
    Q^{-1} at positions where Q is nonzero) via a backward sweep from
    block n to 1.  These entries suffice because dQ/d(theta_k) shares
    the same BTA sparsity as Q, so all other entries contribute zero to
    the trace.

    The selected inversion recurrence (backward from block n to 1)::

        Z_T       = (L_T L_T^T)^{-1}
        Z_C_n     = -Z_T L_C_n L_D_n^{-1}
        Z_D_n     = (L_D_n L_D_n^T)^{-1} - Z_C_n^T L_C_n L_D_n^{-1}
        Z_B_i     = -(Z_D_{i+1} L_B_i + Z_C_{i+1}^T L_C_i) L_D_i^{-1}
        Z_C_i     = -(Z_C_{i+1} L_B_i + Z_T L_C_i) L_D_i^{-1}
        Z_D_i     = (L_D_i L_D_i^T)^{-1} - Z_B_i^T L_B_i L_D_i^{-1}
                    - Z_C_i^T L_C_i L_D_i^{-1}

    At each step, L factors are reconstructed from stored Schur carries
    (same reconstruction as in :func:`backward_sub_from_carries`), and
    gradient contributions are accumulated immediately via::

        d(log|Q_c|)/d(theta_k) = sum_i tr(Z_D_i dD_i/dtheta_k)
                                + 2 sum_i tr(Z_B_i^T dB_i/dtheta_k)
                                + 2 sum_i tr(Z_C_i^T dC_i/dtheta_k)
                                + tr(Z_T dT/dtheta_k)

    Only two Z blocks are ever live at a time (Z_D_{i+1} and Z_D_i),
    giving O(b^2) working memory per block.

    Parameters
    ----------
    stored_cond_schurs : (nt, ns, ns)
        Schur carries from the forward pass.
    stored_arrow_schurs : (nt, n_fe, ns)
        Arrow Schur carries from the forward pass.
    L_tip : (n_fe, n_fe)
        Arrow tip Cholesky factor.
    sc : dict
        Spatial components from :func:`precompute_spatial_components`.
    jac_sc : dict
        Jacobians of spatial components w.r.t. theta_st (from ``jax.jacfwd``).
    nt, ns, n_fe, n_theta_st : int
    likelihood_prec : scalar
        Observation precision tau.
    ata_* : sparse COO data for A^T A blocks.
    ata_tip : (n_fe, n_fe)
    dtype : jnp.dtype

    Returns
    -------
    grad_st : (n_theta_st,)
        d(log|Q_c|)/d(theta_st).
    grad_lik : scalar
        d(log|Q_c|)/d(theta_lik).
    """
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    sc_padded = {**sc,
                 'm0_subdiag': jnp.concatenate([sc['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                 'm1_subdiag': jnp.concatenate([sc['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                 'm2_subdiag': jnp.concatenate([sc['m2_subdiag'], jnp.zeros(1, dtype=dtype)])}

    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)], axis=0)

    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    base_mats = [sc['q3s'], sc['q2s'], sc['q1s']]
    jac_mats = []
    for k in range(n_theta_st):
        jac_mats.extend([
            jac_sc['q3s'][..., k],
            jac_sc['q2s'][..., k],
            jac_sc['q1s'][..., k],
        ])
    all_mats = jnp.stack(base_mats + jac_mats, axis=0)
    n_mats = all_mats.shape[0]

    def reconstruct_L(i):
        q_diag = _reconstruct_diag_block(sc_padded, i)
        q_diag = q_diag.at[ata_diag_rows[i], ata_diag_cols[i]].add(
            likelihood_prec * ata_diag_vals[i])
        q_diag = q_diag + eps_reg * eye_ns - stored_cond_schurs[i]
        L_i = _jax_cholesky(q_diag)

        q_lower = _reconstruct_lower_block(sc_padded, i)
        q_lower = q_lower.at[ata_lower_rows_pad[i], ata_lower_cols_pad[i]].add(
            likelihood_prec * ata_lower_vals_pad[i])
        L_lower_i = jax.scipy.linalg.solve_triangular(L_i, q_lower.T, lower=True).T

        q_arrow = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow = q_arrow.at[ata_arrow_rows[i], ata_arrow_cols[i]].add(
            likelihood_prec * ata_arrow_vals[i])
        q_arrow = q_arrow - stored_arrow_schurs[i]
        L_arrow_i = jax.scipy.linalg.solve_triangular(L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    # --- S_tip ---
    L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, eye_nfe, lower=True)
    S_tip = L_tip_inv.T @ L_tip_inv

    # --- Last block (i = nt-1) ---
    L_last, _, L_arrow_last = reconstruct_L(nt - 1)
    L_blk_inv = jax.scipy.linalg.solve_triangular(L_last, eye_ns, lower=True)
    sa_last = -S_tip @ L_arrow_last @ L_blk_inv
    sd_last = (L_blk_inv.T - sa_last.T @ L_arrow_last) @ L_blk_inv

    traces_d = jnp.einsum('ij,sji->s', sd_last, all_mats)
    m_wt = jnp.tile(jnp.array([m0_d[nt - 1], m1_d[nt - 1], m2_d[nt - 1]], dtype=dtype),
                     1 + n_theta_st)
    acc_d = m_wt * traces_d
    acc_l = jnp.zeros(n_mats, dtype=dtype)

    sp_d = jnp.sum(sd_last[ata_diag_rows[nt - 1], ata_diag_cols[nt - 1]]
                   * ata_diag_vals[nt - 1])
    sp_l = jnp.array(0.0, dtype=dtype)
    sp_a = jnp.sum(sa_last[ata_arrow_rows[nt - 1], ata_arrow_cols[nt - 1]]
                   * ata_arrow_vals[nt - 1])

    # --- Backward loop ---
    def body_fn(i_rev, carry):
        sd_prev, sa_prev, acc_d_, acc_l_, sp_d_, sp_l_, sp_a_ = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i, L_arrow_i = reconstruct_L(i)
        L_blk_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_ns, lower=True)

        sl_i = (-sd_prev @ L_lower_i - sa_prev.T @ L_arrow_i) @ L_blk_inv_i
        sa_i = (-sa_prev @ L_lower_i - S_tip @ L_arrow_i) @ L_blk_inv_i
        sd_i = (L_blk_inv_i.T - sl_i.T @ L_lower_i - sa_i.T @ L_arrow_i) @ L_blk_inv_i

        tr_d = jnp.einsum('ij,sji->s', sd_i, all_mats)
        mw_d = jnp.tile(jnp.array([m0_d[i], m1_d[i], m2_d[i]], dtype=dtype),
                         1 + n_theta_st)
        acc_d_ = acc_d_ + mw_d * tr_d

        tr_l = jnp.einsum('ij,sji->s', sl_i, all_mats)
        mw_l = jnp.tile(jnp.array([m0_s[i], m1_s[i], m2_s[i]], dtype=dtype),
                         1 + n_theta_st)
        acc_l_ = acc_l_ + mw_l * tr_l

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

    # --- Assemble grad_lik ---
    sp_tip = jnp.sum(S_tip * ata_tip)
    grad_lik = likelihood_prec * (sp_d + 2.0 * sp_l + 2.0 * sp_a + sp_tip)

    return grad_st, grad_lik
