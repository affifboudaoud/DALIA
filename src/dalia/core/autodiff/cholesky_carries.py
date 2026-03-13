# Copyright 2024-2025 DALIA authors. All rights reserved.

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
    """BTA Cholesky storing carries instead of L factor blocks.

    Same computation as :func:`lazy_bta_cholesky` but outputs the scan
    carries ``(cond_schur, arrow_schur)`` entering each step instead of
    the L factor blocks ``(L_diag, L_lower, L_arrow)``.  This reduces
    scan output storage from ~64 GiB to ~32 GiB for gst_large.

    L blocks can be reconstructed on-the-fly from stored carries via
    :func:`selected_inversion_grads_from_carries_jax`.

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
    """Fused BTA Cholesky + forward substitution in a single scan.

    Combines the Cholesky factorization and forward solve ``L y = rhs``
    into one :func:`lax.scan`, so L blocks are per-step intermediates
    that are never stored as scan outputs.  Outputs the Schur complement
    carries (~32 GiB for gst_large) and forward-sub solutions y_st (~8 MB).

    Parameters
    ----------
    spatial_comp : dict
        Output of :func:`precompute_spatial_components`.
    nt, ns, n_fe : int
    fe_prec : float
    likelihood_prec : scalar
    rhs_st : (nt, ns)
        Spatio-temporal portion of the right-hand side.
    rhs_fe : (n_fe,)
        Fixed-effects portion of the right-hand side.
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
    stored_arrow_schurs : (nt, n_fe, ns)
    y_st : (nt, ns)
        Forward substitution result for spatio-temporal blocks.
    L_tip : (n_fe, n_fe)
    arrow_rhs_acc : (n_fe,)
        Modified arrow rhs: ``rhs_fe - sum_i L_arrow[i] @ y_i``.
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
    """Backward substitution ``L^T x = y``, reconstructing L from carries.

    Performs the backward solve to obtain ``x = Q_cond^{-1} rhs``, where
    the Cholesky factor L is reconstructed on-the-fly from stored Schur
    complement carries, avoiding materialization of full L arrays.

    Also computes the quadratic form ``rhs^T Q_cond^{-1} rhs = ||y||^2``.

    Parameters
    ----------
    stored_cond_schurs : (nt, ns, ns)
    stored_arrow_schurs : (nt, n_fe, ns)
    L_tip : (n_fe, n_fe)
    y_st : (nt, ns)
        Forward substitution result for spatio-temporal blocks.
    arrow_rhs_acc : (n_fe,)
        Modified arrow rhs: ``rhs_fe - sum_i L_arrow[i] @ y_i``.
    sc : dict
        Output of :func:`precompute_spatial_components` (unpadded).
    likelihood_prec : scalar
    ata_* : sparse COO data
    nt, ns, n_fe : int
    dtype : jnp.dtype

    Returns
    -------
    x : (nt * ns + n_fe,)
    quad : scalar
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
    """Fused selected inversion + gradient accumulation from stored carries.

    Like :func:`selected_inversion_grads_jax` but takes scan carries
    instead of L blocks.  L blocks are reconstructed on-the-fly from
    ``stored_cond_schurs`` and ``stored_arrow_schurs``, reducing peak
    memory from ~64 GiB (full L arrays) to ~32 GiB (carries only).

    Parameters
    ----------
    stored_cond_schurs : (nt, ns, ns)
        Schur complement entering each BTA Cholesky step.
    stored_arrow_schurs : (nt, n_fe, ns)
        Arrow Schur complement entering each step.
    L_tip : (n_fe, n_fe)
        Arrow tip Cholesky factor.
    sc : dict
        Output of :func:`precompute_spatial_components` (unpadded).
    jac_sc : dict
        Jacobians of sc w.r.t. theta_st (from ``jax.jacfwd``).
    nt, ns, n_fe, n_theta_st : int
    likelihood_prec : scalar
    ata_* : sparse COO data for AtA blocks
    ata_tip : (n_fe, n_fe)
    dtype : jnp.dtype

    Returns
    -------
    grad_st : (n_theta_st,)
    grad_lik : scalar
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
