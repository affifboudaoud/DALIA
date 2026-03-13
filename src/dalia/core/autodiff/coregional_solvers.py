# Copyright 2024-2025 DALIA authors. All rights reserved.

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from dalia.core.autodiff.spatial_precompute import (
    _jax_cholesky,
    _reconstruct_diag_block,
    _reconstruct_lower_block,
    _reconstruct_coregional_diag_block,
    _reconstruct_coregional_lower_block,
)


def fused_cholesky_fwd_sub_coregional(
    sc_list, coreg_w, n_models, nt, ns, n_fe, fe_prec,
    likelihood_precs,
    rhs_st, rhs_fe,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_ata_tip, per_model_offsets,
    dtype,
):
    """Fused BTA Cholesky + forward substitution for coregional models.

    Combines the Cholesky factorization and forward solve into one
    :func:`lax.scan`, with super-blocks reconstructed on the fly from
    per-model spatial components.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components (length n_models), padded subdiags.
    coreg_w : (n_models, n_models, n_models)
    n_models, nt, ns, n_fe : int
    fe_prec : float
    likelihood_precs : (n_models,)
    rhs_st : (nt, block_size)
    rhs_fe : (n_fe,)
    per_model_ata_diag_{rows,cols,vals} : list of (nt, max_nnz_m)
    per_model_ata_lower_{rows,cols,vals} : list of (nt, max_nnz_m)
    per_model_ata_arrow_{rows,cols,vals} : list of (nt, max_nnz_m)
    per_model_ata_tip : list of (n_fe, n_fe)
    per_model_offsets : list of int
    dtype : jnp.dtype

    Returns
    -------
    stored_cond_schurs : (nt, block_size, block_size)
    stored_arrow_schurs : (nt, n_fe, block_size)
    y_st : (nt, block_size)
    L_tip : (n_fe, n_fe)
    arrow_rhs_acc : (n_fe,)
    logdet_Q_cond : scalar
    """
    block_size = n_models * ns
    eps = jnp.finfo(dtype).eps
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_bs = jnp.eye(block_size, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    # Pad lower AtA arrays to length nt for each model
    padded_lower_rows = []
    padded_lower_cols = []
    padded_lower_vals = []
    for m in range(n_models):
        lr = per_model_ata_lower_rows[m]
        lc = per_model_ata_lower_cols[m]
        lv = per_model_ata_lower_vals[m]
        padded_lower_rows.append(jnp.concatenate([
            lr, jnp.zeros((1, lr.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_cols.append(jnp.concatenate([
            lc, jnp.zeros((1, lc.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_vals.append(jnp.concatenate([
            lv, jnp.zeros((1, lv.shape[1]), dtype=dtype)], axis=0))

    # Pad sc_list subdiags
    sc_list_padded = []
    for m in range(n_models):
        sc_m = sc_list[m]
        sc_list_padded.append({
            **sc_m,
            'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
        })

    # Arrow tip initial: fe_prec * I + sum_m prec_m * ata_tip_m
    arrow_tip_init = fe_prec * eye_nfe + eps_reg * eye_nfe
    for m in range(n_models):
        arrow_tip_init = arrow_tip_init + likelihood_precs[m] * per_model_ata_tip[m]

    def scan_body(carry, inputs):
        (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
         prev_lower_y, arrow_rhs_acc) = carry
        i = inputs[0]
        rhs_i = inputs[1]

        saved_carries = (cond_schur, arrow_schur)

        # Reconstruct Q_prior super-block on the fly
        q_cond_diag_i = _reconstruct_coregional_diag_block(
            sc_list_padded, coreg_w, n_models, ns, i)

        # Scatter per-model sparse AtA
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            d_rows = per_model_ata_diag_rows[m][i]
            d_cols = per_model_ata_diag_cols[m][i]
            d_vals = per_model_ata_diag_vals[m][i]
            q_cond_diag_i = q_cond_diag_i.at[
                m_off + d_rows, m_off + d_cols
            ].add(likelihood_precs[m] * d_vals)

        q_cond_diag_i = q_cond_diag_i + eps_reg * eye_bs - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        # Lower block
        q_cond_lower_i = _reconstruct_coregional_lower_block(
            sc_list_padded, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            l_rows = padded_lower_rows[m][i]
            l_cols = padded_lower_cols[m][i]
            l_vals = padded_lower_vals[m][i]
            q_cond_lower_i = q_cond_lower_i.at[
                m_off + l_rows, m_off + l_cols
            ].add(likelihood_precs[m] * l_vals)

        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True).T

        # Arrow block
        q_arrow_i = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            a_rows = per_model_ata_arrow_rows[m][i]
            a_cols = per_model_ata_arrow_cols[m][i]
            a_vals = per_model_ata_arrow_vals[m][i]
            q_arrow_i = q_arrow_i.at[a_rows, m_off + a_cols].add(
                likelihood_precs[m] * a_vals)
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(i < nt - 1, new_cond_schur,
                                    jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(i < nt - 1, new_arrow_schur,
                                     jnp.zeros_like(new_arrow_schur))

        # Forward substitution
        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)

        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(i < nt - 1, new_prev_lower_y,
                                      jnp.zeros_like(new_prev_lower_y))
        new_arrow_rhs_acc = arrow_rhs_acc - L_arrow_i @ y_i

        new_carry = (new_cond_schur, new_arrow_tip_acc, new_arrow_schur,
                     logdet_cond, new_prev_lower_y, new_arrow_rhs_acc)
        return new_carry, (saved_carries, y_i)

    init_carry = (
        jnp.zeros((block_size, block_size), dtype=dtype),
        arrow_tip_init,
        jnp.zeros((n_fe, block_size), dtype=dtype),
        jnp.array(0.0, dtype=dtype),
        jnp.zeros(block_size, dtype=dtype),
        rhs_fe,
    )

    scan_inputs = (jnp.arange(nt), rhs_st)

    (_, arrow_tip_final, _, logdet_cond, _, arrow_rhs_acc), \
        ((stored_cond_schurs, stored_arrow_schurs), y_st) = lax.scan(
            scan_body, init_carry, scan_inputs
        )

    L_tip = _jax_cholesky(arrow_tip_final)
    tip_diag = jnp.diag(L_tip)
    safe_tip_diag = jnp.maximum(tip_diag, eps)
    logdet_Q_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_tip_diag))

    return stored_cond_schurs, stored_arrow_schurs, y_st, L_tip, arrow_rhs_acc, logdet_Q_cond


def backward_sub_from_carries_coregional(
    stored_cond_schurs, stored_arrow_schurs, L_tip,
    y_st, arrow_rhs_acc,
    sc_list, coreg_w, likelihood_precs,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_offsets,
    n_models, nt, ns, n_fe, dtype,
):
    """Backward substitution for coregional, reconstructing L from carries.

    Parameters
    ----------
    stored_cond_schurs : (nt, block_size, block_size)
    stored_arrow_schurs : (nt, n_fe, block_size)
    L_tip : (n_fe, n_fe)
    y_st : (nt, block_size)
    arrow_rhs_acc : (n_fe,)
    sc_list : list of dict
        Per-model spatial components (unpadded).
    coreg_w : (n_models, n_models, n_models)
    likelihood_precs : (n_models,)
    per_model_ata_* : per-model sparse COO data
    per_model_offsets : list of int
    n_models, nt, ns, n_fe : int
    dtype : jnp.dtype

    Returns
    -------
    x : (nt * block_size + n_fe,)
    quad : scalar
    """
    block_size = n_models * ns
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_bs = jnp.eye(block_size, dtype=dtype)

    # Pad sc_list and lower AtA
    sc_padded = []
    for m in range(n_models):
        sc_m = sc_list[m]
        sc_padded.append({
            **sc_m,
            'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
        })

    padded_lower_rows = []
    padded_lower_cols = []
    padded_lower_vals = []
    for m in range(n_models):
        lr = per_model_ata_lower_rows[m]
        lc = per_model_ata_lower_cols[m]
        lv = per_model_ata_lower_vals[m]
        padded_lower_rows.append(jnp.concatenate([
            lr, jnp.zeros((1, lr.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_cols.append(jnp.concatenate([
            lc, jnp.zeros((1, lc.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_vals.append(jnp.concatenate([
            lv, jnp.zeros((1, lv.shape[1]), dtype=dtype)], axis=0))

    def reconstruct_L(i):
        q_diag = _reconstruct_coregional_diag_block(sc_padded, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_diag = q_diag.at[
                m_off + per_model_ata_diag_rows[m][i],
                m_off + per_model_ata_diag_cols[m][i]
            ].add(likelihood_precs[m] * per_model_ata_diag_vals[m][i])
        q_diag = q_diag + eps_reg * eye_bs - stored_cond_schurs[i]
        L_i = _jax_cholesky(q_diag)

        q_lower = _reconstruct_coregional_lower_block(sc_padded, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_lower = q_lower.at[
                m_off + padded_lower_rows[m][i],
                m_off + padded_lower_cols[m][i]
            ].add(likelihood_precs[m] * padded_lower_vals[m][i])
        L_lower_i = jax.scipy.linalg.solve_triangular(L_i, q_lower.T, lower=True).T

        q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_arrow = q_arrow.at[
                per_model_ata_arrow_rows[m][i],
                m_off + per_model_ata_arrow_cols[m][i]
            ].add(likelihood_precs[m] * per_model_ata_arrow_vals[m][i])
        q_arrow = q_arrow - stored_arrow_schurs[i]
        L_arrow_i = jax.scipy.linalg.solve_triangular(L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    # Forward sub on tip + quadratic form
    y_fe = jax.scipy.linalg.solve_triangular(L_tip, arrow_rhs_acc, lower=True)
    quad = jnp.sum(y_st ** 2) + jnp.sum(y_fe ** 2)

    # Backward sub
    x_fe = jax.scipy.linalg.solve_triangular(L_tip.T, y_fe, lower=False)

    L_last, _, L_arrow_last = reconstruct_L(nt - 1)
    x_last = jax.scipy.linalg.solve_triangular(
        L_last.T, y_st[nt - 1] - L_arrow_last.T @ x_fe, lower=False)

    x_st = jnp.zeros((nt, block_size), dtype=dtype)
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


def logdet_Q_prior_coregional_scan(
    sc_list, coreg_w, n_models, ns, nt, dtype,
):
    """Compute logdet(Q_prior) for coregional model via BT Cholesky scan.

    Q_prior has no arrowhead, so this is a simple BT Cholesky.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components (padded subdiags).
    coreg_w : (n_models, n_models, n_models)
    n_models, ns, nt : int
    dtype : jnp.dtype

    Returns
    -------
    logdet : scalar
    """
    block_size = n_models * ns
    eps = jnp.finfo(dtype).eps

    @partial(jax.checkpoint, prevent_cse=True)
    def scan_body(carry, i):
        schur, logdet = carry
        q_diag_i = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, i) - schur
        L_i = _jax_cholesky(q_diag_i)
        diag_vals = jnp.diag(L_i)
        safe_vals = jnp.maximum(diag_vals, eps)
        logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_vals))

        q_lower_i = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, i)
        L_inv_lower = jax.scipy.linalg.solve_triangular(
            L_i, q_lower_i.T, lower=True)
        new_schur = L_inv_lower.T @ L_inv_lower
        new_schur = jnp.where(i < nt - 1, new_schur, jnp.zeros_like(new_schur))
        return (new_schur, logdet), None

    init_carry = (jnp.zeros((block_size, block_size), dtype=dtype),
                  jnp.array(0.0, dtype=dtype))
    (_, logdet), _ = lax.scan(scan_body, init_carry, jnp.arange(nt))
    return logdet


def logdet_Q_prior_coregional_grad(
    sc_list, jac_sc_list, coreg_w, jac_coreg_w,
    n_models, ns, nt, dtype,
):
    """Analytical gradient of logdet(Q_prior) for coregional models.

    Uses forward BT Cholesky storing Schur carries, then backward BT
    selected inversion to accumulate tr(Sigma @ dQ/dtheta) block-by-block.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components (padded subdiags).
    jac_sc_list : list of dict
        Per-model Jacobians of spatial components w.r.t. their theta_st_m.
    coreg_w : (n_models, n_models, n_models)
    jac_coreg_w : (n_models, n_models, n_models, n_coreg_params)
        Jacobian of coreg_w w.r.t. coregional params.
    n_models, ns, nt : int
    dtype : jnp.dtype

    Returns
    -------
    grad_per_model_st : list of (n_theta_st_m,) arrays
    grad_coreg : (n_coreg_params,)
    """
    block_size = n_models * ns
    eye_bs = jnp.eye(block_size, dtype=dtype)
    n_coreg = jac_coreg_w.shape[-1]

    # --- Forward BT Cholesky scan, store incoming Schurs ---
    def fwd_body(schur, i):
        q_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, i) - schur
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        new_schur = L_lower_i @ L_lower_i.T
        new_schur = jnp.where(i < nt - 1, new_schur, jnp.zeros_like(new_schur))
        return new_schur, schur

    init_schur = jnp.zeros((block_size, block_size), dtype=dtype)
    _, stored_schurs = lax.scan(fwd_body, init_schur, jnp.arange(nt))

    def _reconstruct_L(i):
        q_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, i) - stored_schurs[i]
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        return L_i, L_lower_i

    # --- Accumulate traces via backward BT selected inversion ---
    # For each sub-block (i,j), tr(S_ij[t] @ Qu_m[t]) contributes to both
    # ST params (via dQu_m/dtheta_st_m) and coreg params (via dw_ijm/dparam).

    # Initialize accumulators (stacked for lax.fori_loop compatibility)
    grad_per_model_st = jnp.zeros((n_models, 3), dtype=dtype)
    grad_coreg = jnp.zeros(n_coreg, dtype=dtype)

    def _accumulate_traces(sd_i, sl_i, t_diag, t_lower):
        """Accumulate gradient contributions from S_diag=sd_i and S_lower=sl_i at time step."""
        g_st = jnp.zeros((n_models, 3), dtype=dtype)
        g_c = jnp.zeros(n_coreg, dtype=dtype)

        for ii in range(n_models):
            for jj in range(n_models):
                # Extract sub-block of Sigma
                sd_ij = sd_i[ii * ns:(ii + 1) * ns, jj * ns:(jj + 1) * ns]

                for m_idx in range(n_models):
                    w_ijm = coreg_w[ii, jj, m_idx]
                    sc_m = sc_list[m_idx]

                    # Trace with per-model reconstruction components
                    qu_diag_m = _reconstruct_diag_block(sc_m, t_diag)
                    tr_val = jnp.sum(sd_ij * qu_diag_m.T)

                    # Coreg gradient: dw_ijm/dparam * tr(S_ij @ Qu_m)
                    dw = jac_coreg_w[ii, jj, m_idx, :]
                    g_c = g_c + dw * tr_val

                    # ST gradient for model m: w_ijm * tr(S_ij @ dQu_m/dtheta_st_m)
                    jsc_m = jac_sc_list[m_idx]
                    scale_m = sc_m['scale']
                    exp_gt_m = sc_m['exp_gt']
                    m0_d_val = sc_m['m0_diag'][t_diag]
                    m1_d_val = sc_m['m1_diag'][t_diag]
                    m2_d_val = sc_m['m2_diag'][t_diag]

                    for k in range(3):
                        dq3s = jsc_m['q3s'][..., k]
                        dq2s = jsc_m['q2s'][..., k]
                        dq1s = jsc_m['q1s'][..., k]
                        d_scale = jsc_m['scale'][k]
                        d_exp_gt = jsc_m['exp_gt'][k]

                        dQu_diag = (
                            d_scale * (m0_d_val * sc_m['q3s'] + exp_gt_m * m1_d_val * sc_m['q2s']
                                       + exp_gt_m**2 * m2_d_val * sc_m['q1s'])
                            + scale_m * (m0_d_val * dq3s + exp_gt_m * m1_d_val * dq2s
                                         + exp_gt_m**2 * m2_d_val * dq1s)
                            + scale_m * d_exp_gt * (m1_d_val * sc_m['q2s']
                                                    + 2.0 * exp_gt_m * m2_d_val * sc_m['q1s'])
                        )
                        tr_jac = jnp.sum(sd_ij * dQu_diag.T)
                        g_st = g_st.at[m_idx, k].add(w_ijm * tr_jac)

                # Lower block contribution (only if sl_i is provided)
                if t_lower is not None:
                    sl_ij = sl_i[ii * ns:(ii + 1) * ns, jj * ns:(jj + 1) * ns]
                    for m_idx in range(n_models):
                        w_ijm = coreg_w[ii, jj, m_idx]
                        sc_m = sc_list[m_idx]

                        qu_lower_m = _reconstruct_lower_block(sc_m, t_lower)
                        tr_val_l = jnp.sum(sl_ij * qu_lower_m.T)

                        dw = jac_coreg_w[ii, jj, m_idx, :]
                        g_c = g_c + 2.0 * dw * tr_val_l

                        jsc_m = jac_sc_list[m_idx]
                        scale_m = sc_m['scale']
                        exp_gt_m = sc_m['exp_gt']
                        m0_s_val = sc_m['m0_subdiag'][t_lower]
                        m1_s_val = sc_m['m1_subdiag'][t_lower]
                        m2_s_val = sc_m['m2_subdiag'][t_lower]

                        for k in range(3):
                            dq3s = jsc_m['q3s'][..., k]
                            dq2s = jsc_m['q2s'][..., k]
                            dq1s = jsc_m['q1s'][..., k]
                            d_scale = jsc_m['scale'][k]
                            d_exp_gt = jsc_m['exp_gt'][k]

                            dQu_lower = (
                                d_scale * (m0_s_val * sc_m['q3s'] + exp_gt_m * m1_s_val * sc_m['q2s']
                                           + exp_gt_m**2 * m2_s_val * sc_m['q1s'])
                                + scale_m * (m0_s_val * dq3s + exp_gt_m * m1_s_val * dq2s
                                             + exp_gt_m**2 * m2_s_val * dq1s)
                                + scale_m * d_exp_gt * (m1_s_val * sc_m['q2s']
                                                        + 2.0 * exp_gt_m * m2_s_val * sc_m['q1s'])
                            )
                            tr_jac_l = jnp.sum(sl_ij * dQu_lower.T)
                            g_st = g_st.at[m_idx, k].add(2.0 * w_ijm * tr_jac_l)

        return g_st, g_c

    # --- Last block ---
    L_last, _ = _reconstruct_L(nt - 1)
    L_inv_last = jax.scipy.linalg.solve_triangular(L_last, eye_bs, lower=True)
    sd_last = L_inv_last.T @ L_inv_last

    g_st_last, g_c_last = _accumulate_traces(sd_last, None, nt - 1, None)
    grad_per_model_st = grad_per_model_st + g_st_last
    grad_coreg = grad_coreg + g_c_last

    # --- Backward loop via lax.fori_loop ---
    def bwd_body(i_rev, carry):
        sd_prev, grad_st_acc, grad_c_acc = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i = _reconstruct_L(i)
        L_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_bs, lower=True)
        sl_i = -sd_prev @ L_lower_i @ L_inv_i
        sd_i = (L_inv_i.T - sl_i.T @ L_lower_i) @ L_inv_i

        g_st_i, g_c_i = _accumulate_traces(sd_i, sl_i, i, i)
        return (sd_i, grad_st_acc + g_st_i, grad_c_acc + g_c_i)

    carry = (sd_last, grad_per_model_st, grad_coreg)
    _, grad_per_model_st, grad_coreg = lax.fori_loop(0, nt - 1, bwd_body, carry)

    return [grad_per_model_st[m] for m in range(n_models)], grad_coreg


def selected_inversion_grads_from_carries_coregional(
    stored_cond_schurs, stored_arrow_schurs, L_tip,
    sc_list, jac_sc_list, coreg_w, jac_coreg_w,
    n_models, nt, ns, n_fe,
    likelihood_precs,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_ata_tip, per_model_offsets,
    dtype,
):
    """Fused selected inversion + gradient accumulation for coregional models.

    Reconstructs L blocks on-the-fly from stored carries and accumulates
    gradient traces for all hyperparameter groups:
    - Per-model ST params (r_s_m, r_t_m, sigma_st_m)
    - Per-model likelihood precision
    - Coregional params (sigmas, lambdas)

    Parameters
    ----------
    stored_cond_schurs : (nt, block_size, block_size)
    stored_arrow_schurs : (nt, n_fe, block_size)
    L_tip : (n_fe, n_fe)
    sc_list : list of dict
        Per-model spatial components (padded subdiags).
    jac_sc_list : list of dict
        Per-model Jacobians from jacfwd.
    coreg_w : (n_models, n_models, n_models)
    jac_coreg_w : (n_models, n_models, n_models, n_coreg_params)
    n_models, nt, ns, n_fe : int
    likelihood_precs : (n_models,)
    per_model_ata_* : per-model sparse COO data
    per_model_ata_tip : list of (n_fe, n_fe)
    per_model_offsets : list of int
    dtype : jnp.dtype

    Returns
    -------
    grad_per_model_st : list of (3,) arrays
    grad_per_model_lik : (n_models,)
    grad_coreg : (n_coreg_params,)
    """
    block_size = n_models * ns
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_bs = jnp.eye(block_size, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)
    n_coreg = jac_coreg_w.shape[-1]

    # Pad lower AtA
    padded_lower_rows = []
    padded_lower_cols = []
    padded_lower_vals = []
    for m in range(n_models):
        lr = per_model_ata_lower_rows[m]
        lc = per_model_ata_lower_cols[m]
        lv = per_model_ata_lower_vals[m]
        padded_lower_rows.append(jnp.concatenate([
            lr, jnp.zeros((1, lr.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_cols.append(jnp.concatenate([
            lc, jnp.zeros((1, lc.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_vals.append(jnp.concatenate([
            lv, jnp.zeros((1, lv.shape[1]), dtype=dtype)], axis=0))

    def reconstruct_L(i):
        q_diag = _reconstruct_coregional_diag_block(sc_list, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_diag = q_diag.at[
                m_off + per_model_ata_diag_rows[m][i],
                m_off + per_model_ata_diag_cols[m][i]
            ].add(likelihood_precs[m] * per_model_ata_diag_vals[m][i])
        q_diag = q_diag + eps_reg * eye_bs - stored_cond_schurs[i]
        L_i = _jax_cholesky(q_diag)

        q_lower = _reconstruct_coregional_lower_block(sc_list, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_lower = q_lower.at[
                m_off + padded_lower_rows[m][i],
                m_off + padded_lower_cols[m][i]
            ].add(likelihood_precs[m] * padded_lower_vals[m][i])
        L_lower_i = jax.scipy.linalg.solve_triangular(L_i, q_lower.T, lower=True).T

        q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_arrow = q_arrow.at[
                per_model_ata_arrow_rows[m][i],
                m_off + per_model_ata_arrow_cols[m][i]
            ].add(likelihood_precs[m] * per_model_ata_arrow_vals[m][i])
        q_arrow = q_arrow - stored_arrow_schurs[i]
        L_arrow_i = jax.scipy.linalg.solve_triangular(L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    def _accumulate_block_grads(sd_i, sl_i, sa_i, t_diag, t_lower):
        """Accumulate gradient contributions from Sigma blocks at time step."""
        g_st = jnp.zeros((n_models, 3), dtype=dtype)
        g_lik = jnp.zeros(n_models, dtype=dtype)
        g_c = jnp.zeros(n_coreg, dtype=dtype)

        # ST and coreg gradients from diagonal super-block
        for ii in range(n_models):
            for jj in range(n_models):
                sd_ij = sd_i[ii * ns:(ii + 1) * ns, jj * ns:(jj + 1) * ns]

                for m_idx in range(n_models):
                    w_ijm = coreg_w[ii, jj, m_idx]
                    sc_m = sc_list[m_idx]

                    qu_diag_m = _reconstruct_diag_block(sc_m, t_diag)
                    tr_val = jnp.sum(sd_ij * qu_diag_m.T)

                    # Coreg gradient
                    dw = jac_coreg_w[ii, jj, m_idx, :]
                    g_c = g_c + dw * tr_val

                    # ST gradient
                    jsc_m = jac_sc_list[m_idx]
                    scale_m = sc_m['scale']
                    exp_gt_m = sc_m['exp_gt']
                    m0_d_val = sc_m['m0_diag'][t_diag]
                    m1_d_val = sc_m['m1_diag'][t_diag]
                    m2_d_val = sc_m['m2_diag'][t_diag]

                    for k in range(3):
                        dq3s = jsc_m['q3s'][..., k]
                        dq2s = jsc_m['q2s'][..., k]
                        dq1s = jsc_m['q1s'][..., k]
                        d_scale = jsc_m['scale'][k]
                        d_exp_gt = jsc_m['exp_gt'][k]

                        dQu = (
                            d_scale * (m0_d_val * sc_m['q3s'] + exp_gt_m * m1_d_val * sc_m['q2s']
                                       + exp_gt_m**2 * m2_d_val * sc_m['q1s'])
                            + scale_m * (m0_d_val * dq3s + exp_gt_m * m1_d_val * dq2s
                                         + exp_gt_m**2 * m2_d_val * dq1s)
                            + scale_m * d_exp_gt * (m1_d_val * sc_m['q2s']
                                                    + 2.0 * exp_gt_m * m2_d_val * sc_m['q1s'])
                        )
                        g_st = g_st.at[m_idx, k].add(
                            w_ijm * jnp.sum(sd_ij * dQu.T))

                # Lower block contribution
                if sl_i is not None and t_lower is not None:
                    sl_ij = sl_i[ii * ns:(ii + 1) * ns, jj * ns:(jj + 1) * ns]
                    for m_idx in range(n_models):
                        w_ijm = coreg_w[ii, jj, m_idx]
                        sc_m = sc_list[m_idx]
                        qu_lower_m = _reconstruct_lower_block(sc_m, t_lower)
                        tr_val_l = jnp.sum(sl_ij * qu_lower_m.T)

                        dw = jac_coreg_w[ii, jj, m_idx, :]
                        g_c = g_c + 2.0 * dw * tr_val_l

                        jsc_m = jac_sc_list[m_idx]
                        scale_m = sc_m['scale']
                        exp_gt_m = sc_m['exp_gt']
                        m0_s_val = sc_m['m0_subdiag'][t_lower]
                        m1_s_val = sc_m['m1_subdiag'][t_lower]
                        m2_s_val = sc_m['m2_subdiag'][t_lower]

                        for k in range(3):
                            dq3s = jsc_m['q3s'][..., k]
                            dq2s = jsc_m['q2s'][..., k]
                            dq1s = jsc_m['q1s'][..., k]
                            d_scale = jsc_m['scale'][k]
                            d_exp_gt = jsc_m['exp_gt'][k]

                            dQu_l = (
                                d_scale * (m0_s_val * sc_m['q3s'] + exp_gt_m * m1_s_val * sc_m['q2s']
                                           + exp_gt_m**2 * m2_s_val * sc_m['q1s'])
                                + scale_m * (m0_s_val * dq3s + exp_gt_m * m1_s_val * dq2s
                                             + exp_gt_m**2 * m2_s_val * dq1s)
                                + scale_m * d_exp_gt * (m1_s_val * sc_m['q2s']
                                                        + 2.0 * exp_gt_m * m2_s_val * sc_m['q1s'])
                            )
                            g_st = g_st.at[m_idx, k].add(
                                2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))

        # Likelihood gradient from sparse AtA
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            # Diagonal AtA contribution
            sp_d = jnp.sum(
                sd_i[m_off + per_model_ata_diag_rows[m][t_diag],
                     m_off + per_model_ata_diag_cols[m][t_diag]]
                * per_model_ata_diag_vals[m][t_diag])
            g_lik = g_lik.at[m].add(sp_d)

            # Arrow AtA contribution
            if sa_i is not None:
                sp_a = jnp.sum(
                    sa_i[per_model_ata_arrow_rows[m][t_diag],
                         m_off + per_model_ata_arrow_cols[m][t_diag]]
                    * per_model_ata_arrow_vals[m][t_diag])
                g_lik = g_lik.at[m].add(2.0 * sp_a)

            # Lower AtA contribution
            if sl_i is not None and t_lower is not None:
                sp_l = jnp.sum(
                    sl_i[m_off + padded_lower_rows[m][t_lower],
                         m_off + padded_lower_cols[m][t_lower]]
                    * padded_lower_vals[m][t_lower])
                g_lik = g_lik.at[m].add(2.0 * sp_l)

        return g_st, g_lik, g_c

    # --- S_tip ---
    L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, eye_nfe, lower=True)
    S_tip = L_tip_inv.T @ L_tip_inv

    # --- Last block ---
    L_last, _, L_arrow_last = reconstruct_L(nt - 1)
    L_blk_inv = jax.scipy.linalg.solve_triangular(L_last, eye_bs, lower=True)
    sa_last = -S_tip @ L_arrow_last @ L_blk_inv
    sd_last = (L_blk_inv.T - sa_last.T @ L_arrow_last) @ L_blk_inv

    g_st_last, g_lik_last, g_c_last = _accumulate_block_grads(
        sd_last, None, sa_last, nt - 1, None)

    # Tip contribution to likelihood
    for m in range(n_models):
        g_lik_last = g_lik_last.at[m].add(jnp.sum(S_tip * per_model_ata_tip[m]))

    grad_per_model_st = g_st_last.copy()
    grad_lik_acc = g_lik_last.copy()
    grad_coreg = g_c_last.copy()

    # --- Backward loop via lax.fori_loop ---
    def bwd_body(i_rev, carry):
        sd_prev, sa_prev, grad_st_acc, grad_lik, grad_c = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i, L_arrow_i = reconstruct_L(i)
        L_blk_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_bs, lower=True)

        sl_i = (-sd_prev @ L_lower_i - sa_prev.T @ L_arrow_i) @ L_blk_inv_i
        sa_i = (-sa_prev @ L_lower_i - S_tip @ L_arrow_i) @ L_blk_inv_i
        sd_i = (L_blk_inv_i.T - sl_i.T @ L_lower_i - sa_i.T @ L_arrow_i) @ L_blk_inv_i

        g_st_i, g_lik_i, g_c_i = _accumulate_block_grads(sd_i, sl_i, sa_i, i, i)

        return (sd_i, sa_i, grad_st_acc + g_st_i, grad_lik + g_lik_i, grad_c + g_c_i)

    carry = (sd_last, sa_last, grad_per_model_st, grad_lik_acc, grad_coreg)
    _, _, grad_per_model_st, grad_lik_acc, grad_coreg = lax.fori_loop(
        0, nt - 1, bwd_body, carry)

    # Scale likelihood grads by prec_m
    grad_per_model_lik = likelihood_precs * grad_lik_acc

    return [grad_per_model_st[m] for m in range(n_models)], grad_per_model_lik, grad_coreg


def _compute_grad_quad_coregional(
    x, sc_list, jac_sc_list, coreg_w, jac_coreg_w,
    n_models, nt, ns, n_fe,
    rhs, likelihood_precs,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_ata_tip, per_model_offsets,
):
    """Compute gradient of quad = rhs^T Q_cond^{-1} rhs for coregional models.

    d(quad)/d(theta_st_m) = -x^T (dQ_st/d(theta_st_m)) x
    d(quad)/d(theta_lik_m) = 2 x^T (d(rhs)/d(theta_lik_m)) - prec_m * x^T AtA_m x
    d(quad)/d(coreg_param) = -x^T (dQ_prior/d(coreg_param)) x

    Parameters
    ----------
    x : (nt * block_size + n_fe,)
    sc_list, jac_sc_list, coreg_w, jac_coreg_w : as above
    n_models, nt, ns, n_fe : int
    rhs : (nt * block_size + n_fe,)
    likelihood_precs : (n_models,)
    per_model_ata_* : per-model sparse COO

    Returns
    -------
    grad_per_model_st : list of (3,) arrays
    grad_per_model_lik : (n_models,)
    grad_coreg : (n_coreg_params,)
    """
    block_size = n_models * ns
    dtype = x.dtype
    n_coreg = jac_coreg_w.shape[-1]

    x_st = x[:nt * block_size].reshape(nt, block_size)
    x_fe = x[nt * block_size:]

    # --- ST gradient: -x^T (dQ_st/dtheta_st_m) x ---
    grad_per_model_st = jnp.zeros((n_models, 3), dtype=dtype)
    grad_coreg = jnp.zeros(n_coreg, dtype=dtype)

    # Pad sc_list subdiags so index nt-1 yields zero for lower blocks
    sc_padded = []
    for m in range(n_models):
        sc_m = sc_list[m]
        sc_padded.append({
            **sc_m,
            'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
        })

    # Pad x_st with a zero row so index nt yields zeros for lower block at t=nt-1
    x_st_padded = jnp.concatenate([x_st, jnp.zeros((1, block_size), dtype=dtype)], axis=0)

    def quad_body(t, carry):
        g_st_acc, g_c_acc = carry
        x_t = x_st[t]
        x_tp1 = x_st_padded[t + 1]
        is_interior = (t < nt - 1).astype(dtype)

        for ii in range(n_models):
            xi = x_t[ii * ns:(ii + 1) * ns]
            xi_next = x_tp1[ii * ns:(ii + 1) * ns]
            for jj in range(n_models):
                xj = x_t[jj * ns:(jj + 1) * ns]
                xj_curr = x_t[jj * ns:(jj + 1) * ns]

                for m_idx in range(n_models):
                    w_ijm = coreg_w[ii, jj, m_idx]
                    sc_m = sc_padded[m_idx]

                    # Diagonal block
                    qu_d = _reconstruct_diag_block(sc_m, t)
                    xQx = xi @ qu_d @ xj

                    dw = jac_coreg_w[ii, jj, m_idx, :]
                    g_c_acc = g_c_acc - dw * xQx

                    # Lower block (zero contribution at t=nt-1 via padding)
                    qu_l = _reconstruct_lower_block(sc_m, t)
                    xQx_l = xi_next @ qu_l @ xj_curr
                    g_c_acc = g_c_acc - 2.0 * is_interior * dw * xQx_l

                    jsc_m = jac_sc_list[m_idx]
                    scale_m = sc_m['scale']
                    exp_gt_m = sc_m['exp_gt']
                    m0_d_val = sc_m['m0_diag'][t]
                    m1_d_val = sc_m['m1_diag'][t]
                    m2_d_val = sc_m['m2_diag'][t]
                    m0_s_val = sc_m['m0_subdiag'][t]
                    m1_s_val = sc_m['m1_subdiag'][t]
                    m2_s_val = sc_m['m2_subdiag'][t]

                    for k in range(3):
                        dq3s = jsc_m['q3s'][..., k]
                        dq2s = jsc_m['q2s'][..., k]
                        dq1s = jsc_m['q1s'][..., k]
                        d_scale = jsc_m['scale'][k]
                        d_exp_gt = jsc_m['exp_gt'][k]

                        dQu = (
                            d_scale * (m0_d_val * sc_m['q3s'] + exp_gt_m * m1_d_val * sc_m['q2s']
                                       + exp_gt_m**2 * m2_d_val * sc_m['q1s'])
                            + scale_m * (m0_d_val * dq3s + exp_gt_m * m1_d_val * dq2s
                                         + exp_gt_m**2 * m2_d_val * dq1s)
                            + scale_m * d_exp_gt * (m1_d_val * sc_m['q2s']
                                                    + 2.0 * exp_gt_m * m2_d_val * sc_m['q1s'])
                        )
                        xdQx = xi @ dQu @ xj
                        g_st_acc = g_st_acc.at[m_idx, k].add(-w_ijm * xdQx)

                        dQu_l = (
                            d_scale * (m0_s_val * sc_m['q3s'] + exp_gt_m * m1_s_val * sc_m['q2s']
                                       + exp_gt_m**2 * m2_s_val * sc_m['q1s'])
                            + scale_m * (m0_s_val * dq3s + exp_gt_m * m1_s_val * dq2s
                                         + exp_gt_m**2 * m2_s_val * dq1s)
                            + scale_m * d_exp_gt * (m1_s_val * sc_m['q2s']
                                                    + 2.0 * exp_gt_m * m2_s_val * sc_m['q1s'])
                        )
                        xdQx_l = xi_next @ dQu_l @ xj_curr
                        g_st_acc = g_st_acc.at[m_idx, k].add(
                            -2.0 * is_interior * w_ijm * xdQx_l)

        return (g_st_acc, g_c_acc)

    grad_per_model_st, grad_coreg = lax.fori_loop(
        0, nt, quad_body, (grad_per_model_st, grad_coreg))

    # --- Likelihood gradient via vectorized sums over time ---
    grad_per_model_lik = jnp.zeros(n_models, dtype=dtype)

    for m in range(n_models):
        m_off = per_model_offsets[m] * ns

        # Vectorized x^T AtA_m x via sparse COO
        xAtAx_d = jnp.sum(
            x_st[jnp.arange(nt)[:, None], m_off + per_model_ata_diag_rows[m]]
            * per_model_ata_diag_vals[m]
            * x_st[jnp.arange(nt)[:, None], m_off + per_model_ata_diag_cols[m]])

        xAtAx_l = jnp.sum(
            x_st[jnp.arange(nt - 1)[:, None] + 1, m_off + per_model_ata_lower_rows[m][:nt - 1]]
            * per_model_ata_lower_vals[m][:nt - 1]
            * x_st[jnp.arange(nt - 1)[:, None], m_off + per_model_ata_lower_cols[m][:nt - 1]])

        xAtAx_a = jnp.sum(
            x_fe[per_model_ata_arrow_rows[m]]
            * per_model_ata_arrow_vals[m]
            * x_st[jnp.arange(nt)[:, None], m_off + per_model_ata_arrow_cols[m]])

        xAtAx_tip = x_fe @ per_model_ata_tip[m] @ x_fe

        xAtAx_m = xAtAx_d + 2.0 * xAtAx_l + 2.0 * xAtAx_a + xAtAx_tip
        grad_per_model_lik = grad_per_model_lik.at[m].set(0.0)

    return [grad_per_model_st[m] for m in range(n_models)], grad_per_model_lik, grad_coreg
