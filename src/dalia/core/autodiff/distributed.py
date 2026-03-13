# Copyright 2024-2025 DALIA authors. All rights reserved.

import jax
import jax.numpy as jnp
from jax import lax

try:
    import mpi4jax
    _MPI4JAX_AVAILABLE = True
except ImportError:
    _MPI4JAX_AVAILABLE = False

from dalia.core.autodiff.spatial_precompute import (
    _jax_cholesky,
    _reconstruct_coregional_diag_block,
    _reconstruct_coregional_lower_block,
)


def pipeline_logdet_Q_prior_coregional_grad(
    sc_list, jac_sc_list, coreg_w, jac_coreg_w,
    n_models, ns, nt_global, dtype,
    rank, comm_size, n_local, start_idx, comm,
):
    """Pipeline-distributed gradient of logdet(Q_prior) for coregional models.

    Forward BT Cholesky storing Schur carries, then backward BT selected
    inversion with boundary communication to accumulate gradient traces.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components (padded subdiags).
    jac_sc_list : list of dict
        Per-model Jacobians.
    coreg_w : (n_models, n_models, n_models)
    jac_coreg_w : (n_models, n_models, n_models, n_coreg_params)
    n_models, ns, nt_global : int
    dtype : jnp.dtype
    rank, comm_size, n_local, start_idx : int
    comm : MPI communicator

    Returns
    -------
    logdet_prior : scalar
        logdet(Q_prior) — accumulated as a byproduct of the forward Cholesky.
    grad_per_model_st : list of (n_theta_st_m,) arrays
    grad_coreg : (n_coreg_params,)
    """
    from mpi4py import MPI

    block_size = n_models * ns
    eye_bs = jnp.eye(block_size, dtype=dtype)
    eps = jnp.finfo(dtype).eps
    n_coreg = jac_coreg_w.shape[-1]
    n_models_cubed = n_models * n_models * n_models

    # Stack spatial components and Jacobians for fori_loop dynamic indexing
    q1s_all = jnp.stack([sc_list[m]['q1s'] for m in range(n_models)])
    q2s_all = jnp.stack([sc_list[m]['q2s'] for m in range(n_models)])
    q3s_all = jnp.stack([sc_list[m]['q3s'] for m in range(n_models)])
    scale_all = jnp.array([sc_list[m]['scale'] for m in range(n_models)])
    exp_gt_all = jnp.array([sc_list[m]['exp_gt'] for m in range(n_models)])
    m0_diag_all = jnp.stack([sc_list[m]['m0_diag'] for m in range(n_models)])
    m1_diag_all = jnp.stack([sc_list[m]['m1_diag'] for m in range(n_models)])
    m2_diag_all = jnp.stack([sc_list[m]['m2_diag'] for m in range(n_models)])
    m0_sub_all = jnp.stack([sc_list[m]['m0_subdiag'] for m in range(n_models)])
    m1_sub_all = jnp.stack([sc_list[m]['m1_subdiag'] for m in range(n_models)])
    m2_sub_all = jnp.stack([sc_list[m]['m2_subdiag'] for m in range(n_models)])
    jac_q1s_all = jnp.stack([jac_sc_list[m]['q1s'] for m in range(n_models)])
    jac_q2s_all = jnp.stack([jac_sc_list[m]['q2s'] for m in range(n_models)])
    jac_q3s_all = jnp.stack([jac_sc_list[m]['q3s'] for m in range(n_models)])
    jac_scale_all = jnp.stack([jac_sc_list[m]['scale'] for m in range(n_models)])
    jac_exp_gt_all = jnp.stack([jac_sc_list[m]['exp_gt'] for m in range(n_models)])

    # --- Forward BT Cholesky, store incoming Schurs + accumulate logdet ---
    global_indices = jnp.arange(start_idx, start_idx + n_local)
    stored_schurs_buf = jnp.zeros((n_local, block_size, block_size), dtype=dtype)

    init_schur = jnp.zeros((block_size, block_size), dtype=dtype)
    init_logdet = jnp.array(0.0, dtype=dtype)
    if rank > 0:
        init_schur = mpi4jax.recv(init_schur, source=rank - 1, tag=30, comm=comm)
        init_logdet = mpi4jax.recv(init_logdet, source=rank - 1, tag=33, comm=comm)

    def fwd_fori_body(j, state):
        schur, stored_, logdet_acc = state
        stored_ = stored_.at[j].set(schur)
        global_i = global_indices[j]
        q_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, global_i) - schur
        L_i = _jax_cholesky(q_diag)
        diag_vals = jnp.diag(L_i)
        safe_vals = jnp.maximum(diag_vals, eps)
        logdet_acc = logdet_acc + 2.0 * jnp.sum(jnp.log(safe_vals))
        q_lower = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, global_i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        new_schur = L_lower_i @ L_lower_i.T
        new_schur = jnp.where(global_i < nt_global - 1, new_schur,
                              jnp.zeros_like(new_schur))
        return new_schur, stored_, logdet_acc

    final_schur, local_stored_schurs, local_logdet = lax.fori_loop(
        0, n_local, fwd_fori_body, (init_schur, stored_schurs_buf, init_logdet))

    if rank < comm_size - 1:
        mpi4jax.send(final_schur, dest=rank + 1, tag=30, comm=comm)
        mpi4jax.send(local_logdet, dest=rank + 1, tag=33, comm=comm)

    logdet_prior = mpi4jax.bcast(local_logdet, root=comm_size - 1, comm=comm)

    def _reconstruct_L_local(local_i):
        global_i = start_idx + local_i
        q_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, global_i) - local_stored_schurs[local_i]
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, global_i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        return L_i, L_lower_i

    def _accumulate_traces(sd_i, sl_i, t_diag, t_lower):
        def _diag_trace_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models

            sd_ij = lax.dynamic_slice(sd_i, (ii * ns, jj * ns), (ns, ns))
            w_ijm = coreg_w[ii, jj, m_idx]

            scale_m = scale_all[m_idx]
            exp_gt_m = exp_gt_all[m_idx]
            m0_d = m0_diag_all[m_idx, t_diag]
            m1_d = m1_diag_all[m_idx, t_diag]
            m2_d = m2_diag_all[m_idx, t_diag]
            q1s_m = q1s_all[m_idx]
            q2s_m = q2s_all[m_idx]
            q3s_m = q3s_all[m_idx]

            base_d = m0_d * q3s_m + exp_gt_m * m1_d * q2s_m + exp_gt_m**2 * m2_d * q1s_m
            tr_val = jnp.sum(sd_ij * (scale_m * base_d).T)

            dw = jac_coreg_w[ii, jj, m_idx, :]
            g_c_ = g_c_ + dw * tr_val

            partial_gt_d = m1_d * q2s_m + 2.0 * exp_gt_m * m2_d * q1s_m
            for k in range(3):
                dq3 = jac_q3s_all[m_idx, :, :, k]
                dq2 = jac_q2s_all[m_idx, :, :, k]
                dq1 = jac_q1s_all[m_idx, :, :, k]
                d_scale = jac_scale_all[m_idx, k]
                d_exp_gt = jac_exp_gt_all[m_idx, k]

                dQu = (
                    d_scale * base_d
                    + scale_m * (m0_d * dq3 + exp_gt_m * m1_d * dq2
                                 + exp_gt_m**2 * m2_d * dq1)
                    + scale_m * d_exp_gt * partial_gt_d
                )
                g_st_ = g_st_.at[m_idx, k].add(w_ijm * jnp.sum(sd_ij * dQu.T))

            return (g_st_, g_c_)

        g_st_init = jnp.zeros((n_models, 3), dtype=dtype)
        g_c_init = jnp.zeros(n_coreg, dtype=dtype)
        g_st, g_c = lax.fori_loop(0, n_models_cubed, _diag_trace_body, (g_st_init, g_c_init))

        if t_lower is not None:
            def _lower_trace_body(flat_idx, acc):
                g_st_, g_c_ = acc
                ii = flat_idx // (n_models * n_models)
                jj = (flat_idx // n_models) % n_models
                m_idx = flat_idx % n_models

                sl_ij = lax.dynamic_slice(sl_i, (ii * ns, jj * ns), (ns, ns))
                w_ijm = coreg_w[ii, jj, m_idx]

                scale_m = scale_all[m_idx]
                exp_gt_m = exp_gt_all[m_idx]
                m0_s = m0_sub_all[m_idx, t_lower]
                m1_s = m1_sub_all[m_idx, t_lower]
                m2_s = m2_sub_all[m_idx, t_lower]
                q1s_m = q1s_all[m_idx]
                q2s_m = q2s_all[m_idx]
                q3s_m = q3s_all[m_idx]

                base_l = m0_s * q3s_m + exp_gt_m * m1_s * q2s_m + exp_gt_m**2 * m2_s * q1s_m
                tr_val_l = jnp.sum(sl_ij * (scale_m * base_l).T)

                dw = jac_coreg_w[ii, jj, m_idx, :]
                g_c_ = g_c_ + 2.0 * dw * tr_val_l

                partial_gt_l = m1_s * q2s_m + 2.0 * exp_gt_m * m2_s * q1s_m
                for k in range(3):
                    dq3 = jac_q3s_all[m_idx, :, :, k]
                    dq2 = jac_q2s_all[m_idx, :, :, k]
                    dq1 = jac_q1s_all[m_idx, :, :, k]
                    d_scale = jac_scale_all[m_idx, k]
                    d_exp_gt = jac_exp_gt_all[m_idx, k]

                    dQu_l = (
                        d_scale * base_l
                        + scale_m * (m0_s * dq3 + exp_gt_m * m1_s * dq2
                                     + exp_gt_m**2 * m2_s * dq1)
                        + scale_m * d_exp_gt * partial_gt_l
                    )
                    g_st_ = g_st_.at[m_idx, k].add(
                        2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))

                return (g_st_, g_c_)

            g_st_l, g_c_l = lax.fori_loop(
                0, n_models_cubed, _lower_trace_body, (g_st_init, g_c_init))
            g_st = g_st + g_st_l
            g_c = g_c + g_c_l

        return g_st, g_c

    # --- Backward SI sweep ---
    sd_boundary = jnp.zeros((block_size, block_size), dtype=dtype)
    if rank < comm_size - 1:
        sd_boundary = mpi4jax.recv(sd_boundary, source=rank + 1, tag=31, comm=comm)

    last_local = n_local - 1
    global_last = start_idx + last_local
    is_global_last = (global_last == nt_global - 1)

    L_last, L_lower_last = _reconstruct_L_local(last_local)
    L_inv_last = jax.scipy.linalg.solve_triangular(L_last, eye_bs, lower=True)

    sd_global_last = L_inv_last.T @ L_inv_last
    sl_from_bnd = -sd_boundary @ L_lower_last @ L_inv_last
    sd_from_bnd = (L_inv_last.T - sl_from_bnd.T @ L_lower_last) @ L_inv_last

    sd_last = jnp.where(is_global_last, sd_global_last, sd_from_bnd)

    g_st_last, g_c_last = _accumulate_traces(sd_last, None, global_last, None)

    safe_idx = jnp.minimum(global_last, nt_global - 2)
    sl_last_for_trace = jnp.where(is_global_last, jnp.zeros_like(sl_from_bnd), sl_from_bnd)
    t_lower_last = jnp.where(is_global_last, global_last, safe_idx)
    g_st_sl, g_c_sl = _accumulate_traces(
        jnp.zeros_like(sd_last), sl_last_for_trace, global_last, t_lower_last)
    g_st_last = g_st_last + jnp.where(is_global_last, jnp.zeros_like(g_st_sl), g_st_sl)
    g_c_last = g_c_last + jnp.where(is_global_last, jnp.zeros_like(g_c_sl), g_c_sl)

    grad_per_model_st = g_st_last
    grad_coreg = g_c_last

    def bwd_body(i_rev, carry):
        sd_prev, grad_st_acc, grad_c_acc = carry
        local_i = last_local - 1 - i_rev
        global_i = start_idx + local_i

        L_i, L_lower_i = _reconstruct_L_local(local_i)
        L_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_bs, lower=True)
        sl_i = -sd_prev @ L_lower_i @ L_inv_i
        sd_i = (L_inv_i.T - sl_i.T @ L_lower_i) @ L_inv_i

        g_st_i, g_c_i = _accumulate_traces(sd_i, sl_i, global_i, global_i)
        return (sd_i, grad_st_acc + g_st_i, grad_c_acc + g_c_i)

    carry = (sd_last, grad_per_model_st, grad_coreg)
    sd_first, grad_per_model_st, grad_coreg = lax.fori_loop(
        0, n_local - 1, bwd_body, carry)

    if rank > 0:
        mpi4jax.send(sd_first, dest=rank - 1, tag=31, comm=comm)

    grad_per_model_st = mpi4jax.allreduce(grad_per_model_st, op=MPI.SUM, comm=comm)
    grad_coreg = mpi4jax.allreduce(grad_coreg, op=MPI.SUM, comm=comm)

    return logdet_prior, [grad_per_model_st[m] for m in range(n_models)], grad_coreg


def pipeline_compute_grad_quad_coregional(
    x, sc_list, jac_sc_list, coreg_w, jac_coreg_w,
    n_models, nt_global, ns, n_fe,
    rhs, likelihood_precs,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_ata_tip, per_model_offsets,
    a_sparse, y, n_observations_idx,
    rank, comm_size, n_local, start_idx, comm,
):
    """Pipeline-distributed gradient of quadratic form for coregional models.

    Parameters
    ----------
    x : (nt_global * block_size + n_fe,)
    sc_list, jac_sc_list : spatial components and Jacobians
    coreg_w, jac_coreg_w : coregional weights and Jacobians
    n_models, nt_global, ns, n_fe : int
    rhs : (nt_global * block_size + n_fe,)
    likelihood_precs : (n_models,)
    per_model_ata_* : local per-model sparse COO data
    per_model_ata_tip : list of (n_fe, n_fe)
    per_model_offsets : list of int
    a_sparse : BCOO sparse matrix
    y : (n_obs,)
    n_observations_idx : list of int
    rank, comm_size, n_local, start_idx : int
    comm : MPI communicator

    Returns
    -------
    grad_per_model_st : list of (3,) arrays
    grad_per_model_lik : (n_models,)
    grad_coreg : (n_coreg_params,)
    """
    from mpi4py import MPI

    block_size = n_models * ns
    dtype = x.dtype
    n_coreg = jac_coreg_w.shape[-1]

    x_st = x[:nt_global * block_size].reshape(nt_global, block_size)
    x_fe = x[nt_global * block_size:]

    x_st_local = x_st[start_idx:start_idx + n_local]

    sc_padded = []
    for m in range(n_models):
        sc_m = sc_list[m]
        sc_padded.append({
            **sc_m,
            'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
        })

    x_st_padded = jnp.concatenate([x_st, jnp.zeros((1, block_size), dtype=dtype)], axis=0)

    n_models_cubed = n_models * n_models * n_models

    # Stack spatial components and Jacobians for fori_loop dynamic indexing
    q1s_all = jnp.stack([sc_padded[m]['q1s'] for m in range(n_models)])
    q2s_all = jnp.stack([sc_padded[m]['q2s'] for m in range(n_models)])
    q3s_all = jnp.stack([sc_padded[m]['q3s'] for m in range(n_models)])
    scale_all = jnp.array([sc_padded[m]['scale'] for m in range(n_models)])
    exp_gt_all = jnp.array([sc_padded[m]['exp_gt'] for m in range(n_models)])
    m0_diag_all = jnp.stack([sc_padded[m]['m0_diag'] for m in range(n_models)])
    m1_diag_all = jnp.stack([sc_padded[m]['m1_diag'] for m in range(n_models)])
    m2_diag_all = jnp.stack([sc_padded[m]['m2_diag'] for m in range(n_models)])
    m0_sub_all = jnp.stack([sc_padded[m]['m0_subdiag'] for m in range(n_models)])
    m1_sub_all = jnp.stack([sc_padded[m]['m1_subdiag'] for m in range(n_models)])
    m2_sub_all = jnp.stack([sc_padded[m]['m2_subdiag'] for m in range(n_models)])
    jac_q1s_all = jnp.stack([jac_sc_list[m]['q1s'] for m in range(n_models)])
    jac_q2s_all = jnp.stack([jac_sc_list[m]['q2s'] for m in range(n_models)])
    jac_q3s_all = jnp.stack([jac_sc_list[m]['q3s'] for m in range(n_models)])
    jac_scale_all = jnp.stack([jac_sc_list[m]['scale'] for m in range(n_models)])
    jac_exp_gt_all = jnp.stack([jac_sc_list[m]['exp_gt'] for m in range(n_models)])

    # --- ST + coreg gradient ---
    grad_per_model_st = jnp.zeros((n_models, 3), dtype=dtype)
    grad_coreg = jnp.zeros(n_coreg, dtype=dtype)

    def quad_body(t_local, carry):
        g_st_acc, g_c_acc = carry
        t = start_idx + t_local
        x_t = x_st[t]
        x_tp1 = x_st_padded[t + 1]
        is_interior = jnp.where(t < nt_global - 1, 1.0, 0.0).astype(dtype)

        def _quad_trace_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models

            xi = lax.dynamic_slice(x_t, (ii * ns,), (ns,))
            xi_next = lax.dynamic_slice(x_tp1, (ii * ns,), (ns,))
            xj = lax.dynamic_slice(x_t, (jj * ns,), (ns,))

            w_ijm = coreg_w[ii, jj, m_idx]

            scale_m = scale_all[m_idx]
            exp_gt_m = exp_gt_all[m_idx]
            m0_d = m0_diag_all[m_idx, t]
            m1_d = m1_diag_all[m_idx, t]
            m2_d = m2_diag_all[m_idx, t]
            m0_s = m0_sub_all[m_idx, t]
            m1_s = m1_sub_all[m_idx, t]
            m2_s = m2_sub_all[m_idx, t]
            q1s_m = q1s_all[m_idx]
            q2s_m = q2s_all[m_idx]
            q3s_m = q3s_all[m_idx]

            base_d = m0_d * q3s_m + exp_gt_m * m1_d * q2s_m + exp_gt_m**2 * m2_d * q1s_m
            qu_d = scale_m * base_d
            xQx = xi @ qu_d @ xj

            dw = jac_coreg_w[ii, jj, m_idx, :]
            g_c_ = g_c_ - dw * xQx

            base_l = m0_s * q3s_m + exp_gt_m * m1_s * q2s_m + exp_gt_m**2 * m2_s * q1s_m
            qu_l = scale_m * base_l
            xQx_l = xi_next @ qu_l @ xj
            g_c_ = g_c_ - 2.0 * is_interior * dw * xQx_l

            partial_gt_d = m1_d * q2s_m + 2.0 * exp_gt_m * m2_d * q1s_m
            partial_gt_l = m1_s * q2s_m + 2.0 * exp_gt_m * m2_s * q1s_m

            for k in range(3):
                dq3 = jac_q3s_all[m_idx, :, :, k]
                dq2 = jac_q2s_all[m_idx, :, :, k]
                dq1 = jac_q1s_all[m_idx, :, :, k]
                d_scale = jac_scale_all[m_idx, k]
                d_exp_gt = jac_exp_gt_all[m_idx, k]

                dQu = (
                    d_scale * base_d
                    + scale_m * (m0_d * dq3 + exp_gt_m * m1_d * dq2
                                 + exp_gt_m**2 * m2_d * dq1)
                    + scale_m * d_exp_gt * partial_gt_d
                )
                g_st_ = g_st_.at[m_idx, k].add(-w_ijm * (xi @ dQu @ xj))

                dQu_l = (
                    d_scale * base_l
                    + scale_m * (m0_s * dq3 + exp_gt_m * m1_s * dq2
                                 + exp_gt_m**2 * m2_s * dq1)
                    + scale_m * d_exp_gt * partial_gt_l
                )
                g_st_ = g_st_.at[m_idx, k].add(
                    -2.0 * is_interior * w_ijm * (xi_next @ dQu_l @ xj))

            return (g_st_, g_c_)

        g_st_acc, g_c_acc = lax.fori_loop(
            0, n_models_cubed, _quad_trace_body, (g_st_acc, g_c_acc))

        return (g_st_acc, g_c_acc)

    grad_per_model_st, grad_coreg = lax.fori_loop(
        0, n_local, quad_body, (grad_per_model_st, grad_coreg))

    # --- Likelihood gradient ---
    grad_per_model_lik = jnp.zeros(n_models, dtype=dtype)

    for m in range(n_models):
        m_off = per_model_offsets[m] * ns
        prec_m = likelihood_precs[m]

        # Local x^T AtA_m x contributions
        local_xAtAx_d = jnp.array(0.0, dtype=dtype)
        for t_local in range(n_local):
            t = start_idx + t_local
            local_xAtAx_d = local_xAtAx_d + jnp.sum(
                x_st[t, m_off + per_model_ata_diag_rows[m][t_local]]
                * per_model_ata_diag_vals[m][t_local]
                * x_st[t, m_off + per_model_ata_diag_cols[m][t_local]])

        local_xAtAx_l = jnp.array(0.0, dtype=dtype)
        for t_local in range(n_local):
            t = start_idx + t_local
            t_next = min(t + 1, nt_global - 1)
            valid = jnp.array(1.0 if t < nt_global - 1 else 0.0, dtype=dtype)
            local_xAtAx_l = local_xAtAx_l + valid * jnp.sum(
                x_st[t_next, m_off + per_model_ata_lower_rows[m][t_local]]
                * per_model_ata_lower_vals[m][t_local]
                * x_st[t, m_off + per_model_ata_lower_cols[m][t_local]])

        local_xAtAx_a = jnp.array(0.0, dtype=dtype)
        for t_local in range(n_local):
            t = start_idx + t_local
            local_xAtAx_a = local_xAtAx_a + jnp.sum(
                x_fe[per_model_ata_arrow_rows[m][t_local]]
                * per_model_ata_arrow_vals[m][t_local]
                * x_st[t, m_off + per_model_ata_arrow_cols[m][t_local]])

        xAtAx_tip = jnp.where(rank == 0, x_fe @ per_model_ata_tip[m] @ x_fe, 0.0)

        local_xAtAx = local_xAtAx_d + 2.0 * local_xAtAx_l + 2.0 * local_xAtAx_a + xAtAx_tip
        xAtAx_m = mpi4jax.allreduce(local_xAtAx, op=MPI.SUM, comm=comm)

        obs_start = n_observations_idx[m]
        obs_end = n_observations_idx[m + 1]
        y_m_weighted = jnp.zeros_like(y)
        y_m_weighted = y_m_weighted.at[obs_start:obs_end].set(y[obs_start:obs_end])
        rhs_m = a_sparse.T @ y_m_weighted
        xTAmy = jnp.dot(x, rhs_m)

        grad_per_model_lik = grad_per_model_lik.at[m].set(
            prec_m * (2.0 * xTAmy - xAtAx_m))

    grad_per_model_st = mpi4jax.allreduce(grad_per_model_st, op=MPI.SUM, comm=comm)
    grad_coreg = mpi4jax.allreduce(grad_coreg, op=MPI.SUM, comm=comm)

    return [grad_per_model_st[m] for m in range(n_models)], grad_per_model_lik, grad_coreg


def twophase_logdet_Q_prior_coregional_scan(
    sc_list, coreg_w, n_models, ns, nt_global, dtype,
    rank, comm_size, n_local, start_idx, comm,
):
    """Two-phase parallel logdet(Q_prior) for coregional model via BT Cholesky.

    Phase 1: All ranks compute BT Cholesky simultaneously (root standard,
    non-root permuted with buffer). Phase 2: Allgather boundary blocks,
    factorize reduced BT system, combine logdets.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components (padded subdiags).
    coreg_w : (n_models, n_models, n_models)
    n_models, ns, nt_global : int
    dtype : jnp.dtype
    rank, comm_size, n_local, start_idx : int
    comm : MPI communicator

    Returns
    -------
    logdet : scalar
    """
    from mpi4py import MPI
    block_size = n_models * ns
    eps = jnp.finfo(dtype).eps

    def _bt_chol_step(schur, global_i):
        q_diag_i = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, global_i) - schur
        L_i = _jax_cholesky(q_diag_i)
        diag_vals = jnp.diag(L_i)
        safe_vals = jnp.maximum(diag_vals, eps)
        logdet_i = 2.0 * jnp.sum(jnp.log(safe_vals))
        q_lower_i = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, global_i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower_i.T, lower=True).T
        new_schur = L_lower_i @ L_lower_i.T
        new_schur = jnp.where(global_i < nt_global - 1, new_schur,
                              jnp.zeros_like(new_schur))
        return L_i, L_lower_i, new_schur, logdet_i

    global_indices = jnp.arange(start_idx, start_idx + n_local)

    if rank == 0:
        # Root: standard BT Cholesky on [0..n_local-2], skip last block
        def root_fwd_body(j, state):
            schur, logdet = state
            global_i = global_indices[j]
            _, _, new_schur, logdet_i = _bt_chol_step(schur, global_i)
            logdet = logdet + logdet_i
            return new_schur, logdet

        init_schur = jnp.zeros((block_size, block_size), dtype=dtype)
        n_interior = jnp.where(n_local > 1, n_local - 1, 0)
        final_schur, local_logdet = lax.fori_loop(
            0, n_interior, root_fwd_body, (init_schur, jnp.array(0.0, dtype=dtype)))

        # Boundary: Schur-complemented last diagonal (not factorized)
        last_gi = start_idx + n_local - 1
        bnd_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, last_gi) - final_schur

        # Pack into reduced system position [1]:
        rs_diag_local = jnp.zeros((2 * comm_size, block_size, block_size), dtype=dtype)
        rs_lower_local = jnp.zeros((2 * comm_size, block_size, block_size), dtype=dtype)
        rs_diag_local = rs_diag_local.at[1].set(bnd_diag)
        last_lower = jnp.where(
            n_local > 1,
            _reconstruct_coregional_lower_block(
                sc_list, coreg_w, n_models, ns, last_gi - 1),
            jnp.zeros((block_size, block_size), dtype=dtype))
        rs_lower_local = rs_lower_local.at[1].set(last_lower)
    else:
        # Non-root: permuted BT Cholesky with buffer on [1..n_local-2]
        # Buffer tracks coupling to block[0]
        first_gi = start_idx
        block0_lower = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, first_gi - 1)
        buffer_init = block0_lower.T  # A_{top, 1} = A_{lower}[start-1].T

        block0_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, first_gi)

        def nonroot_fwd_body(j, state):
            schur, logdet, buf, b0_diag = state
            # j runs from 0 to n_interior-1, mapping to local index j+1
            local_j = j + 1
            global_i = global_indices[local_j]

            L_i, L_lower_i, new_schur, logdet_i = _bt_chol_step(schur, global_i)

            # Buffer propagation: solve_tri(L_i, buf.T, lower=True).T
            buf_solved = jax.scipy.linalg.solve_triangular(
                L_i, buf.T, lower=True).T
            b0_diag = b0_diag - buf_solved @ buf_solved.T
            new_buf = -buf_solved @ L_lower_i.T
            new_buf = jnp.where(global_i < nt_global - 1,
                                new_buf, jnp.zeros_like(new_buf))

            logdet = logdet + logdet_i
            return new_schur, logdet, new_buf, b0_diag

        init_schur_nr = jnp.zeros((block_size, block_size), dtype=dtype)
        # First block (local 1) gets schur from block[0] lower
        first_lower = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, first_gi)
        # We need to factorize block[1] with schur from block[0]
        # but block[0] is a boundary block => we skip block[0] entirely
        # The schur for block[1] comes from the lower block at global first_gi
        # But block[0] is NOT factorized in phase 1, so no schur propagates.
        # Per the algorithm: non-root processes blocks [1..n_local-2]
        # The first block they process (local=1) starts with zero schur
        # because only the boundary block (local=0) connects to previous rank
        n_interior_nr = jnp.where(n_local > 2, n_local - 2, 0)
        final_schur_nr, local_logdet, final_buf, block0_diag_acc = lax.fori_loop(
            0, n_interior_nr, nonroot_fwd_body,
            (init_schur_nr, jnp.array(0.0, dtype=dtype), buffer_init, block0_diag))

        # Boundary extraction
        last_gi = start_idx + n_local - 1
        last_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, last_gi) - final_schur_nr

        rs_diag_local = jnp.zeros((2 * comm_size, block_size, block_size), dtype=dtype)
        rs_lower_local = jnp.zeros((2 * comm_size, block_size, block_size), dtype=dtype)
        rs_diag_local = rs_diag_local.at[2 * rank].set(block0_diag_acc)
        rs_diag_local = rs_diag_local.at[2 * rank + 1].set(last_diag)

        # Lower: buffer[-1].T at position [2*rank], original lower at [2*rank+1]
        rs_lower_local = rs_lower_local.at[2 * rank].set(final_buf.T)
        last_lower = jnp.where(
            rank < comm_size - 1,
            _reconstruct_coregional_lower_block(
                sc_list, coreg_w, n_models, ns, last_gi),
            jnp.zeros((block_size, block_size), dtype=dtype))
        rs_lower_local = rs_lower_local.at[2 * rank + 1].set(last_lower)

    # Phase 2: Allgather + factorize reduced BT system
    rs_diag = mpi4jax.allreduce(rs_diag_local, op=MPI.SUM, comm=comm)
    rs_lower = mpi4jax.allreduce(rs_lower_local, op=MPI.SUM, comm=comm)

    # Factorize reduced BT system [1:2P-1]
    def rs_fwd_body(j, state):
        schur, logdet = state
        idx = j + 1  # reduced system indices [1..2P-1]
        diag_j = rs_diag[idx] - schur
        L_j = _jax_cholesky(diag_j)
        d_vals = jnp.diag(L_j)
        safe_d = jnp.maximum(d_vals, eps)
        logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_d))
        lower_j = rs_lower[idx]
        L_lower_j = jax.scipy.linalg.solve_triangular(
            L_j, lower_j.T, lower=True).T
        new_schur = L_lower_j @ L_lower_j.T
        is_last = (idx == 2 * comm_size - 1)
        new_schur = jnp.where(is_last, jnp.zeros_like(new_schur), new_schur)
        return new_schur, logdet

    _, rs_logdet = lax.fori_loop(
        0, 2 * comm_size - 1, rs_fwd_body,
        (jnp.zeros((block_size, block_size), dtype=dtype), jnp.array(0.0, dtype=dtype)))

    # Allreduce local logdets (each rank computed different interior blocks)
    local_logdet_val = jnp.where(rank == 0, local_logdet, local_logdet)
    total_interior_logdet = mpi4jax.allreduce(local_logdet_val, op=MPI.SUM, comm=comm)

    return total_interior_logdet + rs_logdet
