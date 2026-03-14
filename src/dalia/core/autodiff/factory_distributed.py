# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Callable, Tuple
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from dalia.core.autodiff.config import (
    get_jax_dtype,
    _evaluate_gaussian_likelihood_jax,
    _evaluate_log_prior_hyperparameters_jax,
)
from dalia.core.autodiff.data_extraction import (
    _extract_static_data_distributed_coregional,
)
from dalia.core.autodiff.spatial_precompute import (
    _jax_cholesky,
    precompute_spatial_components,
    precompute_spatial_components_coregional,
    _reconstruct_coregional_diag_block,
    _reconstruct_coregional_lower_block,
)
from dalia.core.autodiff.distributed import (
    twophase_logdet_Q_prior_coregional_scan,
    twophase_logdet_Q_prior_coregional_grad,
    pipeline_logdet_Q_prior_coregional_grad,
    pipeline_compute_grad_quad_coregional,
)

def create_pure_jax_objective_distributed_coregional_twophase(
    dalia_instance, comm, dtype=None
) -> Tuple[Callable, Callable]:
    """Create distributed JAX objective for CoregionalModel using two-phase parallel algorithm.

    All ranks compute simultaneously during local phases, then exchange boundary
    data via collective communication to solve a small reduced system. This gives
    ~3-4x speedup over the pipeline approach for P=4.

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance with CoregionalModel.
    comm : MPI communicator
    dtype : jnp.dtype, optional

    Returns
    -------
    objective_func : Callable
    objective_with_grad : Callable
    """
    from jax import lax
    import mpi4jax
    from mpi4py import MPI

    if dtype is None:
        dtype = get_jax_dtype()
    np_dtype = np.float64 if dtype == jnp.float64 else np.float32

    static_data = _extract_static_data_distributed_coregional(dalia_instance, comm, dtype=dtype)

    n_models = static_data['n_models']
    nt = static_data['nt']
    ns = static_data['ns']
    block_size = static_data['block_size']
    n_fe = static_data['n_fixed_effects_total']
    fe_prec = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a_sparse = static_data['a_sparse']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']

    per_model_ata_diag_rows = static_data['per_model_ata_diag_rows']
    per_model_ata_diag_cols = static_data['per_model_ata_diag_cols']
    per_model_ata_diag_vals = static_data['per_model_ata_diag_vals']
    per_model_ata_lower_rows = static_data['per_model_ata_lower_rows']
    per_model_ata_lower_cols = static_data['per_model_ata_lower_cols']
    per_model_ata_lower_vals = static_data['per_model_ata_lower_vals']
    per_model_ata_arrow_rows = static_data['per_model_ata_arrow_rows']
    per_model_ata_arrow_cols = static_data['per_model_ata_arrow_cols']
    per_model_ata_arrow_vals = static_data['per_model_ata_arrow_vals']
    per_model_ata_tip = static_data['per_model_ata_tip']
    per_model_offsets = static_data['per_model_offsets']

    rank = static_data['rank']
    comm_size = static_data['comm_size']
    n_local = static_data['n_local']
    start_idx = static_data['start_idx']
    _max_n_local = static_data['max_n_local']

    manifolds = [md.get('manifold', 'plane') for md in models_data]
    n_hyperparameters = dalia_instance.model.n_hyperparameters

    sigma_idx = theta_keys.index('sigma_0')
    n_sigmas = n_models
    lambda_keys = [k for k in theta_keys if k.startswith('lambda_')]
    n_lambdas = len(lambda_keys)
    n_coreg_params = n_sigmas + n_lambdas
    lambda_indices = [theta_keys.index(lk) for lk in lambda_keys]

    # Numpy static data for CuPy streaming path
    models_data_np = static_data['models_data_np']
    _ata_diag_rows_np = static_data['per_model_ata_diag_rows_np']
    _ata_diag_cols_np = static_data['per_model_ata_diag_cols_np']
    _ata_diag_vals_np = static_data['per_model_ata_diag_vals_np']
    _ata_lower_rows_np = static_data['per_model_ata_lower_rows_np']
    _ata_lower_cols_np = static_data['per_model_ata_lower_cols_np']
    _ata_lower_vals_np = static_data['per_model_ata_lower_vals_np']
    _ata_arrow_rows_np = static_data['per_model_ata_arrow_rows_np']
    _ata_arrow_cols_np = static_data['per_model_ata_arrow_cols_np']
    _ata_arrow_vals_np = static_data['per_model_ata_arrow_vals_np']
    _ata_tip_np = static_data['per_model_ata_tip_np']
    _offsets_np = static_data['per_model_offsets_np']

    # ---- Shared helpers ----

    def _theta_to_likelihood_precs(theta):
        precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            precs = precs.at[m].set(jnp.exp(theta[prec_idx]))
        return precs

    def _theta_to_sc_coreg(theta):
        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)
        return sc_list, coreg_w

    def _pad_subdiags(sc_list_raw):
        sc_padded = []
        for m_i in range(n_models):
            sc_m = sc_list_raw[m_i]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })
        return sc_padded

    def _theta_to_jac_sc(theta):
        jac_sc_list = []
        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            theta_m = theta[hp_start:hp_end]
            if theta_m.shape[0] == 2:
                theta_m = jnp.concatenate([theta_m, jnp.array([0.0])])
            jac_m = jax.jacfwd(precompute_spatial_components)(
                theta_m, models_data[m]['spatial_matrices'],
                models_data[m]['temporal_matrices'], manifolds[m])
            jac_sc_list.append(jac_m)
        return jac_sc_list

    def _theta_to_jac_sc_padded(theta):
        jac_sc_list = _theta_to_jac_sc(theta)
        jac_sc_list_padded = []
        for m in range(n_models):
            jac_padded = {}
            for key, val in jac_sc_list[m].items():
                if key.endswith('_subdiag'):
                    jac_padded[key] = jnp.concatenate(
                        [val, jnp.zeros((1,) + val.shape[1:], dtype=dtype)], axis=0)
                else:
                    jac_padded[key] = val
            jac_sc_list_padded.append(jac_padded)
        return jac_sc_list_padded

    def _coreg_w_from_params(coreg_params):
        w = jnp.zeros((n_models, n_models, n_models), dtype=dtype)
        sigmas_raw = coreg_params[:n_sigmas]
        sigs = jnp.exp(sigmas_raw)
        if n_models == 2:
            lam01 = coreg_params[n_sigmas]
            s0, s1 = sigs[0], sigs[1]
            w = w.at[0, 0, 0].set(1.0 / s0**2)
            w = w.at[0, 0, 1].set(lam01**2 / s1**2)
            w = w.at[1, 0, 1].set(-lam01 / s1**2)
            w = w.at[0, 1, 1].set(-lam01 / s1**2)
            w = w.at[1, 1, 1].set(1.0 / s1**2)
        elif n_models == 3:
            lam01 = coreg_params[n_sigmas]
            lam02 = coreg_params[n_sigmas + 1]
            lam12 = coreg_params[n_sigmas + 2]
            s0, s1, s2 = sigs[0], sigs[1], sigs[2]
            w = w.at[0, 0, 0].set(1.0 / s0**2)
            w = w.at[0, 0, 1].set(lam01**2 / s1**2)
            w = w.at[0, 0, 2].set(lam12**2 / s2**2)
            w = w.at[1, 0, 1].set(-lam01 / s1**2)
            w = w.at[0, 1, 1].set(-lam01 / s1**2)
            w = w.at[1, 0, 2].set(lam02 * lam12 / s2**2)
            w = w.at[0, 1, 2].set(lam02 * lam12 / s2**2)
            w = w.at[2, 0, 2].set(-lam12 / s2**2)
            w = w.at[0, 2, 2].set(-lam12 / s2**2)
            w = w.at[1, 1, 1].set(1.0 / s1**2)
            w = w.at[1, 1, 2].set(lam02**2 / s2**2)
            w = w.at[2, 1, 2].set(-lam02 / s2**2)
            w = w.at[1, 2, 2].set(-lam02 / s2**2)
            w = w.at[2, 2, 2].set(1.0 / s2**2)
        return w

    def _theta_to_jac_coreg_w(theta):
        coreg_params = jnp.zeros(n_coreg_params, dtype=dtype)
        for m in range(n_models):
            coreg_params = coreg_params.at[m].set(theta[sigma_idx + m])
        for li in range(n_lambdas):
            coreg_params = coreg_params.at[n_sigmas + li].set(theta[lambda_indices[li]])
        return jax.jacfwd(_coreg_w_from_params)(coreg_params)

    def _build_rhs(theta):
        likelihood_precs = _theta_to_likelihood_precs(theta)
        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        return a_sparse.T @ gradient_likelihood

    # ---- Stage 1a: Two-phase Cholesky + forward sub ----

    _padded_lower_rows = []
    _padded_lower_cols = []
    _padded_lower_vals = []
    for m in range(n_models):
        lr = per_model_ata_lower_rows[m]
        lc = per_model_ata_lower_cols[m]
        lv = per_model_ata_lower_vals[m]
        _padded_lower_rows.append(jnp.concatenate([
            lr, jnp.zeros((1, lr.shape[1]), dtype=jnp.int32)], axis=0))
        _padded_lower_cols.append(jnp.concatenate([
            lc, jnp.zeros((1, lc.shape[1]), dtype=jnp.int32)], axis=0))
        _padded_lower_vals.append(jnp.concatenate([
            lv, jnp.zeros((1, lv.shape[1]), dtype=dtype)], axis=0))

    _chol_eye_bs = jnp.eye(block_size, dtype=dtype)
    _chol_eye_nfe = jnp.eye(n_fe, dtype=dtype)
    _chol_eps = jnp.finfo(dtype).eps
    _chol_eps_reg = jnp.where(dtype == jnp.float32, 1e-4, 0.0)

    # Per-block JIT for standard (root) forward Cholesky
    @jax.jit
    def _chol_one_block(
        cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
        prev_lower_y, arrow_rhs_acc_blk,
        j, rhs_st_local,
        q1s_all, q2s_all, q3s_all, scale_all, exp_gt_all,
        m0_diag_all, m1_diag_all, m2_diag_all,
        m0_sub_all, m1_sub_all, m2_sub_all,
        coreg_w_arg, likelihood_precs_arg,
    ):
        global_i = start_idx + j
        rhs_i = rhs_st_local[j]

        sc_loc = []
        for m in range(n_models):
            sc_loc.append({
                'q1s': q1s_all[m], 'q2s': q2s_all[m], 'q3s': q3s_all[m],
                'scale': scale_all[m], 'exp_gt': exp_gt_all[m],
                'm0_diag': m0_diag_all[m], 'm1_diag': m1_diag_all[m],
                'm2_diag': m2_diag_all[m],
                'm0_subdiag': m0_sub_all[m], 'm1_subdiag': m1_sub_all[m],
                'm2_subdiag': m2_sub_all[m],
            })

        q_cond_diag_i = _reconstruct_coregional_diag_block(
            sc_loc, coreg_w_arg, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_cond_diag_i = q_cond_diag_i.at[
                m_off + per_model_ata_diag_rows[m][j],
                m_off + per_model_ata_diag_cols[m][j]
            ].add(likelihood_precs_arg[m] * per_model_ata_diag_vals[m][j])
        q_cond_diag_i = q_cond_diag_i + _chol_eps_reg * _chol_eye_bs - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, _chol_eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        q_cond_lower_i = _reconstruct_coregional_lower_block(
            sc_loc, coreg_w_arg, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_cond_lower_i = q_cond_lower_i.at[
                m_off + _padded_lower_rows[m][j],
                m_off + _padded_lower_cols[m][j]
            ].add(likelihood_precs_arg[m] * _padded_lower_vals[m][j])
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True).T

        q_arrow_i = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_arrow_i = q_arrow_i.at[
                per_model_ata_arrow_rows[m][j],
                m_off + per_model_ata_arrow_cols[m][j]
            ].add(likelihood_precs_arg[m] * per_model_ata_arrow_vals[m][j])
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(global_i < nt - 1,
                                   new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(global_i < nt - 1,
                                    new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)

        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(global_i < nt - 1,
                                     new_prev_lower_y, jnp.zeros_like(new_prev_lower_y))
        new_arrow_rhs_acc = arrow_rhs_acc_blk - L_arrow_i @ y_i

        return (new_cond_schur, new_arrow_tip_acc, new_arrow_schur,
                logdet_cond, new_prev_lower_y, new_arrow_rhs_acc,
                cond_schur, arrow_schur, y_i,
                L_i, L_lower_i, L_arrow_i)

    # Per-block JIT for permuted (non-root) forward Cholesky with buffer
    @jax.jit
    def _chol_one_block_permuted(
        cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
        prev_lower_y, arrow_rhs_acc_blk,
        buffer_in,
        block0_diag_acc, block0_arrow_acc, block0_rhs_acc,
        j, rhs_st_local,
        q1s_all, q2s_all, q3s_all, scale_all, exp_gt_all,
        m0_diag_all, m1_diag_all, m2_diag_all,
        m0_sub_all, m1_sub_all, m2_sub_all,
        coreg_w_arg, likelihood_precs_arg,
    ):
        """Permuted forward Cholesky for non-root ranks with buffer propagation.

        Follows _pobtaf_permuted from serinv: processes interior blocks while
        propagating buffer terms that track coupling to the first (boundary) block.
        """
        global_i = start_idx + j
        rhs_i = rhs_st_local[j]

        sc_loc = []
        for m in range(n_models):
            sc_loc.append({
                'q1s': q1s_all[m], 'q2s': q2s_all[m], 'q3s': q3s_all[m],
                'scale': scale_all[m], 'exp_gt': exp_gt_all[m],
                'm0_diag': m0_diag_all[m], 'm1_diag': m1_diag_all[m],
                'm2_diag': m2_diag_all[m],
                'm0_subdiag': m0_sub_all[m], 'm1_subdiag': m1_sub_all[m],
                'm2_subdiag': m2_sub_all[m],
            })

        # Standard diagonal block reconstruction + Cholesky
        q_cond_diag_i = _reconstruct_coregional_diag_block(
            sc_loc, coreg_w_arg, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_cond_diag_i = q_cond_diag_i.at[
                m_off + per_model_ata_diag_rows[m][j],
                m_off + per_model_ata_diag_cols[m][j]
            ].add(likelihood_precs_arg[m] * per_model_ata_diag_vals[m][j])
        q_cond_diag_i = q_cond_diag_i + _chol_eps_reg * _chol_eye_bs - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, _chol_eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        # Lower factor
        q_cond_lower_i = _reconstruct_coregional_lower_block(
            sc_loc, coreg_w_arg, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_cond_lower_i = q_cond_lower_i.at[
                m_off + _padded_lower_rows[m][j],
                m_off + _padded_lower_cols[m][j]
            ].add(likelihood_precs_arg[m] * _padded_lower_vals[m][j])
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True).T

        # Buffer propagation: L_{top, i} = A_{top, i} @ L_{i,i}^{-T}
        buffer_solved = jax.scipy.linalg.solve_triangular(
            L_i, buffer_in.T, lower=True).T

        # Arrow factor
        q_arrow_i = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_arrow_i = q_arrow_i.at[
                per_model_ata_arrow_rows[m][j],
                m_off + per_model_ata_arrow_cols[m][j]
            ].add(likelihood_precs_arg[m] * per_model_ata_arrow_vals[m][j])
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True).T

        # Standard Schur updates for next block
        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(global_i < nt - 1,
                                   new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(global_i < nt - 1,
                                    new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        # Buffer updates for block[0] accumulation (2-sided pattern)
        # A_{top,top} -= L_{top,i} @ L_{top,i}.T
        new_block0_diag = block0_diag_acc - buffer_solved @ buffer_solved.T
        # A_{top,i+1} = -L_{top,i} @ L_{i+1,i}.T
        new_buffer = -buffer_solved @ L_lower_i.T
        new_buffer = jnp.where(global_i < nt - 1,
                               new_buffer, jnp.zeros_like(new_buffer))
        # A_{ndb+1,top} -= L_{ndb+1,i} @ L_{top,i}.T
        new_block0_arrow = block0_arrow_acc - L_arrow_i @ buffer_solved.T

        # Forward substitution (standard part)
        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)
        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(global_i < nt - 1,
                                     new_prev_lower_y, jnp.zeros_like(new_prev_lower_y))
        new_arrow_rhs_acc = arrow_rhs_acc_blk - L_arrow_i @ y_i

        # Buffer contribution to block[0] RHS
        new_block0_rhs = block0_rhs_acc - buffer_solved @ y_i

        return (new_cond_schur, new_arrow_tip_acc, new_arrow_schur,
                logdet_cond, new_prev_lower_y, new_arrow_rhs_acc,
                buffer_solved, new_buffer,
                new_block0_diag, new_block0_arrow, new_block0_rhs,
                cond_schur, arrow_schur, y_i,
                L_i, L_lower_i, L_arrow_i)

    # ---- Pre-built Q block functions for Path A optimization ----

    @jax.jit
    def _build_q_block(
        j,
        q1s_all, q2s_all, q3s_all, scale_all, exp_gt_all,
        m0_diag_all, m1_diag_all, m2_diag_all,
        m0_sub_all, m1_sub_all, m2_sub_all,
        coreg_w_arg, likelihood_precs_arg,
    ):
        """Build Q_diag, Q_lower, Q_arrow for block j (Q reconstruction only)."""
        global_i = start_idx + j
        sc_loc = []
        for m in range(n_models):
            sc_loc.append({
                'q1s': q1s_all[m], 'q2s': q2s_all[m], 'q3s': q3s_all[m],
                'scale': scale_all[m], 'exp_gt': exp_gt_all[m],
                'm0_diag': m0_diag_all[m], 'm1_diag': m1_diag_all[m],
                'm2_diag': m2_diag_all[m],
                'm0_subdiag': m0_sub_all[m], 'm1_subdiag': m1_sub_all[m],
                'm2_subdiag': m2_sub_all[m],
            })

        q_diag = _reconstruct_coregional_diag_block(
            sc_loc, coreg_w_arg, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_diag = q_diag.at[
                m_off + per_model_ata_diag_rows[m][j],
                m_off + per_model_ata_diag_cols[m][j]
            ].add(likelihood_precs_arg[m] * per_model_ata_diag_vals[m][j])

        q_lower = _reconstruct_coregional_lower_block(
            sc_loc, coreg_w_arg, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_lower = q_lower.at[
                m_off + _padded_lower_rows[m][j],
                m_off + _padded_lower_cols[m][j]
            ].add(likelihood_precs_arg[m] * _padded_lower_vals[m][j])

        q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_arrow = q_arrow.at[
                per_model_ata_arrow_rows[m][j],
                m_off + per_model_ata_arrow_cols[m][j]
            ].add(likelihood_precs_arg[m] * per_model_ata_arrow_vals[m][j])

        return q_diag, q_lower, q_arrow

    @jax.jit
    def _chol_la_block(
        cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
        prev_lower_y, arrow_rhs_acc_blk,
        q_diag, q_lower, q_arrow, rhs_i, global_i,
    ):
        """LA-only Cholesky step on pre-built Q blocks."""
        q_diag = q_diag + _chol_eps_reg * _chol_eye_bs - cond_schur

        L_i = _jax_cholesky(q_diag)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, _chol_eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T

        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, (q_arrow - arrow_schur).T, lower=True).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(global_i < nt - 1,
                                   new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(global_i < nt - 1,
                                    new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)

        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(global_i < nt - 1,
                                     new_prev_lower_y, jnp.zeros_like(new_prev_lower_y))
        new_arrow_rhs_acc = arrow_rhs_acc_blk - L_arrow_i @ y_i

        return (new_cond_schur, new_arrow_tip_acc, new_arrow_schur,
                logdet_cond, new_prev_lower_y, new_arrow_rhs_acc,
                cond_schur, arrow_schur, y_i,
                L_i, L_lower_i, L_arrow_i)

    @jax.jit
    def _chol_la_block_permuted(
        cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
        prev_lower_y, arrow_rhs_acc_blk,
        buffer_in,
        block0_diag_acc, block0_arrow_acc, block0_rhs_acc,
        q_diag, q_lower, q_arrow, rhs_i, global_i,
    ):
        """LA-only permuted Cholesky step on pre-built Q blocks with buffer."""
        q_diag = q_diag + _chol_eps_reg * _chol_eye_bs - cond_schur

        L_i = _jax_cholesky(q_diag)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, _chol_eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T

        buffer_solved = jax.scipy.linalg.solve_triangular(
            L_i, buffer_in.T, lower=True).T

        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, (q_arrow - arrow_schur).T, lower=True).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(global_i < nt - 1,
                                   new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(global_i < nt - 1,
                                    new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        new_block0_diag = block0_diag_acc - buffer_solved @ buffer_solved.T
        new_buffer = -buffer_solved @ L_lower_i.T
        new_buffer = jnp.where(global_i < nt - 1,
                               new_buffer, jnp.zeros_like(new_buffer))
        new_block0_arrow = block0_arrow_acc - L_arrow_i @ buffer_solved.T

        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)
        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(global_i < nt - 1,
                                     new_prev_lower_y, jnp.zeros_like(new_prev_lower_y))
        new_arrow_rhs_acc = arrow_rhs_acc_blk - L_arrow_i @ y_i
        new_block0_rhs = block0_rhs_acc - buffer_solved @ y_i

        return (new_cond_schur, new_arrow_tip_acc, new_arrow_schur,
                logdet_cond, new_prev_lower_y, new_arrow_rhs_acc,
                buffer_solved, new_buffer,
                new_block0_diag, new_block0_arrow, new_block0_rhs,
                cond_schur, arrow_schur, y_i,
                L_i, L_lower_i, L_arrow_i)

    # ---- Configuration flags ----
    _use_prebuild = [False]
    # Auto-select: scan for many blocks (WA1), per-block for few blocks (AP1/WA2)
    _use_fori_loop = [n_local > 48]

    # Chunk size for scan: sized so 4 * 2 * CHUNK * bs^2 * 8 < 50 GiB
    _scan_chunk = max(1, int(50 * 2**30 / (4 * 2 * block_size**2 * 8)))
    _scan_chunk = min(_scan_chunk, 64)

    # Spatial precompute cache: populated by _chol_fn/_chol_fn_fori during the
    # forward Cholesky, then reused by _bwd_and_si_fn, _grad_prior_fn, and
    # _grad_quad_fn to avoid redundant _theta_to_sc_coreg calls (~4s each).
    # Cleared at the start of each objective_with_grad call.
    _sc_cache = {}

    import time as _time_mod  # for diagnostic timing inside _rs_aggregate_and_factorize

    # ---- Reduced system assembly and factorization ----
    # The reduced system has 2*P blocks (2 boundary blocks per rank).
    # For small blocks (WA1, bs=3741): use lax.scan on GPU — all RS data fits.
    # For large blocks (AP1/WA2, bs>10000): use per-block JIT with CPU staging
    # to avoid GPU OOM (RS data would be ~50-70 GiB on GPU).
    n_rs = 2 * comm_size

    @jax.jit
    def _rs_allgather(my_diag, my_lower, my_arrow, my_tip, my_rhs,
                      arrow_tip_initial):
        """Allgather boundary blocks from all ranks."""
        rs_diag = mpi4jax.allgather(my_diag, comm=comm).reshape(n_rs, block_size, block_size)
        rs_lower = mpi4jax.allgather(my_lower, comm=comm).reshape(n_rs, block_size, block_size)
        rs_arrow = mpi4jax.allgather(my_arrow, comm=comm).reshape(n_rs, n_fe, block_size)
        rs_tip = mpi4jax.allreduce(my_tip, op=MPI.SUM, comm=comm)
        rs_rhs = mpi4jax.allgather(my_rhs, comm=comm).reshape(n_rs, block_size)
        rs_tip = rs_tip + arrow_tip_initial
        return rs_diag, rs_lower, rs_arrow, rs_tip, rs_rhs

    @jax.jit
    def _rs_fwd_one_block(schur, arrow_schur, tip_acc, logdet,
                          rhs_carry, arrow_rhs,
                          diag_j, lower_j, arrow_j, rhs_j, is_last):
        """Factorize one block of the reduced system."""
        diag_j = diag_j - schur + _chol_eps_reg * _chol_eye_bs

        L_j = _jax_cholesky(diag_j)
        d_vals = jnp.diag(L_j)
        safe_d = jnp.maximum(d_vals, _chol_eps)
        logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_d))

        L_lower_j = jax.scipy.linalg.solve_triangular(
            L_j, lower_j.T, lower=True).T
        L_arrow_j = jax.scipy.linalg.solve_triangular(
            L_j, (arrow_j - arrow_schur).T, lower=True).T

        new_schur = L_lower_j @ L_lower_j.T
        new_arrow_schur = L_arrow_j @ L_lower_j.T
        tip_acc = tip_acc - L_arrow_j @ L_arrow_j.T

        new_schur = jnp.where(is_last, jnp.zeros_like(new_schur), new_schur)
        new_arrow_schur = jnp.where(is_last,
                                    jnp.zeros_like(new_arrow_schur), new_arrow_schur)

        y_j = jax.scipy.linalg.solve_triangular(L_j, rhs_j - rhs_carry, lower=True)
        new_rhs_carry = L_lower_j @ y_j
        new_rhs_carry = jnp.where(is_last,
                                  jnp.zeros_like(new_rhs_carry), new_rhs_carry)
        arrow_rhs = arrow_rhs - L_arrow_j @ y_j

        return (new_schur, new_arrow_schur, tip_acc, logdet,
                new_rhs_carry, arrow_rhs,
                schur, arrow_schur, y_j, L_j, L_lower_j, L_arrow_j)

    @jax.jit
    def _rs_scan_factorize(rs_diag, rs_lower, rs_arrow, rs_tip, rs_rhs):
        """Scan-based BTA Cholesky on reduced system — all on GPU, no CPU copies."""
        def rs_scan_body(carry, idx):
            schur, arrow_schur, tip_acc, logdet, rhs_carry, arrow_rhs = carry
            diag_j = rs_diag[idx] - schur + _chol_eps_reg * _chol_eye_bs
            L_j = _jax_cholesky(diag_j)
            d_vals = jnp.diag(L_j)
            safe_d = jnp.maximum(d_vals, _chol_eps)
            logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_d))

            lower_j = rs_lower[idx]
            arrow_j = rs_arrow[idx]
            L_lower_j = jax.scipy.linalg.solve_triangular(
                L_j, lower_j.T, lower=True).T
            L_arrow_j = jax.scipy.linalg.solve_triangular(
                L_j, (arrow_j - arrow_schur).T, lower=True).T

            new_schur = L_lower_j @ L_lower_j.T
            new_arrow_schur = L_arrow_j @ L_lower_j.T
            tip_acc = tip_acc - L_arrow_j @ L_arrow_j.T

            is_last = (idx == n_rs - 1)
            new_schur = jnp.where(is_last, jnp.zeros_like(new_schur), new_schur)
            new_arrow_schur = jnp.where(is_last,
                                        jnp.zeros_like(new_arrow_schur), new_arrow_schur)

            rhs_j = rs_rhs[idx]
            y_j = jax.scipy.linalg.solve_triangular(L_j, rhs_j - rhs_carry, lower=True)
            new_rhs_carry = L_lower_j @ y_j
            new_rhs_carry = jnp.where(is_last,
                                      jnp.zeros_like(new_rhs_carry), new_rhs_carry)
            arrow_rhs = arrow_rhs - L_arrow_j @ y_j

            new_carry = (new_schur, new_arrow_schur, tip_acc, logdet,
                         new_rhs_carry, arrow_rhs)
            per_step = (schur, arrow_schur, y_j, L_j, L_lower_j, L_arrow_j)
            return new_carry, per_step

        init_carry = (jnp.zeros((block_size, block_size), dtype=dtype),
                      jnp.zeros((n_fe, block_size), dtype=dtype),
                      rs_tip,
                      jnp.array(0.0, dtype=dtype),
                      jnp.zeros(block_size, dtype=dtype),
                      jnp.zeros(n_fe, dtype=dtype))

        carry_out, (cs_scan, as_scan, y_scan,
                    L_scan, Ll_scan, La_scan) = lax.scan(
            rs_scan_body, init_carry, jnp.arange(1, n_rs))

        (_, _, tip_acc, logdet, _, arrow_rhs) = carry_out

        # Pad scan outputs to full [0..n_rs-1] indexing
        _z_bs2 = jnp.zeros((1, block_size, block_size), dtype=dtype)
        _z_nfe_bs = jnp.zeros((1, n_fe, block_size), dtype=dtype)
        _z_bs1 = jnp.zeros((1, block_size), dtype=dtype)
        rs_stored_cs = jnp.concatenate([_z_bs2, cs_scan], axis=0)
        rs_stored_as = jnp.concatenate([_z_nfe_bs, as_scan], axis=0)
        rs_stored_L = jnp.concatenate([_z_bs2, L_scan], axis=0)
        rs_stored_Ll = jnp.concatenate([_z_bs2, Ll_scan], axis=0)
        rs_stored_La = jnp.concatenate([_z_nfe_bs, La_scan], axis=0)
        rs_y = jnp.concatenate([_z_bs1, y_scan], axis=0)

        L_tip_rs = _jax_cholesky(tip_acc + _chol_eps_reg * _chol_eye_nfe)
        tip_diag = jnp.diag(L_tip_rs)
        safe_tip = jnp.maximum(tip_diag, _chol_eps)
        rs_logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_tip))

        return (rs_stored_cs, rs_stored_as,
                rs_stored_L, rs_stored_Ll, rs_stored_La,
                rs_logdet, L_tip_rs, rs_y, arrow_rhs)

    # RS scan stores all L factors on GPU: (n_rs-1) × 6 arrays × bs² × 8 bytes.
    # For WA1 (bs=3741): ~5 GiB — fits. For AP1/WA2 (bs>12000): ~50+ GiB — OOM.
    _rs_scan_mem = (n_rs - 1) * 6 * block_size**2 * 8
    _use_rs_scan = _rs_scan_mem < 20 * 2**30

    def _rs_aggregate_and_factorize(
        my_diag, my_lower, my_arrow, my_tip, my_rhs,
        arrow_tip_initial,
    ):
        """Allgather boundary blocks from all ranks, then factorize the reduced BTA system.

        Two paths based on block size:
        - Small blocks: single JIT with lax.scan (all data on GPU, no transfers)
        - Large blocks: per-block JIT Python loop with CPU staging (1 block on GPU at a time)

        Returns the same 14-element tuple regardless of path, with JAX or numpy arrays.
        """
        rs_diag, rs_lower, rs_arrow, rs_tip, rs_rhs = _rs_allgather(
            my_diag, my_lower, my_arrow, my_tip, my_rhs, arrow_tip_initial)

        jax.block_until_ready(rs_diag)
        if '_rs_t0' in _chol_detail_timing:
            _chol_detail_timing['rs_allgather'] = _time_mod.perf_counter() - _chol_detail_timing['_rs_t0']
            _chol_detail_timing['_rs_t0'] = _time_mod.perf_counter()

        if _use_rs_scan:
            (rs_stored_cs, rs_stored_as,
             rs_stored_L, rs_stored_Ll, rs_stored_La,
             rs_logdet, L_tip_rs, rs_y, arrow_rhs) = _rs_scan_factorize(
                rs_diag, rs_lower, rs_arrow, rs_tip, rs_rhs)
        else:
            # Per-block JIT fallback for large blocks (keeps 1 block on GPU at a time)
            rs_stored_cs_h = np.zeros((n_rs, block_size, block_size), dtype=np_dtype)
            rs_stored_as_h = np.zeros((n_rs, n_fe, block_size), dtype=np_dtype)
            rs_stored_L_h = np.zeros((n_rs, block_size, block_size), dtype=np_dtype)
            rs_stored_Ll_h = np.zeros((n_rs, block_size, block_size), dtype=np_dtype)
            rs_stored_La_h = np.zeros((n_rs, n_fe, block_size), dtype=np_dtype)
            rs_y_h = np.zeros((n_rs, block_size), dtype=np_dtype)

            schur = jnp.zeros((block_size, block_size), dtype=dtype)
            arrow_schur = jnp.zeros((n_fe, block_size), dtype=dtype)
            tip_acc = rs_tip
            logdet = jnp.array(0.0, dtype=dtype)
            rhs_carry = jnp.zeros(block_size, dtype=dtype)
            arrow_rhs = jnp.zeros(n_fe, dtype=dtype)

            for j_py in range(n_rs - 1):
                idx = j_py + 1
                is_last = jnp.array(idx == n_rs - 1)
                (schur, arrow_schur, tip_acc, logdet,
                 rhs_carry, arrow_rhs,
                 cs_out, as_out, y_out,
                 L_out, Ll_out, La_out) = _rs_fwd_one_block(
                    schur, arrow_schur, tip_acc, logdet,
                    rhs_carry, arrow_rhs,
                    rs_diag[idx], rs_lower[idx], rs_arrow[idx],
                    rs_rhs[idx], is_last)
                rs_stored_cs_h[idx] = np.asarray(cs_out)
                rs_stored_as_h[idx] = np.asarray(as_out)
                rs_stored_L_h[idx] = np.asarray(L_out)
                rs_stored_Ll_h[idx] = np.asarray(Ll_out)
                rs_stored_La_h[idx] = np.asarray(La_out)
                rs_y_h[idx] = np.asarray(y_out)
                del cs_out, as_out, y_out, L_out, Ll_out, La_out

            L_tip_rs = _jax_cholesky(tip_acc + _chol_eps_reg * _chol_eye_nfe)
            tip_diag = jnp.diag(L_tip_rs)
            safe_tip = jnp.maximum(tip_diag, _chol_eps)
            rs_logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_tip))

            rs_stored_cs = rs_stored_cs_h
            rs_stored_as = rs_stored_as_h
            rs_stored_L = rs_stored_L_h
            rs_stored_Ll = rs_stored_Ll_h
            rs_stored_La = rs_stored_La_h
            rs_y = rs_y_h

        return (rs_diag, rs_lower, rs_arrow, rs_tip,
                rs_stored_cs, rs_stored_as,
                rs_stored_L, rs_stored_Ll, rs_stored_La,
                rs_logdet, L_tip_rs, rs_rhs, rs_y, arrow_rhs)

    _chol_detail_timing = {}
    _tp_profile_perblock = [False]
    _use_cupy_streaming = [False]

    def _chol_fn_streaming(theta):
        """CuPy streaming replacement for the two-phase per-block JIT Cholesky loop."""
        import time as _time
        from dalia.core.autodiff.numpy_q_builder import (
            build_q_blocks_np,
            precompute_spatial_components_coregional_np,
            reconstruct_coregional_lower_block_np,
        )
        from dalia.core.autodiff.cupy_streaming_chol import (
            streaming_chol_fwd_sub,
            streaming_chol_fwd_sub_permuted,
        )

        _chol_detail_timing.clear()
        _t0 = _time.perf_counter()

        theta_np = theta.get() if hasattr(theta, 'get') else np.asarray(theta)

        likelihood_precs_np = np.array([
            np.exp(theta_np[hyperparameters_idx[m + 1] - 1])
            for m in range(n_models)])

        q_diag_np, q_lower_np, q_arrow_np, arrow_tip_np, _ = build_q_blocks_np(
            theta_np, n_models, ns, block_size, n_fe, nt,
            models_data_np, hyperparameters_idx, theta_keys, manifolds,
            _ata_diag_rows_np, _ata_diag_cols_np, _ata_diag_vals_np,
            _ata_lower_rows_np, _ata_lower_cols_np, _ata_lower_vals_np,
            _ata_arrow_rows_np, _ata_arrow_cols_np, _ata_arrow_vals_np,
            _ata_tip_np, _offsets_np,
            likelihood_precs_np, float(fe_prec), float(_chol_eps_reg),
            start_idx, n_local, np_dtype,
        )
        arrow_tip_initial_np = arrow_tip_np.copy()

        rhs = _build_rhs(theta)
        rhs_np = np.asarray(rhs)
        rhs_st_global = rhs_np[:nt * block_size].reshape(nt, block_size)
        rhs_fe_np = rhs_np[nt * block_size:]
        rhs_st_local_np = rhs_st_global[start_idx:start_idx + n_local].copy()

        _chol_detail_timing['q_build'] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()

        stored_L_host = np.zeros((n_local, block_size, block_size), dtype=np_dtype)
        stored_Ll_host = np.zeros((n_local, block_size, block_size), dtype=np_dtype)
        stored_La_host = np.zeros((n_local, n_fe, block_size), dtype=np_dtype)
        stored_cs_host = np.empty((n_local, block_size, block_size), dtype=np_dtype)
        stored_as_host = np.empty((n_local, n_fe, block_size), dtype=np_dtype)
        stored_buf_host = np.empty((n_local, block_size, block_size), dtype=np_dtype)

        if rank == 0:
            (L_diag_h, L_lower_h, L_arrow_h, arrow_tip_np,
             y_local_h, arrow_rhs_h, logdet_local,
             cond_schur_h, arrow_schur_h) = streaming_chol_fwd_sub(
                q_diag_np, q_lower_np, q_arrow_np, arrow_tip_np,
                rhs_st_local_np, rhs_fe_np,
                float(_chol_eps_reg), factorize_last=False,
            )

            # Free CuPy GPU memory so JAX can allocate for reduced system
            import gc as _gc
            import cupy as _cp
            _gc.collect()
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()

            # Store L factors for interior blocks [0..n_local-2]
            stored_L_host[:n_local - 1] = L_diag_h[:n_local - 1]
            stored_Ll_host[:n_local - 1] = L_lower_h[:n_local - 1]
            stored_La_host[:n_local - 1] = L_arrow_h[:n_local - 1]

            y_st_host = np.empty((n_local, block_size), dtype=np_dtype)
            y_st_host[:n_local - 1] = y_local_h[:n_local - 1]

            stored_cs_host[n_local - 1] = cond_schur_h
            stored_as_host[n_local - 1] = arrow_schur_h

            # Boundary extraction for reduced system position [1]
            bnd_diag_last = jnp.array(L_diag_h[n_local - 1]) + _chol_eps_reg * _chol_eye_bs
            bnd_lower_last = jnp.array(L_lower_h[n_local - 1])
            bnd_arrow_last = jnp.array(L_arrow_h[n_local - 1])

            if n_local > 1:
                prev_lower_y_np = stored_Ll_host[n_local - 2] @ y_local_h[n_local - 2]
            else:
                prev_lower_y_np = np.zeros(block_size, dtype=np_dtype)
            last_rhs = jnp.array(rhs_st_local_np[n_local - 1] - prev_lower_y_np)

            _zeros_bs = jnp.zeros((block_size, block_size), dtype=dtype)
            _zeros_nfe_bs = jnp.zeros((n_fe, block_size), dtype=dtype)
            _zeros_bs_vec = jnp.zeros(block_size, dtype=dtype)

            my_diag = jnp.stack([_zeros_bs, bnd_diag_last])
            my_lower = jnp.stack([_zeros_bs, bnd_lower_last])
            my_arrow = jnp.stack([_zeros_nfe_bs, bnd_arrow_last])
            arrow_tip_schur = arrow_tip_np - arrow_tip_initial_np
            my_tip = jnp.array(arrow_tip_schur)
            my_rhs = jnp.stack([_zeros_bs_vec, last_rhs])

            arrow_rhs_delta_np = arrow_rhs_h - rhs_fe_np

        else:
            # Build buffer_init for non-root
            sc_list_np, coreg_w_np = precompute_spatial_components_coregional_np(
                theta_np, n_models, ns, models_data_np,
                hyperparameters_idx, theta_keys, manifolds,
            )
            for m in range(n_models):
                sc_list_np[m]['m0_subdiag'] = np.concatenate([
                    sc_list_np[m]['m0_subdiag'], np.zeros(1)])
                sc_list_np[m]['m1_subdiag'] = np.concatenate([
                    sc_list_np[m]['m1_subdiag'], np.zeros(1)])
                sc_list_np[m]['m2_subdiag'] = np.concatenate([
                    sc_list_np[m]['m2_subdiag'], np.zeros(1)])

            buffer_init_np = reconstruct_coregional_lower_block_np(
                sc_list_np, coreg_w_np, n_models, ns, start_idx - 1)
            for m in range(n_models):
                m_off = _offsets_np[m] * ns
                rows = _ata_lower_rows_np[m][0]
                cols = _ata_lower_cols_np[m][0]
                vals = _ata_lower_vals_np[m][0]
                buffer_init_np[m_off + rows, m_off + cols] += (
                    likelihood_precs_np[m] * vals)

            (L_diag_h, L_lower_h, L_arrow_h, buf_solved_h, arrow_tip_np,
             block0_diag_h, block0_arrow_h, block0_rhs_h,
             rhs_last_h,
             y_local_h, arrow_rhs_h, logdet_local,
             cond_schur_h, arrow_schur_h) = streaming_chol_fwd_sub_permuted(
                q_diag_np, q_lower_np, q_arrow_np, arrow_tip_np,
                buffer_init_np, rhs_st_local_np, rhs_fe_np,
                float(_chol_eps_reg),
            )

            # Free CuPy GPU memory so JAX can allocate for reduced system
            import gc as _gc
            import cupy as _cp
            _gc.collect()
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()

            # Store L factors for interior blocks [1..n_local-2]
            stored_L_host[1:n_local - 1] = L_diag_h[1:n_local - 1]
            stored_Ll_host[1:n_local - 1] = L_lower_h[1:n_local - 1]
            stored_La_host[1:n_local - 1] = L_arrow_h[1:n_local - 1]
            stored_buf_host[1:n_local - 1] = buf_solved_h[1:n_local - 1]

            y_st_host = np.empty((n_local, block_size), dtype=np_dtype)
            y_st_host[1:n_local - 1] = y_local_h[1:n_local - 1]

            stored_cs_host[n_local - 1] = cond_schur_h
            stored_as_host[n_local - 1] = arrow_schur_h
            stored_cs_host[0] = np.zeros((block_size, block_size), dtype=np_dtype)
            stored_as_host[0] = np.zeros((n_fe, block_size), dtype=np_dtype)

            # Boundary extraction
            block0_diag_acc = jnp.array(block0_diag_h)
            block0_arrow_acc = jnp.array(block0_arrow_h)
            block0_rhs_acc = jnp.array(block0_rhs_h)

            last_diag = jnp.array(L_diag_h[n_local - 1]) + _chol_eps_reg * _chol_eye_bs
            last_lower = jnp.zeros((block_size, block_size), dtype=dtype)
            if rank < comm_size - 1:
                last_lower = jnp.array(q_lower_np[n_local - 1])
            last_arrow = jnp.array(L_arrow_h[n_local - 1])
            last_rhs = jnp.array(rhs_last_h)

            buffer_cur_np = buf_solved_h[n_local - 1]

            my_diag = jnp.stack([block0_diag_acc, last_diag])
            my_lower = jnp.stack([jnp.array(buffer_cur_np.T), last_lower])
            my_arrow = jnp.stack([block0_arrow_acc, last_arrow])
            arrow_tip_schur = arrow_tip_np - arrow_tip_initial_np
            my_tip = jnp.array(arrow_tip_schur)
            my_rhs = jnp.stack([block0_rhs_acc, last_rhs])

            arrow_rhs_delta_np = arrow_rhs_h - rhs_fe_np

        _chol_detail_timing['local_blocks'] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()

        # Phase 2: Reduced system (stays pure JAX)
        arrow_tip_initial_jax = fe_prec * _chol_eye_nfe + _chol_eps_reg * _chol_eye_nfe
        likelihood_precs = _theta_to_likelihood_precs(theta)
        for m in range(n_models):
            arrow_tip_initial_jax = arrow_tip_initial_jax + likelihood_precs[m] * per_model_ata_tip[m]

        _chol_detail_timing['_rs_t0'] = _time.perf_counter()
        (rs_diag_g, rs_lower_g, rs_arrow_g, rs_tip_final,
         rs_stored_cs, rs_stored_as,
         rs_stored_L, rs_stored_Ll, rs_stored_La,
         rs_logdet, L_tip, rs_rhs_g,
         rs_y, rs_arrow_rhs) = _rs_aggregate_and_factorize(
            my_diag, my_lower, my_arrow, my_tip,
            my_rhs, arrow_tip_initial_jax)

        jax.block_until_ready(rs_logdet)
        if '_rs_t0' in _chol_detail_timing:
            _chol_detail_timing['rs_factorize'] = _time.perf_counter() - _chol_detail_timing['_rs_t0']
        _chol_detail_timing['rs_aggregate'] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()

        # Total logdet
        logdet_local_jax = jnp.array(logdet_local, dtype=dtype)
        total_interior_logdet = mpi4jax.allreduce(logdet_local_jax, op=MPI.SUM, comm=comm)
        logdet_cond_final = total_interior_logdet + rs_logdet

        # Fill y values for boundary blocks from reduced system
        if rank == 0:
            y_st_host[n_local - 1] = np.asarray(rs_y[1])
        else:
            y_st_host[0] = np.asarray(rs_y[2 * rank])
            y_st_host[n_local - 1] = np.asarray(rs_y[2 * rank + 1])
        y_st_local = jnp.array(y_st_host)

        # Update stored_cs/stored_as for boundary blocks
        if rank == 0:
            pass
        else:
            block0_diag_init = q_diag_np[0] + float(_chol_eps_reg) * np.eye(block_size, dtype=np_dtype)
            buffer_schur_np = block0_diag_init - np.asarray(rs_diag_g[2 * rank])
            stored_cs_host[0] = buffer_schur_np + np.asarray(rs_stored_cs[2 * rank])
            arrow_buf_schur_np = q_arrow_np[0] - np.asarray(rs_arrow_g[2 * rank])
            stored_as_host[0] = arrow_buf_schur_np + np.asarray(rs_stored_as[2 * rank])
            stored_cs_host[n_local - 1] = (
                stored_cs_host[n_local - 1] + np.asarray(rs_stored_cs[2 * rank + 1]))
            stored_as_host[n_local - 1] = (
                stored_as_host[n_local - 1] + np.asarray(rs_stored_as[2 * rank + 1]))

        # Arrow RHS
        arrow_rhs_delta_jax = jnp.array(arrow_rhs_delta_np)
        rhs_fe_jax = jnp.array(rhs_fe_np)
        interior_arrow = mpi4jax.allreduce(arrow_rhs_delta_jax, op=MPI.SUM, comm=comm)
        arrow_rhs_global = rhs_fe_jax + interior_arrow + rs_arrow_rhs

        _chol_detail_timing['boundary_update'] = _time.perf_counter() - _t0

        return (stored_L_host, stored_Ll_host, stored_La_host,
                stored_cs_host, stored_as_host, stored_buf_host,
                y_st_local, L_tip, arrow_rhs_global, logdet_cond_final,
                rs_diag_g, rs_lower_g, rs_arrow_g,
                rs_stored_cs, rs_stored_as,
                rs_stored_L, rs_stored_Ll, rs_stored_La,
                rs_y)

    def _chol_fn(theta):
        if _use_cupy_streaming[0]:
            return _chol_fn_streaming(theta)
        import time as _time
        _chol_detail_timing.clear()

        _t0 = _time.perf_counter()
        likelihood_precs = _theta_to_likelihood_precs(theta)
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        sc_padded = _pad_subdiags(sc_list)
        _sc_cache['sc_list'] = sc_list
        _sc_cache['coreg_w'] = coreg_w
        _sc_cache['sc_padded'] = sc_padded
        _sc_cache['likelihood_precs'] = likelihood_precs

        rhs = _build_rhs(theta)
        rhs_st_global = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]
        rhs_st_local = rhs_st_global[start_idx:start_idx + n_local]

        q1s_a = jnp.stack([sc_padded[m]['q1s'] for m in range(n_models)])
        q2s_a = jnp.stack([sc_padded[m]['q2s'] for m in range(n_models)])
        q3s_a = jnp.stack([sc_padded[m]['q3s'] for m in range(n_models)])
        scale_a = jnp.array([sc_padded[m]['scale'] for m in range(n_models)])
        exp_gt_a = jnp.array([sc_padded[m]['exp_gt'] for m in range(n_models)])
        m0d_a = jnp.stack([sc_padded[m]['m0_diag'] for m in range(n_models)])
        m1d_a = jnp.stack([sc_padded[m]['m1_diag'] for m in range(n_models)])
        m2d_a = jnp.stack([sc_padded[m]['m2_diag'] for m in range(n_models)])
        m0s_a = jnp.stack([sc_padded[m]['m0_subdiag'] for m in range(n_models)])
        m1s_a = jnp.stack([sc_padded[m]['m1_subdiag'] for m in range(n_models)])
        m2s_a = jnp.stack([sc_padded[m]['m2_subdiag'] for m in range(n_models)])

        # Initialize carries
        cond_schur = jnp.zeros((block_size, block_size), dtype=dtype)
        arrow_tip_initial = fe_prec * _chol_eye_nfe + _chol_eps_reg * _chol_eye_nfe
        for m in range(n_models):
            arrow_tip_initial = arrow_tip_initial + likelihood_precs[m] * per_model_ata_tip[m]
        arrow_tip_acc = jnp.zeros((n_fe, n_fe), dtype=dtype)  # accumulate Schur updates only
        arrow_schur = jnp.zeros((n_fe, block_size), dtype=dtype)
        logdet_cond = jnp.array(0.0, dtype=dtype)
        prev_lower_y = jnp.zeros(block_size, dtype=dtype)
        arrow_rhs_acc = jnp.zeros(n_fe, dtype=dtype)

        stored_L_host = np.zeros((n_local, block_size, block_size), dtype=np_dtype)
        stored_Ll_host = np.zeros((n_local, block_size, block_size), dtype=np_dtype)
        stored_La_host = np.zeros((n_local, n_fe, block_size), dtype=np_dtype)
        stored_cs_host = np.empty((n_local, block_size, block_size), dtype=np_dtype)
        stored_as_host = np.empty((n_local, n_fe, block_size), dtype=np_dtype)
        stored_buf_host = np.empty((n_local, block_size, block_size), dtype=np_dtype)
        y_st_host = np.empty((n_local, block_size), dtype=np_dtype)

        sc_args = (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
                   m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a)

        _zeros_bs = jnp.zeros((block_size, block_size), dtype=dtype)
        _zeros_nfe_bs = jnp.zeros((n_fe, block_size), dtype=dtype)
        _zeros_bs_vec = jnp.zeros(block_size, dtype=dtype)

        _chol_detail_timing['precompute'] = _time.perf_counter() - _t0

        # Pre-build Q blocks if enabled
        _prebuild = _use_prebuild[0]
        if _prebuild:
            _t0 = _time.perf_counter()
            q_diag_host = np.empty((n_local, block_size, block_size), dtype=np_dtype)
            q_lower_host = np.empty((n_local, block_size, block_size), dtype=np_dtype)
            q_arrow_host = np.empty((n_local, n_fe, block_size), dtype=np_dtype)
            for j_py in range(n_local):
                j_jax = jnp.array(j_py, dtype=jnp.int32)
                qd, ql, qa = _build_q_block(
                    j_jax, *sc_args, coreg_w, likelihood_precs)
                q_diag_host[j_py] = np.asarray(qd)
                q_lower_host[j_py] = np.asarray(ql)
                q_arrow_host[j_py] = np.asarray(qa)
                del qd, ql, qa
            _chol_detail_timing['q_prebuild'] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()

        if rank == 0:
            # Phase 1: Standard forward Cholesky on [0..n_local-2]
            _pb = _tp_profile_perblock[0]
            if _pb:
                import cupy as _cp
                _gpu_sync = _cp.cuda.Device().synchronize
                _blk_compute = []
                _blk_transfer = []
            for j_py in range(n_local - 1):
                if _pb:
                    _gpu_sync()
                    _tb0 = _time.perf_counter()
                if _prebuild:
                    global_i_jax = jnp.array(start_idx + j_py, dtype=jnp.int32)
                    (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                     prev_lower_y, arrow_rhs_acc,
                     cs_out, as_out, y_out,
                     L_out, Ll_out, La_out) = _chol_la_block(
                        cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                        prev_lower_y, arrow_rhs_acc,
                        jnp.array(q_diag_host[j_py]),
                        jnp.array(q_lower_host[j_py]),
                        jnp.array(q_arrow_host[j_py]),
                        rhs_st_local[j_py], global_i_jax)
                else:
                    j_jax = jnp.array(j_py, dtype=jnp.int32)
                    (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                     prev_lower_y, arrow_rhs_acc,
                     cs_out, as_out, y_out,
                     L_out, Ll_out, La_out) = _chol_one_block(
                        cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                        prev_lower_y, arrow_rhs_acc,
                        j_jax, rhs_st_local,
                        *sc_args,
                        coreg_w, likelihood_precs)
                if _pb:
                    jax.block_until_ready((L_out, Ll_out, La_out))
                    _gpu_sync()
                    _blk_compute.append(_time.perf_counter() - _tb0)
                    _tb0 = _time.perf_counter()
                stored_L_host[j_py] = np.asarray(L_out)
                stored_Ll_host[j_py] = np.asarray(Ll_out)
                stored_La_host[j_py] = np.asarray(La_out)
                y_st_host[j_py] = np.asarray(y_out)
                if _pb:
                    _blk_transfer.append(_time.perf_counter() - _tb0)
                del cs_out, as_out, y_out, L_out, Ll_out, La_out
            if _pb:
                _chol_detail_timing['root_blk_compute'] = _blk_compute
                _chol_detail_timing['root_blk_transfer'] = _blk_transfer

            # Last block: store Schur but don't factorize
            stored_cs_host[n_local - 1] = np.asarray(cond_schur)
            stored_as_host[n_local - 1] = np.asarray(arrow_schur)

            # Boundary extraction for reduced system position [1]
            if _prebuild:
                bnd_diag_last = jnp.array(q_diag_host[n_local - 1]) + _chol_eps_reg * _chol_eye_bs - cond_schur
                bnd_lower_last = jnp.array(q_lower_host[n_local - 1])
                bnd_arrow_last = jnp.array(q_arrow_host[n_local - 1]) - arrow_schur
            else:
                global_last = start_idx + n_local - 1
                sc_loc = []
                for m_i in range(n_models):
                    sc_loc.append({
                        'q1s': q1s_a[m_i], 'q2s': q2s_a[m_i], 'q3s': q3s_a[m_i],
                        'scale': scale_a[m_i], 'exp_gt': exp_gt_a[m_i],
                        'm0_diag': m0d_a[m_i], 'm1_diag': m1d_a[m_i], 'm2_diag': m2d_a[m_i],
                        'm0_subdiag': m0s_a[m_i], 'm1_subdiag': m1s_a[m_i], 'm2_subdiag': m2s_a[m_i],
                    })
                bnd_diag_last = _reconstruct_coregional_diag_block(
                    sc_loc, coreg_w, n_models, ns, global_last)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    bnd_diag_last = bnd_diag_last.at[
                        m_off + per_model_ata_diag_rows[m_i][n_local - 1],
                        m_off + per_model_ata_diag_cols[m_i][n_local - 1]
                    ].add(likelihood_precs[m_i] * per_model_ata_diag_vals[m_i][n_local - 1])
                bnd_diag_last = bnd_diag_last + _chol_eps_reg * _chol_eye_bs - cond_schur
                bnd_lower_last = _reconstruct_coregional_lower_block(
                    sc_loc, coreg_w, n_models, ns, global_last)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    bnd_lower_last = bnd_lower_last.at[
                        m_off + _padded_lower_rows[m_i][n_local - 1],
                        m_off + _padded_lower_cols[m_i][n_local - 1]
                    ].add(likelihood_precs[m_i] * _padded_lower_vals[m_i][n_local - 1])
                bnd_arrow_last = jnp.zeros((n_fe, block_size), dtype=dtype)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    bnd_arrow_last = bnd_arrow_last.at[
                        per_model_ata_arrow_rows[m_i][n_local - 1],
                        m_off + per_model_ata_arrow_cols[m_i][n_local - 1]
                    ].add(likelihood_precs[m_i] * per_model_ata_arrow_vals[m_i][n_local - 1])
                bnd_arrow_last = bnd_arrow_last - arrow_schur

            last_rhs = rhs_st_local[n_local - 1] - prev_lower_y

            # Root contributes 1 block at position [1], position [0] is zeros
            my_diag = jnp.stack([_zeros_bs, bnd_diag_last])
            my_lower = jnp.stack([_zeros_bs, bnd_lower_last])
            my_arrow = jnp.stack([_zeros_nfe_bs, bnd_arrow_last])
            my_tip = arrow_tip_acc
            my_rhs = jnp.stack([_zeros_bs_vec, last_rhs])

        else:
            # Phase 1: Permuted forward Cholesky on [1..n_local-2]
            if _prebuild:
                block0_diag = jnp.array(q_diag_host[0]) + _chol_eps_reg * _chol_eye_bs
                block0_arrow = jnp.array(q_arrow_host[0])
                # Buffer init: lower block at start_idx-1
                # _build_q_block for j=0 gives lower at global index start_idx,
                # but we need the lower block at start_idx-1 (incoming from prev rank).
                # This lower block is the SAME as what q_lower_host[0] stores
                # because _build_q_block uses _padded_lower which pads the last entry.
                # Actually, the buffer_init uses start_idx-1 as the global index,
                # which is NOT in our local partition. We must still compute it.
                sc_loc = []
                for m_i in range(n_models):
                    sc_loc.append({
                        'q1s': q1s_a[m_i], 'q2s': q2s_a[m_i], 'q3s': q3s_a[m_i],
                        'scale': scale_a[m_i], 'exp_gt': exp_gt_a[m_i],
                        'm0_diag': m0d_a[m_i], 'm1_diag': m1d_a[m_i], 'm2_diag': m2d_a[m_i],
                        'm0_subdiag': m0s_a[m_i], 'm1_subdiag': m1s_a[m_i], 'm2_subdiag': m2s_a[m_i],
                    })
                buffer_init = _reconstruct_coregional_lower_block(
                    sc_loc, coreg_w, n_models, ns, start_idx - 1)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    buffer_init = buffer_init.at[
                        m_off + _padded_lower_rows[m_i][0],
                        m_off + _padded_lower_cols[m_i][0]
                    ].add(likelihood_precs[m_i] * _padded_lower_vals[m_i][0])
                buffer_cur = buffer_init.T
            else:
                sc_loc = []
                for m_i in range(n_models):
                    sc_loc.append({
                        'q1s': q1s_a[m_i], 'q2s': q2s_a[m_i], 'q3s': q3s_a[m_i],
                        'scale': scale_a[m_i], 'exp_gt': exp_gt_a[m_i],
                        'm0_diag': m0d_a[m_i], 'm1_diag': m1d_a[m_i], 'm2_diag': m2d_a[m_i],
                        'm0_subdiag': m0s_a[m_i], 'm1_subdiag': m1s_a[m_i], 'm2_subdiag': m2s_a[m_i],
                    })
                block0_diag = _reconstruct_coregional_diag_block(
                    sc_loc, coreg_w, n_models, ns, start_idx)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    block0_diag = block0_diag.at[
                        m_off + per_model_ata_diag_rows[m_i][0],
                        m_off + per_model_ata_diag_cols[m_i][0]
                    ].add(likelihood_precs[m_i] * per_model_ata_diag_vals[m_i][0])
                block0_diag = block0_diag + _chol_eps_reg * _chol_eye_bs
                block0_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    block0_arrow = block0_arrow.at[
                        per_model_ata_arrow_rows[m_i][0],
                        m_off + per_model_ata_arrow_cols[m_i][0]
                    ].add(likelihood_precs[m_i] * per_model_ata_arrow_vals[m_i][0])
                buffer_init = _reconstruct_coregional_lower_block(
                    sc_loc, coreg_w, n_models, ns, start_idx - 1)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    buffer_init = buffer_init.at[
                        m_off + _padded_lower_rows[m_i][0],
                        m_off + _padded_lower_cols[m_i][0]
                    ].add(likelihood_precs[m_i] * _padded_lower_vals[m_i][0])
                buffer_cur = buffer_init.T

            block0_diag_acc = block0_diag
            block0_arrow_acc = block0_arrow
            block0_rhs_acc = rhs_st_local[0]

            # Process blocks [1..n_local-2]
            _pb = _tp_profile_perblock[0]
            if _pb:
                import cupy as _cp
                _gpu_sync = _cp.cuda.Device().synchronize
                _blk_compute = []
                _blk_transfer = []
            for j_py in range(1, n_local - 1):
                if _pb:
                    _gpu_sync()
                    _tb0 = _time.perf_counter()
                if _prebuild:
                    global_i_jax = jnp.array(start_idx + j_py, dtype=jnp.int32)
                    (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                     prev_lower_y, arrow_rhs_acc,
                     buf_solved, buffer_cur,
                     block0_diag_acc, block0_arrow_acc, block0_rhs_acc,
                     cs_out, as_out, y_out,
                     L_out, Ll_out, La_out) = _chol_la_block_permuted(
                        cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                        prev_lower_y, arrow_rhs_acc,
                        buffer_cur,
                        block0_diag_acc, block0_arrow_acc, block0_rhs_acc,
                        jnp.array(q_diag_host[j_py]),
                        jnp.array(q_lower_host[j_py]),
                        jnp.array(q_arrow_host[j_py]),
                        rhs_st_local[j_py], global_i_jax)
                else:
                    j_jax = jnp.array(j_py, dtype=jnp.int32)
                    (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                     prev_lower_y, arrow_rhs_acc,
                     buf_solved, buffer_cur,
                     block0_diag_acc, block0_arrow_acc, block0_rhs_acc,
                     cs_out, as_out, y_out,
                     L_out, Ll_out, La_out) = _chol_one_block_permuted(
                        cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                        prev_lower_y, arrow_rhs_acc,
                        buffer_cur,
                        block0_diag_acc, block0_arrow_acc, block0_rhs_acc,
                        j_jax, rhs_st_local,
                        *sc_args,
                        coreg_w, likelihood_precs)
                if _pb:
                    jax.block_until_ready((L_out, Ll_out, La_out))
                    _gpu_sync()
                    _blk_compute.append(_time.perf_counter() - _tb0)
                    _tb0 = _time.perf_counter()
                stored_L_host[j_py] = np.asarray(L_out)
                stored_Ll_host[j_py] = np.asarray(Ll_out)
                stored_La_host[j_py] = np.asarray(La_out)
                stored_buf_host[j_py] = np.asarray(buf_solved)
                y_st_host[j_py] = np.asarray(y_out)
                if _pb:
                    _blk_transfer.append(_time.perf_counter() - _tb0)
                del cs_out, as_out, y_out, buf_solved, L_out, Ll_out, La_out
            if _pb:
                _chol_detail_timing['perm_blk_compute'] = _blk_compute
                _chol_detail_timing['perm_blk_transfer'] = _blk_transfer

            # Last block boundary
            stored_cs_host[n_local - 1] = np.asarray(cond_schur)
            stored_as_host[n_local - 1] = np.asarray(arrow_schur)

            # Block[0] stored schur = 0 (no incoming schur for block[0])
            stored_cs_host[0] = np.zeros((block_size, block_size), dtype=np_dtype)
            stored_as_host[0] = np.zeros((n_fe, block_size), dtype=np_dtype)

            # Boundary extraction
            if _prebuild:
                last_diag = jnp.array(q_diag_host[n_local - 1]) + _chol_eps_reg * _chol_eye_bs - cond_schur
                last_lower = _zeros_bs
                if rank < comm_size - 1:
                    last_lower = jnp.array(q_lower_host[n_local - 1])
                last_arrow = jnp.array(q_arrow_host[n_local - 1]) - arrow_schur
            else:
                global_last = start_idx + n_local - 1
                last_diag = _reconstruct_coregional_diag_block(
                    sc_loc, coreg_w, n_models, ns, global_last)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    last_diag = last_diag.at[
                        m_off + per_model_ata_diag_rows[m_i][n_local - 1],
                        m_off + per_model_ata_diag_cols[m_i][n_local - 1]
                    ].add(likelihood_precs[m_i] * per_model_ata_diag_vals[m_i][n_local - 1])
                last_diag = last_diag + _chol_eps_reg * _chol_eye_bs - cond_schur
                last_lower = _zeros_bs
                if rank < comm_size - 1:
                    last_lower = _reconstruct_coregional_lower_block(
                        sc_loc, coreg_w, n_models, ns, global_last)
                    for m_i in range(n_models):
                        m_off = per_model_offsets[m_i] * ns
                        last_lower = last_lower.at[
                            m_off + _padded_lower_rows[m_i][n_local - 1],
                            m_off + _padded_lower_cols[m_i][n_local - 1]
                        ].add(likelihood_precs[m_i] * _padded_lower_vals[m_i][n_local - 1])
                last_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    last_arrow = last_arrow.at[
                        per_model_ata_arrow_rows[m_i][n_local - 1],
                        m_off + per_model_ata_arrow_cols[m_i][n_local - 1]
                    ].add(likelihood_precs[m_i] * per_model_ata_arrow_vals[m_i][n_local - 1])
                last_arrow = last_arrow - arrow_schur

            last_rhs = rhs_st_local[n_local - 1] - prev_lower_y

            my_diag = jnp.stack([block0_diag_acc, last_diag])
            my_lower = jnp.stack([buffer_cur.T, last_lower])
            my_arrow = jnp.stack([block0_arrow_acc, last_arrow])
            my_tip = arrow_tip_acc
            my_rhs = jnp.stack([block0_rhs_acc, last_rhs])

        _chol_detail_timing['local_blocks'] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()
        _chol_detail_timing['_rs_t0'] = _t0

        # Phase 2: Allgather (2, bs, bs) per rank and factorize reduced system
        (rs_diag_g, rs_lower_g, rs_arrow_g, rs_tip_final,
         rs_stored_cs, rs_stored_as,
         rs_stored_L, rs_stored_Ll, rs_stored_La,
         rs_logdet, L_tip, rs_rhs_g,
         rs_y, rs_arrow_rhs) = _rs_aggregate_and_factorize(
            my_diag, my_lower, my_arrow, my_tip,
            my_rhs, arrow_tip_initial)

        jax.block_until_ready(rs_logdet)
        if '_rs_t0' in _chol_detail_timing:
            _chol_detail_timing['rs_factorize'] = _time.perf_counter() - _chol_detail_timing['_rs_t0']
        _chol_detail_timing['rs_aggregate'] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()

        # Total logdet = local interior logdet + reduced system logdet
        local_logdet_j = jnp.array(logdet_cond) if isinstance(logdet_cond, (float, int)) else logdet_cond
        total_interior_logdet = mpi4jax.allreduce(local_logdet_j, op=MPI.SUM, comm=comm)
        logdet_cond_final = total_interior_logdet + rs_logdet

        # Fill y values for boundary blocks from reduced system
        if rank == 0:
            y_st_host[n_local - 1] = np.asarray(rs_y[1])
        else:
            y_st_host[0] = np.asarray(rs_y[2 * rank])
            y_st_host[n_local - 1] = np.asarray(rs_y[2 * rank + 1])
        y_st_local = jnp.array(y_st_host)

        # Update stored_cs/stored_as for boundary blocks so backward sub
        # can reconstruct the correct L factors
        if rank == 0:
            # Root's last block: rs_stored_cs[1] = 0, no change needed
            pass
        else:
            # Non-root block[0]: stored_cs = buffer_Schur + reduced_system_Schur
            # block0_diag holds initial Q_diag + AtA + eps
            # block0_diag_acc (= my_diag[0] after allgather = rs_diag_g[2*rank]) is the accumulated value
            buffer_schur_np = np.asarray(block0_diag) - np.asarray(rs_diag_g[2 * rank])
            stored_cs_host[0] = buffer_schur_np + np.asarray(rs_stored_cs[2 * rank])
            # Arrow schur similarly
            arrow_buf_schur_np = np.asarray(block0_arrow) - np.asarray(rs_arrow_g[2 * rank])
            stored_as_host[0] = arrow_buf_schur_np + np.asarray(rs_stored_as[2 * rank])
            # Non-root last block: add reduced system Schur
            stored_cs_host[n_local - 1] = (
                stored_cs_host[n_local - 1] + np.asarray(rs_stored_cs[2 * rank + 1]))
            stored_as_host[n_local - 1] = (
                stored_as_host[n_local - 1] + np.asarray(rs_stored_as[2 * rank + 1]))

        # Arrow RHS: rhs_fe + interior contributions + boundary contributions
        interior_arrow = mpi4jax.allreduce(arrow_rhs_acc, op=MPI.SUM, comm=comm)
        arrow_rhs_global = rhs_fe + interior_arrow + rs_arrow_rhs

        _chol_detail_timing['boundary_update'] = _time.perf_counter() - _t0

        return (stored_L_host, stored_Ll_host, stored_La_host,
                stored_cs_host, stored_as_host, stored_buf_host,
                y_st_local, L_tip, arrow_rhs_global, logdet_cond_final,
                rs_diag_g, rs_lower_g, rs_arrow_g,
                rs_stored_cs, rs_stored_as,
                rs_stored_L, rs_stored_Ll, rs_stored_La,
                rs_y)

    # ---- scan-based forward Cholesky ----

    @jax.jit
    def _chol_fori_root_interior(
        cond_schur_init, arrow_tip_init, arrow_schur_init,
        logdet_init, prev_lower_y_init, arrow_rhs_init,
        rhs_st_local,
        q1s_all, q2s_all, q3s_all, scale_all, exp_gt_all,
        m0_diag_all, m1_diag_all, m2_diag_all,
        m0_sub_all, m1_sub_all, m2_sub_all,
        coreg_w_arg, likelihood_precs_arg,
        indices,
    ):
        def body_fn(carry, j):
            (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
             prev_lower_y, arrow_rhs_acc) = carry
            global_i = start_idx + j
            rhs_i = rhs_st_local[j]

            sc_loc = []
            for m in range(n_models):
                sc_loc.append({
                    'q1s': q1s_all[m], 'q2s': q2s_all[m], 'q3s': q3s_all[m],
                    'scale': scale_all[m], 'exp_gt': exp_gt_all[m],
                    'm0_diag': m0_diag_all[m], 'm1_diag': m1_diag_all[m],
                    'm2_diag': m2_diag_all[m],
                    'm0_subdiag': m0_sub_all[m], 'm1_subdiag': m1_sub_all[m],
                    'm2_subdiag': m2_sub_all[m],
                })

            q_diag = _reconstruct_coregional_diag_block(
                sc_loc, coreg_w_arg, n_models, ns, global_i)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_diag = q_diag.at[
                    m_off + per_model_ata_diag_rows[m][j],
                    m_off + per_model_ata_diag_cols[m][j]
                ].add(likelihood_precs_arg[m] * per_model_ata_diag_vals[m][j])
            q_diag = q_diag + _chol_eps_reg * _chol_eye_bs - cond_schur

            L_i = _jax_cholesky(q_diag)
            cond_diag_vals = jnp.diag(L_i)
            safe_cond = jnp.maximum(cond_diag_vals, _chol_eps)
            logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

            q_lower = _reconstruct_coregional_lower_block(
                sc_loc, coreg_w_arg, n_models, ns, global_i)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_lower = q_lower.at[
                    m_off + _padded_lower_rows[m][j],
                    m_off + _padded_lower_cols[m][j]
                ].add(likelihood_precs_arg[m] * _padded_lower_vals[m][j])
            L_lower_i = jax.scipy.linalg.solve_triangular(
                L_i, q_lower.T, lower=True).T

            q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_arrow = q_arrow.at[
                    per_model_ata_arrow_rows[m][j],
                    m_off + per_model_ata_arrow_cols[m][j]
                ].add(likelihood_precs_arg[m] * per_model_ata_arrow_vals[m][j])
            q_arrow = q_arrow - arrow_schur
            L_arrow_i = jax.scipy.linalg.solve_triangular(
                L_i, q_arrow.T, lower=True).T

            new_cond_schur = L_lower_i @ L_lower_i.T
            new_arrow_schur = L_arrow_i @ L_lower_i.T
            new_arrow_tip = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

            new_cond_schur = jnp.where(global_i < nt - 1,
                                       new_cond_schur, jnp.zeros_like(new_cond_schur))
            new_arrow_schur = jnp.where(global_i < nt - 1,
                                        new_arrow_schur, jnp.zeros_like(new_arrow_schur))

            modified_rhs = rhs_i - prev_lower_y
            y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs, lower=True)

            new_prev_lower_y = L_lower_i @ y_i
            new_prev_lower_y = jnp.where(global_i < nt - 1,
                                         new_prev_lower_y, jnp.zeros_like(new_prev_lower_y))
            new_arrow_rhs = arrow_rhs_acc - L_arrow_i @ y_i

            new_carry = (new_cond_schur, new_arrow_tip, new_arrow_schur,
                         logdet_cond, new_prev_lower_y, new_arrow_rhs)
            per_step = (cond_schur, arrow_schur, y_i)
            return new_carry, per_step

        init_carry = (cond_schur_init, arrow_tip_init, arrow_schur_init,
                      logdet_init, prev_lower_y_init, arrow_rhs_init)
        carry_out, (cs_all, as_all, y_all) = lax.scan(
            body_fn, init_carry, indices)

        return carry_out, (cs_all, as_all, y_all)

    @jax.jit
    def _chol_fori_nonroot_interior(
        cond_schur_init, arrow_tip_init, arrow_schur_init,
        logdet_init, prev_lower_y_init, arrow_rhs_init,
        buffer_init_arg,
        block0_diag_init, block0_arrow_init, block0_rhs_init,
        rhs_st_local,
        q1s_all, q2s_all, q3s_all, scale_all, exp_gt_all,
        m0_diag_all, m1_diag_all, m2_diag_all,
        m0_sub_all, m1_sub_all, m2_sub_all,
        coreg_w_arg, likelihood_precs_arg,
        indices,
    ):
        def body_fn(carry, j_offset):
            (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
             prev_lower_y, arrow_rhs_acc,
             buffer_cur,
             block0_diag_acc, block0_arrow_acc, block0_rhs_acc) = carry
            j = j_offset + 1
            global_i = start_idx + j
            rhs_i = rhs_st_local[j]

            sc_loc = []
            for m in range(n_models):
                sc_loc.append({
                    'q1s': q1s_all[m], 'q2s': q2s_all[m], 'q3s': q3s_all[m],
                    'scale': scale_all[m], 'exp_gt': exp_gt_all[m],
                    'm0_diag': m0_diag_all[m], 'm1_diag': m1_diag_all[m],
                    'm2_diag': m2_diag_all[m],
                    'm0_subdiag': m0_sub_all[m], 'm1_subdiag': m1_sub_all[m],
                    'm2_subdiag': m2_sub_all[m],
                })

            q_diag = _reconstruct_coregional_diag_block(
                sc_loc, coreg_w_arg, n_models, ns, global_i)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_diag = q_diag.at[
                    m_off + per_model_ata_diag_rows[m][j],
                    m_off + per_model_ata_diag_cols[m][j]
                ].add(likelihood_precs_arg[m] * per_model_ata_diag_vals[m][j])
            q_diag = q_diag + _chol_eps_reg * _chol_eye_bs - cond_schur

            L_i = _jax_cholesky(q_diag)
            cond_diag_vals = jnp.diag(L_i)
            safe_cond = jnp.maximum(cond_diag_vals, _chol_eps)
            logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

            q_lower = _reconstruct_coregional_lower_block(
                sc_loc, coreg_w_arg, n_models, ns, global_i)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_lower = q_lower.at[
                    m_off + _padded_lower_rows[m][j],
                    m_off + _padded_lower_cols[m][j]
                ].add(likelihood_precs_arg[m] * _padded_lower_vals[m][j])
            L_lower_i = jax.scipy.linalg.solve_triangular(
                L_i, q_lower.T, lower=True).T

            buffer_solved = jax.scipy.linalg.solve_triangular(
                L_i, buffer_cur.T, lower=True).T

            q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_arrow = q_arrow.at[
                    per_model_ata_arrow_rows[m][j],
                    m_off + per_model_ata_arrow_cols[m][j]
                ].add(likelihood_precs_arg[m] * per_model_ata_arrow_vals[m][j])
            q_arrow = q_arrow - arrow_schur
            L_arrow_i = jax.scipy.linalg.solve_triangular(
                L_i, q_arrow.T, lower=True).T

            new_cond_schur = L_lower_i @ L_lower_i.T
            new_arrow_schur = L_arrow_i @ L_lower_i.T
            new_arrow_tip = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

            new_cond_schur = jnp.where(global_i < nt - 1,
                                       new_cond_schur, jnp.zeros_like(new_cond_schur))
            new_arrow_schur = jnp.where(global_i < nt - 1,
                                        new_arrow_schur, jnp.zeros_like(new_arrow_schur))

            new_block0_diag = block0_diag_acc - buffer_solved @ buffer_solved.T
            new_buffer = -buffer_solved @ L_lower_i.T
            new_buffer = jnp.where(global_i < nt - 1,
                                   new_buffer, jnp.zeros_like(new_buffer))
            new_block0_arrow = block0_arrow_acc - L_arrow_i @ buffer_solved.T

            modified_rhs = rhs_i - prev_lower_y
            y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs, lower=True)
            new_prev_lower_y = L_lower_i @ y_i
            new_prev_lower_y = jnp.where(global_i < nt - 1,
                                         new_prev_lower_y, jnp.zeros_like(new_prev_lower_y))
            new_arrow_rhs = arrow_rhs_acc - L_arrow_i @ y_i
            new_block0_rhs = block0_rhs_acc - buffer_solved @ y_i

            new_carry = (new_cond_schur, new_arrow_tip, new_arrow_schur,
                         logdet_cond, new_prev_lower_y, new_arrow_rhs,
                         new_buffer,
                         new_block0_diag, new_block0_arrow, new_block0_rhs)
            per_step = (cond_schur, arrow_schur, y_i, buffer_solved)
            return new_carry, per_step

        init_carry = (cond_schur_init, arrow_tip_init, arrow_schur_init,
                      logdet_init, prev_lower_y_init, arrow_rhs_init,
                      buffer_init_arg,
                      block0_diag_init, block0_arrow_init, block0_rhs_init)
        carry_out, (cs_all, as_all, y_all, buf_all) = lax.scan(
            body_fn, init_carry, indices)

        return carry_out, (cs_all, as_all, y_all, buf_all)

    def _chol_fn_fori(theta):
        import time as _time
        _chol_detail_timing.clear()

        _t0 = _time.perf_counter()
        likelihood_precs = _theta_to_likelihood_precs(theta)
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        sc_padded = _pad_subdiags(sc_list)
        _sc_cache['sc_list'] = sc_list
        _sc_cache['coreg_w'] = coreg_w
        _sc_cache['sc_padded'] = sc_padded
        _sc_cache['likelihood_precs'] = likelihood_precs

        rhs = _build_rhs(theta)
        rhs_st_global = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]
        rhs_st_local = rhs_st_global[start_idx:start_idx + n_local]

        q1s_a = jnp.stack([sc_padded[m]['q1s'] for m in range(n_models)])
        q2s_a = jnp.stack([sc_padded[m]['q2s'] for m in range(n_models)])
        q3s_a = jnp.stack([sc_padded[m]['q3s'] for m in range(n_models)])
        scale_a = jnp.array([sc_padded[m]['scale'] for m in range(n_models)])
        exp_gt_a = jnp.array([sc_padded[m]['exp_gt'] for m in range(n_models)])
        m0d_a = jnp.stack([sc_padded[m]['m0_diag'] for m in range(n_models)])
        m1d_a = jnp.stack([sc_padded[m]['m1_diag'] for m in range(n_models)])
        m2d_a = jnp.stack([sc_padded[m]['m2_diag'] for m in range(n_models)])
        m0s_a = jnp.stack([sc_padded[m]['m0_subdiag'] for m in range(n_models)])
        m1s_a = jnp.stack([sc_padded[m]['m1_subdiag'] for m in range(n_models)])
        m2s_a = jnp.stack([sc_padded[m]['m2_subdiag'] for m in range(n_models)])
        sc_args = (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
                   m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a)

        cond_schur = jnp.zeros((block_size, block_size), dtype=dtype)
        arrow_tip_initial = fe_prec * _chol_eye_nfe + _chol_eps_reg * _chol_eye_nfe
        for m in range(n_models):
            arrow_tip_initial = arrow_tip_initial + likelihood_precs[m] * per_model_ata_tip[m]
        arrow_tip_acc = jnp.zeros((n_fe, n_fe), dtype=dtype)
        arrow_schur = jnp.zeros((n_fe, block_size), dtype=dtype)
        logdet_cond = jnp.array(0.0, dtype=dtype)
        prev_lower_y = jnp.zeros(block_size, dtype=dtype)
        arrow_rhs_acc = jnp.zeros(n_fe, dtype=dtype)

        _chol_detail_timing['precompute'] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()

        chunk = _scan_chunk

        if rank == 0:
            n_interior = n_local - 1
            cs_all_h = np.zeros((n_interior, block_size, block_size), dtype=np_dtype)
            as_all_h = np.zeros((n_interior, n_fe, block_size), dtype=np_dtype)
            y_all_h = np.zeros((n_interior, block_size), dtype=np_dtype)

            carry = (cond_schur, arrow_tip_acc, arrow_schur,
                     logdet_cond, prev_lower_y, arrow_rhs_acc)
            for chunk_start in range(0, n_interior, chunk):
                chunk_end = min(chunk_start + chunk, n_interior)
                indices = jnp.arange(chunk_start, chunk_end)
                carry, (cs_chunk, as_chunk, y_chunk) = _chol_fori_root_interior(
                    *carry, rhs_st_local, *sc_args, coreg_w, likelihood_precs,
                    indices)
                cs_all_h[chunk_start:chunk_end] = np.asarray(cs_chunk)
                as_all_h[chunk_start:chunk_end] = np.asarray(as_chunk)
                y_all_h[chunk_start:chunk_end] = np.asarray(y_chunk)
                del cs_chunk, as_chunk, y_chunk

            (cond_schur, arrow_tip_acc, arrow_schur,
             logdet_cond, prev_lower_y, arrow_rhs_acc) = carry

            y_st_host = np.zeros((n_local, block_size), dtype=np_dtype)
            y_st_host[:n_interior] = y_all_h

            # Boundary: last block not factorized, goes to reduced system
            sc_loc = []
            for m_i in range(n_models):
                sc_loc.append({
                    'q1s': q1s_a[m_i], 'q2s': q2s_a[m_i], 'q3s': q3s_a[m_i],
                    'scale': scale_a[m_i], 'exp_gt': exp_gt_a[m_i],
                    'm0_diag': m0d_a[m_i], 'm1_diag': m1d_a[m_i], 'm2_diag': m2d_a[m_i],
                    'm0_subdiag': m0s_a[m_i], 'm1_subdiag': m1s_a[m_i], 'm2_subdiag': m2s_a[m_i],
                })
            global_last = start_idx + n_local - 1
            bnd_diag_last = _reconstruct_coregional_diag_block(
                sc_loc, coreg_w, n_models, ns, global_last)
            for m_i in range(n_models):
                m_off = per_model_offsets[m_i] * ns
                bnd_diag_last = bnd_diag_last.at[
                    m_off + per_model_ata_diag_rows[m_i][n_local - 1],
                    m_off + per_model_ata_diag_cols[m_i][n_local - 1]
                ].add(likelihood_precs[m_i] * per_model_ata_diag_vals[m_i][n_local - 1])
            bnd_diag_last = bnd_diag_last + _chol_eps_reg * _chol_eye_bs - cond_schur
            bnd_lower_last = _reconstruct_coregional_lower_block(
                sc_loc, coreg_w, n_models, ns, global_last)
            for m_i in range(n_models):
                m_off = per_model_offsets[m_i] * ns
                bnd_lower_last = bnd_lower_last.at[
                    m_off + _padded_lower_rows[m_i][n_local - 1],
                    m_off + _padded_lower_cols[m_i][n_local - 1]
                ].add(likelihood_precs[m_i] * _padded_lower_vals[m_i][n_local - 1])
            bnd_arrow_last = jnp.zeros((n_fe, block_size), dtype=dtype)
            for m_i in range(n_models):
                m_off = per_model_offsets[m_i] * ns
                bnd_arrow_last = bnd_arrow_last.at[
                    per_model_ata_arrow_rows[m_i][n_local - 1],
                    m_off + per_model_ata_arrow_cols[m_i][n_local - 1]
                ].add(likelihood_precs[m_i] * per_model_ata_arrow_vals[m_i][n_local - 1])
            bnd_arrow_last = bnd_arrow_last - arrow_schur

            last_rhs = rhs_st_local[n_local - 1] - prev_lower_y

            _zeros_bs = jnp.zeros((block_size, block_size), dtype=dtype)
            _zeros_nfe_bs = jnp.zeros((n_fe, block_size), dtype=dtype)
            _zeros_bs_vec = jnp.zeros(block_size, dtype=dtype)
            my_diag = jnp.stack([_zeros_bs, bnd_diag_last])
            my_lower = jnp.stack([_zeros_bs, bnd_lower_last])
            my_arrow = jnp.stack([_zeros_nfe_bs, bnd_arrow_last])
            my_tip = arrow_tip_acc
            my_rhs = jnp.stack([_zeros_bs_vec, last_rhs])
            buf_all_h = np.zeros((1, block_size, block_size), dtype=np_dtype)

        else:
            # Compute buffer_init and block0 initial values
            sc_loc = []
            for m_i in range(n_models):
                sc_loc.append({
                    'q1s': q1s_a[m_i], 'q2s': q2s_a[m_i], 'q3s': q3s_a[m_i],
                    'scale': scale_a[m_i], 'exp_gt': exp_gt_a[m_i],
                    'm0_diag': m0d_a[m_i], 'm1_diag': m1d_a[m_i], 'm2_diag': m2d_a[m_i],
                    'm0_subdiag': m0s_a[m_i], 'm1_subdiag': m1s_a[m_i], 'm2_subdiag': m2s_a[m_i],
                })
            block0_diag = _reconstruct_coregional_diag_block(
                sc_loc, coreg_w, n_models, ns, start_idx)
            for m_i in range(n_models):
                m_off = per_model_offsets[m_i] * ns
                block0_diag = block0_diag.at[
                    m_off + per_model_ata_diag_rows[m_i][0],
                    m_off + per_model_ata_diag_cols[m_i][0]
                ].add(likelihood_precs[m_i] * per_model_ata_diag_vals[m_i][0])
            block0_diag = block0_diag + _chol_eps_reg * _chol_eye_bs
            block0_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
            for m_i in range(n_models):
                m_off = per_model_offsets[m_i] * ns
                block0_arrow = block0_arrow.at[
                    per_model_ata_arrow_rows[m_i][0],
                    m_off + per_model_ata_arrow_cols[m_i][0]
                ].add(likelihood_precs[m_i] * per_model_ata_arrow_vals[m_i][0])
            buffer_init = _reconstruct_coregional_lower_block(
                sc_loc, coreg_w, n_models, ns, start_idx - 1)
            for m_i in range(n_models):
                m_off = per_model_offsets[m_i] * ns
                buffer_init = buffer_init.at[
                    m_off + _padded_lower_rows[m_i][0],
                    m_off + _padded_lower_cols[m_i][0]
                ].add(likelihood_precs[m_i] * _padded_lower_vals[m_i][0])
            buffer_cur = buffer_init.T

            n_interior = n_local - 2
            cs_all_h = np.zeros((n_interior, block_size, block_size), dtype=np_dtype)
            as_all_h = np.zeros((n_interior, n_fe, block_size), dtype=np_dtype)
            y_all_h = np.zeros((n_interior, block_size), dtype=np_dtype)
            buf_all_h = np.zeros((n_interior, block_size, block_size), dtype=np_dtype)

            carry = (cond_schur, arrow_tip_acc, arrow_schur,
                     logdet_cond, prev_lower_y, arrow_rhs_acc,
                     buffer_cur,
                     block0_diag, block0_arrow, rhs_st_local[0])
            for chunk_start in range(0, n_interior, chunk):
                chunk_end = min(chunk_start + chunk, n_interior)
                indices = jnp.arange(chunk_start, chunk_end)
                carry, (cs_chunk, as_chunk, y_chunk, buf_chunk) = \
                    _chol_fori_nonroot_interior(
                        *carry, rhs_st_local, *sc_args, coreg_w, likelihood_precs,
                        indices)
                cs_all_h[chunk_start:chunk_end] = np.asarray(cs_chunk)
                as_all_h[chunk_start:chunk_end] = np.asarray(as_chunk)
                y_all_h[chunk_start:chunk_end] = np.asarray(y_chunk)
                buf_all_h[chunk_start:chunk_end] = np.asarray(buf_chunk)
                del cs_chunk, as_chunk, y_chunk, buf_chunk

            (cond_schur, arrow_tip_acc, arrow_schur,
             logdet_cond, prev_lower_y, arrow_rhs_acc,
             buffer_cur,
             block0_diag_acc, block0_arrow_acc, block0_rhs_acc) = carry

            y_st_host = np.zeros((n_local, block_size), dtype=np_dtype)
            y_st_host[1:1 + n_interior] = y_all_h

            # Boundary: last block
            global_last = start_idx + n_local - 1
            last_diag = _reconstruct_coregional_diag_block(
                sc_loc, coreg_w, n_models, ns, global_last)
            for m_i in range(n_models):
                m_off = per_model_offsets[m_i] * ns
                last_diag = last_diag.at[
                    m_off + per_model_ata_diag_rows[m_i][n_local - 1],
                    m_off + per_model_ata_diag_cols[m_i][n_local - 1]
                ].add(likelihood_precs[m_i] * per_model_ata_diag_vals[m_i][n_local - 1])
            last_diag = last_diag + _chol_eps_reg * _chol_eye_bs - cond_schur
            _zeros_bs = jnp.zeros((block_size, block_size), dtype=dtype)
            last_lower = _zeros_bs
            if rank < comm_size - 1:
                last_lower = _reconstruct_coregional_lower_block(
                    sc_loc, coreg_w, n_models, ns, global_last)
                for m_i in range(n_models):
                    m_off = per_model_offsets[m_i] * ns
                    last_lower = last_lower.at[
                        m_off + _padded_lower_rows[m_i][n_local - 1],
                        m_off + _padded_lower_cols[m_i][n_local - 1]
                    ].add(likelihood_precs[m_i] * _padded_lower_vals[m_i][n_local - 1])
            last_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
            for m_i in range(n_models):
                m_off = per_model_offsets[m_i] * ns
                last_arrow = last_arrow.at[
                    per_model_ata_arrow_rows[m_i][n_local - 1],
                    m_off + per_model_ata_arrow_cols[m_i][n_local - 1]
                ].add(likelihood_precs[m_i] * per_model_ata_arrow_vals[m_i][n_local - 1])
            last_arrow = last_arrow - arrow_schur

            last_rhs = rhs_st_local[n_local - 1] - prev_lower_y

            _zeros_nfe_bs = jnp.zeros((n_fe, block_size), dtype=dtype)
            _zeros_bs_vec = jnp.zeros(block_size, dtype=dtype)
            my_diag = jnp.stack([block0_diag_acc, last_diag])
            my_lower = jnp.stack([buffer_cur.T, last_lower])
            my_arrow = jnp.stack([block0_arrow_acc, last_arrow])
            my_tip = arrow_tip_acc
            my_rhs = jnp.stack([block0_rhs_acc, last_rhs])

        jax.block_until_ready(my_diag)
        _chol_detail_timing['local_blocks'] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()
        _chol_detail_timing['_rs_t0'] = _t0

        # Phase 2: Reduced system (split into allgather + scan for diagnostics)
        (rs_diag_g, rs_lower_g, rs_arrow_g, rs_tip_final,
         rs_stored_cs, rs_stored_as,
         rs_stored_L, rs_stored_Ll, rs_stored_La,
         rs_logdet, L_tip, rs_rhs_g,
         rs_y, rs_arrow_rhs) = _rs_aggregate_and_factorize(
            my_diag, my_lower, my_arrow, my_tip,
            my_rhs, arrow_tip_initial)

        jax.block_until_ready(rs_logdet)
        _chol_detail_timing['rs_factorize'] = _time.perf_counter() - _chol_detail_timing['_rs_t0']
        _chol_detail_timing['rs_aggregate'] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()

        local_logdet_j = logdet_cond
        total_interior_logdet = mpi4jax.allreduce(local_logdet_j, op=MPI.SUM, comm=comm)
        logdet_cond_final = total_interior_logdet + rs_logdet

        # Fill y values for boundary blocks from reduced system
        if rank == 0:
            y_st_host[n_local - 1] = np.asarray(rs_y[1])
        else:
            y_st_host[0] = np.asarray(rs_y[2 * rank])
            y_st_host[n_local - 1] = np.asarray(rs_y[2 * rank + 1])
        y_st_local = jnp.array(y_st_host)

        # Arrow RHS
        interior_arrow = mpi4jax.allreduce(arrow_rhs_acc, op=MPI.SUM, comm=comm)
        arrow_rhs_global = rhs_fe + interior_arrow + rs_arrow_rhs

        _chol_detail_timing['boundary_update'] = _time.perf_counter() - _t0

        return (cs_all_h, as_all_h, buf_all_h,
                y_st_local, L_tip, arrow_rhs_global, logdet_cond_final,
                rs_diag_g, rs_lower_g, rs_arrow_g,
                rs_stored_cs, rs_stored_as,
                rs_stored_L, rs_stored_Ll, rs_stored_La,
                rs_y)


    @jax.jit
    def _bwd_one_block(L_i, L_lower_i, L_arrow_i, y_i, x_next, x_fe, local_i):
        global_i = start_idx + local_i
        rhs = y_i - L_arrow_i.T @ x_fe
        rhs = jnp.where(global_i < nt - 1, rhs - L_lower_i.T @ x_next, rhs)
        x_i = jax.scipy.linalg.solve_triangular(L_i.T, rhs, lower=False)
        return x_i

    @jax.jit
    def _bwd_one_block_permuted(L_i, L_lower_i, L_arrow_i, stored_buf_i,
                                y_i, x_next, x_fe, x_block0, local_i):
        """Backward sub for non-root interior blocks with buffer term."""
        global_i = start_idx + local_i
        rhs = y_i - L_arrow_i.T @ x_fe - stored_buf_i.T @ x_block0
        rhs = jnp.where(global_i < nt - 1, rhs - L_lower_i.T @ x_next, rhs)
        x_i = jax.scipy.linalg.solve_triangular(L_i.T, rhs, lower=False)
        return x_i

    @jax.jit
    def _bwd_allgather_reduce(local_x_st, local_quad):
        # Place local x at correct global indices, then allreduce
        x_global = jnp.zeros((nt, block_size), dtype=dtype)
        x_global = x_global.at[start_idx:start_idx + n_local].set(
            local_x_st[:n_local])
        x_st_global = mpi4jax.allreduce(x_global, op=MPI.SUM, comm=comm)
        quad_global = mpi4jax.allreduce(local_quad, op=MPI.SUM, comm=comm)
        return x_st_global, quad_global

    @jax.jit
    def _rs_bwd_one_block_fn(L_j, L_lower_j, L_arrow_j, y_j,
                              x_next, x_fe, is_last):
        """One block of reduced system backward sub."""
        rhs = y_j - L_arrow_j.T @ x_fe
        rhs = jnp.where(is_last, rhs, rhs - L_lower_j.T @ x_next)
        return jax.scipy.linalg.solve_triangular(L_j.T, rhs, lower=False)

    @jax.jit
    def _rs_backward_sub(rs_stored_L, rs_stored_Ll, rs_stored_La,
                         rs_y, L_tip, y_fe, x_fe):
        """Backward sub on the reduced BTA system — single JIT, all on GPU."""
        rs_x = jnp.zeros((n_rs, block_size), dtype=dtype)

        idx_last = n_rs - 1
        rhs_last = rs_y[idx_last] - rs_stored_La[idx_last].T @ x_fe
        x_last = jax.scipy.linalg.solve_triangular(
            rs_stored_L[idx_last].T, rhs_last, lower=False)
        rs_x = rs_x.at[idx_last].set(x_last)

        def bwd_body(j, state):
            rs_x_a, x_next = state
            idx = n_rs - 2 - j
            rhs = rs_y[idx] - rs_stored_La[idx].T @ x_fe - rs_stored_Ll[idx].T @ x_next
            x_j = jax.scipy.linalg.solve_triangular(
                rs_stored_L[idx].T, rhs, lower=False)
            rs_x_a = rs_x_a.at[idx].set(x_j)
            return rs_x_a, x_j

        rs_x, _ = lax.fori_loop(0, n_rs - 2, bwd_body, (rs_x, x_last))
        return rs_x

    @jax.jit
    def _rs_si(rs_stored_L, rs_stored_Ll, rs_stored_La, S_tip):
        """Selected inversion on the reduced BTA system using stored L factors."""
        rs_sd = jnp.zeros((n_rs, block_size, block_size), dtype=dtype)
        rs_sl = jnp.zeros((n_rs, block_size, block_size), dtype=dtype)
        rs_sa = jnp.zeros((n_rs, n_fe, block_size), dtype=dtype)

        idx_last = n_rs - 1
        L_last = rs_stored_L[idx_last]
        L_inv_last = jax.scipy.linalg.solve_triangular(
            L_last, _chol_eye_bs, lower=True)
        L_arrow_last = rs_stored_La[idx_last]

        sa_last = -S_tip @ L_arrow_last @ L_inv_last
        sd_last = (L_inv_last.T - sa_last.T @ L_arrow_last) @ L_inv_last
        rs_sd = rs_sd.at[idx_last].set(sd_last)
        rs_sa = rs_sa.at[idx_last].set(sa_last)

        def rs_si_body(j, state):
            rs_sd_a, rs_sl_a, rs_sa_a, sd_p, sa_p = state
            idx = n_rs - 2 - j

            L_j = rs_stored_L[idx]
            L_inv_j = jax.scipy.linalg.solve_triangular(
                L_j, _chol_eye_bs, lower=True)
            L_lower_j = rs_stored_Ll[idx]
            L_arrow_j = rs_stored_La[idx]

            sl_j = (-sd_p @ L_lower_j - sa_p.T @ L_arrow_j) @ L_inv_j
            sa_j = (-sa_p @ L_lower_j - S_tip @ L_arrow_j) @ L_inv_j
            sd_j = (L_inv_j.T - sl_j.T @ L_lower_j - sa_j.T @ L_arrow_j) @ L_inv_j

            rs_sd_a = rs_sd_a.at[idx].set(sd_j)
            rs_sl_a = rs_sl_a.at[idx].set(sl_j)
            rs_sa_a = rs_sa_a.at[idx].set(sa_j)
            return rs_sd_a, rs_sl_a, rs_sa_a, sd_j, sa_j

        rs_sd, rs_sl, rs_sa, _, _ = lax.fori_loop(
            0, n_rs - 2, rs_si_body,
            (rs_sd, rs_sl, rs_sa, sd_last, sa_last))

        return rs_sd, rs_sl, rs_sa

    def _bwd_fn(theta, stored_L_host, stored_Ll_host, stored_La_host,
                stored_buf_host,
                L_tip, y_st_local, arrow_rhs_total,
                rs_stored_L, rs_stored_Ll, rs_stored_La,
                rs_y):
        y_fe = jax.scipy.linalg.solve_triangular(L_tip, arrow_rhs_total, lower=True)
        local_quad = jnp.sum(y_st_local ** 2)
        y_fe_quad = jnp.where(rank == comm_size - 1, jnp.sum(y_fe ** 2), 0.0)
        local_quad = local_quad + y_fe_quad
        x_fe = jax.scipy.linalg.solve_triangular(L_tip.T, y_fe, lower=False)

        rs_x = _rs_backward_sub(
            rs_stored_L, rs_stored_Ll, rs_stored_La,
            rs_y, L_tip, y_fe, x_fe)

        max_n_local = _max_n_local
        local_x_host = np.zeros((max_n_local, block_size), dtype=np_dtype)

        if rank == 0:
            x_last = jnp.array(rs_x[1])
            local_x_host[n_local - 1] = np.asarray(x_last)
            x_next = x_last
            for local_i_py in range(n_local - 2, -1, -1):
                Li = jnp.array(stored_L_host[local_i_py])
                Lli = jnp.array(stored_Ll_host[local_i_py])
                Lai = jnp.array(stored_La_host[local_i_py])
                local_i_jax = jnp.array(local_i_py, dtype=jnp.int32)
                x_i = _bwd_one_block(Li, Lli, Lai, y_st_local[local_i_py],
                                     x_next, x_fe, local_i_jax)
                local_x_host[local_i_py] = np.asarray(x_i)
                x_next = x_i
                del Li, Lli, Lai
        else:
            x_block0 = jnp.array(rs_x[2 * rank])
            x_last = jnp.array(rs_x[2 * rank + 1])
            local_x_host[0] = np.asarray(x_block0)
            local_x_host[n_local - 1] = np.asarray(x_last)
            x_next = x_last
            for local_i_py in range(n_local - 2, 0, -1):
                Li = jnp.array(stored_L_host[local_i_py])
                Lli = jnp.array(stored_Ll_host[local_i_py])
                Lai = jnp.array(stored_La_host[local_i_py])
                buf_i = jnp.array(stored_buf_host[local_i_py])
                local_i_jax = jnp.array(local_i_py, dtype=jnp.int32)
                x_i = _bwd_one_block_permuted(
                    Li, Lli, Lai, buf_i,
                    y_st_local[local_i_py], x_next, x_fe, x_block0,
                    local_i_jax)
                local_x_host[local_i_py] = np.asarray(x_i)
                x_next = x_i
                del Li, Lli, Lai, buf_i

        local_x_st = jnp.array(local_x_host)
        x_st_global, quad_global = _bwd_allgather_reduce(local_x_st, local_quad)
        x = jnp.concatenate([x_st_global.reshape(-1), x_fe])
        return quad_global, x

    # ---- Stage 1c: logdet Q_prior ----

    @jax.jit
    def _logdet_prior_fn(theta):
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        sc_padded = _pad_subdiags(sc_list)
        return twophase_logdet_Q_prior_coregional_scan(
            sc_padded, coreg_w, n_models, ns, nt, dtype,
            rank, comm_size, n_local, start_idx, comm)

    # ---- Stage 2: SI gradients ----

    n_coreg = n_sigmas + n_lambdas
    n_models_cubed = n_models * n_models * n_models

    @jax.jit
    def _si_allreduce(g_st, g_lik, g_c):
        g_st = mpi4jax.allreduce(g_st, op=MPI.SUM, comm=comm)
        g_lik = mpi4jax.allreduce(g_lik, op=MPI.SUM, comm=comm)
        g_c = mpi4jax.allreduce(g_c, op=MPI.SUM, comm=comm)
        return g_st, g_lik, g_c

    @jax.jit
    def _trace_at_block(sd_i, sl_i, sa_i, S_tip_arg, local_i,
                        sc_args_tuple, coreg_w_arg, likelihood_precs_arg,
                        jac_q1s_a, jac_q2s_a, jac_q3s_a,
                        jac_scale_a, jac_exp_gt_a, jac_coreg_w_arg,
                        is_global_last, include_tip):
        """Compute gradient traces from pre-computed SI values at a boundary block."""
        global_i = start_idx + local_i
        (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
         m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a) = sc_args_tuple

        def _diag_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models
            sd_ij = lax.dynamic_slice(sd_i, (ii * ns, jj * ns), (ns, ns))
            w_ijm = coreg_w_arg[ii, jj, m_idx]
            base_d = (m0d_a[m_idx, global_i] * q3s_a[m_idx]
                      + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * q2s_a[m_idx]
                      + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * q1s_a[m_idx])
            tr_val = jnp.sum(sd_ij * (scale_a[m_idx] * base_d).T)
            dw = jac_coreg_w_arg[ii, jj, m_idx, :]
            g_c_ = g_c_ + dw * tr_val
            partial_gt = (m1d_a[m_idx, global_i] * q2s_a[m_idx]
                          + 2.0 * exp_gt_a[m_idx] * m2d_a[m_idx, global_i] * q1s_a[m_idx])
            for k in range(3):
                dQu = (jac_scale_a[m_idx, k] * base_d
                       + scale_a[m_idx] * (m0d_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                           + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                           + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                       + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                g_st_ = g_st_.at[m_idx, k].add(w_ijm * jnp.sum(sd_ij * dQu.T))
            return (g_st_, g_c_)

        g_st_init = jnp.zeros((n_models, 3), dtype=dtype)
        g_c_init = jnp.zeros(n_coreg, dtype=dtype)
        g_st, g_c = lax.fori_loop(0, n_models_cubed, _diag_body, (g_st_init, g_c_init))

        safe_idx = jnp.minimum(global_i, nt - 2)
        sl_for_trace = jnp.where(is_global_last, jnp.zeros_like(sl_i), sl_i)

        def _lower_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models
            sl_ij = lax.dynamic_slice(sl_for_trace, (ii * ns, jj * ns), (ns, ns))
            w_ijm = coreg_w_arg[ii, jj, m_idx]
            base_l = (m0s_a[m_idx, safe_idx] * q3s_a[m_idx]
                      + exp_gt_a[m_idx] * m1s_a[m_idx, safe_idx] * q2s_a[m_idx]
                      + exp_gt_a[m_idx]**2 * m2s_a[m_idx, safe_idx] * q1s_a[m_idx])
            tr_l = jnp.sum(sl_ij * (scale_a[m_idx] * base_l).T)
            dw = jac_coreg_w_arg[ii, jj, m_idx, :]
            g_c_ = g_c_ + 2.0 * dw * tr_l
            partial_gt = (m1s_a[m_idx, safe_idx] * q2s_a[m_idx]
                          + 2.0 * exp_gt_a[m_idx] * m2s_a[m_idx, safe_idx] * q1s_a[m_idx])
            for k in range(3):
                dQu_l = (jac_scale_a[m_idx, k] * base_l
                         + scale_a[m_idx] * (m0s_a[m_idx, safe_idx] * jac_q3s_a[m_idx, :, :, k]
                                             + exp_gt_a[m_idx] * m1s_a[m_idx, safe_idx] * jac_q2s_a[m_idx, :, :, k]
                                             + exp_gt_a[m_idx]**2 * m2s_a[m_idx, safe_idx] * jac_q1s_a[m_idx, :, :, k])
                         + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                g_st_ = g_st_.at[m_idx, k].add(2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))
            return (g_st_, g_c_)

        g_st_l, g_c_l = lax.fori_loop(0, n_models_cubed, _lower_body, (g_st_init, g_c_init))
        g_st = g_st + jnp.where(is_global_last, jnp.zeros_like(g_st_l), g_st_l)
        g_c = g_c + jnp.where(is_global_last, jnp.zeros_like(g_c_l), g_c_l)

        g_lik = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            sp_d = jnp.sum(sd_i[m_off + per_model_ata_diag_rows[m][local_i],
                                m_off + per_model_ata_diag_cols[m][local_i]]
                           * per_model_ata_diag_vals[m][local_i])
            sp_a = jnp.sum(sa_i[per_model_ata_arrow_rows[m][local_i],
                                m_off + per_model_ata_arrow_cols[m][local_i]]
                           * per_model_ata_arrow_vals[m][local_i])
            sl_lik = jnp.where(
                is_global_last, 0.0,
                jnp.sum(sl_for_trace[m_off + _padded_lower_rows[m][local_i],
                                     m_off + _padded_lower_cols[m][local_i]]
                        * _padded_lower_vals[m][local_i]))
            g_lik = g_lik.at[m].add(sp_d + 2.0 * sp_a + 2.0 * sl_lik)
            tip_val = jnp.where(
                include_tip & (rank == comm_size - 1),
                jnp.sum(S_tip_arg * per_model_ata_tip[m]), 0.0)
            g_lik = g_lik.at[m].add(tip_val)

        return g_st, g_lik, g_c

    # ---- Merged bwd+SI per-block JIT functions ----

    @jax.jit
    def _bwd_si_block(L_i, L_lower_i, L_arrow_i, y_i, x_next, x_fe,
                      sd_prev, sa_prev, S_tip_arg, local_i,
                      sc_args_tuple, coreg_w_arg, likelihood_precs_arg,
                      jac_q1s_a, jac_q2s_a, jac_q3s_a,
                      jac_scale_a, jac_exp_gt_a, jac_coreg_w_arg):
        """Merged backward sub + SI gradient traces (L factors from CPU)."""
        L_blk_inv = jax.scipy.linalg.solve_triangular(L_i, _chol_eye_bs, lower=True)
        global_i = start_idx + local_i

        # Backward sub
        rhs = y_i - L_arrow_i.T @ x_fe
        rhs = jnp.where(global_i < nt - 1, rhs - L_lower_i.T @ x_next, rhs)
        x_i = jax.scipy.linalg.solve_triangular(L_i.T, rhs, lower=False)

        # SI recurrence
        sl_i = (-sd_prev @ L_lower_i - sa_prev.T @ L_arrow_i) @ L_blk_inv
        sa_i = (-sa_prev @ L_lower_i - S_tip_arg @ L_arrow_i) @ L_blk_inv
        sd_i = (L_blk_inv.T - sl_i.T @ L_lower_i - sa_i.T @ L_arrow_i) @ L_blk_inv

        # Gradient traces
        (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
         m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a) = sc_args_tuple

        def _diag_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models
            sd_ij = lax.dynamic_slice(sd_i, (ii * ns, jj * ns), (ns, ns))
            w_ijm = coreg_w_arg[ii, jj, m_idx]
            base_d = (m0d_a[m_idx, global_i] * q3s_a[m_idx]
                      + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * q2s_a[m_idx]
                      + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * q1s_a[m_idx])
            tr_val = jnp.sum(sd_ij * (scale_a[m_idx] * base_d).T)
            dw = jac_coreg_w_arg[ii, jj, m_idx, :]
            g_c_ = g_c_ + dw * tr_val
            partial_gt = (m1d_a[m_idx, global_i] * q2s_a[m_idx]
                          + 2.0 * exp_gt_a[m_idx] * m2d_a[m_idx, global_i] * q1s_a[m_idx])
            for k in range(3):
                dQu = (jac_scale_a[m_idx, k] * base_d
                       + scale_a[m_idx] * (m0d_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                           + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                           + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                       + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                g_st_ = g_st_.at[m_idx, k].add(w_ijm * jnp.sum(sd_ij * dQu.T))
            return (g_st_, g_c_)

        g_st_init = jnp.zeros((n_models, 3), dtype=dtype)
        g_c_init = jnp.zeros(n_coreg, dtype=dtype)
        g_st, g_c = lax.fori_loop(0, n_models_cubed, _diag_body, (g_st_init, g_c_init))

        def _lower_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models
            sl_ij = lax.dynamic_slice(sl_i, (ii * ns, jj * ns), (ns, ns))
            w_ijm = coreg_w_arg[ii, jj, m_idx]
            base_l = (m0s_a[m_idx, global_i] * q3s_a[m_idx]
                      + exp_gt_a[m_idx] * m1s_a[m_idx, global_i] * q2s_a[m_idx]
                      + exp_gt_a[m_idx]**2 * m2s_a[m_idx, global_i] * q1s_a[m_idx])
            tr_l = jnp.sum(sl_ij * (scale_a[m_idx] * base_l).T)
            dw = jac_coreg_w_arg[ii, jj, m_idx, :]
            g_c_ = g_c_ + 2.0 * dw * tr_l
            partial_gt = (m1s_a[m_idx, global_i] * q2s_a[m_idx]
                          + 2.0 * exp_gt_a[m_idx] * m2s_a[m_idx, global_i] * q1s_a[m_idx])
            for k in range(3):
                dQu_l = (jac_scale_a[m_idx, k] * base_l
                         + scale_a[m_idx] * (m0s_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                             + exp_gt_a[m_idx] * m1s_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                             + exp_gt_a[m_idx]**2 * m2s_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                         + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                g_st_ = g_st_.at[m_idx, k].add(2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))
            return (g_st_, g_c_)

        g_st_l, g_c_l = lax.fori_loop(0, n_models_cubed, _lower_body, (g_st_init, g_c_init))
        g_st = g_st + g_st_l
        g_c = g_c + g_c_l

        g_lik = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            sp_d = jnp.sum(sd_i[m_off + per_model_ata_diag_rows[m][local_i],
                                m_off + per_model_ata_diag_cols[m][local_i]]
                           * per_model_ata_diag_vals[m][local_i])
            sp_a = jnp.sum(sa_i[per_model_ata_arrow_rows[m][local_i],
                                m_off + per_model_ata_arrow_cols[m][local_i]]
                           * per_model_ata_arrow_vals[m][local_i])
            sp_l = jnp.sum(sl_i[m_off + _padded_lower_rows[m][local_i],
                                m_off + _padded_lower_cols[m][local_i]]
                           * _padded_lower_vals[m][local_i])
            g_lik = g_lik.at[m].add(sp_d + 2.0 * sp_a + 2.0 * sp_l)

        return x_i, sd_i, sa_i, g_st, g_lik, g_c

    @jax.jit
    def _bwd_si_block_permuted(
            L_i, L_lower_i, L_arrow_i, stored_buf_i,
            y_i, x_next, x_fe, x_block0,
            sd_prev, sa_prev, buf_X_next,
            sd_block0, sa_block0, S_tip_arg, local_i,
            sc_args_tuple, coreg_w_arg, likelihood_precs_arg,
            jac_q1s_a, jac_q2s_a, jac_q3s_a,
            jac_scale_a, jac_exp_gt_a, jac_coreg_w_arg):
        """Merged backward sub + SI gradient traces for non-root interior blocks."""
        L_blk_inv = jax.scipy.linalg.solve_triangular(L_i, _chol_eye_bs, lower=True)
        global_i = start_idx + local_i

        # Backward sub (permuted)
        rhs = y_i - L_arrow_i.T @ x_fe - stored_buf_i.T @ x_block0
        rhs = jnp.where(global_i < nt - 1, rhs - L_lower_i.T @ x_next, rhs)
        x_i = jax.scipy.linalg.solve_triangular(L_i.T, rhs, lower=False)

        # SI recurrence (permuted)
        buf_L_i = stored_buf_i
        sl_i = (-buf_X_next.T @ buf_L_i
                - sd_prev @ L_lower_i
                - sa_prev.T @ L_arrow_i) @ L_blk_inv
        buf_X_i = (-buf_X_next @ L_lower_i
                   - sd_block0 @ buf_L_i
                   - sa_block0.T @ L_arrow_i) @ L_blk_inv
        sa_i = (-sa_prev @ L_lower_i
                - sa_block0 @ buf_L_i
                - S_tip_arg @ L_arrow_i) @ L_blk_inv
        sd_i = (L_blk_inv.T
                - sl_i.T @ L_lower_i
                - buf_X_i.T @ buf_L_i
                - sa_i.T @ L_arrow_i) @ L_blk_inv

        # Gradient traces
        (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
         m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a) = sc_args_tuple

        def _diag_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models
            sd_ij = lax.dynamic_slice(sd_i, (ii * ns, jj * ns), (ns, ns))
            w_ijm = coreg_w_arg[ii, jj, m_idx]
            base_d = (m0d_a[m_idx, global_i] * q3s_a[m_idx]
                      + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * q2s_a[m_idx]
                      + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * q1s_a[m_idx])
            tr_val = jnp.sum(sd_ij * (scale_a[m_idx] * base_d).T)
            dw = jac_coreg_w_arg[ii, jj, m_idx, :]
            g_c_ = g_c_ + dw * tr_val
            partial_gt = (m1d_a[m_idx, global_i] * q2s_a[m_idx]
                          + 2.0 * exp_gt_a[m_idx] * m2d_a[m_idx, global_i] * q1s_a[m_idx])
            for k in range(3):
                dQu = (jac_scale_a[m_idx, k] * base_d
                       + scale_a[m_idx] * (m0d_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                           + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                           + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                       + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                g_st_ = g_st_.at[m_idx, k].add(w_ijm * jnp.sum(sd_ij * dQu.T))
            return (g_st_, g_c_)

        g_st_init = jnp.zeros((n_models, 3), dtype=dtype)
        g_c_init = jnp.zeros(n_coreg, dtype=dtype)
        g_st, g_c = lax.fori_loop(0, n_models_cubed, _diag_body, (g_st_init, g_c_init))

        def _lower_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models
            sl_ij = lax.dynamic_slice(sl_i, (ii * ns, jj * ns), (ns, ns))
            w_ijm = coreg_w_arg[ii, jj, m_idx]
            base_l = (m0s_a[m_idx, global_i] * q3s_a[m_idx]
                      + exp_gt_a[m_idx] * m1s_a[m_idx, global_i] * q2s_a[m_idx]
                      + exp_gt_a[m_idx]**2 * m2s_a[m_idx, global_i] * q1s_a[m_idx])
            tr_l = jnp.sum(sl_ij * (scale_a[m_idx] * base_l).T)
            dw = jac_coreg_w_arg[ii, jj, m_idx, :]
            g_c_ = g_c_ + 2.0 * dw * tr_l
            partial_gt = (m1s_a[m_idx, global_i] * q2s_a[m_idx]
                          + 2.0 * exp_gt_a[m_idx] * m2s_a[m_idx, global_i] * q1s_a[m_idx])
            for k in range(3):
                dQu_l = (jac_scale_a[m_idx, k] * base_l
                         + scale_a[m_idx] * (m0s_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                             + exp_gt_a[m_idx] * m1s_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                             + exp_gt_a[m_idx]**2 * m2s_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                         + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                g_st_ = g_st_.at[m_idx, k].add(2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))
            return (g_st_, g_c_)

        g_st_l, g_c_l = lax.fori_loop(0, n_models_cubed, _lower_body, (g_st_init, g_c_init))
        g_st = g_st + g_st_l
        g_c = g_c + g_c_l

        g_lik = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            sp_d = jnp.sum(sd_i[m_off + per_model_ata_diag_rows[m][local_i],
                                m_off + per_model_ata_diag_cols[m][local_i]]
                           * per_model_ata_diag_vals[m][local_i])
            sp_a = jnp.sum(sa_i[per_model_ata_arrow_rows[m][local_i],
                                m_off + per_model_ata_arrow_cols[m][local_i]]
                           * per_model_ata_arrow_vals[m][local_i])
            sp_l = jnp.sum(sl_i[m_off + _padded_lower_rows[m][local_i],
                                m_off + _padded_lower_cols[m][local_i]]
                           * _padded_lower_vals[m][local_i])
            g_lik = g_lik.at[m].add(sp_d + 2.0 * sp_a + 2.0 * sp_l)

        return x_i, sd_i, sa_i, buf_X_i, g_st, g_lik, g_c

    @jax.jit
    def _rs_bwd_si_last_block_fn(L_last, L_arrow_last, y_last, x_fe, S_tip):
        """Last block of RS backward sub + SI."""
        L_inv = jax.scipy.linalg.solve_triangular(L_last, _chol_eye_bs, lower=True)
        rhs = y_last - L_arrow_last.T @ x_fe
        x_last = jax.scipy.linalg.solve_triangular(L_last.T, rhs, lower=False)
        sa = -S_tip @ L_arrow_last @ L_inv
        sd = (L_inv.T - sa.T @ L_arrow_last) @ L_inv
        return x_last, sd, sa

    @jax.jit
    def _rs_bwd_si_interior_fn(L_j, L_lower_j, L_arrow_j, y_j,
                                x_next, x_fe, sd_prev, sa_prev, S_tip):
        """Interior block of RS backward sub + SI."""
        L_inv = jax.scipy.linalg.solve_triangular(L_j, _chol_eye_bs, lower=True)
        rhs = y_j - L_arrow_j.T @ x_fe - L_lower_j.T @ x_next
        x_j = jax.scipy.linalg.solve_triangular(L_j.T, rhs, lower=False)
        sl = (-sd_prev @ L_lower_j - sa_prev.T @ L_arrow_j) @ L_inv
        sa = (-sa_prev @ L_lower_j - S_tip @ L_arrow_j) @ L_inv
        sd = (L_inv.T - sl.T @ L_lower_j - sa.T @ L_arrow_j) @ L_inv
        return x_j, sd, sl, sa

    if _use_rs_scan:
        @jax.jit
        def _rs_bwd_and_si(rs_stored_L, rs_stored_Ll, rs_stored_La,
                           rs_y, L_tip, y_fe, x_fe, S_tip):
            """Merged RS backward sub + SI — single JIT, all on GPU."""
            idx_last = n_rs - 1
            L_last = rs_stored_L[idx_last]
            L_inv_last = jax.scipy.linalg.solve_triangular(
                L_last, _chol_eye_bs, lower=True)
            L_arrow_last = rs_stored_La[idx_last]

            rhs_last = rs_y[idx_last] - L_arrow_last.T @ x_fe
            x_last = jax.scipy.linalg.solve_triangular(L_last.T, rhs_last, lower=False)

            sa_last = -S_tip @ L_arrow_last @ L_inv_last
            sd_last = (L_inv_last.T - sa_last.T @ L_arrow_last) @ L_inv_last

            rs_x = jnp.zeros((n_rs, block_size), dtype=dtype)
            rs_sd = jnp.zeros((n_rs, block_size, block_size), dtype=dtype)
            rs_sl = jnp.zeros((n_rs, block_size, block_size), dtype=dtype)
            rs_sa = jnp.zeros((n_rs, n_fe, block_size), dtype=dtype)

            rs_x = rs_x.at[idx_last].set(x_last)
            rs_sd = rs_sd.at[idx_last].set(sd_last)
            rs_sa = rs_sa.at[idx_last].set(sa_last)

            def bwd_si_body(j, state):
                rs_x_a, rs_sd_a, rs_sl_a, rs_sa_a, x_next, sd_prev, sa_prev = state
                idx = n_rs - 2 - j
                L_j = rs_stored_L[idx]
                L_inv_j = jax.scipy.linalg.solve_triangular(
                    L_j, _chol_eye_bs, lower=True)
                L_lower_j = rs_stored_Ll[idx]
                L_arrow_j = rs_stored_La[idx]
                rhs_j = rs_y[idx] - L_arrow_j.T @ x_fe - L_lower_j.T @ x_next
                x_j = jax.scipy.linalg.solve_triangular(L_j.T, rhs_j, lower=False)
                sl_j = (-sd_prev @ L_lower_j - sa_prev.T @ L_arrow_j) @ L_inv_j
                sa_j = (-sa_prev @ L_lower_j - S_tip @ L_arrow_j) @ L_inv_j
                sd_j = (L_inv_j.T - sl_j.T @ L_lower_j - sa_j.T @ L_arrow_j) @ L_inv_j
                rs_x_a = rs_x_a.at[idx].set(x_j)
                rs_sd_a = rs_sd_a.at[idx].set(sd_j)
                rs_sl_a = rs_sl_a.at[idx].set(sl_j)
                rs_sa_a = rs_sa_a.at[idx].set(sa_j)
                return rs_x_a, rs_sd_a, rs_sl_a, rs_sa_a, x_j, sd_j, sa_j

            rs_x, rs_sd, rs_sl, rs_sa, _, _, _ = lax.fori_loop(
                0, n_rs - 2, bwd_si_body,
                (rs_x, rs_sd, rs_sl, rs_sa, x_last, sd_last, sa_last))

            return rs_x, rs_sd, rs_sl, rs_sa
    else:
        def _rs_bwd_and_si(rs_stored_L, rs_stored_Ll, rs_stored_La,
                           rs_y, L_tip, y_fe, x_fe, S_tip):
            """Per-block JIT fallback for large blocks."""
            rs_x_h = np.zeros((n_rs, block_size), dtype=np_dtype)
            rs_sd_h = np.zeros((n_rs, block_size, block_size), dtype=np_dtype)
            rs_sl_h = np.zeros((n_rs, block_size, block_size), dtype=np_dtype)
            rs_sa_h = np.zeros((n_rs, n_fe, block_size), dtype=np_dtype)

            idx_last = n_rs - 1
            x_last, sd_last, sa_last = _rs_bwd_si_last_block_fn(
                rs_stored_L[idx_last], rs_stored_La[idx_last],
                rs_y[idx_last], x_fe, S_tip)
            rs_x_h[idx_last] = np.asarray(x_last)
            rs_sd_h[idx_last] = np.asarray(sd_last)
            rs_sa_h[idx_last] = np.asarray(sa_last)

            x_next = x_last
            sd_prev = sd_last
            sa_prev = sa_last
            for j_py in range(n_rs - 2, 0, -1):
                x_j, sd_j, sl_j, sa_j = _rs_bwd_si_interior_fn(
                    rs_stored_L[j_py], rs_stored_Ll[j_py],
                    rs_stored_La[j_py], rs_y[j_py],
                    x_next, x_fe, sd_prev, sa_prev, S_tip)
                rs_x_h[j_py] = np.asarray(x_j)
                rs_sd_h[j_py] = np.asarray(sd_j)
                rs_sl_h[j_py] = np.asarray(sl_j)
                rs_sa_h[j_py] = np.asarray(sa_j)
                x_next = x_j
                sd_prev = sd_j
                sa_prev = sa_j

            return rs_x_h, rs_sd_h, rs_sl_h, rs_sa_h

    # ---- Merged bwd+SI orchestrator ----

    def _bwd_and_si_fn(theta, stored_L_host, stored_Ll_host, stored_La_host,
                       stored_buf_host,
                       L_tip, y_st_local, arrow_rhs_total,
                       rs_stored_L, rs_stored_Ll, rs_stored_La,
                       rs_y):
        """Merged backward sub + SI gradient computation in a single pass."""
        likelihood_precs = _sc_cache['likelihood_precs'] if 'likelihood_precs' in _sc_cache else _theta_to_likelihood_precs(theta)
        if 'sc_padded' in _sc_cache:
            sc_padded = _sc_cache['sc_padded']
            coreg_w = _sc_cache['coreg_w']
        else:
            sc_list, coreg_w = _theta_to_sc_coreg(theta)
            sc_padded = _pad_subdiags(sc_list)
        jac_sc_list = _sc_cache['jac_sc_list'] if 'jac_sc_list' in _sc_cache else _theta_to_jac_sc(theta)
        jac_coreg_w = _sc_cache['jac_coreg_w'] if 'jac_coreg_w' in _sc_cache else _theta_to_jac_coreg_w(theta)
        _sc_cache['jac_sc_list'] = jac_sc_list
        _sc_cache['jac_coreg_w'] = jac_coreg_w

        q1s_a = jnp.stack([sc_padded[m]['q1s'] for m in range(n_models)])
        q2s_a = jnp.stack([sc_padded[m]['q2s'] for m in range(n_models)])
        q3s_a = jnp.stack([sc_padded[m]['q3s'] for m in range(n_models)])
        scale_a = jnp.array([sc_padded[m]['scale'] for m in range(n_models)])
        exp_gt_a = jnp.array([sc_padded[m]['exp_gt'] for m in range(n_models)])
        m0d_a = jnp.stack([sc_padded[m]['m0_diag'] for m in range(n_models)])
        m1d_a = jnp.stack([sc_padded[m]['m1_diag'] for m in range(n_models)])
        m2d_a = jnp.stack([sc_padded[m]['m2_diag'] for m in range(n_models)])
        m0s_a = jnp.stack([sc_padded[m]['m0_subdiag'] for m in range(n_models)])
        m1s_a = jnp.stack([sc_padded[m]['m1_subdiag'] for m in range(n_models)])
        m2s_a = jnp.stack([sc_padded[m]['m2_subdiag'] for m in range(n_models)])
        sc_args_t = (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
                     m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a)

        jac_q1s_a = jnp.stack([jac_sc_list[m]['q1s'] for m in range(n_models)])
        jac_q2s_a = jnp.stack([jac_sc_list[m]['q2s'] for m in range(n_models)])
        jac_q3s_a = jnp.stack([jac_sc_list[m]['q3s'] for m in range(n_models)])
        jac_scale_a = jnp.stack([jac_sc_list[m]['scale'] for m in range(n_models)])
        jac_exp_gt_a = jnp.stack([jac_sc_list[m]['exp_gt'] for m in range(n_models)])

        # Forward solve for arrow block
        y_fe = jax.scipy.linalg.solve_triangular(L_tip, arrow_rhs_total, lower=True)
        local_quad = jnp.sum(y_st_local ** 2)
        y_fe_quad = jnp.where(rank == comm_size - 1, jnp.sum(y_fe ** 2), 0.0)
        local_quad = local_quad + y_fe_quad
        x_fe = jax.scipy.linalg.solve_triangular(L_tip.T, y_fe, lower=False)

        # Tip inverse for SI
        L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, _chol_eye_nfe, lower=True)
        S_tip = L_tip_inv.T @ L_tip_inv

        # Merged RS backward sub + SI
        rs_x, rs_sd, rs_sl, rs_sa = _rs_bwd_and_si(
            rs_stored_L, rs_stored_Ll, rs_stored_La,
            rs_y, L_tip, y_fe, x_fe, S_tip)

        max_n_local = _max_n_local
        local_x_host = np.zeros((max_n_local, block_size), dtype=np_dtype)

        g_st_acc = jnp.zeros((n_models, 3), dtype=dtype)
        g_lik_acc = jnp.zeros(n_models, dtype=dtype)
        g_c_acc = jnp.zeros(n_coreg, dtype=dtype)

        last_local = n_local - 1
        is_last_global = jnp.array(start_idx + last_local == nt - 1)

        if rank == 0:
            # Root: last block from RS
            x_last = jnp.array(rs_x[1])
            local_x_host[n_local - 1] = np.asarray(x_last)

            # SI traces for last block (from RS)
            sd_last = jnp.array(rs_sd[1])
            sa_last = jnp.array(rs_sa[1])
            sl_last = jnp.array(rs_sl[1])
            local_i_last = jnp.array(last_local, dtype=jnp.int32)
            g_st_b, g_lik_b, g_c_b = _trace_at_block(
                sd_last, sl_last, sa_last, S_tip, local_i_last,
                sc_args_t, coreg_w, likelihood_precs,
                jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                jac_coreg_w, is_last_global, jnp.array(True))
            g_st_acc = g_st_acc + g_st_b
            g_lik_acc = g_lik_acc + g_lik_b
            g_c_acc = g_c_acc + g_c_b

            # Merged backward pass on [n_local-2 .. 0]
            x_next = x_last
            sd_prev = sd_last
            sa_prev = sa_last
            for local_i_py in range(n_local - 2, -1, -1):
                Li = jnp.array(stored_L_host[local_i_py])
                Lli = jnp.array(stored_Ll_host[local_i_py])
                Lai = jnp.array(stored_La_host[local_i_py])
                local_i_jax = jnp.array(local_i_py, dtype=jnp.int32)
                x_i, sd_prev, sa_prev, g_st_i, g_lik_i, g_c_i = _bwd_si_block(
                    Li, Lli, Lai, y_st_local[local_i_py], x_next, x_fe,
                    sd_prev, sa_prev, S_tip, local_i_jax,
                    sc_args_t, coreg_w, likelihood_precs,
                    jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                    jac_coreg_w)
                local_x_host[local_i_py] = np.asarray(x_i)
                x_next = x_i
                g_st_acc = g_st_acc + g_st_i
                g_lik_acc = g_lik_acc + g_lik_i
                g_c_acc = g_c_acc + g_c_i
                del Li, Lli, Lai
        else:
            # Non-root: boundary blocks from RS
            x_block0 = jnp.array(rs_x[2 * rank])
            x_last = jnp.array(rs_x[2 * rank + 1])
            local_x_host[0] = np.asarray(x_block0)
            local_x_host[n_local - 1] = np.asarray(x_last)

            # SI traces for last block (from RS)
            sd_last = jnp.array(rs_sd[2 * rank + 1])
            sa_last = jnp.array(rs_sa[2 * rank + 1])
            sl_last = jnp.array(rs_sl[2 * rank + 1])
            local_i_last = jnp.array(last_local, dtype=jnp.int32)
            g_st_b, g_lik_b, g_c_b = _trace_at_block(
                sd_last, sl_last, sa_last, S_tip, local_i_last,
                sc_args_t, coreg_w, likelihood_precs,
                jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                jac_coreg_w, is_last_global, jnp.array(True))
            g_st_acc = g_st_acc + g_st_b
            g_lik_acc = g_lik_acc + g_lik_b
            g_c_acc = g_c_acc + g_c_b

            # SI data for block[0] from RS
            sd_block0 = jnp.array(rs_sd[2 * rank])
            sa_block0 = jnp.array(rs_sa[2 * rank])
            buf_X_next = jnp.array(rs_sl[2 * rank]).T

            # Merged backward pass on [n_local-2 .. 1]
            x_next = x_last
            sd_prev = sd_last
            sa_prev = sa_last
            for local_i_py in range(n_local - 2, 0, -1):
                Li = jnp.array(stored_L_host[local_i_py])
                Lli = jnp.array(stored_Ll_host[local_i_py])
                Lai = jnp.array(stored_La_host[local_i_py])
                buf_i = jnp.array(stored_buf_host[local_i_py])
                local_i_jax = jnp.array(local_i_py, dtype=jnp.int32)
                x_i, sd_prev, sa_prev, buf_X_next, g_st_i, g_lik_i, g_c_i = \
                    _bwd_si_block_permuted(
                        Li, Lli, Lai, buf_i,
                        y_st_local[local_i_py], x_next, x_fe, x_block0,
                        sd_prev, sa_prev, buf_X_next,
                        sd_block0, sa_block0, S_tip, local_i_jax,
                        sc_args_t, coreg_w, likelihood_precs,
                        jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                        jac_coreg_w)
                local_x_host[local_i_py] = np.asarray(x_i)
                x_next = x_i
                g_st_acc = g_st_acc + g_st_i
                g_lik_acc = g_lik_acc + g_lik_i
                g_c_acc = g_c_acc + g_c_i
                del Li, Lli, Lai, buf_i

            # Block[0]: sl = buf_X[1].T
            sl_block0 = buf_X_next.T
            local_i_zero = jnp.array(0, dtype=jnp.int32)
            g_st_0, g_lik_0, g_c_0 = _trace_at_block(
                sd_block0, sl_block0, sa_block0, S_tip, local_i_zero,
                sc_args_t, coreg_w, likelihood_precs,
                jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                jac_coreg_w, jnp.array(False), jnp.array(False))
            g_st_acc = g_st_acc + g_st_0
            g_lik_acc = g_lik_acc + g_lik_0
            g_c_acc = g_c_acc + g_c_0

        # Allreduce for x and quad
        local_x_st = jnp.array(local_x_host)
        x_st_global, quad_global = _bwd_allgather_reduce(local_x_st, local_quad)
        x = jnp.concatenate([x_st_global.reshape(-1), x_fe])

        # Allreduce for SI gradients
        g_st_acc, g_lik_acc, g_c_acc = _si_allreduce(g_st_acc, g_lik_acc, g_c_acc)
        grad_lik = likelihood_precs * g_lik_acc

        return quad_global, x, g_st_acc, grad_lik, g_c_acc

    # ---- scan-based backward sub + SI ----

    @jax.jit
    def _bwd_si_fori_root_interior(
        x_next_init, sd_prev_init, sa_prev_init,
        g_st_init, g_lik_init, g_c_init,
        S_tip_arg, x_fe,
        cs_chunk, as_chunk, y_st_local,
        indices,
        sc_args_tuple, coreg_w_arg, likelihood_precs_arg,
        jac_q1s_a, jac_q2s_a, jac_q3s_a,
        jac_scale_a, jac_exp_gt_a, jac_coreg_w_arg,
    ):
        (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
         m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a) = sc_args_tuple

        def body_fn(carry, xs_i):
            (x_next, sd_prev, sa_prev,
             g_st_acc, g_lik_acc, g_c_acc) = carry
            local_i, cs_i, as_i = xs_i
            global_i = start_idx + local_i

            sc_loc = []
            for m in range(n_models):
                sc_loc.append({
                    'q1s': q1s_a[m], 'q2s': q2s_a[m], 'q3s': q3s_a[m],
                    'scale': scale_a[m], 'exp_gt': exp_gt_a[m],
                    'm0_diag': m0d_a[m], 'm1_diag': m1d_a[m],
                    'm2_diag': m2d_a[m],
                    'm0_subdiag': m0s_a[m], 'm1_subdiag': m1s_a[m],
                    'm2_subdiag': m2s_a[m],
                })

            q_diag = _reconstruct_coregional_diag_block(
                sc_loc, coreg_w_arg, n_models, ns, global_i)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_diag = q_diag.at[
                    m_off + per_model_ata_diag_rows[m][local_i],
                    m_off + per_model_ata_diag_cols[m][local_i]
                ].add(likelihood_precs_arg[m] * per_model_ata_diag_vals[m][local_i])
            q_diag = q_diag + _chol_eps_reg * _chol_eye_bs - cs_i

            L_i = _jax_cholesky(q_diag)
            L_blk_inv = jax.scipy.linalg.solve_triangular(L_i, _chol_eye_bs, lower=True)

            q_lower = _reconstruct_coregional_lower_block(
                sc_loc, coreg_w_arg, n_models, ns, global_i)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_lower = q_lower.at[
                    m_off + _padded_lower_rows[m][local_i],
                    m_off + _padded_lower_cols[m][local_i]
                ].add(likelihood_precs_arg[m] * _padded_lower_vals[m][local_i])
            L_lower_i = jax.scipy.linalg.solve_triangular(
                L_i, q_lower.T, lower=True).T

            q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_arrow = q_arrow.at[
                    per_model_ata_arrow_rows[m][local_i],
                    m_off + per_model_ata_arrow_cols[m][local_i]
                ].add(likelihood_precs_arg[m] * per_model_ata_arrow_vals[m][local_i])
            q_arrow = q_arrow - as_i
            L_arrow_i = jax.scipy.linalg.solve_triangular(
                L_i, q_arrow.T, lower=True).T

            # Backward sub
            rhs = y_st_local[local_i] - L_arrow_i.T @ x_fe
            rhs = jnp.where(global_i < nt - 1, rhs - L_lower_i.T @ x_next, rhs)
            x_i = jax.scipy.linalg.solve_triangular(L_i.T, rhs, lower=False)

            # SI recurrence
            sl_i = (-sd_prev @ L_lower_i - sa_prev.T @ L_arrow_i) @ L_blk_inv
            sa_i = (-sa_prev @ L_lower_i - S_tip_arg @ L_arrow_i) @ L_blk_inv
            sd_i = (L_blk_inv.T - sl_i.T @ L_lower_i - sa_i.T @ L_arrow_i) @ L_blk_inv

            # Gradient traces: diagonal
            def _diag_body(flat_idx, acc):
                g_st_, g_c_ = acc
                ii = flat_idx // (n_models * n_models)
                jj = (flat_idx // n_models) % n_models
                m_idx = flat_idx % n_models
                sd_ij = lax.dynamic_slice(sd_i, (ii * ns, jj * ns), (ns, ns))
                w_ijm = coreg_w_arg[ii, jj, m_idx]
                base_d = (m0d_a[m_idx, global_i] * q3s_a[m_idx]
                          + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * q2s_a[m_idx]
                          + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * q1s_a[m_idx])
                tr_val = jnp.sum(sd_ij * (scale_a[m_idx] * base_d).T)
                dw = jac_coreg_w_arg[ii, jj, m_idx, :]
                g_c_ = g_c_ + dw * tr_val
                partial_gt = (m1d_a[m_idx, global_i] * q2s_a[m_idx]
                              + 2.0 * exp_gt_a[m_idx] * m2d_a[m_idx, global_i] * q1s_a[m_idx])
                for k in range(3):
                    dQu = (jac_scale_a[m_idx, k] * base_d
                           + scale_a[m_idx] * (m0d_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                               + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                               + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                           + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                    g_st_ = g_st_.at[m_idx, k].add(w_ijm * jnp.sum(sd_ij * dQu.T))
                return (g_st_, g_c_)

            g_st_z = jnp.zeros((n_models, 3), dtype=dtype)
            g_c_z = jnp.zeros(n_coreg, dtype=dtype)
            g_st_d, g_c_d = lax.fori_loop(0, n_models_cubed, _diag_body, (g_st_z, g_c_z))

            # Gradient traces: lower
            def _lower_body(flat_idx, acc):
                g_st_, g_c_ = acc
                ii = flat_idx // (n_models * n_models)
                jj = (flat_idx // n_models) % n_models
                m_idx = flat_idx % n_models
                sl_ij = lax.dynamic_slice(sl_i, (ii * ns, jj * ns), (ns, ns))
                w_ijm = coreg_w_arg[ii, jj, m_idx]
                base_l = (m0s_a[m_idx, global_i] * q3s_a[m_idx]
                          + exp_gt_a[m_idx] * m1s_a[m_idx, global_i] * q2s_a[m_idx]
                          + exp_gt_a[m_idx]**2 * m2s_a[m_idx, global_i] * q1s_a[m_idx])
                tr_l = jnp.sum(sl_ij * (scale_a[m_idx] * base_l).T)
                dw = jac_coreg_w_arg[ii, jj, m_idx, :]
                g_c_ = g_c_ + 2.0 * dw * tr_l
                partial_gt = (m1s_a[m_idx, global_i] * q2s_a[m_idx]
                              + 2.0 * exp_gt_a[m_idx] * m2s_a[m_idx, global_i] * q1s_a[m_idx])
                for k in range(3):
                    dQu_l = (jac_scale_a[m_idx, k] * base_l
                             + scale_a[m_idx] * (m0s_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                                 + exp_gt_a[m_idx] * m1s_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                                 + exp_gt_a[m_idx]**2 * m2s_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                             + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                    g_st_ = g_st_.at[m_idx, k].add(2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))
                return (g_st_, g_c_)

            g_st_l, g_c_l = lax.fori_loop(0, n_models_cubed, _lower_body, (g_st_z, g_c_z))
            g_st_blk = g_st_d + g_st_l
            g_c_blk = g_c_d + g_c_l

            g_lik_blk = jnp.zeros(n_models, dtype=dtype)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                sp_d = jnp.sum(sd_i[m_off + per_model_ata_diag_rows[m][local_i],
                                    m_off + per_model_ata_diag_cols[m][local_i]]
                               * per_model_ata_diag_vals[m][local_i])
                sp_a = jnp.sum(sa_i[per_model_ata_arrow_rows[m][local_i],
                                    m_off + per_model_ata_arrow_cols[m][local_i]]
                               * per_model_ata_arrow_vals[m][local_i])
                sp_l = jnp.sum(sl_i[m_off + _padded_lower_rows[m][local_i],
                                    m_off + _padded_lower_cols[m][local_i]]
                               * _padded_lower_vals[m][local_i])
                g_lik_blk = g_lik_blk.at[m].add(sp_d + 2.0 * sp_a + 2.0 * sp_l)

            new_carry = (x_i, sd_i, sa_i,
                         g_st_acc + g_st_blk, g_lik_acc + g_lik_blk, g_c_acc + g_c_blk)
            return new_carry, x_i

        init_carry = (x_next_init, sd_prev_init, sa_prev_init,
                      g_st_init, g_lik_init, g_c_init)
        (x_out, sd_out, sa_out,
         g_st_acc, g_lik_acc, g_c_acc), x_all = lax.scan(
            body_fn, init_carry,
            (indices, cs_chunk, as_chunk),
            reverse=True)

        return x_all, x_out, sd_out, sa_out, g_st_acc, g_lik_acc, g_c_acc

    @jax.jit
    def _bwd_si_fori_nonroot_interior(
        x_next_init, sd_prev_init, sa_prev_init, buf_X_next_init,
        g_st_init, g_lik_init, g_c_init,
        x_block0, sd_block0, sa_block0,
        S_tip_arg, x_fe,
        cs_chunk, as_chunk, buf_chunk, y_st_local,
        indices,
        sc_args_tuple, coreg_w_arg, likelihood_precs_arg,
        jac_q1s_a, jac_q2s_a, jac_q3s_a,
        jac_scale_a, jac_exp_gt_a, jac_coreg_w_arg,
    ):
        (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
         m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a) = sc_args_tuple

        def body_fn(carry, xs_i):
            (x_next, sd_prev, sa_prev, buf_X_next,
             g_st_acc, g_lik_acc, g_c_acc) = carry
            local_i, cs_i, as_i, buf_i = xs_i
            global_i = start_idx + local_i

            sc_loc = []
            for m in range(n_models):
                sc_loc.append({
                    'q1s': q1s_a[m], 'q2s': q2s_a[m], 'q3s': q3s_a[m],
                    'scale': scale_a[m], 'exp_gt': exp_gt_a[m],
                    'm0_diag': m0d_a[m], 'm1_diag': m1d_a[m],
                    'm2_diag': m2d_a[m],
                    'm0_subdiag': m0s_a[m], 'm1_subdiag': m1s_a[m],
                    'm2_subdiag': m2s_a[m],
                })

            q_diag = _reconstruct_coregional_diag_block(
                sc_loc, coreg_w_arg, n_models, ns, global_i)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_diag = q_diag.at[
                    m_off + per_model_ata_diag_rows[m][local_i],
                    m_off + per_model_ata_diag_cols[m][local_i]
                ].add(likelihood_precs_arg[m] * per_model_ata_diag_vals[m][local_i])
            q_diag = q_diag + _chol_eps_reg * _chol_eye_bs - cs_i

            L_i = _jax_cholesky(q_diag)
            L_blk_inv = jax.scipy.linalg.solve_triangular(L_i, _chol_eye_bs, lower=True)

            q_lower = _reconstruct_coregional_lower_block(
                sc_loc, coreg_w_arg, n_models, ns, global_i)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_lower = q_lower.at[
                    m_off + _padded_lower_rows[m][local_i],
                    m_off + _padded_lower_cols[m][local_i]
                ].add(likelihood_precs_arg[m] * _padded_lower_vals[m][local_i])
            L_lower_i = jax.scipy.linalg.solve_triangular(
                L_i, q_lower.T, lower=True).T

            stored_buf_i = buf_i

            q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                q_arrow = q_arrow.at[
                    per_model_ata_arrow_rows[m][local_i],
                    m_off + per_model_ata_arrow_cols[m][local_i]
                ].add(likelihood_precs_arg[m] * per_model_ata_arrow_vals[m][local_i])
            q_arrow = q_arrow - as_i
            L_arrow_i = jax.scipy.linalg.solve_triangular(
                L_i, q_arrow.T, lower=True).T

            # Backward sub (permuted)
            rhs = y_st_local[local_i] - L_arrow_i.T @ x_fe - stored_buf_i.T @ x_block0
            rhs = jnp.where(global_i < nt - 1, rhs - L_lower_i.T @ x_next, rhs)
            x_i = jax.scipy.linalg.solve_triangular(L_i.T, rhs, lower=False)

            # SI recurrence (permuted)
            buf_L_i = stored_buf_i
            sl_i = (-buf_X_next.T @ buf_L_i
                    - sd_prev @ L_lower_i
                    - sa_prev.T @ L_arrow_i) @ L_blk_inv
            buf_X_i = (-buf_X_next @ L_lower_i
                       - sd_block0 @ buf_L_i
                       - sa_block0.T @ L_arrow_i) @ L_blk_inv
            sa_i = (-sa_prev @ L_lower_i
                    - sa_block0 @ buf_L_i
                    - S_tip_arg @ L_arrow_i) @ L_blk_inv
            sd_i = (L_blk_inv.T
                    - sl_i.T @ L_lower_i
                    - buf_X_i.T @ buf_L_i
                    - sa_i.T @ L_arrow_i) @ L_blk_inv

            # Gradient traces: diagonal
            def _diag_body(flat_idx, acc):
                g_st_, g_c_ = acc
                ii = flat_idx // (n_models * n_models)
                jj = (flat_idx // n_models) % n_models
                m_idx = flat_idx % n_models
                sd_ij = lax.dynamic_slice(sd_i, (ii * ns, jj * ns), (ns, ns))
                w_ijm = coreg_w_arg[ii, jj, m_idx]
                base_d = (m0d_a[m_idx, global_i] * q3s_a[m_idx]
                          + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * q2s_a[m_idx]
                          + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * q1s_a[m_idx])
                tr_val = jnp.sum(sd_ij * (scale_a[m_idx] * base_d).T)
                dw = jac_coreg_w_arg[ii, jj, m_idx, :]
                g_c_ = g_c_ + dw * tr_val
                partial_gt = (m1d_a[m_idx, global_i] * q2s_a[m_idx]
                              + 2.0 * exp_gt_a[m_idx] * m2d_a[m_idx, global_i] * q1s_a[m_idx])
                for k in range(3):
                    dQu = (jac_scale_a[m_idx, k] * base_d
                           + scale_a[m_idx] * (m0d_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                               + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                               + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                           + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                    g_st_ = g_st_.at[m_idx, k].add(w_ijm * jnp.sum(sd_ij * dQu.T))
                return (g_st_, g_c_)

            g_st_z = jnp.zeros((n_models, 3), dtype=dtype)
            g_c_z = jnp.zeros(n_coreg, dtype=dtype)
            g_st_d, g_c_d = lax.fori_loop(0, n_models_cubed, _diag_body, (g_st_z, g_c_z))

            # Gradient traces: lower
            def _lower_body(flat_idx, acc):
                g_st_, g_c_ = acc
                ii = flat_idx // (n_models * n_models)
                jj = (flat_idx // n_models) % n_models
                m_idx = flat_idx % n_models
                sl_ij = lax.dynamic_slice(sl_i, (ii * ns, jj * ns), (ns, ns))
                w_ijm = coreg_w_arg[ii, jj, m_idx]
                base_l = (m0s_a[m_idx, global_i] * q3s_a[m_idx]
                          + exp_gt_a[m_idx] * m1s_a[m_idx, global_i] * q2s_a[m_idx]
                          + exp_gt_a[m_idx]**2 * m2s_a[m_idx, global_i] * q1s_a[m_idx])
                tr_l = jnp.sum(sl_ij * (scale_a[m_idx] * base_l).T)
                dw = jac_coreg_w_arg[ii, jj, m_idx, :]
                g_c_ = g_c_ + 2.0 * dw * tr_l
                partial_gt = (m1s_a[m_idx, global_i] * q2s_a[m_idx]
                              + 2.0 * exp_gt_a[m_idx] * m2s_a[m_idx, global_i] * q1s_a[m_idx])
                for k in range(3):
                    dQu_l = (jac_scale_a[m_idx, k] * base_l
                             + scale_a[m_idx] * (m0s_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                                 + exp_gt_a[m_idx] * m1s_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                                 + exp_gt_a[m_idx]**2 * m2s_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                             + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt)
                    g_st_ = g_st_.at[m_idx, k].add(2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))
                return (g_st_, g_c_)

            g_st_l, g_c_l = lax.fori_loop(0, n_models_cubed, _lower_body, (g_st_z, g_c_z))
            g_st_blk = g_st_d + g_st_l
            g_c_blk = g_c_d + g_c_l

            g_lik_blk = jnp.zeros(n_models, dtype=dtype)
            for m in range(n_models):
                m_off = per_model_offsets[m] * ns
                sp_d = jnp.sum(sd_i[m_off + per_model_ata_diag_rows[m][local_i],
                                    m_off + per_model_ata_diag_cols[m][local_i]]
                               * per_model_ata_diag_vals[m][local_i])
                sp_a = jnp.sum(sa_i[per_model_ata_arrow_rows[m][local_i],
                                    m_off + per_model_ata_arrow_cols[m][local_i]]
                               * per_model_ata_arrow_vals[m][local_i])
                sp_l = jnp.sum(sl_i[m_off + _padded_lower_rows[m][local_i],
                                    m_off + _padded_lower_cols[m][local_i]]
                               * _padded_lower_vals[m][local_i])
                g_lik_blk = g_lik_blk.at[m].add(sp_d + 2.0 * sp_a + 2.0 * sp_l)

            new_carry = (x_i, sd_i, sa_i, buf_X_i,
                         g_st_acc + g_st_blk, g_lik_acc + g_lik_blk, g_c_acc + g_c_blk)
            return new_carry, x_i

        init_carry = (x_next_init, sd_prev_init, sa_prev_init, buf_X_next_init,
                      g_st_init, g_lik_init, g_c_init)
        (x_out, sd_out, sa_out, buf_X_final,
         g_st_acc, g_lik_acc, g_c_acc), x_all = lax.scan(
            body_fn, init_carry,
            (indices, cs_chunk, as_chunk, buf_chunk),
            reverse=True)

        return x_all, x_out, sd_out, sa_out, buf_X_final, g_st_acc, g_lik_acc, g_c_acc

    def _bwd_and_si_fn_fori(theta, cs_all_h, as_all_h, buf_all_h,
                            L_tip, y_st_local, arrow_rhs_total,
                            rs_stored_L, rs_stored_Ll, rs_stored_La,
                            rs_y):
        likelihood_precs = _sc_cache['likelihood_precs'] if 'likelihood_precs' in _sc_cache else _theta_to_likelihood_precs(theta)
        if 'sc_padded' in _sc_cache:
            sc_padded = _sc_cache['sc_padded']
            coreg_w = _sc_cache['coreg_w']
        else:
            sc_list, coreg_w = _theta_to_sc_coreg(theta)
            sc_padded = _pad_subdiags(sc_list)
        jac_sc_list = _sc_cache['jac_sc_list'] if 'jac_sc_list' in _sc_cache else _theta_to_jac_sc(theta)
        jac_coreg_w = _sc_cache['jac_coreg_w'] if 'jac_coreg_w' in _sc_cache else _theta_to_jac_coreg_w(theta)
        _sc_cache['jac_sc_list'] = jac_sc_list
        _sc_cache['jac_coreg_w'] = jac_coreg_w

        q1s_a = jnp.stack([sc_padded[m]['q1s'] for m in range(n_models)])
        q2s_a = jnp.stack([sc_padded[m]['q2s'] for m in range(n_models)])
        q3s_a = jnp.stack([sc_padded[m]['q3s'] for m in range(n_models)])
        scale_a = jnp.array([sc_padded[m]['scale'] for m in range(n_models)])
        exp_gt_a = jnp.array([sc_padded[m]['exp_gt'] for m in range(n_models)])
        m0d_a = jnp.stack([sc_padded[m]['m0_diag'] for m in range(n_models)])
        m1d_a = jnp.stack([sc_padded[m]['m1_diag'] for m in range(n_models)])
        m2d_a = jnp.stack([sc_padded[m]['m2_diag'] for m in range(n_models)])
        m0s_a = jnp.stack([sc_padded[m]['m0_subdiag'] for m in range(n_models)])
        m1s_a = jnp.stack([sc_padded[m]['m1_subdiag'] for m in range(n_models)])
        m2s_a = jnp.stack([sc_padded[m]['m2_subdiag'] for m in range(n_models)])
        sc_args_t = (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
                     m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a)

        jac_q1s_a = jnp.stack([jac_sc_list[m]['q1s'] for m in range(n_models)])
        jac_q2s_a = jnp.stack([jac_sc_list[m]['q2s'] for m in range(n_models)])
        jac_q3s_a = jnp.stack([jac_sc_list[m]['q3s'] for m in range(n_models)])
        jac_scale_a = jnp.stack([jac_sc_list[m]['scale'] for m in range(n_models)])
        jac_exp_gt_a = jnp.stack([jac_sc_list[m]['exp_gt'] for m in range(n_models)])

        # Forward solve for arrow block
        y_fe = jax.scipy.linalg.solve_triangular(L_tip, arrow_rhs_total, lower=True)
        local_quad = jnp.sum(y_st_local ** 2)
        y_fe_quad = jnp.where(rank == comm_size - 1, jnp.sum(y_fe ** 2), 0.0)
        local_quad = local_quad + y_fe_quad
        x_fe = jax.scipy.linalg.solve_triangular(L_tip.T, y_fe, lower=False)

        L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, _chol_eye_nfe, lower=True)
        S_tip = L_tip_inv.T @ L_tip_inv

        # RS backward sub + SI
        rs_x, rs_sd, rs_sl, rs_sa = _rs_bwd_and_si(
            rs_stored_L, rs_stored_Ll, rs_stored_La,
            rs_y, L_tip, y_fe, x_fe, S_tip)

        max_n_local = _max_n_local

        g_st_acc = jnp.zeros((n_models, 3), dtype=dtype)
        g_lik_acc = jnp.zeros(n_models, dtype=dtype)
        g_c_acc = jnp.zeros(n_coreg, dtype=dtype)

        last_local = n_local - 1
        is_last_global = jnp.array(start_idx + last_local == nt - 1)
        chunk = _scan_chunk

        if rank == 0:
            x_last = jnp.array(rs_x[1])

            sd_last = jnp.array(rs_sd[1])
            sa_last = jnp.array(rs_sa[1])
            sl_last = jnp.array(rs_sl[1])
            local_i_last = jnp.array(last_local, dtype=jnp.int32)
            g_st_b, g_lik_b, g_c_b = _trace_at_block(
                sd_last, sl_last, sa_last, S_tip, local_i_last,
                sc_args_t, coreg_w, likelihood_precs,
                jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                jac_coreg_w, is_last_global, jnp.array(True))
            g_st_acc = g_st_acc + g_st_b
            g_lik_acc = g_lik_acc + g_lik_b
            g_c_acc = g_c_acc + g_c_b

            n_interior = n_local - 1
            x_all_h = np.zeros((n_interior, block_size), dtype=np_dtype)

            x_next = x_last
            sd_prev = sd_last
            sa_prev = sa_last

            for chunk_start in reversed(range(0, n_interior, chunk)):
                chunk_end = min(chunk_start + chunk, n_interior)
                indices = jnp.arange(chunk_start, chunk_end)
                cs_chunk = jnp.array(cs_all_h[chunk_start:chunk_end])
                as_chunk = jnp.array(as_all_h[chunk_start:chunk_end])

                x_chunk, x_next, sd_prev, sa_prev, g_st_i, g_lik_i, g_c_i = \
                    _bwd_si_fori_root_interior(
                        x_next, sd_prev, sa_prev,
                        jnp.zeros((n_models, 3), dtype=dtype),
                        jnp.zeros(n_models, dtype=dtype),
                        jnp.zeros(n_coreg, dtype=dtype),
                        S_tip, x_fe,
                        cs_chunk, as_chunk, y_st_local,
                        indices,
                        sc_args_t, coreg_w, likelihood_precs,
                        jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                        jac_coreg_w)
                x_all_h[chunk_start:chunk_end] = np.asarray(x_chunk)
                g_st_acc = g_st_acc + g_st_i
                g_lik_acc = g_lik_acc + g_lik_i
                g_c_acc = g_c_acc + g_c_i
                del cs_chunk, as_chunk, x_chunk

            local_x = jnp.zeros((max_n_local, block_size), dtype=dtype)
            local_x = local_x.at[:n_interior].set(jnp.array(x_all_h))
            local_x = local_x.at[n_local - 1].set(x_last)

        else:
            x_block0 = jnp.array(rs_x[2 * rank])
            x_last = jnp.array(rs_x[2 * rank + 1])

            sd_last = jnp.array(rs_sd[2 * rank + 1])
            sa_last = jnp.array(rs_sa[2 * rank + 1])
            sl_last = jnp.array(rs_sl[2 * rank + 1])
            local_i_last = jnp.array(last_local, dtype=jnp.int32)
            g_st_b, g_lik_b, g_c_b = _trace_at_block(
                sd_last, sl_last, sa_last, S_tip, local_i_last,
                sc_args_t, coreg_w, likelihood_precs,
                jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                jac_coreg_w, is_last_global, jnp.array(True))
            g_st_acc = g_st_acc + g_st_b
            g_lik_acc = g_lik_acc + g_lik_b
            g_c_acc = g_c_acc + g_c_b

            sd_block0 = jnp.array(rs_sd[2 * rank])
            sa_block0 = jnp.array(rs_sa[2 * rank])
            buf_X_next = jnp.array(rs_sl[2 * rank]).T

            n_interior = n_local - 2
            x_all_h = np.zeros((n_interior, block_size), dtype=np_dtype)

            x_next = x_last
            sd_prev = sd_last
            sa_prev = sa_last

            for chunk_start in reversed(range(0, n_interior, chunk)):
                chunk_end = min(chunk_start + chunk, n_interior)
                indices = jnp.arange(chunk_start, chunk_end) + 1
                cs_chunk = jnp.array(cs_all_h[chunk_start:chunk_end])
                as_chunk = jnp.array(as_all_h[chunk_start:chunk_end])
                buf_chunk_j = jnp.array(buf_all_h[chunk_start:chunk_end])

                x_chunk, x_next, sd_prev, sa_prev, buf_X_next, \
                    g_st_i, g_lik_i, g_c_i = \
                    _bwd_si_fori_nonroot_interior(
                        x_next, sd_prev, sa_prev, buf_X_next,
                        jnp.zeros((n_models, 3), dtype=dtype),
                        jnp.zeros(n_models, dtype=dtype),
                        jnp.zeros(n_coreg, dtype=dtype),
                        x_block0, sd_block0, sa_block0,
                        S_tip, x_fe,
                        cs_chunk, as_chunk, buf_chunk_j, y_st_local,
                        indices,
                        sc_args_t, coreg_w, likelihood_precs,
                        jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                        jac_coreg_w)
                x_all_h[chunk_start:chunk_end] = np.asarray(x_chunk)
                g_st_acc = g_st_acc + g_st_i
                g_lik_acc = g_lik_acc + g_lik_i
                g_c_acc = g_c_acc + g_c_i
                del cs_chunk, as_chunk, buf_chunk_j, x_chunk

            # Block[0] trace
            sl_block0 = buf_X_next.T
            local_i_zero = jnp.array(0, dtype=jnp.int32)
            g_st_0, g_lik_0, g_c_0 = _trace_at_block(
                sd_block0, sl_block0, sa_block0, S_tip, local_i_zero,
                sc_args_t, coreg_w, likelihood_precs,
                jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                jac_coreg_w, jnp.array(False), jnp.array(False))
            g_st_acc = g_st_acc + g_st_0
            g_lik_acc = g_lik_acc + g_lik_0
            g_c_acc = g_c_acc + g_c_0

            local_x = jnp.zeros((max_n_local, block_size), dtype=dtype)
            local_x = local_x.at[0].set(x_block0)
            local_x = local_x.at[1:1 + n_interior].set(jnp.array(x_all_h))
            local_x = local_x.at[n_local - 1].set(x_last)

        x_st_global, quad_global = _bwd_allgather_reduce(local_x, local_quad)
        x = jnp.concatenate([x_st_global.reshape(-1), x_fe])

        g_st_acc, g_lik_acc, g_c_acc = _si_allreduce(g_st_acc, g_lik_acc, g_c_acc)
        grad_lik = likelihood_precs * g_lik_acc

        return quad_global, x, g_st_acc, grad_lik, g_c_acc

    # ---- Stage 3: logdet Q_prior gradients ----
    # Two paths for grad_prior (BT Cholesky + SI + gradient traces on Q_prior):
    #
    # 1. Two-phase (parallel): all ranks compute simultaneously, then allgather
    #    boundary + RS factorize. Requires storing (n_local, bs, bs) Schurs on GPU.
    #    Used for small blocks (WA1: bs=3741, ~24 GiB fits on 96 GiB GPU).
    #
    # 2. Pipeline (sequential): rank 0 forward → rank 1 forward → ... → backward.
    #    No GPU memory issues but zero parallelism.
    #    Used for large blocks (AP1/WA2: bs>12000, Schurs would be >50 GiB).
    _grad_prior_total_mem = (n_local + 3 * n_rs) * block_size**2 * 8
    _use_twophase_grad_prior_gpu = _grad_prior_total_mem < 40 * 2**30

    @jax.jit
    def _grad_prior_fn_pipeline_jit(theta, sc_padded, coreg_w, jac_sc_list_padded, jac_coreg_w):
        logdet_prior, grad_prior_st, grad_prior_coreg = \
            pipeline_logdet_Q_prior_coregional_grad(
                sc_padded, jac_sc_list_padded, coreg_w, jac_coreg_w,
                n_models, ns, nt, dtype,
                rank, comm_size, n_local, start_idx, comm)
        grad_prior_st_arr = jnp.stack(grad_prior_st)
        return logdet_prior, grad_prior_st_arr, grad_prior_coreg

    if _use_twophase_grad_prior_gpu:
        @jax.jit
        def _grad_prior_fn_jit(theta, sc_padded, coreg_w, jac_sc_list_padded, jac_coreg_w):
            logdet_prior, grad_prior_st, grad_prior_coreg = \
                twophase_logdet_Q_prior_coregional_grad(
                    sc_padded, jac_sc_list_padded, coreg_w, jac_coreg_w,
                    n_models, ns, nt, dtype,
                    rank, comm_size, n_local, start_idx, comm)
            grad_prior_st_arr = jnp.stack(grad_prior_st)
            return logdet_prior, grad_prior_st_arr, grad_prior_coreg

    def _pad_jac_sc(jac_sc_list):
        jac_sc_list_padded = []
        for m in range(n_models):
            jac_padded = {}
            for key, val in jac_sc_list[m].items():
                if key.endswith('_subdiag'):
                    jac_padded[key] = jnp.concatenate(
                        [val, jnp.zeros((1,) + val.shape[1:], dtype=dtype)], axis=0)
                else:
                    jac_padded[key] = val
            jac_sc_list_padded.append(jac_padded)
        return jac_sc_list_padded

    def _grad_prior_fn(theta):
        if 'sc_padded' in _sc_cache:
            sc_padded = _sc_cache['sc_padded']
            coreg_w = _sc_cache['coreg_w']
        else:
            sc_list, coreg_w = _theta_to_sc_coreg(theta)
            sc_padded = _pad_subdiags(sc_list)
        jac_sc_list = _sc_cache['jac_sc_list'] if 'jac_sc_list' in _sc_cache else _theta_to_jac_sc(theta)
        jac_coreg_w = _sc_cache['jac_coreg_w'] if 'jac_coreg_w' in _sc_cache else _theta_to_jac_coreg_w(theta)
        jac_sc_list_padded = _pad_jac_sc(jac_sc_list)
        if _use_twophase_grad_prior_gpu:
            return _grad_prior_fn_jit(theta, sc_padded, coreg_w, jac_sc_list_padded, jac_coreg_w)
        else:
            return _grad_prior_fn_pipeline_jit(theta, sc_padded, coreg_w, jac_sc_list_padded, jac_coreg_w)

    # ---- Stage 4: Quadratic form gradients ----

    @jax.jit
    def _grad_quad_fn_jit(theta, x, sc_list, coreg_w, likelihood_precs,
                          jac_sc_list, jac_coreg_w):
        rhs = _build_rhs(theta)

        grad_quad_st, grad_quad_lik, grad_quad_coreg = \
            pipeline_compute_grad_quad_coregional(
                x, sc_list, jac_sc_list, coreg_w, jac_coreg_w,
                n_models, nt, ns, n_fe,
                rhs, likelihood_precs,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                a_sparse, y, n_observations_idx,
                rank, comm_size, n_local, start_idx, comm)

        grad_quad_st_arr = jnp.stack(grad_quad_st)
        return grad_quad_st_arr, grad_quad_lik, grad_quad_coreg

    def _grad_quad_fn(theta, x):
        likelihood_precs = _sc_cache['likelihood_precs'] if 'likelihood_precs' in _sc_cache else _theta_to_likelihood_precs(theta)
        if 'sc_list' in _sc_cache:
            sc_list = _sc_cache['sc_list']
            coreg_w = _sc_cache['coreg_w']
        else:
            sc_list, coreg_w = _theta_to_sc_coreg(theta)
        jac_sc_list = _sc_cache['jac_sc_list'] if 'jac_sc_list' in _sc_cache else _theta_to_jac_sc(theta)
        jac_coreg_w = _sc_cache['jac_coreg_w'] if 'jac_coreg_w' in _sc_cache else _theta_to_jac_coreg_w(theta)
        return _grad_quad_fn_jit(theta, x, sc_list, coreg_w, likelihood_precs,
                                 jac_sc_list, jac_coreg_w)

    # ---- Stage 5: Scalar gradient terms ----

    @jax.jit
    def _grad_scalar_fn(theta):
        def _scalar_terms(theta_):
            log_prior_hp = _evaluate_log_prior_hyperparameters_jax(theta_, prior_configs)
            eta = jnp.zeros_like(y)
            log_lik = 0.0
            for m in range(n_models):
                obs_start = n_observations_idx[m]
                obs_end = n_observations_idx[m + 1]
                prec_idx = hyperparameters_idx[m + 1] - 1
                log_lik += _evaluate_gaussian_likelihood_jax(
                    eta[obs_start:obs_end], y[obs_start:obs_end], theta_[prec_idx])
            return -(log_prior_hp + log_lik)
        return jax.grad(_scalar_terms)(theta)

    # ---- Gradient combiner ----

    def _combine_gradients(theta_jax,
                           grad_cond_st, grad_cond_lik, grad_cond_coreg,
                           grad_prior_st, grad_prior_coreg,
                           grad_quad_st, grad_quad_lik, grad_quad_coreg,
                           grad_scalar):
        bar_logdet_prior = -0.5
        bar_logdet_cond = 0.5
        bar_quad = -0.5

        bar_theta = grad_scalar.copy()

        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            n_st_m = hp_end - hp_start

            grad_st_m = (
                bar_logdet_prior * grad_prior_st[m, :n_st_m]
                + bar_logdet_cond * grad_cond_st[m, :n_st_m]
                + bar_quad * grad_quad_st[m, :n_st_m]
            )
            bar_theta = bar_theta.at[hp_start:hp_end].add(grad_st_m)

            prec_idx = hyperparameters_idx[m + 1] - 1
            grad_lik_m = (
                bar_logdet_cond * grad_cond_lik[m]
                + bar_quad * grad_quad_lik[m]
            )
            bar_theta = bar_theta.at[prec_idx].add(grad_lik_m)

        grad_coreg_total = (
            bar_logdet_prior * grad_prior_coreg
            + bar_logdet_cond * grad_cond_coreg
            + bar_quad * grad_quad_coreg
        )
        for m in range(n_models):
            bar_theta = bar_theta.at[sigma_idx + m].add(grad_coreg_total[m])
        for li in range(n_lambdas):
            bar_theta = bar_theta.at[lambda_indices[li]].add(
                grad_coreg_total[n_sigmas + li])

        return bar_theta

    # ---- Public API ----

    def _compute_objective(theta_jax, logdet_cond, quad, logdet_prior):
        log_prior_hp = _evaluate_log_prior_hyperparameters_jax(theta_jax, prior_configs)
        log_lik = 0.0
        eta = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            prec_idx = hyperparameters_idx[m + 1] - 1
            log_lik += _evaluate_gaussian_likelihood_jax(
                eta[obs_start:obs_end], y[obs_start:obs_end], theta_jax[prec_idx])
        return -(log_prior_hp + log_lik
                 + 0.5 * logdet_prior - 0.5 * logdet_cond + 0.5 * quad)

    def objective_with_grad(theta):
        _sc_cache.clear()
        theta_jax = jnp.asarray(theta, dtype=dtype)

        if _use_fori_loop[0]:
            (cs_all, as_all, buf_all,
             y_st_local, L_tip, arrow_rhs_total, logdet_cond,
             rs_diag_g, rs_lower_g, rs_arrow_g,
             rs_stored_cs, rs_stored_as,
             rs_stored_L, rs_stored_Ll, rs_stored_La,
             rs_y) = _chol_fn_fori(theta_jax)

            quad, x_val, grad_cond_st, grad_cond_lik, grad_cond_coreg = \
                _bwd_and_si_fn_fori(
                    theta_jax, cs_all, as_all, buf_all,
                    L_tip, y_st_local, arrow_rhs_total,
                    rs_stored_L, rs_stored_Ll, rs_stored_La,
                    rs_y)
            del y_st_local, arrow_rhs_total, rs_y
            del cs_all, as_all, buf_all, L_tip
            del rs_diag_g, rs_lower_g, rs_arrow_g, rs_stored_cs, rs_stored_as
            del rs_stored_L, rs_stored_Ll, rs_stored_La
        else:
            (L_host, Ll_host, La_host,
             cs_host, as_host, buf_host,
             y_st_local, L_tip, arrow_rhs_total, logdet_cond,
             rs_diag_g, rs_lower_g, rs_arrow_g,
             rs_stored_cs, rs_stored_as,
             rs_stored_L, rs_stored_Ll, rs_stored_La,
             rs_y) = _chol_fn(theta_jax)

            quad, x_val, grad_cond_st, grad_cond_lik, grad_cond_coreg = \
                _bwd_and_si_fn(
                    theta_jax, L_host, Ll_host, La_host, buf_host,
                    L_tip, y_st_local, arrow_rhs_total,
                    rs_stored_L, rs_stored_Ll, rs_stored_La,
                    rs_y)
            del y_st_local, arrow_rhs_total, rs_y
            del L_host, Ll_host, La_host, cs_host, as_host, buf_host, L_tip
            del rs_diag_g, rs_lower_g, rs_arrow_g, rs_stored_cs, rs_stored_as
            del rs_stored_L, rs_stored_Ll, rs_stored_La

        logdet_prior, grad_prior_st, grad_prior_coreg = _grad_prior_fn(theta_jax)

        f_val = _compute_objective(theta_jax, logdet_cond, quad, logdet_prior)

        grad_quad_st, grad_quad_lik, grad_quad_coreg = _grad_quad_fn(
            theta_jax, x_val)

        grad_scalar = _grad_scalar_fn(theta_jax)

        grad_val = _combine_gradients(
            theta_jax,
            grad_cond_st, grad_cond_lik, grad_cond_coreg,
            grad_prior_st, grad_prior_coreg,
            grad_quad_st, grad_quad_lik, grad_quad_coreg,
            grad_scalar)

        return float(f_val), np.asarray(grad_val, dtype=np_dtype), np.asarray(x_val, dtype=np_dtype)

    def timed_objective_with_grad(theta):
        import time as _time
        _sc_cache.clear()
        timing = {}
        theta_jax = jnp.asarray(theta, dtype=dtype)

        t0 = _time.perf_counter()
        if _use_fori_loop[0]:
            (cs_all, as_all, buf_all,
             y_st_local, L_tip, arrow_rhs_total, logdet_cond,
             rs_diag_g, rs_lower_g, rs_arrow_g,
             rs_stored_cs, rs_stored_as,
             rs_stored_L, rs_stored_Ll, rs_stored_La,
             rs_y) = _chol_fn_fori(theta_jax)
        else:
            (L_host, Ll_host, La_host,
             cs_host, as_host, buf_host,
             y_st_local, L_tip, arrow_rhs_total, logdet_cond,
             rs_diag_g, rs_lower_g, rs_arrow_g,
             rs_stored_cs, rs_stored_as,
             rs_stored_L, rs_stored_Ll, rs_stored_La,
             rs_y) = _chol_fn(theta_jax)
        jax.block_until_ready(logdet_cond)
        timing["chol_fwd_sub"] = _time.perf_counter() - t0
        timing["chol_detail"] = dict(_chol_detail_timing)

        t0 = _time.perf_counter()
        if _use_fori_loop[0]:
            quad, x_val, grad_cond_st, grad_cond_lik, grad_cond_coreg = \
                _bwd_and_si_fn_fori(
                    theta_jax, cs_all, as_all, buf_all,
                    L_tip, y_st_local, arrow_rhs_total,
                    rs_stored_L, rs_stored_Ll, rs_stored_La,
                    rs_y)
            jax.block_until_ready((quad, x_val))
            timing["bwd_si"] = _time.perf_counter() - t0
            del y_st_local, arrow_rhs_total, rs_y
            del cs_all, as_all, buf_all, L_tip
            del rs_diag_g, rs_lower_g, rs_arrow_g, rs_stored_cs, rs_stored_as
            del rs_stored_L, rs_stored_Ll, rs_stored_La
        else:
            quad, x_val, grad_cond_st, grad_cond_lik, grad_cond_coreg = \
                _bwd_and_si_fn(
                    theta_jax, L_host, Ll_host, La_host, buf_host,
                    L_tip, y_st_local, arrow_rhs_total,
                    rs_stored_L, rs_stored_Ll, rs_stored_La,
                    rs_y)
            jax.block_until_ready((quad, x_val))
            timing["bwd_si"] = _time.perf_counter() - t0
            del y_st_local, arrow_rhs_total, rs_y
            del L_host, Ll_host, La_host, cs_host, as_host, buf_host, L_tip
            del rs_diag_g, rs_lower_g, rs_arrow_g, rs_stored_cs, rs_stored_as
            del rs_stored_L, rs_stored_Ll, rs_stored_La

        t0 = _time.perf_counter()
        logdet_prior, grad_prior_st, grad_prior_coreg = _grad_prior_fn(theta_jax)
        jax.block_until_ready(logdet_prior)
        timing["grad_prior"] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        f_val = _compute_objective(theta_jax, logdet_cond, quad, logdet_prior)
        jax.block_until_ready(f_val)
        timing["compute_f"] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        grad_quad_st, grad_quad_lik, grad_quad_coreg = _grad_quad_fn(
            theta_jax, x_val)
        jax.block_until_ready((grad_quad_st, grad_quad_lik))
        timing["grad_quad"] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        grad_scalar = _grad_scalar_fn(theta_jax)
        jax.block_until_ready(grad_scalar)
        timing["grad_scalar"] = _time.perf_counter() - t0

        grad_val = _combine_gradients(
            theta_jax,
            grad_cond_st, grad_cond_lik, grad_cond_coreg,
            grad_prior_st, grad_prior_coreg,
            grad_quad_st, grad_quad_lik, grad_quad_coreg,
            grad_scalar)

        return float(f_val), np.asarray(grad_val, dtype=np_dtype), \
            np.asarray(x_val, dtype=np_dtype), timing

    def timed_perblock_objective_with_grad(theta):
        _tp_profile_perblock[0] = True
        try:
            f_val, grad_val, x_val, timing = timed_objective_with_grad(theta)
        finally:
            _tp_profile_perblock[0] = False
        return f_val, grad_val, x_val, timing

    def timed_prebuild_objective_with_grad(theta):
        _use_prebuild[0] = True
        try:
            f_val, grad_val, x_val, timing = timed_objective_with_grad(theta)
        finally:
            _use_prebuild[0] = False
        return f_val, grad_val, x_val, timing

    def timed_streaming_objective_with_grad(theta):
        _use_cupy_streaming[0] = True
        try:
            f_val, grad_val, x_val, timing = timed_objective_with_grad(theta)
        finally:
            _use_cupy_streaming[0] = False
        return f_val, grad_val, x_val, timing

    def timed_fori_objective_with_grad(theta):
        _use_fori_loop[0] = True
        try:
            f_val, grad_val, x_val, timing = timed_objective_with_grad(theta)
        finally:
            _use_fori_loop[0] = False
        return f_val, grad_val, x_val, timing

    objective_with_grad.timed = timed_objective_with_grad
    objective_with_grad.timed_perblock = timed_perblock_objective_with_grad
    objective_with_grad.timed_prebuild = timed_prebuild_objective_with_grad
    objective_with_grad.timed_streaming = timed_streaming_objective_with_grad
    objective_with_grad.timed_fori = timed_fori_objective_with_grad
    objective_with_grad._use_cupy_streaming = _use_cupy_streaming
    objective_with_grad._use_fori_loop = _use_fori_loop
    objective_with_grad._scan_chunk = _scan_chunk

    def objective_fn(theta):
        _sc_cache.clear()
        theta_jax = jnp.asarray(theta, dtype=dtype)
        (L_h, Ll_h, La_h,
         cs_host, as_host, buf_host,
         y_st_local, L_tip, arrow_rhs_total, logdet_cond,
         rs_diag_g, rs_lower_g, rs_arrow_g,
         rs_stored_cs, rs_stored_as,
         rs_stored_L, rs_stored_Ll, rs_stored_La,
         rs_y) = _chol_fn(theta_jax)
        quad, x_val = _bwd_fn(
            theta_jax, L_h, Ll_h, La_h, buf_host,
            L_tip, y_st_local, arrow_rhs_total,
            rs_stored_L, rs_stored_Ll, rs_stored_La,
            rs_y)
        logdet_prior = _logdet_prior_fn(theta_jax)
        f_val = _compute_objective(theta_jax, logdet_cond, quad, logdet_prior)
        return float(f_val), np.asarray(x_val, dtype=np_dtype)

    # JIT compilation: trace and compile each stage with dummy inputs.
    # This is a one-time cost; subsequent calls reuse the compiled XLA programs.
    import time as _compile_time
    theta_init = jnp.ones(n_hyperparameters, dtype=dtype)
    from dalia.utils import print_msg

    _use_scan_str = f"fori, chunk={_scan_chunk}" if _use_fori_loop[0] else "per-block"
    _rs_str = "scan" if _use_rs_scan else "per-block"
    _gp_str = "two-phase" if _use_twophase_grad_prior_gpu else "pipeline"
    print_msg(f"Two-phase ({_use_scan_str}, RS={_rs_str}, grad_prior={_gp_str}): "
              f"JIT compiling all stages...")

    _t_compile = _compile_time.perf_counter()

    if _use_fori_loop[0]:
        print_msg("  compiling Cholesky + forward sub...", flush=True)
        (cs0, as0, buf0, yst0, lt0, arr0, logdet0,
         rsd0, rsl0, rsa0, rscs0, rsas0,
         rsL0, rsLl0, rsLa0, rsy0) = _chol_fn_fori(theta_init)
        print_msg("  compiling merged bwd+SI...", flush=True)
        quad0, x0, _, _, _ = _bwd_and_si_fn_fori(
            theta_init, cs0, as0, buf0, lt0, yst0, arr0,
            rsL0, rsLl0, rsLa0, rsy0)
        del yst0, arr0, rsy0
        del cs0, as0, buf0, lt0
        del rsd0, rsl0, rsa0, rscs0, rsas0, rsL0, rsLl0, rsLa0
    else:
        print_msg("  compiling Cholesky + forward sub...", flush=True)
        (L0, Ll0, La0, cs0, as0, buf0, yst0, lt0, arr0, logdet0,
         rsd0, rsl0, rsa0, rscs0, rsas0,
         rsL0, rsLl0, rsLa0, rsy0) = _chol_fn(theta_init)
        print_msg("  compiling merged bwd+SI...", flush=True)
        quad0, x0, _, _, _ = _bwd_and_si_fn(
            theta_init, L0, Ll0, La0, buf0, lt0, yst0, arr0,
            rsL0, rsLl0, rsLa0, rsy0)
        del yst0, arr0, rsy0
        del L0, Ll0, La0, cs0, as0, buf0, lt0
        del rsd0, rsl0, rsa0, rscs0, rsas0, rsL0, rsLl0, rsLa0

    print_msg("  compiling prior logdet + gradient...", flush=True)
    _ = _grad_prior_fn(theta_init)
    print_msg("  compiling quad gradient...", flush=True)
    _ = _grad_quad_fn(theta_init, x0)
    print_msg("  compiling scalar gradient...", flush=True)
    _ = _grad_scalar_fn(theta_init)

    _t_compile = _compile_time.perf_counter() - _t_compile
    print_msg(f"  all stages compiled in {_t_compile:.1f}s.")

    return objective_fn, objective_with_grad
