# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Dict, Any

import numpy as np

import jax.numpy as jnp
from scipy import sparse as scipy_sparse

from dalia.core.autodiff.config import (
    get_jax_dtype,
    _to_numpy,
    _scipy_sparse_to_jax_bcoo,
    _extract_bta_blocks_coregional,
)
from dalia.core.autodiff.q_construction import (
    extract_bta_blocks_sparse_coo,
)
from dalia.core.autodiff.spatial_precompute import (
    extract_bta_blocks_sparse_coo_coregional,
)


def _extract_static_data(dalia_instance, dtype=None) -> Dict[str, Any]:
    """Extract static data from DALIA instance for pure JAX function.

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance.
    dtype : jnp.dtype, optional
        JAX dtype to use. If None, uses the configured dtype from get_jax_dtype().

    Returns
    -------
    static_data : dict
        Dictionary containing model-specific data for all likelihood types.
    """
    if dtype is None:
        dtype = get_jax_dtype()
    model = dalia_instance.model
    likelihood_type = model.likelihood_config.type

    # Check if we have a spatio-temporal model
    has_spatio_temporal = any(
        hasattr(sm, 'submodel_type') and sm.submodel_type == 'spatio_temporal'
        for sm in model.submodels
    )

    # Check if we have a spatial model (but not spatio-temporal)
    has_spatial = any(
        hasattr(sm, 'submodel_type') and sm.submodel_type == 'spatial'
        for sm in model.submodels
    )

    # Determine solver type from config - use sparse only for serinv solver with ST model
    solver_type = dalia_instance.config.solver.type if hasattr(dalia_instance.config.solver, 'type') else 'dense'
    use_sparse_path = has_spatio_temporal and solver_type == 'serinv'

    prior_configs = []
    for i, prior_hp in enumerate(model.prior_hyperparameters):
        prior_type = prior_hp.config.type if hasattr(prior_hp.config, 'type') else 'unknown'

        if prior_type == 'penalized_complexity':
            hp_type = getattr(prior_hp, 'hyperparameter_type', 'unknown')
            alpha = float(prior_hp.config.alpha) if hasattr(prior_hp.config, 'alpha') else 0.01
            u = float(prior_hp.config.u) if hasattr(prior_hp.config, 'u') else 5.0
            lambda_theta = float(prior_hp.lambda_theta)
            prior_configs.append({
                'prior_type': 'penalized_complexity',
                'hyperparameter_type': hp_type,
                'alpha': alpha,
                'u': u,
                'lambda_theta': lambda_theta,
            })
        elif prior_type == 'gaussian':
            mean = float(prior_hp.mean) if hasattr(prior_hp, 'mean') else 0.0
            precision = float(prior_hp.precision) if hasattr(prior_hp, 'precision') else 1.0
            prior_configs.append({
                'prior_type': 'gaussian',
                'mean': mean,
                'precision': precision,
            })
        else:
            prior_configs.append({
                'prior_type': 'unknown',
            })

    fixed_effects_precision = 0.001
    for submodel in model.submodels:
        if hasattr(submodel, 'submodel_type') and submodel.submodel_type == 'regression':
            if hasattr(submodel.config, 'fixed_effects_prior_precision'):
                fixed_effects_precision = float(submodel.config.fixed_effects_prior_precision)
            break

    # For spatio-temporal models with serinv solver, keep A sparse and precompute AtA in BTA format
    if use_sparse_path:
        # Get nt, ns from spatio-temporal submodel
        st_submodel = next(
            sm for sm in model.submodels
            if hasattr(sm, 'submodel_type') and sm.submodel_type == 'spatio_temporal'
        )
        nt = int(st_submodel.nt)
        ns = int(st_submodel.ns)
        n_fixed_effects = int(model.n_fixed_effects)

        # Get sparse A matrix
        if hasattr(model.a, 'get'):
            a_scipy = scipy_sparse.csr_matrix(model.a.get())
        elif hasattr(model.a, 'toarray'):
            a_scipy = scipy_sparse.csr_matrix(model.a)
        else:
            a_scipy = scipy_sparse.csr_matrix(model.a)

        # Precompute A^T @ A as sparse and extract BTA blocks in COO format.
        # Storing COO triplets instead of dense (nt, ns, ns) blocks saves
        # ~60 GB for gst_large where each block has O(ns) nonzeros.
        ata_scipy = a_scipy.T @ a_scipy
        ata_sparse_coo = extract_bta_blocks_sparse_coo(
            ata_scipy, nt, ns, n_fixed_effects, dtype=dtype
        )

        # Store sparse A for A^T @ y operations
        a_sparse = _scipy_sparse_to_jax_bcoo(a_scipy, dtype=dtype)

        static_data = {
            'likelihood_type': likelihood_type,
            'has_spatio_temporal': True,
            'a_sparse': a_sparse,
            **ata_sparse_coo,
            'y': jnp.array(_to_numpy(model.y), dtype=dtype),
            'n_fixed_effects': n_fixed_effects,
            'fixed_effects_precision': float(fixed_effects_precision),
            'prior_configs': prior_configs,
            'n_observations': int(model.n_observations),
            'n_latent_parameters': int(model.n_latent_parameters),
            'inner_iter_tol': float(dalia_instance.config.eps_inner_iteration),
            'inner_iter_max': int(dalia_instance.config.inner_iteration_max_iter),
            'use_sparse_solver': True,
            'nt': nt,
            'ns': ns,
            'manifold': str(st_submodel.manifold),
            'spatial_matrices': {
                'c0': jnp.array(_to_numpy(st_submodel.c0.toarray()), dtype=dtype),
                'g1': jnp.array(_to_numpy(st_submodel.g1.toarray()), dtype=dtype),
                'g2': jnp.array(_to_numpy(st_submodel.g2.toarray()), dtype=dtype),
                'g3': jnp.array(_to_numpy(st_submodel.g3.toarray()), dtype=dtype),
            },
            'temporal_matrices': {
                'm0': jnp.array(_to_numpy(st_submodel.m0.toarray()), dtype=dtype),
                'm1': jnp.array(_to_numpy(st_submodel.m1.toarray()), dtype=dtype),
                'm2': jnp.array(_to_numpy(st_submodel.m2.toarray()), dtype=dtype),
            },
        }
    else:
        # For dense solver, store A as dense
        a_matrix = _to_numpy(model.a.toarray()) if hasattr(model.a, 'toarray') else _to_numpy(model.a)
        static_data = {
            'likelihood_type': likelihood_type,
            'has_spatio_temporal': has_spatio_temporal,
            'has_spatial': has_spatial,
            'a': jnp.array(a_matrix, dtype=dtype),
            'y': jnp.array(_to_numpy(model.y), dtype=dtype),
            'n_fixed_effects': int(model.n_fixed_effects),
            'fixed_effects_precision': float(fixed_effects_precision),
            'prior_configs': prior_configs,
            'n_observations': int(model.n_observations),
            'n_latent_parameters': int(model.n_latent_parameters),
            'inner_iter_tol': float(dalia_instance.config.eps_inner_iteration),
            'inner_iter_max': int(dalia_instance.config.inner_iteration_max_iter),
            'use_sparse_solver': False,
        }

        # Add spatial matrices for dense solver with spatial model
        if has_spatial and not has_spatio_temporal:
            spatial_submodel = next(
                sm for sm in model.submodels
                if hasattr(sm, 'submodel_type') and sm.submodel_type == 'spatial'
            )
            static_data['ns'] = int(spatial_submodel.ns)
            static_data['spatial_matrices'] = {
                'c0': jnp.array(_to_numpy(spatial_submodel.c0.toarray()), dtype=dtype),
                'g1': jnp.array(_to_numpy(spatial_submodel.g1.toarray()), dtype=dtype),
                'g2': jnp.array(_to_numpy(spatial_submodel.g2.toarray()), dtype=dtype),
            }
            # Track submodel ordering - find offsets in latent vector
            fe_offset = 0
            spatial_offset = 0
            current_offset = 0
            for sm in model.submodels:
                if hasattr(sm, 'submodel_type'):
                    if sm.submodel_type == 'regression':
                        fe_offset = current_offset
                    elif sm.submodel_type == 'spatial':
                        spatial_offset = current_offset
                current_offset += sm.n_latent_parameters
            static_data['fe_offset'] = int(fe_offset)
            static_data['spatial_offset'] = int(spatial_offset)

        # Add spatio-temporal matrices for dense solver with ST model
        if has_spatio_temporal:
            st_submodel = next(
                sm for sm in model.submodels
                if hasattr(sm, 'submodel_type') and sm.submodel_type == 'spatio_temporal'
            )
            static_data['nt'] = int(st_submodel.nt)
            static_data['ns'] = int(st_submodel.ns)
            static_data['manifold'] = str(st_submodel.manifold)
            static_data['spatial_matrices'] = {
                'c0': jnp.array(_to_numpy(st_submodel.c0.toarray()), dtype=dtype),
                'g1': jnp.array(_to_numpy(st_submodel.g1.toarray()), dtype=dtype),
                'g2': jnp.array(_to_numpy(st_submodel.g2.toarray()), dtype=dtype),
                'g3': jnp.array(_to_numpy(st_submodel.g3.toarray()), dtype=dtype),
            }
            static_data['temporal_matrices'] = {
                'm0': jnp.array(_to_numpy(st_submodel.m0.toarray()), dtype=dtype),
                'm1': jnp.array(_to_numpy(st_submodel.m1.toarray()), dtype=dtype),
                'm2': jnp.array(_to_numpy(st_submodel.m2.toarray()), dtype=dtype),
            }
            # Track submodel ordering - find offsets in latent vector
            fe_offset = 0
            st_offset = 0
            current_offset = 0
            for sm in model.submodels:
                if hasattr(sm, 'submodel_type'):
                    if sm.submodel_type == 'regression':
                        fe_offset = current_offset
                    elif sm.submodel_type == 'spatio_temporal':
                        st_offset = current_offset
                current_offset += sm.n_latent_parameters
            static_data['fe_offset'] = int(fe_offset)
            static_data['st_offset'] = int(st_offset)

    # Add x_initial for inner iteration initialization (critical for Poisson/Binomial)
    x_initial = _to_numpy(model.x)
    static_data['x_initial'] = jnp.array(x_initial, dtype=dtype)

    if likelihood_type == 'poisson':
        if hasattr(model.likelihood, 'e'):
            e = _to_numpy(model.likelihood.e)
        else:
            e = np.ones(model.n_observations)
        static_data['e'] = jnp.array(e, dtype=dtype)

    elif likelihood_type == 'binomial':
        if hasattr(model.likelihood, 'n_trials'):
            n_trials = _to_numpy(model.likelihood.n_trials)
        else:
            n_trials = np.ones(model.n_observations)
        static_data['n_trials'] = jnp.array(n_trials, dtype=dtype)

    return static_data


def _extract_static_data_coregional(dalia_instance, dtype=None, include_dense_ata=False) -> Dict[str, Any]:
    """Extract static data from DALIA instance for CoregionalModel.

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance with CoregionalModel.
    dtype : jnp.dtype, optional
        JAX dtype to use. If None, uses the configured dtype from get_jax_dtype().

    Returns
    -------
    static_data : dict
        Dictionary containing model-specific data for coregional models.
    """
    if dtype is None:
        dtype = get_jax_dtype()
    model = dalia_instance.model
    n_models = model.n_models
    ns = model.n_spatial_nodes
    nt = model.n_temporal_nodes
    n_fixed_effects_per_model = model.n_fixed_effects_per_model

    # Determine solver type
    solver_type = dalia_instance.config.solver.type if hasattr(dalia_instance.config.solver, 'type') else 'dense'
    use_sparse_path = solver_type == 'serinv'

    # Extract prior configs for all hyperparameters
    prior_configs = []
    for i, prior_hp in enumerate(model.prior_hyperparameters):
        prior_type = prior_hp.config.type if hasattr(prior_hp.config, 'type') else 'unknown'

        if prior_type == 'penalized_complexity':
            hp_type = getattr(prior_hp, 'hyperparameter_type', 'unknown')
            alpha = float(prior_hp.config.alpha) if hasattr(prior_hp.config, 'alpha') else 0.01
            u = float(prior_hp.config.u) if hasattr(prior_hp.config, 'u') else 5.0
            lambda_theta = float(prior_hp.lambda_theta)
            prior_configs.append({
                'prior_type': 'penalized_complexity',
                'hyperparameter_type': hp_type,
                'alpha': alpha,
                'u': u,
                'lambda_theta': lambda_theta,
            })
        elif prior_type == 'gaussian':
            mean = float(prior_hp.mean) if hasattr(prior_hp, 'mean') else 0.0
            precision = float(prior_hp.precision) if hasattr(prior_hp, 'precision') else 1.0
            prior_configs.append({
                'prior_type': 'gaussian',
                'mean': mean,
                'precision': precision,
            })
        else:
            prior_configs.append({
                'prior_type': 'unknown',
            })

    # Detect coregionalization type
    coregionalization_type = getattr(model, 'coregionalization_type', 'spatio_temporal')
    is_spatial_only = coregionalization_type == 'spatial'

    # Extract per-model data
    models_data = []
    fixed_effects_precision = 0.001

    for i, m in enumerate(model.models):
        submodel = m.submodels[0]

        if is_spatial_only:
            model_data = {
                'spatial_matrices': {
                    'c0': jnp.array(_to_numpy(submodel.c0.toarray()), dtype=dtype),
                    'g1': jnp.array(_to_numpy(submodel.g1.toarray()), dtype=dtype),
                    'g2': jnp.array(_to_numpy(submodel.g2.toarray()), dtype=dtype),
                },
                'likelihood_type': m.likelihood_config.type,
                'n_observations': int(m.n_observations) if hasattr(m, 'n_observations') else 0,
            }
        else:
            model_data = {
                'manifold': str(submodel.manifold),
                'spatial_matrices': {
                    'c0': jnp.array(_to_numpy(submodel.c0.toarray()), dtype=dtype),
                    'g1': jnp.array(_to_numpy(submodel.g1.toarray()), dtype=dtype),
                    'g2': jnp.array(_to_numpy(submodel.g2.toarray()), dtype=dtype),
                    'g3': jnp.array(_to_numpy(submodel.g3.toarray()), dtype=dtype),
                },
                'temporal_matrices': {
                    'm0': jnp.array(_to_numpy(submodel.m0.toarray()), dtype=dtype),
                    'm1': jnp.array(_to_numpy(submodel.m1.toarray()), dtype=dtype),
                    'm2': jnp.array(_to_numpy(submodel.m2.toarray()), dtype=dtype),
                },
                'likelihood_type': m.likelihood_config.type,
                'n_observations': int(m.n_observations) if hasattr(m, 'n_observations') else 0,
            }

        if len(m.submodels) > 1 and hasattr(m.submodels[1], 'submodel_type') and m.submodels[1].submodel_type == 'regression':
            if hasattr(m.submodels[1].config, 'fixed_effects_prior_precision'):
                fixed_effects_precision = float(m.submodels[1].config.fixed_effects_prior_precision)

        models_data.append(model_data)

    # Get sparse A matrix
    if hasattr(model.a, 'get'):
        a_scipy = scipy_sparse.csr_matrix(model.a.get())
    elif hasattr(model.a, 'toarray'):
        a_scipy = scipy_sparse.csr_matrix(model.a)
    else:
        a_scipy = scipy_sparse.csr_matrix(model.a)

    # Coregional block size: n_models * ns
    block_size = n_models * ns
    n_blocks = nt
    n_fixed_effects_total = n_fixed_effects_per_model * n_models

    if use_sparse_path:
        # Precompute per-model A_i^T @ A_i contributions in BTA format
        # This allows proper handling of per-model likelihood precisions
        n_observations_idx = [int(x) for x in model.n_observations_idx]

        # Dense per-model AtA (only for non-fused fallback on small models)
        per_model_ata_diag = []
        per_model_ata_lower = []
        per_model_ata_arrow = []
        per_model_ata_tip = []

        # Sparse COO per-model AtA (for fused path)
        pm_coo_diag_rows = []
        pm_coo_diag_cols = []
        pm_coo_diag_vals = []
        pm_coo_lower_rows = []
        pm_coo_lower_cols = []
        pm_coo_lower_vals = []
        pm_coo_arrow_rows = []
        pm_coo_arrow_cols = []
        pm_coo_arrow_vals = []
        pm_coo_tip = []
        pm_offsets = []

        for i in range(n_models):
            obs_start = n_observations_idx[i]
            obs_end = n_observations_idx[i + 1]
            a_i = a_scipy[obs_start:obs_end, :]
            ata_i = a_i.T @ a_i

            if include_dense_ata:
                ata_diag_i, ata_lower_i, ata_arrow_i, ata_tip_i = _extract_bta_blocks_coregional(
                    ata_i, n_blocks, block_size, n_fixed_effects_total, dtype=dtype
                )
                per_model_ata_diag.append(ata_diag_i)
                per_model_ata_lower.append(ata_lower_i)
                per_model_ata_arrow.append(ata_arrow_i)
                per_model_ata_tip.append(ata_tip_i)

            coo_data = extract_bta_blocks_sparse_coo_coregional(
                ata_i, n_models, nt, ns, n_fixed_effects_total, dtype=dtype
            )
            pm_coo_diag_rows.append(coo_data['ata_diag_rows'])
            pm_coo_diag_cols.append(coo_data['ata_diag_cols'])
            pm_coo_diag_vals.append(coo_data['ata_diag_vals'])
            pm_coo_lower_rows.append(coo_data['ata_lower_rows'])
            pm_coo_lower_cols.append(coo_data['ata_lower_cols'])
            pm_coo_lower_vals.append(coo_data['ata_lower_vals'])
            pm_coo_arrow_rows.append(coo_data['ata_arrow_rows'])
            pm_coo_arrow_cols.append(coo_data['ata_arrow_cols'])
            pm_coo_arrow_vals.append(coo_data['ata_arrow_vals'])
            pm_coo_tip.append(coo_data['ata_tip'])
            pm_offsets.append(coo_data['model_offset'])

        if include_dense_ata:
            ata_diag_per_model = jnp.stack(per_model_ata_diag, axis=0)
            ata_lower_per_model = jnp.stack(per_model_ata_lower, axis=0)
            ata_arrow_per_model = jnp.stack(per_model_ata_arrow, axis=0)
            ata_tip_per_model = jnp.stack(per_model_ata_tip, axis=0)

        a_sparse = _scipy_sparse_to_jax_bcoo(a_scipy, dtype=dtype)

        static_data = {
            'is_coregional': True,
            'n_models': n_models,
            'models_data': models_data,
            'a_sparse': a_sparse,
            # Sparse COO per-model data for fused path
            'per_model_ata_diag_rows': pm_coo_diag_rows,
            'per_model_ata_diag_cols': pm_coo_diag_cols,
            'per_model_ata_diag_vals': pm_coo_diag_vals,
            'per_model_ata_lower_rows': pm_coo_lower_rows,
            'per_model_ata_lower_cols': pm_coo_lower_cols,
            'per_model_ata_lower_vals': pm_coo_lower_vals,
            'per_model_ata_arrow_rows': pm_coo_arrow_rows,
            'per_model_ata_arrow_cols': pm_coo_arrow_cols,
            'per_model_ata_arrow_vals': pm_coo_arrow_vals,
            'per_model_ata_tip': pm_coo_tip,
            'per_model_offsets': pm_offsets,
            'use_fused': True,
            'y': jnp.array(_to_numpy(model.y), dtype=dtype),
            'n_fixed_effects_per_model': n_fixed_effects_per_model,
            'n_fixed_effects_total': n_fixed_effects_total,
            'fixed_effects_precision': float(fixed_effects_precision),
            'prior_configs': prior_configs,
            'n_observations': int(model.n_observations),
            'n_observations_idx': n_observations_idx,
            'n_latent_parameters': int(model.n_latent_parameters),
            'inner_iter_tol': float(dalia_instance.config.eps_inner_iteration),
            'inner_iter_max': int(dalia_instance.config.inner_iteration_max_iter),
            'use_sparse_solver': True,
            'nt': nt,
            'ns': ns,
            'block_size': block_size,
            'hyperparameters_idx': [int(x) for x in model.hyperparameters_idx],
            'theta_keys': list(model.theta_keys),
        }

        if include_dense_ata:
            static_data['ata_diag_per_model'] = ata_diag_per_model
            static_data['ata_lower_per_model'] = ata_lower_per_model
            static_data['ata_arrow_per_model'] = ata_arrow_per_model
            static_data['ata_tip_per_model'] = ata_tip_per_model
    else:
        # Dense solver path
        a_dense = jnp.array(a_scipy.toarray(), dtype=dtype)
        n_observations_idx = [int(x) for x in model.n_observations_idx]

        static_data = {
            'is_coregional': True,
            'is_spatial_only': is_spatial_only,
            'n_models': n_models,
            'models_data': models_data,
            'a': a_dense,
            'y': jnp.array(_to_numpy(model.y), dtype=dtype),
            'n_fixed_effects_per_model': n_fixed_effects_per_model,
            'n_fixed_effects_total': n_fixed_effects_total,
            'fixed_effects_precision': float(fixed_effects_precision),
            'prior_configs': prior_configs,
            'n_observations': int(model.n_observations),
            'n_observations_idx': n_observations_idx,
            'n_latent_parameters': int(model.n_latent_parameters),
            'inner_iter_tol': float(dalia_instance.config.eps_inner_iteration),
            'inner_iter_max': int(dalia_instance.config.inner_iteration_max_iter),
            'use_sparse_solver': False,
            'nt': nt,
            'ns': ns,
            'block_size': block_size,
            'hyperparameters_idx': [int(x) for x in model.hyperparameters_idx],
            'theta_keys': list(model.theta_keys),
        }

    # Add x_initial for inner iteration initialization
    x_initial = _to_numpy(model.x)
    static_data['x_initial'] = jnp.array(x_initial, dtype=dtype)

    return static_data


def _extract_static_data_distributed_coregional(dalia_instance, comm, dtype=None) -> Dict[str, Any]:
    """Extract static data with per-rank slicing for distributed coregional JAX autodiff.

    Extends :func:`_extract_static_data_coregional` by splitting per-model
    BTA block arrays across MPI ranks along the time dimension.

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance with CoregionalModel.
    comm : MPI communicator
    dtype : jnp.dtype, optional

    Returns
    -------
    static_data : dict
        Per-rank sliced data plus distribution metadata.
    """
    static_data = _extract_static_data_coregional(dalia_instance, dtype=dtype)

    if not static_data.get('use_sparse_solver', False):
        raise NotImplementedError(
            "Distributed JAX autodiff for CoregionalModel requires serinv solver.")

    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    nt = static_data['nt']
    n_models = static_data['n_models']

    # Balance partition: root uses standard BTA (faster), non-root uses
    # permuted BTA with buffer (~2x slower per block). Give root more
    # blocks so both sides finish their local scan at the same time.
    # Solve: root_n + (P-1)*nonroot_n = nt
    #        (root_n - 1) * 1.0 = (nonroot_n - 2) * ratio
    if comm_size > 1:
        ratio = 2.0
        P = comm_size
        nonroot_n = int(round((nt - 1 + 2 * ratio) / (ratio + P - 1)))
        nonroot_n = max(nonroot_n, 3)
        root_n = nt - (P - 1) * nonroot_n
        root_n = max(root_n, 3)
        remainder = nt - root_n - (P - 1) * nonroot_n
        root_n += remainder
        n_locals = [root_n] + [nonroot_n] * (P - 1)
    else:
        n_locals = [nt]
    start_idx = sum(n_locals[:rank])
    n_local = n_locals[rank]

    sliced_diag_rows = []
    sliced_diag_cols = []
    sliced_diag_vals = []
    sliced_lower_rows = []
    sliced_lower_cols = []
    sliced_lower_vals = []
    sliced_arrow_rows = []
    sliced_arrow_cols = []
    sliced_arrow_vals = []

    for m in range(n_models):
        sliced_diag_rows.append(
            static_data['per_model_ata_diag_rows'][m][start_idx:start_idx + n_local])
        sliced_diag_cols.append(
            static_data['per_model_ata_diag_cols'][m][start_idx:start_idx + n_local])
        sliced_diag_vals.append(
            static_data['per_model_ata_diag_vals'][m][start_idx:start_idx + n_local])

        sliced_arrow_rows.append(
            static_data['per_model_ata_arrow_rows'][m][start_idx:start_idx + n_local])
        sliced_arrow_cols.append(
            static_data['per_model_ata_arrow_cols'][m][start_idx:start_idx + n_local])
        sliced_arrow_vals.append(
            static_data['per_model_ata_arrow_vals'][m][start_idx:start_idx + n_local])

        full_lr = static_data['per_model_ata_lower_rows'][m]
        full_lc = static_data['per_model_ata_lower_cols'][m]
        full_lv = static_data['per_model_ata_lower_vals'][m]

        if rank == 0:
            zero_r = jnp.zeros((1, full_lr.shape[1]), dtype=jnp.int32)
            zero_c = jnp.zeros((1, full_lc.shape[1]), dtype=jnp.int32)
            zero_v = jnp.zeros((1, full_lv.shape[1]), dtype=full_lv.dtype)
            sliced_lower_rows.append(
                jnp.concatenate([zero_r, full_lr[:n_local - 1]], axis=0))
            sliced_lower_cols.append(
                jnp.concatenate([zero_c, full_lc[:n_local - 1]], axis=0))
            sliced_lower_vals.append(
                jnp.concatenate([zero_v, full_lv[:n_local - 1]], axis=0))
        else:
            lower_start = start_idx - 1
            sliced_lower_rows.append(full_lr[lower_start:lower_start + n_local])
            sliced_lower_cols.append(full_lc[lower_start:lower_start + n_local])
            sliced_lower_vals.append(full_lv[lower_start:lower_start + n_local])

    static_data['per_model_ata_diag_rows'] = sliced_diag_rows
    static_data['per_model_ata_diag_cols'] = sliced_diag_cols
    static_data['per_model_ata_diag_vals'] = sliced_diag_vals
    static_data['per_model_ata_lower_rows'] = sliced_lower_rows
    static_data['per_model_ata_lower_cols'] = sliced_lower_cols
    static_data['per_model_ata_lower_vals'] = sliced_lower_vals
    static_data['per_model_ata_arrow_rows'] = sliced_arrow_rows
    static_data['per_model_ata_arrow_cols'] = sliced_arrow_cols
    static_data['per_model_ata_arrow_vals'] = sliced_arrow_vals

    static_data['rank'] = rank
    static_data['comm_size'] = comm_size
    static_data['n_local'] = n_local
    static_data['start_idx'] = start_idx
    static_data['max_n_local'] = max(n_locals)

    # Cache numpy copies of spatial/temporal matrices for CuPy streaming path
    models_data_np = []
    for m_data in static_data['models_data']:
        md_np = {
            'spatial_matrices': {
                k: np.asarray(v) for k, v in m_data['spatial_matrices'].items()
            },
        }
        if 'temporal_matrices' in m_data:
            md_np['temporal_matrices'] = {
                k: np.asarray(v) for k, v in m_data['temporal_matrices'].items()
            }
        if 'manifold' in m_data:
            md_np['manifold'] = m_data['manifold']
        models_data_np.append(md_np)
    static_data['models_data_np'] = models_data_np

    # Numpy copies of AtA arrays for streaming Q build
    static_data['per_model_ata_diag_rows_np'] = [np.asarray(a) for a in sliced_diag_rows]
    static_data['per_model_ata_diag_cols_np'] = [np.asarray(a) for a in sliced_diag_cols]
    static_data['per_model_ata_diag_vals_np'] = [np.asarray(a) for a in sliced_diag_vals]
    static_data['per_model_ata_lower_rows_np'] = [np.asarray(a) for a in sliced_lower_rows]
    static_data['per_model_ata_lower_cols_np'] = [np.asarray(a) for a in sliced_lower_cols]
    static_data['per_model_ata_lower_vals_np'] = [np.asarray(a) for a in sliced_lower_vals]
    static_data['per_model_ata_arrow_rows_np'] = [np.asarray(a) for a in sliced_arrow_rows]
    static_data['per_model_ata_arrow_cols_np'] = [np.asarray(a) for a in sliced_arrow_cols]
    static_data['per_model_ata_arrow_vals_np'] = [np.asarray(a) for a in sliced_arrow_vals]
    static_data['per_model_ata_tip_np'] = [np.asarray(t) for t in static_data['per_model_ata_tip']]
    static_data['per_model_offsets_np'] = [int(o) for o in static_data['per_model_offsets']]

    return static_data
