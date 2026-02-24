# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Callable, Tuple, Dict, Any
import numpy as np

import jax
import jax.numpy as jnp

_JAX_DTYPE = None


def configure_jax_precision(precision: str = "float64"):
    """Configure JAX precision. Call before creating JAX objectives.

    Parameters
    ----------
    precision : str
        Either "float32" or "float64".

    Returns
    -------
    dtype : jnp.dtype
        The configured JAX dtype.
    """
    global _JAX_DTYPE
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
        _JAX_DTYPE = jnp.float64
    else:
        jax.config.update("jax_enable_x64", False)
        _JAX_DTYPE = jnp.float32
    return _JAX_DTYPE


def get_jax_dtype():
    """Get the configured JAX dtype (defaults to float64 if not configured)."""
    global _JAX_DTYPE
    if _JAX_DTYPE is None:
        configure_jax_precision("float64")
    return _JAX_DTYPE


from jax import lax
from jax.experimental import sparse as jax_sparse
from scipy import sparse as scipy_sparse

from dalia.core.jax_sparse_helpers import (
    build_spatio_temporal_Q_jax,
    build_spatio_temporal_Q_bta_jax,
    build_Q_conditional_jax,
    kronecker_to_bta_structure,
    extract_bta_blocks_from_sparse,
    extract_bta_blocks_sparse_coo,
    extract_bta_blocks_sparse_coo_coregional,
    scatter_sparse_ata_into_blocks,
    compute_logdet_from_cholesky_bta_jax,
    solve_bta_system_jax,
    build_coregional_Q_bta_jax,
    quadratic_form_bta_jax,
    precompute_spatial_components,
    precompute_spatial_components_coregional,
    _reconstruct_coregional_diag_block,
    _reconstruct_coregional_lower_block,
    lazy_bta_cholesky,
    lazy_bta_cholesky_carries,
    fused_cholesky_fwd_sub,
    fused_cholesky_fwd_sub_coregional,
    backward_sub_from_carries,
    backward_sub_from_carries_coregional,
    pobtasi_jax,
    logdet_Q_st_scan,
    logdet_Q_prior_coregional_scan,
    logdet_Q_prior_coregional_grad,
    bt_logdet_grad,
    _compute_grad_logdet_cond,
    _compute_grad_quad,
    _compute_grad_quad_coregional,
    selected_inversion_grads_jax,
    selected_inversion_grads_from_carries_jax,
    selected_inversion_grads_from_carries_coregional,
    pipeline_fused_cholesky_fwd_sub,
    pipeline_backward_sub_from_carries,
    pipeline_selected_inversion_grads_from_carries,
    pipeline_bt_logdet_grad,
    pipeline_compute_grad_quad,
    _jax_cholesky,
    pipeline_fused_cholesky_fwd_sub_coregional,
    pipeline_backward_sub_from_carries_coregional,
    pipeline_logdet_Q_prior_coregional_scan,
    pipeline_logdet_Q_prior_coregional_grad,
    pipeline_selected_inversion_grads_from_carries_coregional,
    pipeline_compute_grad_quad_coregional,
)
from serinv.algs.pobtaf_jax import pobtaf_jax_optimized
from serinv.algs.pobtf_jax import pobtf_logdet_jax


def _to_numpy(arr):
    """Convert array to NumPy, handling CuPy arrays."""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def _scipy_sparse_to_jax_bcoo(sp_matrix, dtype=None):
    """Convert scipy sparse matrix to JAX BCOO format."""
    if dtype is None:
        dtype = get_jax_dtype()
    if hasattr(sp_matrix, 'get'):
        sp_matrix = sp_matrix.get()
    coo = scipy_sparse.coo_matrix(sp_matrix)
    indices = jnp.array(np.column_stack([coo.row, coo.col]), dtype=jnp.int32)
    data = jnp.array(coo.data, dtype=dtype)
    return jax_sparse.BCOO((data, indices), shape=coo.shape)


def create_pure_jax_objective(dalia_instance, dtype=None) -> Tuple[Callable, Callable]:
    """Create pure JAX objective function with automatic differentiation.

    Supports:
    - Gaussian, Poisson, and Binomial likelihoods
    - Dense or sparse (serinv) solvers
    - Single-process (no MPI)
    - Models with zero hyperparameters (Poisson/Binomial only)

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance.
    dtype : jnp.dtype, optional
        JAX dtype to use. If None, uses the configured dtype from get_jax_dtype().

    Returns
    -------
    objective_func : Callable
        Pure JAX objective function.
    objective_with_grad : Callable
        Function returning both forward value and gradient.
    """
    if dtype is None:
        dtype = get_jax_dtype()
    np_dtype = np.float64 if dtype == jnp.float64 else np.float32
    static_data = _extract_static_data(dalia_instance, dtype=dtype)
    likelihood_type = static_data['likelihood_type']
    use_sparse = static_data.get('use_sparse_solver', False)
    n_hyperparameters = dalia_instance.model.n_hyperparameters

    # Handle zero hyperparameters case
    if n_hyperparameters == 0:
        if use_sparse:
            raise NotImplementedError(
                "JAX autodiff with zero hyperparameters is not supported for sparse solver. "
                "Use gradient_method='finite_diff' instead."
            )
        if likelihood_type == 'gaussian':
            raise NotImplementedError(
                "JAX autodiff with zero hyperparameters is not supported for Gaussian likelihood "
                "(requires at least observation precision hyperparameter). "
                "Use gradient_method='finite_diff' instead."
            )

        # For Poisson/Binomial with no hyperparameters, create value-only functions
        def objective_pure_jax_no_hp(theta):
            if likelihood_type == 'poisson':
                return _objective_poisson_dense(theta, static_data)
            elif likelihood_type == 'binomial':
                return _objective_binomial_dense(theta, static_data)
            else:
                raise ValueError(f"Unsupported likelihood type for zero hyperparameters: {likelihood_type}")

        objective_pure_jax_no_hp = jax.jit(objective_pure_jax_no_hp)

        # Warmup with empty array
        theta_init = jnp.array([], dtype=dtype)
        _ = objective_pure_jax_no_hp(theta_init)

        def objective_with_grad_no_hp(theta):
            theta_jax = jnp.asarray(theta, dtype=dtype)
            f_val = objective_pure_jax_no_hp(theta_jax)
            return float(f_val), np.array([], dtype=np_dtype)

        return objective_pure_jax_no_hp, objective_with_grad_no_hp

    # Check if model has spatio-temporal or spatial component
    has_st = static_data.get('has_spatio_temporal', False)
    has_spatial = static_data.get('has_spatial', False)

    def objective_pure_jax(theta):
        """Pure JAX objective function - dispatches based on likelihood and solver type

        Returns (objective, x) where x is the latent parameters.
        """
        if use_sparse:
            if likelihood_type == 'gaussian':
                return _objective_gaussian_sparse(theta, static_data)
            else:
                raise ValueError(f"Sparse solver only supports Gaussian likelihood, got {likelihood_type}")
        else:
            if likelihood_type == 'gaussian':
                if has_st:
                    return _objective_gaussian_st_dense(theta, static_data)
                elif has_spatial:
                    return _objective_gaussian_spatial_dense(theta, static_data)
                return _objective_gaussian_dense(theta, static_data)
            elif likelihood_type == 'poisson':
                if has_st:
                    return _objective_poisson_st_dense(theta, static_data)
                return _objective_poisson_dense(theta, static_data)
            elif likelihood_type == 'binomial':
                if has_st:
                    raise NotImplementedError(
                        "JAX autodiff for Binomial likelihood with spatio-temporal models is not yet implemented. "
                        "Use gradient_method='finite_diff' instead."
                    )
                return _objective_binomial_dense(theta, static_data)
            else:
                raise ValueError(f"Unsupported likelihood type: {likelihood_type}")

    # Use has_aux=True since objective functions return (objective, x)
    value_and_grad_fn = jax.value_and_grad(objective_pure_jax, has_aux=True)

    objective_pure_jax = jax.jit(objective_pure_jax)
    value_and_grad_fn = jax.jit(value_and_grad_fn)

    # Warmup JIT compilation
    theta_init = jnp.ones(n_hyperparameters, dtype=dtype)
    _ = value_and_grad_fn(theta_init)

    def objective_with_grad(theta):
        """Returns (f_val, grad, x) where x is the latent parameters."""
        theta_jax = jnp.asarray(theta, dtype=dtype)
        (f_val, x_val), grad_val = value_and_grad_fn(theta_jax)
        return float(f_val), np.asarray(grad_val, dtype=np_dtype), np.asarray(x_val, dtype=np_dtype)

    return objective_pure_jax, objective_with_grad


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


def _extract_static_data_distributed(dalia_instance, comm, dtype=None) -> Dict[str, Any]:
    """Extract static data with per-rank slicing for distributed JAX autodiff.

    Extends :func:`_extract_static_data` by splitting BTA block arrays
    across MPI ranks along the time dimension.

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance.
    comm : MPI communicator
    dtype : jnp.dtype, optional

    Returns
    -------
    static_data : dict
        Per-rank sliced data plus distribution metadata.
    """
    # Start from the full static data (replicated on all ranks)
    static_data = _extract_static_data(dalia_instance, dtype=dtype)

    if not static_data.get('use_sparse_solver', False):
        raise NotImplementedError(
            "Distributed JAX autodiff requires sparse (serinv) solver.")

    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    nt = static_data['nt']

    # Compute per-rank block counts: rank 0 gets remainder
    n_locals = [nt // comm_size] * comm_size
    n_locals[0] += nt % comm_size
    start_idx = sum(n_locals[:rank])
    n_local = n_locals[rank]

    # Slice BTA block arrays to local range
    ata_diag_rows = static_data['ata_diag_rows'][start_idx:start_idx + n_local]
    ata_diag_cols = static_data['ata_diag_cols'][start_idx:start_idx + n_local]
    ata_diag_vals = static_data['ata_diag_vals'][start_idx:start_idx + n_local]

    ata_arrow_rows = static_data['ata_arrow_rows'][start_idx:start_idx + n_local]
    ata_arrow_cols = static_data['ata_arrow_cols'][start_idx:start_idx + n_local]
    ata_arrow_vals = static_data['ata_arrow_vals'][start_idx:start_idx + n_local]

    # Lower blocks: rank r needs lowers [start_idx-1 : start_idx+n_local-1]
    # (the lower block at global index i couples block i+1 to block i)
    # For the scan, lower[local_j] is the lower block entering step start_idx+j,
    # which is ata_lower[start_idx+j-1] in global indexing.
    # Rank 0: first lower is zero (no incoming coupling), then lowers [0..n_local-2]
    # Rank r>0: lowers [start_idx-1..start_idx+n_local-2]
    full_lower_rows = static_data['ata_lower_rows']  # (nt-1, max_nnz)
    full_lower_cols = static_data['ata_lower_cols']
    full_lower_vals = static_data['ata_lower_vals']

    if rank == 0:
        # Pad first row with zeros, then take lowers [0..n_local-2]
        zero_row_r = jnp.zeros((1, full_lower_rows.shape[1]), dtype=jnp.int32)
        zero_row_c = jnp.zeros((1, full_lower_cols.shape[1]), dtype=jnp.int32)
        zero_row_v = jnp.zeros((1, full_lower_vals.shape[1]), dtype=full_lower_vals.dtype)

        local_lower_rows = jnp.concatenate([zero_row_r, full_lower_rows[:n_local - 1]], axis=0)
        local_lower_cols = jnp.concatenate([zero_row_c, full_lower_cols[:n_local - 1]], axis=0)
        local_lower_vals = jnp.concatenate([zero_row_v, full_lower_vals[:n_local - 1]], axis=0)
    else:
        lower_start = start_idx - 1
        local_lower_rows = full_lower_rows[lower_start:lower_start + n_local]
        local_lower_cols = full_lower_cols[lower_start:lower_start + n_local]
        local_lower_vals = full_lower_vals[lower_start:lower_start + n_local]

    # Replace global arrays with local slices
    static_data['ata_diag_rows'] = ata_diag_rows
    static_data['ata_diag_cols'] = ata_diag_cols
    static_data['ata_diag_vals'] = ata_diag_vals
    static_data['ata_lower_rows'] = local_lower_rows
    static_data['ata_lower_cols'] = local_lower_cols
    static_data['ata_lower_vals'] = local_lower_vals
    static_data['ata_arrow_rows'] = ata_arrow_rows
    static_data['ata_arrow_cols'] = ata_arrow_cols
    static_data['ata_arrow_vals'] = ata_arrow_vals

    # Add distribution metadata
    static_data['rank'] = rank
    static_data['comm_size'] = comm_size
    static_data['n_local'] = n_local
    static_data['start_idx'] = start_idx

    # Slice rhs_st will be done at eval time (depends on theta)
    # but we can note that y and a_sparse remain global/replicated

    return static_data


def create_pure_jax_objective_distributed(dalia_instance, comm, dtype=None) -> Tuple[Callable, Callable]:
    """Create distributed JAX objective with analytical gradients via mpi4jax.

    Uses pipeline communication for the forward/backward passes across ranks.
    Each rank stores only its local Schur complement carries, solving the
    OOM problem for large models.

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance.
    comm : MPI communicator
    dtype : jnp.dtype, optional

    Returns
    -------
    objective_func : Callable
    objective_with_grad : Callable
    """
    if dtype is None:
        dtype = get_jax_dtype()
    np_dtype = np.float64 if dtype == jnp.float64 else np.float32

    static_data = _extract_static_data_distributed(dalia_instance, comm, dtype=dtype)

    nt = static_data['nt']
    ns = static_data['ns']
    n_fe = static_data['n_fixed_effects']
    fe_prec = static_data['fixed_effects_precision']
    spatial_matrices = static_data['spatial_matrices']
    temporal_matrices = static_data['temporal_matrices']
    manifold = static_data['manifold']
    y = static_data['y']
    a_sparse = static_data['a_sparse']
    prior_configs = static_data['prior_configs']

    ata_diag_rows = static_data['ata_diag_rows']
    ata_diag_cols = static_data['ata_diag_cols']
    ata_diag_vals = static_data['ata_diag_vals']
    ata_lower_rows = static_data['ata_lower_rows']
    ata_lower_cols = static_data['ata_lower_cols']
    ata_lower_vals = static_data['ata_lower_vals']
    ata_arrow_rows = static_data['ata_arrow_rows']
    ata_arrow_cols = static_data['ata_arrow_cols']
    ata_arrow_vals = static_data['ata_arrow_vals']
    ata_tip = static_data['ata_tip']

    rank = static_data['rank']
    comm_size = static_data['comm_size']
    n_local = static_data['n_local']
    start_idx = static_data['start_idx']

    n_theta_st = 3
    n_hyperparameters = dalia_instance.model.n_hyperparameters

    @jax.custom_vjp
    def fused_core_dist(theta_st, theta_lik):
        lik_prec = jnp.exp(theta_lik)
        sc = precompute_spatial_components(theta_st, spatial_matrices, temporal_matrices, manifold)

        rhs = lik_prec * (a_sparse.T @ y)
        rhs_st_global = rhs[:nt * ns].reshape(nt, ns)
        rhs_fe = rhs[nt * ns:]
        rhs_st_local = rhs_st_global[start_idx:start_idx + n_local]

        stored_cs, stored_as, y_st_local, L_tip, arrow_rhs_acc, logdet_cond = \
            pipeline_fused_cholesky_fwd_sub(
                sc, nt, ns, n_fe, fe_prec, lik_prec,
                rhs_st_local, rhs_fe,
                ata_diag_rows, ata_diag_cols, ata_diag_vals,
                ata_lower_rows, ata_lower_cols, ata_lower_vals,
                ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
                ata_tip, dtype,
                rank, comm_size, n_local, start_idx, comm,
            )

        x_st_global, x_fe, quad = pipeline_backward_sub_from_carries(
            stored_cs, stored_as, L_tip,
            y_st_local, arrow_rhs_acc,
            sc, lik_prec,
            ata_diag_rows, ata_diag_cols, ata_diag_vals,
            ata_lower_rows, ata_lower_cols, ata_lower_vals,
            ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
            nt, ns, n_fe, dtype,
            rank, comm_size, n_local, start_idx, comm,
        )

        x = jnp.concatenate([x_st_global.reshape(-1), x_fe])

        logdet_st = logdet_Q_st_scan(theta_st, spatial_matrices, temporal_matrices, manifold, nt, ns, dtype)

        return logdet_st, logdet_cond, quad, x

    def fused_core_dist_fwd(theta_st, theta_lik):
        lik_prec = jnp.exp(theta_lik)
        sc = precompute_spatial_components(theta_st, spatial_matrices, temporal_matrices, manifold)

        rhs = lik_prec * (a_sparse.T @ y)
        rhs_st_global = rhs[:nt * ns].reshape(nt, ns)
        rhs_fe = rhs[nt * ns:]
        rhs_st_local = rhs_st_global[start_idx:start_idx + n_local]

        stored_cs, stored_as, y_st_local, L_tip, arrow_rhs_acc, logdet_cond = \
            pipeline_fused_cholesky_fwd_sub(
                sc, nt, ns, n_fe, fe_prec, lik_prec,
                rhs_st_local, rhs_fe,
                ata_diag_rows, ata_diag_cols, ata_diag_vals,
                ata_lower_rows, ata_lower_cols, ata_lower_vals,
                ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
                ata_tip, dtype,
                rank, comm_size, n_local, start_idx, comm,
            )

        x_st_global, x_fe, quad = pipeline_backward_sub_from_carries(
            stored_cs, stored_as, L_tip,
            y_st_local, arrow_rhs_acc,
            sc, lik_prec,
            ata_diag_rows, ata_diag_cols, ata_diag_vals,
            ata_lower_rows, ata_lower_cols, ata_lower_vals,
            ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
            nt, ns, n_fe, dtype,
            rank, comm_size, n_local, start_idx, comm,
        )

        x = jnp.concatenate([x_st_global.reshape(-1), x_fe])

        logdet_st = logdet_Q_st_scan(theta_st, spatial_matrices, temporal_matrices, manifold, nt, ns, dtype)

        # Residuals: local carries only (~32 GiB / P per rank)
        residuals = (theta_st, theta_lik, x, stored_cs, stored_as, L_tip)
        return (logdet_st, logdet_cond, quad, x), residuals

    def fused_core_dist_bwd(residuals, g):
        bar_logdet_st, bar_logdet_cond, bar_quad, _bar_x = g
        theta_st_r, theta_lik_r, x_r, stored_cs_r, stored_as_r, L_tip_r = residuals

        lik_prec = jnp.exp(theta_lik_r)

        sc = precompute_spatial_components(theta_st_r, spatial_matrices, temporal_matrices, manifold)
        jac_sc = jax.jacfwd(precompute_spatial_components)(
            theta_st_r, spatial_matrices, temporal_matrices, manifold)

        # Selected inversion gradients (distributed)
        grad_cond_st, grad_cond_lik = pipeline_selected_inversion_grads_from_carries(
            stored_cs_r, stored_as_r, L_tip_r,
            sc, jac_sc,
            nt, ns, n_fe, n_theta_st,
            lik_prec,
            ata_diag_rows, ata_diag_cols, ata_diag_vals,
            ata_lower_rows, ata_lower_cols, ata_lower_vals,
            ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
            ata_tip, dtype,
            rank, comm_size, n_local, start_idx, comm,
        )

        # Quadratic form gradient (distributed)
        rhs = lik_prec * (a_sparse.T @ y)
        grad_quad_st, grad_quad_lik = pipeline_compute_grad_quad(
            x_r, sc, jac_sc,
            nt, ns, n_fe, n_theta_st,
            rhs, lik_prec,
            ata_diag_rows, ata_diag_cols, ata_diag_vals,
            ata_lower_rows, ata_lower_cols, ata_lower_vals,
            ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
            ata_tip,
            rank, comm_size, n_local, start_idx, comm,
        )

        # logdet Q_st gradient (distributed)
        grad_logdet_st = pipeline_bt_logdet_grad(
            theta_st_r, spatial_matrices, temporal_matrices,
            manifold, nt, ns, n_theta_st, dtype,
            rank, comm_size, n_local, start_idx, comm,
        )

        bar_theta_st = (
            bar_logdet_st * grad_logdet_st
            + bar_logdet_cond * grad_cond_st
            + bar_quad * grad_quad_st
        )
        bar_theta_lik = (
            bar_logdet_cond * grad_cond_lik
            + bar_quad * grad_quad_lik
        )

        return bar_theta_st, bar_theta_lik

    fused_core_dist.defvjp(fused_core_dist_fwd, fused_core_dist_bwd)

    def objective_pure_jax(theta):
        theta_st = theta[:-1]
        theta_likelihood = theta[-1]

        logdet_Q_st_val, logdet_Q_cond_val, quad_form, x = fused_core_dist(theta_st, theta_likelihood)

        log_prior_hp = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)
        eta = jnp.zeros_like(y)
        log_lik = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

        objective = -(
            log_prior_hp
            + log_lik
            + 0.5 * logdet_Q_st_val
            - 0.5 * logdet_Q_cond_val
            + 0.5 * quad_form
        )

        return objective, x

    value_and_grad_fn = jax.value_and_grad(objective_pure_jax, has_aux=True)

    objective_pure_jax = jax.jit(objective_pure_jax)
    value_and_grad_fn = jax.jit(value_and_grad_fn)

    # Warmup JIT
    theta_init = jnp.ones(n_hyperparameters, dtype=dtype)
    _ = value_and_grad_fn(theta_init)

    def objective_with_grad(theta):
        theta_jax = jnp.asarray(theta, dtype=dtype)
        (f_val, x_val), grad_val = value_and_grad_fn(theta_jax)
        return float(f_val), np.asarray(grad_val, dtype=np_dtype), np.asarray(x_val, dtype=np_dtype)

    return objective_pure_jax, objective_with_grad


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

    n_locals = [nt // comm_size] * comm_size
    n_locals[0] += nt % comm_size
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

    return static_data


def _extract_bta_blocks_coregional(
    sparse_matrix,
    n_blocks: int,
    block_size: int,
    n_fixed_effects: int,
    dtype=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract BTA blocks from sparse matrix for coregional model.

    inputs:
    sparse_matrix : scipy sparse matrix
    n_blocks : Number of temporal blocks (nt)
    block_size : Size of each block (n_models * ns)
    n_fixed_effects : Total size of arrow tip (n_models * n_fixed_effects_per_model)
    dtype : JAX dtype for output arrays

    Returns:
    diag_blocks : (n_blocks, block_size, block_size)
    lower_diag_blocks : (n_blocks-1, block_size, block_size)
    arrow_bottom_blocks : (n_blocks, n_fixed_effects, block_size)
    arrow_tip : (n_fixed_effects, n_fixed_effects)
    """
    if dtype is None:
        dtype = get_jax_dtype()
    csc = scipy_sparse.csc_matrix(sparse_matrix)
    total_st = n_blocks * block_size

    diag_blocks = jnp.zeros((n_blocks, block_size, block_size), dtype=dtype)
    lower_diag_blocks = jnp.zeros((n_blocks - 1, block_size, block_size), dtype=dtype)
    arrow_bottom_blocks = jnp.zeros((n_blocks, n_fixed_effects, block_size), dtype=dtype)
    arrow_tip = jnp.zeros((n_fixed_effects, n_fixed_effects), dtype=dtype)

    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        block = csc[start:end, start:end].toarray()
        diag_blocks = diag_blocks.at[i].set(jnp.array(block, dtype=dtype))

    for i in range(n_blocks - 1):
        row_start = (i + 1) * block_size
        row_end = (i + 2) * block_size
        col_start = i * block_size
        col_end = (i + 1) * block_size
        block = csc[row_start:row_end, col_start:col_end].toarray()
        lower_diag_blocks = lower_diag_blocks.at[i].set(jnp.array(block, dtype=dtype))

    if n_fixed_effects > 0:
        for i in range(n_blocks):
            col_start = i * block_size
            col_end = (i + 1) * block_size
            block = csc[total_st:, col_start:col_end].toarray()
            arrow_bottom_blocks = arrow_bottom_blocks.at[i].set(jnp.array(block, dtype=dtype))

        arrow_tip = jnp.array(csc[total_st:, total_st:].toarray(), dtype=dtype)

    return diag_blocks, lower_diag_blocks, arrow_bottom_blocks, arrow_tip


def bta_to_dense_jax(diag_blocks, lower_blocks, arrow_blocks, tip_block):
    """Convert BTA (Block-Tridiagonal-Arrowhead) format to dense matrix.

    inputs:
    diag_blocks : (nt, ns, ns) diagonal blocks
    lower_blocks : (nt-1, ns, ns) lower diagonal blocks
    arrow_blocks : (nt, n_fe, ns) arrow blocks
    tip_block : (n_fe, n_fe) tip block

    Returns:
    dense : (nt*ns + n_fe, nt*ns + n_fe) dense matrix
    """
    nt = diag_blocks.shape[0]
    ns = diag_blocks.shape[1]
    n_fe = tip_block.shape[0] if tip_block.ndim > 0 else 0

    n_total = nt * ns + n_fe
    dense = jnp.zeros((n_total, n_total), dtype=diag_blocks.dtype)

    # Place diagonal blocks
    for t in range(nt):
        row_start = t * ns
        row_end = (t + 1) * ns
        dense = dense.at[row_start:row_end, row_start:row_end].set(diag_blocks[t])

    # Place lower and upper diagonal blocks
    for t in range(nt - 1):
        row_start = (t + 1) * ns
        row_end = (t + 2) * ns
        col_start = t * ns
        col_end = (t + 1) * ns
        # Lower block
        dense = dense.at[row_start:row_end, col_start:col_end].set(lower_blocks[t])
        # Upper block (transpose)
        dense = dense.at[col_start:col_end, row_start:row_end].set(lower_blocks[t].T)

    # Place arrow blocks (last rows/columns except tip)
    if n_fe > 0:
        for t in range(nt):
            col_start = t * ns
            col_end = (t + 1) * ns
            row_start = nt * ns
            # Bottom arrow
            dense = dense.at[row_start:row_start+n_fe, col_start:col_end].set(arrow_blocks[t])
            # Right arrow (transpose)
            dense = dense.at[col_start:col_end, row_start:row_start+n_fe].set(arrow_blocks[t].T)

        # Place tip block
        dense = dense.at[nt*ns:, nt*ns:].set(tip_block)

    return dense


def sigmoid_function(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood):
    yEta = eta - y
    log_likelihood = ( 0.5 * theta_likelihood * len(y) - 0.5 * jnp.exp(theta_likelihood) * (yEta.T @ yEta)
    )
    return log_likelihood


def _evaluate_poisson_likelihood_jax(eta, y, e):
    log_likelihood = jnp.dot(eta, y) - jnp.sum(e * jnp.exp(eta))
    return log_likelihood


def _evaluate_binomial_likelihood_jax(eta, y, n_trials):
    linkEta = sigmoid_function(eta)
    log_likelihood = (
        jnp.dot(y, jnp.log(linkEta + 1e-12))
        + jnp.dot(n_trials - y, jnp.log(1.0 - linkEta + 1e-12))
    )
    return log_likelihood


def _gradient_poisson_likelihood_jax(eta, y, e):
    return y - e * jnp.exp(eta)


def _gradient_binomial_likelihood_jax(eta, y, n_trials):
    linkEta = sigmoid_function(eta)
    return y - n_trials * linkEta


def _hessian_diag_poisson_jax(eta, e):
    return -e * jnp.exp(eta)


def _hessian_diag_binomial_jax(eta, n_trials):
    linkEta = sigmoid_function(eta)
    return -n_trials * linkEta * (1.0 - linkEta)


def _evaluate_log_prior_hyperparameters_jax(theta, prior_configs):
    """Evaluate log prior for all hyperparameters matching DALIA's implementation.

    Supports both Penalized Complexity and Gaussian priors:

    Penalized Complexity priors (by hyperparameter type):
    - r_s (spatial range): log(lambda) - lambda * exp(-theta) - theta
    - r_t (temporal range): log(lambda) + log(0.5) - lambda * exp(-0.5*theta) - 0.5*theta
    - sigma_st/sigma_e: log(lambda) - lambda * exp(theta) + theta
    - prec_o (observation precision): log(lambda) - lambda * exp(theta) + theta

    Gaussian prior:
    - -0.5 * precision * (theta - mean)^2

    inputs:
    theta : Hyperparameters array
    prior_configs : List of dicts with prior type and parameters

    Returns:
    log_prior : Sum of log priors for all hyperparameters
    """
    log_prior = 0.0

    for i, config in enumerate(prior_configs):
        prior_type = config.get('prior_type', 'unknown')
        theta_i = theta[i]

        if prior_type == 'penalized_complexity':
            hp_type = config['hyperparameter_type']
            lambda_theta = config['lambda_theta']

            if hp_type == 'r_s':
                log_prior = log_prior + (
                    jnp.log(lambda_theta)
                    - lambda_theta * jnp.exp(-theta_i)
                    - theta_i
                )
            elif hp_type == 'r_t':
                log_prior = log_prior + (
                    jnp.log(lambda_theta)
                    - lambda_theta * jnp.exp(-0.5 * theta_i)
                    + jnp.log(0.5)
                    - 0.5 * theta_i
                )
            elif hp_type in ('sigma_st', 'sigma_e'):
                log_prior = log_prior + (
                    jnp.log(lambda_theta)
                    - lambda_theta * jnp.exp(theta_i)
                    + theta_i
                )
            elif hp_type == 'prec_o':
                log_prior = log_prior + (
                    jnp.log(lambda_theta)
                    - lambda_theta * jnp.exp(theta_i)
                    + theta_i
                )

        elif prior_type == 'gaussian':
            mean = config['mean']
            precision = config['precision']
            log_prior = log_prior + (
                -0.5 * precision * (theta_i - mean) ** 2
            )

    return log_prior


def _inner_iteration_jax(a, y, Q_prior, grad_likelihood_fn, hess_diag_fn, tol, max_iter, x_initial=None):
    """JAX implementation of inner iteration for non-Gaussian likelihoods.

    Uses jax.lax.fori_loop with fixed iteration count for efficient autodiff.
    Newton-Raphson typically converges in 5-10 iterations for well-conditioned problems.

    inputs:
    a : Design matrix
    y : Observations
    Q_prior : Prior precision matrix
    grad_likelihood_fn : Function computing likelihood gradient
    hess_diag_fn : Function computing diagonal of likelihood Hessian
    tol : Convergence tolerance (not used - fixed iterations for autodiff efficiency)
    max_iter : Maximum iterations (not used - fixed iterations for autodiff efficiency)
    x_initial : Initial values for latent parameters (if None, uses zeros)

    Returns:
    Q_conditional : Conditional precision matrix
    x_star : Optimal latent parameters
    eta : Linear predictor (A @ x_star)
    """
    n_latent = Q_prior.shape[0]
    eta_max = 20.0  # Clip eta to prevent exp overflow

    # Pre-compute transpose once
    a_T = a.T

    # Determine if we need FP32 regularization
    is_fp32 = Q_prior.dtype == jnp.float32
    eps_reg = 1e-4 if is_fp32 else 0.0

    def body_fn(i, state):
        """One Newton-Raphson iteration."""
        x_star, Q_conditional = state

        # Compute eta = A @ x with clipping
        eta = a @ x_star
        eta = jnp.clip(eta, -eta_max, eta_max)

        # Compute Hessian diagonal and conditional precision
        D_diag = hess_diag_fn(eta)
        Q_conditional = Q_prior - (a_T * D_diag) @ a

        # Add diagonal regularization for FP32 to stabilize Cholesky
        Q_conditional = Q_conditional + eps_reg * jnp.eye(n_latent, dtype=Q_prior.dtype)

        # Compute RHS and solve
        gradient_likelihood = grad_likelihood_fn(eta)
        rhs = a_T @ gradient_likelihood - Q_prior @ x_star

        # Solve using Cholesky decomposition
        L = jnp.linalg.cholesky(Q_conditional)
        x_update = jax.scipy.linalg.cho_solve((L, True), rhs)

        x_star = x_star + x_update

        return (x_star, Q_conditional)

    # Initialize state
    if x_initial is None:
        x_star = jnp.zeros(n_latent, dtype=Q_prior.dtype)
    else:
        x_star = x_initial
    Q_conditional = Q_prior.copy()

    # Run fixed 10 iterations - sufficient for Newton convergence, minimal backward pass cost
    x_star, Q_conditional = lax.fori_loop(
        0, 10,
        body_fn,
        (x_star, Q_conditional)
    )

    # Compute final eta with clipping
    eta = a @ x_star
    eta = jnp.clip(eta, -eta_max, eta_max)

    return Q_conditional, x_star, eta


def _objective_gaussian_dense(theta, static_data):
    """Pure JAX objective function for Gaussian likelihood with dense solver.

    This implements the INLA objective:
        f(theta) = -[log p(theta) + log p(y|x,theta) + log p(x|theta) - log p(x|y,theta)]

    inputs:
    theta : Hyperparameters.
    static_data : Static data extracted from model.

    Returns:
    f : Objective function value.
    """
    a = static_data['a']
    y = static_data['y']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    prior_configs = static_data['prior_configs']

    theta_likelihood = theta[-1]
    dtype = a.dtype

    Q_prior = jnp.eye(n_fixed_effects, dtype=dtype) * fixed_effects_precision
    
    eta = jnp.zeros_like(y)
    
    D_diag = -jnp.exp(theta_likelihood) * jnp.ones(len(y), dtype=dtype)
    Q_conditional = Q_prior - a.T @ jnp.diag(D_diag) @ a

    gradient_likelihood = jnp.exp(theta_likelihood) * y
    rhs = a.T @ gradient_likelihood

    x = jnp.linalg.solve(Q_conditional, rhs)

    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    log_likelihood = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior

    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * x.T @ Q_conditional @ x

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _build_spatial_Q_prior_jax(theta_spatial, spatial_matrices, ns):
    """Build spatial Q_prior matrix using JAX.

    This implements the SPDE precision matrix construction for 2D spatial domain.

    inputs:
    theta_spatial : [r_s, sigma_e] hyperparameters
    spatial_matrices : dict with 'c0', 'g1', 'g2' matrices
    ns : number of spatial nodes

    Returns:
    Q_spatial : (ns, ns) spatial precision matrix
    """
    r_s = theta_spatial[0]
    sigma_e = theta_spatial[1]

    c0 = spatial_matrices['c0']
    g1 = spatial_matrices['g1']
    g2 = spatial_matrices['g2']

    # Interpretable to compute transformation (2D spatial domain)
    alpha = 2.0
    dim_spatial_domain = 2.0
    nu_s = alpha - dim_spatial_domain / 2.0  # = 1.0 for 2D

    gamma_s = 0.5 * jnp.log(8.0 * nu_s) - r_s

    # gamma_e computation using scipy.special.gamma values precomputed
    # gamma(1) = 1, gamma(2) = 1, so for nu_s=1, alpha=2:
    # log(gamma(1)) = 0, log(gamma(2)) = 0
    log_gamma_nu_s = jax.scipy.special.gammaln(nu_s)
    log_gamma_alpha = jax.scipy.special.gammaln(alpha)

    gamma_e = 0.5 * (
        log_gamma_nu_s
        - (log_gamma_alpha + 0.5 * dim_spatial_domain * jnp.log(4.0 * jnp.pi) + 2.0 * nu_s * gamma_s + 2.0 * sigma_e)
    )

    exp_gamma_s = jnp.exp(gamma_s)
    exp_gamma_e = jnp.exp(gamma_e)

    # Q = exp(gamma_e)^2 * (exp(gamma_s)^4 * c0 + 2 * exp(gamma_s)^2 * g1 + g2)
    q2s = (
        jnp.power(exp_gamma_s, 4) * c0
        + 2.0 * jnp.power(exp_gamma_s, 2) * g1
        + g2
    )
    Q_spatial = jnp.power(exp_gamma_e, 2) * q2s

    return Q_spatial


def _objective_gaussian_spatial_dense(theta, static_data):
    """Pure JAX objective function for Gaussian likelihood with spatial model (dense solver).

    This implements the INLA objective for spatial + regression models:
        f(theta) = -[log p(theta) + log p(y|x,theta) + log p(x|theta) - log p(x|y,theta)]

    inputs:
    theta : Hyperparameters [r_s, sigma_e, prec_o] or [r_s, prec_o] if sigma_e is fixed
    static_data : Static data extracted from model.

    Returns:
    f : Objective function value.
    x : Latent parameters.
    """
    a = static_data['a']
    y = static_data['y']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    prior_configs = static_data['prior_configs']
    ns = static_data['ns']
    spatial_matrices = static_data['spatial_matrices']
    fe_offset = static_data.get('fe_offset', ns)
    spatial_offset = static_data.get('spatial_offset', 0)
    n_latent = static_data['n_latent_parameters']

    # Parse hyperparameters
    # theta layout: [spatial_params..., prec_o]
    # For spatial: [r_s, sigma_e] or just [r_s] if sigma_e is fixed
    n_theta = len(prior_configs)
    theta_likelihood = theta[-1]

    # Determine if sigma_e is a hyperparameter
    n_spatial_params = n_theta - 1  # last one is prec_o
    if n_spatial_params == 2:
        theta_spatial = theta[:2]
    else:
        # sigma_e is fixed at 0
        theta_spatial = jnp.array([theta[0], 0.0])

    # Build spatial Q_prior
    Q_spatial = _build_spatial_Q_prior_jax(theta_spatial, spatial_matrices, ns)

    # Build full Q_prior (block diagonal: spatial + fixed effects)
    Q_prior = jnp.zeros((n_latent, n_latent), dtype=y.dtype)

    # Place spatial block
    Q_prior = Q_prior.at[spatial_offset:spatial_offset+ns, spatial_offset:spatial_offset+ns].set(Q_spatial)

    # Place fixed effects block
    if n_fixed_effects > 0:
        Q_fe = jnp.eye(n_fixed_effects, dtype=y.dtype) * fixed_effects_precision
        Q_prior = Q_prior.at[fe_offset:fe_offset+n_fixed_effects, fe_offset:fe_offset+n_fixed_effects].set(Q_fe)

    # For Gaussian likelihood, use the same formulation as _objective_gaussian_dense:
    # eta = 0 (evaluate likelihood at zero)
    # Q_conditional = Q_prior + prec_o * A^T A
    eta = jnp.zeros_like(y)

    D_diag = -jnp.exp(theta_likelihood) * jnp.ones(len(y), dtype=y.dtype)
    Q_conditional = Q_prior - (a.T * D_diag) @ a  # = Q_prior + prec_o * A^T A

    # Solve for x
    gradient_likelihood = jnp.exp(theta_likelihood) * y
    rhs = a.T @ gradient_likelihood
    x = jnp.linalg.solve(Q_conditional, rhs)

    # Evaluate log prior hyperparameters
    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    # Evaluate log likelihood at eta=0 (matching existing Gaussian dense formulation)
    log_likelihood = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    # Log prior latent: 0.5 * log|Q_prior|
    # Use slogdet which handles near-singular matrices better
    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior

    # Log conditional: 0.5 * log|Q_conditional| - 0.5 * x^T @ Q_conditional @ x
    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * x.T @ Q_conditional @ x

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_gaussian_st_dense(theta, static_data):
    """Pure JAX objective for Gaussian likelihood with spatio-temporal model (dense solver).

    This handles the case where we have a spatio-temporal submodel with Gaussian likelihood
    using a dense solver.

    inputs:
    theta : Hyperparameters [r_s, r_t, sigma_st, prec_o]
    static_data : Static data containing spatial/temporal matrices and model parameters

    Returns:
    objective : INLA objective value
    x : Latent parameters
    """
    from dalia.core.jax_sparse_helpers import build_spatio_temporal_Q_bta_jax

    a = static_data['a']
    y = static_data['y']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    prior_configs = static_data['prior_configs']
    nt = static_data['nt']
    ns = static_data['ns']
    manifold = static_data['manifold']
    spatial_matrices = static_data['spatial_matrices']
    temporal_matrices = static_data['temporal_matrices']
    fe_offset = static_data.get('fe_offset', nt * ns)
    st_offset = static_data.get('st_offset', 0)
    n_latent = static_data['n_latent_parameters']

    # Parse hyperparameters: [r_s, r_t, sigma_st, prec_o]
    theta_st = theta[:3]
    theta_likelihood = theta[-1]

    # Build spatio-temporal Q_prior in BT format, then convert to dense
    diag_blocks, lower_blocks = build_spatio_temporal_Q_bta_jax(
        theta_st, spatial_matrices, temporal_matrices, manifold
    )
    dummy_arrow = jnp.zeros((nt, 0, ns), dtype=y.dtype)
    dummy_tip = jnp.zeros((0, 0), dtype=y.dtype)
    Q_st = bta_to_dense_jax(diag_blocks, lower_blocks, dummy_arrow, dummy_tip)

    # Build full Q_prior (block diagonal: spatio-temporal + fixed effects)
    Q_prior = jnp.zeros((n_latent, n_latent), dtype=y.dtype)

    # Place ST block
    Q_prior = Q_prior.at[st_offset:st_offset+nt*ns, st_offset:st_offset+nt*ns].set(Q_st)

    # Place fixed effects block
    if n_fixed_effects > 0:
        Q_fe = jnp.eye(n_fixed_effects, dtype=y.dtype) * fixed_effects_precision
        Q_prior = Q_prior.at[fe_offset:fe_offset+n_fixed_effects, fe_offset:fe_offset+n_fixed_effects].set(Q_fe)

    # For Gaussian likelihood, use same formulation as _objective_gaussian_dense
    eta = jnp.zeros_like(y)

    D_diag = -jnp.exp(theta_likelihood) * jnp.ones(len(y), dtype=y.dtype)
    Q_conditional = Q_prior - (a.T * D_diag) @ a  # = Q_prior + prec_o * A^T A

    # Solve for x
    gradient_likelihood = jnp.exp(theta_likelihood) * y
    rhs = a.T @ gradient_likelihood
    x = jnp.linalg.solve(Q_conditional, rhs)

    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)
    log_likelihood = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    # Use slogdet for log determinants
    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior

    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * x.T @ Q_conditional @ x

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_poisson_dense(theta, static_data):
    """Pure JAX objective function for Poisson likelihood with dense solver."""
    a = static_data['a']
    y = static_data['y']
    e = static_data['e']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    tol = static_data['inner_iter_tol']
    max_iter = static_data['inner_iter_max']
    x_initial = static_data.get('x_initial', None)

    dtype = y.dtype
    Q_prior = jnp.eye(n_fixed_effects, dtype=dtype) * fixed_effects_precision

    # Add diagonal regularization for FP32
    if dtype == jnp.float32:
        Q_prior = Q_prior + 1e-6 * jnp.eye(n_fixed_effects, dtype=dtype)

    log_prior_hyperparameters = 0.0

    grad_fn = lambda eta: _gradient_poisson_likelihood_jax(eta, y, e)
    hess_fn = lambda eta: _hessian_diag_poisson_jax(eta, e)

    Q_conditional, x, eta = _inner_iteration_jax(
        a, y, Q_prior, grad_fn, hess_fn, tol, max_iter, x_initial
    )

    log_likelihood = _evaluate_poisson_likelihood_jax(eta, y, e)

    # Q_prior is diagonal, logdet = sum of log of diagonal elements
    logdet_Q_prior = n_fixed_effects * jnp.log(fixed_effects_precision)
    log_prior_latent = 0.5 * logdet_Q_prior - 0.5 * fixed_effects_precision * jnp.dot(x, x)

    # Log conditional using Cholesky with safe log (faster than slogdet)
    L_cond = jnp.linalg.cholesky(Q_conditional)
    eps = jnp.finfo(L_cond.dtype).eps
    logdet_Q_conditional = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.diag(L_cond), eps)))
    log_conditional = 0.5 * logdet_Q_conditional

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_poisson_st_dense(theta, static_data):
    """Pure JAX objective for Poisson likelihood with spatio-temporal model (dense solver).

    This handles the case where we have a spatio-temporal submodel with Poisson likelihood
    using a dense solver. The Q_prior is constructed from the hyperparameters theta.

    inputs:
    theta : Hyperparameters [r_s, r_t, sigma_st]
    static_data : Static data containing spatial/temporal matrices and model parameters

    Returns:
    objective : INLA objective value
    x : Latent parameters
    """
    a = static_data['a']
    y = static_data['y']
    e = static_data['e']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    prior_configs = static_data['prior_configs']
    tol = static_data['inner_iter_tol']
    max_iter = static_data['inner_iter_max']
    nt = static_data['nt']
    ns = static_data['ns']
    spatial_matrices = static_data['spatial_matrices']
    temporal_matrices = static_data['temporal_matrices']
    manifold = static_data['manifold']
    fe_offset = static_data['fe_offset']
    st_offset = static_data['st_offset']
    x_initial = static_data['x_initial']

    # Build Q_st from hyperparameters
    Q_st = build_spatio_temporal_Q_jax(theta, spatial_matrices, temporal_matrices, manifold)

    # Build full Q_prior: block diagonal matching submodel ordering
    total_st_size = nt * ns
    n_latent = total_st_size + n_fixed_effects
    dtype = y.dtype
    Q_prior = jnp.zeros((n_latent, n_latent), dtype=dtype)

    # Place Q_st and Q_fe blocks at their correct offsets
    Q_prior = Q_prior.at[st_offset:st_offset+total_st_size, st_offset:st_offset+total_st_size].set(Q_st)
    Q_prior = Q_prior.at[fe_offset:fe_offset+n_fixed_effects, fe_offset:fe_offset+n_fixed_effects].set(
        jnp.eye(n_fixed_effects, dtype=dtype) * fixed_effects_precision
    )

    # Evaluate prior on hyperparameters
    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    # Inner iteration for Poisson likelihood
    grad_fn = lambda eta: _gradient_poisson_likelihood_jax(eta, y, e)
    hess_fn = lambda eta: _hessian_diag_poisson_jax(eta, e)

    # Add diagonal regularization for FP32 to improve numerical stability
    if Q_prior.dtype == jnp.float32:
        eps_reg = 1e-4
        Q_prior = Q_prior + eps_reg * jnp.eye(Q_prior.shape[0], dtype=Q_prior.dtype)

    Q_conditional, x, eta = _inner_iteration_jax(
        a, y, Q_prior, grad_fn, hess_fn, tol, max_iter, x_initial
    )

    # Poisson log-likelihood
    log_likelihood = _evaluate_poisson_likelihood_jax(eta, y, e)

    # Log prior for latent parameters using correct offsets
    # Use Cholesky for logdet (faster than slogdet for positive definite matrices)
    # Add regularization for FP32 and use safe log
    Q_st_reg = Q_st
    if Q_st.dtype == jnp.float32:
        Q_st_reg = Q_st + 1e-6 * jnp.eye(Q_st.shape[0], dtype=Q_st.dtype)
    L_st = jnp.linalg.cholesky(Q_st_reg)
    eps = jnp.finfo(L_st.dtype).eps
    logdet_Q_st = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.diag(L_st), eps)))
    x_st = x[st_offset:st_offset+total_st_size]
    log_prior_st = 0.5 * logdet_Q_st - 0.5 * jnp.dot(x_st, Q_st @ x_st)

    # Fixed effects prior (Q_fe is diagonal, so logdet is simple)
    x_fe = x[fe_offset:fe_offset+n_fixed_effects]
    logdet_Q_fe = n_fixed_effects * jnp.log(fixed_effects_precision)
    log_prior_fe = 0.5 * logdet_Q_fe - 0.5 * fixed_effects_precision * jnp.dot(x_fe, x_fe)

    log_prior_latent = log_prior_st + log_prior_fe

    # Log conditional using Cholesky with safe log
    L_cond = jnp.linalg.cholesky(Q_conditional)
    logdet_Q_conditional = 2.0 * jnp.sum(jnp.log(jnp.maximum(jnp.diag(L_cond), eps)))
    log_conditional = 0.5 * logdet_Q_conditional

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_binomial_dense(theta, static_data):
    """Pure JAX objective function for Binomial likelihood with dense solver."""
    a = static_data['a']
    y = static_data['y']
    n_trials = static_data['n_trials']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    tol = static_data['inner_iter_tol']
    max_iter = static_data['inner_iter_max']
    x_initial = static_data.get('x_initial', None)

    Q_prior = jnp.eye(n_fixed_effects, dtype=y.dtype) * fixed_effects_precision

    log_prior_hyperparameters = 0.0

    grad_fn = lambda eta: _gradient_binomial_likelihood_jax(eta, y, n_trials)
    hess_fn = lambda eta: _hessian_diag_binomial_jax(eta, n_trials)

    Q_conditional, x, eta = _inner_iteration_jax(
        a, y, Q_prior, grad_fn, hess_fn, tol, max_iter, x_initial
    )

    log_likelihood = _evaluate_binomial_likelihood_jax(eta, y, n_trials)

    # Q_prior is diagonal, logdet = sum of log of diagonal elements
    logdet_Q_prior = n_fixed_effects * jnp.log(fixed_effects_precision)
    log_prior_latent = 0.5 * logdet_Q_prior - 0.5 * fixed_effects_precision * jnp.dot(x, x)

    # Log conditional using Cholesky (faster than slogdet)
    L_cond = jnp.linalg.cholesky(Q_conditional)
    logdet_Q_conditional = 2.0 * jnp.sum(jnp.log(jnp.diag(L_cond)))
    log_conditional = 0.5 * logdet_Q_conditional

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _bt_cholesky_step(carry, i):
    """Single step of block-tridiagonal Cholesky factorization for lax.scan."""
    diag_blocks, lower_blocks, nt = carry

    # Cholesky of current diagonal block
    L_i = jnp.linalg.cholesky(diag_blocks[i])
    diag_blocks = diag_blocks.at[i].set(L_i)

    # Update lower block and next diagonal (only if not last block)
    def update_blocks(args):
        diag_blocks, lower_blocks, L_i, i = args
        # Solve L_i @ X = lower_blocks[i].T -> X = L_i^{-1} @ lower_blocks[i].T
        lower_updated = jnp.linalg.solve(L_i, lower_blocks[i].T).T
        lower_blocks = lower_blocks.at[i].set(lower_updated)
        # Update next diagonal: D_{i+1} -= L_i @ L_i^T
        diag_blocks = diag_blocks.at[i + 1].set(
            diag_blocks[i + 1] - lower_updated @ lower_updated.T
        )
        return diag_blocks, lower_blocks

    def no_update(args):
        diag_blocks, lower_blocks, _, _ = args
        return diag_blocks, lower_blocks

    diag_blocks, lower_blocks = lax.cond(
        i < nt - 1,
        update_blocks,
        no_update,
        (diag_blocks, lower_blocks, L_i, i)
    )

    return (diag_blocks, lower_blocks, nt), None


def _objective_gaussian_sparse(theta, static_data):
    """Pure JAX objective for Gaussian likelihood with sparse serinv solver.

    Uses a ``custom_vjp`` (``fused_core``) that computes analytical
    gradients via selected inversion, avoiding JAX AD through the
    expensive BTA Cholesky scan.  Peak memory in the backward pass is
    ~60 GiB (one copy of L/Sigma factors) instead of ~418 GiB.

    Parameters
    ----------
    theta : jnp.ndarray
        Hyperparameters ``[r_s, r_t, sigma_st, theta_likelihood]``.
    static_data : dict
        Static data from :func:`_extract_static_data`.

    Returns
    -------
    objective : float
        INLA objective value.
    x : jnp.ndarray
        Latent parameters.
    """
    nt = static_data['nt']
    ns = static_data['ns']
    n_fe = static_data['n_fixed_effects']
    fe_prec = static_data['fixed_effects_precision']
    spatial_matrices = static_data['spatial_matrices']
    temporal_matrices = static_data['temporal_matrices']
    manifold = static_data['manifold']
    y = static_data['y']
    a_sparse = static_data['a_sparse']
    prior_configs = static_data['prior_configs']

    ata_diag_rows = static_data['ata_diag_rows']
    ata_diag_cols = static_data['ata_diag_cols']
    ata_diag_vals = static_data['ata_diag_vals']
    ata_lower_rows = static_data['ata_lower_rows']
    ata_lower_cols = static_data['ata_lower_cols']
    ata_lower_vals = static_data['ata_lower_vals']
    ata_arrow_rows = static_data['ata_arrow_rows']
    ata_arrow_cols = static_data['ata_arrow_cols']
    ata_arrow_vals = static_data['ata_arrow_vals']
    ata_tip = static_data['ata_tip']

    dtype = y.dtype
    n_theta_st = 3

    # ---- fused_core: custom_vjp function ----
    # Returns (logdet_st, logdet_cond, quad, x).
    # Forward: factorize, solve, compute logdets + quadratic form.
    # Backward: analytical gradients via selected inversion.

    @jax.custom_vjp
    def fused_core(theta_st, theta_lik):
        lik_prec = jnp.exp(theta_lik)
        sc = precompute_spatial_components(theta_st, spatial_matrices, temporal_matrices, manifold)

        rhs = lik_prec * (a_sparse.T @ y)
        rhs_st = rhs[:nt * ns].reshape(nt, ns)
        rhs_fe = rhs[nt * ns:]

        stored_cs, stored_as, y_st, L_tip, arrow_rhs_acc, logdet_cond = \
            fused_cholesky_fwd_sub(
                sc, nt, ns, n_fe, fe_prec, lik_prec,
                rhs_st, rhs_fe,
                ata_diag_rows, ata_diag_cols, ata_diag_vals,
                ata_lower_rows, ata_lower_cols, ata_lower_vals,
                ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
                ata_tip, dtype,
            )

        x, quad = backward_sub_from_carries(
            stored_cs, stored_as, L_tip,
            y_st, arrow_rhs_acc,
            sc, lik_prec,
            ata_diag_rows, ata_diag_cols, ata_diag_vals,
            ata_lower_rows, ata_lower_cols, ata_lower_vals,
            ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
            nt, ns, n_fe, dtype,
        )

        logdet_st = logdet_Q_st_scan(theta_st, spatial_matrices, temporal_matrices, manifold, nt, ns, dtype)

        return logdet_st, logdet_cond, quad, x

    def fused_core_fwd(theta_st, theta_lik):
        lik_prec = jnp.exp(theta_lik)
        sc = precompute_spatial_components(theta_st, spatial_matrices, temporal_matrices, manifold)

        rhs = lik_prec * (a_sparse.T @ y)
        rhs_st = rhs[:nt * ns].reshape(nt, ns)
        rhs_fe = rhs[nt * ns:]

        stored_cs, stored_as, y_st, L_tip, arrow_rhs_acc, logdet_cond = \
            fused_cholesky_fwd_sub(
                sc, nt, ns, n_fe, fe_prec, lik_prec,
                rhs_st, rhs_fe,
                ata_diag_rows, ata_diag_cols, ata_diag_vals,
                ata_lower_rows, ata_lower_cols, ata_lower_vals,
                ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
                ata_tip, dtype,
            )

        x, quad = backward_sub_from_carries(
            stored_cs, stored_as, L_tip,
            y_st, arrow_rhs_acc,
            sc, lik_prec,
            ata_diag_rows, ata_diag_cols, ata_diag_vals,
            ata_lower_rows, ata_lower_cols, ata_lower_vals,
            ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
            nt, ns, n_fe, dtype,
        )

        logdet_st = logdet_Q_st_scan(theta_st, spatial_matrices, temporal_matrices, manifold, nt, ns, dtype)

        # Residuals: theta + x (~8 MB) + carries (~32 GiB) + L_tip (tiny).
        # Carries are reused in backward for selected inversion (no recomputation).
        residuals = (theta_st, theta_lik, x, stored_cs, stored_as, L_tip)
        return (logdet_st, logdet_cond, quad, x), residuals

    def fused_core_bwd(residuals, g):
        bar_logdet_st, bar_logdet_cond, bar_quad, _bar_x = g
        theta_st_r, theta_lik_r, x_r, stored_cs_r, stored_as_r, L_tip_r = residuals

        lik_prec = jnp.exp(theta_lik_r)

        # --- Phase A: selected inversion using residual carries (no recomputation) ---
        sc = precompute_spatial_components(theta_st_r, spatial_matrices, temporal_matrices, manifold)

        jac_sc = jax.jacfwd(precompute_spatial_components)(
            theta_st_r, spatial_matrices, temporal_matrices, manifold
        )

        grad_cond_st, grad_cond_lik = selected_inversion_grads_from_carries_jax(
            stored_cs_r, stored_as_r, L_tip_r,
            sc, jac_sc,
            nt, ns, n_fe, n_theta_st,
            lik_prec,
            ata_diag_rows, ata_diag_cols, ata_diag_vals,
            ata_lower_rows, ata_lower_cols, ata_lower_vals,
            ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
            ata_tip, dtype,
        )

        rhs = lik_prec * (a_sparse.T @ y)

        grad_quad_st, grad_quad_lik = _compute_grad_quad(
            x_r, sc, jac_sc,
            nt, ns, n_fe, n_theta_st,
            rhs, lik_prec,
            ata_diag_rows, ata_diag_cols, ata_diag_vals,
            ata_lower_rows, ata_lower_cols, ata_lower_vals,
            ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
            ata_tip,
        )

        # --- Phase B: logdet_Q_st gradient via analytical BT inversion ---
        grad_logdet_st = bt_logdet_grad(
            theta_st_r, spatial_matrices, temporal_matrices,
            manifold, nt, ns, n_theta_st, dtype,
        )

        # --- Combine ---
        bar_theta_st = (
            bar_logdet_st * grad_logdet_st
            + bar_logdet_cond * grad_cond_st
            + bar_quad * grad_quad_st
        )
        bar_theta_lik = (
            bar_logdet_cond * grad_cond_lik
            + bar_quad * grad_quad_lik
        )

        return bar_theta_st, bar_theta_lik

    fused_core.defvjp(fused_core_fwd, fused_core_bwd)

    # ---- Assemble objective ----
    theta_st = theta[:-1]
    theta_likelihood = theta[-1]

    logdet_Q_st_val, logdet_Q_cond_val, quad_form, x = fused_core(theta_st, theta_likelihood)

    log_prior_hp = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)
    eta = jnp.zeros_like(y)
    log_lik = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    objective = -(
        log_prior_hp
        + log_lik
        + 0.5 * logdet_Q_st_val
        - 0.5 * logdet_Q_cond_val
        + 0.5 * quad_form
    )

    return objective, x


def _objective_gaussian_coregional_sparse(theta, static_data):
    """Pure JAX objective function for Gaussian CoregionalModel with sparse serinv solver.

    Uses block-tridiagonal-arrowhead structure with coregional block size (n_models * ns).

    inputs:
    theta : Hyperparameters arranged as:
            [model_0_params..., model_1_params..., ..., sigmas..., lambdas...]
    static_data : Static data extracted from CoregionalModel

    Returns:
    objective : INLA objective value
    x : Latent parameters
    """
    n_models = static_data['n_models']
    nt = static_data['nt']
    ns = static_data['ns']
    block_size = static_data['block_size']
    n_fixed_effects_total = static_data['n_fixed_effects_total']
    fixed_effects_precision = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a_sparse = static_data['a_sparse']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']

    # Per-model AtA contributions: shape (n_models, n_blocks, block_size, block_size)
    ata_diag_per_model = static_data['ata_diag_per_model']
    ata_lower_per_model = static_data['ata_lower_per_model']
    ata_arrow_per_model = static_data['ata_arrow_per_model']
    ata_tip_per_model = static_data['ata_tip_per_model']

    # Build Q_prior in BTA format using coregional structure
    q_prior_diag, q_prior_lower = build_coregional_Q_bta_jax(
        theta, n_models, ns, nt, models_data, hyperparameters_idx, theta_keys
    )

    # Compute likelihood precisions for each model
    likelihood_precisions = jnp.zeros(n_models, dtype=y.dtype)
    for i in range(n_models):
        prec_idx = hyperparameters_idx[i + 1] - 1
        likelihood_precisions = likelihood_precisions.at[i].set(jnp.exp(theta[prec_idx]))

    # Compute weighted AtDA contribution: sum_i(prec_i * A_i^T @ A_i)
    # Using einsum to weight per-model contributions by their precisions
    # ata_diag_per_model: (n_models, n_blocks, block_size, block_size)
    # likelihood_precisions: (n_models,)
    weighted_ata_diag = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_diag_per_model)
    weighted_ata_lower = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_lower_per_model)
    weighted_ata_arrow = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_arrow_per_model)
    weighted_ata_tip = jnp.einsum('m,mij->ij', likelihood_precisions, ata_tip_per_model)

    # Build Q_conditional blocks (save for quadratic form computation later)
    q_cond_diag = q_prior_diag + weighted_ata_diag
    q_cond_lower = q_prior_lower + weighted_ata_lower
    q_cond_arrow = weighted_ata_arrow
    q_cond_tip = fixed_effects_precision * jnp.eye(n_fixed_effects_total, dtype=y.dtype) + weighted_ata_tip

    # Prepare blocks for Cholesky
    diag_blocks = q_cond_diag
    lower_diag_blocks = q_cond_lower
    lower_arrow_blocks = q_cond_arrow
    arrow_tip = q_cond_tip

    # Add small diagonal regularization for FP32 to improve numerical stability
    if diag_blocks.dtype == jnp.float32:
        eps_reg = 1e-4
        identity_block = jnp.eye(diag_blocks.shape[-1], dtype=diag_blocks.dtype)
        diag_blocks = diag_blocks + eps_reg * identity_block[None, :, :]
        arrow_tip = arrow_tip + eps_reg * jnp.eye(n_fixed_effects_total, dtype=arrow_tip.dtype)

    # Cholesky factorization in BTA format
    L_diag, L_lower, L_arrow, L_tip = pobtaf_jax_optimized(
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip
    )

    # Log determinant from Cholesky factors
    logdet_Q_conditional = compute_logdet_from_cholesky_bta_jax(L_diag, L_tip)

    # Solve for x using BTA system
    # RHS = A^T @ D @ y where D is diagonal with per-model precisions
    # Build weighted gradient: D @ y
    gradient_likelihood = jnp.zeros_like(y)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
            likelihood_precisions[i] * y[obs_start:obs_end]
        )
    rhs = a_sparse.T @ gradient_likelihood

    x = solve_bta_system_jax(L_diag, L_lower, L_arrow, L_tip, rhs)

    # Evaluate log prior on hyperparameters
    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    # Evaluate likelihood (using eta=0 simplification for Gaussian at mode)
    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        y_i = y[obs_start:obs_end]
        eta_i = eta[obs_start:obs_end]
        prec_idx = hyperparameters_idx[i + 1] - 1
        theta_lik_i = theta[prec_idx]
        log_likelihood += _evaluate_gaussian_likelihood_jax(eta_i, y_i, theta_lik_i)

    # Log prior for latent parameters - compute logdet of Q_prior
    logdet_Q_prior_st = pobtf_logdet_jax(q_prior_diag, q_prior_lower)
    log_prior_latent = 0.5 * logdet_Q_prior_st

    # Quadratic form x^T Q_conditional x using Cholesky factors L
    #
    # MEMORY OPTIMIZATION: Instead of computing x^T @ Q @ x directly (which requires
    # keeping the Q_conditional blocks q_cond_diag, q_cond_lower, q_cond_arrow, q_cond_tip
    # in memory), we use the identity:
    #
    #     x^T @ Q @ x = x^T @ L @ L^T @ x = ||L^T @ x||^2
    #
    # where L is the Cholesky factor we already computed (L_diag, L_lower, L_arrow, L_tip).
    #
    # This saves ~60 GB for gst_large scale problems by allowing the Q_conditional blocks
    # to be freed after the Cholesky factorization.
    #
    # Note: For coregional models, block_size = n_models * ns (spatial nodes per model).
    quad_form = quadratic_form_bta_jax(
        L_diag, L_lower, L_arrow, L_tip, x, nt, block_size
    )

    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quad_form

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _build_coregional_Q_prior_spatial_dense(
    theta: jnp.ndarray,
    n_models: int,
    ns: int,
    models_data: list,
    hyperparameters_idx: list,
    theta_keys: list,
) -> jnp.ndarray:
    """Build coregional Q_prior for spatial-only models as dense matrix.

    For 2-model case:
        Q_11 = (1/sigma_0^2)*Qu_0 + (lambda_01^2/sigma_1^2)*Qu_1
        Q_12 = -(lambda_01/sigma_1^2)*Qu_1
        Q_21 = -(lambda_01/sigma_1^2)*Qu_1
        Q_22 = (1/sigma_1^2)*Qu_1

    For 3-model case, similar pattern with more terms.
    """
    block_size = n_models * ns

    # Build individual Qu matrices for each model
    Qu_list = []
    for i in range(n_models):
        model_data = models_data[i]
        hp_start = hyperparameters_idx[i]
        hp_end = hyperparameters_idx[i + 1] - 1

        theta_model = theta[hp_start:hp_end]

        # For spatial models, theta_model contains [r_s]
        r_s = theta_model[0]
        sigma_e = 0.0  # Fixed at 0 for spatial coregional

        # Build spatial precision matrix
        c0 = model_data['spatial_matrices']['c0']
        g1 = model_data['spatial_matrices']['g1']
        g2 = model_data['spatial_matrices']['g2']

        alpha = 2.0
        dim_spatial_domain = 2.0
        nu_s = alpha - dim_spatial_domain / 2.0
        gamma_s = 0.5 * jnp.log(8.0 * nu_s) - r_s
        log_gamma_nu_s = jax.scipy.special.gammaln(nu_s)
        log_gamma_alpha = jax.scipy.special.gammaln(alpha)
        gamma_e = 0.5 * (log_gamma_nu_s - (
            log_gamma_alpha + 0.5 * dim_spatial_domain * jnp.log(4.0 * jnp.pi)
            + 2.0 * nu_s * gamma_s + 2.0 * sigma_e
        ))

        exp_gamma_s = jnp.exp(gamma_s)
        exp_gamma_e = jnp.exp(gamma_e)

        q2s = jnp.power(exp_gamma_s, 4) * c0 + 2.0 * jnp.power(exp_gamma_s, 2) * g1 + g2
        Qu = jnp.power(exp_gamma_e, 2) * q2s
        Qu_list.append(Qu)

    # Extract sigmas and lambdas from theta
    sigma_idx = theta_keys.index('sigma_0')
    sigmas = []
    for i in range(n_models):
        sigmas.append(jnp.exp(theta[sigma_idx + i]))

    lambda_01_idx = theta_keys.index('lambda_0_1')
    lambda_01 = theta[lambda_01_idx]

    if n_models == 3:
        lambda_02_idx = theta_keys.index('lambda_0_2')
        lambda_12_idx = theta_keys.index('lambda_1_2')
        lambda_02 = theta[lambda_02_idx]
        lambda_12 = theta[lambda_12_idx]

    # Build coregional Q_prior matrix
    Q_prior = jnp.zeros((block_size, block_size), dtype=theta.dtype)

    if n_models == 2:
        sigma_0, sigma_1 = sigmas[0], sigmas[1]

        coef_11_0 = 1.0 / (sigma_0 ** 2)
        coef_11_1 = (lambda_01 ** 2) / (sigma_1 ** 2)
        coef_12 = -lambda_01 / (sigma_1 ** 2)
        coef_22 = 1.0 / (sigma_1 ** 2)

        q11 = coef_11_0 * Qu_list[0] + coef_11_1 * Qu_list[1]
        q12 = coef_12 * Qu_list[1]
        q22 = coef_22 * Qu_list[1]

        Q_prior = Q_prior.at[:ns, :ns].set(q11)
        Q_prior = Q_prior.at[:ns, ns:].set(q12)
        Q_prior = Q_prior.at[ns:, :ns].set(q12)  # Q_21 = Q_12
        Q_prior = Q_prior.at[ns:, ns:].set(q22)

    elif n_models == 3:
        sigma_0, sigma_1, sigma_2 = sigmas[0], sigmas[1], sigmas[2]

        coef_11_0 = 1.0 / (sigma_0 ** 2)
        coef_11_1 = (lambda_01 ** 2) / (sigma_1 ** 2)
        coef_11_2 = (lambda_12 ** 2) / (sigma_2 ** 2)

        coef_21_1 = -lambda_01 / (sigma_1 ** 2)
        coef_21_2 = (lambda_02 * lambda_12) / (sigma_2 ** 2)

        coef_31 = -lambda_12 / (sigma_2 ** 2)

        coef_22_1 = 1.0 / (sigma_1 ** 2)
        coef_22_2 = (lambda_02 ** 2) / (sigma_2 ** 2)

        coef_32 = -lambda_02 / (sigma_2 ** 2)

        coef_33 = 1.0 / (sigma_2 ** 2)

        q11 = coef_11_0 * Qu_list[0] + coef_11_1 * Qu_list[1] + coef_11_2 * Qu_list[2]
        q21 = coef_21_1 * Qu_list[1] + coef_21_2 * Qu_list[2]
        q31 = coef_31 * Qu_list[2]
        q22 = coef_22_1 * Qu_list[1] + coef_22_2 * Qu_list[2]
        q32 = coef_32 * Qu_list[2]
        q33 = coef_33 * Qu_list[2]

        Q_prior = Q_prior.at[:ns, :ns].set(q11)
        Q_prior = Q_prior.at[ns:2*ns, :ns].set(q21)
        Q_prior = Q_prior.at[:ns, ns:2*ns].set(q21)  # Symmetric
        Q_prior = Q_prior.at[2*ns:, :ns].set(q31)
        Q_prior = Q_prior.at[:ns, 2*ns:].set(q31)  # Symmetric
        Q_prior = Q_prior.at[ns:2*ns, ns:2*ns].set(q22)
        Q_prior = Q_prior.at[2*ns:, ns:2*ns].set(q32)
        Q_prior = Q_prior.at[ns:2*ns, 2*ns:].set(q32)  # Symmetric
        Q_prior = Q_prior.at[2*ns:, 2*ns:].set(q33)

    return Q_prior


def _objective_gaussian_coregional_spatial_dense(theta, static_data):
    """Pure JAX objective for Gaussian spatial CoregionalModel with dense solver.

    Uses dense matrix operations for spatial-only coregional models.
    """
    n_models = static_data['n_models']
    ns = static_data['ns']
    n_fixed_effects_total = static_data['n_fixed_effects_total']
    fixed_effects_precision = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a = static_data['a']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']
    n_latent = static_data['n_latent_parameters']

    # Build coregional Q_prior for spatial fields
    Q_prior_spatial = _build_coregional_Q_prior_spatial_dense(
        theta, n_models, ns, models_data, hyperparameters_idx, theta_keys
    )

    # Build full Q_prior (coregional spatial + fixed effects)
    Q_prior = jnp.zeros((n_latent, n_latent), dtype=y.dtype)
    n_spatial = n_models * ns
    Q_prior = Q_prior.at[:n_spatial, :n_spatial].set(Q_prior_spatial)

    if n_fixed_effects_total > 0:
        Q_fe = jnp.eye(n_fixed_effects_total, dtype=y.dtype) * fixed_effects_precision
        Q_prior = Q_prior.at[n_spatial:, n_spatial:].set(Q_fe)

    # Compute likelihood precisions for each model
    likelihood_precisions = []
    for i in range(n_models):
        prec_idx = hyperparameters_idx[i + 1] - 1
        likelihood_precisions.append(jnp.exp(theta[prec_idx]))

    # Build D diagonal (per-observation precision)
    n_obs = len(y)
    D_diag = jnp.zeros(n_obs, dtype=y.dtype)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        D_diag = D_diag.at[obs_start:obs_end].set(-likelihood_precisions[i])

    # Q_conditional = Q_prior - A^T @ D @ A
    Q_conditional = Q_prior - (a.T * D_diag) @ a

    # Compute RHS = A^T @ (prec * y)
    gradient_likelihood = jnp.zeros_like(y)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
            likelihood_precisions[i] * y[obs_start:obs_end]
        )
    rhs = a.T @ gradient_likelihood

    # Solve for x
    x = jnp.linalg.solve(Q_conditional, rhs)

    # Evaluate log prior on hyperparameters
    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    # Evaluate likelihood (using eta=0 simplification)
    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        y_i = y[obs_start:obs_end]
        eta_i = eta[obs_start:obs_end]
        prec_idx = hyperparameters_idx[i + 1] - 1
        theta_lik_i = theta[prec_idx]
        log_likelihood += _evaluate_gaussian_likelihood_jax(eta_i, y_i, theta_lik_i)

    # Log prior for latent parameters
    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior

    # Log conditional
    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    quad_form = x.T @ Q_conditional @ x
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quad_form

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_gaussian_coregional_st_dense(theta, static_data):
    """Pure JAX objective for Gaussian spatio-temporal CoregionalModel with dense solver.

    Builds the coregional Q_prior in BTA format, converts to dense, then uses
    standard dense linear algebra. This is used as a baseline to compare against
    the structure-preserving sparse solver.
    """
    n_models = static_data['n_models']
    nt = static_data['nt']
    ns = static_data['ns']
    block_size = static_data['block_size']
    n_fixed_effects_total = static_data['n_fixed_effects_total']
    fixed_effects_precision = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a = static_data['a']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']
    n_latent = static_data['n_latent_parameters']

    # Build coregional Q_prior in BTA format
    q_prior_diag, q_prior_lower = build_coregional_Q_bta_jax(
        theta, n_models, ns, nt, models_data, hyperparameters_idx, theta_keys
    )

    # Convert block-tridiagonal to dense
    n_st = nt * block_size
    Q_prior_st = jnp.zeros((n_st, n_st), dtype=y.dtype)
    for t in range(nt):
        s = t * block_size
        e = s + block_size
        Q_prior_st = Q_prior_st.at[s:e, s:e].set(q_prior_diag[t])
    for t in range(nt - 1):
        s1 = (t + 1) * block_size
        e1 = s1 + block_size
        s0 = t * block_size
        e0 = s0 + block_size
        Q_prior_st = Q_prior_st.at[s1:e1, s0:e0].set(q_prior_lower[t])
        Q_prior_st = Q_prior_st.at[s0:e0, s1:e1].set(q_prior_lower[t].T)

    # Build full Q_prior (coregional ST + fixed effects)
    Q_prior = jnp.zeros((n_latent, n_latent), dtype=y.dtype)
    Q_prior = Q_prior.at[:n_st, :n_st].set(Q_prior_st)
    if n_fixed_effects_total > 0:
        Q_fe = jnp.eye(n_fixed_effects_total, dtype=y.dtype) * fixed_effects_precision
        Q_prior = Q_prior.at[n_st:, n_st:].set(Q_fe)

    # Compute likelihood precisions for each model
    likelihood_precisions = []
    for i in range(n_models):
        prec_idx = hyperparameters_idx[i + 1] - 1
        likelihood_precisions.append(jnp.exp(theta[prec_idx]))

    # Build D diagonal (per-observation precision)
    n_obs = len(y)
    D_diag = jnp.zeros(n_obs, dtype=y.dtype)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        D_diag = D_diag.at[obs_start:obs_end].set(-likelihood_precisions[i])

    Q_conditional = Q_prior - (a.T * D_diag) @ a

    # Compute RHS = A^T @ (prec * y)
    gradient_likelihood = jnp.zeros_like(y)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
            likelihood_precisions[i] * y[obs_start:obs_end]
        )
    rhs = a.T @ gradient_likelihood

    x = jnp.linalg.solve(Q_conditional, rhs)

    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        y_i = y[obs_start:obs_end]
        eta_i = eta[obs_start:obs_end]
        prec_idx = hyperparameters_idx[i + 1] - 1
        theta_lik_i = theta[prec_idx]
        log_likelihood += _evaluate_gaussian_likelihood_jax(eta_i, y_i, theta_lik_i)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior

    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    quad_form = x.T @ Q_conditional @ x
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quad_form

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_gaussian_coregional_sparse_fused(theta, static_data):
    """Memory-efficient JAX objective for coregional Gaussian with custom_vjp.

    Uses carry-based Cholesky (no L storage), per-model sparse COO AtA,
    and on-the-fly super-block reconstruction to fit within GPU memory.

    Parameters
    ----------
    theta : jnp.ndarray
        Full hyperparameter vector.
    static_data : dict
        Static data from :func:`_extract_static_data_coregional`.

    Returns
    -------
    objective : scalar
    x : jnp.ndarray
    """
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

    # Per-model sparse COO data
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

    manifolds = [md.get('manifold', 'plane') for md in models_data]
    dtype = y.dtype

    # Count hyperparameter groups
    n_theta_st_per_model = 3
    n_theta_lik_per_model = 1

    # Identify coregional parameter indices
    sigma_idx = theta_keys.index('sigma_0')
    n_sigmas = n_models
    lambda_keys = [k for k in theta_keys if k.startswith('lambda_')]
    n_lambdas = len(lambda_keys)
    n_coreg_params = n_sigmas + n_lambdas

    @jax.custom_vjp
    def fused_core_coregional(theta_full):
        # Split theta
        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_full[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_full, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        # Build RHS
        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood
        rhs_st = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]

        stored_cs, stored_as, y_st, L_tip, arrow_rhs_acc, logdet_cond = \
            fused_cholesky_fwd_sub_coregional(
                sc_list, coreg_w, n_models, nt, ns, n_fe, fe_prec,
                likelihood_precs,
                rhs_st, rhs_fe,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype)

        x, quad = backward_sub_from_carries_coregional(
            stored_cs, stored_as, L_tip,
            y_st, arrow_rhs_acc,
            sc_list, coreg_w, likelihood_precs,
            per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
            per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
            per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
            per_model_offsets,
            n_models, nt, ns, n_fe, dtype)

        # Pad sc_list subdiags for logdet scan
        sc_padded = []
        for m in range(n_models):
            sc_m = sc_list[m]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        logdet_prior = logdet_Q_prior_coregional_scan(
            sc_padded, coreg_w, n_models, ns, nt, dtype)

        return logdet_prior, logdet_cond, quad, x

    def fused_core_fwd(theta_full):
        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_full[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_full, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood
        rhs_st = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]

        stored_cs, stored_as, y_st, L_tip, arrow_rhs_acc, logdet_cond = \
            fused_cholesky_fwd_sub_coregional(
                sc_list, coreg_w, n_models, nt, ns, n_fe, fe_prec,
                likelihood_precs,
                rhs_st, rhs_fe,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype)

        x, quad = backward_sub_from_carries_coregional(
            stored_cs, stored_as, L_tip,
            y_st, arrow_rhs_acc,
            sc_list, coreg_w, likelihood_precs,
            per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
            per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
            per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
            per_model_offsets,
            n_models, nt, ns, n_fe, dtype)

        sc_padded = []
        for m in range(n_models):
            sc_m = sc_list[m]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        logdet_prior = logdet_Q_prior_coregional_scan(
            sc_padded, coreg_w, n_models, ns, nt, dtype)

        residuals = (theta_full, x, stored_cs, stored_as, L_tip)
        return (logdet_prior, logdet_cond, quad, x), residuals

    def fused_core_bwd(residuals, g):
        bar_logdet_prior, bar_logdet_cond, bar_quad, _bar_x = g
        theta_r, x_r, stored_cs_r, stored_as_r, L_tip_r = residuals

        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_r[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_r, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        # Jacobians of spatial components per model
        jac_sc_list = []
        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            theta_m = theta_r[hp_start:hp_end]
            if theta_m.shape[0] == 2:
                theta_m = jnp.concatenate([theta_m, jnp.array([0.0])])
            jac_m = jax.jacfwd(precompute_spatial_components)(
                theta_m, models_data[m]['spatial_matrices'],
                models_data[m]['temporal_matrices'], manifolds[m])
            jac_sc_list.append(jac_m)

        # Jacobian of coreg_w w.r.t. coregional params (sigmas + lambdas)
        def _coreg_w_from_params(coreg_params):
            """Extract coreg_w from just the coregional parameters."""
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

        # Build coreg_params vector
        coreg_params = jnp.zeros(n_coreg_params, dtype=dtype)
        for m in range(n_models):
            coreg_params = coreg_params.at[m].set(theta_r[sigma_idx + m])
        for li, lk in enumerate(lambda_keys):
            coreg_params = coreg_params.at[n_sigmas + li].set(
                theta_r[theta_keys.index(lk)])

        jac_coreg_w = jax.jacfwd(_coreg_w_from_params)(coreg_params)

        # Pad sc_list subdiags
        sc_padded = []
        for m in range(n_models):
            sc_m = sc_list[m]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        # --- Phase A: logdet_cond gradient via selected inversion ---
        grad_cond_st, grad_cond_lik, grad_cond_coreg = \
            selected_inversion_grads_from_carries_coregional(
                stored_cs_r, stored_as_r, L_tip_r,
                sc_padded, jac_sc_list, coreg_w, jac_coreg_w,
                n_models, nt, ns, n_fe,
                likelihood_precs,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype)

        # --- Phase B: quadratic form gradient ---
        # Rebuild rhs
        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood

        grad_quad_st, grad_quad_lik_placeholder, grad_quad_coreg = \
            _compute_grad_quad_coregional(
                x_r, sc_list, jac_sc_list, coreg_w, jac_coreg_w,
                n_models, nt, ns, n_fe,
                rhs, likelihood_precs,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets)

        # Compute per-model likelihood quad gradient properly
        x_st = x_r[:nt * block_size].reshape(nt, block_size)
        x_fe = x_r[nt * block_size:]
        grad_quad_lik = jnp.zeros(n_models, dtype=dtype)

        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            prec_m = likelihood_precs[m]

            # x^T AtA_m x
            xAtAx_d = jnp.array(0.0, dtype=dtype)
            for t in range(nt):
                xAtAx_d = xAtAx_d + jnp.sum(
                    x_st[t, m_off + per_model_ata_diag_rows[m][t]]
                    * per_model_ata_diag_vals[m][t]
                    * x_st[t, m_off + per_model_ata_diag_cols[m][t]])

            xAtAx_l = jnp.array(0.0, dtype=dtype)
            for t in range(nt - 1):
                xAtAx_l = xAtAx_l + jnp.sum(
                    x_st[t + 1, m_off + per_model_ata_lower_rows[m][t]]
                    * per_model_ata_lower_vals[m][t]
                    * x_st[t, m_off + per_model_ata_lower_cols[m][t]])

            xAtAx_a = jnp.array(0.0, dtype=dtype)
            for t in range(nt):
                xAtAx_a = xAtAx_a + jnp.sum(
                    x_fe[per_model_ata_arrow_rows[m][t]]
                    * per_model_ata_arrow_vals[m][t]
                    * x_st[t, m_off + per_model_ata_arrow_cols[m][t]])

            xAtAx_tip = x_fe @ per_model_ata_tip[m] @ x_fe
            xAtAx_m = xAtAx_d + 2.0 * xAtAx_l + 2.0 * xAtAx_a + xAtAx_tip

            # x^T A_m^T y_m: reconstruct from sparse COO structure
            # Actually use: d(quad)/d(theta_lik_m) = 2*x^T*rhs_m - prec_m*x^T*AtA_m*x
            # where rhs_m = prec_m * A_m^T y_m
            # Reconstruct rhs_m from AtA structure: rhs_m is the contribution from
            # model m alone. We can get x^T A_m^T y_m from:
            # Note that x^T rhs = sum_m x^T rhs_m, and for the gradient we need:
            # d(quad)/d(theta_lik_m) = 2*prec_m*(x^T A_m^T y_m) - prec_m*xAtAx_m
            # = prec_m * (2*(x^T rhs_m / prec_m) - xAtAx_m)
            # Hmm, simpler: rhs_m = prec_m * A_m^T y_m, so x^T A_m^T y_m = x^T rhs_m / prec_m
            # And x^T rhs_m can be computed: build rhs_m
            rhs_m = jnp.zeros(nt * block_size + n_fe, dtype=dtype)
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            y_m_weighted = jnp.zeros_like(y)
            y_m_weighted = y_m_weighted.at[obs_start:obs_end].set(y[obs_start:obs_end])
            rhs_m = a_sparse.T @ y_m_weighted

            xTAmy = jnp.dot(x_r, rhs_m)

            grad_quad_lik = grad_quad_lik.at[m].set(
                prec_m * (2.0 * xTAmy - xAtAx_m))

        # --- Phase C: logdet_prior gradient ---
        jac_sc_list_padded = []
        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            theta_m = theta_r[hp_start:hp_end]
            if theta_m.shape[0] == 2:
                theta_m = jnp.concatenate([theta_m, jnp.array([0.0])])
            jac_m = jax.jacfwd(precompute_spatial_components)(
                theta_m, models_data[m]['spatial_matrices'],
                models_data[m]['temporal_matrices'], manifolds[m])
            # Pad the Jacobian subdiags
            jac_padded = {}
            for key, val in jac_m.items():
                if key.endswith('_subdiag'):
                    jac_padded[key] = jnp.concatenate([val, jnp.zeros((1,) + val.shape[1:], dtype=dtype)], axis=0)
                else:
                    jac_padded[key] = val
            jac_sc_list_padded.append(jac_padded)

        grad_prior_st, grad_prior_coreg = logdet_Q_prior_coregional_grad(
            sc_padded, jac_sc_list_padded, coreg_w, jac_coreg_w,
            n_models, ns, nt, dtype)

        # --- Combine into full gradient ---
        n_theta = theta_r.shape[0]
        bar_theta = jnp.zeros(n_theta, dtype=dtype)

        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            n_st_m = hp_end - hp_start

            grad_st_m = (
                bar_logdet_prior * grad_prior_st[m][:n_st_m]
                + bar_logdet_cond * grad_cond_st[m][:n_st_m]
                + bar_quad * grad_quad_st[m][:n_st_m]
            )
            bar_theta = bar_theta.at[hp_start:hp_end].set(grad_st_m)

            # Likelihood param
            prec_idx = hyperparameters_idx[m + 1] - 1
            grad_lik_m = (
                bar_logdet_cond * grad_cond_lik[m]
                + bar_quad * grad_quad_lik[m]
            )
            bar_theta = bar_theta.at[prec_idx].set(grad_lik_m)

        # Coregional params
        grad_coreg_total = (
            bar_logdet_prior * grad_prior_coreg
            + bar_logdet_cond * grad_cond_coreg
            + bar_quad * grad_quad_coreg
        )
        for m in range(n_models):
            bar_theta = bar_theta.at[sigma_idx + m].add(grad_coreg_total[m])
        for li, lk in enumerate(lambda_keys):
            bar_theta = bar_theta.at[theta_keys.index(lk)].add(
                grad_coreg_total[n_sigmas + li])

        return (bar_theta,)

    fused_core_coregional.defvjp(fused_core_fwd, fused_core_bwd)

    # --- Assemble objective ---
    logdet_prior, logdet_cond, quad_form, x = fused_core_coregional(theta)

    log_prior_hp = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for m in range(n_models):
        obs_start = n_observations_idx[m]
        obs_end = n_observations_idx[m + 1]
        prec_idx = hyperparameters_idx[m + 1] - 1
        log_likelihood += _evaluate_gaussian_likelihood_jax(
            eta[obs_start:obs_end], y[obs_start:obs_end], theta[prec_idx])

    objective = -(
        log_prior_hp
        + log_likelihood
        + 0.5 * logdet_prior
        - 0.5 * logdet_cond
        + 0.5 * quad_form
    )

    return objective, x


def create_pure_jax_objective_coregional(dalia_instance, dtype=None) -> Tuple[Callable, Callable]:
    """Create pure JAX objective function for CoregionalModel with automatic differentiation.

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance with CoregionalModel.
    dtype : jnp.dtype, optional
        JAX dtype to use. If None, uses the configured dtype from get_jax_dtype().

    Returns
    -------
    objective_func : Callable
        Pure JAX objective function.
    objective_with_grad : Callable
        Function returning both forward value and gradient.
    """
    if dtype is None:
        dtype = get_jax_dtype()
    np_dtype = np.float64 if dtype == jnp.float64 else np.float32
    static_data = _extract_static_data_coregional(dalia_instance, dtype=dtype)
    n_hyperparameters = dalia_instance.model.n_hyperparameters

    # Verify all likelihoods are Gaussian
    for model_data in static_data['models_data']:
        if model_data['likelihood_type'] != 'gaussian':
            raise NotImplementedError(
                f"JAX autodiff for CoregionalModel only supports Gaussian likelihoods. "
                f"Found: {model_data['likelihood_type']}"
            )

    use_sparse = static_data.get('use_sparse_solver', True)
    is_spatial_only = static_data.get('is_spatial_only', False)
    use_fused = use_sparse and static_data.get('use_fused', True)

    def objective_pure_jax(theta):
        if use_fused:
            return _objective_gaussian_coregional_sparse_fused(theta, static_data)
        elif use_sparse:
            return _objective_gaussian_coregional_sparse(theta, static_data)
        elif is_spatial_only:
            return _objective_gaussian_coregional_spatial_dense(theta, static_data)
        else:
            return _objective_gaussian_coregional_st_dense(theta, static_data)

    value_and_grad_fn = jax.value_and_grad(objective_pure_jax, has_aux=True)

    objective_pure_jax = jax.jit(objective_pure_jax)
    value_and_grad_fn = jax.jit(value_and_grad_fn)

    # Warmup JIT compilation
    theta_init = jnp.ones(n_hyperparameters, dtype=dtype)
    _ = value_and_grad_fn(theta_init)

    def objective_with_grad(theta):
        theta_jax = jnp.asarray(theta, dtype=dtype)
        (f_val, x_val), grad_val = value_and_grad_fn(theta_jax)
        return float(f_val), np.asarray(grad_val, dtype=np_dtype), np.asarray(x_val, dtype=np_dtype)

    return objective_pure_jax, objective_with_grad


def create_pure_jax_objective_distributed_coregional(dalia_instance, comm, dtype=None) -> Tuple[Callable, Callable]:
    """Create distributed JAX objective with analytical gradients for CoregionalModel.

    Uses pipeline communication for the forward/backward passes across ranks.
    Each rank stores only its local Schur complement carries, solving the
    OOM problem for large coregional models.

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

    manifolds = [md.get('manifold', 'plane') for md in models_data]
    n_hyperparameters = dalia_instance.model.n_hyperparameters

    sigma_idx = theta_keys.index('sigma_0')
    n_sigmas = n_models
    lambda_keys = [k for k in theta_keys if k.startswith('lambda_')]
    n_lambdas = len(lambda_keys)
    n_coreg_params = n_sigmas + n_lambdas

    @jax.custom_vjp
    def fused_core_dist_coregional(theta_full):
        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_full[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_full, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood
        rhs_st_global = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]
        rhs_st_local = rhs_st_global[start_idx:start_idx + n_local]

        stored_cs, stored_as, y_st_local, L_tip, arrow_rhs_acc, logdet_cond = \
            pipeline_fused_cholesky_fwd_sub_coregional(
                sc_list, coreg_w, n_models, nt, ns, n_fe, fe_prec,
                likelihood_precs,
                rhs_st_local, rhs_fe,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype,
                rank, comm_size, n_local, start_idx, comm)

        x_st_global, x_fe, quad = pipeline_backward_sub_from_carries_coregional(
            stored_cs, stored_as, L_tip,
            y_st_local, arrow_rhs_acc,
            sc_list, coreg_w, likelihood_precs,
            per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
            per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
            per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
            per_model_offsets,
            n_models, nt, ns, n_fe, dtype,
            rank, comm_size, n_local, start_idx, comm)

        x = jnp.concatenate([x_st_global.reshape(-1), x_fe])

        sc_padded = []
        for m_i in range(n_models):
            sc_m = sc_list[m_i]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        logdet_prior = pipeline_logdet_Q_prior_coregional_scan(
            sc_padded, coreg_w, n_models, ns, nt, dtype,
            rank, comm_size, n_local, start_idx, comm)

        return logdet_prior, logdet_cond, quad, x

    def fused_core_dist_fwd(theta_full):
        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_full[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_full, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood
        rhs_st_global = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]
        rhs_st_local = rhs_st_global[start_idx:start_idx + n_local]

        stored_cs, stored_as, y_st_local, L_tip, arrow_rhs_acc, logdet_cond = \
            pipeline_fused_cholesky_fwd_sub_coregional(
                sc_list, coreg_w, n_models, nt, ns, n_fe, fe_prec,
                likelihood_precs,
                rhs_st_local, rhs_fe,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype,
                rank, comm_size, n_local, start_idx, comm)

        x_st_global, x_fe, quad = pipeline_backward_sub_from_carries_coregional(
            stored_cs, stored_as, L_tip,
            y_st_local, arrow_rhs_acc,
            sc_list, coreg_w, likelihood_precs,
            per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
            per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
            per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
            per_model_offsets,
            n_models, nt, ns, n_fe, dtype,
            rank, comm_size, n_local, start_idx, comm)

        x = jnp.concatenate([x_st_global.reshape(-1), x_fe])

        sc_padded = []
        for m_i in range(n_models):
            sc_m = sc_list[m_i]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        logdet_prior = pipeline_logdet_Q_prior_coregional_scan(
            sc_padded, coreg_w, n_models, ns, nt, dtype,
            rank, comm_size, n_local, start_idx, comm)

        residuals = (theta_full, x, stored_cs, stored_as, L_tip)
        return (logdet_prior, logdet_cond, quad, x), residuals

    def fused_core_dist_bwd(residuals, g):
        bar_logdet_prior, bar_logdet_cond, bar_quad, _bar_x = g
        theta_r, x_r, stored_cs_r, stored_as_r, L_tip_r = residuals

        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_r[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_r, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        jac_sc_list = []
        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            theta_m = theta_r[hp_start:hp_end]
            if theta_m.shape[0] == 2:
                theta_m = jnp.concatenate([theta_m, jnp.array([0.0])])
            jac_m = jax.jacfwd(precompute_spatial_components)(
                theta_m, models_data[m]['spatial_matrices'],
                models_data[m]['temporal_matrices'], manifolds[m])
            jac_sc_list.append(jac_m)

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

        coreg_params = jnp.zeros(n_coreg_params, dtype=dtype)
        for m in range(n_models):
            coreg_params = coreg_params.at[m].set(theta_r[sigma_idx + m])
        for li, lk in enumerate(lambda_keys):
            coreg_params = coreg_params.at[n_sigmas + li].set(
                theta_r[theta_keys.index(lk)])

        jac_coreg_w = jax.jacfwd(_coreg_w_from_params)(coreg_params)

        sc_padded = []
        for m in range(n_models):
            sc_m = sc_list[m]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        # Selected inversion gradients (distributed)
        grad_cond_st, grad_cond_lik, grad_cond_coreg = \
            pipeline_selected_inversion_grads_from_carries_coregional(
                stored_cs_r, stored_as_r, L_tip_r,
                sc_padded, jac_sc_list, coreg_w, jac_coreg_w,
                n_models, nt, ns, n_fe,
                likelihood_precs,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype,
                rank, comm_size, n_local, start_idx, comm)

        # Quadratic form gradient (distributed)
        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood

        grad_quad_st, grad_quad_lik, grad_quad_coreg = \
            pipeline_compute_grad_quad_coregional(
                x_r, sc_list, jac_sc_list, coreg_w, jac_coreg_w,
                n_models, nt, ns, n_fe,
                rhs, likelihood_precs,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                a_sparse, y, n_observations_idx,
                rank, comm_size, n_local, start_idx, comm)

        # logdet Q_prior gradient (distributed)
        jac_sc_list_padded = []
        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            theta_m = theta_r[hp_start:hp_end]
            if theta_m.shape[0] == 2:
                theta_m = jnp.concatenate([theta_m, jnp.array([0.0])])
            jac_m = jax.jacfwd(precompute_spatial_components)(
                theta_m, models_data[m]['spatial_matrices'],
                models_data[m]['temporal_matrices'], manifolds[m])
            jac_padded = {}
            for key, val in jac_m.items():
                if key.endswith('_subdiag'):
                    jac_padded[key] = jnp.concatenate(
                        [val, jnp.zeros((1,) + val.shape[1:], dtype=dtype)], axis=0)
                else:
                    jac_padded[key] = val
            jac_sc_list_padded.append(jac_padded)

        grad_prior_st, grad_prior_coreg = pipeline_logdet_Q_prior_coregional_grad(
            sc_padded, jac_sc_list_padded, coreg_w, jac_coreg_w,
            n_models, ns, nt, dtype,
            rank, comm_size, n_local, start_idx, comm)

        # Combine into full gradient
        n_theta = theta_r.shape[0]
        bar_theta = jnp.zeros(n_theta, dtype=dtype)

        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            n_st_m = hp_end - hp_start

            grad_st_m = (
                bar_logdet_prior * grad_prior_st[m][:n_st_m]
                + bar_logdet_cond * grad_cond_st[m][:n_st_m]
                + bar_quad * grad_quad_st[m][:n_st_m]
            )
            bar_theta = bar_theta.at[hp_start:hp_end].set(grad_st_m)

            prec_idx = hyperparameters_idx[m + 1] - 1
            grad_lik_m = (
                bar_logdet_cond * grad_cond_lik[m]
                + bar_quad * grad_quad_lik[m]
            )
            bar_theta = bar_theta.at[prec_idx].set(grad_lik_m)

        grad_coreg_total = (
            bar_logdet_prior * grad_prior_coreg
            + bar_logdet_cond * grad_cond_coreg
            + bar_quad * grad_quad_coreg
        )
        for m in range(n_models):
            bar_theta = bar_theta.at[sigma_idx + m].add(grad_coreg_total[m])
        for li, lk in enumerate(lambda_keys):
            bar_theta = bar_theta.at[theta_keys.index(lk)].add(
                grad_coreg_total[n_sigmas + li])

        return (bar_theta,)

    fused_core_dist_coregional.defvjp(fused_core_dist_fwd, fused_core_dist_bwd)

    def objective_pure_jax_dist(theta):
        logdet_prior, logdet_cond, quad_form, x = fused_core_dist_coregional(theta)

        log_prior_hp = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

        eta = jnp.zeros_like(y)
        log_likelihood = 0.0
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            prec_idx = hyperparameters_idx[m + 1] - 1
            log_likelihood += _evaluate_gaussian_likelihood_jax(
                eta[obs_start:obs_end], y[obs_start:obs_end], theta[prec_idx])

        objective = -(
            log_prior_hp
            + log_likelihood
            + 0.5 * logdet_prior
            - 0.5 * logdet_cond
            + 0.5 * quad_form
        )

        return objective, x

    value_and_grad_fn = jax.value_and_grad(objective_pure_jax_dist, has_aux=True)

    objective_pure_jax_dist = jax.jit(objective_pure_jax_dist)
    value_and_grad_fn = jax.jit(value_and_grad_fn)

    theta_init = jnp.ones(n_hyperparameters, dtype=dtype)
    _ = value_and_grad_fn(theta_init)

    def objective_with_grad(theta):
        theta_jax = jnp.asarray(theta, dtype=dtype)
        (f_val, x_val), grad_val = value_and_grad_fn(theta_jax)
        return float(f_val), np.asarray(grad_val, dtype=np_dtype), np.asarray(x_val, dtype=np_dtype)

    return objective_pure_jax_dist, objective_with_grad


def create_pure_jax_objective_distributed_coregional_splitjit(
    dalia_instance, comm, dtype=None
) -> Tuple[Callable, Callable]:
    """Create distributed JAX objective for CoregionalModel using split JIT.

    Unlike ``create_pure_jax_objective_distributed_coregional`` which uses
    ``custom_vjp`` (all gradient computations in a single XLA program), this
    version compiles forward and each gradient component as **separate** JIT
    functions.  XLA can then free memory between stages, reducing peak GPU
    memory from ~126 GiB to ~35 GiB for AP1 with P=4.

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

    manifolds = [md.get('manifold', 'plane') for md in models_data]
    n_hyperparameters = dalia_instance.model.n_hyperparameters

    sigma_idx = theta_keys.index('sigma_0')
    n_sigmas = n_models
    lambda_keys = [k for k in theta_keys if k.startswith('lambda_')]
    n_lambdas = len(lambda_keys)
    n_coreg_params = n_sigmas + n_lambdas
    lambda_indices = [theta_keys.index(lk) for lk in lambda_keys]

    # ---- Shared helpers (closed over static_data) ----

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

    # ---- Stage 1a: Cholesky + forward sub ----
    # NOT @jax.jit: uses Python-level loop with per-block JIT to avoid XLA
    # keeping all per-iteration intermediates alive (121 GiB for AP1).

    # Pre-compute static AtA padding (closed over by per-block JIT)
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

        # Reconstruct sc_list from stacked arrays
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

        # Reconstruct diagonal block
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

        # Reconstruct lower block
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

        # Arrow block
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

        # Schur complement updates
        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(global_i < nt - 1,
                                   new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(global_i < nt - 1,
                                    new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        # Forward substitution
        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)

        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(global_i < nt - 1,
                                     new_prev_lower_y, jnp.zeros_like(new_prev_lower_y))
        new_arrow_rhs_acc = arrow_rhs_acc_blk - L_arrow_i @ y_i

        return (new_cond_schur, new_arrow_tip_acc, new_arrow_schur,
                logdet_cond, new_prev_lower_y, new_arrow_rhs_acc,
                cond_schur, arrow_schur, y_i)

    @jax.jit
    def _chol_recv_carries(cs, at, a_s, ld, ply, ara):
        cs = mpi4jax.recv(cs, source=rank - 1, tag=0, comm=comm)
        at = mpi4jax.recv(at, source=rank - 1, tag=1, comm=comm)
        a_s = mpi4jax.recv(a_s, source=rank - 1, tag=2, comm=comm)
        ld = mpi4jax.recv(ld, source=rank - 1, tag=3, comm=comm)
        ply = mpi4jax.recv(ply, source=rank - 1, tag=4, comm=comm)
        ara = mpi4jax.recv(ara, source=rank - 1, tag=5, comm=comm)
        return cs, at, a_s, ld, ply, ara

    @jax.jit
    def _chol_send_carries(cs, at, a_s, ld, ply, ara):
        mpi4jax.send(cs, dest=rank + 1, tag=0, comm=comm)
        mpi4jax.send(at, dest=rank + 1, tag=1, comm=comm)
        mpi4jax.send(a_s, dest=rank + 1, tag=2, comm=comm)
        mpi4jax.send(ld, dest=rank + 1, tag=3, comm=comm)
        mpi4jax.send(ply, dest=rank + 1, tag=4, comm=comm)
        mpi4jax.send(ara, dest=rank + 1, tag=5, comm=comm)

    @jax.jit
    def _chol_finalize(logdet_local, arrow_tip_final, arrow_rhs_final):
        logdet_global = mpi4jax.bcast(logdet_local, root=comm_size - 1, comm=comm)
        L_tip = _jax_cholesky(arrow_tip_final)
        tip_diag = jnp.diag(L_tip)
        safe_tip = jnp.maximum(tip_diag, _chol_eps)
        logdet_tip = 2.0 * jnp.sum(jnp.log(safe_tip))
        L_tip = mpi4jax.bcast(L_tip, root=comm_size - 1, comm=comm)
        arr_global = mpi4jax.bcast(arrow_rhs_final, root=comm_size - 1, comm=comm)
        logdet_tip = mpi4jax.bcast(logdet_tip, root=comm_size - 1, comm=comm)
        return logdet_global + logdet_tip, L_tip, arr_global

    def _chol_fn(theta):
        likelihood_precs = _theta_to_likelihood_precs(theta)
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        sc_padded = _pad_subdiags(sc_list)

        rhs = _build_rhs(theta)
        rhs_st_global = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]
        rhs_st_local = rhs_st_global[start_idx:start_idx + n_local]

        # Stack sc_list into arrays for JIT arguments
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

        # Initialize carry
        cond_schur = jnp.zeros((block_size, block_size), dtype=dtype)
        arrow_tip_acc = fe_prec * _chol_eye_nfe + _chol_eps_reg * _chol_eye_nfe
        for m in range(n_models):
            arrow_tip_acc = arrow_tip_acc + likelihood_precs[m] * per_model_ata_tip[m]
        arrow_schur = jnp.zeros((n_fe, block_size), dtype=dtype)
        logdet_cond = jnp.array(0.0, dtype=dtype)
        prev_lower_y = jnp.zeros(block_size, dtype=dtype)
        arrow_rhs_acc = rhs_fe

        # MPI recv carries from previous rank
        if rank > 0:
            cond_schur, arrow_tip_acc, arrow_schur, logdet_cond, \
                prev_lower_y, arrow_rhs_acc = _chol_recv_carries(
                    cond_schur, arrow_tip_acc, arrow_schur,
                    logdet_cond, prev_lower_y, arrow_rhs_acc)

        # Python loop over local blocks — each block is a separate JIT call.
        # Accumulate results in CPU (host) memory to avoid GPU OOM when
        # n_local blocks of (block_size, block_size) don't fit simultaneously.
        stored_cs_host = np.empty((n_local, block_size, block_size), dtype=np_dtype)
        stored_as_host = np.empty((n_local, n_fe, block_size), dtype=np_dtype)
        y_st_host = np.empty((n_local, block_size), dtype=np_dtype)
        sc_args = (q1s_all, q2s_all, q3s_all, scale_all, exp_gt_all,
                   m0_diag_all, m1_diag_all, m2_diag_all,
                   m0_sub_all, m1_sub_all, m2_sub_all)

        for j_py in range(n_local):
            j_jax = jnp.array(j_py, dtype=jnp.int32)
            (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
             prev_lower_y, arrow_rhs_acc,
             cs_out, as_out, y_out) = _chol_one_block(
                cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
                prev_lower_y, arrow_rhs_acc,
                j_jax, rhs_st_local,
                *sc_args,
                coreg_w, likelihood_precs)
            stored_cs_host[j_py] = np.asarray(cs_out)
            stored_as_host[j_py] = np.asarray(as_out)
            y_st_host[j_py] = np.asarray(y_out)
            del cs_out, as_out, y_out

        # y_st is small — transfer to GPU; stored_cs/as stay on CPU
        y_st_local = jnp.array(y_st_host)
        del y_st_host

        # MPI send carries to next rank
        if rank < comm_size - 1:
            _chol_send_carries(cond_schur, arrow_tip_acc, arrow_schur,
                               logdet_cond, prev_lower_y, arrow_rhs_acc)

        # Finalize: bcast logdet, L_tip, arrow_rhs
        logdet_cond_final, L_tip, arrow_rhs_global = _chol_finalize(
            logdet_cond, arrow_tip_acc, arrow_rhs_acc)

        return stored_cs_host, stored_as_host, y_st_local, L_tip, arrow_rhs_global, logdet_cond_final

    # ---- Shared block reconstruction (traced into calling JIT) ----

    def _reconstruct_L_at_block(stored_cs_i, stored_as_i, local_i,
                                sc_args_tuple, coreg_w_arg, likelihood_precs_arg):
        (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
         m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a) = sc_args_tuple
        global_i = start_idx + local_i
        sc_loc = []
        for m in range(n_models):
            sc_loc.append({
                'q1s': q1s_a[m], 'q2s': q2s_a[m], 'q3s': q3s_a[m],
                'scale': scale_a[m], 'exp_gt': exp_gt_a[m],
                'm0_diag': m0d_a[m], 'm1_diag': m1d_a[m], 'm2_diag': m2d_a[m],
                'm0_subdiag': m0s_a[m], 'm1_subdiag': m1s_a[m], 'm2_subdiag': m2s_a[m],
            })

        q_diag = _reconstruct_coregional_diag_block(
            sc_loc, coreg_w_arg, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_diag = q_diag.at[
                m_off + per_model_ata_diag_rows[m][local_i],
                m_off + per_model_ata_diag_cols[m][local_i]
            ].add(likelihood_precs_arg[m] * per_model_ata_diag_vals[m][local_i])
        q_diag = q_diag + _chol_eps_reg * _chol_eye_bs - stored_cs_i
        L_i = _jax_cholesky(q_diag)

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
        q_arrow = q_arrow - stored_as_i
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    # ---- Stage 1b: Backward sub (Python loop with per-block JIT) ----

    @jax.jit
    def _bwd_one_block(stored_cs_i, stored_as_i, y_i, x_next, x_fe, local_i,
                       sc_args_tuple, coreg_w_arg, likelihood_precs_arg):
        L_i, L_lower_i, L_arrow_i = _reconstruct_L_at_block(
            stored_cs_i, stored_as_i, local_i,
            sc_args_tuple, coreg_w_arg, likelihood_precs_arg)
        global_i = start_idx + local_i
        rhs = y_i - L_arrow_i.T @ x_fe
        rhs = jnp.where(global_i < nt - 1, rhs - L_lower_i.T @ x_next, rhs)
        x_i = jax.scipy.linalg.solve_triangular(L_i.T, rhs, lower=False)
        return x_i

    @jax.jit
    def _bwd_recv_x_next(x_next):
        return mpi4jax.recv(x_next, source=rank + 1, tag=10, comm=comm)

    @jax.jit
    def _bwd_send_x_first(x_first):
        mpi4jax.send(x_first, dest=rank - 1, tag=10, comm=comm)

    @jax.jit
    def _bwd_allgather_reduce(local_x_st, local_quad):
        from mpi4py import MPI as _MPI
        x_gathered = mpi4jax.allgather(local_x_st, comm=comm)
        x_st_global = x_gathered.reshape(-1, block_size)[:nt]
        quad_global = mpi4jax.allreduce(local_quad, op=_MPI.SUM, comm=comm)
        return x_st_global, quad_global

    def _bwd_fn(theta, stored_cs_host, stored_as_host, L_tip, y_st_local, arrow_rhs_acc):
        likelihood_precs = _theta_to_likelihood_precs(theta)
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        sc_padded = _pad_subdiags(sc_list)

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

        y_fe = jax.scipy.linalg.solve_triangular(L_tip, arrow_rhs_acc, lower=True)
        local_quad = jnp.sum(y_st_local ** 2)
        y_fe_quad = jnp.where(rank == comm_size - 1, jnp.sum(y_fe ** 2), 0.0)
        local_quad = local_quad + y_fe_quad
        x_fe = jax.scipy.linalg.solve_triangular(L_tip.T, y_fe, lower=False)

        x_next = jnp.zeros(block_size, dtype=dtype)
        if rank < comm_size - 1:
            x_next = _bwd_recv_x_next(x_next)

        max_n_local = (nt + comm_size - 1) // comm_size
        local_x_host = np.zeros((max_n_local, block_size), dtype=np_dtype)

        for local_i_py in range(n_local - 1, -1, -1):
            cs_i = jnp.array(stored_cs_host[local_i_py])
            as_i = jnp.array(stored_as_host[local_i_py])
            local_i_jax = jnp.array(local_i_py, dtype=jnp.int32)
            x_i = _bwd_one_block(cs_i, as_i, y_st_local[local_i_py],
                                 x_next, x_fe, local_i_jax,
                                 sc_args_t, coreg_w, likelihood_precs)
            local_x_host[local_i_py] = np.asarray(x_i)
            x_next = x_i
            del cs_i, as_i

        if rank > 0:
            _bwd_send_x_first(x_next)

        local_x_st = jnp.array(local_x_host)
        x_st_global, quad_global = _bwd_allgather_reduce(local_x_st, local_quad)
        x = jnp.concatenate([x_st_global.reshape(-1), x_fe])
        return quad_global, x

    # ---- Stage 1b: logdet Q_prior (separate JIT to avoid storing both Schur sets) ----

    @jax.jit
    def _logdet_prior_fn(theta):
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        sc_padded = _pad_subdiags(sc_list)
        return pipeline_logdet_Q_prior_coregional_scan(
            sc_padded, coreg_w, n_models, ns, nt, dtype,
            rank, comm_size, n_local, start_idx, comm)

    # ---- Stage 2: Selected inversion gradients (Python loop) ----

    n_coreg = n_sigmas + n_lambdas
    n_models_cubed = n_models * n_models * n_models

    @jax.jit
    def _si_last_block(stored_cs_i, stored_as_i, sd_boundary, sa_boundary,
                       S_tip_arg, L_tip_arg, local_i,
                       sc_args_tuple, coreg_w_arg, likelihood_precs_arg,
                       jac_q1s_a, jac_q2s_a, jac_q3s_a,
                       jac_scale_a, jac_exp_gt_a, jac_coreg_w_arg):
        """Process the last local block for SI grads."""
        L_i, L_lower_i, L_arrow_i = _reconstruct_L_at_block(
            stored_cs_i, stored_as_i, local_i,
            sc_args_tuple, coreg_w_arg, likelihood_precs_arg)
        L_blk_inv = jax.scipy.linalg.solve_triangular(L_i, _chol_eye_bs, lower=True)

        global_i = start_idx + local_i
        is_global_last = (global_i == nt - 1)

        sa_global_last = -S_tip_arg @ L_arrow_i @ L_blk_inv
        sd_global_last = (L_blk_inv.T - sa_global_last.T @ L_arrow_i) @ L_blk_inv

        sl_bnd = (-sd_boundary @ L_lower_i - sa_boundary.T @ L_arrow_i) @ L_blk_inv
        sa_bnd = (-sa_boundary @ L_lower_i - S_tip_arg @ L_arrow_i) @ L_blk_inv
        sd_bnd = (L_blk_inv.T - sl_bnd.T @ L_lower_i - sa_bnd.T @ L_arrow_i) @ L_blk_inv

        sd_last = jnp.where(is_global_last, sd_global_last, sd_bnd)
        sa_last = jnp.where(is_global_last, sa_global_last, sa_bnd)

        (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
         m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a) = sc_args_tuple

        # --- Diagonal trace accumulation ---
        def _diag_body(flat_idx, acc):
            g_st_, g_c_ = acc
            ii = flat_idx // (n_models * n_models)
            jj = (flat_idx // n_models) % n_models
            m_idx = flat_idx % n_models
            sd_ij = lax.dynamic_slice(sd_last, (ii * ns, jj * ns), (ns, ns))
            w_ijm = coreg_w_arg[ii, jj, m_idx]
            base_d = (m0d_a[m_idx, global_i] * q3s_a[m_idx]
                      + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * q2s_a[m_idx]
                      + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * q1s_a[m_idx])
            tr_val = jnp.sum(sd_ij * (scale_a[m_idx] * base_d).T)
            dw = jac_coreg_w_arg[ii, jj, m_idx, :]
            g_c_ = g_c_ + dw * tr_val
            partial_gt_d = (m1d_a[m_idx, global_i] * q2s_a[m_idx]
                            + 2.0 * exp_gt_a[m_idx] * m2d_a[m_idx, global_i] * q1s_a[m_idx])
            for k in range(3):
                dQu = (jac_scale_a[m_idx, k] * base_d
                       + scale_a[m_idx] * (m0d_a[m_idx, global_i] * jac_q3s_a[m_idx, :, :, k]
                                           + exp_gt_a[m_idx] * m1d_a[m_idx, global_i] * jac_q2s_a[m_idx, :, :, k]
                                           + exp_gt_a[m_idx]**2 * m2d_a[m_idx, global_i] * jac_q1s_a[m_idx, :, :, k])
                       + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt_d)
                g_st_ = g_st_.at[m_idx, k].add(w_ijm * jnp.sum(sd_ij * dQu.T))
            return (g_st_, g_c_)

        g_st_init = jnp.zeros((n_models, 3), dtype=dtype)
        g_c_init = jnp.zeros(n_coreg, dtype=dtype)
        g_st, g_c = lax.fori_loop(0, n_models_cubed, _diag_body, (g_st_init, g_c_init))

        # Arrow + diag likelihood
        g_lik = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            sp_d = jnp.sum(sd_last[m_off + per_model_ata_diag_rows[m][local_i],
                                   m_off + per_model_ata_diag_cols[m][local_i]]
                           * per_model_ata_diag_vals[m][local_i])
            sp_a = jnp.sum(sa_last[per_model_ata_arrow_rows[m][local_i],
                                   m_off + per_model_ata_arrow_cols[m][local_i]]
                           * per_model_ata_arrow_vals[m][local_i])
            g_lik = g_lik.at[m].add(sp_d + 2.0 * sp_a)
            tip_val = jnp.where(rank == comm_size - 1,
                                jnp.sum(S_tip_arg * per_model_ata_tip[m]), 0.0)
            g_lik = g_lik.at[m].add(tip_val)

        # Lower block trace at last step (non-global-last only)
        safe_idx = jnp.minimum(global_i, nt - 2)
        sl_for_trace = jnp.where(is_global_last, jnp.zeros_like(sl_bnd), sl_bnd)

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
            partial_gt_l = (m1s_a[m_idx, safe_idx] * q2s_a[m_idx]
                            + 2.0 * exp_gt_a[m_idx] * m2s_a[m_idx, safe_idx] * q1s_a[m_idx])
            for k in range(3):
                dQu_l = (jac_scale_a[m_idx, k] * base_l
                         + scale_a[m_idx] * (m0s_a[m_idx, safe_idx] * jac_q3s_a[m_idx, :, :, k]
                                             + exp_gt_a[m_idx] * m1s_a[m_idx, safe_idx] * jac_q2s_a[m_idx, :, :, k]
                                             + exp_gt_a[m_idx]**2 * m2s_a[m_idx, safe_idx] * jac_q1s_a[m_idx, :, :, k])
                         + scale_a[m_idx] * jac_exp_gt_a[m_idx, k] * partial_gt_l)
                g_st_ = g_st_.at[m_idx, k].add(2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))
            return (g_st_, g_c_)

        g_st_l, g_c_l = lax.fori_loop(0, n_models_cubed, _lower_body, (g_st_init, g_c_init))
        g_st = g_st + jnp.where(is_global_last, jnp.zeros_like(g_st_l), g_st_l)
        g_c = g_c + jnp.where(is_global_last, jnp.zeros_like(g_c_l), g_c_l)

        # Lower AtA likelihood
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            sl_lik = jnp.where(
                is_global_last, 0.0,
                jnp.sum(sl_bnd[m_off + _padded_lower_rows[m][local_i],
                               m_off + _padded_lower_cols[m][local_i]]
                        * _padded_lower_vals[m][local_i]))
            g_lik = g_lik.at[m].add(2.0 * sl_lik)

        return sd_last, sa_last, g_st, g_lik, g_c

    @jax.jit
    def _si_inner_block(stored_cs_i, stored_as_i, sd_prev, sa_prev, S_tip_arg,
                        local_i,
                        sc_args_tuple, coreg_w_arg, likelihood_precs_arg,
                        jac_q1s_a, jac_q2s_a, jac_q3s_a,
                        jac_scale_a, jac_exp_gt_a, jac_coreg_w_arg):
        """Process an inner block for SI grads."""
        L_i, L_lower_i, L_arrow_i = _reconstruct_L_at_block(
            stored_cs_i, stored_as_i, local_i,
            sc_args_tuple, coreg_w_arg, likelihood_precs_arg)
        L_blk_inv = jax.scipy.linalg.solve_triangular(L_i, _chol_eye_bs, lower=True)

        global_i = start_idx + local_i
        sl_i = (-sd_prev @ L_lower_i - sa_prev.T @ L_arrow_i) @ L_blk_inv
        sa_i = (-sa_prev @ L_lower_i - S_tip_arg @ L_arrow_i) @ L_blk_inv
        sd_i = (L_blk_inv.T - sl_i.T @ L_lower_i - sa_i.T @ L_arrow_i) @ L_blk_inv

        (q1s_a, q2s_a, q3s_a, scale_a, exp_gt_a,
         m0d_a, m1d_a, m2d_a, m0s_a, m1s_a, m2s_a) = sc_args_tuple

        # Diagonal trace
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

        # Lower trace
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

        # Likelihood from sparse AtA
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

        return sd_i, sa_i, g_st, g_lik, g_c

    @jax.jit
    def _si_recv_boundary(sd_b, sa_b):
        sd_b = mpi4jax.recv(sd_b, source=rank + 1, tag=20, comm=comm)
        sa_b = mpi4jax.recv(sa_b, source=rank + 1, tag=21, comm=comm)
        return sd_b, sa_b

    @jax.jit
    def _si_send_boundary(sd_b, sa_b):
        mpi4jax.send(sd_b, dest=rank - 1, tag=20, comm=comm)
        mpi4jax.send(sa_b, dest=rank - 1, tag=21, comm=comm)

    @jax.jit
    def _si_allreduce(g_st, g_lik, g_c):
        from mpi4py import MPI as _MPI
        g_st = mpi4jax.allreduce(g_st, op=_MPI.SUM, comm=comm)
        g_lik = mpi4jax.allreduce(g_lik, op=_MPI.SUM, comm=comm)
        g_c = mpi4jax.allreduce(g_c, op=_MPI.SUM, comm=comm)
        return g_st, g_lik, g_c

    def _grad_si_fn(theta, stored_cs_host, stored_as_host, L_tip):
        likelihood_precs = _theta_to_likelihood_precs(theta)
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        sc_padded = _pad_subdiags(sc_list)
        jac_sc_list = _theta_to_jac_sc(theta)
        jac_coreg_w = _theta_to_jac_coreg_w(theta)

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

        L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, _chol_eye_nfe, lower=True)
        S_tip = L_tip_inv.T @ L_tip_inv

        sd_boundary = jnp.zeros((block_size, block_size), dtype=dtype)
        sa_boundary = jnp.zeros((n_fe, block_size), dtype=dtype)
        if rank < comm_size - 1:
            sd_boundary, sa_boundary = _si_recv_boundary(sd_boundary, sa_boundary)

        last_local = n_local - 1
        cs_last = jnp.array(stored_cs_host[last_local])
        as_last = jnp.array(stored_as_host[last_local])
        local_i_last = jnp.array(last_local, dtype=jnp.int32)

        sd_prev, sa_prev, g_st_acc, g_lik_acc, g_c_acc = _si_last_block(
            cs_last, as_last, sd_boundary, sa_boundary, S_tip, L_tip,
            local_i_last, sc_args_t, coreg_w, likelihood_precs,
            jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
            jac_coreg_w)
        del cs_last, as_last

        for local_i_py in range(n_local - 2, -1, -1):
            cs_i = jnp.array(stored_cs_host[local_i_py])
            as_i = jnp.array(stored_as_host[local_i_py])
            local_i_jax = jnp.array(local_i_py, dtype=jnp.int32)

            sd_prev, sa_prev, g_st_i, g_lik_i, g_c_i = _si_inner_block(
                cs_i, as_i, sd_prev, sa_prev, S_tip,
                local_i_jax, sc_args_t, coreg_w, likelihood_precs,
                jac_q1s_a, jac_q2s_a, jac_q3s_a, jac_scale_a, jac_exp_gt_a,
                jac_coreg_w)
            g_st_acc = g_st_acc + g_st_i
            g_lik_acc = g_lik_acc + g_lik_i
            g_c_acc = g_c_acc + g_c_i
            del cs_i, as_i

        if rank > 0:
            _si_send_boundary(sd_prev, sa_prev)

        g_st_acc, g_lik_acc, g_c_acc = _si_allreduce(g_st_acc, g_lik_acc, g_c_acc)
        grad_lik = likelihood_precs * g_lik_acc
        return g_st_acc, grad_lik, g_c_acc

    # ---- Stage 3: logdet Q_prior gradients (JIT-compiled) ----

    @jax.jit
    def _grad_prior_fn(theta):
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        sc_padded = _pad_subdiags(sc_list)
        jac_sc_list_padded = _theta_to_jac_sc_padded(theta)
        jac_coreg_w = _theta_to_jac_coreg_w(theta)

        grad_prior_st, grad_prior_coreg = pipeline_logdet_Q_prior_coregional_grad(
            sc_padded, jac_sc_list_padded, coreg_w, jac_coreg_w,
            n_models, ns, nt, dtype,
            rank, comm_size, n_local, start_idx, comm)

        grad_prior_st_arr = jnp.stack(grad_prior_st)
        return grad_prior_st_arr, grad_prior_coreg

    # ---- Stage 4: Quadratic form gradients (JIT-compiled) ----

    @jax.jit
    def _grad_quad_fn(theta, x):
        likelihood_precs = _theta_to_likelihood_precs(theta)
        sc_list, coreg_w = _theta_to_sc_coreg(theta)
        jac_sc_list = _theta_to_jac_sc(theta)
        jac_coreg_w = _theta_to_jac_coreg_w(theta)
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

    # ---- Stage 5: Scalar gradient terms (log prior + log likelihood) ----

    @jax.jit
    def _grad_scalar_fn(theta):
        """Gradient of scalar terms: log_prior_hyperparameters + log_likelihood."""
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

    # ---- Gradient combiner (NOT JIT-compiled — runs on host) ----

    def _combine_gradients(theta_jax,
                           grad_cond_st, grad_cond_lik, grad_cond_coreg,
                           grad_prior_st, grad_prior_coreg,
                           grad_quad_st, grad_quad_lik, grad_quad_coreg,
                           grad_scalar):
        bar_logdet_prior = -0.5
        bar_logdet_cond = 0.5
        bar_quad = -0.5

        n_theta = theta_jax.shape[0]
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
        theta_jax = jnp.asarray(theta, dtype=dtype)

        # Stage 1a: Cholesky + forward sub (stored_cs/as on CPU)
        cs_host, as_host, y_st_local, L_tip, arrow_rhs_acc, logdet_cond = \
            _chol_fn(theta_jax)

        # Stage 1b: Backward sub (fetches stored_cs from CPU one block at a time)
        quad, x_val = _bwd_fn(
            theta_jax, cs_host, as_host, L_tip, y_st_local, arrow_rhs_acc)
        del y_st_local, arrow_rhs_acc

        # Stage 1c: logdet prior
        logdet_prior = _logdet_prior_fn(theta_jax)

        f_val = _compute_objective(theta_jax, logdet_cond, quad, logdet_prior)

        # Stage 2: SI gradients (fetches stored_cs from CPU one block at a time)
        grad_cond_st, grad_cond_lik, grad_cond_coreg = _grad_si_fn(
            theta_jax, cs_host, as_host, L_tip)
        del cs_host, as_host, L_tip

        # Stage 3: logdet prior gradients
        grad_prior_st, grad_prior_coreg = _grad_prior_fn(theta_jax)

        # Stage 4: Quad gradients (uses x from forward)
        grad_quad_st, grad_quad_lik, grad_quad_coreg = _grad_quad_fn(
            theta_jax, x_val)

        # Stage 5: Scalar term gradients
        grad_scalar = _grad_scalar_fn(theta_jax)

        # Combine
        grad_val = _combine_gradients(
            theta_jax,
            grad_cond_st, grad_cond_lik, grad_cond_coreg,
            grad_prior_st, grad_prior_coreg,
            grad_quad_st, grad_quad_lik, grad_quad_coreg,
            grad_scalar)

        return float(f_val), np.asarray(grad_val, dtype=np_dtype), np.asarray(x_val, dtype=np_dtype)

    def objective_fn(theta):
        theta_jax = jnp.asarray(theta, dtype=dtype)
        cs_host, as_host, y_st_local, L_tip, arrow_rhs_acc, logdet_cond = \
            _chol_fn(theta_jax)
        quad, x_val = _bwd_fn(
            theta_jax, cs_host, as_host, L_tip, y_st_local, arrow_rhs_acc)
        logdet_prior = _logdet_prior_fn(theta_jax)
        f_val = _compute_objective(theta_jax, logdet_cond, quad, logdet_prior)
        return float(f_val), np.asarray(x_val, dtype=np_dtype)

    # Warmup: compile each stage
    theta_init = jnp.ones(n_hyperparameters, dtype=dtype)
    from dalia.utils import print_msg
    print_msg("Split-JIT: compiling Cholesky + forward sub...")
    cs0, as0, yst0, lt0, arr0, logdet0 = _chol_fn(theta_init)
    print_msg("Split-JIT: compiling backward sub...")
    quad0, x0 = _bwd_fn(theta_init, cs0, as0, lt0, yst0, arr0)
    del yst0, arr0
    print_msg("Split-JIT: compiling logdet prior...")
    _ = _logdet_prior_fn(theta_init)
    print_msg("Split-JIT: compiling SI gradient...")
    _ = _grad_si_fn(theta_init, cs0, as0, lt0)
    del cs0, as0, lt0
    print_msg("Split-JIT: compiling prior gradient...")
    _ = _grad_prior_fn(theta_init)
    print_msg("Split-JIT: compiling quad gradient...")
    _ = _grad_quad_fn(theta_init, x0)
    print_msg("Split-JIT: compiling scalar gradient...")
    _ = _grad_scalar_fn(theta_init)
    print_msg("Split-JIT: all stages compiled.")

    return objective_fn, objective_with_grad
