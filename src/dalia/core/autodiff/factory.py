# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Factory functions for creating single-GPU INLA objectives with AD.

These functions extract static data from a DALIA model instance, select
the appropriate objective function based on the model configuration
(likelihood type, solver type, number of variables), JIT-compile it
together with ``jax.value_and_grad``, and return callable functions for
use in L-BFGS-B optimization.

Three AD strategies are supported via the ``ad_mode`` parameter:

- ``"default"`` (AD-BTA): custom backward pass via ``jax.custom_vjp``
  using the structure-preserving gradient decomposition (Phases A/B/C).
  Most memory-efficient; scales to million-variable models.

- ``"scan"``: JAX AD through the ``lax.scan`` loop (AD-Loop).
  Simple but stores all loop carries; OOMs on large models.

- ``"scan_ckpt"``: JAX AD with gradient checkpointing (AD-Loop-Ckpt).
  Reduces memory vs ``"scan"`` by recomputing forward intermediates.

For distributed (multi-GPU) models, see :mod:`factory_distributed`.
"""

from typing import Callable, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from dalia.core.autodiff.config import get_jax_dtype
from dalia.core.autodiff.data_extraction import (
    _extract_static_data,
    _extract_static_data_coregional,
)
from dalia.core.autodiff.objectives_univariate import (
    _objective_gaussian_dense,
    _objective_gaussian_spatial_dense,
    _objective_gaussian_st_dense,
    _objective_poisson_dense,
    _objective_poisson_st_dense,
    _objective_binomial_dense,
    _objective_gaussian_sparse,
    _objective_gaussian_scan_baseline,
)
from dalia.core.autodiff.objectives_coregional import (
    _objective_gaussian_coregional_scan_baseline,
    _objective_gaussian_coregional_sparse,
    _objective_gaussian_coregional_sparse_fused,
    _objective_gaussian_coregional_spatial_dense,
    _objective_gaussian_coregional_st_dense,
)


def create_pure_jax_objective(dalia_instance, dtype=None, ad_mode="default") -> Tuple[Callable, Callable]:
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
    ad_mode : str, optional
        AD strategy. ``"default"`` uses custom VJP for sparse solver.
        ``"scan"`` uses naive ``lax.scan`` AD (no custom VJP).
        ``"scan_ckpt"`` uses ``lax.scan`` AD with ``jax.checkpoint``.

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
                if ad_mode == "scan":
                    return _objective_gaussian_scan_baseline(theta, static_data, checkpoint=False)
                elif ad_mode == "scan_ckpt":
                    return _objective_gaussian_scan_baseline(theta, static_data, checkpoint=True)
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

    # JIT compilation (one-time cost)
    from dalia.utils import print_msg
    import time as _t
    print_msg("JIT compiling objective + gradient...", flush=True)
    _t0 = _t.perf_counter()
    theta_init = jnp.ones(n_hyperparameters, dtype=dtype)
    _ = value_and_grad_fn(theta_init)
    print_msg(f"JIT compilation done in {_t.perf_counter() - _t0:.1f}s.")

    def objective_with_grad(theta):
        """Returns (f_val, grad, x) where x is the latent parameters."""
        theta_jax = jnp.asarray(theta, dtype=dtype)
        (f_val, x_val), grad_val = value_and_grad_fn(theta_jax)
        return float(f_val), np.asarray(grad_val, dtype=np_dtype), np.asarray(x_val, dtype=np_dtype)

    return objective_pure_jax, objective_with_grad


def create_pure_jax_objective_coregional(dalia_instance, dtype=None, ad_mode="default") -> Tuple[Callable, Callable]:
    """Create pure JAX objective function for CoregionalModel with automatic differentiation.

    Parameters
    ----------
    dalia_instance : DALIA
        DALIA instance with CoregionalModel.
    dtype : jnp.dtype, optional
        JAX dtype to use. If None, uses the configured dtype from get_jax_dtype().
    ad_mode : str, optional
        AD strategy. ``"default"`` uses custom VJP for sparse solver.
        ``"scan"`` uses naive ``lax.fori_loop`` AD (no custom VJP).
        ``"scan_ckpt"`` uses ``lax.fori_loop`` AD with ``jax.checkpoint``.

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
    need_dense_ata = ad_mode in ("scan", "scan_ckpt")
    static_data = _extract_static_data_coregional(
        dalia_instance, dtype=dtype, include_dense_ata=need_dense_ata)
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
        if ad_mode in ("scan", "scan_ckpt"):
            ckpt = (ad_mode == "scan_ckpt")
            return _objective_gaussian_coregional_scan_baseline(theta, static_data, checkpoint=ckpt)
        elif use_fused:
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

    # JIT compilation (one-time cost)
    from dalia.utils import print_msg
    import time as _t
    print_msg("JIT compiling coregional objective + gradient...", flush=True)
    _t0 = _t.perf_counter()
    theta_init = jnp.ones(n_hyperparameters, dtype=dtype)
    _ = value_and_grad_fn(theta_init)
    print_msg(f"JIT compilation done in {_t.perf_counter() - _t0:.1f}s.")

    def objective_with_grad(theta):
        theta_jax = jnp.asarray(theta, dtype=dtype)
        (f_val, x_val), grad_val = value_and_grad_fn(theta_jax)
        return float(f_val), np.asarray(grad_val, dtype=np_dtype), np.asarray(x_val, dtype=np_dtype)

    return objective_pure_jax, objective_with_grad
