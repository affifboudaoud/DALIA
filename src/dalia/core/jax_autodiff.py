# Copyright 2024-2025 DALIA authors. All rights reserved.
#
# Backward-compatibility shim. All code has moved to dalia.core.autodiff.

from dalia.core.autodiff.config import (
    configure_jax_precision,
    get_jax_dtype,
    _to_numpy,
    _scipy_sparse_to_jax_bcoo,
    _extract_bta_blocks_coregional,
    bta_to_dense_jax,
    sigmoid_function,
    _evaluate_gaussian_likelihood_jax,
    _evaluate_poisson_likelihood_jax,
    _evaluate_binomial_likelihood_jax,
    _gradient_poisson_likelihood_jax,
    _gradient_binomial_likelihood_jax,
    _hessian_diag_poisson_jax,
    _hessian_diag_binomial_jax,
    _evaluate_log_prior_hyperparameters_jax,
    _inner_iteration_jax,
)
from dalia.core.autodiff.data_extraction import (
    _extract_static_data,
    _extract_static_data_coregional,
    _extract_static_data_distributed_coregional,
)
from dalia.core.autodiff.objectives_univariate import (
    _objective_gaussian_dense,
    _build_spatial_Q_prior_jax,
    _objective_gaussian_spatial_dense,
    _objective_gaussian_st_dense,
    _objective_poisson_dense,
    _objective_poisson_st_dense,
    _objective_binomial_dense,
    _bt_cholesky_step,
    _objective_gaussian_sparse,
    _objective_gaussian_scan_baseline,
)
from dalia.core.autodiff.objectives_coregional import (
    _objective_gaussian_coregional_scan_baseline,
    _objective_gaussian_coregional_sparse,
    _build_coregional_Q_prior_spatial_dense,
    _objective_gaussian_coregional_spatial_dense,
    _objective_gaussian_coregional_st_dense,
    _objective_gaussian_coregional_sparse_fused,
)
from dalia.core.autodiff.factory import (
    create_pure_jax_objective,
    create_pure_jax_objective_coregional,
)
from dalia.core.autodiff.factory_distributed import (
    create_pure_jax_objective_distributed_coregional_twophase,
)
