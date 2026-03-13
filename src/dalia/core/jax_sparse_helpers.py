# Copyright 2024-2025 DALIA authors. All rights reserved.
#
# Backward-compatibility shim. All code has moved to dalia.core.autodiff.

from dalia.core.autodiff.spatial_precompute import (
    _interpretable_to_compute_jax,
    _jax_cholesky,
    precompute_spatial_components,
    _reconstruct_diag_block,
    _reconstruct_lower_block,
    extract_bta_blocks_sparse_coo_coregional,
    precompute_spatial_components_coregional,
    _reconstruct_coregional_diag_block,
    _reconstruct_coregional_lower_block,
)
from dalia.core.autodiff.q_construction import (
    kronecker_to_bta_structure,
    build_spatio_temporal_Q_jax,
    build_spatio_temporal_Q_bta_jax,
    extract_bta_blocks_from_sparse,
    build_Q_conditional_jax,
    compute_logdet_from_cholesky_bta_jax,
    solve_bta_system_jax,
    quadratic_form_bta_jax,
    build_coregional_Q_bta_jax,
    extract_bta_blocks_sparse_coo,
    scatter_sparse_ata_into_blocks,
)
from dalia.core.autodiff.cholesky import (
    lazy_bta_cholesky,
    pobtasi_jax,
    logdet_Q_st_scan,
)
from dalia.core.autodiff.gradients import (
    bt_logdet_grad,
    _spatial_traces,
    _compute_grad_logdet_cond,
    _compute_grad_quad,
    selected_inversion_grads_jax,
)
from dalia.core.autodiff.cholesky_carries import (
    lazy_bta_cholesky_carries,
    fused_cholesky_fwd_sub,
    backward_sub_from_carries,
    selected_inversion_grads_from_carries_jax,
)
from dalia.core.autodiff.coregional_solvers import (
    fused_cholesky_fwd_sub_coregional,
    backward_sub_from_carries_coregional,
    logdet_Q_prior_coregional_scan,
    logdet_Q_prior_coregional_grad,
    selected_inversion_grads_from_carries_coregional,
    _compute_grad_quad_coregional,
)
from dalia.core.autodiff.distributed import (
    pipeline_logdet_Q_prior_coregional_grad,
    pipeline_compute_grad_quad_coregional,
    twophase_logdet_Q_prior_coregional_scan,
)
