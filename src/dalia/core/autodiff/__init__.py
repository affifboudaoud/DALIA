# Copyright 2024-2025 DALIA authors. All rights reserved.
"""Automatic differentiation backend for DALIA (ADELIA).

Structure-preserving reverse-mode AD for the INLA objective over
block-tridiagonal arrowhead (BTA) precision matrices.  Computes exact
gradients via an analytical decomposition into three independent phases:

- **Phase A**: Selected inversion + log-determinant gradient for Q_c.
- **Phase B**: Quadratic form gradient from the posterior mode x*.
- **Phase C**: Prior log-determinant gradient for Q_p via BT SI.

Key design choices:

1. **Fused forward pass**: Cholesky factorization and forward substitution
   are combined in a single sweep; per-block L factors are temporaries,
   only Schur complement carries are stored (~n blocks of b x b).

2. **Carry-based L reconstruction**: During the backward pass, L factors
   are reconstructed on-the-fly from stored carries, trading one extra
   Cholesky per block for halved memory.

3. **Coregional extension**: Super-blocks of size k*n_s handle multivariate
   models; the k x k sub-block structure is exploited to separate
   per-variable and coupling hyperparameter gradients.

4. **Two-phase distribution**: For models exceeding single-GPU memory, a
   domain decomposition with MPI allgather distributes the factorization
   and its backward pass across P GPUs with CPU-staged carries.

Modules
-------
config
    Precision management, likelihood functions, utilities.
data_extraction
    Static data extraction from DALIA model instances.
spatial_precompute
    Lazy BTA block reconstruction from spatial/temporal FEM matrices.
q_construction
    Full BTA matrix construction and direct operations.
cholesky
    Full-factor BTA Cholesky and selected inversion (AD-Loop variants).
cholesky_carries
    Carry-based fused Cholesky, backward sub, and SI (AD-BTA strategy).
gradients
    Analytical gradient routines (Phases A, B, C) for univariate models.
coregional_solvers
    All BTA operations extended to coregional super-blocks.
distributed
    MPI-distributed gradient routines (pipeline and two-phase).
objectives_univariate
    INLA objective functions for single-variable models.
objectives_coregional
    INLA objective functions for multivariate (coregional) models.
factory
    Factory for single-GPU JIT-compiled objective + gradient functions.
factory_distributed
    Factory for multi-GPU distributed objective + gradient functions.
"""

from dalia.core.autodiff.config import configure_jax_precision, get_jax_dtype
from dalia.core.autodiff.factory import (
    create_pure_jax_objective,
    create_pure_jax_objective_coregional,
)
from dalia.core.autodiff.factory_distributed import (
    create_pure_jax_objective_distributed_coregional_twophase,
)

__all__ = [
    "configure_jax_precision",
    "get_jax_dtype",
    "create_pure_jax_objective",
    "create_pure_jax_objective_coregional",
    "create_pure_jax_objective_distributed_coregional_twophase",
]
