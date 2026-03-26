# Copyright 2024-2025 DALIA authors. All rights reserved.
"""INLA objective functions for univariate models (single response variable).

Each function implements the INLA objective::

    f(theta) = 1/2 log|Q_p| - 1/2 log|Q_c| - 1/2 x*^T Q_p x*
             + log p(y|x*,theta) + log pi(theta)

where Q_p is the prior precision, Q_c = Q_p + tau A^T A is the conditional
precision, and x* = Q_c^{-1} r is the posterior mode.

Multiple variants are provided, corresponding to different differentiation
strategies and solver backends:

- ``_objective_gaussian_dense``: Dense solver, JAX AD through dense Cholesky.
  Corresponds to the AD-Dense strategy.

- ``_objective_gaussian_scan_baseline``: BTA solver via ``lax.scan``, JAX AD
  through the scan (AD-Loop / AD-Loop-Ckpt strategies).

- ``_objective_gaussian_sparse``: BTA solver with ``custom_vjp`` that
  uses the structure-preserving backward pass (AD-BTA strategy).  The
  ``custom_vjp`` registers ``fused_core_bwd`` which computes analytical
  gradients via the three-phase decomposition:
    - Phase A: SI + log|Q_c| gradient (from Schur carries)
    - Phase B: quadratic form gradient (from posterior mode x*)
    - Phase C: prior log|Q_p| gradient (separate BT sweep)
"""

import jax
import jax.numpy as jnp
from jax import lax

from dalia.core.autodiff.config import (
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
from dalia.core.autodiff.q_construction import (
    build_spatio_temporal_Q_jax,
    build_spatio_temporal_Q_bta_jax,
    solve_bta_system_jax,
    quadratic_form_bta_jax,
)
from dalia.core.autodiff.spatial_precompute import (
    precompute_spatial_components,
)
from dalia.core.autodiff.cholesky import (
    lazy_bta_cholesky,
    logdet_Q_st_scan,
)
from dalia.core.autodiff.cholesky_carries import (
    fused_cholesky_fwd_sub,
    backward_sub_from_carries,
    selected_inversion_grads_from_carries_jax,
)
from dalia.core.autodiff.gradients import (
    bt_logdet_grad,
    _compute_grad_quad,
)
from serinv.algs.pobtaf_jax import pobtaf_jax_optimized
from serinv.algs.pobtf_jax import pobtf_logdet_jax


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
    """INLA objective for Gaussian likelihood using structure-preserving AD (AD-BTA).

    Registers a ``jax.custom_vjp`` (``fused_core``) for the computational
    core (factorization + solve + log-determinants).  The custom backward
    pass ``fused_core_bwd`` computes exact analytical gradients via the
    three-phase decomposition, replacing JAX's automatic tape with
    structure-preserving operations:

    **Forward (``fused_core_fwd``)**: fused BTA Cholesky + forward sub
    (stores Schur carries + forward-sub vectors z_i), backward sub
    (recovers posterior mode x*), and BT Cholesky for log|Q_p|.
    Residuals saved: theta, x*, Schur carries, L_tip.

    **Backward (``fused_core_bwd``)**:

    - Phase A: selected inversion + tr(Z dQ_c/dtheta) from carries
    - Phase B: x*^T (dQ_p/dtheta) x* from posterior mode
    - Phase C: tr(Z_p dQ_p/dtheta) via separate BT SI sweep

    Peak memory is dominated by the n stored Schur carries (n x b x b),
    approximately half the memory of the full Cholesky factor.

    Parameters
    ----------
    theta : jnp.ndarray
        Hyperparameters [r_s, r_t, sigma_st, theta_lik].
    static_data : dict
        Static data from :func:`_extract_static_data`.

    Returns
    -------
    objective : scalar
        INLA objective value (negated for minimization by L-BFGS-B).
    x : jnp.ndarray
        Posterior mode x* = Q_c^{-1} r.
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

        # Residuals: theta (d floats) + x (nt*ns) + carries (nt * ns * ns) + L_tip (n_fe * n_fe).
        # Carries dominate memory. L factors are reconstructed from carries during backward.
        residuals = (theta_st, theta_lik, x, stored_cs, stored_as, L_tip)
        return (logdet_st, logdet_cond, quad, x), residuals

    def fused_core_bwd(residuals, g):
        bar_logdet_st, bar_logdet_cond, bar_quad, _bar_x = g
        theta_st_r, theta_lik_r, x_r, stored_cs_r, stored_as_r, L_tip_r = residuals

        lik_prec = jnp.exp(theta_lik_r)

        # --- Phase A: SI gradients for log|Q_c| ---
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

        # --- Phase B: quadratic form gradients for x*^T Q_p x* ---
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

        # --- Phase C: prior log-determinant gradients for log|Q_p| ---
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

    # Negate for minimization (L-BFGS-B minimizes, f(θ) is maximized)
    objective = -(
        log_prior_hp
        + log_lik
        + 0.5 * logdet_Q_st_val
        - 0.5 * logdet_Q_cond_val
        + 0.5 * quad_form
    )

    return objective, x


def _objective_gaussian_scan_baseline(theta, static_data, checkpoint=False):
    """Pure JAX objective using ``lax.scan`` BTA Cholesky — no custom VJP.

    JAX AD differentiates through the scan directly, storing the full carry
    trajectory in the backward pass.  This serves as a baseline to measure
    the memory cost avoided by our ``custom_vjp`` approach. 
    This is kept for testing and benchmarking, 
    but is not used in the final implementation due to high memory usage.

    Parameters
    ----------
    theta : jnp.ndarray
        Hyperparameters ``[r_s, r_t, sigma_st, theta_likelihood]``.
    static_data : dict
        Static data from :func:`_extract_static_data`.
    checkpoint : bool
        If True, wrap the scan body with ``jax.checkpoint``.

    Returns
    -------
    objective : scalar
    x : jnp.ndarray
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

    theta_st = theta[:-1]
    theta_likelihood = theta[-1]
    lik_prec = jnp.exp(theta_likelihood)

    sc = precompute_spatial_components(
        theta_st, spatial_matrices, temporal_matrices, manifold)

    L_diag, L_lower, L_arrow, L_tip, logdet_Q_cond = lazy_bta_cholesky(
        sc, nt, ns, n_fe, fe_prec, lik_prec,
        ata_diag_rows, ata_diag_cols, ata_diag_vals,
        ata_lower_rows, ata_lower_cols, ata_lower_vals,
        ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
        ata_tip, dtype, checkpoint=checkpoint,
    )

    rhs = lik_prec * (a_sparse.T @ y)
    x = solve_bta_system_jax(L_diag, L_lower, L_arrow, L_tip, rhs)

    logdet_Q_st = logdet_Q_st_scan(
        theta_st, spatial_matrices, temporal_matrices, manifold, nt, ns, dtype)

    quad_form = quadratic_form_bta_jax(
        L_diag, L_lower, L_arrow, L_tip, x, nt, ns)

    log_prior_hp = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)
    eta = jnp.zeros_like(y)
    log_lik = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    objective = -(
        log_prior_hp
        + log_lik
        + 0.5 * logdet_Q_st
        - 0.5 * logdet_Q_cond
        + 0.5 * quad_form
    )

    return objective, x
