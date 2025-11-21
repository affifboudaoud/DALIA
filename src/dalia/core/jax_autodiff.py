# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Callable, Tuple, Dict, Any
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from dalia.core.jax_sparse_helpers import (
    build_spatio_temporal_Q_jax,
    build_Q_conditional_jax,
    kronecker_to_bta_structure,
    compute_logdet_from_cholesky_bta_jax,
    solve_bta_system_jax,
)
from serinv.algs import pobtaf


def create_pure_jax_objective(dalia_instance) -> Tuple[Callable, Callable]:
    """Create pure JAX objective function with automatic differentiation.

    Supports:
    - Gaussian, Poisson, and Binomial likelihoods
    - Dense or sparse (serinv) solvers
    - Single-process (no MPI)

    inputs:
    dalia_instance : DALIA instance.

    Returns:
    objective_func : Pure JAX objective function.
    objective_with_grad : Function returning both forward value and gradient.
    """
    static_data = _extract_static_data(dalia_instance)
    likelihood_type = static_data['likelihood_type']
    use_sparse = static_data.get('use_sparse_solver', False)

    def objective_pure_jax(theta):
        """Pure JAX objective function - dispatches based on likelihood and solver type"""
        if use_sparse:
            if likelihood_type == 'gaussian':
                return _objective_gaussian_sparse(theta, static_data)
            else:
                raise ValueError(f"Sparse solver only supports Gaussian likelihood, got {likelihood_type}")
        else:
            if likelihood_type == 'gaussian':
                return _objective_gaussian_dense(theta, static_data)
            elif likelihood_type == 'poisson':
                return _objective_poisson_dense(theta, static_data)
            elif likelihood_type == 'binomial':
                return _objective_binomial_dense(theta, static_data)
            else:
                raise ValueError(f"Unsupported likelihood type: {likelihood_type}")

    value_and_grad_fn = jax.value_and_grad(objective_pure_jax)

    objective_pure_jax = jax.jit(objective_pure_jax)
    value_and_grad_fn = jax.jit(value_and_grad_fn)

    # Warmup JIT compilation
    theta_init = jnp.ones(dalia_instance.model.n_hyperparameters, dtype=jnp.float64)
    _ = value_and_grad_fn(theta_init)

    def objective_with_grad(theta):
        theta_jax = jnp.asarray(theta, dtype=jnp.float64)
        f_val, grad_val = value_and_grad_fn(theta_jax)
        return float(f_val), np.asarray(grad_val, dtype=np.float64)

    return objective_pure_jax, objective_with_grad


def _extract_static_data(dalia_instance) -> Dict[str, Any]:
    """Extract static data from DALIA instance for pure JAX function.

    Inputs:
    dalia_instance : DALIA instance.

    Returns:
    static_data : Dictionary containing model-specific data for all likelihood types.
    """
    model = dalia_instance.model
    likelihood_type = model.likelihood_config.type

    if len(model.submodels) > 0:
        submodel = model.submodels[0]
        if hasattr(submodel, 'a'):
            a_matrix = submodel.a.toarray() if hasattr(submodel.a, 'toarray') else np.array(submodel.a)
        else:
            a_matrix = np.eye(model.n_observations)
    else:
        a_matrix = np.eye(model.n_observations)

    if len(model.prior_hyperparameters) > 0:
        pc_prior = model.prior_hyperparameters[0]
        pc_prior_alpha = float(pc_prior.config.alpha) if hasattr(pc_prior.config, 'alpha') else 0.01
        pc_prior_u = float(pc_prior.config.u) if hasattr(pc_prior.config, 'u') else 5.0
    else:
        pc_prior_alpha = 0.01
        pc_prior_u = 5.0

    fixed_effects_precision = 0.001
    if len(model.submodels) > 0 and hasattr(model.submodels[0].config, 'fixed_effects_prior_precision'):
        fixed_effects_precision = float(model.submodels[0].config.fixed_effects_prior_precision)

    static_data = {
        'likelihood_type': likelihood_type,
        'a': jnp.array(a_matrix, dtype=jnp.float64),
        'y': jnp.array(np.array(model.y), dtype=jnp.float64),
        'n_fixed_effects': int(model.n_fixed_effects),
        'fixed_effects_precision': float(fixed_effects_precision),
        'pc_prior_alpha': float(pc_prior_alpha),
        'pc_prior_u': float(pc_prior_u),
        'n_observations': int(model.n_observations),
        'inner_iter_tol': float(dalia_instance.config.eps_inner_iteration),
        'inner_iter_max': int(dalia_instance.config.inner_iteration_max_iter),
        'use_sparse_solver': False,
    }

    for submodel in model.submodels:
        if hasattr(submodel, 'type') and submodel.type == 'spatio_temporal':
            static_data['use_sparse_solver'] = True
            static_data['nt'] = int(submodel.nt)
            static_data['ns'] = int(submodel.ns)
            static_data['manifold'] = str(submodel.manifold)

            static_data['spatial_matrices'] = {
                'c0': jnp.array(submodel.c0.toarray(), dtype=jnp.float64),
                'g1': jnp.array(submodel.g1.toarray(), dtype=jnp.float64),
                'g2': jnp.array(submodel.g2.toarray(), dtype=jnp.float64),
                'g3': jnp.array(submodel.g3.toarray(), dtype=jnp.float64),
            }

            static_data['temporal_matrices'] = {
                'm0': jnp.array(submodel.m0.toarray(), dtype=jnp.float64),
                'm1': jnp.array(submodel.m1.toarray(), dtype=jnp.float64),
                'm2': jnp.array(submodel.m2.toarray(), dtype=jnp.float64),
            }
            break

    if likelihood_type == 'poisson':
        if hasattr(model.likelihood, 'e'):
            e = np.array(model.likelihood.e)
        else:
            e = np.ones(model.n_observations)
        static_data['e'] = jnp.array(e, dtype=jnp.float64)

    elif likelihood_type == 'binomial':
        if hasattr(model.likelihood, 'n_trials'):
            n_trials = np.array(model.likelihood.n_trials)
        else:
            n_trials = np.ones(model.n_observations)
        static_data['n_trials'] = jnp.array(n_trials, dtype=jnp.float64)

    return static_data


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


def _inner_iteration_jax(a, y, Q_prior, grad_likelihood_fn, hess_diag_fn, tol, max_iter):
    """JAX implementation of inner iteration for non-Gaussian likelihoods.

    Uses jax.lax.while_loop for differentiable iterative optimization.

    inputs:
    a : Design matrix
    y : Observations
    Q_prior : Prior precision matrix
    grad_likelihood_fn : Function computing likelihood gradient
    hess_diag_fn : Function computing diagonal of likelihood Hessian
    tol : Convergence tolerance
    max_iter : Maximum iterations

    Returns:
    Q_conditional : Conditional precision matrix
    x_star : Optimal latent parameters
    eta : Linear predictor (A @ x_star)
    """
    n_latent = Q_prior.shape[0]

    def cond_fn(state):
        _, _, counter, norm = state
        return (norm >= tol) & (counter < max_iter)

    def body_fn(state):
        """One iteration of Newton-Raphson"""
        x_star, x_update, counter, _ = state

        x_star = x_star + x_update
        eta = a @ x_star

        D_diag = hess_diag_fn(eta)
        Q_conditional = Q_prior - a.T @ jnp.diag(D_diag) @ a

        gradient_likelihood = grad_likelihood_fn(eta)
        rhs = -Q_prior @ x_star + a.T @ gradient_likelihood

        x_update = jnp.linalg.solve(Q_conditional, rhs)
        norm = jnp.linalg.norm(x_update)

        return (x_star, x_update, counter + 1, norm)

    # Initialize state
    x_star = jnp.zeros(n_latent, dtype=jnp.float64)
    x_update = jnp.zeros(n_latent, dtype=jnp.float64)
    counter = 0
    norm = 1.0

    # Run iteration
    x_star, x_update, counter, norm = lax.while_loop(
        cond_fn,
        body_fn,
        (x_star, x_update, counter, norm)
    )

    # Compute final values
    x_star = x_star + x_update
    eta = a @ x_star
    D_diag = hess_diag_fn(eta)
    Q_conditional = Q_prior - a.T @ jnp.diag(D_diag) @ a

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
    pc_prior_alpha = static_data['pc_prior_alpha']
    pc_prior_u = static_data['pc_prior_u']

    theta_likelihood = theta[-1]

    Q_prior = jnp.eye(n_fixed_effects) * fixed_effects_precision

    eta = jnp.zeros_like(y)

    D_diag = -jnp.exp(theta_likelihood) * jnp.ones(len(y))
    Q_conditional = Q_prior - a.T @ jnp.diag(D_diag) @ a

    rhs = -a.T @ (D_diag * y)

    x = jnp.linalg.solve(Q_conditional, rhs)

    lambda_theta = -jnp.log(pc_prior_alpha) / pc_prior_u
    log_prior_hyperparameters = (
        jnp.log(lambda_theta)
        - lambda_theta * jnp.exp(theta_likelihood)
        + theta_likelihood
    )

    log_likelihood = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior - 0.5 * x.T @ Q_prior @ x

    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * x.T @ Q_conditional @ x

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective


def _objective_poisson_dense(theta, static_data):
    """Pure JAX objective function for Poisson likelihood with dense solver."""
    a = static_data['a']
    y = static_data['y']
    e = static_data['e']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    tol = static_data['inner_iter_tol']
    max_iter = static_data['inner_iter_max']

    Q_prior = jnp.eye(n_fixed_effects) * fixed_effects_precision

    # No hyperparameters for Poisson (or use them if available)
    log_prior_hyperparameters = 0.0

    # Inner iteration for non-Gaussian likelihood
    grad_fn = lambda eta: _gradient_poisson_likelihood_jax(eta, y, e)
    hess_fn = lambda eta: _hessian_diag_poisson_jax(eta, e)

    Q_conditional, x, eta = _inner_iteration_jax(
        a, y, Q_prior, grad_fn, hess_fn, tol, max_iter
    )

    log_likelihood = _evaluate_poisson_likelihood_jax(eta, y, e)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior - 0.5 * x.T @ Q_prior @ x

    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * x.T @ Q_conditional @ x

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective


def _objective_binomial_dense(theta, static_data):
    """Pure JAX objective function for Binomial likelihood with dense solver."""
    a = static_data['a']
    y = static_data['y']
    n_trials = static_data['n_trials']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    tol = static_data['inner_iter_tol']
    max_iter = static_data['inner_iter_max']

    Q_prior = jnp.eye(n_fixed_effects) * fixed_effects_precision

    # No hyperparameters for Binomial (or use them if available)
    log_prior_hyperparameters = 0.0

    # Inner iteration for non-Gaussian likelihood
    grad_fn = lambda eta: _gradient_binomial_likelihood_jax(eta, y, n_trials)
    hess_fn = lambda eta: _hessian_diag_binomial_jax(eta, n_trials)

    Q_conditional, x, eta = _inner_iteration_jax(
        a, y, Q_prior, grad_fn, hess_fn, tol, max_iter
    )

    log_likelihood = _evaluate_binomial_likelihood_jax(eta, y, n_trials)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior - 0.5 * x.T @ Q_prior @ x

    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * x.T @ Q_conditional @ x

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective


def _objective_gaussian_sparse(theta, static_data):
    """Pure JAX objective function for Gaussian likelihood with sparse serinv solver.

    Uses block-tridiagonal-arrowhead structure for spatio-temporal models.

    inputs:
    theta : Hyperparameters [gamma_s, gamma_t, gamma_st, theta_likelihood]
    static_data : Static data containing spatial/temporal matrices and model parameters

    Returns:
    objective : INLA objective value
    """
    nt = static_data['nt']
    ns = static_data['ns']
    n_fixed_effects = static_data['n_fixed_effects']
    fixed_effects_precision = static_data['fixed_effects_precision']
    spatial_matrices = static_data['spatial_matrices']
    temporal_matrices = static_data['temporal_matrices']
    manifold = static_data['manifold']
    y = static_data['y']
    pc_prior_alpha = static_data['pc_prior_alpha']
    pc_prior_u = static_data['pc_prior_u']

    theta_st = theta[:-1]
    theta_likelihood = theta[-1]

    Q_st = build_spatio_temporal_Q_jax(
        theta_st, spatial_matrices, temporal_matrices, manifold
    )

    Q_prior_full = build_Q_conditional_jax(
        Q_st, fixed_effects_precision, n_fixed_effects, nt, ns
    )

    D_diag = -jnp.exp(theta_likelihood) * jnp.ones(len(y))
    Q_conditional_full = Q_prior_full.copy()
    for i in range(len(y)):
        Q_conditional_full = Q_conditional_full.at[i, i].add(D_diag[i])

    diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip = \
        kronecker_to_bta_structure(Q_conditional_full, nt, ns, n_fixed_effects)

    diag_blocks = diag_blocks.copy()
    lower_diag_blocks = lower_diag_blocks.copy()
    lower_arrow_blocks = lower_arrow_blocks.copy()
    arrow_tip = arrow_tip.copy()

    result = pobtaf(diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip)
    if result is not None:
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip = result

    logdet_Q_conditional = compute_logdet_from_cholesky_bta_jax(
        diag_blocks, arrow_tip
    )

    rhs = jnp.zeros(nt * ns + n_fixed_effects)
    for i in range(len(y)):
        rhs = rhs.at[i].set(-D_diag[i] * y[i])

    x = solve_bta_system_jax(
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip, rhs
    )

    lambda_theta = -jnp.log(pc_prior_alpha) / pc_prior_u
    log_prior_hyperparameters = (
        jnp.log(lambda_theta)
        - lambda_theta * jnp.exp(theta_likelihood)
        + theta_likelihood
    )

    eta = x[:len(y)]
    log_likelihood = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior_full)
    log_prior_latent = 0.5 * logdet_Q_prior - 0.5 * x.T @ Q_prior_full @ x

    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * x.T @ Q_conditional_full @ x

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective
