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
from serinv.algs.pobtaf_jax import pobtaf_jax_optimized


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

    a_matrix = model.a.toarray() if hasattr(model.a, 'toarray') else np.array(model.a)

    prior_configs = []
    for i, prior_hp in enumerate(model.prior_hyperparameters):
        hp_type = getattr(prior_hp, 'hyperparameter_type', 'unknown')
        alpha = float(prior_hp.config.alpha) if hasattr(prior_hp.config, 'alpha') else 0.01
        u = float(prior_hp.config.u) if hasattr(prior_hp.config, 'u') else 5.0
        lambda_theta = float(prior_hp.lambda_theta)
        prior_configs.append({
            'hyperparameter_type': hp_type,
            'alpha': alpha,
            'u': u,
            'lambda_theta': lambda_theta,
        })

    fixed_effects_precision = 0.001
    for submodel in model.submodels:
        if hasattr(submodel, 'submodel_type') and submodel.submodel_type == 'regression':
            if hasattr(submodel.config, 'fixed_effects_prior_precision'):
                fixed_effects_precision = float(submodel.config.fixed_effects_prior_precision)
            break

    static_data = {
        'likelihood_type': likelihood_type,
        'a': jnp.array(a_matrix, dtype=jnp.float64),
        'y': jnp.array(np.array(model.y), dtype=jnp.float64),
        'n_fixed_effects': int(model.n_fixed_effects),
        'fixed_effects_precision': float(fixed_effects_precision),
        'prior_configs': prior_configs,
        'n_observations': int(model.n_observations),
        'n_latent_parameters': int(model.n_latent_parameters),
        'inner_iter_tol': float(dalia_instance.config.eps_inner_iteration),
        'inner_iter_max': int(dalia_instance.config.inner_iteration_max_iter),
        'use_sparse_solver': False,
    }

    for submodel in model.submodels:
        if hasattr(submodel, 'submodel_type') and submodel.submodel_type == 'spatio_temporal':
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


def _evaluate_log_prior_hyperparameters_jax(theta, prior_configs):
    """Evaluate log prior for all hyperparameters matching DALIA's implementation.

    Each hyperparameter type has a different PC prior formula:
    - r_s (spatial range): log(lambda) - lambda * exp(-theta) - theta
    - r_t (temporal range): log(lambda) + log(0.5) - lambda * exp(-0.5*theta) - 0.5*theta
    - sigma_st/sigma_e: log(lambda) - lambda * exp(theta) + theta
    - prec_o (observation precision): log(lambda) - lambda * exp(theta) + theta

    inputs:
    theta : Hyperparameters array [gamma_s, gamma_t, gamma_st, theta_likelihood]
    prior_configs : List of dicts with 'hyperparameter_type', 'lambda_theta' for each prior

    Returns:
    log_prior : Sum of log priors for all hyperparameters
    """
    log_prior = 0.0

    for i, config in enumerate(prior_configs):
        hp_type = config['hyperparameter_type']
        lambda_theta = config['lambda_theta']
        theta_i = theta[i]

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

    return log_prior


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
    prior_configs = static_data['prior_configs']

    theta_likelihood = theta[-1]

    Q_prior = jnp.eye(n_fixed_effects) * fixed_effects_precision

    eta = jnp.zeros_like(y)

    D_diag = -jnp.exp(theta_likelihood) * jnp.ones(len(y))
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
    a = static_data['a']
    prior_configs = static_data['prior_configs']

    theta_st = theta[:-1]
    theta_likelihood = theta[-1]

    Q_st = build_spatio_temporal_Q_jax(
        theta_st, spatial_matrices, temporal_matrices, manifold
    )

    Q_prior_full = build_Q_conditional_jax(
        Q_st, fixed_effects_precision, n_fixed_effects, nt, ns
    )

    likelihood_precision = jnp.exp(theta_likelihood)
    D_diag = -likelihood_precision * jnp.ones(len(y))
    AtDA = a.T @ jnp.diag(D_diag) @ a
    Q_conditional_full = Q_prior_full - AtDA

    diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip = \
        kronecker_to_bta_structure(Q_conditional_full, nt, ns, n_fixed_effects)

    diag_blocks = diag_blocks.copy()
    lower_diag_blocks = lower_diag_blocks.copy()
    lower_arrow_blocks = lower_arrow_blocks.copy()
    arrow_tip = arrow_tip.copy()

    diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip = pobtaf_jax_optimized(
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip
    )

    logdet_Q_conditional = compute_logdet_from_cholesky_bta_jax(
        diag_blocks, arrow_tip
    )

    gradient_likelihood = likelihood_precision * y
    rhs = a.T @ gradient_likelihood

    x = solve_bta_system_jax(
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip, rhs
    )

    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    eta = jnp.zeros_like(y)
    log_likelihood = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    _, logdet_Q_st = jnp.linalg.slogdet(Q_st)
    log_prior_latent = 0.5 * logdet_Q_st

    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * x.T @ Q_conditional_full @ x

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective
