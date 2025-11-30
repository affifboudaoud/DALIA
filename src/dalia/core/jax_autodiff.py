# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Callable, Tuple, Dict, Any
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jax_sparse
from scipy import sparse as scipy_sparse

from dalia.core.jax_sparse_helpers import (
    build_spatio_temporal_Q_jax,
    build_spatio_temporal_Q_bta_jax,
    build_Q_conditional_jax,
    kronecker_to_bta_structure,
    extract_bta_blocks_from_sparse,
    compute_logdet_from_cholesky_bta_jax,
    solve_bta_system_jax,
)
from serinv.algs.pobtaf_jax import pobtaf_jax_optimized


def _to_numpy(arr):
    """Convert array to NumPy, handling CuPy arrays."""
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def _scipy_sparse_to_jax_bcoo(sp_matrix):
    """Convert scipy sparse matrix to JAX BCOO format."""
    if hasattr(sp_matrix, 'get'):
        sp_matrix = sp_matrix.get()
    coo = scipy_sparse.coo_matrix(sp_matrix)
    indices = jnp.array(np.column_stack([coo.row, coo.col]), dtype=jnp.int32)
    data = jnp.array(coo.data, dtype=jnp.float64)
    return jax_sparse.BCOO((data, indices), shape=coo.shape)


def create_pure_jax_objective(dalia_instance) -> Tuple[Callable, Callable]:
    """Create pure JAX objective function with automatic differentiation.

    Supports:
    - Gaussian, Poisson, and Binomial likelihoods
    - Dense or sparse (serinv) solvers
    - Single-process (no MPI)
    - Models with zero hyperparameters (Poisson/Binomial only)

    inputs:
    dalia_instance : DALIA instance.

    Returns:
    objective_func : Pure JAX objective function.
    objective_with_grad : Function returning both forward value and gradient.
    """
    static_data = _extract_static_data(dalia_instance)
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
        theta_init = jnp.array([], dtype=jnp.float64)
        _ = objective_pure_jax_no_hp(theta_init)

        def objective_with_grad_no_hp(theta):
            theta_jax = jnp.asarray(theta, dtype=jnp.float64)
            f_val = objective_pure_jax_no_hp(theta_jax)
            return float(f_val), np.array([], dtype=np.float64)

        return objective_pure_jax_no_hp, objective_with_grad_no_hp

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
                return _objective_gaussian_dense(theta, static_data)
            elif likelihood_type == 'poisson':
                return _objective_poisson_dense(theta, static_data)
            elif likelihood_type == 'binomial':
                return _objective_binomial_dense(theta, static_data)
            else:
                raise ValueError(f"Unsupported likelihood type: {likelihood_type}")

    # Use has_aux=True since objective functions return (objective, x)
    value_and_grad_fn = jax.value_and_grad(objective_pure_jax, has_aux=True)

    objective_pure_jax = jax.jit(objective_pure_jax)
    value_and_grad_fn = jax.jit(value_and_grad_fn)

    # Warmup JIT compilation
    theta_init = jnp.ones(n_hyperparameters, dtype=jnp.float64)
    _ = value_and_grad_fn(theta_init)

    def objective_with_grad(theta):
        """Returns (f_val, grad, x) where x is the latent parameters."""
        theta_jax = jnp.asarray(theta, dtype=jnp.float64)
        (f_val, x_val), grad_val = value_and_grad_fn(theta_jax)
        return float(f_val), np.asarray(grad_val, dtype=np.float64), np.asarray(x_val, dtype=np.float64)

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

    # Check if we have a spatio-temporal model (use sparse A)
    has_spatio_temporal = any(
        hasattr(sm, 'submodel_type') and sm.submodel_type == 'spatio_temporal'
        for sm in model.submodels
    )

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

    # For spatio-temporal models, keep A sparse and precompute AtA in BTA format
    if has_spatio_temporal:
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

        # Precompute A^T @ A as sparse and extract BTA blocks
        ata_scipy = a_scipy.T @ a_scipy
        ata_diag, ata_lower, ata_arrow, ata_tip = extract_bta_blocks_from_sparse(
            ata_scipy, nt, ns, n_fixed_effects
        )

        # Store sparse A for A^T @ y operations
        a_sparse = _scipy_sparse_to_jax_bcoo(a_scipy)

        static_data = {
            'likelihood_type': likelihood_type,
            'a_sparse': a_sparse,
            'ata_diag_blocks': ata_diag,
            'ata_lower_blocks': ata_lower,
            'ata_arrow_blocks': ata_arrow,
            'ata_tip_block': ata_tip,
            'y': jnp.array(_to_numpy(model.y), dtype=jnp.float64),
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
                'c0': jnp.array(_to_numpy(st_submodel.c0.toarray()), dtype=jnp.float64),
                'g1': jnp.array(_to_numpy(st_submodel.g1.toarray()), dtype=jnp.float64),
                'g2': jnp.array(_to_numpy(st_submodel.g2.toarray()), dtype=jnp.float64),
                'g3': jnp.array(_to_numpy(st_submodel.g3.toarray()), dtype=jnp.float64),
            },
            'temporal_matrices': {
                'm0': jnp.array(_to_numpy(st_submodel.m0.toarray()), dtype=jnp.float64),
                'm1': jnp.array(_to_numpy(st_submodel.m1.toarray()), dtype=jnp.float64),
                'm2': jnp.array(_to_numpy(st_submodel.m2.toarray()), dtype=jnp.float64),
            },
        }
    else:
        # For dense solver, store A as dense
        a_matrix = _to_numpy(model.a.toarray()) if hasattr(model.a, 'toarray') else _to_numpy(model.a)
        static_data = {
            'likelihood_type': likelihood_type,
            'a': jnp.array(a_matrix, dtype=jnp.float64),
            'y': jnp.array(_to_numpy(model.y), dtype=jnp.float64),
            'n_fixed_effects': int(model.n_fixed_effects),
            'fixed_effects_precision': float(fixed_effects_precision),
            'prior_configs': prior_configs,
            'n_observations': int(model.n_observations),
            'n_latent_parameters': int(model.n_latent_parameters),
            'inner_iter_tol': float(dalia_instance.config.eps_inner_iteration),
            'inner_iter_max': int(dalia_instance.config.inner_iteration_max_iter),
            'use_sparse_solver': False,
        }

    if likelihood_type == 'poisson':
        if hasattr(model.likelihood, 'e'):
            e = _to_numpy(model.likelihood.e)
        else:
            e = np.ones(model.n_observations)
        static_data['e'] = jnp.array(e, dtype=jnp.float64)

    elif likelihood_type == 'binomial':
        if hasattr(model.likelihood, 'n_trials'):
            n_trials = _to_numpy(model.likelihood.n_trials)
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

    Q_prior = jnp.eye(n_fixed_effects) * fixed_effects_precision

    log_prior_hyperparameters = 0.0

    grad_fn = lambda eta: _gradient_poisson_likelihood_jax(eta, y, e)
    hess_fn = lambda eta: _hessian_diag_poisson_jax(eta, e)

    Q_conditional, x, eta = _inner_iteration_jax(
        a, y, Q_prior, grad_fn, hess_fn, tol, max_iter
    )

    log_likelihood = _evaluate_poisson_likelihood_jax(eta, y, e)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior - 0.5 * x.T @ Q_prior @ x

    # For non-Gaussian, DALIA uses x=None, x_mean=None -> quadratic_form=0
    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
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

    Q_prior = jnp.eye(n_fixed_effects) * fixed_effects_precision

    log_prior_hyperparameters = 0.0

    grad_fn = lambda eta: _gradient_binomial_likelihood_jax(eta, y, n_trials)
    hess_fn = lambda eta: _hessian_diag_binomial_jax(eta, n_trials)

    Q_conditional, x, eta = _inner_iteration_jax(
        a, y, Q_prior, grad_fn, hess_fn, tol, max_iter
    )

    log_likelihood = _evaluate_binomial_likelihood_jax(eta, y, n_trials)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior - 0.5 * x.T @ Q_prior @ x

    # For non-Gaussian, DALIA uses x=None, x_mean=None -> quadratic_form=0
    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
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
    """Pure JAX objective function for Gaussian likelihood with sparse serinv solver.

    Uses block-tridiagonal-arrowhead structure directly, avoiding dense matrices.

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
    a_sparse = static_data['a_sparse']
    prior_configs = static_data['prior_configs']

    # Precomputed AtA BTA blocks
    ata_diag = static_data['ata_diag_blocks']
    ata_lower = static_data['ata_lower_blocks']
    ata_arrow = static_data['ata_arrow_blocks']
    ata_tip = static_data['ata_tip_block']

    theta_st = theta[:-1]
    theta_likelihood = theta[-1]

    # Build Q_st directly in BTA block format (memory efficient)
    q_st_diag, q_st_lower = build_spatio_temporal_Q_bta_jax(
        theta_st, spatial_matrices, temporal_matrices, manifold
    )

    # Build Q_conditional blocks:
    # Q_conditional = Q_prior - A^T @ D @ A
    # where Q_prior has Q_st in spatio-temporal part and fixed_effects_precision on diagonal
    # and D = -exp(theta_likelihood) * I
    likelihood_precision = jnp.exp(theta_likelihood)

    # Diagonal blocks: Q_st_diag - (-prec * AtA_diag) = Q_st_diag + prec * AtA_diag
    diag_blocks = q_st_diag + likelihood_precision * ata_diag

    # Lower diagonal blocks
    lower_diag_blocks = q_st_lower + likelihood_precision * ata_lower

    # Arrow blocks (Q_prior has zeros here, only AtA contributes)
    lower_arrow_blocks = likelihood_precision * ata_arrow

    # Arrow tip: fixed_effects_precision * I + prec * AtA_tip
    arrow_tip = fixed_effects_precision * jnp.eye(n_fixed_effects) + likelihood_precision * ata_tip

    # Cholesky factorization in BTA format
    diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip = pobtaf_jax_optimized(
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip
    )

    # Log determinant from Cholesky factors (vectorized)
    logdet_Q_conditional = compute_logdet_from_cholesky_bta_jax(
        diag_blocks, arrow_tip
    )

    # Solve for x using BTA system
    gradient_likelihood = likelihood_precision * y
    rhs = a_sparse.T @ gradient_likelihood

    x = solve_bta_system_jax(
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip, rhs
    )

    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    eta = jnp.zeros_like(y)
    log_likelihood = _evaluate_gaussian_likelihood_jax(eta, y, theta_likelihood)

    # Compute log(det(Q_st)) using BT Cholesky with lax.fori_loop
    q_st_diag_copy = q_st_diag.copy()
    q_st_lower_copy = q_st_lower.copy()

    def bt_chol_body(i, carry):
        diag_b, lower_b = carry
        L_i = jnp.linalg.cholesky(diag_b[i])
        diag_b = diag_b.at[i].set(L_i)

        # Update lower and next diagonal (masked for last iteration)
        lower_updated = jnp.linalg.solve(L_i, lower_b[i].T).T
        lower_b = lower_b.at[i].set(lower_updated)

        # Only update next diagonal if not at last block
        next_diag = diag_b[i + 1] - lower_updated @ lower_updated.T
        # Use where to conditionally update (avoid out of bounds)
        diag_b = lax.cond(
            i < nt - 1,
            lambda d: d.at[i + 1].set(next_diag),
            lambda d: d,
            diag_b
        )
        return (diag_b, lower_b)

    q_st_diag_copy, q_st_lower_copy = lax.fori_loop(
        0, nt, bt_chol_body, (q_st_diag_copy, q_st_lower_copy)
    )

    # Log det from BT Cholesky (vectorized)
    diag_vals_st = jnp.diagonal(q_st_diag_copy, axis1=1, axis2=2)
    logdet_Q_st = 2.0 * jnp.sum(jnp.log(diag_vals_st))
    log_prior_latent = 0.5 * logdet_Q_st

    # Quadratic form x^T Q_conditional x using BTA blocks (vectorized)
    q_cond_diag = q_st_diag + likelihood_precision * ata_diag
    q_cond_lower = q_st_lower + likelihood_precision * ata_lower
    q_cond_arrow = likelihood_precision * ata_arrow
    q_cond_tip = fixed_effects_precision * jnp.eye(n_fixed_effects) + likelihood_precision * ata_tip

    x_st = x[:nt * ns].reshape(nt, ns)
    x_fe = x[nt * ns:]

    # Diagonal contribution: sum_i x_st[i] @ q_cond_diag[i] @ x_st[i]
    # Using einsum: 'bi,bij,bj->b' then sum
    quad_diag = jnp.einsum('bi,bij,bj->', x_st, q_cond_diag, x_st)

    # Lower diagonal contribution: 2 * sum_i x_st[i+1] @ q_cond_lower[i] @ x_st[i]
    # x_st[1:] @ q_cond_lower @ x_st[:-1]
    quad_lower = 2.0 * jnp.einsum('bi,bij,bj->', x_st[1:], q_cond_lower, x_st[:-1])

    # Arrow contribution: 2 * sum_i x_fe @ q_cond_arrow[i] @ x_st[i]
    # = 2 * x_fe @ (sum_i q_cond_arrow[i] @ x_st[i])
    arrow_matvec = jnp.einsum('bij,bj->i', q_cond_arrow, x_st)
    quad_arrow = 2.0 * jnp.dot(x_fe, arrow_matvec)

    # Arrow tip contribution
    quad_tip = x_fe @ q_cond_tip @ x_fe

    quad_form = quad_diag + quad_lower + quad_arrow + quad_tip

    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quad_form

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x
