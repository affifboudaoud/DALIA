# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Callable, Tuple, Dict, Any
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def create_pure_jax_objective(dalia_instance) -> Tuple[Callable, Callable]:
    """Create pure JAX objective function with automatic differentiation.

    For now, This only works for:
    - Gaussian likelihood
    - Dense solver
    - Single-process (no MPI)

    inputs:
    dalia_instance : DALIA instance.

    Returns:
    objective_func : Pure JAX objective function.
    objective_with_grad : Function returning both forward value and gradient.
    """
    static_data = _extract_static_data(dalia_instance)

    def objective_pure_jax(theta):
        """Pure JAX objective function"""
        return _objective_gaussian_dense(theta, static_data)

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
    static_data : Dictionary containing:
        - a: Design matrix (numpy array)
        - y: Observations (numpy array)
        - n_fixed_effects: Number of fixed effects
        - fixed_effects_precision: Prior precision for fixed effects
        - pc_prior_alpha: PC prior alpha parameter
        - pc_prior_u: PC prior u parameter
        - n_observations: Number of observations
    """
    model = dalia_instance.model

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
        'a': jnp.array(a_matrix, dtype=jnp.float64),
        'y': jnp.array(np.array(model.y), dtype=jnp.float64),
        'n_fixed_effects': int(model.n_fixed_effects),
        'fixed_effects_precision': float(fixed_effects_precision),
        'pc_prior_alpha': float(pc_prior_alpha),
        'pc_prior_u': float(pc_prior_u),
        'n_observations': int(model.n_observations),
    }

    return static_data


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

    yEta = eta - y
    log_likelihood = (
        0.5 * theta_likelihood * len(y)
        - 0.5 * jnp.exp(theta_likelihood) * (yEta.T @ yEta)
    )

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
