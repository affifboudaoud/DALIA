# Copyright 2024-2025 DALIA authors. All rights reserved.

from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jax_sparse
from scipy import sparse as scipy_sparse


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
