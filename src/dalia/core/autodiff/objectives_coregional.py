# Copyright 2024-2025 DALIA authors. All rights reserved.

import jax
import jax.numpy as jnp
from jax import lax

from dalia.core.autodiff.config import (
    bta_to_dense_jax,
    _evaluate_gaussian_likelihood_jax,
    _evaluate_log_prior_hyperparameters_jax,
)
from dalia.core.autodiff.q_construction import (
    build_coregional_Q_bta_jax,
    compute_logdet_from_cholesky_bta_jax,
    solve_bta_system_jax,
    quadratic_form_bta_jax,
)
from dalia.core.autodiff.spatial_precompute import (
    _jax_cholesky,
    precompute_spatial_components,
    precompute_spatial_components_coregional,
    _reconstruct_coregional_diag_block,
    _reconstruct_coregional_lower_block,
)
from dalia.core.autodiff.coregional_solvers import (
    fused_cholesky_fwd_sub_coregional,
    backward_sub_from_carries_coregional,
    logdet_Q_prior_coregional_scan,
    logdet_Q_prior_coregional_grad,
    selected_inversion_grads_from_carries_coregional,
    _compute_grad_quad_coregional,
)
from serinv.algs.pobtaf_jax import pobtaf_jax_optimized
from serinv.algs.pobtf_jax import pobtf_logdet_jax


def _objective_gaussian_coregional_scan_baseline(theta, static_data, checkpoint=False):
    """Pure JAX objective for coregional Gaussian — no custom VJP.

    Uses ``_pobtaf_impl`` (``lax.fori_loop``) instead of
    ``pobtaf_jax_optimized`` (which has a ``custom_vjp``), so JAX AD
    differentiates through the loop body directly.

    Parameters
    ----------
    theta : jnp.ndarray
        Full hyperparameter vector.
    static_data : dict
        Static data from :func:`_extract_static_data_coregional`.
    checkpoint : bool
        If True, wrap the fori_loop body with ``jax.checkpoint``.

    Returns
    -------
    objective : scalar
    x : jnp.ndarray
    """
    from serinv.algs.pobtaf_jax import _pobtaf_impl

    n_models = static_data['n_models']
    nt = static_data['nt']
    ns = static_data['ns']
    block_size = static_data['block_size']
    n_fixed_effects_total = static_data['n_fixed_effects_total']
    fixed_effects_precision = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a_sparse = static_data['a_sparse']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']

    ata_diag_per_model = static_data['ata_diag_per_model']
    ata_lower_per_model = static_data['ata_lower_per_model']
    ata_arrow_per_model = static_data['ata_arrow_per_model']
    ata_tip_per_model = static_data['ata_tip_per_model']

    q_prior_diag, q_prior_lower = build_coregional_Q_bta_jax(
        theta, n_models, ns, nt, models_data, hyperparameters_idx, theta_keys
    )

    likelihood_precisions = jnp.zeros(n_models, dtype=y.dtype)
    for i in range(n_models):
        prec_idx = hyperparameters_idx[i + 1] - 1
        likelihood_precisions = likelihood_precisions.at[i].set(jnp.exp(theta[prec_idx]))

    weighted_ata_diag = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_diag_per_model)
    weighted_ata_lower = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_lower_per_model)
    weighted_ata_arrow = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_arrow_per_model)
    weighted_ata_tip = jnp.einsum('m,mij->ij', likelihood_precisions, ata_tip_per_model)

    q_cond_diag = q_prior_diag + weighted_ata_diag
    q_cond_lower = q_prior_lower + weighted_ata_lower
    q_cond_arrow = weighted_ata_arrow
    q_cond_tip = fixed_effects_precision * jnp.eye(n_fixed_effects_total, dtype=y.dtype) + weighted_ata_tip

    if q_cond_diag.dtype == jnp.float32:
        eps_reg = 1e-4
        identity_block = jnp.eye(q_cond_diag.shape[-1], dtype=q_cond_diag.dtype)
        q_cond_diag = q_cond_diag + eps_reg * identity_block[None, :, :]
        q_cond_tip = q_cond_tip + eps_reg * jnp.eye(n_fixed_effects_total, dtype=q_cond_tip.dtype)

    if checkpoint:
        from serinv.algs.pobtaf_jax import _pobtaf_impl as _pobtaf_base
        L_diag, L_lower, L_arrow, L_tip = jax.checkpoint(
            _pobtaf_base, prevent_cse=True)(
            q_cond_diag, q_cond_lower, q_cond_arrow, q_cond_tip)
    else:
        L_diag, L_lower, L_arrow, L_tip = _pobtaf_impl(
            q_cond_diag, q_cond_lower, q_cond_arrow, q_cond_tip)

    logdet_Q_conditional = compute_logdet_from_cholesky_bta_jax(L_diag, L_tip)

    gradient_likelihood = jnp.zeros_like(y)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
            likelihood_precisions[i] * y[obs_start:obs_end]
        )
    rhs = a_sparse.T @ gradient_likelihood

    x = solve_bta_system_jax(L_diag, L_lower, L_arrow, L_tip, rhs)

    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        y_i = y[obs_start:obs_end]
        eta_i = eta[obs_start:obs_end]
        prec_idx = hyperparameters_idx[i + 1] - 1
        theta_lik_i = theta[prec_idx]
        log_likelihood += _evaluate_gaussian_likelihood_jax(eta_i, y_i, theta_lik_i)

    logdet_Q_prior_st = pobtf_logdet_jax(q_prior_diag, q_prior_lower)
    log_prior_latent = 0.5 * logdet_Q_prior_st

    quad_form = quadratic_form_bta_jax(
        L_diag, L_lower, L_arrow, L_tip, x, nt, block_size)

    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quad_form

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_gaussian_coregional_sparse(theta, static_data):
    """Pure JAX objective function for Gaussian CoregionalModel with sparse serinv solver.

    Uses block-tridiagonal-arrowhead structure with coregional block size (n_models * ns).

    inputs:
    theta : Hyperparameters arranged as:
            [model_0_params..., model_1_params..., ..., sigmas..., lambdas...]
    static_data : Static data extracted from CoregionalModel

    Returns:
    objective : INLA objective value
    x : Latent parameters
    """
    n_models = static_data['n_models']
    nt = static_data['nt']
    ns = static_data['ns']
    block_size = static_data['block_size']
    n_fixed_effects_total = static_data['n_fixed_effects_total']
    fixed_effects_precision = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a_sparse = static_data['a_sparse']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']

    # Per-model AtA contributions: shape (n_models, n_blocks, block_size, block_size)
    ata_diag_per_model = static_data['ata_diag_per_model']
    ata_lower_per_model = static_data['ata_lower_per_model']
    ata_arrow_per_model = static_data['ata_arrow_per_model']
    ata_tip_per_model = static_data['ata_tip_per_model']

    # Build Q_prior in BTA format using coregional structure
    q_prior_diag, q_prior_lower = build_coregional_Q_bta_jax(
        theta, n_models, ns, nt, models_data, hyperparameters_idx, theta_keys
    )

    # Compute likelihood precisions for each model
    likelihood_precisions = jnp.zeros(n_models, dtype=y.dtype)
    for i in range(n_models):
        prec_idx = hyperparameters_idx[i + 1] - 1
        likelihood_precisions = likelihood_precisions.at[i].set(jnp.exp(theta[prec_idx]))

    # Compute weighted AtDA contribution: sum_i(prec_i * A_i^T @ A_i)
    # Using einsum to weight per-model contributions by their precisions
    # ata_diag_per_model: (n_models, n_blocks, block_size, block_size)
    # likelihood_precisions: (n_models,)
    weighted_ata_diag = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_diag_per_model)
    weighted_ata_lower = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_lower_per_model)
    weighted_ata_arrow = jnp.einsum('m,mbij->bij', likelihood_precisions, ata_arrow_per_model)
    weighted_ata_tip = jnp.einsum('m,mij->ij', likelihood_precisions, ata_tip_per_model)

    # Build Q_conditional blocks (save for quadratic form computation later)
    q_cond_diag = q_prior_diag + weighted_ata_diag
    q_cond_lower = q_prior_lower + weighted_ata_lower
    q_cond_arrow = weighted_ata_arrow
    q_cond_tip = fixed_effects_precision * jnp.eye(n_fixed_effects_total, dtype=y.dtype) + weighted_ata_tip

    # Prepare blocks for Cholesky
    diag_blocks = q_cond_diag
    lower_diag_blocks = q_cond_lower
    lower_arrow_blocks = q_cond_arrow
    arrow_tip = q_cond_tip

    # Add small diagonal regularization for FP32 to improve numerical stability
    if diag_blocks.dtype == jnp.float32:
        eps_reg = 1e-4
        identity_block = jnp.eye(diag_blocks.shape[-1], dtype=diag_blocks.dtype)
        diag_blocks = diag_blocks + eps_reg * identity_block[None, :, :]
        arrow_tip = arrow_tip + eps_reg * jnp.eye(n_fixed_effects_total, dtype=arrow_tip.dtype)

    # Cholesky factorization in BTA format
    L_diag, L_lower, L_arrow, L_tip = pobtaf_jax_optimized(
        diag_blocks, lower_diag_blocks, lower_arrow_blocks, arrow_tip
    )

    # Log determinant from Cholesky factors
    logdet_Q_conditional = compute_logdet_from_cholesky_bta_jax(L_diag, L_tip)

    # Solve for x using BTA system
    # RHS = A^T @ D @ y where D is diagonal with per-model precisions
    # Build weighted gradient: D @ y
    gradient_likelihood = jnp.zeros_like(y)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
            likelihood_precisions[i] * y[obs_start:obs_end]
        )
    rhs = a_sparse.T @ gradient_likelihood

    x = solve_bta_system_jax(L_diag, L_lower, L_arrow, L_tip, rhs)

    # Evaluate log prior on hyperparameters
    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    # Evaluate likelihood (using eta=0 simplification for Gaussian at mode)
    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        y_i = y[obs_start:obs_end]
        eta_i = eta[obs_start:obs_end]
        prec_idx = hyperparameters_idx[i + 1] - 1
        theta_lik_i = theta[prec_idx]
        log_likelihood += _evaluate_gaussian_likelihood_jax(eta_i, y_i, theta_lik_i)

    # Log prior for latent parameters - compute logdet of Q_prior
    logdet_Q_prior_st = pobtf_logdet_jax(q_prior_diag, q_prior_lower)
    log_prior_latent = 0.5 * logdet_Q_prior_st

    # Quadratic form x^T Q_conditional x using Cholesky factors L
    #
    # MEMORY OPTIMIZATION: Instead of computing x^T @ Q @ x directly (which requires
    # keeping the Q_conditional blocks q_cond_diag, q_cond_lower, q_cond_arrow, q_cond_tip
    # in memory), we use the identity:
    #
    #     x^T @ Q @ x = x^T @ L @ L^T @ x = ||L^T @ x||^2
    #
    # where L is the Cholesky factor we already computed (L_diag, L_lower, L_arrow, L_tip).
    #
    # This saves ~60 GB for gst_large scale problems by allowing the Q_conditional blocks
    # to be freed after the Cholesky factorization.
    #
    # Note: For coregional models, block_size = n_models * ns (spatial nodes per model).
    quad_form = quadratic_form_bta_jax(
        L_diag, L_lower, L_arrow, L_tip, x, nt, block_size
    )

    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quad_form

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _build_coregional_Q_prior_spatial_dense(
    theta: jnp.ndarray,
    n_models: int,
    ns: int,
    models_data: list,
    hyperparameters_idx: list,
    theta_keys: list,
) -> jnp.ndarray:
    """Build coregional Q_prior for spatial-only models as dense matrix.

    For 2-model case:
        Q_11 = (1/sigma_0^2)*Qu_0 + (lambda_01^2/sigma_1^2)*Qu_1
        Q_12 = -(lambda_01/sigma_1^2)*Qu_1
        Q_21 = -(lambda_01/sigma_1^2)*Qu_1
        Q_22 = (1/sigma_1^2)*Qu_1

    For 3-model case, similar pattern with more terms.
    """
    block_size = n_models * ns

    # Build individual Qu matrices for each model
    Qu_list = []
    for i in range(n_models):
        model_data = models_data[i]
        hp_start = hyperparameters_idx[i]
        hp_end = hyperparameters_idx[i + 1] - 1

        theta_model = theta[hp_start:hp_end]

        # For spatial models, theta_model contains [r_s]
        r_s = theta_model[0]
        sigma_e = 0.0  # Fixed at 0 for spatial coregional

        # Build spatial precision matrix
        c0 = model_data['spatial_matrices']['c0']
        g1 = model_data['spatial_matrices']['g1']
        g2 = model_data['spatial_matrices']['g2']

        alpha = 2.0
        dim_spatial_domain = 2.0
        nu_s = alpha - dim_spatial_domain / 2.0
        gamma_s = 0.5 * jnp.log(8.0 * nu_s) - r_s
        log_gamma_nu_s = jax.scipy.special.gammaln(nu_s)
        log_gamma_alpha = jax.scipy.special.gammaln(alpha)
        gamma_e = 0.5 * (log_gamma_nu_s - (
            log_gamma_alpha + 0.5 * dim_spatial_domain * jnp.log(4.0 * jnp.pi)
            + 2.0 * nu_s * gamma_s + 2.0 * sigma_e
        ))

        exp_gamma_s = jnp.exp(gamma_s)
        exp_gamma_e = jnp.exp(gamma_e)

        q2s = jnp.power(exp_gamma_s, 4) * c0 + 2.0 * jnp.power(exp_gamma_s, 2) * g1 + g2
        Qu = jnp.power(exp_gamma_e, 2) * q2s
        Qu_list.append(Qu)

    # Extract sigmas and lambdas from theta
    sigma_idx = theta_keys.index('sigma_0')
    sigmas = []
    for i in range(n_models):
        sigmas.append(jnp.exp(theta[sigma_idx + i]))

    lambda_01_idx = theta_keys.index('lambda_0_1')
    lambda_01 = theta[lambda_01_idx]

    if n_models == 3:
        lambda_02_idx = theta_keys.index('lambda_0_2')
        lambda_12_idx = theta_keys.index('lambda_1_2')
        lambda_02 = theta[lambda_02_idx]
        lambda_12 = theta[lambda_12_idx]

    # Build coregional Q_prior matrix
    Q_prior = jnp.zeros((block_size, block_size), dtype=theta.dtype)

    if n_models == 2:
        sigma_0, sigma_1 = sigmas[0], sigmas[1]

        coef_11_0 = 1.0 / (sigma_0 ** 2)
        coef_11_1 = (lambda_01 ** 2) / (sigma_1 ** 2)
        coef_12 = -lambda_01 / (sigma_1 ** 2)
        coef_22 = 1.0 / (sigma_1 ** 2)

        q11 = coef_11_0 * Qu_list[0] + coef_11_1 * Qu_list[1]
        q12 = coef_12 * Qu_list[1]
        q22 = coef_22 * Qu_list[1]

        Q_prior = Q_prior.at[:ns, :ns].set(q11)
        Q_prior = Q_prior.at[:ns, ns:].set(q12)
        Q_prior = Q_prior.at[ns:, :ns].set(q12)  # Q_21 = Q_12
        Q_prior = Q_prior.at[ns:, ns:].set(q22)

    elif n_models == 3:
        sigma_0, sigma_1, sigma_2 = sigmas[0], sigmas[1], sigmas[2]

        coef_11_0 = 1.0 / (sigma_0 ** 2)
        coef_11_1 = (lambda_01 ** 2) / (sigma_1 ** 2)
        coef_11_2 = (lambda_12 ** 2) / (sigma_2 ** 2)

        coef_21_1 = -lambda_01 / (sigma_1 ** 2)
        coef_21_2 = (lambda_02 * lambda_12) / (sigma_2 ** 2)

        coef_31 = -lambda_12 / (sigma_2 ** 2)

        coef_22_1 = 1.0 / (sigma_1 ** 2)
        coef_22_2 = (lambda_02 ** 2) / (sigma_2 ** 2)

        coef_32 = -lambda_02 / (sigma_2 ** 2)

        coef_33 = 1.0 / (sigma_2 ** 2)

        q11 = coef_11_0 * Qu_list[0] + coef_11_1 * Qu_list[1] + coef_11_2 * Qu_list[2]
        q21 = coef_21_1 * Qu_list[1] + coef_21_2 * Qu_list[2]
        q31 = coef_31 * Qu_list[2]
        q22 = coef_22_1 * Qu_list[1] + coef_22_2 * Qu_list[2]
        q32 = coef_32 * Qu_list[2]
        q33 = coef_33 * Qu_list[2]

        Q_prior = Q_prior.at[:ns, :ns].set(q11)
        Q_prior = Q_prior.at[ns:2*ns, :ns].set(q21)
        Q_prior = Q_prior.at[:ns, ns:2*ns].set(q21)  # Symmetric
        Q_prior = Q_prior.at[2*ns:, :ns].set(q31)
        Q_prior = Q_prior.at[:ns, 2*ns:].set(q31)  # Symmetric
        Q_prior = Q_prior.at[ns:2*ns, ns:2*ns].set(q22)
        Q_prior = Q_prior.at[2*ns:, ns:2*ns].set(q32)
        Q_prior = Q_prior.at[ns:2*ns, 2*ns:].set(q32)  # Symmetric
        Q_prior = Q_prior.at[2*ns:, 2*ns:].set(q33)

    return Q_prior


def _objective_gaussian_coregional_spatial_dense(theta, static_data):
    """Pure JAX objective for Gaussian spatial CoregionalModel with dense solver.

    Uses dense matrix operations for spatial-only coregional models.
    """
    n_models = static_data['n_models']
    ns = static_data['ns']
    n_fixed_effects_total = static_data['n_fixed_effects_total']
    fixed_effects_precision = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a = static_data['a']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']
    n_latent = static_data['n_latent_parameters']

    # Build coregional Q_prior for spatial fields
    Q_prior_spatial = _build_coregional_Q_prior_spatial_dense(
        theta, n_models, ns, models_data, hyperparameters_idx, theta_keys
    )

    # Build full Q_prior (coregional spatial + fixed effects)
    Q_prior = jnp.zeros((n_latent, n_latent), dtype=y.dtype)
    n_spatial = n_models * ns
    Q_prior = Q_prior.at[:n_spatial, :n_spatial].set(Q_prior_spatial)

    if n_fixed_effects_total > 0:
        Q_fe = jnp.eye(n_fixed_effects_total, dtype=y.dtype) * fixed_effects_precision
        Q_prior = Q_prior.at[n_spatial:, n_spatial:].set(Q_fe)

    # Compute likelihood precisions for each model
    likelihood_precisions = []
    for i in range(n_models):
        prec_idx = hyperparameters_idx[i + 1] - 1
        likelihood_precisions.append(jnp.exp(theta[prec_idx]))

    # Build D diagonal (per-observation precision)
    n_obs = len(y)
    D_diag = jnp.zeros(n_obs, dtype=y.dtype)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        D_diag = D_diag.at[obs_start:obs_end].set(-likelihood_precisions[i])

    # Q_conditional = Q_prior - A^T @ D @ A
    Q_conditional = Q_prior - (a.T * D_diag) @ a

    # Compute RHS = A^T @ (prec * y)
    gradient_likelihood = jnp.zeros_like(y)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
            likelihood_precisions[i] * y[obs_start:obs_end]
        )
    rhs = a.T @ gradient_likelihood

    # Solve for x
    x = jnp.linalg.solve(Q_conditional, rhs)

    # Evaluate log prior on hyperparameters
    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    # Evaluate likelihood (using eta=0 simplification)
    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        y_i = y[obs_start:obs_end]
        eta_i = eta[obs_start:obs_end]
        prec_idx = hyperparameters_idx[i + 1] - 1
        theta_lik_i = theta[prec_idx]
        log_likelihood += _evaluate_gaussian_likelihood_jax(eta_i, y_i, theta_lik_i)

    # Log prior for latent parameters
    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior

    # Log conditional
    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    quad_form = x.T @ Q_conditional @ x
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quad_form

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_gaussian_coregional_st_dense(theta, static_data):
    """Pure JAX objective for Gaussian spatio-temporal CoregionalModel with dense solver.

    Builds the coregional Q_prior in BTA format, converts to dense, then uses
    standard dense linear algebra. This is used as a baseline to compare against
    the structure-preserving sparse solver.
    """
    n_models = static_data['n_models']
    nt = static_data['nt']
    ns = static_data['ns']
    block_size = static_data['block_size']
    n_fixed_effects_total = static_data['n_fixed_effects_total']
    fixed_effects_precision = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a = static_data['a']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']
    n_latent = static_data['n_latent_parameters']

    # Build coregional Q_prior in BTA format
    q_prior_diag, q_prior_lower = build_coregional_Q_bta_jax(
        theta, n_models, ns, nt, models_data, hyperparameters_idx, theta_keys
    )

    # Convert block-tridiagonal to dense
    n_st = nt * block_size
    Q_prior_st = jnp.zeros((n_st, n_st), dtype=y.dtype)
    for t in range(nt):
        s = t * block_size
        e = s + block_size
        Q_prior_st = Q_prior_st.at[s:e, s:e].set(q_prior_diag[t])
    for t in range(nt - 1):
        s1 = (t + 1) * block_size
        e1 = s1 + block_size
        s0 = t * block_size
        e0 = s0 + block_size
        Q_prior_st = Q_prior_st.at[s1:e1, s0:e0].set(q_prior_lower[t])
        Q_prior_st = Q_prior_st.at[s0:e0, s1:e1].set(q_prior_lower[t].T)

    # Build full Q_prior (coregional ST + fixed effects)
    Q_prior = jnp.zeros((n_latent, n_latent), dtype=y.dtype)
    Q_prior = Q_prior.at[:n_st, :n_st].set(Q_prior_st)
    if n_fixed_effects_total > 0:
        Q_fe = jnp.eye(n_fixed_effects_total, dtype=y.dtype) * fixed_effects_precision
        Q_prior = Q_prior.at[n_st:, n_st:].set(Q_fe)

    # Compute likelihood precisions for each model
    likelihood_precisions = []
    for i in range(n_models):
        prec_idx = hyperparameters_idx[i + 1] - 1
        likelihood_precisions.append(jnp.exp(theta[prec_idx]))

    # Build D diagonal (per-observation precision)
    n_obs = len(y)
    D_diag = jnp.zeros(n_obs, dtype=y.dtype)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        D_diag = D_diag.at[obs_start:obs_end].set(-likelihood_precisions[i])

    Q_conditional = Q_prior - (a.T * D_diag) @ a

    # Compute RHS = A^T @ (prec * y)
    gradient_likelihood = jnp.zeros_like(y)
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
            likelihood_precisions[i] * y[obs_start:obs_end]
        )
    rhs = a.T @ gradient_likelihood

    x = jnp.linalg.solve(Q_conditional, rhs)

    log_prior_hyperparameters = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for i in range(n_models):
        obs_start = n_observations_idx[i]
        obs_end = n_observations_idx[i + 1]
        y_i = y[obs_start:obs_end]
        eta_i = eta[obs_start:obs_end]
        prec_idx = hyperparameters_idx[i + 1] - 1
        theta_lik_i = theta[prec_idx]
        log_likelihood += _evaluate_gaussian_likelihood_jax(eta_i, y_i, theta_lik_i)

    _, logdet_Q_prior = jnp.linalg.slogdet(Q_prior)
    log_prior_latent = 0.5 * logdet_Q_prior

    _, logdet_Q_conditional = jnp.linalg.slogdet(Q_conditional)
    quad_form = x.T @ Q_conditional @ x
    log_conditional = 0.5 * logdet_Q_conditional - 0.5 * quad_form

    objective = -(
        log_prior_hyperparameters
        + log_likelihood
        + log_prior_latent
        - log_conditional
    )

    return objective, x


def _objective_gaussian_coregional_sparse_fused(theta, static_data):
    """Memory-efficient JAX objective for coregional Gaussian with custom_vjp.

    Uses carry-based Cholesky (no L storage), per-model sparse COO AtA,
    and on-the-fly super-block reconstruction to fit within GPU memory.

    Parameters
    ----------
    theta : jnp.ndarray
        Full hyperparameter vector.
    static_data : dict
        Static data from :func:`_extract_static_data_coregional`.

    Returns
    -------
    objective : scalar
    x : jnp.ndarray
    """
    n_models = static_data['n_models']
    nt = static_data['nt']
    ns = static_data['ns']
    block_size = static_data['block_size']
    n_fe = static_data['n_fixed_effects_total']
    fe_prec = static_data['fixed_effects_precision']
    models_data = static_data['models_data']
    y = static_data['y']
    a_sparse = static_data['a_sparse']
    prior_configs = static_data['prior_configs']
    hyperparameters_idx = static_data['hyperparameters_idx']
    theta_keys = static_data['theta_keys']
    n_observations_idx = static_data['n_observations_idx']

    # Per-model sparse COO data
    per_model_ata_diag_rows = static_data['per_model_ata_diag_rows']
    per_model_ata_diag_cols = static_data['per_model_ata_diag_cols']
    per_model_ata_diag_vals = static_data['per_model_ata_diag_vals']
    per_model_ata_lower_rows = static_data['per_model_ata_lower_rows']
    per_model_ata_lower_cols = static_data['per_model_ata_lower_cols']
    per_model_ata_lower_vals = static_data['per_model_ata_lower_vals']
    per_model_ata_arrow_rows = static_data['per_model_ata_arrow_rows']
    per_model_ata_arrow_cols = static_data['per_model_ata_arrow_cols']
    per_model_ata_arrow_vals = static_data['per_model_ata_arrow_vals']
    per_model_ata_tip = static_data['per_model_ata_tip']
    per_model_offsets = static_data['per_model_offsets']

    manifolds = [md.get('manifold', 'plane') for md in models_data]
    dtype = y.dtype

    # Count hyperparameter groups
    n_theta_st_per_model = 3
    n_theta_lik_per_model = 1

    # Identify coregional parameter indices
    sigma_idx = theta_keys.index('sigma_0')
    n_sigmas = n_models
    lambda_keys = [k for k in theta_keys if k.startswith('lambda_')]
    n_lambdas = len(lambda_keys)
    n_coreg_params = n_sigmas + n_lambdas

    @jax.custom_vjp
    def fused_core_coregional(theta_full):
        # Split theta
        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_full[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_full, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        # Build RHS
        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood
        rhs_st = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]

        stored_cs, stored_as, y_st, L_tip, arrow_rhs_acc, logdet_cond = \
            fused_cholesky_fwd_sub_coregional(
                sc_list, coreg_w, n_models, nt, ns, n_fe, fe_prec,
                likelihood_precs,
                rhs_st, rhs_fe,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype)

        x, quad = backward_sub_from_carries_coregional(
            stored_cs, stored_as, L_tip,
            y_st, arrow_rhs_acc,
            sc_list, coreg_w, likelihood_precs,
            per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
            per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
            per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
            per_model_offsets,
            n_models, nt, ns, n_fe, dtype)

        # Pad sc_list subdiags for logdet scan
        sc_padded = []
        for m in range(n_models):
            sc_m = sc_list[m]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        logdet_prior = logdet_Q_prior_coregional_scan(
            sc_padded, coreg_w, n_models, ns, nt, dtype)

        return logdet_prior, logdet_cond, quad, x

    def fused_core_fwd(theta_full):
        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_full[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_full, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood
        rhs_st = rhs[:nt * block_size].reshape(nt, block_size)
        rhs_fe = rhs[nt * block_size:]

        stored_cs, stored_as, y_st, L_tip, arrow_rhs_acc, logdet_cond = \
            fused_cholesky_fwd_sub_coregional(
                sc_list, coreg_w, n_models, nt, ns, n_fe, fe_prec,
                likelihood_precs,
                rhs_st, rhs_fe,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype)

        x, quad = backward_sub_from_carries_coregional(
            stored_cs, stored_as, L_tip,
            y_st, arrow_rhs_acc,
            sc_list, coreg_w, likelihood_precs,
            per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
            per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
            per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
            per_model_offsets,
            n_models, nt, ns, n_fe, dtype)

        sc_padded = []
        for m in range(n_models):
            sc_m = sc_list[m]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        logdet_prior = logdet_Q_prior_coregional_scan(
            sc_padded, coreg_w, n_models, ns, nt, dtype)

        residuals = (theta_full, x, stored_cs, stored_as, L_tip)
        return (logdet_prior, logdet_cond, quad, x), residuals

    def fused_core_bwd(residuals, g):
        bar_logdet_prior, bar_logdet_cond, bar_quad, _bar_x = g
        theta_r, x_r, stored_cs_r, stored_as_r, L_tip_r = residuals

        likelihood_precs = jnp.zeros(n_models, dtype=dtype)
        for m in range(n_models):
            prec_idx = hyperparameters_idx[m + 1] - 1
            likelihood_precs = likelihood_precs.at[m].set(jnp.exp(theta_r[prec_idx]))

        sc_list, coreg_w = precompute_spatial_components_coregional(
            theta_r, n_models, ns, models_data, hyperparameters_idx,
            theta_keys, manifolds)

        # Jacobians of spatial components per model
        jac_sc_list = []
        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            theta_m = theta_r[hp_start:hp_end]
            if theta_m.shape[0] == 2:
                theta_m = jnp.concatenate([theta_m, jnp.array([0.0])])
            jac_m = jax.jacfwd(precompute_spatial_components)(
                theta_m, models_data[m]['spatial_matrices'],
                models_data[m]['temporal_matrices'], manifolds[m])
            jac_sc_list.append(jac_m)

        # Jacobian of coreg_w w.r.t. coregional params (sigmas + lambdas)
        def _coreg_w_from_params(coreg_params):
            """Extract coreg_w from just the coregional parameters."""
            w = jnp.zeros((n_models, n_models, n_models), dtype=dtype)
            sigmas_raw = coreg_params[:n_sigmas]
            sigs = jnp.exp(sigmas_raw)
            if n_models == 2:
                lam01 = coreg_params[n_sigmas]
                s0, s1 = sigs[0], sigs[1]
                w = w.at[0, 0, 0].set(1.0 / s0**2)
                w = w.at[0, 0, 1].set(lam01**2 / s1**2)
                w = w.at[1, 0, 1].set(-lam01 / s1**2)
                w = w.at[0, 1, 1].set(-lam01 / s1**2)
                w = w.at[1, 1, 1].set(1.0 / s1**2)
            elif n_models == 3:
                lam01 = coreg_params[n_sigmas]
                lam02 = coreg_params[n_sigmas + 1]
                lam12 = coreg_params[n_sigmas + 2]
                s0, s1, s2 = sigs[0], sigs[1], sigs[2]
                w = w.at[0, 0, 0].set(1.0 / s0**2)
                w = w.at[0, 0, 1].set(lam01**2 / s1**2)
                w = w.at[0, 0, 2].set(lam12**2 / s2**2)
                w = w.at[1, 0, 1].set(-lam01 / s1**2)
                w = w.at[0, 1, 1].set(-lam01 / s1**2)
                w = w.at[1, 0, 2].set(lam02 * lam12 / s2**2)
                w = w.at[0, 1, 2].set(lam02 * lam12 / s2**2)
                w = w.at[2, 0, 2].set(-lam12 / s2**2)
                w = w.at[0, 2, 2].set(-lam12 / s2**2)
                w = w.at[1, 1, 1].set(1.0 / s1**2)
                w = w.at[1, 1, 2].set(lam02**2 / s2**2)
                w = w.at[2, 1, 2].set(-lam02 / s2**2)
                w = w.at[1, 2, 2].set(-lam02 / s2**2)
                w = w.at[2, 2, 2].set(1.0 / s2**2)
            return w

        # Build coreg_params vector
        coreg_params = jnp.zeros(n_coreg_params, dtype=dtype)
        for m in range(n_models):
            coreg_params = coreg_params.at[m].set(theta_r[sigma_idx + m])
        for li, lk in enumerate(lambda_keys):
            coreg_params = coreg_params.at[n_sigmas + li].set(
                theta_r[theta_keys.index(lk)])

        jac_coreg_w = jax.jacfwd(_coreg_w_from_params)(coreg_params)

        # Pad sc_list subdiags
        sc_padded = []
        for m in range(n_models):
            sc_m = sc_list[m]
            sc_padded.append({
                **sc_m,
                'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
            })

        # --- Phase A: logdet_cond gradient via selected inversion ---
        grad_cond_st, grad_cond_lik, grad_cond_coreg = \
            selected_inversion_grads_from_carries_coregional(
                stored_cs_r, stored_as_r, L_tip_r,
                sc_padded, jac_sc_list, coreg_w, jac_coreg_w,
                n_models, nt, ns, n_fe,
                likelihood_precs,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets,
                dtype)

        # --- Phase B: quadratic form gradient ---
        # Rebuild rhs
        gradient_likelihood = jnp.zeros_like(y)
        for m in range(n_models):
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            gradient_likelihood = gradient_likelihood.at[obs_start:obs_end].set(
                likelihood_precs[m] * y[obs_start:obs_end])
        rhs = a_sparse.T @ gradient_likelihood

        grad_quad_st, grad_quad_lik_placeholder, grad_quad_coreg = \
            _compute_grad_quad_coregional(
                x_r, sc_list, jac_sc_list, coreg_w, jac_coreg_w,
                n_models, nt, ns, n_fe,
                rhs, likelihood_precs,
                per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
                per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
                per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
                per_model_ata_tip, per_model_offsets)

        # d(quad)/d(prec_m) = prec_m * (2*x^T A_m^T y_m - x^T AtA_m x)
        #                   = prec_m * eta_m^T (2*y_m - eta_m)   where eta_m = A_m x
        # The naive formula subtracts two O(1e9) scalars to get an O(1e3) result,
        # losing ~6 digits to cancellation. The eta formulation avoids this.
        eta = a_sparse @ x_r
        grad_quad_lik = jnp.zeros(n_models, dtype=dtype)

        for m in range(n_models):
            prec_m = likelihood_precs[m]
            obs_start = n_observations_idx[m]
            obs_end = n_observations_idx[m + 1]
            eta_m = eta[obs_start:obs_end]
            y_m = y[obs_start:obs_end]
            grad_quad_lik = grad_quad_lik.at[m].set(
                prec_m * jnp.dot(eta_m, 2.0 * y_m - eta_m))

        # --- Phase C: logdet_prior gradient ---
        jac_sc_list_padded = []
        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            theta_m = theta_r[hp_start:hp_end]
            if theta_m.shape[0] == 2:
                theta_m = jnp.concatenate([theta_m, jnp.array([0.0])])
            jac_m = jax.jacfwd(precompute_spatial_components)(
                theta_m, models_data[m]['spatial_matrices'],
                models_data[m]['temporal_matrices'], manifolds[m])
            # Pad the Jacobian subdiags
            jac_padded = {}
            for key, val in jac_m.items():
                if key.endswith('_subdiag'):
                    jac_padded[key] = jnp.concatenate([val, jnp.zeros((1,) + val.shape[1:], dtype=dtype)], axis=0)
                else:
                    jac_padded[key] = val
            jac_sc_list_padded.append(jac_padded)

        grad_prior_st, grad_prior_coreg = logdet_Q_prior_coregional_grad(
            sc_padded, jac_sc_list_padded, coreg_w, jac_coreg_w,
            n_models, ns, nt, dtype)

        # --- Combine into full gradient ---
        n_theta = theta_r.shape[0]
        bar_theta = jnp.zeros(n_theta, dtype=dtype)

        for m in range(n_models):
            hp_start = hyperparameters_idx[m]
            hp_end = hyperparameters_idx[m + 1] - 1
            n_st_m = hp_end - hp_start

            grad_st_m = (
                bar_logdet_prior * grad_prior_st[m][:n_st_m]
                + bar_logdet_cond * grad_cond_st[m][:n_st_m]
                + bar_quad * grad_quad_st[m][:n_st_m]
            )
            bar_theta = bar_theta.at[hp_start:hp_end].set(grad_st_m)

            # Likelihood param
            prec_idx = hyperparameters_idx[m + 1] - 1
            grad_lik_m = (
                bar_logdet_cond * grad_cond_lik[m]
                + bar_quad * grad_quad_lik[m]
            )
            bar_theta = bar_theta.at[prec_idx].set(grad_lik_m)

        # Coregional params
        grad_coreg_total = (
            bar_logdet_prior * grad_prior_coreg
            + bar_logdet_cond * grad_cond_coreg
            + bar_quad * grad_quad_coreg
        )
        for m in range(n_models):
            bar_theta = bar_theta.at[sigma_idx + m].add(grad_coreg_total[m])
        for li, lk in enumerate(lambda_keys):
            bar_theta = bar_theta.at[theta_keys.index(lk)].add(
                grad_coreg_total[n_sigmas + li])

        return (bar_theta,)

    fused_core_coregional.defvjp(fused_core_fwd, fused_core_bwd)

    # --- Assemble objective ---
    logdet_prior, logdet_cond, quad_form, x = fused_core_coregional(theta)

    log_prior_hp = _evaluate_log_prior_hyperparameters_jax(theta, prior_configs)

    eta = jnp.zeros_like(y)
    log_likelihood = 0.0
    for m in range(n_models):
        obs_start = n_observations_idx[m]
        obs_end = n_observations_idx[m + 1]
        prec_idx = hyperparameters_idx[m + 1] - 1
        log_likelihood += _evaluate_gaussian_likelihood_jax(
            eta[obs_start:obs_end], y[obs_start:obs_end], theta[prec_idx])

    objective = -(
        log_prior_hp
        + log_likelihood
        + 0.5 * logdet_prior
        - 0.5 * logdet_cond
        + 0.5 * quad_form
    )

    return objective, x
