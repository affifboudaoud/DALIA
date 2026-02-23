
# Copyright 2024-2025 DALIA authors. All rights reserved.
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax import lax
from typing import Tuple
from functools import partial


def kronecker_to_bta_structure(
    kron_result: jnp.ndarray,
    nt: int,
    ns: int,
    n_fixed_effects: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert Kronecker product result to block-tridiagonal-arrowhead format.

    The spatio-temporal precision matrix from kron(M, S) has a natural
    block-tridiagonal structure when M is tridiagonal (temporal AR(1) structure).

    inputs: kron_result : Result of Kronecker products, shape (nt*ns + n_fixed_effects, nt*ns + n_fixed_effects)
    nt :   Number of temporal blocks
    ns :   Size of each spatial block
    n_fixed_effects :   Size of fixed effects (arrowhead)
    Returns:
    diagonal_blocks :   Shape (nt, ns, ns)
    lower_diagonal_blocks :   Shape (nt-1, ns, ns)
    lower_arrow_blocks :   Shape (nt, n_fixed_effects, ns) if n_fixed_effects > 0, else zeros
    arrow_tip_block :   Shape (n_fixed_effects, n_fixed_effects) if n_fixed_effects > 0, else zeros
    """
    total_st_size = nt * ns
    dtype = kron_result.dtype

    diagonal_blocks = jnp.zeros((nt, ns, ns), dtype=dtype)
    lower_diagonal_blocks = jnp.zeros((nt - 1, ns, ns), dtype=dtype)

    for i in range(nt):
        start_i = i * ns
        end_i = (i + 1) * ns
        diagonal_blocks = diagonal_blocks.at[i].set(
            kron_result[start_i:end_i, start_i:end_i]
        )

        if i < nt - 1:
            start_ip1 = (i + 1) * ns
            end_ip1 = (i + 2) * ns
            lower_diagonal_blocks = lower_diagonal_blocks.at[i].set(
                kron_result[start_ip1:end_ip1, start_i:end_i]
            )

    if n_fixed_effects > 0:
        lower_arrow_blocks = jnp.zeros((nt, n_fixed_effects, ns), dtype=dtype)
        for i in range(nt):
            start_i = i * ns
            end_i = (i + 1) * ns
            lower_arrow_blocks = lower_arrow_blocks.at[i].set(
                kron_result[total_st_size:, start_i:end_i]
            )

        arrow_tip_block = kron_result[total_st_size:, total_st_size:]
    else:
        lower_arrow_blocks = jnp.zeros((nt, 1, ns), dtype=dtype)
        arrow_tip_block = jnp.zeros((1, 1), dtype=dtype)

    return diagonal_blocks, lower_diagonal_blocks, lower_arrow_blocks, arrow_tip_block


def _interpretable_to_compute_jax(r_s, r_t, sigma_st, manifold="plane"):
    """Transform interpretable parameters to computational parameters.

    Matches DALIA's SpatioTemporalSubModel._interpretable2compute.

    Inputs:
    r_s : Spatial range parameter (log-scale)
    r_t : Temporal range parameter (log-scale)
    sigma_st : Spatio-temporal marginal std (log-scale)
    manifold : Either "sphere" or "plane"

    Returns:
    gamma_s, gamma_t, gamma_st : Computational parameters
    """
    import jax.scipy.special as jax_special

    alpha_s = 2
    alpha_t = 1
    alpha_e = 1

    alpha = alpha_e + alpha_s * (alpha_t - 0.5)

    nu_s = alpha - 1
    nu_t = alpha_t - 0.5

    gamma_s = 0.5 * jnp.log(8 * nu_s) - r_s
    gamma_t = r_t - 0.5 * jnp.log(8 * nu_t) + alpha_s * gamma_s

    if manifold == "sphere":
        # Use JAX's gammaln for precision consistency: gamma(x) = exp(gammaln(x))
        log_cR_t = jax_special.gammaln(nu_t) - jax_special.gammaln(alpha_t) - 0.5 * jnp.log(4.0 * jnp.pi)
        # Vectorized sum: k = 0, 1, ..., 49
        k_vals = jnp.arange(50)
        exp_gamma_s_sq = jnp.exp(gamma_s) ** 2
        c_s = jnp.sum(
            (2.0 * k_vals + 1.0) / (4.0 * jnp.pi * jnp.power(exp_gamma_s_sq + k_vals * (k_vals + 1), alpha))
        )
        gamma_st = 0.5 * log_cR_t + 0.5 * jnp.log(c_s) - 0.5 * gamma_t - sigma_st
    elif manifold == "plane":
        # Use JAX's gammaln: log(c1) = gammaln(nu_t) + gammaln(nu_s) - gammaln(alpha_t) - gammaln(alpha) - 1.5*log(4*pi)
        log_c1 = (
            jax_special.gammaln(nu_t) + jax_special.gammaln(nu_s)
            - jax_special.gammaln(alpha_t) - jax_special.gammaln(alpha)
            - 1.5 * jnp.log(4.0 * jnp.pi)
        )
        gamma_st = 0.5 * log_c1 - 0.5 * gamma_t - nu_s * gamma_s - sigma_st
    else:
        raise ValueError(f"Manifold not supported: {manifold}")

    return gamma_s, gamma_t, gamma_st


def build_spatio_temporal_Q_jax(
    theta: jnp.ndarray,
    spatial_matrices: dict,
    temporal_matrices: dict,
    manifold: str = "sphere",
) -> jnp.ndarray:
    """Build spatio-temporal precision matrix using JAX operations.

    Implements the INLA spatio-temporal precision matrix construction:
    Q = exp(gamma_st)^2 * (kron(m0, q3s) + exp(gamma_t)*kron(m1, q2s) + exp(gamma_t)^2*kron(m2, q1s))

    inputs:
    theta : Hyperparameters [r_s, r_t, sigma_st] (interpretable parameters)
    spatial_matrices : Dictionary with keys 'c0', 'g1', 'g2', 'g3' (spatial FEM matrices)
    temporal_matrices : Dictionary with keys 'm0', 'm1', 'm2' (temporal FEM matrices)
    manifold : Either "sphere" or "plane"

    Returns:
    Q : Spatio-temporal precision matrix
    """
    r_s = theta[0]
    r_t = theta[1]
    sigma_st = theta[2]

    gamma_s, gamma_t, gamma_st = _interpretable_to_compute_jax(r_s, r_t, sigma_st, manifold)

    c0 = spatial_matrices['c0']
    g1 = spatial_matrices['g1']
    g2 = spatial_matrices['g2']
    g3 = spatial_matrices['g3']

    m0 = temporal_matrices['m0']
    m1 = temporal_matrices['m1']
    m2 = temporal_matrices['m2']

    exp_gamma_s = jnp.exp(gamma_s)
    exp_gamma_t = jnp.exp(gamma_t)
    exp_gamma_st = jnp.exp(gamma_st)

    q1s = exp_gamma_s**2 * c0 + g1
    q2s = exp_gamma_s**4 * c0 + 2 * exp_gamma_s**2 * g1 + g2
    q3s = exp_gamma_s**6 * c0 + 3 * exp_gamma_s**4 * g1 + 3 * exp_gamma_s**2 * g2 + g3

    Q = exp_gamma_st**2 * (
        jnp.kron(m0, q3s) +
        exp_gamma_t * jnp.kron(m1, q2s) +
        exp_gamma_t**2 * jnp.kron(m2, q1s)
    )

    return Q


def build_spatio_temporal_Q_bta_jax(
    theta: jnp.ndarray,
    spatial_matrices: dict,
    temporal_matrices: dict,
    manifold: str = "sphere",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build spatio-temporal precision matrix directly in BTA block format.

    This avoids creating the full dense Kronecker product by exploiting the
    block-tridiagonal structure of kron(M_tridiag, S).

    For kron(M, S) where M is tridiagonal:
    - Diagonal blocks[i] = M[i,i] * S
    - Lower diagonal blocks[i] = M[i+1,i] * S

    inputs:
    theta : Hyperparameters [r_s, r_t, sigma_st]
    spatial_matrices : Dict with 'c0', 'g1', 'g2', 'g3'
    temporal_matrices : Dict with 'm0', 'm1', 'm2'
    manifold : "sphere" or "plane"

    Returns:
    diag_blocks : Shape (nt, ns, ns) - diagonal blocks of Q_st
    lower_diag_blocks : Shape (nt-1, ns, ns) - lower diagonal blocks of Q_st
    """
    r_s = theta[0]
    r_t = theta[1]
    sigma_st = theta[2]

    gamma_s, gamma_t, gamma_st = _interpretable_to_compute_jax(r_s, r_t, sigma_st, manifold)

    c0 = spatial_matrices['c0']
    g1 = spatial_matrices['g1']
    g2 = spatial_matrices['g2']
    g3 = spatial_matrices['g3']

    m0 = temporal_matrices['m0']
    m1 = temporal_matrices['m1']
    m2 = temporal_matrices['m2']

    exp_gamma_s = jnp.exp(gamma_s)
    exp_gamma_t = jnp.exp(gamma_t)
    exp_gamma_st = jnp.exp(gamma_st)

    # Build spatial precision matrices
    q1s = exp_gamma_s**2 * c0 + g1
    q2s = exp_gamma_s**4 * c0 + 2 * exp_gamma_s**2 * g1 + g2
    q3s = exp_gamma_s**6 * c0 + 3 * exp_gamma_s**4 * g1 + 3 * exp_gamma_s**2 * g2 + g3

    # Scale factor
    scale = exp_gamma_st**2

    # Extract diagonals of temporal matrices - shape (nt,)
    m0_diag = jnp.diag(m0)
    m1_diag = jnp.diag(m1)
    m2_diag = jnp.diag(m2)

    # Build diagonal blocks vectorized: diag_blocks[i] = scale * (m0[i,i]*q3s + ...)
    # Shape: (nt, 1, 1) * (ns, ns) -> (nt, ns, ns) via broadcasting
    diag_blocks = scale * (
        m0_diag[:, None, None] * q3s[None, :, :] +
        exp_gamma_t * m1_diag[:, None, None] * q2s[None, :, :] +
        exp_gamma_t**2 * m2_diag[:, None, None] * q1s[None, :, :]
    )

    # Extract sub-diagonals of temporal matrices - shape (nt-1,)
    # m[i+1, i] for i in range(nt-1) is the first sub-diagonal
    m0_subdiag = jnp.diag(m0, k=-1)
    m1_subdiag = jnp.diag(m1, k=-1)
    m2_subdiag = jnp.diag(m2, k=-1)

    # Build lower diagonal blocks vectorized
    lower_diag_blocks = scale * (
        m0_subdiag[:, None, None] * q3s[None, :, :] +
        exp_gamma_t * m1_subdiag[:, None, None] * q2s[None, :, :] +
        exp_gamma_t**2 * m2_subdiag[:, None, None] * q1s[None, :, :]
    )

    return diag_blocks, lower_diag_blocks


def extract_bta_blocks_from_sparse(
    sparse_matrix,
    nt: int,
    ns: int,
    n_fixed_effects: int,
    dtype=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract BTA blocks from a scipy sparse matrix.

    inputs:
    sparse_matrix : scipy sparse matrix (CSR/CSC/COO)
    nt : Number of temporal blocks
    ns : Size of each spatial block
    n_fixed_effects : Size of arrow tip
    dtype : JAX dtype for output arrays (default: infer from sparse_matrix)

    Returns:
    diag_blocks : (nt, ns, ns)
    lower_diag_blocks : (nt-1, ns, ns)
    arrow_bottom_blocks : (nt, n_fixed_effects, ns)
    arrow_tip : (n_fixed_effects, n_fixed_effects)
    """
    from scipy import sparse as sp_sparse
    from dalia.core.jax_autodiff import get_jax_dtype
    csc = sp_sparse.csc_matrix(sparse_matrix)

    if dtype is None:
        dtype = get_jax_dtype()

    total_st = nt * ns

    diag_blocks = jnp.zeros((nt, ns, ns), dtype=dtype)
    lower_diag_blocks = jnp.zeros((nt - 1, ns, ns), dtype=dtype)
    arrow_bottom_blocks = jnp.zeros((nt, n_fixed_effects, ns), dtype=dtype)
    arrow_tip = jnp.zeros((n_fixed_effects, n_fixed_effects), dtype=dtype)

    # Extract diagonal blocks
    for i in range(nt):
        start = i * ns
        end = (i + 1) * ns
        block = csc[start:end, start:end].toarray()
        diag_blocks = diag_blocks.at[i].set(jnp.array(block, dtype=dtype))

    # Extract lower diagonal blocks
    for i in range(nt - 1):
        row_start = (i + 1) * ns
        row_end = (i + 2) * ns
        col_start = i * ns
        col_end = (i + 1) * ns
        block = csc[row_start:row_end, col_start:col_end].toarray()
        lower_diag_blocks = lower_diag_blocks.at[i].set(jnp.array(block, dtype=dtype))

    # Extract arrow bottom blocks
    if n_fixed_effects > 0:
        for i in range(nt):
            col_start = i * ns
            col_end = (i + 1) * ns
            block = csc[total_st:, col_start:col_end].toarray()
            arrow_bottom_blocks = arrow_bottom_blocks.at[i].set(jnp.array(block, dtype=dtype))

        # Extract arrow tip
        arrow_tip = jnp.array(csc[total_st:, total_st:].toarray(), dtype=dtype)

    return diag_blocks, lower_diag_blocks, arrow_bottom_blocks, arrow_tip


def build_Q_conditional_jax(
    Q_st: jnp.ndarray,
    fixed_effects_precision: float,
    n_fixed_effects: int,
    nt: int,
    ns: int,
) -> jnp.ndarray:
    """Build Q_conditional by adding fixed effects precision.

    inputs:
    Q_st : Spatio-temporal precision matrix (nt*ns, nt*ns)
    fixed_effects_precision : Prior precision for fixed effects
    n_fixed_effects : Number of fixed effects
    nt : Number of temporal blocks
    ns : Number of spatial nodes

    Returns:
    Q_conditional : Full precision matrix with fixed effects block
    """
    total_st_size = nt * ns
    total_size = total_st_size + n_fixed_effects
    dtype = Q_st.dtype

    Q_conditional = jnp.zeros((total_size, total_size), dtype=dtype)

    Q_conditional = Q_conditional.at[:total_st_size, :total_st_size].set(Q_st)

    Q_fe = jnp.eye(n_fixed_effects, dtype=dtype) * fixed_effects_precision
    Q_conditional = Q_conditional.at[total_st_size:, total_st_size:].set(Q_fe)

    return Q_conditional


def compute_logdet_from_cholesky_bta_jax(
    diagonal_blocks: jnp.ndarray,
    arrow_tip_block: jnp.ndarray,
) -> float:
    """Compute log-determinant from Cholesky factorization in BTA format.

    log(det(Q)) = 2 * sum(log(diag(L)))

    Uses safe log computation to prevent NaN when Cholesky diagonal elements
    become non-positive due to FP32 precision loss. Non-positive values are
    clamped to a small epsilon, which effectively returns a large penalty.

    inputs:
        diagonal_blocks : Diagonal blocks of Cholesky factor, shape (n_blocks, block_size, block_size)
        arrow_tip_block : Arrow tip block of Cholesky factor, shape (arrow_size, arrow_size)
    Returns:
        logdet : Log-determinant of the precision matrix
    """
    # Extract diagonals from all blocks at once using vmap
    # diagonal_blocks has shape (n_blocks, block_size, block_size)
    # We want the diagonal of each block
    diag_vals = jnp.diagonal(diagonal_blocks, axis1=1, axis2=2)  # (n_blocks, block_size)

    # Safe log: clamp to small positive value to prevent NaN from non-positive values
    # This can happen in FP32 when Cholesky factorization loses precision
    eps = jnp.finfo(diag_vals.dtype).eps
    safe_diag_vals = jnp.maximum(diag_vals, eps)
    logdet = 2.0 * jnp.sum(jnp.log(safe_diag_vals))

    # Add arrow tip contribution if present
    arrow_diag = jnp.diag(arrow_tip_block)
    safe_arrow_diag = jnp.maximum(arrow_diag, eps)
    logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_arrow_diag))

    return logdet


def solve_bta_system_jax(
    diagonal_blocks: jnp.ndarray,
    lower_diagonal_blocks: jnp.ndarray,
    lower_arrow_blocks: jnp.ndarray,
    arrow_tip_block: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve Q*x = rhs using Cholesky factors in BTA format.

    Performs L * L^T * x = rhs by:
    1. Forward solve: L * y = rhs
    2. Backward solve: L^T * x = y

    inputs:
        diagonal_blocks : Cholesky diagonal blocks
        lower_diagonal_blocks : Cholesky lower diagonal blocks
        lower_arrow_blocks : Cholesky lower arrow blocks
        arrow_tip_block : Cholesky arrow tip block
        rhs : Right-hand side vector

    Returns:
        x : Solution vector
    """
    from serinv.algs.pobtas_jax import pobtas_jax_optimized

    # Forward solve
    result = pobtas_jax_optimized(
        diagonal_blocks,
        lower_diagonal_blocks,
        lower_arrow_blocks,
        arrow_tip_block,
        rhs,
        trans="N",
    )

    # Backward solve
    result = pobtas_jax_optimized(
        diagonal_blocks,
        lower_diagonal_blocks,
        lower_arrow_blocks,
        arrow_tip_block,
        result,
        trans="C",
    )

    return result


def quadratic_form_bta_jax(
    L_diag: jnp.ndarray,
    L_lower: jnp.ndarray,
    L_arrow: jnp.ndarray,
    L_tip: jnp.ndarray,
    x: jnp.ndarray,
    nt: int,
    ns: int,
) -> float:
    """Compute x^T @ Q @ x using Cholesky factors L where Q = L @ L^T.

    This exploits the identity: x^T @ Q @ x = x^T @ L @ L^T @ x = ||L^T @ x||^2

    This is more memory-efficient than computing the quadratic form directly
    with Q, because we can reuse the Cholesky factors L that were already
    computed for the solve step, avoiding the need to keep Q in memory.

    For the BTA (Block-Tridiagonal-Arrowhead) structure:
        L = [L_00                              ]
            [L_10  L_11                        ]
            [      L_21  L_22                  ]
            [            ...                   ]
            [                  L_{n-1,n-1}     ]
            [L_a0  L_a1  ...  L_{a,n-1}  L_aa  ]

    The transpose L^T is upper triangular, and z = L^T @ x gives:
        z_i = L_ii^T @ x_i + L_{i+1,i}^T @ x_{i+1} + L_ai^T @ x_fe  (for i < n-1)
        z_{n-1} = L_{n-1,n-1}^T @ x_{n-1} + L_{a,n-1}^T @ x_fe
        z_fe = L_aa^T @ x_fe

    Then: x^T @ Q @ x = ||z||^2 = sum_i ||z_i||^2 + ||z_fe||^2

    Memory savings: This avoids keeping the full Q_conditional blocks (~60 GB
    for gst_large) in memory after Cholesky factorization.

    Parameters
    ----------
    L_diag : (nt, ns, ns) Cholesky diagonal blocks
    L_lower : (nt-1, ns, ns) Cholesky lower diagonal blocks
    L_arrow : (nt, n_fe, ns) Cholesky arrow blocks
    L_tip : (n_fe, n_fe) Cholesky arrow tip block
    x : (nt*ns + n_fe,) Solution vector
    nt : Number of temporal blocks
    ns : Size of each spatial block

    Returns
    -------
    quad_form : Scalar value of x^T @ Q @ x
    """
    n_fe = L_tip.shape[0]

    # Reshape x into temporal blocks and fixed effects
    x_st = x[:nt * ns].reshape(nt, ns)  # (nt, ns)
    x_fe = x[nt * ns:]  # (n_fe,)

    # Compute z = L^T @ x block by block
    # z_i = L_diag[i]^T @ x_st[i] + L_lower[i]^T @ x_st[i+1] + L_arrow[i]^T @ x_fe

    # Diagonal contribution: L_diag[i]^T @ x_st[i] for all i
    # Using einsum: 'bji,bj->bi' (transpose via swapped indices)
    z_diag = jnp.einsum('bji,bj->bi', L_diag, x_st)  # (nt, ns)

    # Lower diagonal contribution: L_lower[i]^T @ x_st[i+1] for i=0..nt-2
    # This contributes to z_0..z_{nt-2}
    z_lower_contrib = jnp.einsum('bji,bj->bi', L_lower, x_st[1:])  # (nt-1, ns)

    # Arrow contribution: L_arrow[i]^T @ x_fe for all i
    # L_arrow is (nt, n_fe, ns), x_fe is (n_fe,)
    # L_arrow[i]^T is (ns, n_fe), so L_arrow[i]^T @ x_fe is (ns,)
    z_arrow_contrib = jnp.einsum('bji,j->bi', L_arrow, x_fe)  # (nt, ns)

    # Combine contributions for z_st
    # z[i] = z_diag[i] + (z_lower_contrib[i] if i < nt-1 else 0) + z_arrow_contrib[i]
    z_st = z_diag + z_arrow_contrib  # (nt, ns)
    # Add lower diagonal contribution to first nt-1 blocks
    z_st = z_st.at[:-1].add(z_lower_contrib)

    # Arrow tip contribution: z_fe = L_tip^T @ x_fe
    z_fe = L_tip.T @ x_fe  # (n_fe,)

    # Compute ||z||^2 = sum_i ||z_st[i]||^2 + ||z_fe||^2
    quad_form = jnp.sum(z_st ** 2) + jnp.sum(z_fe ** 2)

    return quad_form


def build_coregional_Q_bta_jax(
    theta: jnp.ndarray,
    n_models: int,
    ns: int,
    nt: int,
    models_data: list,
    hyperparameters_idx: list,
    theta_keys: list,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build coregional Q_prior directly in BTA format.

    The coregional Q has block structure where each "super-block"
    contains contributions from all models according to coregionalization math.

    For 2-model case:
        Q_11 = (1/sigma_0^2)*Qu_0 + (lambda_01^2/sigma_1^2)*Qu_1
        Q_12 = -(lambda_01/sigma_1^2)*Qu_1
        Q_21 = -(lambda_01/sigma_1^2)*Qu_1
        Q_22 = (1/sigma_1^2)*Qu_1

    For 3-model case, similar pattern with more terms.

    inputs:
    theta : All hyperparameters
    n_models : Number of models (2 or 3)
    ns : Number of spatial nodes
    nt : Number of temporal nodes
    models_data : List of dicts with per-model matrices
    hyperparameters_idx : List of hyperparameter index boundaries
    theta_keys : List of hyperparameter names

    Returns:
    diag_blocks : Shape (nt, n_models*ns, n_models*ns)
    lower_diag_blocks : Shape (nt-1, n_models*ns, n_models*ns)
    """
    block_size = n_models * ns

    # Build individual Qu matrices for each model in BTA format
    Qu_diag_list = []
    Qu_lower_list = []

    for i in range(n_models):
        model_data = models_data[i]
        hp_start = hyperparameters_idx[i]
        hp_end = hyperparameters_idx[i + 1] - 1

        theta_model = theta[hp_start:hp_end]

        # Coregional models remove sigma_st from each model's params
        # Add default sigma_st=0.0 if only r_s, r_t are present
        if theta_model.shape[0] == 2:
            theta_model = jnp.concatenate([theta_model, jnp.array([0.0])])

        qu_diag, qu_lower = build_spatio_temporal_Q_bta_jax(
            theta_model,
            model_data['spatial_matrices'],
            model_data['temporal_matrices'],
            model_data['manifold'],
        )
        Qu_diag_list.append(qu_diag)
        Qu_lower_list.append(qu_lower)

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

    # Build coregional super-blocks - infer dtype from input theta
    dtype = theta.dtype
    diag_blocks = jnp.zeros((nt, block_size, block_size), dtype=dtype)
    lower_diag_blocks = jnp.zeros((nt - 1, block_size, block_size), dtype=dtype)

    if n_models == 2:
        sigma_0, sigma_1 = sigmas[0], sigmas[1]

        # Q_11 block: (1/sigma_0^2)*Qu_0 + (lambda_01^2/sigma_1^2)*Qu_1
        coef_11_0 = 1.0 / (sigma_0 ** 2)
        coef_11_1 = (lambda_01 ** 2) / (sigma_1 ** 2)

        # Q_12/Q_21 block: -(lambda_01/sigma_1^2)*Qu_1
        coef_12 = -lambda_01 / (sigma_1 ** 2)

        # Q_22 block: (1/sigma_1^2)*Qu_1
        coef_22 = 1.0 / (sigma_1 ** 2)

        for t in range(nt):
            block = jnp.zeros((block_size, block_size), dtype=dtype)

            q11 = coef_11_0 * Qu_diag_list[0][t] + coef_11_1 * Qu_diag_list[1][t]
            q12 = coef_12 * Qu_diag_list[1][t]
            q21 = coef_12 * Qu_diag_list[1][t]
            q22 = coef_22 * Qu_diag_list[1][t]

            block = block.at[:ns, :ns].set(q11)
            block = block.at[:ns, ns:].set(q12)
            block = block.at[ns:, :ns].set(q21)
            block = block.at[ns:, ns:].set(q22)

            diag_blocks = diag_blocks.at[t].set(block)

        for t in range(nt - 1):
            block = jnp.zeros((block_size, block_size), dtype=dtype)

            q11 = coef_11_0 * Qu_lower_list[0][t] + coef_11_1 * Qu_lower_list[1][t]
            q12 = coef_12 * Qu_lower_list[1][t]
            q21 = coef_12 * Qu_lower_list[1][t]
            q22 = coef_22 * Qu_lower_list[1][t]

            block = block.at[:ns, :ns].set(q11)
            block = block.at[:ns, ns:].set(q12)
            block = block.at[ns:, :ns].set(q21)
            block = block.at[ns:, ns:].set(q22)

            lower_diag_blocks = lower_diag_blocks.at[t].set(block)

    elif n_models == 3:
        sigma_0, sigma_1, sigma_2 = sigmas[0], sigmas[1], sigmas[2]

        # Q_11: (1/sigma_0^2)*Qu_0 + (lambda_01^2/sigma_1^2)*Qu_1 + (lambda_12^2/sigma_2^2)*Qu_2
        coef_11_0 = 1.0 / (sigma_0 ** 2)
        coef_11_1 = (lambda_01 ** 2) / (sigma_1 ** 2)
        coef_11_2 = (lambda_12 ** 2) / (sigma_2 ** 2)

        # Q_21: -(lambda_01/sigma_1^2)*Qu_1 + (lambda_02*lambda_12/sigma_2^2)*Qu_2
        coef_21_1 = -lambda_01 / (sigma_1 ** 2)
        coef_21_2 = (lambda_02 * lambda_12) / (sigma_2 ** 2)

        # Q_31: -(lambda_12/sigma_2^2)*Qu_2
        coef_31 = -lambda_12 / (sigma_2 ** 2)

        # Q_22: (1/sigma_1^2)*Qu_1 + (lambda_02^2/sigma_2^2)*Qu_2
        coef_22_1 = 1.0 / (sigma_1 ** 2)
        coef_22_2 = (lambda_02 ** 2) / (sigma_2 ** 2)

        # Q_32: -(lambda_02/sigma_2^2)*Qu_2
        coef_32 = -lambda_02 / (sigma_2 ** 2)

        # Q_33: (1/sigma_2^2)*Qu_2
        coef_33 = 1.0 / (sigma_2 ** 2)

        for t in range(nt):
            block = jnp.zeros((block_size, block_size), dtype=dtype)

            q11 = coef_11_0 * Qu_diag_list[0][t] + coef_11_1 * Qu_diag_list[1][t] + coef_11_2 * Qu_diag_list[2][t]
            q21 = coef_21_1 * Qu_diag_list[1][t] + coef_21_2 * Qu_diag_list[2][t]
            q31 = coef_31 * Qu_diag_list[2][t]
            q22 = coef_22_1 * Qu_diag_list[1][t] + coef_22_2 * Qu_diag_list[2][t]
            q32 = coef_32 * Qu_diag_list[2][t]
            q33 = coef_33 * Qu_diag_list[2][t]

            block = block.at[:ns, :ns].set(q11)
            block = block.at[:ns, ns:2*ns].set(q21.T)
            block = block.at[:ns, 2*ns:].set(q31.T)
            block = block.at[ns:2*ns, :ns].set(q21)
            block = block.at[ns:2*ns, ns:2*ns].set(q22)
            block = block.at[ns:2*ns, 2*ns:].set(q32.T)
            block = block.at[2*ns:, :ns].set(q31)
            block = block.at[2*ns:, ns:2*ns].set(q32)
            block = block.at[2*ns:, 2*ns:].set(q33)

            diag_blocks = diag_blocks.at[t].set(block)

        for t in range(nt - 1):
            block = jnp.zeros((block_size, block_size), dtype=dtype)

            q11 = coef_11_0 * Qu_lower_list[0][t] + coef_11_1 * Qu_lower_list[1][t] + coef_11_2 * Qu_lower_list[2][t]
            q21 = coef_21_1 * Qu_lower_list[1][t] + coef_21_2 * Qu_lower_list[2][t]
            q31 = coef_31 * Qu_lower_list[2][t]
            q22 = coef_22_1 * Qu_lower_list[1][t] + coef_22_2 * Qu_lower_list[2][t]
            q32 = coef_32 * Qu_lower_list[2][t]
            q33 = coef_33 * Qu_lower_list[2][t]

            block = block.at[:ns, :ns].set(q11)
            block = block.at[:ns, ns:2*ns].set(q21.T)
            block = block.at[:ns, 2*ns:].set(q31.T)
            block = block.at[ns:2*ns, :ns].set(q21)
            block = block.at[ns:2*ns, ns:2*ns].set(q22)
            block = block.at[ns:2*ns, 2*ns:].set(q32.T)
            block = block.at[2*ns:, :ns].set(q31)
            block = block.at[2*ns:, ns:2*ns].set(q32)
            block = block.at[2*ns:, 2*ns:].set(q33)

            lower_diag_blocks = lower_diag_blocks.at[t].set(block)

    return diag_blocks, lower_diag_blocks


def extract_bta_blocks_sparse_coo(
    sparse_matrix,
    nt: int,
    ns: int,
    n_fixed_effects: int,
    dtype=None,
) -> dict:
    """Extract BTA blocks from a scipy sparse matrix in padded COO format.

    Instead of storing each block as a dense (ns, ns) array, this
    returns per-block COO triplets (rows, cols, vals) padded to a
    common ``max_nnz`` for each block type.  For typical observation
    matrices, each diagonal block of A^T A has O(ns) nonzeros rather
    than O(ns^2), saving ~99.9 % of memory for large ns.

    Parameters
    ----------
    sparse_matrix : scipy sparse matrix
        The matrix to decompose (typically A^T @ A).
    nt : int
        Number of temporal blocks.
    ns : int
        Spatial block size.
    n_fixed_effects : int
        Arrow-tip size.
    dtype : jnp.dtype, optional
        JAX dtype for value arrays.

    Returns
    -------
    dict
        Keys: ``ata_diag_{rows,cols,vals}``, ``ata_lower_{rows,cols,vals}``,
        ``ata_arrow_{rows,cols,vals}``, ``ata_tip``.
        Row/col arrays are int32; val arrays use *dtype*.
    """
    from scipy import sparse as sp_sparse
    from dalia.core.jax_autodiff import get_jax_dtype
    import numpy as np

    if dtype is None:
        dtype = get_jax_dtype()
    np_dtype = np.float64 if dtype == jnp.float64 else np.float32

    csc = sp_sparse.csc_matrix(sparse_matrix)
    total_st = nt * ns

    # --- Diagonal blocks ---------------------------------------------------
    diag_coo_list = []
    for i in range(nt):
        s = i * ns
        e = s + ns
        blk = csc[s:e, s:e].tocoo()
        diag_coo_list.append((blk.row.astype(np.int32),
                              blk.col.astype(np.int32),
                              blk.data.astype(np_dtype)))
    max_nnz_diag = max(len(v) for _, _, v in diag_coo_list) if diag_coo_list else 0
    if max_nnz_diag == 0:
        max_nnz_diag = 1  # avoid zero-length arrays

    def _pad(arr, length, fill=0):
        out = np.full(length, fill, dtype=arr.dtype)
        out[:len(arr)] = arr
        return out

    ata_diag_rows = np.zeros((nt, max_nnz_diag), dtype=np.int32)
    ata_diag_cols = np.zeros((nt, max_nnz_diag), dtype=np.int32)
    ata_diag_vals = np.zeros((nt, max_nnz_diag), dtype=np_dtype)
    for i, (r, c, v) in enumerate(diag_coo_list):
        ata_diag_rows[i] = _pad(r, max_nnz_diag)
        ata_diag_cols[i] = _pad(c, max_nnz_diag)
        ata_diag_vals[i] = _pad(v, max_nnz_diag)

    # --- Lower-diagonal blocks ---------------------------------------------
    lower_coo_list = []
    for i in range(nt - 1):
        rs = (i + 1) * ns
        re = rs + ns
        cs = i * ns
        ce = cs + ns
        blk = csc[rs:re, cs:ce].tocoo()
        lower_coo_list.append((blk.row.astype(np.int32),
                               blk.col.astype(np.int32),
                               blk.data.astype(np_dtype)))
    max_nnz_lower = max(len(v) for _, _, v in lower_coo_list) if lower_coo_list else 0
    if max_nnz_lower == 0:
        max_nnz_lower = 1

    ata_lower_rows = np.zeros((nt - 1, max_nnz_lower), dtype=np.int32)
    ata_lower_cols = np.zeros((nt - 1, max_nnz_lower), dtype=np.int32)
    ata_lower_vals = np.zeros((nt - 1, max_nnz_lower), dtype=np_dtype)
    for i, (r, c, v) in enumerate(lower_coo_list):
        ata_lower_rows[i] = _pad(r, max_nnz_lower)
        ata_lower_cols[i] = _pad(c, max_nnz_lower)
        ata_lower_vals[i] = _pad(v, max_nnz_lower)

    # --- Arrow-bottom blocks -----------------------------------------------
    if n_fixed_effects > 0:
        arrow_coo_list = []
        for i in range(nt):
            cs = i * ns
            ce = cs + ns
            blk = csc[total_st:, cs:ce].tocoo()
            arrow_coo_list.append((blk.row.astype(np.int32),
                                   blk.col.astype(np.int32),
                                   blk.data.astype(np_dtype)))
        max_nnz_arrow = max(len(v) for _, _, v in arrow_coo_list) if arrow_coo_list else 0
        if max_nnz_arrow == 0:
            max_nnz_arrow = 1

        ata_arrow_rows = np.zeros((nt, max_nnz_arrow), dtype=np.int32)
        ata_arrow_cols = np.zeros((nt, max_nnz_arrow), dtype=np.int32)
        ata_arrow_vals = np.zeros((nt, max_nnz_arrow), dtype=np_dtype)
        for i, (r, c, v) in enumerate(arrow_coo_list):
            ata_arrow_rows[i] = _pad(r, max_nnz_arrow)
            ata_arrow_cols[i] = _pad(c, max_nnz_arrow)
            ata_arrow_vals[i] = _pad(v, max_nnz_arrow)

        ata_tip = jnp.array(csc[total_st:, total_st:].toarray(), dtype=dtype)
    else:
        max_nnz_arrow = 1
        ata_arrow_rows = np.zeros((nt, 1), dtype=np.int32)
        ata_arrow_cols = np.zeros((nt, 1), dtype=np.int32)
        ata_arrow_vals = np.zeros((nt, 1), dtype=np_dtype)
        ata_tip = jnp.zeros((n_fixed_effects, n_fixed_effects), dtype=dtype)

    return {
        'ata_diag_rows': jnp.array(ata_diag_rows),
        'ata_diag_cols': jnp.array(ata_diag_cols),
        'ata_diag_vals': jnp.array(ata_diag_vals, dtype=dtype),
        'ata_lower_rows': jnp.array(ata_lower_rows),
        'ata_lower_cols': jnp.array(ata_lower_cols),
        'ata_lower_vals': jnp.array(ata_lower_vals, dtype=dtype),
        'ata_arrow_rows': jnp.array(ata_arrow_rows),
        'ata_arrow_cols': jnp.array(ata_arrow_cols),
        'ata_arrow_vals': jnp.array(ata_arrow_vals, dtype=dtype),
        'ata_tip': ata_tip,
    }


def scatter_sparse_ata_into_blocks(
    blocks: jnp.ndarray,
    prec: float,
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    vals: jnp.ndarray,
) -> jnp.ndarray:
    """Add scaled sparse COO values into dense BTA blocks.

    Computes ``blocks[i, rows[i], cols[i]] += prec * vals[i]`` for every
    temporal block *i*, using a single vectorised scatter.

    Parameters
    ----------
    blocks : jnp.ndarray
        Dense blocks to update, shape ``(n_blocks, block_size, block_size)``
        or ``(n_blocks, n_fe, block_size)`` depending on the block type.
    prec : float
        Scalar multiplier (likelihood precision).
    rows, cols : jnp.ndarray
        Int32 index arrays, shape ``(n_blocks, max_nnz)``.
    vals : jnp.ndarray
        Value arrays, shape ``(n_blocks, max_nnz)``.

    Returns
    -------
    jnp.ndarray
        Updated blocks with sparse contributions added.
    """
    n_blocks = blocks.shape[0]
    block_idx = jnp.arange(n_blocks)[:, None]  # (n_blocks, 1)
    return blocks.at[block_idx, rows, cols].add(prec * vals)


_jax_cholesky = partial(jax.scipy.linalg.cholesky, lower=True)


def precompute_spatial_components(theta_st, spatial_matrices, temporal_matrices, manifold):
    """Precompute spatial/temporal components for lazy block reconstruction.

    Instead of materializing the full (nt, ns, ns) Q_st arrays, we store
    three (ns, ns) spatial matrices and short temporal coefficient vectors
    from which each block can be reconstructed on the fly.

    Parameters
    ----------
    theta_st : jnp.ndarray
        Spatio-temporal hyperparameters [r_s, r_t, sigma_st].
    spatial_matrices : dict
        Dict with keys 'c0', 'g1', 'g2', 'g3'.
    temporal_matrices : dict
        Dict with keys 'm0', 'm1', 'm2'.
    manifold : str
        Either "sphere" or "plane".

    Returns
    -------
    dict
        Keys: q1s, q2s, q3s (ns, ns), scale, exp_gt (scalars),
        m0_diag, m1_diag, m2_diag (nt,),
        m0_subdiag, m1_subdiag, m2_subdiag (nt-1,).
    """
    r_s = theta_st[0]
    r_t = theta_st[1]
    sigma_st = theta_st[2]

    gamma_s, gamma_t, gamma_st = _interpretable_to_compute_jax(r_s, r_t, sigma_st, manifold)

    c0 = spatial_matrices['c0']
    g1 = spatial_matrices['g1']
    g2 = spatial_matrices['g2']
    g3 = spatial_matrices['g3']

    m0 = temporal_matrices['m0']
    m1 = temporal_matrices['m1']
    m2 = temporal_matrices['m2']

    exp_gamma_s = jnp.exp(gamma_s)
    exp_gamma_t = jnp.exp(gamma_t)
    exp_gamma_st = jnp.exp(gamma_st)

    q1s = exp_gamma_s**2 * c0 + g1
    q2s = exp_gamma_s**4 * c0 + 2 * exp_gamma_s**2 * g1 + g2
    q3s = exp_gamma_s**6 * c0 + 3 * exp_gamma_s**4 * g1 + 3 * exp_gamma_s**2 * g2 + g3

    scale = exp_gamma_st**2

    m0_diag = jnp.diag(m0)
    m1_diag = jnp.diag(m1)
    m2_diag = jnp.diag(m2)

    m0_subdiag = jnp.diag(m0, k=-1)
    m1_subdiag = jnp.diag(m1, k=-1)
    m2_subdiag = jnp.diag(m2, k=-1)

    return {
        'q1s': q1s,
        'q2s': q2s,
        'q3s': q3s,
        'scale': scale,
        'exp_gt': exp_gamma_t,
        'm0_diag': m0_diag,
        'm1_diag': m1_diag,
        'm2_diag': m2_diag,
        'm0_subdiag': m0_subdiag,
        'm1_subdiag': m1_subdiag,
        'm2_subdiag': m2_subdiag,
    }


def _reconstruct_diag_block(sc, i):
    """Reconstruct diagonal block i of Q_st from spatial components."""
    return sc['scale'] * (
        sc['m0_diag'][i] * sc['q3s']
        + sc['exp_gt'] * sc['m1_diag'][i] * sc['q2s']
        + sc['exp_gt']**2 * sc['m2_diag'][i] * sc['q1s']
    )


def _reconstruct_lower_block(sc, i):
    """Reconstruct lower-diagonal block i of Q_st from spatial components."""
    return sc['scale'] * (
        sc['m0_subdiag'][i] * sc['q3s']
        + sc['exp_gt'] * sc['m1_subdiag'][i] * sc['q2s']
        + sc['exp_gt']**2 * sc['m2_subdiag'][i] * sc['q1s']
    )


def lazy_bta_cholesky(
    spatial_comp, nt, ns, n_fe, fe_prec, likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip, dtype,
):
    """Q_cond BTA Cholesky via ``lax.scan`` with lazy block reconstruction.

    Each Q_cond block is reconstructed on the fly from three (ns, ns)
    spatial matrices, avoiding the full (nt, ns, ns) materialization.

    Parameters
    ----------
    spatial_comp : dict
        Output of :func:`precompute_spatial_components`.
    nt, ns, n_fe : int
        Number of temporal blocks, spatial block size, fixed-effects size.
    fe_prec : float
        Fixed-effects prior precision.
    likelihood_prec : scalar
        Likelihood precision (exp(theta_likelihood)).
    ata_diag_rows, ata_diag_cols, ata_diag_vals : jnp.ndarray
        Sparse COO for diagonal AtA blocks, shape (nt, max_nnz).
    ata_lower_rows, ata_lower_cols, ata_lower_vals : jnp.ndarray
        Sparse COO for lower-diagonal AtA blocks, shape (nt-1, max_nnz).
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals : jnp.ndarray
        Sparse COO for arrow AtA blocks, shape (nt, max_nnz).
    ata_tip : jnp.ndarray
        Arrow tip AtA block, shape (n_fe, n_fe).
    dtype : jnp.dtype
        Working dtype.

    Returns
    -------
    L_diag : (nt, ns, ns)
    L_lower : (nt-1, ns, ns)
    L_arrow : (nt, n_fe, ns)
    L_tip : (n_fe, n_fe)
    logdet_Q_cond : scalar
    """
    eps = jnp.finfo(dtype).eps
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    # Pad subdiagonal vectors to length nt
    m0_sub_pad = jnp.concatenate([spatial_comp['m0_subdiag'], jnp.zeros(1, dtype=dtype)])
    m1_sub_pad = jnp.concatenate([spatial_comp['m1_subdiag'], jnp.zeros(1, dtype=dtype)])
    m2_sub_pad = jnp.concatenate([spatial_comp['m2_subdiag'], jnp.zeros(1, dtype=dtype)])

    # Pad lower AtA to length nt (extra zero row at end)
    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)
    ], axis=0)

    sc_padded = {**spatial_comp,
                 'm0_subdiag': m0_sub_pad,
                 'm1_subdiag': m1_sub_pad,
                 'm2_subdiag': m2_sub_pad}

    def scan_body(carry, inputs):
        (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond) = carry
        i, d_rows, d_cols, d_vals, l_rows, l_cols, l_vals, a_rows, a_cols, a_vals = inputs

        # --- Q_cond BTA Cholesky step ---
        q_cond_diag_i = _reconstruct_diag_block(sc_padded, i)
        q_cond_diag_i = q_cond_diag_i.at[d_rows, d_cols].add(likelihood_prec * d_vals)
        q_cond_diag_i = q_cond_diag_i + eps_reg * eye_ns - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        # Lower block
        q_cond_lower_i = _reconstruct_lower_block(sc_padded, i)
        q_cond_lower_i = q_cond_lower_i.at[l_rows, l_cols].add(likelihood_prec * l_vals)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True
        ).T

        # Arrow block
        q_arrow_i = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow_i = q_arrow_i.at[a_rows, a_cols].add(likelihood_prec * a_vals)
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True
        ).T

        # Schur updates for Q_cond
        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(i < nt - 1, new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(i < nt - 1, new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        new_carry = (new_cond_schur, new_arrow_tip_acc, new_arrow_schur, logdet_cond)
        return new_carry, (L_i, L_lower_i, L_arrow_i)

    init_carry = (
        jnp.zeros((ns, ns), dtype=dtype),     # cond_schur
        fe_prec * eye_nfe + likelihood_prec * ata_tip + eps_reg * eye_nfe,  # arrow_tip_acc
        jnp.zeros((n_fe, ns), dtype=dtype),    # arrow_schur
        jnp.array(0.0, dtype=dtype),           # logdet_cond
    )

    scan_inputs = (
        jnp.arange(nt),
        ata_diag_rows,
        ata_diag_cols,
        ata_diag_vals,
        ata_lower_rows_pad,
        ata_lower_cols_pad,
        ata_lower_vals_pad,
        ata_arrow_rows,
        ata_arrow_cols,
        ata_arrow_vals,
    )

    (_, arrow_tip_final, _, logdet_cond), \
        (L_diag_all, L_lower_all, L_arrow_all) = lax.scan(
            scan_body, init_carry, scan_inputs
        )

    L_lower = L_lower_all[:nt - 1]

    # Factorize arrow tip and add its logdet contribution
    L_tip = _jax_cholesky(arrow_tip_final)
    tip_diag = jnp.diag(L_tip)
    safe_tip_diag = jnp.maximum(tip_diag, eps)
    logdet_Q_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_tip_diag))

    return L_diag_all, L_lower, L_arrow_all, L_tip, logdet_Q_cond


def pobtasi_jax(L_diag, L_lower, L_arrow, L_tip):
    """Selected inversion of a BTA matrix from its Cholesky factors.

    Computes the selected elements of the inverse (diagonal, lower-diagonal,
    arrow-bottom and arrow-tip blocks) of a symmetric positive-definite
    block-tridiagonal-arrowhead matrix given its lower Cholesky factor.

    This is a pure-JAX port of :func:`serinv.algs.pobtasi._pobtasi`.
    Arrays are **not** modified in-place; new arrays are returned.

    Parameters
    ----------
    L_diag : (nt, ns, ns)
        Diagonal blocks of the Cholesky factor.
    L_lower : (nt-1, ns, ns)
        Lower-diagonal blocks of the Cholesky factor.
    L_arrow : (nt, n_fe, ns)
        Arrow-bottom blocks of the Cholesky factor.
    L_tip : (n_fe, n_fe)
        Arrow-tip block of the Cholesky factor.

    Returns
    -------
    S_diag : (nt, ns, ns)
    S_lower : (nt-1, ns, ns)
    S_arrow : (nt, n_fe, ns)
    S_tip : (n_fe, n_fe)
    """
    nt = L_diag.shape[0]
    ns = L_diag.shape[1]
    n_fe = L_tip.shape[0]
    eye_ns = jnp.eye(ns, dtype=L_diag.dtype)

    # Invert tip: S_tip = L_tip^{-T} @ L_tip^{-1}
    L_tip_inv = jax.scipy.linalg.solve_triangular(
        L_tip, jnp.eye(n_fe, dtype=L_tip.dtype), lower=True
    )
    S_tip = L_tip_inv.T @ L_tip_inv

    # Last block
    L_blk_inv = jax.scipy.linalg.solve_triangular(
        L_diag[nt - 1], eye_ns, lower=True
    )

    S_arrow_last = -S_tip @ L_arrow[nt - 1] @ L_blk_inv
    S_diag_last = (
        L_blk_inv.T - S_arrow_last.T @ L_arrow[nt - 1]
    ) @ L_blk_inv

    S_diag = L_diag.at[nt - 1].set(S_diag_last)
    S_arrow = L_arrow.at[nt - 1].set(S_arrow_last)

    # Backward loop: i = nt-2 down to 0
    # Carry: (S_diag, S_lower, S_arrow)
    # S_tip is constant throughout the loop (captured via closure).
    S_lower = jnp.zeros_like(L_lower)

    def body_fn(i_rev, carry):
        sd, sl, sa = carry
        i = nt - 2 - i_rev

        Li = L_diag[i]
        L_blk_inv_i = jax.scipy.linalg.solve_triangular(Li, eye_ns, lower=True)

        # Off-diagonal: S_lower[i] = (-S_diag[i+1] @ L_lower[i] - S_arrow[i+1]^T @ L_arrow[i]) @ L_diag[i]^{-1}
        sl_i = (
            -sd[i + 1] @ L_lower[i]
            - sa[i + 1].T @ L_arrow[i]
        ) @ L_blk_inv_i

        # Arrow: S_arrow[i] = (-S_arrow[i+1] @ L_lower[i] - S_tip @ L_arrow[i]) @ L_diag[i]^{-1}
        sa_i = (
            -sa[i + 1] @ L_lower[i]
            - S_tip @ L_arrow[i]
        ) @ L_blk_inv_i

        # Diagonal: S_diag[i] = (L_diag[i]^{-T} - S_lower[i]^T @ L_lower[i] - S_arrow[i]^T @ L_arrow[i]) @ L_diag[i]^{-1}
        sd_i = (
            L_blk_inv_i.T
            - sl_i.T @ L_lower[i]
            - sa_i.T @ L_arrow[i]
        ) @ L_blk_inv_i

        sd = sd.at[i].set(sd_i)
        sl = sl.at[i].set(sl_i)
        sa = sa.at[i].set(sa_i)

        return (sd, sl, sa)

    S_diag, S_lower, S_arrow = lax.fori_loop(
        0, nt - 1, body_fn, (S_diag, S_lower, S_arrow)
    )

    return S_diag, S_lower, S_arrow, S_tip


def logdet_Q_st_scan(theta_st, spatial_matrices, temporal_matrices, manifold, nt, ns, dtype):
    """Compute logdet(Q_st) via a checkpointed BT Cholesky scan.

    This is a standalone differentiable function whose gradient w.r.t.
    ``theta_st`` is computed by JAX AD (``jax.grad``).  The scan body is
    wrapped with ``jax.checkpoint(prevent_cse=True)`` so that per-step
    intermediates are recomputed during the backward pass instead of
    stored (memory: ~32 GiB carry trajectory for gst_large).

    Parameters
    ----------
    theta_st : jnp.ndarray
        Spatio-temporal hyperparameters ``[r_s, r_t, sigma_st]``.
    spatial_matrices, temporal_matrices : dict
        FEM matrices.
    manifold : str
        ``"sphere"`` or ``"plane"``.
    nt, ns : int
        Number of temporal blocks and spatial block size.
    dtype : jnp.dtype
        Working precision.

    Returns
    -------
    logdet : scalar
        ``log|Q_st|``.
    """
    sc = precompute_spatial_components(theta_st, spatial_matrices, temporal_matrices, manifold)
    eps = jnp.finfo(dtype).eps

    @partial(jax.checkpoint, prevent_cse=True)
    def scan_body(carry, i):
        schur, logdet = carry
        q_diag_i = _reconstruct_diag_block(sc, i) - schur
        L_i = _jax_cholesky(q_diag_i)
        diag_vals = jnp.diag(L_i)
        safe_vals = jnp.maximum(diag_vals, eps)
        logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_vals))

        q_lower_i = _reconstruct_lower_block(sc, i)
        L_inv_lower = jax.scipy.linalg.solve_triangular(
            L_i, q_lower_i.T, lower=True
        )
        new_schur = L_inv_lower.T @ L_inv_lower
        new_schur = jnp.where(i < nt - 1, new_schur, jnp.zeros_like(new_schur))
        return (new_schur, logdet), None

    # Pad subdiagonal vectors to length nt (last entry unused)
    sc_padded = {
        **sc,
        'm0_subdiag': jnp.concatenate([sc['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
        'm1_subdiag': jnp.concatenate([sc['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
        'm2_subdiag': jnp.concatenate([sc['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
    }
    # Override sc reference in closure
    sc.update(sc_padded)

    init_carry = (jnp.zeros((ns, ns), dtype=dtype), jnp.array(0.0, dtype=dtype))
    (_, logdet), _ = lax.scan(scan_body, init_carry, jnp.arange(nt))
    return logdet


def bt_logdet_grad(
    theta_st, spatial_matrices, temporal_matrices, manifold, nt, ns, n_theta_st, dtype
):
    """Analytical gradient of logdet(Q_st) via BT selected inversion.

    Replaces ``jax.grad(logdet_Q_st_scan)`` with an analytical computation
    that avoids the 32 GiB scan carry trajectory from AD.

    Algorithm:
      1. Forward BT Cholesky scan storing Schur carries (``nt`` blocks of
         ``(ns, ns)``).
      2. Backward BT selected inversion sweep, reconstructing L blocks from
         stored carries and accumulating ``tr(Σ_block @ ∂Q_st/∂θ)``
         block-by-block.  Only one S block is live at a time.

    Parameters
    ----------
    theta_st : (n_theta_st,)
    spatial_matrices, temporal_matrices : dict
    manifold : str
    nt, ns, n_theta_st : int
    dtype : jnp.dtype

    Returns
    -------
    grad_logdet_st : (n_theta_st,)
        d(logdet Q_st) / d(theta_st)
    """
    sc = precompute_spatial_components(theta_st, spatial_matrices, temporal_matrices, manifold)
    jac_sc = jax.jacfwd(precompute_spatial_components)(
        theta_st, spatial_matrices, temporal_matrices, manifold
    )

    eye_ns = jnp.eye(ns, dtype=dtype)
    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    # Pad subdiagonal to length nt
    m0_s_pad = jnp.concatenate([m0_s, jnp.zeros(1, dtype=dtype)])
    m1_s_pad = jnp.concatenate([m1_s, jnp.zeros(1, dtype=dtype)])
    m2_s_pad = jnp.concatenate([m2_s, jnp.zeros(1, dtype=dtype)])

    sc_pad = {**sc,
              'm0_subdiag': m0_s_pad,
              'm1_subdiag': m1_s_pad,
              'm2_subdiag': m2_s_pad}

    # --- 1. Forward BT Cholesky scan, store incoming Schur as outputs ---
    def fwd_body(schur, i):
        q_diag = _reconstruct_diag_block(sc_pad, i) - schur
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_lower_block(sc_pad, i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        new_schur = L_lower_i @ L_lower_i.T
        new_schur = jnp.where(i < nt - 1, new_schur, jnp.zeros_like(new_schur))
        return new_schur, schur  # output = incoming Schur for reconstruction

    init_schur = jnp.zeros((ns, ns), dtype=dtype)
    _, stored_schurs = lax.scan(fwd_body, init_schur, jnp.arange(nt))
    # stored_schurs[i] = Schur complement subtracted from Q_st_diag[i]

    # --- Helper: reconstruct L_diag[i] and L_lower[i] from stored carry ---
    def _reconstruct_L(i):
        q_diag = _reconstruct_diag_block(sc_pad, i) - stored_schurs[i]
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_lower_block(sc_pad, i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        return L_i, L_lower_i

    # --- Trace target matrices ---
    base_mats = [sc['q3s'], sc['q2s'], sc['q1s']]
    jac_mats = []
    for k in range(n_theta_st):
        jac_mats.extend([
            jac_sc['q3s'][..., k], jac_sc['q2s'][..., k], jac_sc['q1s'][..., k],
        ])
    all_mats = jnp.stack(base_mats + jac_mats, axis=0)
    n_mats = all_mats.shape[0]

    # --- 2. Last block (i = nt-1) ---
    L_last, _ = _reconstruct_L(nt - 1)
    L_inv_last = jax.scipy.linalg.solve_triangular(L_last, eye_ns, lower=True)
    sd_last = L_inv_last.T @ L_inv_last

    tr_d_last = jnp.einsum('ij,sji->s', sd_last, all_mats)
    mw_last = jnp.tile(
        jnp.array([m0_d[nt - 1], m1_d[nt - 1], m2_d[nt - 1]], dtype=dtype),
        1 + n_theta_st)
    acc_d = mw_last * tr_d_last
    acc_l = jnp.zeros(n_mats, dtype=dtype)

    # --- 3. Backward BT selected inversion + trace accumulation ---
    def bwd_body(i_rev, carry):
        sd_prev, acc_d_, acc_l_ = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i = _reconstruct_L(i)
        L_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_ns, lower=True)

        sl_i = -sd_prev @ L_lower_i @ L_inv_i
        sd_i = (L_inv_i.T - sl_i.T @ L_lower_i) @ L_inv_i

        tr_d = jnp.einsum('ij,sji->s', sd_i, all_mats)
        mw_d = jnp.tile(
            jnp.array([m0_d[i], m1_d[i], m2_d[i]], dtype=dtype), 1 + n_theta_st)
        acc_d_ = acc_d_ + mw_d * tr_d

        tr_l = jnp.einsum('ij,sji->s', sl_i, all_mats)
        mw_l = jnp.tile(
            jnp.array([m0_s[i], m1_s[i], m2_s[i]], dtype=dtype), 1 + n_theta_st)
        acc_l_ = acc_l_ + mw_l * tr_l

        return (sd_i, acc_d_, acc_l_)

    _, acc_d, acc_l = lax.fori_loop(
        0, nt - 1, bwd_body, (sd_last, acc_d, acc_l))

    # Factor of 2: lower block S_l[t] appears twice (lower + upper transpose)
    acc_l = 2.0 * acc_l

    # --- 4. Assemble gradient ---
    jac_scale = jac_sc['scale']
    jac_exp_gt = jac_sc['exp_gt']

    weighted_diag = acc_d[0] + exp_gt * acc_d[1] + exp_gt**2 * acc_d[2]
    weighted_lower = acc_l[0] + exp_gt * acc_l[1] + exp_gt**2 * acc_l[2]

    grad_st = jnp.zeros(n_theta_st, dtype=dtype)
    for k in range(n_theta_st):
        term_scale = jac_scale[k] * (weighted_diag + weighted_lower)

        off = 3 + 3 * k
        term_spatial_diag = scale * (
            acc_d[off] + exp_gt * acc_d[off + 1] + exp_gt**2 * acc_d[off + 2])
        term_spatial_lower = scale * (
            acc_l[off] + exp_gt * acc_l[off + 1] + exp_gt**2 * acc_l[off + 2])

        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            acc_d[1] + 2.0 * exp_gt * acc_d[2])
        term_exp_gt_lower = scale * jac_exp_gt[k] * (
            acc_l[1] + 2.0 * exp_gt * acc_l[2])

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    return grad_st


def _spatial_traces(S_diag, S_lower, sc, nt):
    """Compute traces tr(Sigma_block @ spatial_matrix) for all blocks.

    For diagonal blocks:
        tr_diag[i, j] = tr(S_diag[i] @ Sj)  for Sj in {q3s, q2s, q1s}

    For lower-diagonal blocks:
        tr_lower[i, j] = tr(S_lower[i] @ Sj)

    Parameters
    ----------
    S_diag : (nt, ns, ns)
    S_lower : (nt-1, ns, ns)
    sc : dict from precompute_spatial_components
    nt : int

    Returns
    -------
    tr_diag : (nt, 3)  traces with [q3s, q2s, q1s]
    tr_lower : (nt-1, 3)
    """
    spatial_mats = jnp.stack([sc['q3s'], sc['q2s'], sc['q1s']], axis=0)  # (3, ns, ns)

    # tr(A @ B) = sum(A * B^T) = einsum('ij,ji->')
    # Vectorized over blocks and spatial matrices:
    # S_diag: (nt, ns, ns), spatial_mats: (3, ns, ns)
    tr_diag = jnp.einsum('bij,sji->bs', S_diag, spatial_mats)  # (nt, 3)
    tr_lower = jnp.einsum('bij,sji->bs', S_lower, spatial_mats)  # (nt-1, 3)

    return tr_diag, tr_lower


def _compute_grad_logdet_cond(
    S_diag, S_lower, S_arrow, S_tip,
    sc, jac_sc,
    nt, ns, n_fe, n_theta_st,
    likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip,
):
    """Compute ∂(logdet Q_cond)/∂θ_st and ∂(logdet Q_cond)/∂θ_lik analytically.

    Uses the identity: ∂(logdet Q)/∂θ = tr(Q^{-1} ∂Q/∂θ) = tr(Σ ∂Q/∂θ).

    For θ_st: ∂Q_cond/∂θ_st = ∂Q_st/∂θ_st (likelihood term doesn't depend on θ_st).
    For θ_lik: ∂Q_cond/∂θ_lik = lik_prec * AtA.

    Parameters
    ----------
    S_diag, S_lower, S_arrow, S_tip : Sigma factors from selected inversion.
    sc : dict from precompute_spatial_components.
    jac_sc : dict of Jacobians of sc w.r.t. theta_st (each value has leading dim n_theta_st).
    nt, ns, n_fe, n_theta_st : int
    likelihood_prec : scalar
    ata_* : sparse COO data for AtA blocks.
    ata_tip : (n_fe, n_fe) dense AtA tip.

    Returns
    -------
    grad_st : (n_theta_st,) gradient w.r.t. theta_st
    grad_lik : scalar gradient w.r.t. theta_lik
    """
    # --- Gradient w.r.t. theta_st ---
    # Q_st_diag[i] = scale * (m0_d[i]*q3s + exp_gt*m1_d[i]*q2s + exp_gt^2*m2_d[i]*q1s)
    # ∂Q_st_diag[i]/∂θ_k = d_scale_k * (...) + scale * (m0_d[i]*d_q3s_k + d_exp_gt_k*m1_d[i]*q2s + ...)
    # tr(Σ_diag[i] @ ∂Q_st_diag[i]/∂θ_k) can be reduced to sums of precomputed spatial traces.

    tr_diag, tr_lower = _spatial_traces(S_diag, S_lower, sc, nt)
    # tr_diag[:, 0] = tr(S_diag[i] @ q3s), [:, 1] = ... @ q2s, [:, 2] = ... @ q1s

    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    # Temporal coefficients for diagonal blocks (nt,)
    coeff_q3s_diag = m0_d                          # coefficient of q3s in diag block
    coeff_q2s_diag = exp_gt * m1_d                  # coefficient of q2s
    coeff_q1s_diag = exp_gt**2 * m2_d               # coefficient of q1s

    # For lower blocks (nt-1,)
    coeff_q3s_lower = m0_s
    coeff_q2s_lower = exp_gt * m1_s
    coeff_q1s_lower = exp_gt**2 * m2_s

    # Weighted spatial traces: sum over blocks of temporal_coeff[i] * tr_spatial[i, j]
    # This gives tr(Σ @ scale * temporal_term_j) for each spatial matrix j
    # diag_contrib[j] = sum_i coeff_j_diag[i] * tr_diag[i, j]
    weighted_diag = (
        jnp.sum(coeff_q3s_diag * tr_diag[:, 0])
        + jnp.sum(coeff_q2s_diag * tr_diag[:, 1])
        + jnp.sum(coeff_q1s_diag * tr_diag[:, 2])
    )
    # Factor of 2 on lower: each S_l[t] counts for both lower and upper blocks
    weighted_lower = 2.0 * (
        jnp.sum(coeff_q3s_lower * tr_lower[:, 0])
        + jnp.sum(coeff_q2s_lower * tr_lower[:, 1])
        + jnp.sum(coeff_q1s_lower * tr_lower[:, 2])
    )

    # Jacobians of spatial components w.r.t. theta_st.
    # jacfwd puts the input dimension last: q3s has shape (ns, ns, n_theta_st), etc.
    jac_q3s = jac_sc['q3s']   # (ns, ns, n_theta_st)
    jac_q2s = jac_sc['q2s']
    jac_q1s = jac_sc['q1s']
    jac_scale = jac_sc['scale']   # (n_theta_st,)
    jac_exp_gt = jac_sc['exp_gt']  # (n_theta_st,)

    grad_st = jnp.zeros(n_theta_st, dtype=S_diag.dtype)

    for k in range(n_theta_st):
        dq3s_k = jac_q3s[..., k]  # (ns, ns)
        dq2s_k = jac_q2s[..., k]
        dq1s_k = jac_q1s[..., k]

        # Term 1: d_scale[k] * (original sum of traces)
        term_scale = jac_scale[k] * (weighted_diag + weighted_lower)

        # Term 2: scale * traces with Jacobian spatial matrices
        tr_S_dq3s = jnp.einsum('bij,ji->b', S_diag, dq3s_k)  # (nt,)
        tr_S_dq2s = jnp.einsum('bij,ji->b', S_diag, dq2s_k)
        tr_S_dq1s = jnp.einsum('bij,ji->b', S_diag, dq1s_k)

        tr_Sl_dq3s = jnp.einsum('bij,ji->b', S_lower, dq3s_k)  # (nt-1,)
        tr_Sl_dq2s = jnp.einsum('bij,ji->b', S_lower, dq2s_k)
        tr_Sl_dq1s = jnp.einsum('bij,ji->b', S_lower, dq1s_k)

        term_spatial_diag = scale * (
            jnp.sum(m0_d * tr_S_dq3s)
            + exp_gt * jnp.sum(m1_d * tr_S_dq2s)
            + exp_gt**2 * jnp.sum(m2_d * tr_S_dq1s)
        )
        term_spatial_lower = 2.0 * scale * (
            jnp.sum(m0_s * tr_Sl_dq3s)
            + exp_gt * jnp.sum(m1_s * tr_Sl_dq2s)
            + exp_gt**2 * jnp.sum(m2_s * tr_Sl_dq1s)
        )

        # Term 3: exp_gt derivative contributions
        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            jnp.sum(m1_d * tr_diag[:, 1])
            + 2.0 * exp_gt * jnp.sum(m2_d * tr_diag[:, 2])
        )
        term_exp_gt_lower = 2.0 * scale * jac_exp_gt[k] * (
            jnp.sum(m1_s * tr_lower[:, 1])
            + 2.0 * exp_gt * jnp.sum(m2_s * tr_lower[:, 2])
        )

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    # --- Gradient w.r.t. theta_lik ---
    # ∂Q_cond/∂θ_lik = lik_prec * AtA (since Q_cond = Q_st + lik_prec * AtA)
    # tr(Σ @ lik_prec * AtA) = lik_prec * tr(Σ @ AtA)
    # Compute sparse trace using COO data:
    # tr(Σ_diag[i] @ AtA_diag[i]) = sum_j Σ_diag[i, rows[i,j], cols[i,j]] * vals[i,j]
    n_blocks_diag = ata_diag_rows.shape[0]
    block_idx_d = jnp.arange(n_blocks_diag)[:, None]
    sparse_tr_diag = jnp.sum(S_diag[block_idx_d, ata_diag_rows, ata_diag_cols] * ata_diag_vals)

    n_blocks_lower = ata_lower_rows.shape[0]
    block_idx_l = jnp.arange(n_blocks_lower)[:, None]
    sparse_tr_lower = jnp.sum(S_lower[block_idx_l, ata_lower_rows, ata_lower_cols] * ata_lower_vals)

    n_blocks_arrow = ata_arrow_rows.shape[0]
    block_idx_a = jnp.arange(n_blocks_arrow)[:, None]
    sparse_tr_arrow = jnp.sum(S_arrow[block_idx_a, ata_arrow_rows, ata_arrow_cols] * ata_arrow_vals)

    sparse_tr_tip = jnp.sum(S_tip * ata_tip)

    # Lower blocks contribute twice (symmetry: tr(Σ @ Q) counts both off-diag blocks)
    grad_lik = likelihood_prec * (sparse_tr_diag + 2.0 * sparse_tr_lower + 2.0 * sparse_tr_arrow + sparse_tr_tip)

    return grad_st, grad_lik


def _compute_grad_quad(
    x, sc, jac_sc,
    nt, ns, n_fe, n_theta_st,
    rhs, likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip,
):
    """Compute total derivative of quad = rhs^T Q_cond^{-1} rhs w.r.t. theta.

    Since x = Q_cond^{-1} rhs and quad = x^T Q_cond x = rhs^T Q_cond^{-1} rhs:
    - d(quad)/d(θ_st[k]) = -x^T (∂Q_st/∂θ_st[k]) x
    - d(quad)/d(θ_lik) = 2 x^T rhs - lik_prec * x^T AtA x

    Parameters
    ----------
    x : (nt*ns + n_fe,) solution vector
    sc, jac_sc : spatial components and their Jacobians
    nt, ns, n_fe, n_theta_st : int
    rhs : (nt*ns + n_fe,) right-hand side vector
    likelihood_prec : scalar
    ata_* : sparse COO data

    Returns
    -------
    grad_st : (n_theta_st,) gradient w.r.t. theta_st
    grad_lik : scalar gradient w.r.t. theta_lik
    """
    x_st = x[:nt * ns].reshape(nt, ns)  # (nt, ns)

    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    # Precompute x^T S_j x for each spatial matrix, for each block
    spatial_mats = jnp.stack([sc['q3s'], sc['q2s'], sc['q1s']], axis=0)  # (3, ns, ns)
    # x_st[i]^T @ S_j @ x_st[i] = einsum('j,jk,k->', x_st[i], S_j, x_st[i])
    # Vectorized: (nt, 3)
    xSx_diag = jnp.einsum('bi,sij,bj->bs', x_st, spatial_mats, x_st)  # (nt, 3)

    # Lower blocks: x_st[i+1]^T @ S_j @ x_st[i]
    xSx_lower = jnp.einsum('bi,sij,bj->bs', x_st[1:], spatial_mats, x_st[:-1])  # (nt-1, 3)

    # jacfwd puts input dim last: q3s is (ns, ns, n_theta_st)
    jac_q3s = jac_sc['q3s']   # (ns, ns, n_theta_st)
    jac_q2s = jac_sc['q2s']
    jac_q1s = jac_sc['q1s']
    jac_scale = jac_sc['scale']   # (n_theta_st,)
    jac_exp_gt = jac_sc['exp_gt']  # (n_theta_st,)

    coeff_q3s_diag = m0_d
    coeff_q2s_diag = exp_gt * m1_d
    coeff_q1s_diag = exp_gt**2 * m2_d
    coeff_q3s_lower = m0_s
    coeff_q2s_lower = exp_gt * m1_s
    coeff_q1s_lower = exp_gt**2 * m2_s

    weighted_diag = (
        jnp.sum(coeff_q3s_diag * xSx_diag[:, 0])
        + jnp.sum(coeff_q2s_diag * xSx_diag[:, 1])
        + jnp.sum(coeff_q1s_diag * xSx_diag[:, 2])
    )
    weighted_lower = (
        jnp.sum(coeff_q3s_lower * xSx_lower[:, 0])
        + jnp.sum(coeff_q2s_lower * xSx_lower[:, 1])
        + jnp.sum(coeff_q1s_lower * xSx_lower[:, 2])
    )

    grad_st = jnp.zeros(n_theta_st, dtype=x.dtype)

    for k in range(n_theta_st):
        dq3s_k = jac_q3s[..., k]  # (ns, ns)
        dq2s_k = jac_q2s[..., k]
        dq1s_k = jac_q1s[..., k]

        # Term 1: d_scale[k] contribution
        term_scale = jac_scale[k] * (weighted_diag + 2.0 * weighted_lower)

        # Term 2: Jacobians of spatial matrices
        xDqx_diag_q3 = jnp.einsum('bi,ij,bj->b', x_st, dq3s_k, x_st)  # (nt,)
        xDqx_diag_q2 = jnp.einsum('bi,ij,bj->b', x_st, dq2s_k, x_st)
        xDqx_diag_q1 = jnp.einsum('bi,ij,bj->b', x_st, dq1s_k, x_st)

        xDqx_lower_q3 = jnp.einsum('bi,ij,bj->b', x_st[1:], dq3s_k, x_st[:-1])  # (nt-1,)
        xDqx_lower_q2 = jnp.einsum('bi,ij,bj->b', x_st[1:], dq2s_k, x_st[:-1])
        xDqx_lower_q1 = jnp.einsum('bi,ij,bj->b', x_st[1:], dq1s_k, x_st[:-1])

        term_spatial_diag = scale * (
            jnp.sum(m0_d * xDqx_diag_q3)
            + exp_gt * jnp.sum(m1_d * xDqx_diag_q2)
            + exp_gt**2 * jnp.sum(m2_d * xDqx_diag_q1)
        )
        term_spatial_lower = 2.0 * scale * (
            jnp.sum(m0_s * xDqx_lower_q3)
            + exp_gt * jnp.sum(m1_s * xDqx_lower_q2)
            + exp_gt**2 * jnp.sum(m2_s * xDqx_lower_q1)
        )

        # Term 3: exp_gt derivative
        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            jnp.sum(m1_d * xSx_diag[:, 1])
            + 2.0 * exp_gt * jnp.sum(m2_d * xSx_diag[:, 2])
        )
        term_exp_gt_lower = 2.0 * scale * jac_exp_gt[k] * (
            jnp.sum(m1_s * xSx_lower[:, 1])
            + 2.0 * exp_gt * jnp.sum(m2_s * xSx_lower[:, 2])
        )

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    # Negate: total derivative is -x^T (dQ_st/dtheta_st) x
    grad_st = -grad_st

    # --- Gradient w.r.t. theta_lik ---
    # d(quad)/d(theta_lik) = 2*x^T*rhs - lik_prec * x^T*AtA*x
    # Compute x^T AtA x using sparse COO data.
    x_fe = x[nt * ns:]

    n_blocks_diag = ata_diag_rows.shape[0]
    block_idx_d = jnp.arange(n_blocks_diag)[:, None]
    xAtAx_diag = jnp.sum(x_st[block_idx_d, ata_diag_rows] * ata_diag_vals * x_st[block_idx_d, ata_diag_cols])

    n_blocks_lower = ata_lower_rows.shape[0]
    block_idx_l = jnp.arange(n_blocks_lower)[:, None]
    xAtAx_lower = jnp.sum(x_st[1:][block_idx_l, ata_lower_rows] * ata_lower_vals * x_st[:-1][block_idx_l, ata_lower_cols])

    n_blocks_arrow = ata_arrow_rows.shape[0]
    block_idx_a = jnp.arange(n_blocks_arrow)[:, None]
    xAtAx_arrow = jnp.sum(x_fe[ata_arrow_rows] * ata_arrow_vals * x_st[block_idx_a, ata_arrow_cols])

    xAtAx_tip = x_fe @ ata_tip @ x_fe

    xAtAx = xAtAx_diag + 2.0 * xAtAx_lower + 2.0 * xAtAx_arrow + xAtAx_tip

    grad_lik = 2.0 * jnp.dot(x, rhs) - likelihood_prec * xAtAx

    return grad_st, grad_lik


def selected_inversion_grads_jax(
    L_diag, L_lower, L_arrow, L_tip,
    sc, jac_sc,
    nt, ns, n_fe, n_theta_st,
    likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip,
):
    """Fused selected inversion + logdet-gradient accumulation.

    Combines :func:`pobtasi_jax` and :func:`_compute_grad_logdet_cond`
    into a single backward sweep so that only **one block** of Sigma is
    live at a time.  This reduces peak memory from
    ``L (64 GiB) + S (64 GiB) = 128 GiB`` to
    ``L (64 GiB) + one S block (~128 MB) = ~64 GiB``.

    Parameters
    ----------
    L_diag : (nt, ns, ns)
    L_lower : (nt-1, ns, ns)
    L_arrow : (nt, n_fe, ns)
    L_tip : (n_fe, n_fe)
    sc : dict from :func:`precompute_spatial_components`
    jac_sc : dict of Jacobians (from ``jax.jacfwd``, input dim last)
    nt, ns, n_fe, n_theta_st : int
    likelihood_prec : scalar
    ata_* : sparse COO data for AtA blocks
    ata_tip : (n_fe, n_fe)

    Returns
    -------
    grad_st : (n_theta_st,)
        d(logdet Q_cond) / d(theta_st)
    grad_lik : scalar
        d(logdet Q_cond) / d(theta_lik)
    """
    dtype = L_diag.dtype
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    # Build trace-target matrix stack: (n_mats, ns, ns)
    # Layout: [q3s, q2s, q1s, dq3s_0, dq2s_0, dq1s_0, dq3s_1, ..., dq1s_2]
    base_mats = [sc['q3s'], sc['q2s'], sc['q1s']]
    jac_mats = []
    for k in range(n_theta_st):
        jac_mats.extend([
            jac_sc['q3s'][..., k],
            jac_sc['q2s'][..., k],
            jac_sc['q1s'][..., k],
        ])
    all_mats = jnp.stack(base_mats + jac_mats, axis=0)  # (3 + 3*n_theta_st, ns, ns)
    n_mats = all_mats.shape[0]

    # --- S_tip ---
    L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, eye_nfe, lower=True)
    S_tip = L_tip_inv.T @ L_tip_inv

    # --- Last block (i = nt-1) ---
    L_blk_inv = jax.scipy.linalg.solve_triangular(
        L_diag[nt - 1], eye_ns, lower=True
    )
    sa_last = -S_tip @ L_arrow[nt - 1] @ L_blk_inv
    sd_last = (L_blk_inv.T - sa_last.T @ L_arrow[nt - 1]) @ L_blk_inv

    # Accumulate traces from last diagonal block
    traces_d = jnp.einsum('ij,sji->s', sd_last, all_mats)
    m_wt = jnp.tile(jnp.array([m0_d[nt - 1], m1_d[nt - 1], m2_d[nt - 1]], dtype=dtype),
                     1 + n_theta_st)
    acc_d = m_wt * traces_d

    # No lower block for last position
    acc_l = jnp.zeros(n_mats, dtype=dtype)

    # Sparse traces from last block
    sp_d = jnp.sum(sd_last[ata_diag_rows[nt - 1], ata_diag_cols[nt - 1]]
                   * ata_diag_vals[nt - 1])
    sp_l = jnp.array(0.0, dtype=dtype)
    sp_a = jnp.sum(sa_last[ata_arrow_rows[nt - 1], ata_arrow_cols[nt - 1]]
                   * ata_arrow_vals[nt - 1])

    # --- Backward loop ---
    def body_fn(i_rev, carry):
        sd_prev, sa_prev, acc_d_, acc_l_, sp_d_, sp_l_, sp_a_ = carry
        i = nt - 2 - i_rev

        Li = L_diag[i]
        L_blk_inv_i = jax.scipy.linalg.solve_triangular(Li, eye_ns, lower=True)

        sl_i = (-sd_prev @ L_lower[i] - sa_prev.T @ L_arrow[i]) @ L_blk_inv_i
        sa_i = (-sa_prev @ L_lower[i] - S_tip @ L_arrow[i]) @ L_blk_inv_i
        sd_i = (L_blk_inv_i.T - sl_i.T @ L_lower[i] - sa_i.T @ L_arrow[i]) @ L_blk_inv_i

        # Diagonal traces
        tr_d = jnp.einsum('ij,sji->s', sd_i, all_mats)
        mw_d = jnp.tile(jnp.array([m0_d[i], m1_d[i], m2_d[i]], dtype=dtype),
                         1 + n_theta_st)
        acc_d_ = acc_d_ + mw_d * tr_d

        # Lower traces
        tr_l = jnp.einsum('ij,sji->s', sl_i, all_mats)
        mw_l = jnp.tile(jnp.array([m0_s[i], m1_s[i], m2_s[i]], dtype=dtype),
                         1 + n_theta_st)
        acc_l_ = acc_l_ + mw_l * tr_l

        # Sparse COO traces
        sp_d_ = sp_d_ + jnp.sum(
            sd_i[ata_diag_rows[i], ata_diag_cols[i]] * ata_diag_vals[i])
        sp_l_ = sp_l_ + jnp.sum(
            sl_i[ata_lower_rows[i], ata_lower_cols[i]] * ata_lower_vals[i])
        sp_a_ = sp_a_ + jnp.sum(
            sa_i[ata_arrow_rows[i], ata_arrow_cols[i]] * ata_arrow_vals[i])

        return (sd_i, sa_i, acc_d_, acc_l_, sp_d_, sp_l_, sp_a_)

    init_carry = (sd_last, sa_last, acc_d, acc_l, sp_d, sp_l, sp_a)
    _, _, acc_d, acc_l, sp_d, sp_l, sp_a = lax.fori_loop(
        0, nt - 1, body_fn, init_carry
    )

    # Factor of 2: lower block S_l[t] appears twice (lower + upper transpose)
    acc_l = 2.0 * acc_l

    # --- Assemble grad_st ---
    jac_scale = jac_sc['scale']    # (n_theta_st,)
    jac_exp_gt = jac_sc['exp_gt']  # (n_theta_st,)

    weighted_diag = acc_d[0] + exp_gt * acc_d[1] + exp_gt**2 * acc_d[2]
    weighted_lower = acc_l[0] + exp_gt * acc_l[1] + exp_gt**2 * acc_l[2]

    grad_st = jnp.zeros(n_theta_st, dtype=dtype)
    for k in range(n_theta_st):
        term_scale = jac_scale[k] * (weighted_diag + weighted_lower)

        off = 3 + 3 * k
        term_spatial_diag = scale * (
            acc_d[off] + exp_gt * acc_d[off + 1] + exp_gt**2 * acc_d[off + 2])
        term_spatial_lower = scale * (
            acc_l[off] + exp_gt * acc_l[off + 1] + exp_gt**2 * acc_l[off + 2])

        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            acc_d[1] + 2.0 * exp_gt * acc_d[2])
        term_exp_gt_lower = scale * jac_exp_gt[k] * (
            acc_l[1] + 2.0 * exp_gt * acc_l[2])

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    # --- Assemble grad_lik ---
    sp_tip = jnp.sum(S_tip * ata_tip)
    grad_lik = likelihood_prec * (sp_d + 2.0 * sp_l + 2.0 * sp_a + sp_tip)

    return grad_st, grad_lik


def lazy_bta_cholesky_carries(
    spatial_comp, nt, ns, n_fe, fe_prec, likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip, dtype,
):
    """BTA Cholesky storing carries instead of L factor blocks.

    Same computation as :func:`lazy_bta_cholesky` but outputs the scan
    carries ``(cond_schur, arrow_schur)`` entering each step instead of
    the L factor blocks ``(L_diag, L_lower, L_arrow)``.  This reduces
    scan output storage from ~64 GiB to ~32 GiB for gst_large.

    L blocks can be reconstructed on-the-fly from stored carries via
    :func:`selected_inversion_grads_from_carries_jax`.

    Parameters
    ----------
    spatial_comp : dict
        Output of :func:`precompute_spatial_components`.
    nt, ns, n_fe : int
    fe_prec : float
    likelihood_prec : scalar
    ata_diag_rows, ata_diag_cols, ata_diag_vals : jnp.ndarray
        Sparse COO for diagonal AtA blocks, shape (nt, max_nnz).
    ata_lower_rows, ata_lower_cols, ata_lower_vals : jnp.ndarray
        Sparse COO for lower-diagonal AtA blocks, shape (nt-1, max_nnz).
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals : jnp.ndarray
        Sparse COO for arrow AtA blocks, shape (nt, max_nnz).
    ata_tip : jnp.ndarray
        Arrow tip AtA block, shape (n_fe, n_fe).
    dtype : jnp.dtype

    Returns
    -------
    stored_cond_schurs : (nt, ns, ns)
        Schur complement entering each step (subtracted from diagonal).
    stored_arrow_schurs : (nt, n_fe, ns)
        Arrow Schur complement entering each step.
    L_tip : (n_fe, n_fe)
        Arrow tip Cholesky factor.
    logdet_Q_cond : scalar
    """
    eps = jnp.finfo(dtype).eps
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    m0_sub_pad = jnp.concatenate([spatial_comp['m0_subdiag'], jnp.zeros(1, dtype=dtype)])
    m1_sub_pad = jnp.concatenate([spatial_comp['m1_subdiag'], jnp.zeros(1, dtype=dtype)])
    m2_sub_pad = jnp.concatenate([spatial_comp['m2_subdiag'], jnp.zeros(1, dtype=dtype)])

    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)
    ], axis=0)

    sc_padded = {**spatial_comp,
                 'm0_subdiag': m0_sub_pad,
                 'm1_subdiag': m1_sub_pad,
                 'm2_subdiag': m2_sub_pad}

    def scan_body(carry, inputs):
        (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond) = carry
        i, d_rows, d_cols, d_vals, l_rows, l_cols, l_vals, a_rows, a_cols, a_vals = inputs

        saved = (cond_schur, arrow_schur)

        q_cond_diag_i = _reconstruct_diag_block(sc_padded, i)
        q_cond_diag_i = q_cond_diag_i.at[d_rows, d_cols].add(likelihood_prec * d_vals)
        q_cond_diag_i = q_cond_diag_i + eps_reg * eye_ns - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        q_cond_lower_i = _reconstruct_lower_block(sc_padded, i)
        q_cond_lower_i = q_cond_lower_i.at[l_rows, l_cols].add(likelihood_prec * l_vals)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True
        ).T

        q_arrow_i = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow_i = q_arrow_i.at[a_rows, a_cols].add(likelihood_prec * a_vals)
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True
        ).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(i < nt - 1, new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(i < nt - 1, new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        new_carry = (new_cond_schur, new_arrow_tip_acc, new_arrow_schur, logdet_cond)
        return new_carry, saved

    init_carry = (
        jnp.zeros((ns, ns), dtype=dtype),
        fe_prec * eye_nfe + likelihood_prec * ata_tip + eps_reg * eye_nfe,
        jnp.zeros((n_fe, ns), dtype=dtype),
        jnp.array(0.0, dtype=dtype),
    )

    scan_inputs = (
        jnp.arange(nt),
        ata_diag_rows,
        ata_diag_cols,
        ata_diag_vals,
        ata_lower_rows_pad,
        ata_lower_cols_pad,
        ata_lower_vals_pad,
        ata_arrow_rows,
        ata_arrow_cols,
        ata_arrow_vals,
    )

    (_, arrow_tip_final, _, logdet_cond), \
        (stored_cond_schurs, stored_arrow_schurs) = lax.scan(
            scan_body, init_carry, scan_inputs
        )

    L_tip = _jax_cholesky(arrow_tip_final)
    tip_diag = jnp.diag(L_tip)
    safe_tip_diag = jnp.maximum(tip_diag, eps)
    logdet_Q_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_tip_diag))

    return stored_cond_schurs, stored_arrow_schurs, L_tip, logdet_Q_cond


def fused_cholesky_fwd_sub(
    spatial_comp, nt, ns, n_fe, fe_prec, likelihood_prec,
    rhs_st, rhs_fe,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip, dtype,
):
    """Fused BTA Cholesky + forward substitution in a single scan.

    Combines the Cholesky factorization and forward solve ``L y = rhs``
    into one :func:`lax.scan`, so L blocks are per-step intermediates
    that are never stored as scan outputs.  Outputs the Schur complement
    carries (~32 GiB for gst_large) and forward-sub solutions y_st (~8 MB).

    Parameters
    ----------
    spatial_comp : dict
        Output of :func:`precompute_spatial_components`.
    nt, ns, n_fe : int
    fe_prec : float
    likelihood_prec : scalar
    rhs_st : (nt, ns)
        Spatio-temporal portion of the right-hand side.
    rhs_fe : (n_fe,)
        Fixed-effects portion of the right-hand side.
    ata_diag_rows, ata_diag_cols, ata_diag_vals : jnp.ndarray
        Sparse COO for diagonal AtA blocks, shape (nt, max_nnz).
    ata_lower_rows, ata_lower_cols, ata_lower_vals : jnp.ndarray
        Sparse COO for lower-diagonal AtA blocks, shape (nt-1, max_nnz).
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals : jnp.ndarray
        Sparse COO for arrow AtA blocks, shape (nt, max_nnz).
    ata_tip : jnp.ndarray
        Arrow tip AtA block, shape (n_fe, n_fe).
    dtype : jnp.dtype

    Returns
    -------
    stored_cond_schurs : (nt, ns, ns)
    stored_arrow_schurs : (nt, n_fe, ns)
    y_st : (nt, ns)
        Forward substitution result for spatio-temporal blocks.
    L_tip : (n_fe, n_fe)
    arrow_rhs_acc : (n_fe,)
        Modified arrow rhs: ``rhs_fe - sum_i L_arrow[i] @ y_i``.
    logdet_Q_cond : scalar
    """
    eps = jnp.finfo(dtype).eps
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    m0_sub_pad = jnp.concatenate([spatial_comp['m0_subdiag'], jnp.zeros(1, dtype=dtype)])
    m1_sub_pad = jnp.concatenate([spatial_comp['m1_subdiag'], jnp.zeros(1, dtype=dtype)])
    m2_sub_pad = jnp.concatenate([spatial_comp['m2_subdiag'], jnp.zeros(1, dtype=dtype)])

    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)
    ], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)
    ], axis=0)

    sc_padded = {**spatial_comp,
                 'm0_subdiag': m0_sub_pad,
                 'm1_subdiag': m1_sub_pad,
                 'm2_subdiag': m2_sub_pad}

    def scan_body(carry, inputs):
        (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
         prev_lower_y, arrow_rhs_acc) = carry
        (i, rhs_i, d_rows, d_cols, d_vals,
         l_rows, l_cols, l_vals, a_rows, a_cols, a_vals) = inputs

        saved_carries = (cond_schur, arrow_schur)

        # --- Cholesky step ---
        q_cond_diag_i = _reconstruct_diag_block(sc_padded, i)
        q_cond_diag_i = q_cond_diag_i.at[d_rows, d_cols].add(likelihood_prec * d_vals)
        q_cond_diag_i = q_cond_diag_i + eps_reg * eye_ns - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        q_cond_lower_i = _reconstruct_lower_block(sc_padded, i)
        q_cond_lower_i = q_cond_lower_i.at[l_rows, l_cols].add(likelihood_prec * l_vals)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True
        ).T

        q_arrow_i = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow_i = q_arrow_i.at[a_rows, a_cols].add(likelihood_prec * a_vals)
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True
        ).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(i < nt - 1, new_cond_schur, jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(i < nt - 1, new_arrow_schur, jnp.zeros_like(new_arrow_schur))

        # --- Forward substitution step ---
        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)

        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(i < nt - 1, new_prev_lower_y,
                                     jnp.zeros_like(new_prev_lower_y))

        new_arrow_rhs_acc = arrow_rhs_acc - L_arrow_i @ y_i

        new_carry = (new_cond_schur, new_arrow_tip_acc, new_arrow_schur, logdet_cond,
                     new_prev_lower_y, new_arrow_rhs_acc)
        return new_carry, (saved_carries, y_i)

    init_carry = (
        jnp.zeros((ns, ns), dtype=dtype),
        fe_prec * eye_nfe + likelihood_prec * ata_tip + eps_reg * eye_nfe,
        jnp.zeros((n_fe, ns), dtype=dtype),
        jnp.array(0.0, dtype=dtype),
        jnp.zeros(ns, dtype=dtype),
        rhs_fe,
    )

    scan_inputs = (
        jnp.arange(nt),
        rhs_st,
        ata_diag_rows,
        ata_diag_cols,
        ata_diag_vals,
        ata_lower_rows_pad,
        ata_lower_cols_pad,
        ata_lower_vals_pad,
        ata_arrow_rows,
        ata_arrow_cols,
        ata_arrow_vals,
    )

    (_, arrow_tip_final, _, logdet_cond, _, arrow_rhs_acc), \
        ((stored_cond_schurs, stored_arrow_schurs), y_st) = lax.scan(
            scan_body, init_carry, scan_inputs
        )

    L_tip = _jax_cholesky(arrow_tip_final)
    tip_diag = jnp.diag(L_tip)
    safe_tip_diag = jnp.maximum(tip_diag, eps)
    logdet_Q_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_tip_diag))

    return stored_cond_schurs, stored_arrow_schurs, y_st, L_tip, arrow_rhs_acc, logdet_Q_cond


def backward_sub_from_carries(
    stored_cond_schurs, stored_arrow_schurs, L_tip,
    y_st, arrow_rhs_acc,
    sc, likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    nt, ns, n_fe, dtype,
):
    """Backward substitution ``L^T x = y``, reconstructing L from carries.

    Performs the backward solve to obtain ``x = Q_cond^{-1} rhs``, where
    the Cholesky factor L is reconstructed on-the-fly from stored Schur
    complement carries, avoiding materialization of full L arrays.

    Also computes the quadratic form ``rhs^T Q_cond^{-1} rhs = ||y||^2``.

    Parameters
    ----------
    stored_cond_schurs : (nt, ns, ns)
    stored_arrow_schurs : (nt, n_fe, ns)
    L_tip : (n_fe, n_fe)
    y_st : (nt, ns)
        Forward substitution result for spatio-temporal blocks.
    arrow_rhs_acc : (n_fe,)
        Modified arrow rhs: ``rhs_fe - sum_i L_arrow[i] @ y_i``.
    sc : dict
        Output of :func:`precompute_spatial_components` (unpadded).
    likelihood_prec : scalar
    ata_* : sparse COO data
    nt, ns, n_fe : int
    dtype : jnp.dtype

    Returns
    -------
    x : (nt * ns + n_fe,)
    quad : scalar
    """
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)

    sc_padded = {**sc,
                 'm0_subdiag': jnp.concatenate([sc['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                 'm1_subdiag': jnp.concatenate([sc['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                 'm2_subdiag': jnp.concatenate([sc['m2_subdiag'], jnp.zeros(1, dtype=dtype)])}

    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)], axis=0)

    def reconstruct_L(i):
        q_diag = _reconstruct_diag_block(sc_padded, i)
        q_diag = q_diag.at[ata_diag_rows[i], ata_diag_cols[i]].add(
            likelihood_prec * ata_diag_vals[i])
        q_diag = q_diag + eps_reg * eye_ns - stored_cond_schurs[i]
        L_i = _jax_cholesky(q_diag)

        q_lower = _reconstruct_lower_block(sc_padded, i)
        q_lower = q_lower.at[ata_lower_rows_pad[i], ata_lower_cols_pad[i]].add(
            likelihood_prec * ata_lower_vals_pad[i])
        L_lower_i = jax.scipy.linalg.solve_triangular(L_i, q_lower.T, lower=True).T

        q_arrow = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow = q_arrow.at[ata_arrow_rows[i], ata_arrow_cols[i]].add(
            likelihood_prec * ata_arrow_vals[i])
        q_arrow = q_arrow - stored_arrow_schurs[i]
        L_arrow_i = jax.scipy.linalg.solve_triangular(L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    # Forward sub on tip + quadratic form
    y_fe = jax.scipy.linalg.solve_triangular(L_tip, arrow_rhs_acc, lower=True)
    quad = jnp.sum(y_st ** 2) + jnp.sum(y_fe ** 2)

    # Backward sub: L^T x = y
    x_fe = jax.scipy.linalg.solve_triangular(L_tip.T, y_fe, lower=False)

    L_last, _, L_arrow_last = reconstruct_L(nt - 1)
    x_last = jax.scipy.linalg.solve_triangular(
        L_last.T, y_st[nt - 1] - L_arrow_last.T @ x_fe, lower=False
    )

    x_st = jnp.zeros((nt, ns), dtype=dtype)
    x_st = x_st.at[nt - 1].set(x_last)

    def body_fn(i_rev, carry):
        x_st_, x_next = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i, L_arrow_i = reconstruct_L(i)
        x_i = jax.scipy.linalg.solve_triangular(
            L_i.T,
            y_st[i] - L_lower_i.T @ x_next - L_arrow_i.T @ x_fe,
            lower=False,
        )
        x_st_ = x_st_.at[i].set(x_i)
        return (x_st_, x_i)

    x_st, _ = lax.fori_loop(0, nt - 1, body_fn, (x_st, x_last))

    x = jnp.concatenate([x_st.reshape(-1), x_fe])
    return x, quad


def selected_inversion_grads_from_carries_jax(
    stored_cond_schurs, stored_arrow_schurs, L_tip,
    sc, jac_sc,
    nt, ns, n_fe, n_theta_st,
    likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip, dtype,
):
    """Fused selected inversion + gradient accumulation from stored carries.

    Like :func:`selected_inversion_grads_jax` but takes scan carries
    instead of L blocks.  L blocks are reconstructed on-the-fly from
    ``stored_cond_schurs`` and ``stored_arrow_schurs``, reducing peak
    memory from ~64 GiB (full L arrays) to ~32 GiB (carries only).

    Parameters
    ----------
    stored_cond_schurs : (nt, ns, ns)
        Schur complement entering each BTA Cholesky step.
    stored_arrow_schurs : (nt, n_fe, ns)
        Arrow Schur complement entering each step.
    L_tip : (n_fe, n_fe)
        Arrow tip Cholesky factor.
    sc : dict
        Output of :func:`precompute_spatial_components` (unpadded).
    jac_sc : dict
        Jacobians of sc w.r.t. theta_st (from ``jax.jacfwd``).
    nt, ns, n_fe, n_theta_st : int
    likelihood_prec : scalar
    ata_* : sparse COO data for AtA blocks
    ata_tip : (n_fe, n_fe)
    dtype : jnp.dtype

    Returns
    -------
    grad_st : (n_theta_st,)
    grad_lik : scalar
    """
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_ns = jnp.eye(ns, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    sc_padded = {**sc,
                 'm0_subdiag': jnp.concatenate([sc['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
                 'm1_subdiag': jnp.concatenate([sc['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
                 'm2_subdiag': jnp.concatenate([sc['m2_subdiag'], jnp.zeros(1, dtype=dtype)])}

    ata_lower_rows_pad = jnp.concatenate([
        ata_lower_rows, jnp.zeros((1, ata_lower_rows.shape[1]), dtype=jnp.int32)], axis=0)
    ata_lower_cols_pad = jnp.concatenate([
        ata_lower_cols, jnp.zeros((1, ata_lower_cols.shape[1]), dtype=jnp.int32)], axis=0)
    ata_lower_vals_pad = jnp.concatenate([
        ata_lower_vals, jnp.zeros((1, ata_lower_vals.shape[1]), dtype=dtype)], axis=0)

    scale = sc['scale']
    exp_gt = sc['exp_gt']
    m0_d = sc['m0_diag']
    m1_d = sc['m1_diag']
    m2_d = sc['m2_diag']
    m0_s = sc['m0_subdiag']
    m1_s = sc['m1_subdiag']
    m2_s = sc['m2_subdiag']

    base_mats = [sc['q3s'], sc['q2s'], sc['q1s']]
    jac_mats = []
    for k in range(n_theta_st):
        jac_mats.extend([
            jac_sc['q3s'][..., k],
            jac_sc['q2s'][..., k],
            jac_sc['q1s'][..., k],
        ])
    all_mats = jnp.stack(base_mats + jac_mats, axis=0)
    n_mats = all_mats.shape[0]

    def reconstruct_L(i):
        q_diag = _reconstruct_diag_block(sc_padded, i)
        q_diag = q_diag.at[ata_diag_rows[i], ata_diag_cols[i]].add(
            likelihood_prec * ata_diag_vals[i])
        q_diag = q_diag + eps_reg * eye_ns - stored_cond_schurs[i]
        L_i = _jax_cholesky(q_diag)

        q_lower = _reconstruct_lower_block(sc_padded, i)
        q_lower = q_lower.at[ata_lower_rows_pad[i], ata_lower_cols_pad[i]].add(
            likelihood_prec * ata_lower_vals_pad[i])
        L_lower_i = jax.scipy.linalg.solve_triangular(L_i, q_lower.T, lower=True).T

        q_arrow = jnp.zeros((n_fe, ns), dtype=dtype)
        q_arrow = q_arrow.at[ata_arrow_rows[i], ata_arrow_cols[i]].add(
            likelihood_prec * ata_arrow_vals[i])
        q_arrow = q_arrow - stored_arrow_schurs[i]
        L_arrow_i = jax.scipy.linalg.solve_triangular(L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    # --- S_tip ---
    L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, eye_nfe, lower=True)
    S_tip = L_tip_inv.T @ L_tip_inv

    # --- Last block (i = nt-1) ---
    L_last, _, L_arrow_last = reconstruct_L(nt - 1)
    L_blk_inv = jax.scipy.linalg.solve_triangular(L_last, eye_ns, lower=True)
    sa_last = -S_tip @ L_arrow_last @ L_blk_inv
    sd_last = (L_blk_inv.T - sa_last.T @ L_arrow_last) @ L_blk_inv

    traces_d = jnp.einsum('ij,sji->s', sd_last, all_mats)
    m_wt = jnp.tile(jnp.array([m0_d[nt - 1], m1_d[nt - 1], m2_d[nt - 1]], dtype=dtype),
                     1 + n_theta_st)
    acc_d = m_wt * traces_d
    acc_l = jnp.zeros(n_mats, dtype=dtype)

    sp_d = jnp.sum(sd_last[ata_diag_rows[nt - 1], ata_diag_cols[nt - 1]]
                   * ata_diag_vals[nt - 1])
    sp_l = jnp.array(0.0, dtype=dtype)
    sp_a = jnp.sum(sa_last[ata_arrow_rows[nt - 1], ata_arrow_cols[nt - 1]]
                   * ata_arrow_vals[nt - 1])

    # --- Backward loop ---
    def body_fn(i_rev, carry):
        sd_prev, sa_prev, acc_d_, acc_l_, sp_d_, sp_l_, sp_a_ = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i, L_arrow_i = reconstruct_L(i)
        L_blk_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_ns, lower=True)

        sl_i = (-sd_prev @ L_lower_i - sa_prev.T @ L_arrow_i) @ L_blk_inv_i
        sa_i = (-sa_prev @ L_lower_i - S_tip @ L_arrow_i) @ L_blk_inv_i
        sd_i = (L_blk_inv_i.T - sl_i.T @ L_lower_i - sa_i.T @ L_arrow_i) @ L_blk_inv_i

        tr_d = jnp.einsum('ij,sji->s', sd_i, all_mats)
        mw_d = jnp.tile(jnp.array([m0_d[i], m1_d[i], m2_d[i]], dtype=dtype),
                         1 + n_theta_st)
        acc_d_ = acc_d_ + mw_d * tr_d

        tr_l = jnp.einsum('ij,sji->s', sl_i, all_mats)
        mw_l = jnp.tile(jnp.array([m0_s[i], m1_s[i], m2_s[i]], dtype=dtype),
                         1 + n_theta_st)
        acc_l_ = acc_l_ + mw_l * tr_l

        sp_d_ = sp_d_ + jnp.sum(
            sd_i[ata_diag_rows[i], ata_diag_cols[i]] * ata_diag_vals[i])
        sp_l_ = sp_l_ + jnp.sum(
            sl_i[ata_lower_rows[i], ata_lower_cols[i]] * ata_lower_vals[i])
        sp_a_ = sp_a_ + jnp.sum(
            sa_i[ata_arrow_rows[i], ata_arrow_cols[i]] * ata_arrow_vals[i])

        return (sd_i, sa_i, acc_d_, acc_l_, sp_d_, sp_l_, sp_a_)

    init_carry = (sd_last, sa_last, acc_d, acc_l, sp_d, sp_l, sp_a)
    _, _, acc_d, acc_l, sp_d, sp_l, sp_a = lax.fori_loop(
        0, nt - 1, body_fn, init_carry
    )

    # Factor of 2: lower block S_l[t] appears twice (lower + upper transpose)
    acc_l = 2.0 * acc_l

    # --- Assemble grad_st ---
    jac_scale = jac_sc['scale']
    jac_exp_gt = jac_sc['exp_gt']

    weighted_diag = acc_d[0] + exp_gt * acc_d[1] + exp_gt**2 * acc_d[2]
    weighted_lower = acc_l[0] + exp_gt * acc_l[1] + exp_gt**2 * acc_l[2]

    grad_st = jnp.zeros(n_theta_st, dtype=dtype)
    for k in range(n_theta_st):
        term_scale = jac_scale[k] * (weighted_diag + weighted_lower)

        off = 3 + 3 * k
        term_spatial_diag = scale * (
            acc_d[off] + exp_gt * acc_d[off + 1] + exp_gt**2 * acc_d[off + 2])
        term_spatial_lower = scale * (
            acc_l[off] + exp_gt * acc_l[off + 1] + exp_gt**2 * acc_l[off + 2])

        term_exp_gt_diag = scale * jac_exp_gt[k] * (
            acc_d[1] + 2.0 * exp_gt * acc_d[2])
        term_exp_gt_lower = scale * jac_exp_gt[k] * (
            acc_l[1] + 2.0 * exp_gt * acc_l[2])

        grad_st = grad_st.at[k].set(
            term_scale + term_spatial_diag + term_spatial_lower
            + term_exp_gt_diag + term_exp_gt_lower
        )

    # --- Assemble grad_lik ---
    sp_tip = jnp.sum(S_tip * ata_tip)
    grad_lik = likelihood_prec * (sp_d + 2.0 * sp_l + 2.0 * sp_a + sp_tip)

    return grad_st, grad_lik


# =========================================================================
# Coregional memory-efficient functions
# =========================================================================


def extract_bta_blocks_sparse_coo_coregional(
    sparse_matrix,
    n_models: int,
    nt: int,
    ns: int,
    n_fixed_effects_total: int,
    dtype=None,
) -> dict:
    """Extract per-model BTA blocks from coregional A_i^T A_i in padded COO format.

    For a coregional model each A_i^T A_i only has nonzeros in the
    i-th (ns, ns) sub-block within each (n_models*ns, n_models*ns)
    super-block.  We extract COO triplets with local indices within
    that sub-block (rows/cols in [0, ns)), saving ~n_models^2x memory
    vs. storing full super-block dense blocks.

    Parameters
    ----------
    sparse_matrix : scipy sparse matrix
        Per-model A_i^T A_i (full size: n_latent x n_latent).
    n_models : int
    nt : int
    ns : int
        Per-model spatial block size.
    n_fixed_effects_total : int
    dtype : jnp.dtype, optional

    Returns
    -------
    dict
        Keys: ``ata_diag_{rows,cols,vals}``, ``ata_lower_{rows,cols,vals}``,
        ``ata_arrow_{rows,cols,vals}``, ``ata_tip``, ``model_offset``.
    """
    from scipy import sparse as sp_sparse
    from dalia.core.jax_autodiff import get_jax_dtype
    import numpy as np

    if dtype is None:
        dtype = get_jax_dtype()
    np_dtype = np.float64 if dtype == jnp.float64 else np.float32

    csc = sp_sparse.csc_matrix(sparse_matrix)
    block_size = n_models * ns
    total_st = nt * block_size

    def _pad(arr, length, fill=0):
        out = np.full(length, fill, dtype=arr.dtype)
        out[:len(arr)] = arr
        return out

    # Detect which sub-block offset has nonzeros
    detected_model = None
    for m in range(n_models):
        ms = m * ns
        me = ms + ns
        blk = csc[ms:me, ms:me]
        if blk.nnz > 0:
            detected_model = m
            break
    if detected_model is None:
        detected_model = 0
    m_off = detected_model * ns

    # --- Diagonal super-blocks: extract sub-block at (m*ns, m*ns) ---
    diag_coo_list = []
    for t in range(nt):
        s = t * block_size + m_off
        blk = csc[s:s + ns, s:s + ns].tocoo()
        diag_coo_list.append((blk.row.astype(np.int32),
                              blk.col.astype(np.int32),
                              blk.data.astype(np_dtype)))

    max_nnz_diag = max(len(v) for _, _, v in diag_coo_list) if diag_coo_list else 0
    if max_nnz_diag == 0:
        max_nnz_diag = 1

    ata_diag_rows = np.zeros((nt, max_nnz_diag), dtype=np.int32)
    ata_diag_cols = np.zeros((nt, max_nnz_diag), dtype=np.int32)
    ata_diag_vals = np.zeros((nt, max_nnz_diag), dtype=np_dtype)
    for i, (r, c, v) in enumerate(diag_coo_list):
        ata_diag_rows[i] = _pad(r, max_nnz_diag)
        ata_diag_cols[i] = _pad(c, max_nnz_diag)
        ata_diag_vals[i] = _pad(v, max_nnz_diag)

    # --- Lower-diagonal super-blocks ---
    lower_coo_list = []
    for t in range(nt - 1):
        rs = (t + 1) * block_size + m_off
        cs_start = t * block_size + m_off
        blk = csc[rs:rs + ns, cs_start:cs_start + ns].tocoo()
        lower_coo_list.append((blk.row.astype(np.int32),
                               blk.col.astype(np.int32),
                               blk.data.astype(np_dtype)))

    max_nnz_lower = max(len(v) for _, _, v in lower_coo_list) if lower_coo_list else 0
    if max_nnz_lower == 0:
        max_nnz_lower = 1

    ata_lower_rows = np.zeros((nt - 1, max_nnz_lower), dtype=np.int32)
    ata_lower_cols = np.zeros((nt - 1, max_nnz_lower), dtype=np.int32)
    ata_lower_vals = np.zeros((nt - 1, max_nnz_lower), dtype=np_dtype)
    for i, (r, c, v) in enumerate(lower_coo_list):
        ata_lower_rows[i] = _pad(r, max_nnz_lower)
        ata_lower_cols[i] = _pad(c, max_nnz_lower)
        ata_lower_vals[i] = _pad(v, max_nnz_lower)

    # --- Arrow-bottom blocks ---
    if n_fixed_effects_total > 0:
        arrow_coo_list = []
        for t in range(nt):
            cs_start = t * block_size + m_off
            blk = csc[total_st:, cs_start:cs_start + ns].tocoo()
            arrow_coo_list.append((blk.row.astype(np.int32),
                                   blk.col.astype(np.int32),
                                   blk.data.astype(np_dtype)))
        max_nnz_arrow = max(len(v) for _, _, v in arrow_coo_list) if arrow_coo_list else 0
        if max_nnz_arrow == 0:
            max_nnz_arrow = 1

        ata_arrow_rows = np.zeros((nt, max_nnz_arrow), dtype=np.int32)
        ata_arrow_cols = np.zeros((nt, max_nnz_arrow), dtype=np.int32)
        ata_arrow_vals = np.zeros((nt, max_nnz_arrow), dtype=np_dtype)
        for i, (r, c, v) in enumerate(arrow_coo_list):
            ata_arrow_rows[i] = _pad(r, max_nnz_arrow)
            ata_arrow_cols[i] = _pad(c, max_nnz_arrow)
            ata_arrow_vals[i] = _pad(v, max_nnz_arrow)

        ata_tip = jnp.array(csc[total_st:, total_st:].toarray(), dtype=dtype)
    else:
        max_nnz_arrow = 1
        ata_arrow_rows = np.zeros((nt, 1), dtype=np.int32)
        ata_arrow_cols = np.zeros((nt, 1), dtype=np.int32)
        ata_arrow_vals = np.zeros((nt, 1), dtype=np_dtype)
        ata_tip = jnp.zeros((max(n_fixed_effects_total, 1),
                              max(n_fixed_effects_total, 1)), dtype=dtype)

    return {
        'ata_diag_rows': jnp.array(ata_diag_rows),
        'ata_diag_cols': jnp.array(ata_diag_cols),
        'ata_diag_vals': jnp.array(ata_diag_vals, dtype=dtype),
        'ata_lower_rows': jnp.array(ata_lower_rows),
        'ata_lower_cols': jnp.array(ata_lower_cols),
        'ata_lower_vals': jnp.array(ata_lower_vals, dtype=dtype),
        'ata_arrow_rows': jnp.array(ata_arrow_rows),
        'ata_arrow_cols': jnp.array(ata_arrow_cols),
        'ata_arrow_vals': jnp.array(ata_arrow_vals, dtype=dtype),
        'ata_tip': ata_tip,
        'model_offset': detected_model,
    }


def precompute_spatial_components_coregional(
    theta, n_models, ns, models_data, hyperparameters_idx, theta_keys, manifolds,
):
    """Precompute spatial components for all models + coregional coefficients.

    Calls :func:`precompute_spatial_components` per model and extracts the
    coregional weight matrix ``w[i,j,m]`` from theta.

    Parameters
    ----------
    theta : jnp.ndarray
        Full hyperparameter vector.
    n_models : int
    ns : int
    models_data : list of dict
    hyperparameters_idx : list of int
    theta_keys : list of str
    manifolds : list of str

    Returns
    -------
    sc_list : list of dict
        Per-model spatial components (length n_models).
    coreg_w : jnp.ndarray
        Weight matrix, shape (n_models, n_models, n_models).
        ``coreg_w[i, j, m]`` = contribution of model m to sub-block (i, j).
    """
    sc_list = []
    for m in range(n_models):
        hp_start = hyperparameters_idx[m]
        hp_end = hyperparameters_idx[m + 1] - 1
        theta_model = theta[hp_start:hp_end]
        if theta_model.shape[0] == 2:
            theta_model = jnp.concatenate([theta_model, jnp.array([0.0])])

        sc_m = precompute_spatial_components(
            theta_model,
            models_data[m]['spatial_matrices'],
            models_data[m]['temporal_matrices'],
            manifolds[m],
        )
        sc_list.append(sc_m)

    # Extract sigmas and lambdas
    sigma_idx = theta_keys.index('sigma_0')
    sigmas = jnp.array([jnp.exp(theta[sigma_idx + i]) for i in range(n_models)])

    coreg_w = jnp.zeros((n_models, n_models, n_models), dtype=theta.dtype)

    if n_models == 2:
        lambda_01 = theta[theta_keys.index('lambda_0_1')]
        s0, s1 = sigmas[0], sigmas[1]
        # w[i, j, m]
        coreg_w = coreg_w.at[0, 0, 0].set(1.0 / s0**2)
        coreg_w = coreg_w.at[0, 0, 1].set(lambda_01**2 / s1**2)
        coreg_w = coreg_w.at[1, 0, 1].set(-lambda_01 / s1**2)
        coreg_w = coreg_w.at[0, 1, 1].set(-lambda_01 / s1**2)
        coreg_w = coreg_w.at[1, 1, 1].set(1.0 / s1**2)

    elif n_models == 3:
        lambda_01 = theta[theta_keys.index('lambda_0_1')]
        lambda_02 = theta[theta_keys.index('lambda_0_2')]
        lambda_12 = theta[theta_keys.index('lambda_1_2')]
        s0, s1, s2 = sigmas[0], sigmas[1], sigmas[2]

        coreg_w = coreg_w.at[0, 0, 0].set(1.0 / s0**2)
        coreg_w = coreg_w.at[0, 0, 1].set(lambda_01**2 / s1**2)
        coreg_w = coreg_w.at[0, 0, 2].set(lambda_12**2 / s2**2)

        coreg_w = coreg_w.at[1, 0, 1].set(-lambda_01 / s1**2)
        coreg_w = coreg_w.at[0, 1, 1].set(-lambda_01 / s1**2)
        coreg_w = coreg_w.at[1, 0, 2].set(lambda_02 * lambda_12 / s2**2)
        coreg_w = coreg_w.at[0, 1, 2].set(lambda_02 * lambda_12 / s2**2)

        coreg_w = coreg_w.at[2, 0, 2].set(-lambda_12 / s2**2)
        coreg_w = coreg_w.at[0, 2, 2].set(-lambda_12 / s2**2)

        coreg_w = coreg_w.at[1, 1, 1].set(1.0 / s1**2)
        coreg_w = coreg_w.at[1, 1, 2].set(lambda_02**2 / s2**2)

        coreg_w = coreg_w.at[2, 1, 2].set(-lambda_02 / s2**2)
        coreg_w = coreg_w.at[1, 2, 2].set(-lambda_02 / s2**2)

        coreg_w = coreg_w.at[2, 2, 2].set(1.0 / s2**2)

    return sc_list, coreg_w


def _reconstruct_coregional_diag_block(sc_list, coreg_w, n_models, ns, t):
    """Reconstruct diagonal super-block t of coregional Q_prior.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components.
    coreg_w : (n_models, n_models, n_models)
        Coregional weight matrix.
    n_models, ns : int
    t : int or scalar
        Temporal block index.

    Returns
    -------
    block : (n_models*ns, n_models*ns)
    """
    block_size = n_models * ns
    dtype = sc_list[0]['q3s'].dtype
    block = jnp.zeros((block_size, block_size), dtype=dtype)

    for i in range(n_models):
        for j in range(i + 1):
            sub = jnp.zeros((ns, ns), dtype=dtype)
            for m_idx in range(n_models):
                sub = sub + coreg_w[i, j, m_idx] * _reconstruct_diag_block(sc_list[m_idx], t)
            block = block.at[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns].set(sub)
            if i != j:
                block = block.at[j * ns:(j + 1) * ns, i * ns:(i + 1) * ns].set(sub)

    return block


def _reconstruct_coregional_lower_block(sc_list, coreg_w, n_models, ns, t):
    """Reconstruct lower-diagonal super-block t of coregional Q_prior.

    Parameters
    ----------
    sc_list : list of dict
    coreg_w : (n_models, n_models, n_models)
    n_models, ns : int
    t : int or scalar

    Returns
    -------
    block : (n_models*ns, n_models*ns)
    """
    block_size = n_models * ns
    dtype = sc_list[0]['q3s'].dtype
    block = jnp.zeros((block_size, block_size), dtype=dtype)

    for i in range(n_models):
        for j in range(n_models):
            sub = jnp.zeros((ns, ns), dtype=dtype)
            for m_idx in range(n_models):
                sub = sub + coreg_w[i, j, m_idx] * _reconstruct_lower_block(sc_list[m_idx], t)
            block = block.at[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns].set(sub)

    return block


def fused_cholesky_fwd_sub_coregional(
    sc_list, coreg_w, n_models, nt, ns, n_fe, fe_prec,
    likelihood_precs,
    rhs_st, rhs_fe,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_ata_tip, per_model_offsets,
    dtype,
):
    """Fused BTA Cholesky + forward substitution for coregional models.

    Combines the Cholesky factorization and forward solve into one
    :func:`lax.scan`, with super-blocks reconstructed on the fly from
    per-model spatial components.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components (length n_models), padded subdiags.
    coreg_w : (n_models, n_models, n_models)
    n_models, nt, ns, n_fe : int
    fe_prec : float
    likelihood_precs : (n_models,)
    rhs_st : (nt, block_size)
    rhs_fe : (n_fe,)
    per_model_ata_diag_{rows,cols,vals} : list of (nt, max_nnz_m)
    per_model_ata_lower_{rows,cols,vals} : list of (nt, max_nnz_m)
    per_model_ata_arrow_{rows,cols,vals} : list of (nt, max_nnz_m)
    per_model_ata_tip : list of (n_fe, n_fe)
    per_model_offsets : list of int
    dtype : jnp.dtype

    Returns
    -------
    stored_cond_schurs : (nt, block_size, block_size)
    stored_arrow_schurs : (nt, n_fe, block_size)
    y_st : (nt, block_size)
    L_tip : (n_fe, n_fe)
    arrow_rhs_acc : (n_fe,)
    logdet_Q_cond : scalar
    """
    block_size = n_models * ns
    eps = jnp.finfo(dtype).eps
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_bs = jnp.eye(block_size, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)

    # Pad lower AtA arrays to length nt for each model
    padded_lower_rows = []
    padded_lower_cols = []
    padded_lower_vals = []
    for m in range(n_models):
        lr = per_model_ata_lower_rows[m]
        lc = per_model_ata_lower_cols[m]
        lv = per_model_ata_lower_vals[m]
        padded_lower_rows.append(jnp.concatenate([
            lr, jnp.zeros((1, lr.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_cols.append(jnp.concatenate([
            lc, jnp.zeros((1, lc.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_vals.append(jnp.concatenate([
            lv, jnp.zeros((1, lv.shape[1]), dtype=dtype)], axis=0))

    # Pad sc_list subdiags
    sc_list_padded = []
    for m in range(n_models):
        sc_m = sc_list[m]
        sc_list_padded.append({
            **sc_m,
            'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
        })

    # Arrow tip initial: fe_prec * I + sum_m prec_m * ata_tip_m
    arrow_tip_init = fe_prec * eye_nfe + eps_reg * eye_nfe
    for m in range(n_models):
        arrow_tip_init = arrow_tip_init + likelihood_precs[m] * per_model_ata_tip[m]

    def scan_body(carry, inputs):
        (cond_schur, arrow_tip_acc, arrow_schur, logdet_cond,
         prev_lower_y, arrow_rhs_acc) = carry
        i = inputs[0]
        rhs_i = inputs[1]

        saved_carries = (cond_schur, arrow_schur)

        # Reconstruct Q_prior super-block on the fly
        q_cond_diag_i = _reconstruct_coregional_diag_block(
            sc_list_padded, coreg_w, n_models, ns, i)

        # Scatter per-model sparse AtA
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            d_rows = per_model_ata_diag_rows[m][i]
            d_cols = per_model_ata_diag_cols[m][i]
            d_vals = per_model_ata_diag_vals[m][i]
            q_cond_diag_i = q_cond_diag_i.at[
                m_off + d_rows, m_off + d_cols
            ].add(likelihood_precs[m] * d_vals)

        q_cond_diag_i = q_cond_diag_i + eps_reg * eye_bs - cond_schur

        L_i = _jax_cholesky(q_cond_diag_i)
        cond_diag_vals = jnp.diag(L_i)
        safe_cond = jnp.maximum(cond_diag_vals, eps)
        logdet_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_cond))

        # Lower block
        q_cond_lower_i = _reconstruct_coregional_lower_block(
            sc_list_padded, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            l_rows = padded_lower_rows[m][i]
            l_cols = padded_lower_cols[m][i]
            l_vals = padded_lower_vals[m][i]
            q_cond_lower_i = q_cond_lower_i.at[
                m_off + l_rows, m_off + l_cols
            ].add(likelihood_precs[m] * l_vals)

        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_cond_lower_i.T, lower=True).T

        # Arrow block
        q_arrow_i = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            a_rows = per_model_ata_arrow_rows[m][i]
            a_cols = per_model_ata_arrow_cols[m][i]
            a_vals = per_model_ata_arrow_vals[m][i]
            q_arrow_i = q_arrow_i.at[a_rows, m_off + a_cols].add(
                likelihood_precs[m] * a_vals)
        q_arrow_i = q_arrow_i - arrow_schur
        L_arrow_i = jax.scipy.linalg.solve_triangular(
            L_i, q_arrow_i.T, lower=True).T

        new_cond_schur = L_lower_i @ L_lower_i.T
        new_arrow_schur = L_arrow_i @ L_lower_i.T
        new_arrow_tip_acc = arrow_tip_acc - L_arrow_i @ L_arrow_i.T

        new_cond_schur = jnp.where(i < nt - 1, new_cond_schur,
                                    jnp.zeros_like(new_cond_schur))
        new_arrow_schur = jnp.where(i < nt - 1, new_arrow_schur,
                                     jnp.zeros_like(new_arrow_schur))

        # Forward substitution
        modified_rhs_i = rhs_i - prev_lower_y
        y_i = jax.scipy.linalg.solve_triangular(L_i, modified_rhs_i, lower=True)

        new_prev_lower_y = L_lower_i @ y_i
        new_prev_lower_y = jnp.where(i < nt - 1, new_prev_lower_y,
                                      jnp.zeros_like(new_prev_lower_y))
        new_arrow_rhs_acc = arrow_rhs_acc - L_arrow_i @ y_i

        new_carry = (new_cond_schur, new_arrow_tip_acc, new_arrow_schur,
                     logdet_cond, new_prev_lower_y, new_arrow_rhs_acc)
        return new_carry, (saved_carries, y_i)

    init_carry = (
        jnp.zeros((block_size, block_size), dtype=dtype),
        arrow_tip_init,
        jnp.zeros((n_fe, block_size), dtype=dtype),
        jnp.array(0.0, dtype=dtype),
        jnp.zeros(block_size, dtype=dtype),
        rhs_fe,
    )

    scan_inputs = (jnp.arange(nt), rhs_st)

    (_, arrow_tip_final, _, logdet_cond, _, arrow_rhs_acc), \
        ((stored_cond_schurs, stored_arrow_schurs), y_st) = lax.scan(
            scan_body, init_carry, scan_inputs
        )

    L_tip = _jax_cholesky(arrow_tip_final)
    tip_diag = jnp.diag(L_tip)
    safe_tip_diag = jnp.maximum(tip_diag, eps)
    logdet_Q_cond = logdet_cond + 2.0 * jnp.sum(jnp.log(safe_tip_diag))

    return stored_cond_schurs, stored_arrow_schurs, y_st, L_tip, arrow_rhs_acc, logdet_Q_cond


def backward_sub_from_carries_coregional(
    stored_cond_schurs, stored_arrow_schurs, L_tip,
    y_st, arrow_rhs_acc,
    sc_list, coreg_w, likelihood_precs,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_offsets,
    n_models, nt, ns, n_fe, dtype,
):
    """Backward substitution for coregional, reconstructing L from carries.

    Parameters
    ----------
    stored_cond_schurs : (nt, block_size, block_size)
    stored_arrow_schurs : (nt, n_fe, block_size)
    L_tip : (n_fe, n_fe)
    y_st : (nt, block_size)
    arrow_rhs_acc : (n_fe,)
    sc_list : list of dict
        Per-model spatial components (unpadded).
    coreg_w : (n_models, n_models, n_models)
    likelihood_precs : (n_models,)
    per_model_ata_* : per-model sparse COO data
    per_model_offsets : list of int
    n_models, nt, ns, n_fe : int
    dtype : jnp.dtype

    Returns
    -------
    x : (nt * block_size + n_fe,)
    quad : scalar
    """
    block_size = n_models * ns
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_bs = jnp.eye(block_size, dtype=dtype)

    # Pad sc_list and lower AtA
    sc_padded = []
    for m in range(n_models):
        sc_m = sc_list[m]
        sc_padded.append({
            **sc_m,
            'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
        })

    padded_lower_rows = []
    padded_lower_cols = []
    padded_lower_vals = []
    for m in range(n_models):
        lr = per_model_ata_lower_rows[m]
        lc = per_model_ata_lower_cols[m]
        lv = per_model_ata_lower_vals[m]
        padded_lower_rows.append(jnp.concatenate([
            lr, jnp.zeros((1, lr.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_cols.append(jnp.concatenate([
            lc, jnp.zeros((1, lc.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_vals.append(jnp.concatenate([
            lv, jnp.zeros((1, lv.shape[1]), dtype=dtype)], axis=0))

    def reconstruct_L(i):
        q_diag = _reconstruct_coregional_diag_block(sc_padded, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_diag = q_diag.at[
                m_off + per_model_ata_diag_rows[m][i],
                m_off + per_model_ata_diag_cols[m][i]
            ].add(likelihood_precs[m] * per_model_ata_diag_vals[m][i])
        q_diag = q_diag + eps_reg * eye_bs - stored_cond_schurs[i]
        L_i = _jax_cholesky(q_diag)

        q_lower = _reconstruct_coregional_lower_block(sc_padded, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_lower = q_lower.at[
                m_off + padded_lower_rows[m][i],
                m_off + padded_lower_cols[m][i]
            ].add(likelihood_precs[m] * padded_lower_vals[m][i])
        L_lower_i = jax.scipy.linalg.solve_triangular(L_i, q_lower.T, lower=True).T

        q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_arrow = q_arrow.at[
                per_model_ata_arrow_rows[m][i],
                m_off + per_model_ata_arrow_cols[m][i]
            ].add(likelihood_precs[m] * per_model_ata_arrow_vals[m][i])
        q_arrow = q_arrow - stored_arrow_schurs[i]
        L_arrow_i = jax.scipy.linalg.solve_triangular(L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    # Forward sub on tip + quadratic form
    y_fe = jax.scipy.linalg.solve_triangular(L_tip, arrow_rhs_acc, lower=True)
    quad = jnp.sum(y_st ** 2) + jnp.sum(y_fe ** 2)

    # Backward sub
    x_fe = jax.scipy.linalg.solve_triangular(L_tip.T, y_fe, lower=False)

    L_last, _, L_arrow_last = reconstruct_L(nt - 1)
    x_last = jax.scipy.linalg.solve_triangular(
        L_last.T, y_st[nt - 1] - L_arrow_last.T @ x_fe, lower=False)

    x_st = jnp.zeros((nt, block_size), dtype=dtype)
    x_st = x_st.at[nt - 1].set(x_last)

    def body_fn(i_rev, carry):
        x_st_, x_next = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i, L_arrow_i = reconstruct_L(i)
        x_i = jax.scipy.linalg.solve_triangular(
            L_i.T,
            y_st[i] - L_lower_i.T @ x_next - L_arrow_i.T @ x_fe,
            lower=False,
        )
        x_st_ = x_st_.at[i].set(x_i)
        return (x_st_, x_i)

    x_st, _ = lax.fori_loop(0, nt - 1, body_fn, (x_st, x_last))

    x = jnp.concatenate([x_st.reshape(-1), x_fe])
    return x, quad


def logdet_Q_prior_coregional_scan(
    sc_list, coreg_w, n_models, ns, nt, dtype,
):
    """Compute logdet(Q_prior) for coregional model via BT Cholesky scan.

    Q_prior has no arrowhead, so this is a simple BT Cholesky.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components (padded subdiags).
    coreg_w : (n_models, n_models, n_models)
    n_models, ns, nt : int
    dtype : jnp.dtype

    Returns
    -------
    logdet : scalar
    """
    block_size = n_models * ns
    eps = jnp.finfo(dtype).eps

    @partial(jax.checkpoint, prevent_cse=True)
    def scan_body(carry, i):
        schur, logdet = carry
        q_diag_i = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, i) - schur
        L_i = _jax_cholesky(q_diag_i)
        diag_vals = jnp.diag(L_i)
        safe_vals = jnp.maximum(diag_vals, eps)
        logdet = logdet + 2.0 * jnp.sum(jnp.log(safe_vals))

        q_lower_i = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, i)
        L_inv_lower = jax.scipy.linalg.solve_triangular(
            L_i, q_lower_i.T, lower=True)
        new_schur = L_inv_lower.T @ L_inv_lower
        new_schur = jnp.where(i < nt - 1, new_schur, jnp.zeros_like(new_schur))
        return (new_schur, logdet), None

    init_carry = (jnp.zeros((block_size, block_size), dtype=dtype),
                  jnp.array(0.0, dtype=dtype))
    (_, logdet), _ = lax.scan(scan_body, init_carry, jnp.arange(nt))
    return logdet


def logdet_Q_prior_coregional_grad(
    sc_list, jac_sc_list, coreg_w, jac_coreg_w,
    n_models, ns, nt, dtype,
):
    """Analytical gradient of logdet(Q_prior) for coregional models.

    Uses forward BT Cholesky storing Schur carries, then backward BT
    selected inversion to accumulate tr(Sigma @ dQ/dtheta) block-by-block.

    Parameters
    ----------
    sc_list : list of dict
        Per-model spatial components (padded subdiags).
    jac_sc_list : list of dict
        Per-model Jacobians of spatial components w.r.t. their theta_st_m.
    coreg_w : (n_models, n_models, n_models)
    jac_coreg_w : (n_models, n_models, n_models, n_coreg_params)
        Jacobian of coreg_w w.r.t. coregional params.
    n_models, ns, nt : int
    dtype : jnp.dtype

    Returns
    -------
    grad_per_model_st : list of (n_theta_st_m,) arrays
    grad_coreg : (n_coreg_params,)
    """
    block_size = n_models * ns
    eye_bs = jnp.eye(block_size, dtype=dtype)
    n_coreg = jac_coreg_w.shape[-1]

    # --- Forward BT Cholesky scan, store incoming Schurs ---
    def fwd_body(schur, i):
        q_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, i) - schur
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        new_schur = L_lower_i @ L_lower_i.T
        new_schur = jnp.where(i < nt - 1, new_schur, jnp.zeros_like(new_schur))
        return new_schur, schur

    init_schur = jnp.zeros((block_size, block_size), dtype=dtype)
    _, stored_schurs = lax.scan(fwd_body, init_schur, jnp.arange(nt))

    def _reconstruct_L(i):
        q_diag = _reconstruct_coregional_diag_block(
            sc_list, coreg_w, n_models, ns, i) - stored_schurs[i]
        L_i = _jax_cholesky(q_diag)
        q_lower = _reconstruct_coregional_lower_block(
            sc_list, coreg_w, n_models, ns, i)
        L_lower_i = jax.scipy.linalg.solve_triangular(
            L_i, q_lower.T, lower=True).T
        return L_i, L_lower_i

    # --- Accumulate traces via backward BT selected inversion ---
    # For each sub-block (i,j), tr(S_ij[t] @ Qu_m[t]) contributes to both
    # ST params (via dQu_m/dtheta_st_m) and coreg params (via dw_ijm/dparam).

    # Initialize accumulators (stacked for lax.fori_loop compatibility)
    grad_per_model_st = jnp.zeros((n_models, 3), dtype=dtype)
    grad_coreg = jnp.zeros(n_coreg, dtype=dtype)

    def _accumulate_traces(sd_i, sl_i, t_diag, t_lower):
        """Accumulate gradient contributions from S_diag=sd_i and S_lower=sl_i at time step."""
        g_st = jnp.zeros((n_models, 3), dtype=dtype)
        g_c = jnp.zeros(n_coreg, dtype=dtype)

        for ii in range(n_models):
            for jj in range(n_models):
                # Extract sub-block of Sigma
                sd_ij = sd_i[ii * ns:(ii + 1) * ns, jj * ns:(jj + 1) * ns]

                for m_idx in range(n_models):
                    w_ijm = coreg_w[ii, jj, m_idx]
                    sc_m = sc_list[m_idx]

                    # Trace with per-model reconstruction components
                    qu_diag_m = _reconstruct_diag_block(sc_m, t_diag)
                    tr_val = jnp.sum(sd_ij * qu_diag_m.T)

                    # Coreg gradient: dw_ijm/dparam * tr(S_ij @ Qu_m)
                    dw = jac_coreg_w[ii, jj, m_idx, :]
                    g_c = g_c + dw * tr_val

                    # ST gradient for model m: w_ijm * tr(S_ij @ dQu_m/dtheta_st_m)
                    jsc_m = jac_sc_list[m_idx]
                    scale_m = sc_m['scale']
                    exp_gt_m = sc_m['exp_gt']
                    m0_d_val = sc_m['m0_diag'][t_diag]
                    m1_d_val = sc_m['m1_diag'][t_diag]
                    m2_d_val = sc_m['m2_diag'][t_diag]

                    for k in range(3):
                        dq3s = jsc_m['q3s'][..., k]
                        dq2s = jsc_m['q2s'][..., k]
                        dq1s = jsc_m['q1s'][..., k]
                        d_scale = jsc_m['scale'][k]
                        d_exp_gt = jsc_m['exp_gt'][k]

                        dQu_diag = (
                            d_scale * (m0_d_val * sc_m['q3s'] + exp_gt_m * m1_d_val * sc_m['q2s']
                                       + exp_gt_m**2 * m2_d_val * sc_m['q1s'])
                            + scale_m * (m0_d_val * dq3s + exp_gt_m * m1_d_val * dq2s
                                         + exp_gt_m**2 * m2_d_val * dq1s)
                            + scale_m * d_exp_gt * (m1_d_val * sc_m['q2s']
                                                    + 2.0 * exp_gt_m * m2_d_val * sc_m['q1s'])
                        )
                        tr_jac = jnp.sum(sd_ij * dQu_diag.T)
                        g_st = g_st.at[m_idx, k].add(w_ijm * tr_jac)

                # Lower block contribution (only if sl_i is provided)
                if t_lower is not None:
                    sl_ij = sl_i[ii * ns:(ii + 1) * ns, jj * ns:(jj + 1) * ns]
                    for m_idx in range(n_models):
                        w_ijm = coreg_w[ii, jj, m_idx]
                        sc_m = sc_list[m_idx]

                        qu_lower_m = _reconstruct_lower_block(sc_m, t_lower)
                        tr_val_l = jnp.sum(sl_ij * qu_lower_m.T)

                        dw = jac_coreg_w[ii, jj, m_idx, :]
                        g_c = g_c + 2.0 * dw * tr_val_l

                        jsc_m = jac_sc_list[m_idx]
                        scale_m = sc_m['scale']
                        exp_gt_m = sc_m['exp_gt']
                        m0_s_val = sc_m['m0_subdiag'][t_lower]
                        m1_s_val = sc_m['m1_subdiag'][t_lower]
                        m2_s_val = sc_m['m2_subdiag'][t_lower]

                        for k in range(3):
                            dq3s = jsc_m['q3s'][..., k]
                            dq2s = jsc_m['q2s'][..., k]
                            dq1s = jsc_m['q1s'][..., k]
                            d_scale = jsc_m['scale'][k]
                            d_exp_gt = jsc_m['exp_gt'][k]

                            dQu_lower = (
                                d_scale * (m0_s_val * sc_m['q3s'] + exp_gt_m * m1_s_val * sc_m['q2s']
                                           + exp_gt_m**2 * m2_s_val * sc_m['q1s'])
                                + scale_m * (m0_s_val * dq3s + exp_gt_m * m1_s_val * dq2s
                                             + exp_gt_m**2 * m2_s_val * dq1s)
                                + scale_m * d_exp_gt * (m1_s_val * sc_m['q2s']
                                                        + 2.0 * exp_gt_m * m2_s_val * sc_m['q1s'])
                            )
                            tr_jac_l = jnp.sum(sl_ij * dQu_lower.T)
                            g_st = g_st.at[m_idx, k].add(2.0 * w_ijm * tr_jac_l)

        return g_st, g_c

    # --- Last block ---
    L_last, _ = _reconstruct_L(nt - 1)
    L_inv_last = jax.scipy.linalg.solve_triangular(L_last, eye_bs, lower=True)
    sd_last = L_inv_last.T @ L_inv_last

    g_st_last, g_c_last = _accumulate_traces(sd_last, None, nt - 1, None)
    grad_per_model_st = grad_per_model_st + g_st_last
    grad_coreg = grad_coreg + g_c_last

    # --- Backward loop via lax.fori_loop ---
    def bwd_body(i_rev, carry):
        sd_prev, grad_st_acc, grad_c_acc = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i = _reconstruct_L(i)
        L_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_bs, lower=True)
        sl_i = -sd_prev @ L_lower_i @ L_inv_i
        sd_i = (L_inv_i.T - sl_i.T @ L_lower_i) @ L_inv_i

        g_st_i, g_c_i = _accumulate_traces(sd_i, sl_i, i, i)
        return (sd_i, grad_st_acc + g_st_i, grad_c_acc + g_c_i)

    carry = (sd_last, grad_per_model_st, grad_coreg)
    _, grad_per_model_st, grad_coreg = lax.fori_loop(0, nt - 1, bwd_body, carry)

    return [grad_per_model_st[m] for m in range(n_models)], grad_coreg


def selected_inversion_grads_from_carries_coregional(
    stored_cond_schurs, stored_arrow_schurs, L_tip,
    sc_list, jac_sc_list, coreg_w, jac_coreg_w,
    n_models, nt, ns, n_fe,
    likelihood_precs,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_ata_tip, per_model_offsets,
    dtype,
):
    """Fused selected inversion + gradient accumulation for coregional models.

    Reconstructs L blocks on-the-fly from stored carries and accumulates
    gradient traces for all hyperparameter groups:
    - Per-model ST params (r_s_m, r_t_m, sigma_st_m)
    - Per-model likelihood precision
    - Coregional params (sigmas, lambdas)

    Parameters
    ----------
    stored_cond_schurs : (nt, block_size, block_size)
    stored_arrow_schurs : (nt, n_fe, block_size)
    L_tip : (n_fe, n_fe)
    sc_list : list of dict
        Per-model spatial components (padded subdiags).
    jac_sc_list : list of dict
        Per-model Jacobians from jacfwd.
    coreg_w : (n_models, n_models, n_models)
    jac_coreg_w : (n_models, n_models, n_models, n_coreg_params)
    n_models, nt, ns, n_fe : int
    likelihood_precs : (n_models,)
    per_model_ata_* : per-model sparse COO data
    per_model_ata_tip : list of (n_fe, n_fe)
    per_model_offsets : list of int
    dtype : jnp.dtype

    Returns
    -------
    grad_per_model_st : list of (3,) arrays
    grad_per_model_lik : (n_models,)
    grad_coreg : (n_coreg_params,)
    """
    block_size = n_models * ns
    is_fp32 = (dtype == jnp.float32)
    eps_reg = jnp.where(is_fp32, 1e-4, 0.0)
    eye_bs = jnp.eye(block_size, dtype=dtype)
    eye_nfe = jnp.eye(n_fe, dtype=dtype)
    n_coreg = jac_coreg_w.shape[-1]

    # Pad lower AtA
    padded_lower_rows = []
    padded_lower_cols = []
    padded_lower_vals = []
    for m in range(n_models):
        lr = per_model_ata_lower_rows[m]
        lc = per_model_ata_lower_cols[m]
        lv = per_model_ata_lower_vals[m]
        padded_lower_rows.append(jnp.concatenate([
            lr, jnp.zeros((1, lr.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_cols.append(jnp.concatenate([
            lc, jnp.zeros((1, lc.shape[1]), dtype=jnp.int32)], axis=0))
        padded_lower_vals.append(jnp.concatenate([
            lv, jnp.zeros((1, lv.shape[1]), dtype=dtype)], axis=0))

    def reconstruct_L(i):
        q_diag = _reconstruct_coregional_diag_block(sc_list, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_diag = q_diag.at[
                m_off + per_model_ata_diag_rows[m][i],
                m_off + per_model_ata_diag_cols[m][i]
            ].add(likelihood_precs[m] * per_model_ata_diag_vals[m][i])
        q_diag = q_diag + eps_reg * eye_bs - stored_cond_schurs[i]
        L_i = _jax_cholesky(q_diag)

        q_lower = _reconstruct_coregional_lower_block(sc_list, coreg_w, n_models, ns, i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_lower = q_lower.at[
                m_off + padded_lower_rows[m][i],
                m_off + padded_lower_cols[m][i]
            ].add(likelihood_precs[m] * padded_lower_vals[m][i])
        L_lower_i = jax.scipy.linalg.solve_triangular(L_i, q_lower.T, lower=True).T

        q_arrow = jnp.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            q_arrow = q_arrow.at[
                per_model_ata_arrow_rows[m][i],
                m_off + per_model_ata_arrow_cols[m][i]
            ].add(likelihood_precs[m] * per_model_ata_arrow_vals[m][i])
        q_arrow = q_arrow - stored_arrow_schurs[i]
        L_arrow_i = jax.scipy.linalg.solve_triangular(L_i, q_arrow.T, lower=True).T

        return L_i, L_lower_i, L_arrow_i

    def _accumulate_block_grads(sd_i, sl_i, sa_i, t_diag, t_lower):
        """Accumulate gradient contributions from Sigma blocks at time step."""
        g_st = jnp.zeros((n_models, 3), dtype=dtype)
        g_lik = jnp.zeros(n_models, dtype=dtype)
        g_c = jnp.zeros(n_coreg, dtype=dtype)

        # ST and coreg gradients from diagonal super-block
        for ii in range(n_models):
            for jj in range(n_models):
                sd_ij = sd_i[ii * ns:(ii + 1) * ns, jj * ns:(jj + 1) * ns]

                for m_idx in range(n_models):
                    w_ijm = coreg_w[ii, jj, m_idx]
                    sc_m = sc_list[m_idx]

                    qu_diag_m = _reconstruct_diag_block(sc_m, t_diag)
                    tr_val = jnp.sum(sd_ij * qu_diag_m.T)

                    # Coreg gradient
                    dw = jac_coreg_w[ii, jj, m_idx, :]
                    g_c = g_c + dw * tr_val

                    # ST gradient
                    jsc_m = jac_sc_list[m_idx]
                    scale_m = sc_m['scale']
                    exp_gt_m = sc_m['exp_gt']
                    m0_d_val = sc_m['m0_diag'][t_diag]
                    m1_d_val = sc_m['m1_diag'][t_diag]
                    m2_d_val = sc_m['m2_diag'][t_diag]

                    for k in range(3):
                        dq3s = jsc_m['q3s'][..., k]
                        dq2s = jsc_m['q2s'][..., k]
                        dq1s = jsc_m['q1s'][..., k]
                        d_scale = jsc_m['scale'][k]
                        d_exp_gt = jsc_m['exp_gt'][k]

                        dQu = (
                            d_scale * (m0_d_val * sc_m['q3s'] + exp_gt_m * m1_d_val * sc_m['q2s']
                                       + exp_gt_m**2 * m2_d_val * sc_m['q1s'])
                            + scale_m * (m0_d_val * dq3s + exp_gt_m * m1_d_val * dq2s
                                         + exp_gt_m**2 * m2_d_val * dq1s)
                            + scale_m * d_exp_gt * (m1_d_val * sc_m['q2s']
                                                    + 2.0 * exp_gt_m * m2_d_val * sc_m['q1s'])
                        )
                        g_st = g_st.at[m_idx, k].add(
                            w_ijm * jnp.sum(sd_ij * dQu.T))

                # Lower block contribution
                if sl_i is not None and t_lower is not None:
                    sl_ij = sl_i[ii * ns:(ii + 1) * ns, jj * ns:(jj + 1) * ns]
                    for m_idx in range(n_models):
                        w_ijm = coreg_w[ii, jj, m_idx]
                        sc_m = sc_list[m_idx]
                        qu_lower_m = _reconstruct_lower_block(sc_m, t_lower)
                        tr_val_l = jnp.sum(sl_ij * qu_lower_m.T)

                        dw = jac_coreg_w[ii, jj, m_idx, :]
                        g_c = g_c + 2.0 * dw * tr_val_l

                        jsc_m = jac_sc_list[m_idx]
                        scale_m = sc_m['scale']
                        exp_gt_m = sc_m['exp_gt']
                        m0_s_val = sc_m['m0_subdiag'][t_lower]
                        m1_s_val = sc_m['m1_subdiag'][t_lower]
                        m2_s_val = sc_m['m2_subdiag'][t_lower]

                        for k in range(3):
                            dq3s = jsc_m['q3s'][..., k]
                            dq2s = jsc_m['q2s'][..., k]
                            dq1s = jsc_m['q1s'][..., k]
                            d_scale = jsc_m['scale'][k]
                            d_exp_gt = jsc_m['exp_gt'][k]

                            dQu_l = (
                                d_scale * (m0_s_val * sc_m['q3s'] + exp_gt_m * m1_s_val * sc_m['q2s']
                                           + exp_gt_m**2 * m2_s_val * sc_m['q1s'])
                                + scale_m * (m0_s_val * dq3s + exp_gt_m * m1_s_val * dq2s
                                             + exp_gt_m**2 * m2_s_val * dq1s)
                                + scale_m * d_exp_gt * (m1_s_val * sc_m['q2s']
                                                        + 2.0 * exp_gt_m * m2_s_val * sc_m['q1s'])
                            )
                            g_st = g_st.at[m_idx, k].add(
                                2.0 * w_ijm * jnp.sum(sl_ij * dQu_l.T))

        # Likelihood gradient from sparse AtA
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            # Diagonal AtA contribution
            sp_d = jnp.sum(
                sd_i[m_off + per_model_ata_diag_rows[m][t_diag],
                     m_off + per_model_ata_diag_cols[m][t_diag]]
                * per_model_ata_diag_vals[m][t_diag])
            g_lik = g_lik.at[m].add(sp_d)

            # Arrow AtA contribution
            if sa_i is not None:
                sp_a = jnp.sum(
                    sa_i[per_model_ata_arrow_rows[m][t_diag],
                         m_off + per_model_ata_arrow_cols[m][t_diag]]
                    * per_model_ata_arrow_vals[m][t_diag])
                g_lik = g_lik.at[m].add(2.0 * sp_a)

            # Lower AtA contribution
            if sl_i is not None and t_lower is not None:
                sp_l = jnp.sum(
                    sl_i[m_off + padded_lower_rows[m][t_lower],
                         m_off + padded_lower_cols[m][t_lower]]
                    * padded_lower_vals[m][t_lower])
                g_lik = g_lik.at[m].add(2.0 * sp_l)

        return g_st, g_lik, g_c

    # --- S_tip ---
    L_tip_inv = jax.scipy.linalg.solve_triangular(L_tip, eye_nfe, lower=True)
    S_tip = L_tip_inv.T @ L_tip_inv

    # --- Last block ---
    L_last, _, L_arrow_last = reconstruct_L(nt - 1)
    L_blk_inv = jax.scipy.linalg.solve_triangular(L_last, eye_bs, lower=True)
    sa_last = -S_tip @ L_arrow_last @ L_blk_inv
    sd_last = (L_blk_inv.T - sa_last.T @ L_arrow_last) @ L_blk_inv

    g_st_last, g_lik_last, g_c_last = _accumulate_block_grads(
        sd_last, None, sa_last, nt - 1, None)

    # Tip contribution to likelihood
    for m in range(n_models):
        g_lik_last = g_lik_last.at[m].add(jnp.sum(S_tip * per_model_ata_tip[m]))

    grad_per_model_st = g_st_last.copy()
    grad_lik_acc = g_lik_last.copy()
    grad_coreg = g_c_last.copy()

    # --- Backward loop via lax.fori_loop ---
    def bwd_body(i_rev, carry):
        sd_prev, sa_prev, grad_st_acc, grad_lik, grad_c = carry
        i = nt - 2 - i_rev

        L_i, L_lower_i, L_arrow_i = reconstruct_L(i)
        L_blk_inv_i = jax.scipy.linalg.solve_triangular(L_i, eye_bs, lower=True)

        sl_i = (-sd_prev @ L_lower_i - sa_prev.T @ L_arrow_i) @ L_blk_inv_i
        sa_i = (-sa_prev @ L_lower_i - S_tip @ L_arrow_i) @ L_blk_inv_i
        sd_i = (L_blk_inv_i.T - sl_i.T @ L_lower_i - sa_i.T @ L_arrow_i) @ L_blk_inv_i

        g_st_i, g_lik_i, g_c_i = _accumulate_block_grads(sd_i, sl_i, sa_i, i, i)

        return (sd_i, sa_i, grad_st_acc + g_st_i, grad_lik + g_lik_i, grad_c + g_c_i)

    carry = (sd_last, sa_last, grad_per_model_st, grad_lik_acc, grad_coreg)
    _, _, grad_per_model_st, grad_lik_acc, grad_coreg = lax.fori_loop(
        0, nt - 1, bwd_body, carry)

    # Scale likelihood grads by prec_m
    grad_per_model_lik = likelihood_precs * grad_lik_acc

    return [grad_per_model_st[m] for m in range(n_models)], grad_per_model_lik, grad_coreg


def _compute_grad_quad_coregional(
    x, sc_list, jac_sc_list, coreg_w, jac_coreg_w,
    n_models, nt, ns, n_fe,
    rhs, likelihood_precs,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_ata_tip, per_model_offsets,
):
    """Compute gradient of quad = rhs^T Q_cond^{-1} rhs for coregional models.

    d(quad)/d(theta_st_m) = -x^T (dQ_st/d(theta_st_m)) x
    d(quad)/d(theta_lik_m) = 2 x^T (d(rhs)/d(theta_lik_m)) - prec_m * x^T AtA_m x
    d(quad)/d(coreg_param) = -x^T (dQ_prior/d(coreg_param)) x

    Parameters
    ----------
    x : (nt * block_size + n_fe,)
    sc_list, jac_sc_list, coreg_w, jac_coreg_w : as above
    n_models, nt, ns, n_fe : int
    rhs : (nt * block_size + n_fe,)
    likelihood_precs : (n_models,)
    per_model_ata_* : per-model sparse COO

    Returns
    -------
    grad_per_model_st : list of (3,) arrays
    grad_per_model_lik : (n_models,)
    grad_coreg : (n_coreg_params,)
    """
    block_size = n_models * ns
    dtype = x.dtype
    n_coreg = jac_coreg_w.shape[-1]

    x_st = x[:nt * block_size].reshape(nt, block_size)
    x_fe = x[nt * block_size:]

    # --- ST gradient: -x^T (dQ_st/dtheta_st_m) x ---
    grad_per_model_st = jnp.zeros((n_models, 3), dtype=dtype)
    grad_coreg = jnp.zeros(n_coreg, dtype=dtype)

    # Pad sc_list subdiags so index nt-1 yields zero for lower blocks
    sc_padded = []
    for m in range(n_models):
        sc_m = sc_list[m]
        sc_padded.append({
            **sc_m,
            'm0_subdiag': jnp.concatenate([sc_m['m0_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm1_subdiag': jnp.concatenate([sc_m['m1_subdiag'], jnp.zeros(1, dtype=dtype)]),
            'm2_subdiag': jnp.concatenate([sc_m['m2_subdiag'], jnp.zeros(1, dtype=dtype)]),
        })

    # Pad x_st with a zero row so index nt yields zeros for lower block at t=nt-1
    x_st_padded = jnp.concatenate([x_st, jnp.zeros((1, block_size), dtype=dtype)], axis=0)

    def quad_body(t, carry):
        g_st_acc, g_c_acc = carry
        x_t = x_st[t]
        x_tp1 = x_st_padded[t + 1]
        is_interior = (t < nt - 1).astype(dtype)

        for ii in range(n_models):
            xi = x_t[ii * ns:(ii + 1) * ns]
            xi_next = x_tp1[ii * ns:(ii + 1) * ns]
            for jj in range(n_models):
                xj = x_t[jj * ns:(jj + 1) * ns]
                xj_curr = x_t[jj * ns:(jj + 1) * ns]

                for m_idx in range(n_models):
                    w_ijm = coreg_w[ii, jj, m_idx]
                    sc_m = sc_padded[m_idx]

                    # Diagonal block
                    qu_d = _reconstruct_diag_block(sc_m, t)
                    xQx = xi @ qu_d @ xj

                    dw = jac_coreg_w[ii, jj, m_idx, :]
                    g_c_acc = g_c_acc - dw * xQx

                    # Lower block (zero contribution at t=nt-1 via padding)
                    qu_l = _reconstruct_lower_block(sc_m, t)
                    xQx_l = xi_next @ qu_l @ xj_curr
                    g_c_acc = g_c_acc - 2.0 * is_interior * dw * xQx_l

                    jsc_m = jac_sc_list[m_idx]
                    scale_m = sc_m['scale']
                    exp_gt_m = sc_m['exp_gt']
                    m0_d_val = sc_m['m0_diag'][t]
                    m1_d_val = sc_m['m1_diag'][t]
                    m2_d_val = sc_m['m2_diag'][t]
                    m0_s_val = sc_m['m0_subdiag'][t]
                    m1_s_val = sc_m['m1_subdiag'][t]
                    m2_s_val = sc_m['m2_subdiag'][t]

                    for k in range(3):
                        dq3s = jsc_m['q3s'][..., k]
                        dq2s = jsc_m['q2s'][..., k]
                        dq1s = jsc_m['q1s'][..., k]
                        d_scale = jsc_m['scale'][k]
                        d_exp_gt = jsc_m['exp_gt'][k]

                        dQu = (
                            d_scale * (m0_d_val * sc_m['q3s'] + exp_gt_m * m1_d_val * sc_m['q2s']
                                       + exp_gt_m**2 * m2_d_val * sc_m['q1s'])
                            + scale_m * (m0_d_val * dq3s + exp_gt_m * m1_d_val * dq2s
                                         + exp_gt_m**2 * m2_d_val * dq1s)
                            + scale_m * d_exp_gt * (m1_d_val * sc_m['q2s']
                                                    + 2.0 * exp_gt_m * m2_d_val * sc_m['q1s'])
                        )
                        xdQx = xi @ dQu @ xj
                        g_st_acc = g_st_acc.at[m_idx, k].add(-w_ijm * xdQx)

                        dQu_l = (
                            d_scale * (m0_s_val * sc_m['q3s'] + exp_gt_m * m1_s_val * sc_m['q2s']
                                       + exp_gt_m**2 * m2_s_val * sc_m['q1s'])
                            + scale_m * (m0_s_val * dq3s + exp_gt_m * m1_s_val * dq2s
                                         + exp_gt_m**2 * m2_s_val * dq1s)
                            + scale_m * d_exp_gt * (m1_s_val * sc_m['q2s']
                                                    + 2.0 * exp_gt_m * m2_s_val * sc_m['q1s'])
                        )
                        xdQx_l = xi_next @ dQu_l @ xj_curr
                        g_st_acc = g_st_acc.at[m_idx, k].add(
                            -2.0 * is_interior * w_ijm * xdQx_l)

        return (g_st_acc, g_c_acc)

    grad_per_model_st, grad_coreg = lax.fori_loop(
        0, nt, quad_body, (grad_per_model_st, grad_coreg))

    # --- Likelihood gradient via vectorized sums over time ---
    grad_per_model_lik = jnp.zeros(n_models, dtype=dtype)

    for m in range(n_models):
        m_off = per_model_offsets[m] * ns

        # Vectorized x^T AtA_m x via sparse COO
        xAtAx_d = jnp.sum(
            x_st[jnp.arange(nt)[:, None], m_off + per_model_ata_diag_rows[m]]
            * per_model_ata_diag_vals[m]
            * x_st[jnp.arange(nt)[:, None], m_off + per_model_ata_diag_cols[m]])

        xAtAx_l = jnp.sum(
            x_st[jnp.arange(nt - 1)[:, None] + 1, m_off + per_model_ata_lower_rows[m][:nt - 1]]
            * per_model_ata_lower_vals[m][:nt - 1]
            * x_st[jnp.arange(nt - 1)[:, None], m_off + per_model_ata_lower_cols[m][:nt - 1]])

        xAtAx_a = jnp.sum(
            x_fe[per_model_ata_arrow_rows[m]]
            * per_model_ata_arrow_vals[m]
            * x_st[jnp.arange(nt)[:, None], m_off + per_model_ata_arrow_cols[m]])

        xAtAx_tip = x_fe @ per_model_ata_tip[m] @ x_fe

        xAtAx_m = xAtAx_d + 2.0 * xAtAx_l + 2.0 * xAtAx_a + xAtAx_tip
        grad_per_model_lik = grad_per_model_lik.at[m].set(0.0)

    return [grad_per_model_st[m] for m in range(n_models)], grad_per_model_lik, grad_coreg
