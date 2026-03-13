# Copyright 2024-2025 DALIA authors. All rights reserved.

import jax.numpy as jnp
from typing import Tuple

from dalia.core.autodiff.spatial_precompute import _interpretable_to_compute_jax


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
    from dalia.core.autodiff.config import get_jax_dtype
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
    from dalia.core.autodiff.config import get_jax_dtype
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
