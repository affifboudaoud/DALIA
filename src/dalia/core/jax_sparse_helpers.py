
# Copyright 2024-2025 DALIA authors. All rights reserved.
import jax.numpy as jnp
from typing import Tuple


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

    diagonal_blocks = jnp.zeros((nt, ns, ns))
    lower_diagonal_blocks = jnp.zeros((nt - 1, ns, ns))

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
        lower_arrow_blocks = jnp.zeros((nt, n_fixed_effects, ns))
        for i in range(nt):
            start_i = i * ns
            end_i = (i + 1) * ns
            lower_arrow_blocks = lower_arrow_blocks.at[i].set(
                kron_result[total_st_size:, start_i:end_i]
            )

        arrow_tip_block = kron_result[total_st_size:, total_st_size:]
    else:
        lower_arrow_blocks = jnp.zeros((nt, 1, ns))
        arrow_tip_block = jnp.zeros((1, 1))

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
    import math
    from scipy import special as sp_special

    alpha_s = 2
    alpha_t = 1
    alpha_e = 1

    alpha = alpha_e + alpha_s * (alpha_t - 0.5)

    nu_s = alpha - 1
    nu_t = alpha_t - 0.5

    gamma_s = 0.5 * jnp.log(8 * nu_s) - r_s
    gamma_t = r_t - 0.5 * jnp.log(8 * nu_t) + alpha_s * gamma_s

    if manifold == "sphere":
        cR_t = sp_special.gamma(nu_t) / (sp_special.gamma(alpha_t) * pow(4 * math.pi, 0.5))
        # Vectorized sum: k = 0, 1, ..., 49
        k_vals = jnp.arange(50)
        exp_gamma_s_sq = jnp.exp(gamma_s) ** 2
        c_s = jnp.sum(
            (2.0 * k_vals + 1.0) / (4.0 * math.pi * jnp.power(exp_gamma_s_sq + k_vals * (k_vals + 1), alpha))
        )
        gamma_st = 0.5 * jnp.log(cR_t) + 0.5 * jnp.log(c_s) - 0.5 * gamma_t - sigma_st
    elif manifold == "plane":
        c1_scaling_constant = pow(4 * math.pi, 1.5)
        c1 = (
            sp_special.gamma(nu_t) * sp_special.gamma(nu_s)
            / (sp_special.gamma(alpha_t) * sp_special.gamma(alpha) * c1_scaling_constant)
        )
        gamma_st = 0.5 * jnp.log(c1) - 0.5 * gamma_t - nu_s * gamma_s - sigma_st
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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract BTA blocks from a scipy sparse matrix.

    inputs:
    sparse_matrix : scipy sparse matrix (CSR/CSC/COO)
    nt : Number of temporal blocks
    ns : Size of each spatial block
    n_fixed_effects : Size of arrow tip

    Returns:
    diag_blocks : (nt, ns, ns)
    lower_diag_blocks : (nt-1, ns, ns)
    arrow_bottom_blocks : (nt, n_fixed_effects, ns)
    arrow_tip : (n_fixed_effects, n_fixed_effects)
    """
    from scipy import sparse as sp_sparse
    csc = sp_sparse.csc_matrix(sparse_matrix)

    total_st = nt * ns

    diag_blocks = jnp.zeros((nt, ns, ns))
    lower_diag_blocks = jnp.zeros((nt - 1, ns, ns))
    arrow_bottom_blocks = jnp.zeros((nt, n_fixed_effects, ns))
    arrow_tip = jnp.zeros((n_fixed_effects, n_fixed_effects))

    # Extract diagonal blocks
    for i in range(nt):
        start = i * ns
        end = (i + 1) * ns
        block = csc[start:end, start:end].toarray()
        diag_blocks = diag_blocks.at[i].set(jnp.array(block))

    # Extract lower diagonal blocks
    for i in range(nt - 1):
        row_start = (i + 1) * ns
        row_end = (i + 2) * ns
        col_start = i * ns
        col_end = (i + 1) * ns
        block = csc[row_start:row_end, col_start:col_end].toarray()
        lower_diag_blocks = lower_diag_blocks.at[i].set(jnp.array(block))

    # Extract arrow bottom blocks
    if n_fixed_effects > 0:
        for i in range(nt):
            col_start = i * ns
            col_end = (i + 1) * ns
            block = csc[total_st:, col_start:col_end].toarray()
            arrow_bottom_blocks = arrow_bottom_blocks.at[i].set(jnp.array(block))

        # Extract arrow tip
        arrow_tip = jnp.array(csc[total_st:, total_st:].toarray())

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

    Q_conditional = jnp.zeros((total_size, total_size))

    Q_conditional = Q_conditional.at[:total_st_size, :total_st_size].set(Q_st)

    Q_fe = jnp.eye(n_fixed_effects) * fixed_effects_precision
    Q_conditional = Q_conditional.at[total_st_size:, total_st_size:].set(Q_fe)

    return Q_conditional


def compute_logdet_from_cholesky_bta_jax(
    diagonal_blocks: jnp.ndarray,
    arrow_tip_block: jnp.ndarray,
) -> float:
    """Compute log-determinant from Cholesky factorization in BTA format.

    log(det(Q)) = 2 * sum(log(diag(L)))

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
    logdet = 2.0 * jnp.sum(jnp.log(diag_vals))

    # Add arrow tip contribution if present
    arrow_diag = jnp.diag(arrow_tip_block)
    logdet = logdet + 2.0 * jnp.sum(jnp.log(arrow_diag))

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
