# Copyright 2024-2025 DALIA authors. All rights reserved.

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from functools import partial
from typing import Tuple


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
    from dalia.core.autodiff.config import get_jax_dtype
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

    Uses concatenation instead of scatter updates to avoid intermediate
    copies of the full (block_size, block_size) matrix.

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
    dtype = sc_list[0]['q3s'].dtype

    # Compute lower triangle sub-blocks (including diagonal)
    subs = [[None] * n_models for _ in range(n_models)]
    for i in range(n_models):
        for j in range(i + 1):
            sub = jnp.zeros((ns, ns), dtype=dtype)
            for m_idx in range(n_models):
                sub = sub + coreg_w[i, j, m_idx] * _reconstruct_diag_block(sc_list[m_idx], t)
            subs[i][j] = sub

    # Copy lower triangle to upper triangle (symmetric)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            subs[i][j] = subs[j][i]

    block_rows = []
    for i in range(n_models):
        block_rows.append(jnp.concatenate(subs[i], axis=1))
    return jnp.concatenate(block_rows, axis=0)


def _reconstruct_coregional_lower_block(sc_list, coreg_w, n_models, ns, t):
    """Reconstruct lower-diagonal super-block t of coregional Q_prior.

    Uses concatenation instead of scatter updates to avoid intermediate
    copies of the full (block_size, block_size) matrix.

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
    dtype = sc_list[0]['q3s'].dtype

    block_rows = []
    for i in range(n_models):
        cols = []
        for j in range(n_models):
            sub = jnp.zeros((ns, ns), dtype=dtype)
            for m_idx in range(n_models):
                sub = sub + coreg_w[i, j, m_idx] * _reconstruct_lower_block(sc_list[m_idx], t)
            cols.append(sub)
        block_rows.append(jnp.concatenate(cols, axis=1))
    return jnp.concatenate(block_rows, axis=0)
