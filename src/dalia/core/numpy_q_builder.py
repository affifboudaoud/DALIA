"""NumPy Q block reconstruction for CuPy streaming forward pass.

Translates JAX Q reconstruction functions from jax_sparse_helpers.py to NumPy,
enabling CPU-side Q block construction that feeds directly into CuPy streaming.
"""

import numpy as np
from scipy.special import gammaln


def _interpretable_to_compute_np(r_s, r_t, sigma_st, manifold="plane"):
    """Transform interpretable parameters to computational parameters.

    NumPy translation of ``_interpretable_to_compute_jax``.

    Parameters
    ----------
    r_s : float
        Spatial range parameter (log-scale).
    r_t : float
        Temporal range parameter (log-scale).
    sigma_st : float
        Spatio-temporal marginal std (log-scale).
    manifold : str
        Either "sphere" or "plane".

    Returns
    -------
    gamma_s, gamma_t, gamma_st : float
    """
    alpha_s = 2
    alpha_t = 1
    alpha_e = 1

    alpha = alpha_e + alpha_s * (alpha_t - 0.5)

    nu_s = alpha - 1
    nu_t = alpha_t - 0.5

    gamma_s = 0.5 * np.log(8 * nu_s) - r_s
    gamma_t = r_t - 0.5 * np.log(8 * nu_t) + alpha_s * gamma_s

    if manifold == "sphere":
        log_cR_t = gammaln(nu_t) - gammaln(alpha_t) - 0.5 * np.log(4.0 * np.pi)
        k_vals = np.arange(50)
        exp_gamma_s_sq = np.exp(gamma_s) ** 2
        c_s = np.sum(
            (2.0 * k_vals + 1.0) / (4.0 * np.pi * np.power(exp_gamma_s_sq + k_vals * (k_vals + 1), alpha))
        )
        gamma_st = 0.5 * log_cR_t + 0.5 * np.log(c_s) - 0.5 * gamma_t - sigma_st
    elif manifold == "plane":
        log_c1 = (
            gammaln(nu_t) + gammaln(nu_s)
            - gammaln(alpha_t) - gammaln(alpha)
            - 1.5 * np.log(4.0 * np.pi)
        )
        gamma_st = 0.5 * log_c1 - 0.5 * gamma_t - nu_s * gamma_s - sigma_st
    else:
        raise ValueError(f"Manifold not supported: {manifold}")

    return gamma_s, gamma_t, gamma_st


def precompute_spatial_components_np(theta_st, spatial_matrices, temporal_matrices, manifold):
    """Precompute spatial/temporal components for lazy block reconstruction.

    NumPy translation of ``precompute_spatial_components``.

    Parameters
    ----------
    theta_st : array-like
        Spatio-temporal hyperparameters [r_s, r_t, sigma_st].
    spatial_matrices : dict
        Dict with keys 'c0', 'g1', 'g2', 'g3' as numpy arrays.
    temporal_matrices : dict
        Dict with keys 'm0', 'm1', 'm2' as numpy arrays.
    manifold : str

    Returns
    -------
    dict
    """
    r_s = float(theta_st[0])
    r_t = float(theta_st[1])
    sigma_st = float(theta_st[2])

    gamma_s, gamma_t, gamma_st = _interpretable_to_compute_np(r_s, r_t, sigma_st, manifold)

    c0 = spatial_matrices['c0']
    g1 = spatial_matrices['g1']
    g2 = spatial_matrices['g2']
    g3 = spatial_matrices['g3']

    m0 = temporal_matrices['m0']
    m1 = temporal_matrices['m1']
    m2 = temporal_matrices['m2']

    exp_gamma_s = np.exp(gamma_s)
    exp_gamma_t = np.exp(gamma_t)
    exp_gamma_st = np.exp(gamma_st)

    q1s = exp_gamma_s**2 * c0 + g1
    q2s = exp_gamma_s**4 * c0 + 2 * exp_gamma_s**2 * g1 + g2
    q3s = exp_gamma_s**6 * c0 + 3 * exp_gamma_s**4 * g1 + 3 * exp_gamma_s**2 * g2 + g3

    scale = exp_gamma_st**2

    m0_diag = np.diag(m0)
    m1_diag = np.diag(m1)
    m2_diag = np.diag(m2)

    m0_subdiag = np.diag(m0, k=-1)
    m1_subdiag = np.diag(m1, k=-1)
    m2_subdiag = np.diag(m2, k=-1)

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


def _reconstruct_diag_block_np(sc, i):
    """Reconstruct diagonal block i of Q_st from spatial components."""
    return sc['scale'] * (
        sc['m0_diag'][i] * sc['q3s']
        + sc['exp_gt'] * sc['m1_diag'][i] * sc['q2s']
        + sc['exp_gt']**2 * sc['m2_diag'][i] * sc['q1s']
    )


def _reconstruct_lower_block_np(sc, i):
    """Reconstruct lower-diagonal block i of Q_st from spatial components."""
    return sc['scale'] * (
        sc['m0_subdiag'][i] * sc['q3s']
        + sc['exp_gt'] * sc['m1_subdiag'][i] * sc['q2s']
        + sc['exp_gt']**2 * sc['m2_subdiag'][i] * sc['q1s']
    )


def precompute_spatial_components_coregional_np(
    theta, n_models, ns, models_data_np, hyperparameters_idx, theta_keys, manifolds,
):
    """Precompute spatial components for all models + coregional coefficients.

    NumPy translation of ``precompute_spatial_components_coregional``.

    Parameters
    ----------
    theta : np.ndarray
        Full hyperparameter vector.
    n_models : int
    ns : int
    models_data_np : list of dict
        Per-model data with numpy spatial/temporal matrices.
    hyperparameters_idx : list of int
    theta_keys : list of str
    manifolds : list of str

    Returns
    -------
    sc_list : list of dict
    coreg_w : np.ndarray, shape (n_models, n_models, n_models)
    """
    sc_list = []
    for m in range(n_models):
        hp_start = hyperparameters_idx[m]
        hp_end = hyperparameters_idx[m + 1] - 1
        theta_model = theta[hp_start:hp_end]
        if theta_model.shape[0] == 2:
            theta_model = np.concatenate([theta_model, np.array([0.0])])

        sc_m = precompute_spatial_components_np(
            theta_model,
            models_data_np[m]['spatial_matrices'],
            models_data_np[m]['temporal_matrices'],
            manifolds[m],
        )
        sc_list.append(sc_m)

    sigma_idx = theta_keys.index('sigma_0')
    sigmas = np.array([np.exp(theta[sigma_idx + i]) for i in range(n_models)])

    coreg_w = np.zeros((n_models, n_models, n_models), dtype=theta.dtype)

    if n_models == 2:
        lambda_01 = theta[theta_keys.index('lambda_0_1')]
        s0, s1 = sigmas[0], sigmas[1]
        coreg_w[0, 0, 0] = 1.0 / s0**2
        coreg_w[0, 0, 1] = lambda_01**2 / s1**2
        coreg_w[1, 0, 1] = -lambda_01 / s1**2
        coreg_w[0, 1, 1] = -lambda_01 / s1**2
        coreg_w[1, 1, 1] = 1.0 / s1**2

    elif n_models == 3:
        lambda_01 = theta[theta_keys.index('lambda_0_1')]
        lambda_02 = theta[theta_keys.index('lambda_0_2')]
        lambda_12 = theta[theta_keys.index('lambda_1_2')]
        s0, s1, s2 = sigmas[0], sigmas[1], sigmas[2]

        coreg_w[0, 0, 0] = 1.0 / s0**2
        coreg_w[0, 0, 1] = lambda_01**2 / s1**2
        coreg_w[0, 0, 2] = lambda_12**2 / s2**2

        coreg_w[1, 0, 1] = -lambda_01 / s1**2
        coreg_w[0, 1, 1] = -lambda_01 / s1**2
        coreg_w[1, 0, 2] = lambda_02 * lambda_12 / s2**2
        coreg_w[0, 1, 2] = lambda_02 * lambda_12 / s2**2

        coreg_w[2, 0, 2] = -lambda_12 / s2**2
        coreg_w[0, 2, 2] = -lambda_12 / s2**2

        coreg_w[1, 1, 1] = 1.0 / s1**2
        coreg_w[1, 1, 2] = lambda_02**2 / s2**2

        coreg_w[2, 1, 2] = -lambda_02 / s2**2
        coreg_w[1, 2, 2] = -lambda_02 / s2**2

        coreg_w[2, 2, 2] = 1.0 / s2**2

    return sc_list, coreg_w


def reconstruct_coregional_diag_block_np(sc_list, coreg_w, n_models, ns, t):
    """Reconstruct diagonal super-block t of coregional Q_prior.

    NumPy translation of ``_reconstruct_coregional_diag_block``.

    Parameters
    ----------
    sc_list : list of dict
    coreg_w : np.ndarray, shape (n_models, n_models, n_models)
    n_models, ns : int
    t : int

    Returns
    -------
    block : np.ndarray, shape (n_models*ns, n_models*ns)
    """
    dtype = sc_list[0]['q3s'].dtype

    subs = [[None] * n_models for _ in range(n_models)]
    for i in range(n_models):
        for j in range(i + 1):
            sub = np.zeros((ns, ns), dtype=dtype)
            for m_idx in range(n_models):
                sub = sub + coreg_w[i, j, m_idx] * _reconstruct_diag_block_np(sc_list[m_idx], t)
            subs[i][j] = sub

    for i in range(n_models):
        for j in range(i + 1, n_models):
            subs[i][j] = subs[j][i]

    block_rows = []
    for i in range(n_models):
        block_rows.append(np.concatenate(subs[i], axis=1))
    return np.concatenate(block_rows, axis=0)


def reconstruct_coregional_lower_block_np(sc_list, coreg_w, n_models, ns, t):
    """Reconstruct lower-diagonal super-block t of coregional Q_prior.

    NumPy translation of ``_reconstruct_coregional_lower_block``.

    Parameters
    ----------
    sc_list : list of dict
    coreg_w : np.ndarray, shape (n_models, n_models, n_models)
    n_models, ns : int
    t : int

    Returns
    -------
    block : np.ndarray, shape (n_models*ns, n_models*ns)
    """
    dtype = sc_list[0]['q3s'].dtype

    block_rows = []
    for i in range(n_models):
        cols = []
        for j in range(n_models):
            sub = np.zeros((ns, ns), dtype=dtype)
            for m_idx in range(n_models):
                sub = sub + coreg_w[i, j, m_idx] * _reconstruct_lower_block_np(sc_list[m_idx], t)
            cols.append(sub)
        block_rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(block_rows, axis=0)


def build_q_blocks_np(
    theta_np, n_models, ns, block_size, n_fe, nt,
    models_data_np, hyperparameters_idx, theta_keys, manifolds,
    per_model_ata_diag_rows, per_model_ata_diag_cols, per_model_ata_diag_vals,
    per_model_ata_lower_rows, per_model_ata_lower_cols, per_model_ata_lower_vals,
    per_model_ata_arrow_rows, per_model_ata_arrow_cols, per_model_ata_arrow_vals,
    per_model_ata_tip, per_model_offsets,
    likelihood_precs_np, fe_prec, eps_reg,
    start_idx, n_local, dtype,
):
    """Build all local Q blocks in numpy.

    Parameters
    ----------
    theta_np : np.ndarray
        Full hyperparameter vector.
    n_models, ns, block_size, n_fe, nt : int
    models_data_np : list of dict
        Per-model numpy spatial/temporal matrices.
    hyperparameters_idx, theta_keys, manifolds : list
    per_model_ata_* : list of np.ndarray
        Sparse AtA block data (already sliced to local partition).
    per_model_ata_tip : list of np.ndarray
    per_model_offsets : list of int
    likelihood_precs_np : np.ndarray, shape (n_models,)
    fe_prec : float
    eps_reg : float
    start_idx, n_local : int
    dtype : np.dtype

    Returns
    -------
    q_diag : np.ndarray, shape (n_local, block_size, block_size)
    q_lower : np.ndarray, shape (n_local, block_size, block_size)
    q_arrow : np.ndarray, shape (n_local, n_fe, block_size)
    arrow_tip : np.ndarray, shape (n_fe, n_fe)
    """
    sc_list, coreg_w = precompute_spatial_components_coregional_np(
        theta_np, n_models, ns, models_data_np,
        hyperparameters_idx, theta_keys, manifolds,
    )
    # Pad subdiags for safe indexing at last block
    for m in range(n_models):
        sc_list[m]['m0_subdiag'] = np.concatenate([sc_list[m]['m0_subdiag'], np.zeros(1)])
        sc_list[m]['m1_subdiag'] = np.concatenate([sc_list[m]['m1_subdiag'], np.zeros(1)])
        sc_list[m]['m2_subdiag'] = np.concatenate([sc_list[m]['m2_subdiag'], np.zeros(1)])

    q_diag = np.empty((n_local, block_size, block_size), dtype=dtype)
    q_lower = np.empty((n_local, block_size, block_size), dtype=dtype)
    q_arrow = np.empty((n_local, n_fe, block_size), dtype=dtype)

    for j in range(n_local):
        global_i = start_idx + j

        # Diagonal block
        q_d = reconstruct_coregional_diag_block_np(sc_list, coreg_w, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            rows = per_model_ata_diag_rows[m][j]
            cols = per_model_ata_diag_cols[m][j]
            vals = per_model_ata_diag_vals[m][j]
            q_d[m_off + rows, m_off + cols] += likelihood_precs_np[m] * vals
        q_diag[j] = q_d

        # Lower block
        q_l = reconstruct_coregional_lower_block_np(sc_list, coreg_w, n_models, ns, global_i)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            rows = per_model_ata_lower_rows[m][j]
            cols = per_model_ata_lower_cols[m][j]
            vals = per_model_ata_lower_vals[m][j]
            q_l[m_off + rows, m_off + cols] += likelihood_precs_np[m] * vals
        q_lower[j] = q_l

        # Arrow block
        q_a = np.zeros((n_fe, block_size), dtype=dtype)
        for m in range(n_models):
            m_off = per_model_offsets[m] * ns
            rows = per_model_ata_arrow_rows[m][j]
            cols = per_model_ata_arrow_cols[m][j]
            vals = per_model_ata_arrow_vals[m][j]
            q_a[rows, m_off + cols] += likelihood_precs_np[m] * vals
        q_arrow[j] = q_a

    # Arrow tip
    arrow_tip = fe_prec * np.eye(n_fe, dtype=dtype) + eps_reg * np.eye(n_fe, dtype=dtype)
    for m in range(n_models):
        arrow_tip = arrow_tip + likelihood_precs_np[m] * per_model_ata_tip[m]

    return q_diag, q_lower, q_arrow, arrow_tip, coreg_w
