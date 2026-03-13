# Copyright 2024-2025 DALIA authors. All rights reserved.

import jax.numpy as jnp
from jax import lax
from functools import partial

from dalia.core.autodiff.spatial_precompute import (
    _jax_cholesky,
    precompute_spatial_components,
    _reconstruct_diag_block,
    _reconstruct_lower_block,
)

import jax
import jax.scipy.linalg


def lazy_bta_cholesky(
    spatial_comp, nt, ns, n_fe, fe_prec, likelihood_prec,
    ata_diag_rows, ata_diag_cols, ata_diag_vals,
    ata_lower_rows, ata_lower_cols, ata_lower_vals,
    ata_arrow_rows, ata_arrow_cols, ata_arrow_vals,
    ata_tip, dtype, checkpoint=False,
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
    checkpoint : bool, optional
        If True, wrap scan body with ``jax.checkpoint`` to trade compute
        for memory — JAX will recompute the forward pass during backward
        instead of storing the full carry trajectory. Default False.

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

    scan_fn = scan_body
    if checkpoint:
        from functools import partial
        scan_fn = jax.checkpoint(scan_body, prevent_cse=True)

    (_, arrow_tip_final, _, logdet_cond), \
        (L_diag_all, L_lower_all, L_arrow_all) = lax.scan(
            scan_fn, init_carry, scan_inputs
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
