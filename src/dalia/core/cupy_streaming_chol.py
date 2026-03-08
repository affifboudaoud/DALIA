"""CuPy streaming Cholesky factorization + forward substitution.

Replicates serinv's triple-buffered streaming pattern from ``_pobtaf_streaming``
and ``_pobtaf_permuted_streaming`` (pobtaf.py), with fused forward substitution.

All triangular solves use pre-allocated F-order workspace arrays and
``overwrite_b=True`` to eliminate CuPy-internal ``cudaMalloc`` calls during the
streaming loop, preventing CUDA virtual-address-space fragmentation when
coexisting with JAX's BFC allocator.
"""

import numpy as np


def streaming_chol_fwd_sub(
    Q_diag,
    Q_lower,
    Q_arrow,
    arrow_tip,
    rhs_local,
    rhs_arrow,
    eps_reg,
    factorize_last,
):
    """Streaming Cholesky + forward substitution for standard (root rank) BTA.

    Uses triple-buffered async streaming on 3 CUDA streams (compute, H2D, D2H)
    following serinv's ``_pobtaf_streaming`` pattern, with fused forward sub.

    Parameters
    ----------
    Q_diag : np.ndarray, shape (n_blocks, bs, bs)
        Q_cond diagonal blocks (numpy, host memory).
    Q_lower : np.ndarray, shape (n_blocks, bs, bs)
        Q_cond lower-diagonal blocks. Last block may be zero.
    Q_arrow : np.ndarray, shape (n_blocks, n_fe, bs)
        Arrow blocks.
    arrow_tip : np.ndarray, shape (n_fe, n_fe)
        Arrow tip accumulator (modified in-place).
    rhs_local : np.ndarray, shape (n_blocks, bs)
        RHS vectors for forward substitution.
    rhs_arrow : np.ndarray, shape (n_fe,)
        Arrow RHS initial value.
    eps_reg : float
        Regularization epsilon.
    factorize_last : bool
        Whether to factorize the last diagonal block.

    Returns
    -------
    L_diag : np.ndarray, shape (n_blocks, bs, bs)
    L_lower : np.ndarray, shape (n_blocks, bs, bs)
    L_arrow : np.ndarray, shape (n_blocks, n_fe, bs)
    arrow_tip : np.ndarray, shape (n_fe, n_fe)
    y_local : np.ndarray, shape (n_blocks, bs)
    arrow_rhs : np.ndarray, shape (n_fe,)
    logdet : float
    cond_schur : np.ndarray, shape (bs, bs)
    arrow_schur : np.ndarray, shape (n_fe, bs)
    """
    import cupy as cp
    import cupyx.scipy.linalg as cu_la
    from serinv import _get_cholesky

    cholesky = _get_cholesky("cupy")
    n_blocks = Q_diag.shape[0]
    bs = Q_diag.shape[1]
    n_fe = Q_arrow.shape[1]
    dtype = Q_diag.dtype

    L_diag = np.empty_like(Q_diag)
    L_lower = np.empty_like(Q_lower)
    L_arrow = np.empty_like(Q_arrow)
    y_local = np.empty_like(rhs_local)
    cond_schur_out = np.zeros((bs, bs), dtype=dtype)
    arrow_schur_out = np.zeros((n_fe, bs), dtype=dtype)

    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diag_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_diag_events = [cp.cuda.Event(), cp.cuda.Event()]

    cp_diag_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_lower_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_arrow_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]

    diag_d = cp.empty((2, bs, bs), dtype=dtype)
    lower_d = cp.empty((2, bs, bs), dtype=dtype)
    arrow_d = cp.empty((2, n_fe, bs), dtype=dtype)
    tip_d = cp.empty((n_fe, n_fe), dtype=dtype)

    logdet_d = cp.zeros(1, dtype=dtype)
    eye_d = eps_reg * cp.eye(bs, dtype=dtype)
    prev_lower_y_d = cp.zeros(bs, dtype=dtype)

    L_diag_d = diag_d
    L_lower_d = lower_d
    L_arrow_d = arrow_d

    # F-order workspaces for solve_triangular (no cudaMalloc during loop)
    ws_L = cp.empty((bs, bs), dtype=dtype, order='F')
    ws_mat = cp.empty((bs, bs), dtype=dtype, order='F')
    ws_arrow = cp.empty((bs, n_fe), dtype=dtype, order='F')

    # --- Initial transfers ---
    cp_lower_h2d_events[1].record(stream=compute_stream)
    cp_arrow_h2d_events[1].record(stream=compute_stream)
    tip_d.set(arr=arrow_tip, stream=compute_stream)

    diag_d[0].set(arr=Q_diag[0], stream=h2d_stream)
    h2d_diag_events[0].record(stream=h2d_stream)
    arrow_d[0].set(arr=Q_arrow[0], stream=h2d_stream)
    h2d_arrow_events[0].record(stream=h2d_stream)

    if n_blocks > 1:
        lower_d[0].set(arr=Q_lower[0], stream=h2d_stream)
        h2d_lower_events[0].record(stream=h2d_stream)

    d2h_diag_events[1].record(stream=d2h_stream)

    # --- Main loop: blocks 0 .. n_blocks-2 ---
    for i in range(n_blocks - 1):
        b = i % 2
        bn = (i + 1) % 2

        # L_{i,i} = chol(A_{i,i})
        with compute_stream:
            compute_stream.wait_event(h2d_diag_events[b])
            diag_d[b] += eye_d
            L_diag_d[b] = cholesky(diag_d[b])
            ws_L[:] = L_diag_d[b]
            cp_diag_events[b].record(stream=compute_stream)

        d2h_stream.wait_event(cp_diag_events[b])
        L_diag_d[b].get(out=L_diag[i], stream=d2h_stream, blocking=False)
        d2h_diag_events[b].record(stream=d2h_stream)

        # H2D: pre-fetch next lower
        if i + 1 < n_blocks - 1:
            h2d_stream.wait_event(cp_lower_h2d_events[bn])
            lower_d[bn].set(arr=Q_lower[i + 1], stream=h2d_stream)
            h2d_lower_events[bn].record(stream=h2d_stream)

        # L_{i+1,i} = A_{i+1,i} @ L_{i,i}^{-T}
        with compute_stream:
            compute_stream.wait_event(h2d_lower_events[b])
            ws_mat[:] = lower_d[b].T
            cu_la.solve_triangular(ws_L, ws_mat, lower=True, overwrite_b=True)
            L_lower_d[b][:] = ws_mat.T
            cp_lower_events[b].record(stream=compute_stream)

        d2h_stream.wait_event(cp_lower_events[b])
        L_lower_d[b].get(out=L_lower[i], stream=d2h_stream, blocking=False)

        # H2D: pre-fetch next arrow
        h2d_stream.wait_event(cp_arrow_h2d_events[bn])
        arrow_d[bn].set(arr=Q_arrow[i + 1], stream=h2d_stream)
        h2d_arrow_events[bn].record(stream=h2d_stream)

        # L_{ndb+1,i} = A_{ndb+1,i} @ L_{i,i}^{-T}
        with compute_stream:
            compute_stream.wait_event(h2d_arrow_events[b])
            ws_arrow[:] = arrow_d[b].T
            cu_la.solve_triangular(ws_L, ws_arrow, lower=True, overwrite_b=True)
            L_arrow_d[b][:] = ws_arrow.T
            cp_arrow_events[b].record(stream=compute_stream)

        d2h_stream.wait_event(cp_arrow_events[b])
        L_arrow_d[b].get(out=L_arrow[i], stream=d2h_stream, blocking=False)

        # Forward substitution: y_i = L^{-1}(rhs_i - L_lower_prev @ y_prev)
        with compute_stream:
            rhs_i_d = cp.asarray(rhs_local[i])
            rhs_i_d -= prev_lower_y_d
            y_i_d = cu_la.solve_triangular(
                ws_L, rhs_i_d, lower=True, overwrite_b=True)

        y_i_d.get(out=y_local[i], stream=compute_stream, blocking=False)

        # Logdet
        with compute_stream:
            diag_vals = cp.diag(L_diag_d[b])
            logdet_d += 2.0 * cp.sum(cp.log(cp.maximum(diag_vals, cp.finfo(dtype).eps)))

        # H2D: pre-fetch next diagonal
        h2d_stream.wait_event(d2h_diag_events[bn])
        diag_d[bn].set(arr=Q_diag[i + 1], stream=h2d_stream)
        h2d_diag_events[bn].record(stream=h2d_stream)

        # Schur updates
        with compute_stream:
            compute_stream.wait_event(h2d_diag_events[bn])
            diag_d[bn] -= L_lower_d[b] @ L_lower_d[b].T
            arrow_d[bn] -= L_arrow_d[b] @ L_lower_d[b].T
            cp_lower_h2d_events[b].record(stream=compute_stream)
            tip_d -= L_arrow_d[b] @ L_arrow_d[b].T
            cp_arrow_h2d_events[b].record(stream=compute_stream)
            prev_lower_y_d = L_lower_d[b] @ y_i_d

    # --- Last block ---
    b_last = (n_blocks - 1) % 2

    # Load last lower for Schur carry computation
    h2d_stream.wait_event(cp_lower_h2d_events[b_last])
    lower_d[b_last].set(arr=Q_lower[n_blocks - 1], stream=h2d_stream)
    h2d_lower_events[b_last].record(stream=h2d_stream)

    with compute_stream:
        compute_stream.wait_event(h2d_diag_events[b_last])
        if factorize_last:
            diag_d[b_last] += eye_d
            L_diag_d[b_last] = cholesky(diag_d[b_last])
            ws_L[:] = L_diag_d[b_last]
        cp_diag_events[b_last].record(stream=compute_stream)

    d2h_stream.wait_event(cp_diag_events[b_last])
    L_diag_d[b_last].get(out=L_diag[n_blocks - 1], stream=d2h_stream, blocking=False)

    # Lower solve for last block
    with compute_stream:
        compute_stream.wait_event(h2d_lower_events[b_last])
        if factorize_last:
            ws_mat[:] = lower_d[b_last].T
            cu_la.solve_triangular(ws_L, ws_mat, lower=True, overwrite_b=True)
            L_lower_d[b_last][:] = ws_mat.T
        cp_lower_events[b_last].record(stream=compute_stream)

    d2h_stream.wait_event(cp_lower_events[b_last])
    L_lower_d[b_last].get(out=L_lower[n_blocks - 1], stream=d2h_stream, blocking=False)

    # Arrow solve for last block
    with compute_stream:
        compute_stream.wait_event(h2d_arrow_events[b_last])
        if factorize_last:
            ws_arrow[:] = arrow_d[b_last].T
            cu_la.solve_triangular(ws_L, ws_arrow, lower=True, overwrite_b=True)
            L_arrow_d[b_last][:] = ws_arrow.T
        cp_arrow_events[b_last].record(stream=compute_stream)

    d2h_stream.wait_event(cp_arrow_events[b_last])
    L_arrow_d[b_last].get(out=L_arrow[n_blocks - 1], stream=d2h_stream, blocking=False)

    # Forward sub for last block
    with compute_stream:
        if factorize_last:
            rhs_last_d = cp.asarray(rhs_local[n_blocks - 1])
            rhs_last_d -= prev_lower_y_d
            y_last_d = cu_la.solve_triangular(
                ws_L, rhs_last_d, lower=True, overwrite_b=True)
            y_last_d.get(out=y_local[n_blocks - 1], stream=compute_stream, blocking=False)

            diag_vals = cp.diag(L_diag_d[b_last])
            logdet_d += 2.0 * cp.sum(cp.log(cp.maximum(diag_vals, cp.finfo(dtype).eps)))
            tip_d -= L_arrow_d[b_last] @ L_arrow_d[b_last].T

            # Schur carry from last factorized block
            cond_schur_d = L_lower_d[b_last] @ L_lower_d[b_last].T
            arrow_schur_d = L_arrow_d[b_last] @ L_lower_d[b_last].T
            cond_schur_d.get(out=cond_schur_out, stream=compute_stream, blocking=False)
            arrow_schur_d.get(out=arrow_schur_out, stream=compute_stream, blocking=False)
        else:
            # Schur carry from last loop iteration (block n_blocks-2)
            b_prev = (n_blocks - 2) % 2
            cond_schur_d = L_lower_d[b_prev] @ L_lower_d[b_prev].T
            arrow_schur_d = L_arrow_d[b_prev] @ L_lower_d[b_prev].T
            cond_schur_d.get(out=cond_schur_out, stream=compute_stream, blocking=False)
            arrow_schur_d.get(out=arrow_schur_out, stream=compute_stream, blocking=False)

    tip_d.get(out=arrow_tip, stream=compute_stream, blocking=False)

    cp.cuda.Device().synchronize()

    logdet = float(logdet_d.get())

    del diag_d, lower_d, arrow_d, tip_d, logdet_d, eye_d, prev_lower_y_d
    del L_diag_d, L_lower_d, L_arrow_d
    del ws_L, ws_mat, ws_arrow
    del compute_stream, h2d_stream, d2h_stream
    cp.get_default_memory_pool().free_all_blocks()

    # Arrow RHS: accumulated on CPU (n_fe is tiny, O(10))
    arrow_rhs = rhs_arrow.copy()
    n_factorized = n_blocks if factorize_last else n_blocks - 1
    for i in range(n_factorized):
        arrow_rhs -= L_arrow[i] @ y_local[i]

    return (L_diag, L_lower, L_arrow, arrow_tip,
            y_local, arrow_rhs, logdet, cond_schur_out, arrow_schur_out)


def streaming_chol_fwd_sub_permuted(
    Q_diag,
    Q_lower,
    Q_arrow,
    arrow_tip,
    buffer_init,
    rhs_local,
    rhs_arrow,
    eps_reg,
):
    """Streaming Cholesky + forward substitution for permuted (non-root) BTA.

    Uses triple-buffered async streaming following serinv's
    ``_pobtaf_permuted_streaming`` pattern, with fused forward sub.

    Parameters
    ----------
    Q_diag : np.ndarray, shape (n_blocks, bs, bs)
    Q_lower : np.ndarray, shape (n_blocks, bs, bs)
    Q_arrow : np.ndarray, shape (n_blocks, n_fe, bs)
    arrow_tip : np.ndarray, shape (n_fe, n_fe)
        Modified in-place.
    buffer_init : np.ndarray, shape (bs, bs)
        Initial buffer = Q_lower[boundary].T (from prev rank boundary).
    rhs_local : np.ndarray, shape (n_blocks, bs)
    rhs_arrow : np.ndarray, shape (n_fe,)
    eps_reg : float

    Returns
    -------
    L_diag : np.ndarray, shape (n_blocks, bs, bs)
    L_lower : np.ndarray, shape (n_blocks, bs, bs)
    L_arrow : np.ndarray, shape (n_blocks, n_fe, bs)
    buf_solved : np.ndarray, shape (n_blocks, bs, bs)
    arrow_tip : np.ndarray, shape (n_fe, n_fe)
    block0_diag : np.ndarray, shape (bs, bs)
    block0_arrow : np.ndarray, shape (n_fe, bs)
    block0_rhs : np.ndarray, shape (bs,)
    y_local : np.ndarray, shape (n_blocks, bs)
    arrow_rhs : np.ndarray, shape (n_fe,)
    logdet : float
    cond_schur : np.ndarray, shape (bs, bs)
    arrow_schur : np.ndarray, shape (n_fe, bs)
    """
    import cupy as cp
    import cupyx.scipy.linalg as cu_la
    from serinv import _get_cholesky

    cholesky = _get_cholesky("cupy")
    n_blocks = Q_diag.shape[0]
    bs = Q_diag.shape[1]
    n_fe = Q_arrow.shape[1]
    dtype = Q_diag.dtype

    L_diag = np.empty_like(Q_diag)
    L_lower = np.empty_like(Q_lower)
    L_arrow = np.empty_like(Q_arrow)
    buf_solved = np.empty((n_blocks, bs, bs), dtype=dtype)
    y_local = np.empty_like(rhs_local)
    block0_diag_out = np.empty((bs, bs), dtype=dtype)
    block0_arrow_out = np.empty((n_fe, bs), dtype=dtype)
    block0_rhs_out = np.empty(bs, dtype=dtype)
    cond_schur_out = np.zeros((bs, bs), dtype=dtype)
    arrow_schur_out = np.zeros((n_fe, bs), dtype=dtype)

    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diag_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_diag_events = [cp.cuda.Event(), cp.cuda.Event()]

    cp_diag_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_lower_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_arrow_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_buf_events = [cp.cuda.Event(), cp.cuda.Event()]

    diag_d = cp.empty((2, bs, bs), dtype=dtype)
    lower_d = cp.empty((2, bs, bs), dtype=dtype)
    arrow_d = cp.empty((2, n_fe, bs), dtype=dtype)
    buf_d = cp.empty((2, bs, bs), dtype=dtype)
    tip_d = cp.empty((n_fe, n_fe), dtype=dtype)
    block0_diag_d = cp.empty((bs, bs), dtype=dtype)
    block0_arrow_d = cp.empty((n_fe, bs), dtype=dtype)
    block0_rhs_d = cp.empty(bs, dtype=dtype)
    eye_d = eps_reg * cp.eye(bs, dtype=dtype)
    logdet_d = cp.zeros(1, dtype=dtype)
    prev_lower_y_d = cp.zeros(bs, dtype=dtype)

    L_diag_d = diag_d
    L_lower_d = lower_d
    L_arrow_d = arrow_d

    # F-order workspaces for solve_triangular (no cudaMalloc during loop)
    ws_L = cp.empty((bs, bs), dtype=dtype, order='F')
    ws_mat = cp.empty((bs, bs), dtype=dtype, order='F')
    ws_arrow = cp.empty((bs, n_fe), dtype=dtype, order='F')

    # --- Initial transfers ---
    cp_lower_h2d_events[1].record(stream=compute_stream)
    cp_arrow_h2d_events[1].record(stream=compute_stream)

    # Block 0 data (boundary, not factorized)
    block0_diag_d.set(arr=Q_diag[0] + eps_reg * np.eye(bs, dtype=dtype), stream=h2d_stream)
    block0_arrow_d.set(arr=Q_arrow[0], stream=h2d_stream)
    block0_rhs_d.set(arr=rhs_local[0], stream=h2d_stream)
    tip_d.set(arr=arrow_tip, stream=h2d_stream)

    # Block 1 data
    diag_d[1].set(arr=Q_diag[1], stream=h2d_stream)
    buf_d[1].set(arr=buffer_init.T, stream=h2d_stream)
    h2d_diag_events[1].record(stream=h2d_stream)

    if n_blocks > 2:
        lower_d[1].set(arr=Q_lower[1], stream=h2d_stream)
        h2d_lower_events[1].record(stream=h2d_stream)

    arrow_d[1].set(arr=Q_arrow[1], stream=h2d_stream)
    h2d_arrow_events[1].record(stream=h2d_stream)

    d2h_diag_events[1].record(stream=d2h_stream)

    # --- Main loop: blocks 1 .. n_blocks-2 ---
    for i in range(1, n_blocks - 1):
        b = i % 2
        bn = (i + 1) % 2

        # L_{i,i} = chol(A_{i,i})
        with compute_stream:
            compute_stream.wait_event(h2d_diag_events[b])
            diag_d[b] += eye_d
            L_diag_d[b] = cholesky(diag_d[b])
            ws_L[:] = L_diag_d[b]
            cp_diag_events[b].record(stream=compute_stream)

        d2h_stream.wait_event(cp_diag_events[b])
        L_diag_d[b].get(out=L_diag[i], stream=d2h_stream, blocking=False)
        d2h_diag_events[b].record(stream=d2h_stream)

        # H2D: pre-fetch next lower
        if i + 1 < n_blocks - 1:
            h2d_stream.wait_event(cp_lower_h2d_events[bn])
            lower_d[bn].set(arr=Q_lower[i + 1], stream=h2d_stream)
            h2d_lower_events[bn].record(stream=h2d_stream)

        # L_{i+1,i}
        with compute_stream:
            compute_stream.wait_event(h2d_lower_events[b])
            ws_mat[:] = lower_d[b].T
            cu_la.solve_triangular(ws_L, ws_mat, lower=True, overwrite_b=True)
            L_lower_d[b][:] = ws_mat.T
            cp_lower_events[b].record(stream=compute_stream)

        d2h_stream.wait_event(cp_lower_events[b])
        L_lower_d[b].get(out=L_lower[i], stream=d2h_stream, blocking=False)

        # H2D: pre-fetch next arrow
        h2d_stream.wait_event(cp_arrow_h2d_events[bn])
        arrow_d[bn].set(arr=Q_arrow[i + 1], stream=h2d_stream)
        h2d_arrow_events[bn].record(stream=h2d_stream)

        # L_{ndb+1,i}
        with compute_stream:
            compute_stream.wait_event(h2d_arrow_events[b])
            ws_arrow[:] = arrow_d[b].T
            cu_la.solve_triangular(ws_L, ws_arrow, lower=True, overwrite_b=True)
            L_arrow_d[b][:] = ws_arrow.T
            cp_arrow_events[b].record(stream=compute_stream)

        d2h_stream.wait_event(cp_arrow_events[b])
        L_arrow_d[b].get(out=L_arrow[i], stream=d2h_stream, blocking=False)

        # Buffer solve: L_{top,i} = buf @ L_{i,i}^{-T}
        with compute_stream:
            ws_mat[:] = buf_d[b].T
            cu_la.solve_triangular(ws_L, ws_mat, lower=True, overwrite_b=True)
            buf_d[b][:] = ws_mat.T
            cp_buf_events[b].record(stream=compute_stream)

        d2h_stream.wait_event(cp_buf_events[b])
        buf_d[b].get(out=buf_solved[i], stream=d2h_stream, blocking=False)

        # Forward sub: y_i = L^{-1}(rhs_i - L_lower_prev @ y_prev)
        with compute_stream:
            rhs_i_d = cp.asarray(rhs_local[i])
            rhs_i_d -= prev_lower_y_d
            y_i_d = cu_la.solve_triangular(
                ws_L, rhs_i_d, lower=True, overwrite_b=True)

        y_i_d.get(out=y_local[i], stream=compute_stream, blocking=False)

        # Logdet
        with compute_stream:
            diag_vals = cp.diag(L_diag_d[b])
            logdet_d += 2.0 * cp.sum(cp.log(cp.maximum(diag_vals, cp.finfo(dtype).eps)))

        # H2D: pre-fetch next diagonal
        h2d_stream.wait_event(d2h_diag_events[bn])
        diag_d[bn].set(arr=Q_diag[i + 1], stream=h2d_stream)
        h2d_diag_events[bn].record(stream=h2d_stream)

        # Schur and buffer updates
        with compute_stream:
            compute_stream.wait_event(h2d_diag_events[bn])
            # Standard Schur
            diag_d[bn] -= L_lower_d[b] @ L_lower_d[b].T
            arrow_d[bn] -= L_arrow_d[b] @ L_lower_d[b].T
            # Buffer: A_{top,i+1} = -L_{top,i} @ L_{i+1,i}.T
            buf_d[bn] = -buf_d[b] @ L_lower_d[b].T
            cp_lower_h2d_events[b].record(stream=compute_stream)

            tip_d -= L_arrow_d[b] @ L_arrow_d[b].T
            # A_{ndb+1,top} -= L_{ndb+1,i} @ L_{top,i}.T
            block0_arrow_d -= L_arrow_d[b] @ buf_d[b].T
            cp_arrow_h2d_events[b].record(stream=compute_stream)
            # A_{top,top} -= L_{top,i} @ L_{top,i}.T
            block0_diag_d -= buf_d[b] @ buf_d[b].T

            # Forward sub carries
            prev_lower_y_d = L_lower_d[b] @ y_i_d
            block0_rhs_d -= buf_d[b] @ y_i_d

    # --- D2H for last block data (unfactorized, for boundary extraction) ---
    b_last = (n_blocks - 1) % 2

    d2h_stream.wait_event(cp_lower_h2d_events[(n_blocks - 2) % 2])
    diag_d[b_last].get(out=L_diag[n_blocks - 1], stream=d2h_stream, blocking=False)
    arrow_d[b_last].get(out=L_arrow[n_blocks - 1], stream=d2h_stream, blocking=False)
    buf_d[b_last].get(out=buf_solved[n_blocks - 1], stream=d2h_stream, blocking=False)

    d2h_stream.wait_event(cp_arrow_h2d_events[(n_blocks - 2) % 2])
    block0_arrow_d.get(out=block0_arrow_out, stream=d2h_stream, blocking=False)
    block0_diag_d.get(out=block0_diag_out, stream=compute_stream, blocking=False)
    block0_rhs_d.get(out=block0_rhs_out, stream=compute_stream, blocking=False)

    # Last block RHS carry
    with compute_stream:
        rhs_last_d = cp.asarray(rhs_local[n_blocks - 1])
        rhs_last_d -= prev_lower_y_d
    rhs_last_out = np.empty(bs, dtype=dtype)
    rhs_last_d.get(out=rhs_last_out, stream=compute_stream, blocking=False)

    # Schur carry from last loop iteration (block n_blocks-2)
    with compute_stream:
        b_prev = (n_blocks - 2) % 2
        cond_schur_d = L_lower_d[b_prev] @ L_lower_d[b_prev].T
        arrow_schur_d = L_arrow_d[b_prev] @ L_lower_d[b_prev].T
        cond_schur_d.get(out=cond_schur_out, stream=compute_stream, blocking=False)
        arrow_schur_d.get(out=arrow_schur_out, stream=compute_stream, blocking=False)

    tip_d.get(out=arrow_tip, stream=d2h_stream, blocking=False)

    cp.cuda.Device().synchronize()

    logdet = float(logdet_d.get())

    del diag_d, lower_d, arrow_d, tip_d, buf_d, logdet_d, eye_d, prev_lower_y_d
    del L_diag_d, L_lower_d, L_arrow_d
    del block0_diag_d, block0_arrow_d, block0_rhs_d
    del ws_L, ws_mat, ws_arrow
    del compute_stream, h2d_stream, d2h_stream
    cp.get_default_memory_pool().free_all_blocks()

    # Arrow RHS: accumulated on CPU
    arrow_rhs = rhs_arrow.copy()
    for i in range(1, n_blocks - 1):
        arrow_rhs -= L_arrow[i] @ y_local[i]

    # Block 0 and last block have no factorized L values
    L_diag[0] = 0.0
    L_lower[0] = 0.0
    L_arrow[0] = 0.0
    y_local[0] = 0.0
    y_local[n_blocks - 1] = 0.0

    return (L_diag, L_lower, L_arrow, buf_solved, arrow_tip,
            block0_diag_out, block0_arrow_out, block0_rhs_out,
            rhs_last_out,
            y_local, arrow_rhs, logdet,
            cond_schur_out, arrow_schur_out)
