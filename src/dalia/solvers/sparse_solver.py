# Copyright 2024-2025 DALIA authors. All rights reserved.

import time

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver


class SparseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the solver."""
        super().__init__(config)

        self.L: sp.sparse.spmatrix = None

        self.t_cholesky = 0.0
        self.t_solve = 0.0

    def cholesky(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        """Compute Cholesky factor of input matrix."""
        tic = time.perf_counter()

        A = sp.sparse.csc_matrix(A)

        LU = sp.sparse.linalg.splu(A, diag_pivot_thresh=0, permc_spec="NATURAL")

        if (LU.U.diagonal() > 0).all():  # Check the matrix A is positive definite.
            self.L = LU.L.dot(sp.sparse.diags(LU.U.diagonal() ** 0.5))
        else:
            raise ValueError("The matrix is not positive definite")

        self.t_cholesky = time.perf_counter() - tic

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Solve linear system using Cholesky factor."""
        tic = time.perf_counter()

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        sp.sparse.linalg.spsolve_triangular(self.L, rhs, lower=True, overwrite_b=True)
        sp.sparse.linalg.spsolve_triangular(
            self.L.T, rhs, lower=False, overwrite_b=True
        )

        self.t_solve = time.perf_counter() - tic

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        """Compute logdet of input matrix using Cholesky factor."""

        if self.L is None:
            raise ValueError("Cholesky factor not computed")

        return 2 * xp.sum(xp.log(self.L.diagonal()))

    def selected_inversion(self, **kwargs):
        # Placeholder for the selected inversion method.
        return super().selected_inversion(**kwargs)

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes"""
        if self.L is None:
            return 0

        return self.L.data.nbytes + self.L.indptr.nbytes + self.L.indices.nbytes