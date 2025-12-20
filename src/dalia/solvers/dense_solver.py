# Copyright 2024-2025 DALIA authors. All rights reserved.

import time

from dalia import NDArray, sp, xp
from dalia.configs.dalia_config import SolverConfig
from dalia.core.solver import Solver


class DenseSolver(Solver):
    def __init__(
        self,
        config: SolverConfig,
        **kwargs,
    ) -> None:
        """Initializes the DenseSolver class.

        Parameters
        ----------
        config : SolverConfig
            Configuration object for the solver.
        n : int
            Size of the matrix.

        Returns
        -------
        None
        """
        super().__init__(config)

        self.n: int = kwargs.get("n", None)
        assert self.n is not None, "The size of the matrix must be provided."

        self.L: NDArray = xp.zeros((self.n, self.n), dtype=xp.float64)
        self.A_inv = None

        self.t_cholesky = 0.0
        self.t_solve = 0.0

    def cholesky(self, A: NDArray, **kwargs) -> None:
        tic = time.perf_counter()
        self.L[:] = A.todense()
        self.L = xp.linalg.cholesky(self.L)
        self.t_cholesky = time.perf_counter() - tic

    def solve(
        self,
        rhs: NDArray,
        **kwargs,
    ) -> NDArray:
        tic = time.perf_counter()
        rhs[:] = sp.linalg.solve_triangular(self.L, rhs, lower=True, overwrite_b=True)
        rhs[:] = sp.linalg.solve_triangular(
            self.L.T, rhs, lower=False, overwrite_b=True
        )
        self.t_solve = time.perf_counter() - tic

        return rhs

    def logdet(
        self,
        **kwargs,
    ) -> float:
        return 2 * xp.sum(xp.log(xp.diag(self.L)))

    # TODO: optimize for memory??
    def selected_inversion(self, **kwargs) -> None:

        L_inv = xp.eye(self.L.shape[0])
        L_inv[:] = sp.linalg.solve_triangular(
            self.L, L_inv, lower=True, overwrite_b=True
        )
        self.A_inv = L_inv.T @ L_inv

        return self.A_inv

    def _structured_to_spmatrix(self, A: sp.sparse.spmatrix, **kwargs) -> None:
        B = A.tocoo()
        B.data = self.A_inv[B.row, B.col]

        return B

    def get_solver_memory(self) -> int:
        """Return the memory used by the solver in number of bytes."""
        solver_mem = 2 * self.n * self.n * xp.dtype(xp.float64).itemsize

        return solver_mem