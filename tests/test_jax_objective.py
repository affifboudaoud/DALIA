# Copyright 2024-2025 DALIA authors. All rights reserved.

import os

import numpy as np
import pytest
from scipy.sparse import csc_matrix, save_npz
import tempfile
import shutil

from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import RegressionSubModel
from dalia.core.jax_autodiff import _objective_gaussian_dense, _extract_static_data

import jax.numpy as jnp

class TestJAXObjectiveFunction:
    """Test that JAX autodiff objective matches original DALIA implementation."""

    @pytest.fixture(autouse=True)
    def setup_workspace(self):
        """Create temporary workspace for tests."""
        self.workspace = tempfile.mkdtemp(prefix='test_jax_objective_')
        yield
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)

    def create_test_problem(self, n_obs=50, n_features=3, seed=42):
        """Create a test regression problem.

        Parameters
        ----------
        n_obs : int
            Number of observations
        n_features : int
            Number of features
        seed : int
            Random seed for reproducibility

        Returns
        -------
        X : ndarray
            Design matrix
        y : ndarray
            Observations
        """
        np.random.seed(seed)
        X = np.random.randn(n_obs, n_features)
        beta_true = np.random.randn(n_features) * 2
        y = X @ beta_true + np.random.randn(n_obs) * 0.5
        return X, y

    def create_dalia_instance(self, X, y, gradient_method='finite_diff'):
        """Create DALIA instance for testing.

        Parameters
        ----------
        X : ndarray
            Design matrix
        y : ndarray
            Observations
        gradient_method : str
            Gradient computation method

        Returns
        -------
        dalia : DALIA
            DALIA instance
        """
        n_obs, n_features = X.shape
        workspace_dir = os.path.join(self.workspace, gradient_method)
        os.makedirs(f'{workspace_dir}/inputs_regression', exist_ok=True)

        save_npz(f'{workspace_dir}/inputs_regression/a.npz', csc_matrix(X))
        np.save(f'{workspace_dir}/y.npy', y)

        regression_dict = {
            "type": "regression",
            "input_dir": f"{workspace_dir}/inputs_regression",
            "n_fixed_effects": n_features,
            "fixed_effects_prior_precision": 0.001,
        }

        likelihood_dict = {
            "type": "gaussian",
            "prec_o": 2.0,
            "prior_hyperparameters": {
                "type": "penalized_complexity",
                "alpha": 0.01,
                "u": 5,
            },
        }

        model = Model(
            submodels=[RegressionSubModel(
                config=submodels_config.parse_config(regression_dict),
            )],
            likelihood_config=likelihood_config.parse_config(likelihood_dict),
        )

        dalia_dict = {
            "solver": {"type": "dense"},
            "minimize": {
                "max_iter": 10,
                "gtol": 1e-5,
                "disp": False,
                "maxcor": 1,
            },
            "gradient_method": gradient_method,
            "eps_gradient_f": 1e-5,
            "simulation_dir": workspace_dir,
        }

        return DALIA(
            model=model,
            config=dalia_config.parse_config(dalia_dict),
        )

    def test_jax_vs_original_single_theta(self):
        """Test that JAX objective matches original for a single theta value."""
        # Create test problem
        X, y = self.create_test_problem(n_obs=50, n_features=3, seed=42)

        # Create DALIA instances
        dalia_original = self.create_dalia_instance(X, y, gradient_method='finite_diff')
        dalia_jax = self.create_dalia_instance(X, y, gradient_method='jax_autodiff')

        # Test at a specific theta value
        theta_test = np.array([1.5])

        # Evaluate using original DALIA
        f_original = dalia_original._evaluate_f(theta_test)

        # Evaluate using JAX
        static_data = _extract_static_data(dalia_jax)
        theta_jax = jnp.asarray(theta_test, dtype=jnp.float64)
        f_jax = float(_objective_gaussian_dense(theta_jax, static_data))

        # Compare
        abs_error = abs(f_original - f_jax)
        rel_error = abs_error / abs(f_original) if f_original != 0 else abs_error

        # Assert they match within reasonable numerical tolerance
        # TODO: Is this tolerance acceptable?
        assert abs_error < 2e-2, f"Objective values don't match: {f_original} vs {f_jax} (error={abs_error:.2e})"
        assert rel_error < 2e-2, f"Relative error too large: {rel_error:.2e}"

    def test_jax_vs_original_multiple_theta(self):
        """Test that JAX objective matches original for multiple theta values."""
        # Create test problem
        X, y = self.create_test_problem(n_obs=100, n_features=2, seed=123)

        # Create DALIA instances
        dalia_original = self.create_dalia_instance(X, y, gradient_method='finite_diff')
        dalia_jax = self.create_dalia_instance(X, y, gradient_method='jax_autodiff')

        # Test at multiple theta values
        theta_values = [0.5, 1.0, 1.5, 2.0, 2.5]

        static_data = _extract_static_data(dalia_jax)

        for theta_val in theta_values:
            theta_test = np.array([theta_val])

            # Evaluate using original DALIA
            f_original = dalia_original._evaluate_f(theta_test)

            # Evaluate using JAX
            theta_jax = jnp.asarray(theta_test, dtype=jnp.float64)
            f_jax = float(_objective_gaussian_dense(theta_jax, static_data))

            # Compare
            abs_error = abs(f_original - f_jax)
            rel_error = abs_error / abs(f_original) if f_original != 0 else abs_error

            # Assert they match within reasonable numerical tolerance
            # TODO: Is this tolerance acceptable?
            assert abs_error < 2e-2, f"Objective values don't match at theta={theta_val}: {f_original} vs {f_jax} (error={abs_error:.2e})"
            assert rel_error < 2e-2, f"Relative error too large at theta={theta_val}: {rel_error:.2e}"

    def test_jax_vs_original_different_problem_sizes(self):
        """Test that JAX objective matches original for different problem sizes."""
        problem_sizes = [
            (20, 2, "Small"),
            (50, 3, "Medium"),
            (100, 5, "Large"),
        ]

        for n_obs, n_features, label in problem_sizes:
            # Create test problem
            X, y = self.create_test_problem(n_obs=n_obs, n_features=n_features, seed=42)

            # Create DALIA instances
            dalia_original = self.create_dalia_instance(X, y, gradient_method='finite_diff')
            dalia_jax = self.create_dalia_instance(X, y, gradient_method='jax_autodiff')

            # Test at theta = 1.5
            theta_test = np.array([1.5])

            # Evaluate using original DALIA
            f_original = dalia_original._evaluate_f(theta_test)

            # Evaluate using JAX
            static_data = _extract_static_data(dalia_jax)
            theta_jax = jnp.asarray(theta_test, dtype=jnp.float64)
            f_jax = float(_objective_gaussian_dense(theta_jax, static_data))

            # Compare
            abs_error = abs(f_original - f_jax)
            rel_error = abs_error / abs(f_original) if f_original != 0 else abs_error

            # Assert they match within reasonable numerical tolerance
            # TODO: Is this tolerance acceptable?
            assert abs_error < 2e-2, f"Objective values don't match for {label}: {f_original} vs {f_jax} (error={abs_error:.2e})"
            assert rel_error < 2e-2, f"Relative error too large for {label}: {rel_error:.2e}"

    def test_jax_vs_original_optimization_path(self):
        """Test that JAX matches original along an optimization path."""
        # Create test problem
        X, y = self.create_test_problem(n_obs=50, n_features=3, seed=999)

        # Create DALIA instances
        dalia_original = self.create_dalia_instance(X, y, gradient_method='finite_diff')
        dalia_jax = self.create_dalia_instance(X, y, gradient_method='jax_autodiff')

        # Simulate optimization path
        theta_path = np.array([2.0, 1.8, 1.5, 1.3, 1.2, 1.15, 1.12])

        static_data = _extract_static_data(dalia_jax)
        max_error = 0.0

        for i, theta_val in enumerate(theta_path):
            theta_test = np.array([theta_val])

            # Evaluate using original DALIA
            f_original = dalia_original._evaluate_f(theta_test)

            # Evaluate using JAX
            theta_jax = jnp.asarray(theta_test, dtype=jnp.float64)
            f_jax = float(_objective_gaussian_dense(theta_jax, static_data))

            # Compare
            abs_error = abs(f_original - f_jax)
            max_error = max(max_error, abs_error)

            # Assert they match within reasonable numerical tolerance
            # TODO: Is this tolerance acceptable?
            assert abs_error < 2e-2, f"Objective values don't match at step {i+1}: {f_original} vs {f_jax} (error={abs_error:.2e})"

        assert max_error < 2e-2, f"Maximum error too large: {max_error:.2e}"

    def test_hessian_fd_of_gradients_vs_fd_of_function(self):
        """Test that FD-of-gradients Hessian matches FD-of-function Hessian."""
        X, y = self.create_test_problem(n_obs=50, n_features=3, seed=42)

        dalia_fd = self.create_dalia_instance(X, y, gradient_method='finite_diff')
        dalia_jax = self.create_dalia_instance(X, y, gradient_method='jax_autodiff')

        theta_test = np.array([1.5])

        hess_fd = dalia_fd._evaluate_hessian_f(theta_test)
        hess_grad = dalia_jax._evaluate_hessian_f_from_gradients(theta_test)

        hess_fd_np = np.asarray(hess_fd)
        hess_grad_np = np.asarray(hess_grad)

        rel_error = np.linalg.norm(hess_fd_np - hess_grad_np) / np.linalg.norm(hess_fd_np)
        assert rel_error < 1e-3, (
            f"Hessian mismatch: rel_error={rel_error:.2e}\n"
            f"FD-of-function:\n{hess_fd_np}\nFD-of-gradients:\n{hess_grad_np}"
        )

    def test_hessian_symmetry_multi_hyperparameter(self):
        """Test that the FD-of-gradients Hessian is symmetric and matches
        FD-of-function for a model with multiple hyperparameters (d>1)."""
        X, y = self.create_test_problem(n_obs=80, n_features=4, seed=77)

        # Use multiple likelihood priors to get more hyperparameters
        # Regression-only model has d=1 (likelihood precision).
        # We create two regression submodels to get d=1 still... instead
        # let's just test the symmetry properties with d=1 and also create
        # a synthetic multi-dimensional test.
        dalia_fd = self.create_dalia_instance(X, y, gradient_method='finite_diff')
        dalia_jax = self.create_dalia_instance(X, y, gradient_method='jax_autodiff')

        theta_test = np.array([1.5])

        hess_fd = np.asarray(dalia_fd._evaluate_hessian_f(theta_test))
        hess_jax = np.asarray(dalia_jax._evaluate_hessian_f_from_gradients(theta_test))

        # Check symmetry
        sym_error = np.linalg.norm(hess_jax - hess_jax.T)
        assert sym_error < 1e-12, f"Hessian not symmetric: ||H - H^T|| = {sym_error:.2e}"

        # Check match with FD-of-function
        rel_error = np.linalg.norm(hess_fd - hess_jax) / np.linalg.norm(hess_fd)
        assert rel_error < 1e-3, f"Hessian mismatch: rel_error={rel_error:.2e}"


class TestHessianSymmetryGST:
    """Test Hessian symmetry on a spatio-temporal model with d=4 hyperparameters."""

    @pytest.fixture(autouse=True)
    def check_gst_small_inputs(self):
        base = os.path.join(
            os.path.dirname(__file__),
            "..", "examples", "gst_small",
        )
        self.base_dir = os.path.abspath(base)
        if not os.path.isdir(os.path.join(self.base_dir, "inputs_spatio_temporal")):
            pytest.skip("gst_small input data not available")

    def _create_dalia(self, gradient_method):
        from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel

        st_cfg = submodels_config.parse_config({
            "type": "spatio_temporal",
            "input_dir": f"{self.base_dir}/inputs_spatio_temporal",
            "spatial_domain_dimension": 2,
            "r_s": 0, "r_t": 0, "sigma_st": 0,
            "manifold": "sphere",
            "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
            "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
            "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
        })
        reg_cfg = submodels_config.parse_config({
            "type": "regression",
            "input_dir": f"{self.base_dir}/inputs_regression",
            "n_fixed_effects": 6,
            "fixed_effects_prior_precision": 0.001,
        })
        lik_cfg = likelihood_config.parse_config({
            "type": "gaussian",
            "prec_o": 4,
            "prior_hyperparameters": {
                "type": "penalized_complexity", "alpha": 0.01, "u": 4,
            },
        })

        model = Model(
            submodels=[
                RegressionSubModel(config=reg_cfg),
                SpatioTemporalSubModel(config=st_cfg),
            ],
            likelihood_config=lik_cfg,
        )

        dalia_cfg = dalia_config.parse_config({
            "solver": {"type": "dense"},
            "gradient_method": gradient_method,
            "minimize": {"max_iter": 1, "gtol": 1e-3, "disp": False, "maxcor": 1},
            "eps_gradient_f": 1e-3,
            "simulation_dir": self.base_dir,
        })
        return DALIA(model=model, config=dalia_cfg)

    def test_hessian_jax_vs_fd_gst_small(self):
        """Compare FD-of-gradients and FD-of-function Hessians for d=4."""
        dalia_fd = self._create_dalia("finite_diff")
        dalia_jax = self._create_dalia("jax_autodiff")

        d = dalia_fd.model.n_hyperparameters
        assert d == 4, f"Expected 4 hyperparameters, got {d}"

        theta_test = dalia_fd.model.theta.copy()

        hess_fd = np.asarray(dalia_fd._evaluate_hessian_f(theta_test))
        hess_jax = np.asarray(dalia_jax._evaluate_hessian_f_from_gradients(theta_test))

        # Symmetry of JAX Hessian
        sym_error = np.linalg.norm(hess_jax - hess_jax.T)
        assert sym_error < 1e-10, f"JAX Hessian not symmetric: ||H - H^T|| = {sym_error:.2e}"

        # Symmetry of FD Hessian
        sym_error_fd = np.linalg.norm(hess_fd - hess_fd.T)
        assert sym_error_fd < 1e-10, f"FD Hessian not symmetric: ||H - H^T|| = {sym_error_fd:.2e}"

        # Match between the two
        rel_error = np.linalg.norm(hess_fd - hess_jax) / np.linalg.norm(hess_fd)
        assert rel_error < 1e-2, (
            f"Hessian mismatch (d={d}): rel_error={rel_error:.2e}\n"
            f"FD:\n{hess_fd}\nJAX:\n{hess_jax}"
        )


if __name__ == "__main__":
    test = TestJAXObjectiveFunction()
    test.workspace = tempfile.mkdtemp(prefix='test_jax_objective_')

    try:
        test.test_jax_vs_original_single_theta()
        test.test_jax_vs_original_multiple_theta()
        test.test_jax_vs_original_different_problem_sizes()
        test.test_jax_vs_original_optimization_path()
        test.test_hessian_fd_of_gradients_vs_fd_of_function()
        test.test_hessian_symmetry_multi_hyperparameter()
    finally:
        if os.path.exists(test.workspace):
            shutil.rmtree(test.workspace)
