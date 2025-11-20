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


if __name__ == "__main__":
    test = TestJAXObjectiveFunction()
    test.workspace = tempfile.mkdtemp(prefix='test_jax_objective_')

    try:

        test.test_jax_vs_original_single_theta()
        test.test_jax_vs_original_multiple_theta()
        test.test_jax_vs_original_different_problem_sizes()
        test.test_jax_vs_original_optimization_path()
        
    finally:
        if os.path.exists(test.workspace):
            shutil.rmtree(test.workspace)
