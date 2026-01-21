import sys
import os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from examples_utils.parser_utils import parse_args

args = parse_args()

from dalia.core.jax_autodiff import configure_jax_precision
configure_jax_precision(args.precision)

import numpy as np
import jax

from dalia import xp, backend_flags
from dalia.configs import (
    likelihood_config,
    models_config,
    dalia_config,
    submodels_config,
)

print(f"Array module: {xp.__name__}, Backend flags: {backend_flags}")
print(f"JAX devices: {jax.devices()}")

from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.models import CoregionalModel
from dalia.utils import print_msg
from dalia.submodels import RegressionSubModel, SpatialSubModel
from examples_utils.jax_utils import (
    get_first_forward_and_gradient,
    profile_jax_execution,
    print_jax_ir,
)

SEED = 63
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_model():
    """Create the Gaussian Coregional (2 variates) spatial model with regression."""
    nv = 2
    ns = 1818
    nt = 1
    nb = 2
    dim_theta = 7

    theta_ref_file = f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_ref.npy"
    theta_ref = np.load(theta_ref_file)
    theta_initial = theta_ref + 0.3 * np.random.randn(dim_theta)

    # Spatial submodel 1
    spatial_1_dict = {
        "type": "spatial",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[0],
        "sigma_e": 0,
        "ph_s": {"type": "gaussian", "mean": theta_ref[0], "precision": 0.5},
        "ph_e": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }
    spatial_1 = SpatialSubModel(
        config=submodels_config.parse_config(spatial_1_dict),
    )
    # Regression submodel 1
    regression_1_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_1/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression_1 = RegressionSubModel(
        config=submodels_config.parse_config(regression_1_dict),
    )
    # Likelihood submodel 1
    likelihood_1_dict = {
        "type": "gaussian",
        "prec_o": theta_initial[1],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[1],
            "precision": 0.5,
        },
    }
    model_1 = Model(
        submodels=[spatial_1, regression_1],
        likelihood_config=likelihood_config.parse_config(likelihood_1_dict),
    )

    # Spatial submodel 2
    spatial_2_dict = {
        "type": "spatial",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_spatial",
        "spatial_domain_dimension": 2,
        "r_s": theta_initial[2],
        "sigma_e": 0,
        "ph_s": {"type": "gaussian", "mean": theta_ref[2], "precision": 0.5},
        "ph_e": {"type": "gaussian", "mean": 0.0, "precision": 0.5},
    }
    spatial_2 = SpatialSubModel(
        config=submodels_config.parse_config(spatial_2_dict),
    )
    # Regression submodel 2
    regression_2_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/model_2/inputs_regression",
        "n_fixed_effects": 1,
        "fixed_effects_prior_precision": 0.001,
    }
    regression_2 = RegressionSubModel(
        config=submodels_config.parse_config(regression_2_dict),
    )
    # Likelihood submodel 2
    likelihood_2_dict = {
        "type": "gaussian",
        "prec_o": theta_initial[3],
        "prior_hyperparameters": {
            "type": "gaussian",
            "mean": theta_ref[3],
            "precision": 0.5,
        },
    }
    model_2 = Model(
        submodels=[spatial_2, regression_2],
        likelihood_config=likelihood_config.parse_config(likelihood_2_dict),
    )

    # Create coregional model
    coreg_dict = {
        "type": "coregional",
        "n_models": 2,
        "sigmas": [theta_initial[4], theta_initial[5]],
        "lambdas": [theta_initial[6]],
        "ph_sigmas": [
            {"type": "gaussian", "mean": theta_ref[4], "precision": 0.5},
            {"type": "gaussian", "mean": theta_ref[5], "precision": 0.5},
        ],
        "ph_lambdas": [
            {"type": "gaussian", "mean": 0.0, "precision": 0.5},
        ],
    }
    coreg_model = CoregionalModel(
        models=[model_1, model_2],
        coregional_model_config=models_config.parse_config(coreg_dict),
    )

    return coreg_model


def run_with_method(model, gradient_method, max_iter, verbose=True):
    """Run DALIA optimization with specified gradient method."""
    dalia_dict = {
        "solver": {"type": "dense"},
        "gradient_method": gradient_method,
        "minimize": {
            "max_iter": max_iter,
            "gtol": 1e-3,
            "disp": verbose,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "eps_hessian_f": 5 * 1e-3,
        "simulation_dir": ".",
    }

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    return dalia


if __name__ == "__main__":
    print_msg("--- JAX vs Finite Differences Comparison ---")
    print_msg("--- Gaussian Coregional (2 variates) Spatial Model with Regression ---\n")

    nv = 2
    ns = 1818
    nt = 1
    nb = 2

    model = create_model()
    print_msg(model)

    initial_theta = model.theta.copy()
    print_msg(f"\nInitial theta: {initial_theta}")
    print_msg(f"Number of hyperparameters: {len(initial_theta)}")

    # --- Run with Finite Differences ---
    print_msg("\n" + "="*70)
    print_msg("RUNNING WITH FINITE DIFFERENCES")
    print_msg("="*70)

    np.random.seed(SEED)
    model_fd = create_model()
    dalia_fd = run_with_method(model_fd, "finite_diff", args.max_iter, verbose=False)

    t0 = time.perf_counter()
    f_fd, grad_fd = get_first_forward_and_gradient(dalia_fd)
    t_first_fd = time.perf_counter() - t0

    print_msg(f"First forward pass: {f_fd:.6f}")
    print_msg(f"First gradient: {grad_fd}")
    print_msg(f"Time for first forward+gradient: {t_first_fd:.4f}s")

    t0 = time.perf_counter()
    results_fd = dalia_fd.run()
    t_total_fd = time.perf_counter() - t0

    print_msg(f"\nOptimization completed in {t_total_fd:.2f}s")
    print_msg(f"Final theta: {results_fd['theta']}")
    print_msg(f"Final objective: {results_fd['f']:.6f}")

    # --- Run with JAX Autodiff ---
    print_msg("\n" + "="*70)
    print_msg("RUNNING WITH JAX AUTODIFF")
    print_msg("="*70)

    np.random.seed(SEED)
    model_jax = create_model()
    dalia_jax = run_with_method(model_jax, "jax_autodiff", args.max_iter, verbose=False)

    ir_output_file = os.path.join(BASE_DIR, "jax_ir_output.txt")
    print_jax_ir(dalia_jax, output_file=ir_output_file)

    if args.profile:
        profile_results = profile_jax_execution(dalia_jax, output_dir=BASE_DIR, num_runs=10)

    t0 = time.perf_counter()
    f_jax, grad_jax = get_first_forward_and_gradient(dalia_jax)
    t_first_jax = time.perf_counter() - t0

    print_msg(f"First forward pass: {f_jax:.6f}")
    print_msg(f"First gradient: {grad_jax}")
    print_msg(f"Time for first forward+gradient: {t_first_jax:.4f}s")

    t0 = time.perf_counter()
    results_jax = dalia_jax.run()
    t_total_jax = time.perf_counter() - t0

    print_msg(f"\nOptimization completed in {t_total_jax:.2f}s")
    print_msg(f"Final theta: {results_jax['theta']}")
    print_msg(f"Final objective: {results_jax['f']:.6f}")

    # --- Numerical Validation ---
    print_msg("\n" + "="*70)
    print_msg("NUMERICAL VALIDATION")
    print_msg("="*70)

    f_diff = abs(f_fd - f_jax)
    f_rel_diff = f_diff / abs(f_fd) if f_fd != 0 else f_diff
    print_msg(f"\nFirst Forward Pass Comparison:")
    print_msg(f"  Finite Diff: {f_fd:.10f}")
    print_msg(f"  JAX Autodiff: {f_jax:.10f}")
    print_msg(f"  Absolute diff: {f_diff:.2e}")
    print_msg(f"  Relative diff: {f_rel_diff:.2e}")

    forward_pass_match = f_rel_diff < 1e-6
    print_msg(f"  Status: {'PASS' if forward_pass_match else 'FAIL'} (tol=1e-6)")

    grad_diff = np.abs(grad_fd - grad_jax)
    grad_rel_diff = grad_diff / (np.abs(grad_fd) + 1e-10)

    print_msg(f"\nFirst Gradient Comparison:")
    print_msg(f"  Finite Diff: {grad_fd}")
    print_msg(f"  JAX Autodiff: {grad_jax}")
    print_msg(f"  Absolute diff: {grad_diff}")
    print_msg(f"  Relative diff: {grad_rel_diff}")
    print_msg(f"  Max relative diff: {np.max(grad_rel_diff):.2e}")

    gradient_reasonable = np.max(grad_rel_diff) < 0.1
    print_msg(f"  Status: {'PASS' if gradient_reasonable else 'CHECK'} (tol=10%)")
    print_msg("  Note: Some difference expected (finite diff has O(eps^2) error)")

    # --- Performance Comparison ---
    print_msg("\n" + "="*70)
    print_msg("PERFORMANCE COMPARISON")
    print_msg("="*70)

    n_iter_fd = len(results_fd.get('f_values', [1]))
    n_iter_jax = len(results_jax.get('f_values', [1]))

    print_msg(f"\nIterations to convergence:")
    print_msg(f"  Finite Diff: {n_iter_fd}")
    print_msg(f"  JAX Autodiff: {n_iter_jax}")

    print_msg(f"\nTotal optimization time:")
    print_msg(f"  Finite Diff: {t_total_fd:.2f}s")
    print_msg(f"  JAX Autodiff: {t_total_jax:.2f}s")
    if t_total_jax > 0:
        print_msg(f"  Speedup: {t_total_fd / t_total_jax:.2f}x")

    print_msg(f"\nFirst forward+gradient time:")
    print_msg(f"  Finite Diff: {t_first_fd:.4f}s")
    print_msg(f"  JAX Autodiff: {t_first_jax:.4f}s")
    if t_first_jax > 0:
        print_msg(f"  Speedup: {t_first_fd / t_first_jax:.2f}x")

    print_msg(f"\nTotal wall-clock time (JIT + optimization):")
    print_msg(f"  Finite Diff: {t_first_fd + t_total_fd:.2f}s")
    print_msg(f"  JAX Autodiff: {t_first_jax + t_total_jax:.2f}s")
    if (t_first_jax + t_total_jax) > 0:
        print_msg(f"  Speedup: {(t_first_fd + t_total_fd) / (t_first_jax + t_total_jax):.2f}x")

    print_msg(f"\nFinal objective values:")
    print_msg(f"  Finite Diff: {results_fd['f']:.6f}")
    print_msg(f"  JAX Autodiff: {results_jax['f']:.6f}")

    theta_diff = np.linalg.norm(results_fd['theta'] - results_jax['theta'])
    print_msg(f"\nFinal theta difference (L2 norm): {theta_diff:.2e}")

    # Compare with reference
    print_msg("\n" + "="*70)
    print_msg("COMPARISON WITH REFERENCE")
    print_msg("="*70)

    theta_ref = np.load(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/theta_ref.npy")
    x_ref = np.load(f"{BASE_DIR}/inputs_nv{nv}_ns{ns}_nt{nt}_nb{nb}/reference_outputs/x_ref.npy")

    print_msg(f"\nTheta error (vs reference):")
    print_msg(f"  Finite Diff: {np.linalg.norm(results_fd['theta'] - theta_ref):.4e}")
    print_msg(f"  JAX Autodiff: {np.linalg.norm(results_jax['theta'] - theta_ref):.4e}")

    print_msg(f"\nLatent param error (vs reference):")
    print_msg(f"  Finite Diff: {np.linalg.norm(results_fd['x'] - x_ref):.4e}")
    print_msg(f"  JAX Autodiff: {np.linalg.norm(results_jax['x'] - x_ref):.4e}")

    print_msg("\n--- Finished ---")
