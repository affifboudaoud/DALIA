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

from dalia import xp, backend_flags
from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import RegressionSubModel, SpatioTemporalSubModel
from dalia.utils import print_msg
from examples_utils.jax_utils import (
    get_first_forward_and_gradient,
    profile_jax_execution,
    print_jax_ir,
)
from examples_utils.benchmark_utils import run_benchmark

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Array module: {xp.__name__}, Backend flags: {backend_flags}")

def create_model():
    """Create the spatio-temporal + regression model."""
    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0.0,
        "r_t": 0.0,
        "sigma_st": 0.0,
        "manifold": "plane",
        "ph_s": {"type": "gaussian", "mean": 0.03972077083991806, "precision": 0.5},
        "ph_t": {"type": "gaussian", "mean": 2.3931471805599456, "precision": 0.5},
        "ph_st": {"type": "gaussian", "mean": 1.4379142862353824, "precision": 0.5},
    }
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_dict),
    )

    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_regression",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 4,
        "prior_hyperparameters": {"type": "gaussian", "mean": 1.4, "precision": 0.5},
    }

    model = Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    return model


def run_with_method(model, gradient_method, max_iter, verbose=True):
    """Run DALIA optimization with specified gradient method."""
    dalia_dict = {
        "solver": {"type": "serinv"},
        "gradient_method": gradient_method,
        "minimize": {
            "max_iter": max_iter,
            "gtol": 1e-3,
            "disp": verbose,
            "maxcor": len(model.theta),
        },
        "f_reduction_tol": 1e-4,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }

    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

    return dalia


if __name__ == "__main__":
    print_msg("--- JAX vs Finite Differences Comparison ---")
    print_msg("--- Gaussian Spatio-Temporal Model (Medium) with Regression ---\n")

    model = create_model()
    print_msg(model)

    initial_theta = model.theta.copy()
    print_msg(f"\nInitial theta: {initial_theta}")

    if args.benchmark_mode:
        dalia_fd = None
        dalia_jax = None
        if args.benchmark_method in ["finite_diff", "both"]:
            model_fd = create_model()
            dalia_fd = run_with_method(model_fd, "finite_diff", args.max_iter, verbose=False)
        if args.benchmark_method in ["jax_autodiff", "both"]:
            model_jax = create_model()
            dalia_jax = run_with_method(model_jax, "jax_autodiff", args.max_iter, verbose=False)
        run_benchmark(dalia_fd, dalia_jax, args, "gst_medium")
        exit(0)

    # --- Run with Finite Differences ---
    print_msg("\n" + "="*70)
    print_msg("RUNNING WITH FINITE DIFFERENCES")
    print_msg("="*70)

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
    print_msg(f"Number of iterations: {len(results_fd.get('f_values', []))}")

    # --- Run with JAX Autodiff ---
    print_msg("\n" + "="*70)
    print_msg("RUNNING WITH JAX AUTODIFF")
    print_msg("="*70)

    model_jax = create_model()
    dalia_jax = run_with_method(model_jax, "jax_autodiff", args.max_iter, verbose=False)

    # Print JAX IR to file
    ir_output_file = os.path.join(BASE_DIR, "jax_ir_output.txt")
    print_jax_ir(dalia_jax, output_file=ir_output_file)

    # Profile JAX execution (optional)
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
    print_msg(f"Number of iterations: {len(results_jax.get('f_values', []))}")

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

    print_msg(f"\nTotal optimization time:")
    print_msg(f"  Finite Diff: {t_total_fd:.2f}s")
    print_msg(f"  JAX Autodiff: {t_total_jax:.2f}s")
    print_msg(f"  Speedup: {t_total_fd / t_total_jax:.2f}x")

    print_msg(f"\nIterations to convergence:")
    print_msg(f"  Finite Diff: {n_iter_fd}")
    print_msg(f"  JAX Autodiff: {n_iter_jax}")

    print_msg(f"\nTime per iteration:")
    print_msg(f"  Finite Diff: {t_total_fd / max(n_iter_fd, 1):.4f}s")
    print_msg(f"  JAX Autodiff: {t_total_jax / max(n_iter_jax, 1):.4f}s")

    print_msg(f"\nFirst forward+gradient time:")
    print_msg(f"  Finite Diff: {t_first_fd:.4f}s")
    print_msg(f"  JAX Autodiff: {t_first_jax:.4f}s")
    print_msg(f"  Speedup: {t_first_fd / t_first_jax:.2f}x")

    print_msg(f"\nTotal wall-clock time (JIT + optimization):")
    print_msg(f"  Finite Diff: {t_first_fd + t_total_fd:.2f}s")
    print_msg(f"  JAX Autodiff: {t_first_jax + t_total_jax:.2f}s")
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

    theta_ref = np.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    x_ref = np.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")

    print_msg(f"\nTheta error (vs reference):")
    print_msg(f"  Finite Diff: {np.linalg.norm(results_fd['theta'] - theta_ref):.4e}")
    print_msg(f"  JAX Autodiff: {np.linalg.norm(results_jax['theta'] - theta_ref):.4e}")

    print_msg(f"\nLatent param error (vs reference):")
    print_msg(f"  Finite Diff: {np.linalg.norm(results_fd['x'] - x_ref):.4e}")
    print_msg(f"  JAX Autodiff: {np.linalg.norm(results_jax['x'] - x_ref):.4e}")

    print_msg("\n--- Finished ---")
