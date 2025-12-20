import sys
import os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np
import jax

from dalia import xp, backend_flags
from dalia.configs import likelihood_config, dalia_config, submodels_config

print(f"Array module: {xp.__name__}, Backend flags: {backend_flags}")
print(f"JAX devices: {jax.devices()}")

from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import RegressionSubModel
from dalia.utils import get_host, print_msg
from examples_utils.parser_utils import parse_args
from examples_utils.jax_utils import (
    get_first_forward_value,
    profile_jax_execution,
    print_jax_ir,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_model():
    """Create the Poisson regression model."""
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    likelihood_dict = {
        "type": "poisson",
        "input_dir": f"{BASE_DIR}",
    }

    model = Model(
        submodels=[regression],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    return model


def run_with_method(model, gradient_method, max_iter, verbose=True):
    """Run DALIA optimization with specified gradient method."""
    dalia_dict = {
        "solver": {"type": "dense"},
        "gradient_method": gradient_method,
        "minimize": {
            "max_iter": max_iter,
            "gtol": 1e-1,
            "disp": verbose,
        },
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
    print_msg("--- Poisson Regression Model (Zero Hyperparameters) ---\n")

    args = parse_args()

    model = create_model()
    print_msg(model)

    initial_theta = model.theta.copy()
    print_msg(f"\nInitial theta: {initial_theta}")
    print_msg(f"Number of hyperparameters: {len(initial_theta)}")

    if len(initial_theta) > 0:
        print_msg("\nWARNING: Expected zero hyperparameters for pure Poisson regression!")

    # --- Run with Finite Differences ---
    print_msg("\n" + "="*70)
    print_msg("RUNNING WITH FINITE DIFFERENCES")
    print_msg("="*70)

    model_fd = create_model()
    dalia_fd = run_with_method(model_fd, "finite_diff", args.max_iter, verbose=False)

    t0 = time.perf_counter()
    f_fd = get_first_forward_value(dalia_fd)
    t_first_fd = time.perf_counter() - t0

    print_msg(f"First forward pass: {f_fd:.6f}")
    print_msg(f"Time for first forward: {t_first_fd:.4f}s")

    t0 = time.perf_counter()
    results_fd = dalia_fd.minimize()
    t_total_fd = time.perf_counter() - t0

    print_msg(f"\nOptimization completed in {t_total_fd:.2f}s")
    print_msg(f"Final latent params (x): {results_fd['x']}")

    # --- Run with JAX Autodiff ---
    print_msg("\n" + "="*70)
    print_msg("RUNNING WITH JAX AUTODIFF")
    print_msg("="*70)

    model_jax = create_model()
    dalia_jax = run_with_method(model_jax, "jax_autodiff", args.max_iter, verbose=False)

    ir_output_file = os.path.join(BASE_DIR, "jax_ir_output.txt")
    print_jax_ir(dalia_jax, output_file=ir_output_file)

    if args.profile:
        profile_results = profile_jax_execution(dalia_jax, output_dir=BASE_DIR, num_runs=10)

    t0 = time.perf_counter()
    f_jax = get_first_forward_value(dalia_jax)
    t_first_jax = time.perf_counter() - t0

    print_msg(f"First forward pass: {f_jax:.6f}")
    print_msg(f"Time for first forward: {t_first_jax:.4f}s")

    t0 = time.perf_counter()
    results_jax = dalia_jax.minimize()
    t_total_jax = time.perf_counter() - t0

    print_msg(f"\nOptimization completed in {t_total_jax:.2f}s")
    print_msg(f"Final latent params (x): {results_jax['x']}")

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

    print_msg(f"\nFirst forward time:")
    print_msg(f"  Finite Diff: {t_first_fd:.4f}s")
    print_msg(f"  JAX Autodiff: {t_first_jax:.4f}s")
    if t_first_jax > 0:
        print_msg(f"  Speedup: {t_first_fd / t_first_jax:.2f}x")

    print_msg(f"\nTotal wall-clock time (JIT + optimization):")
    print_msg(f"  Finite Diff: {t_first_fd + t_total_fd:.2f}s")
    print_msg(f"  JAX Autodiff: {t_first_jax + t_total_jax:.2f}s")
    if (t_first_jax + t_total_jax) > 0:
        print_msg(f"  Speedup: {(t_first_fd + t_total_fd) / (t_first_jax + t_total_jax):.2f}x")

    x_fd = get_host(results_fd['x'])
    x_jax = get_host(results_jax['x'])
    x_diff = np.linalg.norm(x_fd - x_jax)
    print_msg(f"\nFinal x difference (L2 norm): {x_diff:.2e}")

    # Compare with reference
    print_msg("\n" + "="*70)
    print_msg("COMPARISON WITH REFERENCE")
    print_msg("="*70)

    x_ref = np.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")

    print_msg(f"\nLatent param error (vs reference):")
    print_msg(f"  Finite Diff: {np.linalg.norm(x_fd - x_ref):.4e}")
    print_msg(f"  JAX Autodiff: {np.linalg.norm(x_jax - x_ref):.4e}")

    print_msg("\n--- Finished ---")
