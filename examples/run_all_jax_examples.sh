#!/bin/bash

# Script to run all jax_run.py examples
# Each example saves output to its own A100_run_output_<precision>.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ARRAY_MODULE=cupy

# Precision levels to test
PRECISIONS=("float64")

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "Running all JAX examples"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Precisions: ${PRECISIONS[*]}"
echo "========================================"
echo ""

# Examples to skip
SKIP_EXAMPLES=("gst_large")

# Find all directories containing jax_run.py
for dir in "$SCRIPT_DIR"/*/; do
    if [ -f "${dir}jax_run.py" ]; then
        example_name=$(basename "$dir")

        # Skip excluded examples
        if [[ " ${SKIP_EXAMPLES[*]} " =~ " ${example_name} " ]]; then
            echo "Skipping: $example_name"
            continue
        fi

        # Create outputs directory if it doesn't exist
        mkdir -p "${dir}outputs"

        for precision in "${PRECISIONS[@]}"; do
            output_file="${dir}outputs/A100_log_${TIMESTAMP}_${precision}.txt"

            echo "Running: $example_name (precision: $precision)"

            # Create output file with header
            {
                echo "========================================"
                echo "Example: $example_name"
                echo "Precision: $precision"
                echo "Date: $(date)"
                echo "Host: $(hostname)"
                echo "========================================"
                echo ""
            } > "$output_file"

            # Run the script with MPI and capture both stdout and stderr
            cd "$dir"
            python jax_run.py --precision "$precision" 2>&1 | tee -a "$output_file"
            exit_code=${PIPESTATUS[0]}

            # Append footer
            {
                echo ""
                echo "========================================"
                echo "Exit code: $exit_code"
                echo "Completed: $(date)"
                echo "========================================"
            } >> "$output_file"

            echo "  -> Output saved to: $output_file"
            echo ""

            cd "$SCRIPT_DIR"
        done
    fi
done

echo "========================================"
echo "All examples completed at: $(date)"
echo "Precisions tested: ${PRECISIONS[*]}"
echo "========================================"
