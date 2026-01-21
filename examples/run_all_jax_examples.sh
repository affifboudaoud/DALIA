#!/bin/bash

# Script to run all jax_run.py examples
# Each example saves output to its own GH200_run_output.txt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ARRAY_MODULE=cupy

echo "========================================"
echo "Running all JAX examples"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "========================================"
echo ""

# Find all directories containing jax_run.py
for dir in "$SCRIPT_DIR"/*/; do
    if [ -f "${dir}jax_run.py" ]; then
        example_name=$(basename "$dir")

        # Create outputs directory if it doesn't exist
        mkdir -p "${dir}outputs"
        output_file="${dir}outputs/GH200_run_output.txt"

        echo "Running: $example_name"

        # Create output file with header
        {
            echo "========================================"
            echo "Example: $example_name"
            echo "Date: $(date)"
            echo "Host: $(hostname)"
            echo "========================================"
            echo ""
        } > "$output_file"

        # Run the script with MPI and capture both stdout and stderr
        cd "$dir"
        srun python jax_run.py 2>&1 | tee -a "$output_file"
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
    fi
done

echo "========================================"
echo "All examples completed at: $(date)"
echo "========================================"
