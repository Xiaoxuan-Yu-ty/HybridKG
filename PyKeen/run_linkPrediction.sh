#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

PYTHON_SCRIPT="link_prediction.py"

# Define the arrays of parameters you want to iterate through
KGS=("Prime_KGs" "Patient_KGs" "PPI_KGs")
DATASETS=("adni" "geo" "adni_OldTarget")
SCORING_TYPES=("all" "ecdf") # "std")
MODELS=("RotatE" "TransE" "TransR" "HolE" "ComplEx")

# Optional: Define standard paths if you want to override defaults
GRAPH_PATH="../datasets/"
OUTPUT_DIR="../PyKeen/results"

echo "======================================================="
echo "Starting Knowledge Graph Embedding ..."
echo "======================================================="

# Nested loops to iterate through all combinations
for kg in "${KGS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for scoring in "${SCORING_TYPES[@]}"; do
            for model in "${MODELS[@]}"; do

                echo ""
                echo "-------------------------------------------------------"
                echo "Running: KG=$kg, Dataset=$dataset, Scoring=$scoring, Model=$model"
                echo "-------------------------------------------------------"

                # Run the Python script with the current combination of arguments
                python "$PYTHON_SCRIPT" \
                    --graph_path "$GRAPH_PATH" \
                    --kg "$kg" \
                    --dataset "$dataset" \
                    --scoring_type "$scoring" \
                    --config "$CONFIG_DIR" \
                    --model "$model" \
                    --output_dir "$OUTPUT_DIR"

                # Check if the python script executed successfully
                if [ $? -eq 0 ]; then
                    echo "Successfully finished run for $model on $dataset ($kg - $scoring)."
                else
                    echo "ERROR: Run failed for KG=$kg, Dataset=$dataset, Scoring=$scoring, Model=$model"
                    # stop the whole script if one run fails
                    exit 1
                fi

            done
        done
    done
done

echo "======================================================="
echo "All experiment combinations have completed!"
echo "======================================================="