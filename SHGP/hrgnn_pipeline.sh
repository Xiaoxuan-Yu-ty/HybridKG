#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

GRAPH_DIR="../datasets/Patient_KGs"
BASE_OUT_DIR="../results/HRGNN"

# Predefined grids
DATASETS=("adni")
SCORINGS=("ecdf") # "std" "logfc")
MODELS=("gat" "gcn") # "hgt" "sage")
METHODS=("dual_hybrid" "merge")

for dataset in "${DATASETS[@]}"; do
    for scoring in "${SCORINGS[@]}"; do
        for method in "${METHODS[@]}"; do
        
            # Build the graph paths for this combination
            graph_file="${GRAPH_DIR}/G_${dataset}_${method}_${scoring}.pkl"

            # Skip checking files that aren't actually existing for the active method
            if [ ! -f "$graph_file" ]; then
                echo "Skipping: Missing graph file ($graph_file) for $dataset + $scoring + $method"
                continue
            fi
            
            for model in "${MODELS[@]}"; do
    
                python train_hrgnn.py \
                    --graph_path "$graph_file" \
                    --output_dir "$BASE_OUT_DIR" \
                    --dataset "$dataset" \
                    --scoring "$scoring" \
                    --model "$model" \
                    --method "$method"

            done
        done
    done
done

# make the file excutable
#`chmod +x hrgnn_pipeline.sh`
# run script
#`./hrgnn_pipeline.sh`