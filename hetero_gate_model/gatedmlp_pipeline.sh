#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

GRAPH_DIR="../datasets/Patient_KGs"
BASE_OUT_DIR="../results/GatedHeteroMLP"

# Predefined grids
DATASETS=("geo" "adni")
SCORINGS=("ecdf") # "std" "logfc")
MODELS=("gat") # "hgt" "sage")
METHODS=("composite" "dual_hybrid" "merge")

for dataset in "${DATASETS[@]}"; do
    for scoring in "${SCORINGS[@]}"; do
        for method in "${METHODS[@]}"; do
        
            # Build the graph paths for this combination
            disease_file="${GRAPH_DIR}/G_${dataset}_ADKG_${scoring}.pkl"
            healthy_file="${GRAPH_DIR}/G_${dataset}_HealthyKG_${scoring}.pkl"
            graph_file="${GRAPH_DIR}/G_${dataset}_${method}_${scoring}.pkl"

            # Skip checking files that aren't actually needed for the active method
            if [ "$method" = "composite" ]; then
                if [ ! -f "$disease_file" ] || [ ! -f "$healthy_file" ]; then
                    echo "Skipping: Missing composite files (disease/healthy) for $dataset + $scoring"
                    continue
                fi
            else
                if [ ! -f "$graph_file" ]; then
                    echo "Skipping: Missing graph file ($graph_file) for $dataset + $scoring + $method"
                    continue
                fi
            fi

            for model in "${MODELS[@]}"; do
                    
                # Construct dynamic subdirectory 
                output_path="${BASE_OUT_DIR}/${dataset}/${scoring}/${model}/${method}"
                
                # Check method to run corresponding script
                if [ "$method" = "composite" ]; then
                    python train_gatedmlp.py \
                        --graph_path_disease "$disease_file" \
                        --graph_path_healthy "$healthy_file" \
                        --output_dir "$BASE_OUT_DIR" \
                        --dataset "$dataset" \
                        --scoring "$scoring" \
                        --model "$model" \
                        --method "$method"
                else
                    python train_gatedmlp_hetero.py \
                        --graph_path "$graph_file" \
                        --output_dir "$BASE_OUT_DIR" \
                        --dataset "$dataset" \
                        --scoring "$scoring" \
                        --model "$model" \
                        --method "$method"
                fi

            done
        done
    done
done

# make the file excutable
#`chmod +x gatedmlp_pipeline.sh`
# run script
#`./gatedmlp_pipeline.sh`