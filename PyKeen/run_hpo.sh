#!/bin/bash
set -e

# Configuration Grids
DATASETS=("geo" "adni" "adni_OldTarget")
SCORINGS=("ecdf" "all")
METHODS=("ADKG") # "hybrid" "dual_hybrid" "merge" "HealthyKG")
MODELS=("RotatE") # "TransE" "TransR" "HolE" "ComplEx")

# Base directory for the graph files
GRAPH_PATH="../datasets/Patient_KGs"
OUTPUT_PATH="../PyKeen/results/PatientKG"

for dataset in "${DATASETS[@]}"; do
    for scoring in "${SCORINGS[@]}"; do
        for method in "${METHODS[@]}"; do
            
            # Define the expected graph filename
            graph_file="${GRAPH_PATH}/G_${dataset}_${method}_${scoring}.pkl"
            
            # Check if graph exists before triggering Python
            if [ -f "$graph_file" ]; then
                echo "Found graph: $graph_file. Launching HPO jobs..."
                
                for model in "${MODELS[@]}"; do
                    echo "--- Starting HPO for $model on $dataset ($method, $scoring) ---"
                    
                    python hpo.py \
                        --graph_path "$GRAPH_PATH" \
                        --dataset "$dataset" \
                        --scoring_type "$scoring" \
                        --method "$method" \
                        --model "$model" \
                        --output_dir "$OUTPUT_PATH"
                done
            else
                echo "Skipping: $graph_file not found."
            fi
            
        done
    done
done

# make the file excutable
#`chmod +x run_hpo.sh`
# run script
#`./run_hpo.sh`