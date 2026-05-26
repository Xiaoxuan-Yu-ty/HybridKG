#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

disease_kg="../data/KG/ad_network_with_reverse_edges.pkl"
healthy_kg="../data/KG/healthy_aging_reversed_remove_noncausal.pkl"

BASE_OUT_DIR="../datasets/Patient_KGs"

# Predefined grids
DATASETS=("adni_OldTarget" "geo" "adni")
SCORINGS=("std" "all")
METHODS=("dual_hybrid" "ADKG" "HealthyKG")

for dataset in "${DATASETS[@]}"; do

    if [ "$dataset" = "geo" ]; then
        DATASET_DIR="../data/GEO/GSE33000_ad_hd/sample_scoring"
        EXP_PATH=""../data/GEO/GSE33000_ad_hd/GSE33000_exp_2cls.csv""
    elif [ "$dataset" = "adni" ]; then
        DATASET_DIR="../data/ADNI/sample_scoring"
        EXP_PATH="../data/ADNI/adni_exp_2cls.csv"
    else
        # This handles "adni" and any other default cases
        DATASET_DIR="../data/ADNI/old_target"
        EXP_PATH="../data/ADNI/cleaned_gene_expression_data.csv"
    fi

    for scoring in "${SCORINGS[@]}"; do
    
        # Build the graph paths for this combination
        scoring_file="${DATASET_DIR}/sample_scoring_${scoring}.csv"

        # Skip checking files that aren't actually needed for the active method
        if [ ! -f "$scoring_file" ]; then
            echo "Skipping: Missing sample scoring file ($scoring_file)"
            continue
        fi
    

        for method in "${METHODS[@]}"; do   
            
            echo "\nRunning: $dataset | $scoring | $method"

            python pyg_graph_prep.py \
                --kg_disease "$disease_kg" \
                --kg_healthy "$healthy_kg" \
                --exp_path "$EXP_PATH" \
                --output_dir "$BASE_OUT_DIR" \
                --dataset "$dataset" \
                --scoring_path "$scoring_file" \
                --scoring_type "$scoring" \
                --method "$method"
              
        done
    done
done

# make the file excutable
#`chmod +x pyg_graph_generate.sh`
# run script
#`./pyg_graph_generate.sh`