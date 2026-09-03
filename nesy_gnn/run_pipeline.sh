#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define arrays for looping parameters
DATASETS=("AdNeuralMed" "ADNI" "GEO" "ANMerge")
K_VALUES=(4 8 12 16 20)
MODELS=("GAT" "GCN" "GIN" "RuleOnly")

# Base directory for outputs
OUTPUT_DIR="./nesy_gnn/results/explain"
KG_PATH="./data/KG/ad_network_with_reverse_edges.pkl"
KG_NAME="AD_KG"

# Loop through each dataset, k-value, and model type combination
for dataset in "${DATASETS[@]}"; do
    
    # Set paths dynamically based on the dataset
    if [ "$dataset" == "ANMerge" ]; then
        EXP_PATH="./data/ANMerge/ANMerge_TX_preprocessed/AnmergeTX_cleaned_exp.csv"
        DESIGN_PATH="./data/ANMerge/design_2cls_3cl_target.csv"
    elif [ "$dataset" == "ADNI" ]; then
        EXP_PATH="./data/ADNI/cleaned_gene_expression_data.csv"
        DESIGN_PATH="./data/ADNI/design_with_real_target.tsv"
    elif [ "$dataset" == "AdNeuralMed" ]; then
        EXP_PATH="./data/AdNeuralMed/allbatches_expression_mean.csv"
        DESIGN_PATH="./data/AdNeuralMed/allbatches_metadata_cleaned.csv"
    elif [ "$dataset" == "GEO" ]; then
        EXP_PATH="./data/GEO/GSE33000_ad_hd/GSE33000_exp_2cls.csv"
        DESIGN_PATH="./data/GEO/GSE33000_ad_hd/GSE33000_meta_2cls.csv"
    else
        echo "Unknown dataset: $dataset"
        continue
    fi

    for k in "${K_VALUES[@]}"; do
        for model in "${MODELS[@]}"; do

            # check if target directory exist
            TARGET_DIR="${OUTPUT_DIR}/${KG_NAME}/${dataset}/${k}/${model}"
            if [ -d "$TARGET_DIR" ]; then
                echo "Skippping Dataset=$dataset | k=$k | Model=$model (Directory already exists: $TARGET_DIR)"
                continue
            fi

            echo "========================================================"
            echo "Running Pipeline:"
            echo "  Dataset    : $dataset"
            echo "  k (KNN)    : $k"
            echo "  Model Type : $model"
            echo "========================================================"

            python -m nesy_gnn.train_pipeline \
                --KG "$KG_NAME" \
                --kg_path "$KG_PATH" \
                --exp_path "$EXP_PATH" \
                --design "$DESIGN_PATH" \
                --k "$k" \
                --output_dir "$OUTPUT_DIR" \
                --dataset "$dataset" \
                --model_type "$model" \
                --use_kg \
                --num_trial 100 \
                --n_jobs 1 \
                --n_splits 5 \
                --epochs 200 \
                --seed 42

            echo "Finished execution for Dataset=$dataset, k=$k, Model=$model"
            echo "--------------------------------------------------------"
            
        done
    done
done

echo "All pipeline runs completed successfully!"