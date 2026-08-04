#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

GRAPH_DIR="../datasets/Patient_KGs"
BASE_OUT_DIR="../results/HRGNN"

# Predefined grids
DATASETS=("ANMerge" "ADNI" "AdNeuralMed" "GEO")
SCORINGS=("ecdf") # "std" "logfc")
MODELS=("hrgat" "hrgcn") # "hgt" "sage")
METHODS=("dual_hybrid" "merge")

# Base directory for outputs
OUTPUT_DIR="./nesy_gnn/results"

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

    for method in "${METHODs[@]}"; do
        for model in "${MODELS[@]}"; do
            
            echo "========================================================"
            echo "Running Pipeline:"
            echo "  Dataset    : $dataset"
            echo "  Method    : $method"
            echo "  Model Type : $model"
            echo "========================================================"

            python -m GateEmbeddingTask.HRGNN.train_pipeline \
                --exp_path "$EXP_PATH" \
                --design "$DESIGN_PATH" \
                --output_dir "$OUTPUT_DIR" \
                --dataset "$dataset" \
                --model_type "$model" \
                --num_trial 100 \
                --n_jobs 1 \
                --n_splits 3 \
                --epochs 200 \
                --seed 42

            echo "Finished execution for Dataset=$dataset, Method=$mothod, Model=$model"
            echo "--------------------------------------------------------"
            
        done
    done
done

echo "All pipeline runs completed successfully!"