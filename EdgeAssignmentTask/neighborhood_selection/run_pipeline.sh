#!/bin/bash

# Define your parameter combinations
DISEASE_KGS=("AD_KG" "PPI_KG" "Prime_KG")
DATASETS=("adni" "ANMerge" "geo")
AGGREGATORS=("fuzzy_attention")

for kg in "${DISEASE_KGS[@]}"; do
    # Set paths based on DiseaseKG
    case $kg in
        "AD_KG")   graph_path="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge/AD_KG/G_merge_AD_KG_healthy.pkl";   embedding_path="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge/AD_KG/embedding.csv" ;;
        "PPI_KG")  graph_path="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge/PPI_KG/G_merge_PPI_KG_healthy.pkl";  embedding_path="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge/PPI_KG/embedding.csv" ;;
        "Prime_KG") graph_path="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge/Prime_KG/G_merge_Prime_KG_healthy.pkl"; embedding_path="./EdgeAssignmentTask/neighborhood_selection/results/pykeen_kge/Prime_KG/embedding.csv" ;;
    esac

    for ds in "${DATASETS[@]}"; do
        # Set paths based on dataset
        case $ds in
            "adni")    exp_path="./data/ADNI/cleaned_gene_expression_data.csv";    design_path="./data/ADNI/design_with_real_target.tsv" ;;
            "geo")     exp_path="./data/GEO/GSE33000_ad_hd/GSE33000_exp_2cls.csv";     design_path="./data/GEO/GSE33000_ad_hd/GSE33000_meta_2cls.csv" ;;
            "ANMerge") exp_path="./data/ANMerge/ANMerge_exp_cleaned.csv";   design_path="./data/ANMerge/design_2cls_3cl_target.csv";;
        esac

        for agg in "${AGGREGATORS[@]}"; do
            echo "--------------------------------------------------------"
            echo "Running: KG=$kg | Dataset=$ds | Aggregator=$agg"
            echo "--------------------------------------------------------"

            python -m EdgeAssignmentTask.neighborhood_selection.train_pipeline \
                --DiseaseKG "$kg" \
                --dataset "$ds" \
                --aggregator "$agg" \
                --graph_path "$graph_path" \
                --embedding_path "$embedding_path" \
                --exp_path "$exp_path" \
                --design "$design_path" \
                --output_dir "./EdgeAssignmentTask/neighborhood_selection/results/label_homophily" \
                --num_trial 20


            if [ $? -ne 0 ]; then
                echo "Pipeline failed for $kg, $ds, $agg."
            fi
        done
    done
done

echo "All jobs completed."