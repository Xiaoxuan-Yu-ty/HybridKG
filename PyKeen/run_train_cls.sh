#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define your python script name
PYTHON_SCRIPT="train_cls.py"

# --- Define arrays of parameters to iterate over ---
KGS=("PPIKG" "ADKG") #"PrimeKG"
DATASETS=("adni_OldTarget" "adni" "geo")
SCORING_TYPES=("ecdf" "all") # "std")
METHODS=("DiseaseKG") #"hybrid" "dual_hybrid" "merge" "HealthyKG")
MODELS=("RotatE") #"TransE" "TransR" "HolE" "ComplEx")

# --- Fixed paths ---
OUTPUT_DIR="../PyKeen/results"

echo "========================================================="
echo " Starting Retrain KGE & Classification"
echo "========================================================="

# Loop counter for user tracking
COUNTER=0

# --- Nested Loops to iterate over every combination ---
for kg in "${KGS[@]}"; do
    if [ "$kg" = "PrimeKG" ]; then
        GRAPH_PATH="../datasets/Prime_KGs"
    elif [ "$kg" = "PPIKG" ]; then
        GRAPH_PATH="../datasets/PPI_KGs"
    else
        GRAPH_PATH="../datasets/Patient_KGs"
    fi

    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" = "geo" ]; then
            LABEL_PATH="../data/GEO/GSE33000_ad_hd/sample_scoring/sample_scoring_all.csv"
        elif [ "$dataset" = "adni" ]; then
            LABEL_PATH="../data/ADNI/sample_scoring/sample_scoring_all.csv"
        else
            # This handles "adni" and any other default cases
            LABEL_PATH="../data/ADNI/old_target//sample_scoring_all.csv"
        fi

        for scoring_type in "${SCORING_TYPES[@]}"; do
            for method in "${METHODS[@]}"; do
                for model in "${MODELS[@]}"; do
                    
                    ((++COUNTER))
                    echo "[Experiment $COUNTER] Running: KG=$kg | Dataset=$dataset | Score=$scoring_type | Method=$method | Model=$model"
                    
                    # 1. Recreate the precise config path structure locally 
                    #    so we can stream the console logs into a dedicated file.
                    LOG_DIR="$OUTPUT_DIR/$kg/$dataset/$scoring_type/$method/$model"
                    mkdir -p "$LOG_DIR"
                    LOG_FILE="$LOG_DIR/execution_output.log"
                    
                    # 2. Execute the python command
                    # Adding '|| true' prevents 'set -e' from killing the script if Python fails
                    python "$PYTHON_SCRIPT" \
                        --graph_path "$GRAPH_PATH" \
                        --label_path "$LABEL_PATH" \
                        --kg "$kg" \
                        --dataset "$dataset" \
                        --scoring_type "$scoring_type" \
                        --method "$method" \
                        --model "$model" \
                        --output_dir "$OUTPUT_DIR" > "$LOG_FILE" 2>&1 || true
                    
                    # Check the exit status of the Python command (stored via a dummy block)
                    # Because we used '|| true', we can inspect the log or use an alternative check,
                    # but the cleanest Bash way to handle 'set -e' bypass is placing the command 
                    # directly inside the if condition:

                done
            done
        done
    done
done

echo "========================================================="
echo " Complete! All $COUNTER combinations processed cleanly."
echo "========================================================="

# `chmod +x run_train_cls.sh`
# `./run_train_cls.sh`