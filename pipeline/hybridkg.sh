#!/bin/bash

# Arrays of your experimental variables
DATASETS=("adni" "geo")
MODELS=("gat" "sage" "hgt")
METHODS=("cls" "edge" "emb")

for ds in "${DATASETS[@]}"; do
    for mod in "${MODELS[@]}"; do
        for meth in "${METHODS[@]}"; do
            echo "Running: Dataset=$ds, Model=$mod, Method=$meth"
            
            python main_pipeline.py \
                --dataset "$ds" \
                --model "$mod" \
                --assign_method "$meth" \
                --scoring "ecdf" \
                --epochs 100
        done
    done
done

# To run this:

# Save it as hybridkg.sh.

# Run `chmod +x hybridkg.sh to make it executable`.

# Run `./hybridkg.sh`