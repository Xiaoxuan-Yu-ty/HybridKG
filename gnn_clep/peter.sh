#!/bin/bash

#SBATCH --job-name=ad_gnn
#SBATCH --output=peter_logs/train_%j.out
#SBATCH --error=peter_logs/train_%j.err
#SBATCH --time=48:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=5          # Recommendation: start with 1, increase if using MPI
#SBATCH --gpus=2
#SBATCH --mem=80G
#SBATCH --partition=gpu-nv

#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=xiaoxuan.yu@scai-extern.fraunhofer.de
 
# Load modules
module load Miniforge3
module load CUDA

# Dynamically source conda.sh using the base path
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate firegnn

# Navigate to project directory
cd "/home/xyu/thesis/HybridKG/"
 
export CUDA_LAUNCH_BLOCKING=1

KG="AD_KG"
AD_PATH="./datasets/base_kgs/ad_kg_with_reverse_edges.pkl"
PPI_PATH="./datasets/base_kgs/ppi_old_with_reverse_edges.pkl"
PRIME_PATH="./datasets/base_kgs/prime_kg_with_reverse_edges.pkl"

echo "====================================================================="
echo "Running GNN CLEP"
echo "$(date +"%D %T")"
echo "Python path: $(which python)"
echo "---------------------------------------------------------------------"

/home/xyu/.conda/envs/firegnn/bin/python -m gnn_clep.train \
    --DiseaseKG "$KG" \
    --kg_disease "$AD_PATH" \
    --hpo \
    --epochs 200 \
    --num_trial 100

echo "---------------------------------------------------------------------"
echo "$(date +"%D %T")"
echo "====================================================================="
