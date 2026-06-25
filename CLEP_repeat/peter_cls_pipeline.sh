#!/bin/bash

#SBATCH --job-name=rotate(EL)
#SBATCH --output=peter_logs/train_%j.out
#SBATCH --error=peter_logs/train_%j.err
#SBATCH --time=48:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1          # Recommendation: start with 1, increase if using MPI
#SBATCH --gpus=2
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

echo "====================================================================="
echo "Running Testing import script"
echo "$(date +"%D %T")"
echo "Python path: $(which python)"
echo "---------------------------------------------------------------------"

/home/xyu/.conda/envs/firegnn/bin/python -m CLEP_repeat.retrain_pipeline --num_trials=500 --n_jobs=5 --retrain_edgelist

echo "---------------------------------------------------------------------"
echo "$(date +"%D %T")"
echo "====================================================================="
