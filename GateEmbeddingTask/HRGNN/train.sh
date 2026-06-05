#!/bin/bash

#SBATCH -n 1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres gpu:a100:1
#SBATCH -t 48:00:00
#SBATCH -J firegnn
#SBATCH -p gpu-amd
#SBATCH --mail-type=begin        # send a mail when job begins
#SBATCH --mail-type=end          # send a mail when job ends
#SBATCH --mail-type=fail         # send a mail if job fails
#SBATCH --mail-user=xiaoxuan.yu@scai-extern.fraunhofer.de
 
module load Miniforge3
# Activate system-wide Conda (Miniforge3)
source /opt/software/easybuild/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh

conda activate firegnn

cd "/home/xyu/thesis/HybridKG/SHGP/"
 
module load CUDA
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1

DATADIR=$1
OUTDIR=$2

echo "====================================================================================================="
echo "Running Testing import script"
echo `date +"%D %T"`
echo "-------------------------------------------------------------------"
echo "Python path:"
which python

/home/xyu/.conda/envs/firegnn/bin/python hpo_hrgnn.py

echo "-------------------------------------------------------------------"
echo `date +"%D %T"`
echo "====================================================================================================="
