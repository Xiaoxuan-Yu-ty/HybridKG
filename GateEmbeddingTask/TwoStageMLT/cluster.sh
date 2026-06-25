#!/bin/bash

#SBATCH -n 1
#SBATCH --output=leo_logs/train_%j.out
#SBATCH --error=leo_logs/train_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH -t 48:00:00
#SBATCH -J MLT
#SBATCH -p gpu-amd
#SBATCH --mail-type=begin        # send a mail when job begins
#SBATCH --mail-type=end          # send a mail when job ends
#SBATCH --mail-type=fail         # send a mail if job fails
#SBATCH --mail-user=xiaoxuan.yu@scai-extern.fraunhofer.de
 
# Load Miniforge and activate environment
module load Miniforge3
source /opt/software/easybuild/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate firegnn

# Navigate to project directory
cd "/home/xyu/thesis/HybridKG/"
 
# Load CUDA environment
module load CUDA
# Note: CUDA_VISIBLE_DEVICES is automatically set by Slurm based on --gres=gpu:a100:1
export CUDA_LAUNCH_BLOCKING=1

MODEL=('hrgat') # 'hrgcn' 'rgcn' 'rgat' 'hgt' 'hgat' 'graphsage')

echo "====================================================================================================="
echo "Running Testing import script"
echo "$(date +"%D %T")"
echo "-------------------------------------------------------------------"
echo "Python path:"
which python

for model in "${MODEL[@]}"; do
    echo "Running model: $model"
    # standard 'python' will now point to your activated firegnn environment automatically
    /home/xyu/.conda/envs/firegnn/bin/python -m GateEmbeddingTask.TwoStageMLT.train_pipeline --num_trial 50  --epochs 200
done

echo "-------------------------------------------------------------------"
echo "$(date +"%D %T")"
echo "====================================================================================================="