#!/bin/bash
#SBATCH --job-name=drought_train
#SBATCH --output=logs/hpc_%j.log
#SBATCH --error=logs/hpc_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ======================================================
# HPC SLURM script for Drought Detection Training
# College GPU cluster (NVIDIA GPU required)
#
# Usage:
#   sbatch hpc_train.sh
#   sbatch hpc_ablation.sh    # for sequence length ablation
# ======================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Start:  $(date)"
echo "============================================"

# ---- Load modules (adjust for your HPC module system) ----
# module load python/3.10
# module load cuda/12.8
# module load anaconda3

# ---- Activate your conda environment ----
# Option A: if you have a conda env named 'drought':
#   conda activate drought
# Option B: use system python with packages installed:
source activate drought 2>/dev/null || true

# ---- Go to project directory ----
cd "$SLURM_SUBMIT_DIR"

# ---- Verify GPU ----
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
nvidia-smi

# ---- Install dependencies if needed ----
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q
pip install xgboost scikit-learn pandas numpy scipy joblib -q

# ---- Run full training pipeline ----
echo ""
echo ">>> Starting full training pipeline..."
python src/training/train_all.py

echo ""
echo ">>> Training complete: $(date)"
echo ">>> Results in results/"
