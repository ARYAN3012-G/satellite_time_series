#!/bin/bash
#SBATCH --job-name=drought_all_lengths
#SBATCH --output=logs/hpc_%j.log
#SBATCH --error=logs/hpc_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ======================================================
# HPC SLURM script -- Drought Detection
# Trains ALL 4 models for 30d, 60d, 90d windows
# Finds the overall best (model + window) combination
#
# Usage:
#   sbatch hpc_train.sh
# ======================================================

echo "============================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "GPUs    : $CUDA_VISIBLE_DEVICES"
echo "Start   : $(date)"
echo "============================================"

# ---- Load modules (adjust for YOUR HPC) ----
# Run "module avail" on your HPC to see exact names
module load anaconda3   2>/dev/null || true
module load cuda/12.8   2>/dev/null || true

# ---- Activate conda environment ----
source activate drought 2>/dev/null || conda activate drought 2>/dev/null || true

# ---- Go to project directory ----
cd "$SLURM_SUBMIT_DIR"

# ---- Install dependencies if not already installed ----
python -c "import torch" 2>/dev/null || {
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q
}
python -c "import xgboost" 2>/dev/null || {
    echo "Installing remaining packages..."
    pip install xgboost scikit-learn pandas numpy scipy joblib -q
}

# ---- Verify GPU ----
echo ""
echo ">>> GPU Check:"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ---- Run training for ALL sequence lengths (30d, 60d, 90d) ----
echo ""
echo ">>> Starting training for 30d, 60d, 90d windows..."
python src/training/run_all_lengths.py

echo ""
echo "============================================"
echo "DONE: $(date)"
echo "Results saved in results/"
echo "  results/all_lengths_comparison.csv"
echo "  results/best_overall_summary.json"
echo "  results/30d/  results/60d/  results/90d/"
echo "============================================"
