#!/bin/bash
#SBATCH --job-name=drought_ablation
#SBATCH --output=logs/ablation_%j.log
#SBATCH --error=logs/ablation_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

echo "Starting sequence length ablation: $(date)"
cd "$SLURM_SUBMIT_DIR"
source activate drought 2>/dev/null || true
python src/training/ablation_sequence_length.py
echo "Ablation complete: $(date)"
