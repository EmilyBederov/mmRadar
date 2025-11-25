#!/bin/bash
#SBATCH --job-name=transcribe
#SBATCH --output=transcribe_%j.out
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

# Activate your conda env (change name)
source ~/.bashrc
conda activate mmradar

# Show GPU
nvidia-smi

# Run your script
python ../src/utils/transcribe.py

echo "Finished at: $(date)"