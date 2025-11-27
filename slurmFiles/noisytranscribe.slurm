#!/bin/bash
#SBATCH --job-name=transcribe
#SBATCH --output=transcribe_%j.out
#SBATCH --partition=debug          # no * here
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

# Load shell config so 'conda' is available
source ~/.bashrc

# Activate your conda env
conda activate mmradar

# Show GPU status
nvidia-smi

# Run your script (relative to slurmFiles directory)
python ../noisy_transcribe.py

echo "Finished at: $(date)"