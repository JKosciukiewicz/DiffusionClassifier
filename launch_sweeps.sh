#!/bin/bash
#SBATCH --job-name=Sweeps
#SBATCH --time=24:00:00
#SBATCH --account=plgwtln2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu


source /net/pr2/projects/plgrid/plggwtln/jk/DiffusionClassifier
cd /net/people/plgrid/plgjkosciukiewi/DiffusionClassifier/

python scripts/train_sweep_plgrid.py
