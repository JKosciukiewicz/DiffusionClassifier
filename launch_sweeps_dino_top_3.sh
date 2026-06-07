#!/bin/bash
#SBATCH --job-name=SweepsDino3
#SBATCH --time=24:00:00
#SBATCH --account=plgwtln2-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu


source /net/pr2/projects/plgrid/plggwtln/jk/DiffusionClassifier/bin/activate
cd /net/people/plgrid/plgjkosciukiewi/DiffusionClassifier/

python scripts/train_sweep_plgrid_dino_top3.py
