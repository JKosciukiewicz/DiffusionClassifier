#!/bin/bash
#SBATCH --job-name=conf_bray_train
#SBATCH --time=24:00:00
#SBATCH --account=plgomenn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu

source $PLG_GROUPS_STORAGE/plggicv/jk/envs/hcs-dfc/.venv/bin/activate
export PYTHONPATH='/net/people/plgrid/plgjkosciukiewi/HCS-DFC'
python scripts/train_conformal.py fit -c configs/conformal_bray_plg.yaml