#!/bin/bash
#SBATCH --job-name=diff_bbbc_train
#SBATCH --time=24:00:00
#SBATCH --account=plgomenn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu

source $PLG_GROUPS_STORAGE/plggicv/jk/envs/hcs-dfc/.venv/bin/activate
export PYTHONPATH='/net/people/plgrid/plgjkosciukiewi/HCS-DFC'
python scripts/train_diffusion.py fit -c ./configs/diffusion_bbbc_plg.yaml