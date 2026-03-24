#!/bin/bash
#SBATCH --job-name=eval_diffusion_bray
#SBATCH --time=24:00:00
#SBATCH --account=plgomenn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu

source $PLG_GROUPS_STORAGE/plggicv/jk/envs/hcs-dfc/.venv/bin/activate
export PYTHONPATH='/net/people/plgrid/plgjkosciukiewi/HCS-DFC'
python notebooks/eval_diffusion.py