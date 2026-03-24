#!/bin/bash
#SBATCH --job-name=grid-search-mlp-cloome
#SBATCH --time=24:00:00
#SBATCH --account=plgomenn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu

source $PLG_GROUPS_STORAGE/plggicv/jk/envs/hcs-dfc/.venv/bin/activate
export PYTHONPATH='/net/people/plgrid/plgjkosciukiewi/HCS-DFC'
python notebooks/grid_search_cloome.py