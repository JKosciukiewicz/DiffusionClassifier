#!/bin/bash

for alpha in $(seq 0.05 0.05 0.55); do
    echo "Running for alpha=${alpha}"
    python -m scripts.train_diffusion test \
        -c configs/diffusion_bbbc_cloome.yaml \
        --ckpt_path checkpoints/bbbc_cloome/diffusion/last.ckpt \
        --model.alpha $alpha \
        --model.results_path ./results/bbbc_cloome/diffusion/results_${alpha}.csv
done
