#!/bin/bash

for alpha in $(seq 0.05 0.05 0.5); do
    echo "Running for alpha=${alpha}"
    python -m scripts.train_diffusion test \
        -c configs/diffusion_bray_cloome.yaml \
        --ckpt_path checkpoints/bray_cloome/diffusion/last.ckpt \
        --model.alpha $alpha \
        --model.results_path ./results/bray_cloome/diffusion/results_${alpha}.csv
done
