#!/bin/bash

for alpha in $(seq 0.05 0.05 1); do
    echo "Running for alpha=${alpha}"
    python -m scripts.train_conformal test \
        -c configs/conformal_bray_dino.yaml \
        --ckpt_path checkpoints/bray_dino/conformal/last-v5.ckpt \
        --model.alpha $alpha \
        --model.results_path ./results/bray_dino/conformal_new/results_${alpha}.csv
done
