#!/bin/bash

for alpha in $(seq 0.05 0.05 1); do
    echo "Running for alpha=${alpha}"
    python -m scripts.train_conformal test \
        -c configs/conformal_bbbc.yaml \
        --ckpt_path checkpoints/bbbc/conformal/last-v4.ckpt \
        --model.alpha $alpha \
        --model.results_path ./results/bray_dino_2/conformal/results_${alpha}.csv
done
