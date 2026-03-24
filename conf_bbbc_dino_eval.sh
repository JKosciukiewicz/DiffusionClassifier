#!/bin/bash

for alpha in $(seq 0.05 0.05 1); do
    echo "Running for alpha=${alpha}"
    python -m scripts.train_conformal test \
        -c configs/conformal_bbbc_dino.yaml \
        --ckpt_path checkpoints/bbbc_dino_roc_train/conformal/epoch=29-validation_loss=0.21.ckpt \
        --model.alpha $alpha \
        --model.results_path ./results/bbbc_dino_train/conformal/ep30/results_${alpha}.csv
done
