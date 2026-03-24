import os
import subprocess
import numpy as np
import pandas as pd

modality_data = {
    "cp-model": "bray",
    "bray-model": "bray",
    "bbbc-model": "bbbc",
    "dino-model": "dino",
    "dino-small-model": "dino_small",
}
activation_data = {"relu": "torch.nn.ReLU", "gelu": "torch.nn.GELU"}
dataset_names = {
    "bray_cellprofiler_features": "bray",
    "bbbc_cellprofiler_features": "bbbc",
    "bray_dino_features": "bray_dino",
    "bray_cloome_features": "bray_cloome",
}

# datasets = [os.path.join('/Users/jkosciukiewicz/Developer/UJ/Projects/HCS-DFC/checkpoints/', ckpt_path) for ckpt_path in os.listdir('/Users/jkosciukiewicz/Developer/UJ/Projects/HCS-DFC/checkpoints') if ckpt_path not in ['bray_cellprofiler_features', 'bbbc_cellprofiler_features']]

# Process experiment information
for dataset in ["bbbc_cloome", "bbbc_dino"]:
    for modality in ["bray", "bbbc", "dino", "dino_small"]:
        for activation in ["torch.nn.ReLU", "torch.nn.GELU"]:
            for residual in ["True", "False"]:
                conformal_config_path = f"/Users/jkosciukiewicz/Developer/UJ/Projects/HCS-DFC/configs/conformal_{dataset}.yaml"
                diffusion_config_path = f"/Users/jkosciukiewicz/Developer/UJ/Projects/HCS-DFC/configs/diffusion_{dataset}.yaml"

                checkpoint_dir = f"/checkpoints_ablation/{dataset}/{modality}_{activation}_{residual}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                cmd = [
                    "python",
                    "-m",
                    "scripts.train_conformal",
                    "fit",
                    "-c",
                    conformal_config_path,
                    "--model.size",
                    modality,
                    "--model.activation_fn",
                    str(activation),
                    "--model.residual",
                    residual,
                    "--trainer.callbacks.dirpath",
                    f"{checkpoint_dir}/conformal",
                ]
                print(
                    f"Training conformal for {dataset}{modality}{activation}{residual}"
                )
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    if result.stdout:
                        print("STDOUT:", result.stdout)
                except subprocess.CalledProcessError as e:
                    print("STDERR:", e.stderr)
                    continue

                cmd = [
                    "python",
                    "-m",
                    "scripts.train_diffusion",
                    "fit",
                    "-c",
                    diffusion_config_path,
                    "--model.size",
                    modality,
                    "--model.activation_fn",
                    str(activation),
                    "--model.residual",
                    residual,
                    "--trainer.callbacks.dirpath",
                    f"{checkpoint_dir}/diffusion",
                ]
                print(
                    f"Training diffusion for {dataset}{modality}{activation}{residual}"
                )
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    if result.stdout:
                        print("STDOUT:", result.stdout)
                except subprocess.CalledProcessError as e:
                    print("STDERR:", e.stderr)
                    continue
