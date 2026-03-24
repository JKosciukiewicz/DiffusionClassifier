import os
import subprocess
import numpy as np
import pandas as pd

config_path = "configs/diffusion_bbbc_dino_plg.yaml"
result_dir_path = "./results_diffusion_bbbc_dino"
os.makedirs(result_dir_path, exist_ok=True)
checkpoint_path_full = "/net/pr2/projects/plgrid/plggicv/jk/checkpoints_gridsearch_new/bbbc_dino/diffusion/sm/1e-4/last.ckpt"
# Run for each alpha value
for alpha in np.arange(0.05, 0.55, 0.05):  # Fixed range to include 1.0
    alpha = round(alpha, 2)  # Avoid floating point precision issues
    results_path = f"{result_dir_path}/results_alpha={alpha}.csv"

    if not os.path.isfile(results_path):
        cmd = [
            "python",
            "-m",
            "scripts.train_diffusion",
            "test",
            "-c",
            config_path,
            "--ckpt_path",
            checkpoint_path_full,
            "--model.alpha",
            str(alpha),
            "--model.size",
            "sm",
            "--model.results_path",
            results_path,
        ]
        print(f"Running: alpha={alpha}")
        print(f"Results will be saved to: {results_path}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Completed alpha={alpha}")
            if result.stdout:
                print("STDOUT:", result.stdout)
            # result_data = pd.read_csv(results_path)
            # if result_data["ROC_AUC"].item() == 0:
            #     print(f"Stopping loop early: ROC_AUC == 0 at alpha={alpha}")
            #     break
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed alpha={alpha}: {e}")
            print("STDERR:", e.stderr)
            continue
    else:
        print(f"Found results for {results_path}, skipping")

