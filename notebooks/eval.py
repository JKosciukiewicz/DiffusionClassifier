import os
import subprocess
import numpy as np
import pandas as pd

# # Process experiment information
# for dataset in datasets:
#     conformal_model_sizes = os.path.join(checkpoint_path, dataset, 'conformal')
#     #diffusion_model_sizes = os.path.join(dataset, 'diffusion')
#
#     for conformal_model in [size for size in os.listdir(conformal_model_sizes) if size not in ['lg', 'md']]:
#         conformal_model_path = os.path.join(conformal_model_sizes, conformal_model)
#         for checkpoint_path in os.listdir(conformal_model_path):
#             if "epoch=4" in checkpoint_path:
#                 checkpoint_path_full = os.path.join(conformal_model_path, checkpoint_path)
#                 checkpoint_epoch = checkpoint_path.split('-val_loss')[0].split('epoch=epoch=')[1]
#                 if dataset != "bray_cp":
#                     config_path = os.path.join("/Users/jkosciukiewicz/Developer/UJ/Projects/HCS-DFC/configs", f"conformal_{dataset}.yaml")
#                 else:
#                     config_path = "/Users/jkosciukiewicz/Developer/UJ/Projects/HCS-DFC/configs/conformal_bray.yaml"

config_path = "configs/conformal_bray_plg.yaml"
result_dir_path = "./results_conformal_cp"
os.makedirs(result_dir_path, exist_ok=True)
checkpoint_path_full = "/net/pr2/projects/plgrid/plggicv/jk/checkpoints_gridsearch_new/bray/mlp/md/1e-3/epoch=epoch=49-val_roc_auc=validation_roc_auc=0.68.ckpt"
# Run for each alpha value
for alpha in np.arange(0.05, 1.05, 0.05):  # Fixed range to include 1.0
    alpha = round(alpha, 2)  # Avoid floating point precision issues
    results_path = f"{result_dir_path}/results_alpha={alpha}.csv"

    if not os.path.isfile(results_path):
        cmd = [
            "python",
            "-m",
            "scripts.train_conformal",
            "test",
            "-c",
            config_path,
            "--ckpt_path",
            checkpoint_path_full,
            "--model.alpha",
            str(alpha),
            "--model.size",
            "md",
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
            result_data = pd.read_csv(results_path)
            if result_data["ROC_AUC"].item() == 0:
                print(f"Stopping loop early: ROC_AUC == 0 at alpha={alpha}")
                break
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed alpha={alpha}: {e}")
            print("STDERR:", e.stderr)
            continue
    else:
        print(f"Found results for {results_path}, skipping")

        # for alpha in np.arange(0.05, 0.55, 0.05):  # Fixed range to include 1.0
        #     alpha = round(alpha, 2)  # Avoid floating point precision issues
        #     results_path = f"{ diffusion_results_dir}/results_{alpha}.csv"
        #     if not os.path.isfile(results_path):
        #         cmd = [
        #             "python", "-m", "scripts.train_diffusion", "test",
        #             "-c", diffusion_config_path,
        #             "--ckpt_path", diffusion_checkpoint_path,
        #             "--model.alpha", str(alpha),
        #             "--model.activation_fn", str(activation),
        #             "--model.size", modality,
        #             "--model.residual", str(residual_bool),
        #             "--model.results_path", results_path
        #         ]
        #
        #         print(f"Running: alpha={alpha}, size={modality}, residual={residual_bool}")
        #         print(f"Results will be saved to: {results_path}")
        #
        #         try:
        #             result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        #             print(f"✓ Completed alpha={alpha}")
        #             if result.stdout:
        #                 print("STDOUT:", result.stdout)
        #                 result_data = pd.read_csv(results_path)
        #             if result_data['ROC_AUC'].item() == 0:
        #                 print(f"Stopping loop early: ROC_AUC == 0 at alpha={alpha}")
        #                 break
        #         except subprocess.CalledProcessError as e:
        #             print(f"✗ Failed alpha={alpha}: {e}")
        #             print("STDERR:", e.stderr)
        #             continue
        #
        #     else:
        #         print(f"Found results for {results_path}, skipping")
