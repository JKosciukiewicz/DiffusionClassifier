import subprocess

learning_rates = ["1e-5"]
model_size = ["xsm", "sm", "md", "lg", "xlg"]

config_path = "configs/conformal_bbbc_plg.yaml"
for lr in learning_rates:
    for model in model_size:
        checkpoint_path = f"/net/pr2/projects/plgrid/plggicv/jk/checkpoints_gridsearch_new/checkpoints_gridsearch_new/bbbc_cp/mlp/{model}/{lr}"
        run_name = f"gridsearch-bbbc_cp-{model}-{lr}"
        cmd = [
            "python",
            "-m",
            "scripts.train_conformal",
            "fit",
            "-c",
            config_path,
            "--trainer.callbacks.dirpath",
            checkpoint_path,
            "--trainer.logger.name",
            run_name,
            "--trainer.logger.version",
            run_name,
            "--model.size",
            model,
            "--model.lr",
            lr,
        ]

        print(f"Running: lr={lr}, size={model}, checkpoint={checkpoint_path}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Completed  lr={lr}, size={model}")
            if result.stdout:
                print("STDOUT:", result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"✗ Failed  lr={lr}, size={model}: {e}")
            print("STDERR:", e.stderr)
            continue
