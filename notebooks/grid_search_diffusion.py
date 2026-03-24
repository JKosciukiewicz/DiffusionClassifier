import subprocess

learning_rates = ["1e-3"]
model_size = ["md"]

config_path = "configs/diffusion_bray_plg.yaml"
for lr in learning_rates:
    for model in model_size:
        checkpoint_path = f"./diff_god_please_mse/{model}/{lr}"
        run_name = f"gridsearch-diff_god_please_mse_fix-{model}-{lr}"
        cmd = [
            "python",
            "-m",
            "scripts.train_diffusion",
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
