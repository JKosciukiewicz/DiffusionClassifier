import torch
import lightning as L
import wandb
from lightning.pytorch.cli import LightningCLI
from datamodules.base_data_module import BaseDataModule
from lightning_models.base_model import BaseModel
from lightning_training.base_trainer import BaseTrainer

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)


class CalibratedCLI(LightningCLI):
    def before_test(self):
        ckpt_path = getattr(self.config.test, "ckpt_path", None)
        if not ckpt_path:
            raise ValueError("Please specify --ckpt_path when running test.")

        print(f"Loading model from checkpoint: {ckpt_path}")
        self.model.load_state_dict(
            torch.load(ckpt_path, weights_only=True, map_location=device)["state_dict"]
        )
        datamodule = self.datamodule
        datamodule.setup(stage="test")
        self.model.eval()
        for batch in datamodule.calibration_dataloader():
            self.model.calibration_step(batch, 0)
        self.model.compute_thresholds()


def main():
    CalibratedCLI(
        model_class=BaseModel,
        datamodule_class=BaseDataModule,
        trainer_class=BaseTrainer,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
