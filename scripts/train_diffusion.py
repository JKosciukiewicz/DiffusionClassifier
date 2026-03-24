from lightning.pytorch.cli import LightningCLI
from datamodules.base_data_module import BaseDataModule
from lightning_models.base_model import BaseModel
from lightning_training.base_trainer import BaseTrainer

"""
CLI for model training
"""


def main():
    LightningCLI(
        model_class=BaseModel,
        datamodule_class=BaseDataModule,
        trainer_class=BaseTrainer,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
