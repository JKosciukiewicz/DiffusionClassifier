import lightning as L


class BaseDataModule(L.LightningDataModule):
    """Marker base class for LightningCLI to identify datamodule subclasses."""

    pass
