import lightning as L


class BaseTrainer(L.Trainer):
    """Marker base class for LightningCLI to identify trainer subclasses."""

    pass
