import lightning as L

# models/base.py
import pytorch_lightning as pl


class BaseModel(L.LightningModule):
    """Marker base class for LightningCLI to identify model subclasses."""

    pass
