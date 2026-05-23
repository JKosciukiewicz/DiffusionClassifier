from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datamodules.bray_data_module import BrayDataModule
from datamodules.bray_dino_data_module import BrayDinoDataModule

# Configuration for Bray DINO
data_file = "_data/bray_dino/bray_dino_complete.csv"
split_col = "hier_split_0"  # Can be hier_split_0 to hier_split_4

bray_datamodule = BrayDinoDataModule(
    batch_size=1024,
    data_file=data_file,
    split_col=split_col,
    mask_uncertain=True,
)
bray_datamodule.setup("fit")

# Use a standard PyTorch DataLoader to iterate over the dataset
train_dataset = bray_datamodule.train_dataset
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 2. Corrected Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Encoder: 4558 -> 1024 -> 256 -> 64
        self.encoder = nn.Sequential(
            nn.Linear(384, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        # Decoder: 64 -> 256 -> 1024 -> 4558 (Mirrors the encoder)
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 384),
            # Note: Add an activation layer here (e.g., nn.Sigmoid()) if
            # your input data features are normalized between 0 and 1.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# 3. Initialization & Hardware Setup
device = torch.device("mps")
model = Autoencoder().to(device)

# Using Mean Squared Error (MSE) to minimize reconstruction loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 4. Pure PyTorch Training Loop
num_epochs = 20

print(f"Starting training on device: {device}")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Depending on your dataset structure, batch might be a tuple (x, y) or just x.
    # Assuming standard supervised/unsupervised layout where batch[0] is the feature tensor.
    for batch_idx, batch in enumerate(train_loader):
        # Handle tuple unpack if dataset returns targets, otherwise keep raw batch
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device).float()

        # Zero out the parameter gradients
        optimizer.zero_grad()

        # Forward pass: compute reconstruction
        x_hat = model(x)

        # Calculate loss (difference between original x and reconstructed x_hat)
        loss = criterion(x_hat, x)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

print("Training finished!")
