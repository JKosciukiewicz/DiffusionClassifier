import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Conv block with a skip connection. Handles channel mismatches via a 1x1 projection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        # Project the skip connection if channel dims differ
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))


class CNNMultiLabel(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=128):
        super().__init__()

        self.conv1 = ResidualBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        # FC layers with a residual connection (dims match at 128)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.fc_skip = (
            nn.Linear(128, embedding_dim) if 128 != embedding_dim else nn.Identity()
        )
        self.relu = nn.ReLU()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.BatchNorm1d(int(embedding_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim / 2), num_classes),
            nn.Sigmoid(),
        )

    def extract_features(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)

        # Residual FC block
        h = self.relu(self.fc1(x))
        return self.relu(self.fc2(h) + self.fc_skip(h))

    def forward(self, x):
        return self.classifier(self.extract_features(x))
