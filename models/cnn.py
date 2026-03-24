from torch import nn


class CNNMultiLabel(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=128):
        super().__init__()
        # CNN for feature extraction and Diffusion conditioning
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, padding=1
            ),  # Input: 1 channel, Output: 32 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces spatial size by half
            nn.Conv2d(
                32, 64, kernel_size=3, padding=1
            ),  # Input: 32 channels, Output: 64 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces spatial size by half again
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # 64 feature maps of size 7x7 (for 28x28 input)
            nn.ReLU(),
            nn.Linear(128, embedding_dim),  # Output layer with embedding_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.Sigmoid(),  # Outputs independent probabilities for each class
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def extract_features(self, x):
        return self.feature_extractor(x)
