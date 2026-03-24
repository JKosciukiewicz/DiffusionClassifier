import torch
import torch.nn as nn


# Variant 1: One additional large layer
class MLPLarge(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, dropout_rate: float = 0.3):
        """
        MLP Classifier with one additional layer (9 layers total).

        Args:
            num_classes: Number of classes to predict
            embedding_dim: Dimension of the input features
            dropout_rate: Dropout probability (default: 0.3)
        """
        super().__init__()
        # Define layers with an additional layer
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 512)
        self.fc4 = nn.Linear(512, 384)
        self.fc5 = nn.Linear(384, 256)
        self.fc6 = nn.Linear(256, 192)
        self.fc7 = nn.Linear(192, 128)
        self.fc8 = nn.Linear(128, 96)  # Additional layer
        self.fc9 = nn.Linear(96, num_classes)  # Output layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

        # Projection layers for residual connections
        self.res_proj1 = nn.Linear(embedding_dim, 1024)
        self.res_proj2 = nn.Linear(1024, 768)
        self.res_proj3 = nn.Linear(768, 512)
        self.res_proj4 = nn.Linear(512, 384)
        self.res_proj5 = nn.Linear(384, 256)
        self.res_proj6 = nn.Linear(256, 192)
        self.res_proj7 = nn.Linear(192, 128)
        self.res_proj8 = nn.Linear(128, 96)  # New projection

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP with residual connections.

        Args:
            features: Input feature tensor of shape [batch_size, embedding_dim]

        Returns:
            Output probability tensor of shape [batch_size, num_classes]
        """
        # First layer with residual connection
        x = self.fc1(features)
        residual = self.res_proj1(features)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Second layer with residual connection
        residual = self.res_proj2(x)
        x = self.fc2(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Third layer with residual connection
        residual = self.res_proj3(x)
        x = self.fc3(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Fourth layer with residual connection
        residual = self.res_proj4(x)
        x = self.fc4(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Fifth layer with residual connection
        residual = self.res_proj5(x)
        x = self.fc5(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Sixth layer with residual connection
        residual = self.res_proj6(x)
        x = self.fc6(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Seventh layer with residual connection
        residual = self.res_proj7(x)
        x = self.fc7(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Eighth layer with residual connection (additional)
        residual = self.res_proj8(x)
        x = self.fc8(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Final layer
        x = self.fc9(x)
        x = self.sigmoid(x)

        return x


# Variant 2: One less layer
class MLPMedium(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, dropout_rate: float = 0.3):
        """
        MLP Classifier with one less layer (7 layers total).

        Args:
            num_classes: Number of classes to predict
            embedding_dim: Dimension of the input features
            dropout_rate: Dropout probability (default: 0.3)
        """
        super().__init__()
        # Define layers with one less layer
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 512)
        self.fc4 = nn.Linear(512, 384)
        self.fc5 = nn.Linear(384, 256)
        self.fc6 = nn.Linear(256, 128)  # Skip 192 size
        self.fc7 = nn.Linear(128, num_classes)  # Output layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

        # Projection layers for residual connections
        self.res_proj1 = nn.Linear(embedding_dim, 1024)
        self.res_proj2 = nn.Linear(1024, 768)
        self.res_proj3 = nn.Linear(768, 512)
        self.res_proj4 = nn.Linear(512, 384)
        self.res_proj5 = nn.Linear(384, 256)
        self.res_proj6 = nn.Linear(256, 128)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP with residual connections.

        Args:
            features: Input feature tensor of shape [batch_size, embedding_dim]

        Returns:
            Output probability tensor of shape [batch_size, num_classes]
        """
        # First layer with residual connection
        x = self.fc1(features)
        residual = self.res_proj1(features)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Second layer with residual connection
        residual = self.res_proj2(x)
        x = self.fc2(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Third layer with residual connection
        residual = self.res_proj3(x)
        x = self.fc3(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Fourth layer with residual connection
        residual = self.res_proj4(x)
        x = self.fc4(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Fifth layer with residual connection
        residual = self.res_proj5(x)
        x = self.fc5(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Sixth layer with residual connection
        residual = self.res_proj6(x)
        x = self.fc6(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Final layer
        x = self.fc7(x)
        x = self.sigmoid(x)

        return x


# Variant 3: Two less layers
class MLPSmall(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, dropout_rate: float = 0.3):
        """
        MLP Classifier with two less layers (6 layers total).

        Args:
            num_classes: Number of classes to predict
            embedding_dim: Dimension of the input features
            dropout_rate: Dropout probability (default: 0.3)
        """
        super().__init__()
        # Define layers with two less layers
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 512)
        self.fc4 = nn.Linear(512, 256)  # Skip 384 size
        self.fc5 = nn.Linear(256, 128)  # Skip 192 size
        self.fc6 = nn.Linear(128, num_classes)  # Output layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

        # Projection layers for residual connections
        self.res_proj1 = nn.Linear(embedding_dim, 1024)
        self.res_proj2 = nn.Linear(1024, 768)
        self.res_proj3 = nn.Linear(768, 512)
        self.res_proj4 = nn.Linear(512, 256)
        self.res_proj5 = nn.Linear(256, 128)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP with residual connections.

        Args:
            features: Input feature tensor of shape [batch_size, embedding_dim]

        Returns:
            Output probability tensor of shape [batch_size, num_classes]
        """
        # First layer with residual connection
        x = self.fc1(features)
        residual = self.res_proj1(features)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Second layer with residual connection
        residual = self.res_proj2(x)
        x = self.fc2(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Third layer with residual connection
        residual = self.res_proj3(x)
        x = self.fc3(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Fourth layer with residual connection
        residual = self.res_proj4(x)
        x = self.fc4(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Fifth layer with residual connection
        residual = self.res_proj5(x)
        x = self.fc5(x)
        x = self.relu(x + residual)
        x = self.dropout(x)

        # Final layer
        x = self.fc6(x)
        x = self.sigmoid(x)

        return x
