import torch.nn as nn
import torch


class DiffusionMLP(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=128):
        super().__init__()
        # MLP for class prediction
        # Define the layers for the MLP with residual connections
        self.fc1 = nn.Linear(embedding_dim + num_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Projection layers for residual connections
        self.res_proj1 = nn.Linear(embedding_dim + num_classes, 256)
        self.res_proj2 = nn.Linear(256, 128)
        self.res_proj3 = nn.Linear(128, 64)
        self.res_proj4 = nn.Linear(64, 32)
        self.res_proj5 = nn.Linear(32, 16)

    def forward(self, features, labels):
        """
        Args:
            features: Noisy input features used as conditioning, shape [batch_size, embedding_dim]
            labels: Noisy input labels [batch_size, num_classes]
        Returns:
            Predicted noise per class, shape [batch_size, num_classes]
        """
        # Concatenate features and labels
        class_condition = torch.cat((features, labels), dim=1)
        print(class_condition.shape)
        # First layer with residual connection
        x = self.fc1(class_condition)
        residual = self.res_proj1(class_condition)
        x = self.relu(x + residual)

        # Second layer with residual connection
        residual = self.res_proj2(x)
        x = self.fc2(x)
        x = self.relu(x + residual)

        # Third layer with residual connection
        residual = self.res_proj3(x)
        x = self.fc3(x)
        x = self.relu(x + residual)

        # Fourth layer with residual connection
        residual = self.res_proj4(x)
        x = self.fc4(x)
        x = self.relu(x + residual)

        # Fifth layer with residual connection
        residual = self.res_proj5(x)
        x = self.fc5(x)
        x = self.relu(x + residual)

        # Final layer (no residual connection here)
        x = self.fc6(x)
        x = self.sigmoid(x)

        return x
