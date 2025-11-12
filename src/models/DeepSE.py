import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSE(nn.Module):
    def __init__(self):
        super(DeepSE, self).__init__()

        # Layer 1: Conv1D -> MaxPooling1D -> Conv1D -> MaxPooling1D -> Dropout
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        # Dropout after pooling layer
        self.dropout1 = nn.Dropout(0.5)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Dense layer with regularization (L2)
        self.dense1 = nn.Linear(1536, 640)  # This is based on input size (300) and kernel sizes in the Keras model
        self.dropout2 = nn.Dropout(0.25)

        # Output layer
        self.output = nn.Linear(640, 1)

        # L2 Regularization can be added during optimizer setup

    def forward(self, x):
        # Convolution -> Max Pooling
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Dropout after pooling
        x = self.dropout1(x)

        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)

        # Output layer
        x = torch.sigmoid(self.output(x))
        return x


# # Example of creating the model and printing summary
# model = CNNModel()
# print(model)
