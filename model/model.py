import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ECGModel(BaseModel):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(1376, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ECGModel1(BaseModel):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv_blocks = nn.ModuleList([self._create_block(1 if i == 0 else 32, 32) for i in range(5)])
        self.max_pool = nn.MaxPool1d(kernel_size=5, stride=2)

        # Flatten and dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 5)  # Assuming 5 output classes
        self.softmax = nn.Softmax(dim=1)

    def _create_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        for block in self.conv_blocks:
            residual = x
            x = block(x)
            x += residual  # Residual connection
            x = F.relu(x)  # Activation function after addition
            x = self.max_pool(x)  # Max pooling after the block

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)


class ECGModel2(BaseModel):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv7 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv8 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv9 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv10 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.conv11 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)

        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x1 = self.leaky_relu(x1)
        x1 = self.conv3(x1)
        x = x1 + x
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x1 = self.leaky_relu(x)
        x1 = self.conv5(x1)
        x = x1 + x
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv6(x)
        x1 = self.leaky_relu(x)
        x1 = self.conv7(x1)
        x = x1 + x
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv8(x)
        x1 = self.leaky_relu(x)
        x1 = self.conv9(x1)
        x = x1 + x
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv10(x)
        x1 = self.leaky_relu(x)
        x1 = self.conv11(x1)
        x = x1 + x
        x = F.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
