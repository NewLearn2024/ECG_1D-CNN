import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# 논문 재현
class ECGModel(BaseModel):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2) for _ in range(10)])
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)

        for i in range(0, len(self.conv_layers), 2):
            x1 = self.conv_layers[i](x)
            x1 = self.leaky_relu(x1)
            x1 = self.conv_layers[i + 1](x1)
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


# conv 2개 붙여서 만든 모델
class ECGModel2(BaseModel):
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
