import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ECGModel(BaseModel):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(32 * 87, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
