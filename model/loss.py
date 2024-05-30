import torch.nn as nn


def ce_loss(output, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)
