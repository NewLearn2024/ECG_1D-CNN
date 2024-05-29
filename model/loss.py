import torch


def bce_loss(output, target):
    loss_fn = torch.nn.BCELoss()
    return loss_fn(output, target)
