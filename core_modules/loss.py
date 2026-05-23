import torch
import torch.nn as nn


def mse_loss(input, target, reduction="mean", weight=None):
    diff = input - target
    out = diff * diff
    if weight is not None:
        out = out * weight
    if reduction == "none":
        return out
    elif reduction == "mean":
        return torch.mean(out)
    elif reduction == "sum":
        return torch.sum(out)
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return mse_loss(input, target, self.reduction)


def l1_loss(input, target, reduction="mean", weight=None):
    out = torch.abs(input - target)
    if weight is not None:
        out = out * weight
    if reduction == "none":
        return out
    elif reduction == "mean":
        return torch.mean(out)
    elif reduction == "sum":
        return torch.sum(out)
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class L1Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return l1_loss(input, target, self.reduction)
