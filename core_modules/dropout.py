import torch
import torch.nn as nn


def dropout(input, p=0.5, training=True, inplace=False):
    if training:
        mask = torch.empty_like(input)
        mask.bernoulli_(1 - p)
        if inplace:
            input.mul_(mask).div_(1 - p)
            return input
        else:
            return input * mask / (1 - p)
    else:
        return input


class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return dropout(input, self.p, self.training, self.inplace)
