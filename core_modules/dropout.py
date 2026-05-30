import torch
import torch.nn as nn


class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p, training, inplace):
        mask = None
        if training:
            out = input if inplace else input.clone()
            # Use boolean mask to save memory. Using rng state saves more but computing
            # RN for large tensor is computationally expensive and unclear trade-off.
            mask = torch.rand_like(input) > p
            out.mul_(mask).div_(1 - p)
        else:
            out = input

        if inplace:
            ctx.mark_dirty(input)
        if ctx.needs_input_grad[0]:
            ctx.p = p
            ctx.training = training
            ctx.save_for_backward(mask)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        if ctx.training:
            grad_input = grad_output * mask
            grad_input.div_(1 - ctx.p)
        else:
            grad_input = grad_output
        return grad_input, None, None, None


def dropout(input, p=0.5, training=True, inplace=False):
    return DropoutFunction.apply(input, p, training, inplace)


class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return dropout(input, self.p, self.training, self.inplace)


class Dropout1d(nn.Module):
    pass
