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


# Use customized backward as default backward() will take multiple unnecessary derivative
# on indicator function and increase number of ops.
class HuberLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, reduction="mean", delta=1.0, weight=None):
        diff = input - target
        abs_diff = torch.abs(diff)
        out = torch.where(
            abs_diff < delta,
            0.5 * abs_diff * abs_diff,
            delta * (abs_diff - 0.5 * delta),
        )
        if weight is not None:
            out = out * weight

        if ctx.needs_input_grad[0]:
            ctx.reduction = reduction
            ctx.delta = delta
            ctx.has_weight = weight is not None
            # save diff instead of input, target compared to activation unit as residual
            # often makes saving input, target saving overhead ~0, but in loss, saving
            # diff is cheaper.
            if weight is None:
                ctx.save_for_backward(diff)
            else:
                ctx.save_for_backward(diff, weight)

        if reduction == "none":
            return out
        elif reduction == "mean":
            return torch.mean(out)
        elif reduction == "sum":
            return torch.sum(out)
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_weight:
            diff, weight = ctx.saved_tensors
        else:
            (diff,) = ctx.saved_tensors
            weight = None
        abs_diff = torch.abs(diff)
        grad_input = grad_output * torch.where(
            abs_diff < ctx.delta, diff, torch.sign(diff) * ctx.delta
        )
        if weight is not None:
            grad_input = grad_input * weight
        if ctx.reduction == "mean":
            grad_input = grad_input / diff.numel()
        return grad_input, None, None, None, None


def huber_loss(input, target, reduction="mean", delta=1.0, weight=None):
    return HuberLossFunction.apply(input, target, reduction, delta, weight)


class HuberLoss(nn.Module):
    def __init__(self, reduction="mean", delta=1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, input, target):
        return huber_loss(input, target, self.reduction, self.delta)
