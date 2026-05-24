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

        if any(ctx.needs_input_grad):
            ctx.reduction = reduction
            ctx.delta = delta
            ctx.has_weight = weight is not None
            # A tensor T is alive in memory at backward time iff any one of the following holds a reference:
            # 1. Python user scope — pred = model(x) binds pred as a local.
            # 2. An upstream op's save_for_backward — keeps T alive via its SavedVariable.
            # 3. Our own save_for_backward — the choice we're making.
            # If (1) or (2) already holds, then (3) is free. Otherwise (3) costs T.numel() * dtype_size
            #
            # target is typically saved by 1), thus saving input, target usually wins in
            # memory as if 1) or 2) happens to input (when people do pred = model(x)),
            # but saving diff is clearer in code and predictable in memory usage.
            if weight is None:
                ctx.save_for_backward(input, target)
            else:
                ctx.save_for_backward(input, target, weight)

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
            input, target, weight = ctx.saved_tensors
        else:
            input, target = ctx.saved_tensors
            weight = None
        # Use clamp instead of where to save ops more,
        # Option 1: grad_output * torch.where(abs_diff < ctx.delta, diff, torch.sign(diff) * ctx.delta) - 6 kernels
        # Option 2: grad_output * diff.clamp(-ctx.delta, ctx.delta) - 2 kernels as clamp is fused into 1.
        diff = input - target
        grad_input = grad_output * diff.clamp(-ctx.delta, ctx.delta)
        if weight is not None:
            grad_input = grad_input * weight
        if ctx.reduction == "mean":
            grad_input = grad_input / diff.numel()

        grad_target = -grad_input if ctx.needs_input_grad[1] else None
        return grad_input, grad_target, None, None, None


def huber_loss(input, target, reduction="mean", delta=1.0, weight=None):
    return HuberLossFunction.apply(input, target, reduction, delta, weight)


class HuberLoss(nn.Module):
    def __init__(self, reduction="mean", delta=1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, input, target):
        return huber_loss(input, target, self.reduction, self.delta)


def smooth_l1_loss(input, target, reduction="mean", beta=1.0):
    if beta == 0:
        return l1_loss(input, target, reduction)
    return huber_loss(input, target, reduction, beta) / beta


class SmoothL1Loss(nn.Module):
    def __init__(self, reduction="mean", beta=1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, input, target):
        return smooth_l1_loss(input, target, self.reduction, self.beta)


class BCELossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, weight, reduction):
        eps = max(torch.finfo(input.dtype).eps, 1e-100)
        out = -target * torch.log(input.clamp(min=eps)) - (1 - target) * torch.log(
            (1 - input).clamp(min=eps)
        )
        if weight is not None:
            out = out * weight

        if any(ctx.needs_input_grad):
            ctx.reduction = reduction
            ctx.has_weight = weight is not None
            if weight is None:
                ctx.save_for_backward(input, target)
            else:
                ctx.save_for_backward(input, target, weight)

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
            input, target, weight = ctx.saved_tensors
        else:
            input, target = ctx.saved_tensors
            weight = None

        eps = max(torch.finfo(input.dtype).eps, 1e-100)
        clamp_input = input.clamp(min=eps)
        clamp_1minput = (1 - input).clamp(min=eps)
        grad_input = (
            grad_output * (clamp_input - target) / (clamp_input * clamp_1minput)
        )
        grad_input = BCELossFunction._post_process(grad_input, weight, ctx.reduction)

        grad_target = None
        if ctx.needs_input_grad[1]:
            grad_target = grad_output * torch.log(clamp_1minput / clamp_input)
            grad_target = BCELossFunction._post_process(
                grad_target, weight, ctx.reduction
            )
        return grad_input, grad_target, None, None

    @staticmethod
    def _post_process(grad, weight, reduction):
        out = grad
        if weight is not None:
            out = out * weight
        if reduction == "mean":
            out = out / out.numel()
        return out


def binary_cross_entropy(input, target, weight=None, reduction="mean"):
    return BCELossFunction.apply(input, target, weight, reduction)


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return binary_cross_entropy(input, target, self.weight, self.reduction)
