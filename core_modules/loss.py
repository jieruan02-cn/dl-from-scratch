import torch
import torch.nn as nn
from activation import logsigmoid, sigmoid, log_softmax, softmax


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
        input, target, weight = ctx.saved_tensors
        # Use clamp instead of where to save ops more,
        # Option 1: grad_output * torch.where(abs_diff < ctx.delta, diff, torch.sign(diff) * ctx.delta) - 6 kernels
        # Option 2: grad_output * diff.clamp(-ctx.delta, ctx.delta) - 2 kernels as clamp is fused into 1.
        diff = input - target
        grad_input = grad_output * diff.clamp(-ctx.delta, ctx.delta)
        if weight is not None:
            grad_input.mul_(weight)
        if ctx.reduction == "mean":
            grad_input.div_(diff.numel())

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


def _post_process(grad, weight, reduction):
    if grad is None:
        return grad
    if weight is not None:
        grad.mul_(weight)
    if reduction == "mean":
        grad.div_(grad.numel())
    return grad


class BCELossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, weight, reduction):
        eps = max(torch.finfo(input.dtype).tiny, 1e-100)
        out = -target * torch.log(input.clamp(min=eps)) - (1 - target) * torch.log(
            (1 - input).clamp(min=eps)
        )
        if weight is not None:
            out = out * weight

        if any(ctx.needs_input_grad):
            ctx.reduction = reduction
            ctx.has_weight = weight is not None
            ctx.save_for_backward(input, target)

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
        input, target, weight = ctx.saved_tensors

        # 2. Clamp/gradient inconsistency at the boundary (subtle, possibly intentional).
        eps = max(torch.finfo(input.dtype).tiny, 1e-100)
        clamp_input = input.clamp(min=eps)
        clamp_1minput = (1 - input).clamp(min=eps)
        grad_input = (
            grad_output * (clamp_input - target) / (clamp_input * clamp_1minput)
        )
        grad_input = _post_process(grad_input, weight, ctx.reduction)

        grad_target = None
        if ctx.needs_input_grad[1]:
            grad_target = grad_output * torch.log(clamp_1minput / clamp_input)
            grad_target = _post_process(grad_target, weight, ctx.reduction)
        return grad_input, grad_target, None, None


def binary_cross_entropy(input, target, weight=None, reduction="mean"):
    return BCELossFunction.apply(input, target, weight, reduction)


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return binary_cross_entropy(input, target, self.weight, self.reduction)


class BCEWithLogitsLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, weight, reduction, pos_weight):
        out = input * (1 - target)
        log_sig = logsigmoid(input)
        if pos_weight is None:
            out = out - log_sig
        else:
            out = out - (1 - target + pos_weight * target) * log_sig
        if weight is not None:
            out = out * weight

        if any(ctx.needs_input_grad):
            ctx.reduction = reduction
            ctx.save_for_backward(input, target, weight, pos_weight)

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
        input, target, weight, pos_weight = ctx.saved_tensors

        if pos_weight is None:
            grad_input = grad_output * (sigmoid(input) - target)
            grad_target = -grad_output * input if ctx.needs_input_grad[1] else None
        else:
            scale = 1 - target + pos_weight * target
            grad_input = grad_output * (scale * sigmoid(input) - pos_weight * target)
            grad_target = (
                grad_output * ((1 - pos_weight) * logsigmoid(input) - input)
                if ctx.needs_input_grad[1]
                else None
            )
        grad_input = _post_process(grad_input, weight, ctx.reduction)
        grad_target = _post_process(grad_target, weight, ctx.reduction)
        return grad_input, grad_target, None, None, None


def binary_cross_entropy_with_logits(
    input, target, weight=None, reduction="mean", pos_weight=None
):
    return BCEWithLogitsLossFunction.apply(input, target, weight, reduction, pos_weight)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, reduction="mean", pos_weight=None):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input, target):
        return binary_cross_entropy_with_logits(
            input, target, self.weight, self.reduction, self.pos_weight
        )


class NLLLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, weight, ignore_index, reduction):
        dim = 0 if input.dim() == 1 else 1
        clamped_target = target.clamp(0, input.shape[dim] - 1)
        out = -input.gather(dim, clamped_target.unsqueeze(dim)).squeeze(dim)
        target_weight = (target != ignore_index).to(input.dtype)
        if weight is not None:
            target_weight = target_weight * weight[clamped_target]
        out = out * target_weight

        if ctx.needs_input_grad[0]:
            ctx.dim = dim
            ctx.input_shape = input.shape
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.num_classes = input.shape[dim]
            ctx.save_for_backward(target, weight)

        if reduction == "none":
            return out
        elif reduction == "mean":
            return torch.sum(out) / torch.sum(target_weight)
        elif reduction == "sum":
            return torch.sum(out)
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def backward(ctx, grad_output):
        target, weight = ctx.saved_tensors
        clamped_target = target.clamp(0, ctx.num_classes - 1)
        target_weight = (target != ctx.ignore_index).to(grad_output.dtype)
        if weight is not None:
            target_weight = target_weight * weight[clamped_target]
        grad_input = torch.zeros(
            ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        grad_input.scatter_(
            ctx.dim,
            index=clamped_target.unsqueeze(ctx.dim),
            src=-(grad_output * target_weight).unsqueeze(ctx.dim),
        )
        if ctx.reduction == "mean":
            grad_input.div_(torch.sum(target_weight))
        return grad_input, None, None, None, None


def nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean"):
    return NLLLossFunction.apply(input, target, weight, ignore_index, reduction)


class NLLLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        return nll_loss(input, target, self.weight, self.ignore_index, self.reduction)


def _smooth_target(input, target, dim, label_smoothing):
    if target.dim() == input.dim():
        return target
    scale = 1.0 / input.shape[dim]
    smoothed_target = torch.full_like(input, label_smoothing * scale)
    target_view = target.unsqueeze(dim)
    src = torch.full(
        target_view.shape,
        1.0 - (1.0 - scale) * label_smoothing,
        device=input.device,
        dtype=input.dtype,
    )
    smoothed_target.scatter_(dim=dim, index=target_view, src=src)
    return smoothed_target


class CrossEntropyFunction(torch.autograd.Function):
    # Because torch.autograd.Function's forward run in torch.no_grad, while normal torch
    # module's forward doesn't. We choose to compute smoothed target inside Function's
    # forward for efficiency.
    @staticmethod
    def forward(ctx, input, target, weight, reduction, label_smoothing):
        dim = 0 if input.dim() == 1 else 1
        smoothed_target = _smooth_target(input, target, dim, label_smoothing)
        out = -log_softmax(input, dim) * smoothed_target

        weight_shape, num_classes = None, input.shape[dim]
        weight_sum = input.numel() // num_classes
        if weight is not None:
            weight_shape = [1] * input.dim()
            weight_shape[dim] = num_classes
            weight_shape = tuple(weight_shape)
            out.mul_(weight.reshape(weight_shape))
            if target.dim() < input.dim():
                weight_sum = torch.sum(weight[target])

        if ctx.needs_input_grad[0]:
            ctx.dim = dim
            ctx.reduction = reduction
            ctx.weight_sum = weight_sum
            ctx.label_smoothing = label_smoothing
            ctx.weight_shape = weight_shape
            weight_view = weight if weight is None else weight.reshape(ctx.weight_shape)
            ctx.save_for_backward(input, target, weight_view)

        if reduction == "none":
            return torch.sum(out, dim)
        elif reduction == "mean":
            return torch.sum(out) / weight_sum
        elif reduction == "sum":
            return torch.sum(out)
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reduction == "none":
            grad_output = grad_output.unsqueeze(ctx.dim)

        input, target, weight = ctx.saved_tensors
        smoothed_target = _smooth_target(input, target, ctx.dim, ctx.label_smoothing)
        if weight is not None:
            y_dot_w = torch.sum(smoothed_target * weight, ctx.dim, keepdim=True)
            grad_input = softmax(input, ctx.dim) * y_dot_w - smoothed_target * weight
        else:
            grad_input = softmax(input, ctx.dim) - smoothed_target
        if ctx.reduction == "mean":
            grad_input.div_(ctx.weight_sum)
        grad_input.mul_(grad_output)

        grad_target = None
        if ctx.needs_input_grad[1]:
            grad_target = -grad_output * log_softmax(input, ctx.dim)
            if weight is not None:
                grad_target.mul_(weight.reshape(ctx.weight_shape))
            if ctx.reduction == "mean":
                grad_target.div_(ctx.weight_sum)
        return grad_input, grad_target, None, None, None


def cross_entropy(
    input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0
):
    if target.dim() < input.dim() and label_smoothing == 0.0:
        dim = 0 if input.dim() == 1 else 1
        return nll_loss(
            log_softmax(input, dim), target, weight, ignore_index, reduction
        )
    return CrossEntropyFunction.apply(input, target, weight, reduction, label_smoothing)


class CrossEntropyLoss(nn.Module):
    def __init__(
        self, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return cross_entropy(
            input,
            target,
            self.weight,
            self.ignore_index,
            self.reduction,
            self.label_smoothing,
        )
