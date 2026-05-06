import torch
import torch.nn as nn
import warnings


# Lessons:
# 1. In-place op on a leaf tensor with requires_grad=True (e.g. nn.Parameter, or any user-created tensor with
#    requires_grad=True) raises: "a leaf Variable that requires grad is being used in an in-place operation."
#    Two reasons:
#      1) Graph correctness: backward needs the original input values (ReLU's backward needs to know which entries
#         were <= 0). In-place overwrites them, so the saved tensor on the autograd tape would be corrupted.
#      2) Leaf identity: optimizer.step() updates params via no_grad in-place writes and assumes params stay leaves.
#         Mutating a leaf inside forward bumps its _version mid-graph and breaks that assumption.
# 2. In-place is safe on non-leaf tensors (outputs of previous ops) when no downstream backward needs the
#    pre-mutation values. ReLU qualifies: its backward only needs output > 0, which the in-place result still gives.
#    Counter-example: log_'s backward needs the input value, so log_ on a tensor reused later is unsafe.
# 3. PyTorch tracks each tensor's _version; if a saved tensor's version changes before .backward(), it raises then.
def relu(input, inplace=False):
    return input.clamp_(min=0) if inplace else input.clamp(min=0)


def relu_(input):
    return input.clamp_(min=0)


class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return relu(x, self.inplace)


# Customized backward is used to avoid numerical overflow. When x is very small x (negative), regular autograd will
# compute exp(-x) / (1 + exp(-x))^2, leading to overflow. using out * (1 - out) with out in [0, 1] avoids this.
class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return 1 / (1 + torch.exp(-x))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out)


def sigmoid(input):
    return SigmoidFunction.apply(input)


class Sigmoid(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return sigmoid(input)


# TanhFunction is mainly for optimal efficiency for tanh, otherwise reusing sigmoid for tanh is better practice:
# tanh(x) = 2sigmoid(2x) - 1
class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(x):
        # (exp(x) - exp(-x))/(exp(x) + exp(-x)) is worse numerically as any of the inf will cause issue, here inf in
        # denominator will result 0.
        return 2 / (1 + torch.exp(-2 * x)) - 1

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (1 - out**2)


def tanh(input):
    return TanhFunction.apply(input)


class Tanh(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return tanh(input)


def _default_softmax_dim(input):
    return 0 if input.dim() in (0, 1, 3) else 1


def _resolve_dim(input, dim, _stacklevel):
    if dim is None:
        warnings.warn(
            "Implicit dimension choice for softmax has been deprecated. "
            "Change the call to include dim=X as an argument.",
            stacklevel=_stacklevel,
        )
        dim = _default_softmax_dim(input)
    return dim


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, dim, dtype=None):
        if dtype is not None:
            input = input.to(dtype)
        out = torch.exp(input - input.max(dim=dim, keepdim=True).values)
        return out / out.sum(dim=dim, keepdim=True)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.dim = inputs[1]
        ctx.input_dtype = inputs[0].dtype
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        # Compute the jacobian explicitly will explode the memory, avoid that.
        grad_input = grad_output * out
        grad_input = grad_input - grad_input.sum(dim=ctx.dim, keepdim=True) * out
        return grad_input.to(ctx.input_dtype), None, None


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    dim = _resolve_dim(input, dim, _stacklevel)

    # # Regular impl without pedantical customized backward for learning.
    # if dtype is not None:
    #     input = input.to(dtype)
    # out = torch.exp(input - input.max(dim=dim, keepdim=True).values)
    # return out / out.sum(dim=dim, keepdim=True)

    return SoftmaxFunction.apply(input, dim, dtype)


class Softmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return softmax(input, self.dim)


class LogSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, dim, dtype=None):
        if dtype is not None:
            input = input.to(dtype)
        out = input - input.max(dim=dim, keepdim=True).values
        # Use torch.logsumexp for simplicity and speed, keep it plain for learning.
        return out - torch.log(torch.exp(out).sum(dim=dim, keepdim=True))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.dim = inputs[1]
        ctx.input_dtype = inputs[0].dtype
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = grad_output - grad_output.sum(
            dim=ctx.dim, keepdim=True
        ) * torch.exp(out)
        return grad_input.to(ctx.input_dtype), None, None


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    dim = _resolve_dim(input, dim, _stacklevel)

    # # Regular impl without pedantical customized backward for learning.
    # if dtype is not None:
    #     input = input.to(dtype)
    # out = input - input.max(dim=dim, keepdim=True).values
    # return out - torch.log(torch.exp(out).sum(dim=dim, keepdim=True))

    return LogSoftmaxFunction.apply(input, dim, dtype)


class LogSoftmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return log_softmax(input, self.dim)
