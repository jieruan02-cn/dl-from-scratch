import math
import torch
import torch.nn as nn
import warnings


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


class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return relu(x, self.inplace)


class LeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, negative_slope, inplace):
        ctx.negative_slope = negative_slope
        ctx.save_for_backward(input >= 0)
        if inplace:
            input[input < 0] *= negative_slope
            ctx.mark_dirty(input)
            return input
        return torch.where(input >= 0, input, negative_slope * input)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_input = torch.where(mask, grad_output, ctx.negative_slope * grad_output)
        return grad_input, None, None


def leaky_relu(input, negative_slope=0.01, inplace=False):
    return LeakyReLUFunction.apply(input, negative_slope, inplace)


class LeakyRelu(nn.Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return leaky_relu(input, self.negative_slope, self.inplace)


class PReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        if input.dim() == 0:
            shape = []
        else:
            # For 0D, an empty shape keeps weight 0-D too; using [-1] would force
            # input * weight.view([-1]) to broadcast 0-D × (1,) → (1,), changing output rank.
            shape = [1] * input.dim()
            shape[min(1, len(shape) - 1)] = -1
        ctx.shape = shape

        mask = input >= 0
        ctx.save_for_backward(mask, input, weight)
        return torch.where(mask, input, input * weight.view(shape))

    @staticmethod
    def backward(ctx, grad_output):
        (mask, input, weight) = ctx.saved_tensors
        weight_view = weight.view(ctx.shape)
        grad_input = torch.where(mask, grad_output, grad_output * weight_view)
        grad_weight = (
            torch.where(mask, 0, grad_output * input)
            .sum([i for (i, x) in enumerate(weight_view.shape) if x == 1])
            .view_as(weight)
        )
        return grad_input, grad_weight


class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(
            torch.full((num_parameters,), init, device=device, dtype=dtype)
        )

    def forward(self, input):
        return PReLUFunction.apply(input, self.weight)


class ELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, inplace):
        mask = input <= 0
        if inplace:
            input[mask] = alpha * (torch.exp(input[mask]) - 1)
            ctx.mark_dirty(input)
            out = input
        else:
            out = torch.where(mask, alpha * (torch.exp(input) - 1), input)
        # calling save_for_backward will save only the last tensors.
        ctx.save_for_backward(mask, out)
        ctx.alpha = alpha
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (mask, out) = ctx.saved_tensors
        grad_input = torch.where(mask, grad_output * (out + ctx.alpha), grad_output)
        return grad_input, None, None


def elu(input, alpha=1.0, inplace=False):
    return ELUFunction.apply(input, alpha, inplace)


class ELU(nn.Module):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return elu(input, self.alpha, self.inplace)


def _gaussian_cdf(input, approximate):
    if approximate == "none":
        return 0.5 * (1 + torch.erf(input / math.sqrt(2)))
    elif approximate == "tanh":
        return 0.5 * (1 + tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * input**3)))
    else:
        raise ValueError("unrecognized approximate method")


class GELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, approximate="none"):
        return input * _gaussian_cdf(input, approximate)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (input, approximate) = inputs
        ctx.approximate = approximate
        ctx.save_for_backward(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        gaussian_out = _gaussian_cdf(input, ctx.approximate)
        if ctx.approximate == "none":
            density = torch.exp(-(input**2) * 0.5) / math.sqrt(2 * math.pi)
        elif ctx.approximate == "tanh":
            # use tanh^'(x) = 1 - tanh^2(x)
            density = (
                4
                * gaussian_out
                * (1 - gaussian_out)
                * (1 + 3 * 0.044715 * input**2)
                / math.sqrt(2 * math.pi)
            )
        else:
            raise ValueError("unrecognized approximate method")
        grad_input = grad_output * (gaussian_out + input * density)
        return grad_input, None


def gelu(input, approximate="none"):
    return GELUFunction.apply(input, approximate)


class GELU(nn.Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return gelu(input, self.approximate)


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
