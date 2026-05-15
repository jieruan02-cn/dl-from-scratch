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


class ReLU6Function(torch.autograd.Function):
    @staticmethod
    def forward(input, inplace):
        return input.clamp_(0.0, 6.0) if inplace else input.clamp(0.0, 6.0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, inplace = inputs
        if inplace:
            ctx.mark_dirty(input)
        if input.requires_grad:
            # saves output instead of mask as output already in VRAM and saves memory in
            # fact compared to constructing a new mask tensor
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = torch.where((out > 0.0) & (out < 6.0), grad_output, 0.0)
        return grad_input, None


def relu6(input, inplace=False):
    return ReLU6Function.apply(input, inplace)


class ReLU6(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return relu6(input, self.inplace)


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
    def forward(input, alpha, inplace):
        mask = input <= 0.0
        out = input if inplace else input.clone()
        out[mask] = alpha * torch.expm1(out[mask])
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, alpha, inplace = inputs
        ctx.alpha = alpha
        if inplace:
            ctx.mark_dirty(input)
        if input.requires_grad:
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = torch.where(
            out <= 0.0, grad_output * (out + ctx.alpha), grad_output
        )
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


class SiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace=False):
        needs_grad = input.requires_grad
        if inplace:
            if needs_grad:
                ctx.save_for_backward(input.clone())
            input.mul_(sigmoid(input))
            ctx.mark_dirty(input)
            return input
        else:
            if needs_grad:
                ctx.save_for_backward(input)
            return input * sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        sig = sigmoid(input)
        grad_input = grad_output * sig * (1 + input - input * sig)
        return grad_input, None


def silu(input, inplace=False):
    return SiLUFunction.apply(input, inplace)


class SiLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return silu(input, self.inplace)


# Note: GLU(a, b) = a * sigmoid(b) as PyTorch but not as the rest GLU in industry standard.
class GLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        assert input.shape[dim] % 2 == 0
        a, b = torch.chunk(input, 2, dim)
        sig_b = sigmoid(b)
        out = a * sig_b

        ctx.dim = dim
        # saving torch.narrow(input, dim, 0, length) will cause saving whole input as
        # pytorch doesn't support save half of allocation.
        ctx.save_for_backward(out, sig_b)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out, sig_b) = ctx.saved_tensors
        grad_a = grad_output * sig_b
        grad_input = torch.cat((grad_a, grad_output * out * (1 - sig_b)), ctx.dim)
        return grad_input, None


def glu(input, dim=-1):
    return GLUFunction.apply(input, dim)


class GLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return glu(input, self.dim)


def reglu(input, dim=-1):
    assert input.shape[dim] % 2 == 0
    a, b = torch.chunk(input, 2, dim)
    return relu(a) * b


class ReGLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return reglu(input, self.dim)


def geglu(input, dim=-1):
    assert input.shape[dim] % 2 == 0
    a, b = torch.chunk(input, 2, dim)
    return gelu(a) * b


class GeGLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return geglu(input, self.dim)


def swish(input, beta=1):
    if isinstance(beta, (int, float)) and beta == 1.0:
        return silu(input)

    return input * sigmoid(beta * input)


def swishglu(input, dim=-1, beta=1):
    assert input.shape[dim] % 2 == 0
    a, b = torch.chunk(input, 2, dim)
    return swish(a, beta) * b


class SwishGLU(nn.Module):
    def __init__(
        self, dim=-1, learnable_beta=False, beta_init=1.0, device=None, dtype=None
    ):
        super().__init__()
        self.dim = dim
        if learnable_beta:
            self.beta = nn.Parameter(
                torch.tensor(beta_init, device=device, dtype=dtype)
            )
        else:
            self.beta = beta_init

    def forward(self, input):
        return swishglu(input, self.dim, self.beta)


class SoftplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, beta, threshold):
        mask = input * beta <= threshold
        out = input.clone()
        out[mask] = torch.log1p(torch.exp(out[mask] * beta)) / beta
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, beta, threshold = inputs
        ctx.beta = beta
        ctx.threshold = threshold
        if input.requires_grad:
            ctx.save_for_backward(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        mask = input * ctx.beta <= ctx.threshold
        grad_input = grad_output * torch.where(mask, sigmoid(ctx.beta * input), 1.0)
        return grad_input, None, None


def softplus(input, beta=1.0, threshold=20.0):
    assert beta > 0
    return SoftplusFunction.apply(input, beta, threshold)


class Softplus(nn.Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        return softplus(input, self.beta, self.threshold)


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace):
        needs_grad = input.requires_grad
        tanh_softplus = tanh(softplus(input))
        if inplace:
            if needs_grad:
                ctx.save_for_backward(input.clone())
            input.mul_(tanh_softplus)
            ctx.mark_dirty(input)
            return input
        else:
            if needs_grad:
                ctx.save_for_backward(input)
            return input * tanh_softplus

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tanh_softplus = tanh(softplus(input))
        # use sigmoid(input) instead of torch.where(input <= 20.0, sigmoid(input), 1.0)
        # as sigmoid saturated to near 1.0 after x >= 20, no need for piecewise.
        grad_input = grad_output * (
            tanh_softplus + input * (1 - tanh_softplus**2) * sigmoid(input)
        )
        return grad_input, None


def mish(input, inplace=False):
    return MishFunction.apply(input, inplace)


class Mish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return mish(input, self.inplace)


class SELUFunction(torch.autograd.Function):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    @staticmethod
    def forward(input, inplace):
        mask = input <= 0.0
        out = input if inplace else input.clone()
        # use torch.expm1 instead of (torch.exp(out[mask]) - 1) to preserve numerical
        # accuracy for very small negative x ~ -10^-8
        out[mask] = SELUFunction.alpha * torch.expm1(out[mask])
        out *= SELUFunction.scale
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, inplace = inputs
        if inplace:
            ctx.mark_dirty(input)
        if input.requires_grad:
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        mask = out <= 0.0
        grad_input = grad_output * torch.where(
            mask, out + SELUFunction.alpha * SELUFunction.scale, SELUFunction.scale
        )
        return grad_input, None


def selu(input, inplace=False):
    return SELUFunction.apply(input, inplace)


class SELU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return selu(input, self.inplace)


class HardtanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, min_val, max_val, inplace):
        return (
            input.clamp_(min_val, max_val) if inplace else input.clamp(min_val, max_val)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, min_val, max_val, inplace = inputs
        if inplace:
            ctx.mark_dirty(input)
        if input.requires_grad:
            ctx.min_val = min_val
            ctx.max_val = max_val
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        mask = (out <= ctx.min_val) | (out >= ctx.max_val)
        grad_input = torch.where(mask, 0.0, grad_output)
        return grad_input, None, None, None


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    return HardtanhFunction.apply(input, min_val, max_val, inplace)


class Hardtanh(nn.Module):
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def forward(self, input):
        return hardtanh(input, self.min_val, self.max_val, self.inplace)


class HardsigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, inplace):
        out = input if inplace else input.clone()
        out.add_(3).clamp_(0, 6).div_(6)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, inplace = inputs
        if input.requires_grad:
            ctx.save_for_backward(output)
        if inplace:
            ctx.mark_dirty(input)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = torch.where((out > 0.0) & (out < 1.0), grad_output / 6.0, 0.0)
        return grad_input, None


def hardsigmoid(input, inplace=False):
    return HardsigmoidFunction.apply(input, inplace)


class Hardsigmoid(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return hardsigmoid(input, self.inplace)


class HardswishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace):
        if input.requires_grad:
            ctx.save_for_backward(input.clone() if inplace else input)

        hardsig_component = input.add(3.0).clamp_(0.0, 6.0).div_(6.0)
        if inplace:
            input.mul_(hardsig_component)
            ctx.mark_dirty(input)
            return input
        else:
            return input * hardsig_component

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output * torch.where(
            input <= -3, 0.0, torch.where(input >= 3, 1.0, input / 3 + 0.5)
        )
        return grad_input, None


def hardswish(input, inplace=False):
    return HardswishFunction.apply(input, inplace)


class Hardswish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return hardswish(input, self.inplace)


class LogSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(input):
        mask = input > 0.0
        out = torch.empty_like(input)
        out[mask] = -torch.log1p(torch.exp(-input[mask]))
        out[~mask] = input[~mask] - torch.log1p(torch.exp(input[~mask]))
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        if inputs[0].requires_grad:
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (1.0 - torch.exp(out))


def logsigmoid(input):
    return LogSigmoidFunction.apply(input)


class LogSigmoid(nn.Module):
    def forward(self, input):
        return logsigmoid(input)


class SoftsignFunction(torch.autograd.Function):
    @staticmethod
    def forward(input):
        return input / (1.0 + torch.abs(input))

    @staticmethod
    def setup_context(ctx, inputs, output):
        if inputs[0].requires_grad:
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (1.0 - torch.abs(out)) ** 2


def softsign(input):
    return SoftsignFunction.apply(input)


class Softsign(nn.Module):
    def forward(self, input):
        return softsign(input)


class TanhshrinkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        tanh_input = tanh(input)
        if input.requires_grad:
            ctx.save_for_backward(tanh_input)
        return input - tanh_input

    @staticmethod
    def backward(ctx, grad_output):
        (tanh_input,) = ctx.saved_tensors
        return grad_output * tanh_input**2


def tanhshrink(input):
    return TanhshrinkFunction.apply(input)


class Tanhshrink(nn.Module):
    def forward(self, input):
        return tanhshrink(input)


class HardshrinkFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, lambd):
        return torch.where((input < -lambd) | (input > lambd), input, 0.0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        if inputs[0].requires_grad:
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = grad_output * (out != 0.0)
        return grad_input, None


def hardshrink(input, lambd=0.5):
    return HardshrinkFunction.apply(input, lambd)


class Hardshrink(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return hardshrink(input, self.lambd)


class SoftshrinkFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, lambd):
        return torch.where(
            input > lambd,
            input - lambd,
            torch.where(input < -lambd, input + lambd, 0.0),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        if inputs[0].requires_grad:
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = grad_output * (out != 0.0)
        return grad_input, None


def softshrink(input, lambd=0.5):
    return SoftshrinkFunction.apply(input, lambd)


class Softshrink(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return softshrink(input, self.lambd)


class CELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, alpha, inplace):
        mask = input < 0.0
        out = input if inplace else input.clone()
        out[mask] = alpha * torch.expm1(out[mask] / alpha)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, alpha, inplace = inputs
        ctx.alpha = alpha
        if inplace:
            ctx.mark_dirty(input)
        if input.requires_grad:
            ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = grad_output * torch.where(out < 0.0, 1.0 + out / ctx.alpha, 1.0)
        return grad_input, None, None


def celu(input, alpha=1.0, inplace=False):
    return CELUFunction.apply(input, alpha, inplace)


class CELU(nn.Module):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return celu(input, self.alpha, self.inplace)
