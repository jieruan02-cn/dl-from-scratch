import math
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, weight, bias):
        if bias is not None:
            if input.dim() >= 2:
                *batch_shape, in_features = input.shape
                input_view = input.reshape(-1, in_features)
                out = torch.addmm(bias, input_view, weight.mT)
                out = out.view(*batch_shape, weight.size(0))
            else:
                out = torch.addmv(bias, weight, input)
        else:
            out = input @ weight.mT
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight = None, None
        if ctx.needs_input_grad[0]:
            weight = inputs[1]
        if ctx.needs_input_grad[1]:
            input = inputs[0]
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        if ctx.needs_input_grad[1]:
            out_features, in_features = grad_output.size(-1), input.size(-1)
            grad_weight = grad_output.reshape(-1, out_features).mT @ input.reshape(
                -1, in_features
            )
        if ctx.needs_input_grad[2]:
            if grad_output.dim() > 1:
                grad_bias = grad_output.sum(dim=tuple(range(0, grad_output.dim() - 1)))
            else:
                grad_bias = grad_output
        return grad_input, grad_weight, grad_bias


def linear(input, weight, bias=None):
    return LinearFunction.apply(input, weight, bias)


# Lessons:
# 1. torch.matmul's broadcast rule requires row vector multiplication for batched input vectors.
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return linear(x, self.weight, self.bias)


def bilinear(input1, input2, weight, bias=None):
    # # Regular impl
    # # input1[:, None, None, :] fail shape generality if B's dimesnion is more than 1
    # out = (input1.unsqueeze(-2).unsqueeze(-2) @ weight).squeeze(-2)
    # out = (out @ input2.unsqueeze(-1)).squeeze(-1)

    # Superior einsum impl
    # b: batch, i: in1, j:in2, o: out
    out = torch.einsum("bi,oij,bj->bo", input1, weight, input2)
    return out if bias is None else out + bias  # preferred than out += bias


class LazyLinear(Linear):
    def __init__(self, out_features, bias=True, device=None, dtype=None):
        nn.Module.__init__(self)
        self.out_features = out_features
        self.in_features = 0
        self.weight = nn.UninitializedParameter(device=device, dtype=dtype)
        if bias:
            self.bias = nn.UninitializedParameter(device=device, dtype=dtype)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if isinstance(self.weight, nn.UninitializedParameter):
            self.in_features = x.shape[-1]
            self.weight.materialize((self.out_features, self.in_features))
            if self.bias is not None:
                self.bias.materialize((self.out_features,))
            self.reset_parameters()
        return linear(x, self.weight, self.bias)


class Bilinear(nn.Module):
    def __init__(
        self,
        in1_features,
        in2_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in1_features, in2_features), device=device, dtype=dtype
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, input1, input2):
        return bilinear(input1, input2, self.weight, self.bias)

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in1_features) if self.in1_features > 0 else 0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
