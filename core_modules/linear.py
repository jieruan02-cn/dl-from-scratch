import math
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class LinearFunction(torch.autograd.Function):
    # allows the function to support torch.vmap()
    generate_vmap_rule = True

    @staticmethod
    def forward(input, weight, bias):
        if bias is not None:
            if input.dim() >= 2:
                *batch_shape, _ = input.shape
                # flatten(0, -2) is better than reshape(-1, in_featurs) for in_features
                # = 0. Note this requires input.dim() >= 2.
                out = torch.addmm(bias, input.flatten(0, -2), weight.mT)
                out = out.view(*batch_shape, weight.size(0))
            else:
                out = torch.addmv(bias, weight, input)
        else:
            out = input @ weight.mT
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        saved_input, saved_weight = None, None
        if ctx.needs_input_grad[0]:
            saved_weight = inputs[1]
        if ctx.needs_input_grad[1]:
            saved_input = inputs[0]
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(saved_input, saved_weight)
            ctx.input_dtype = inputs[0].dtype
            ctx.weight_dtype = inputs[1].dtype
            ctx.bias_dtype = None if inputs[2] is None else inputs[2].dtype

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None
        # For handling AMP (Automatic Mixed Precision) where output is bf16 while
        # input/weight is fp32
        output_dtype = grad_output.dtype
        if ctx.needs_input_grad[0]:
            grad_input = (grad_output @ weight.to(output_dtype)).to(ctx.input_dtype)
        if ctx.needs_input_grad[1]:
            input = input.to(output_dtype)
            if grad_output.dim() > 1:
                grad_weight = grad_output.flatten(0, -2).mT @ input.flatten(0, -2)
            else:
                grad_weight = torch.outer(grad_output, input)
            grad_weight = grad_weight.to(ctx.weight_dtype)
        if ctx.needs_input_grad[2]:
            if grad_output.dim() > 1:
                grad_bias = grad_output.flatten(0, -2).sum(dim=0)
            else:
                grad_bias = grad_output
            grad_bias = grad_bias.to(ctx.bias_dtype)
        return grad_input, grad_weight, grad_bias


def linear(input, weight, bias=None):
    return LinearFunction.apply(input, weight, bias)


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

    def forward(self, input):
        return linear(input, self.weight, self.bias)


# Classical use cases:
# 1. first FFN after a conv/flatten stack, without laziness, we have to hand-compute the
# length of conv flatten output, which is annoying and prone to bugs.
# 2. Config/data-driven model, in_features = len(columns).
# Laziness has a cost that anything needs shapes early breaks, e.g. DDP wrapping. The
# standard discipline is run one dummy forward to materialize everything, then build the
# optimizer / wrap in DDP / load weights. Great for experiemnt but not production code.
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

    def forward(self, input):
        if isinstance(self.weight, nn.UninitializedParameter):
            self.in_features = input.shape[-1]
            self.weight.materialize((self.out_features, self.in_features))
            if self.bias is not None:
                self.bias.materialize((self.out_features,))
            self.reset_parameters()
        return linear(input, self.weight, self.bias)


class BilinearFunction(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(input1, input2, weight, bias):
        # # Regular impl
        # # input1[:, None, None, :] fail shape generality if B's dimesnion is more than 1
        # out = (input1.unsqueeze(-2).unsqueeze(-2) @ weight).squeeze(-2)
        # out = (out @ input2.unsqueeze(-1)).squeeze(-1)

        # Superior einsum impl
        out = torch.einsum("...i,oij,...j->...o", input1, weight, input2)
        if bias is not None:
            # preferred than out += bias to avoid inplace-modification complication for grad.
            out = out + bias
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        input1, input2, weight = None, None, None
        if ctx.needs_input_grad[0]:
            input2, weight = inputs[1], inputs[2]
        if ctx.needs_input_grad[1]:
            input1, weight = inputs[0], inputs[2]
        if ctx.needs_input_grad[2]:
            input1, input2 = inputs[0], inputs[1]

        ctx.save_for_backward(input1, input2, weight)
        ctx.input1_dtype = inputs[0].dtype
        ctx.input2_dtype = inputs[1].dtype
        ctx.weight_dtype = inputs[2].dtype
        ctx.bias_dtype = None if inputs[3] is None else inputs[3].dtype

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, weight = ctx.saved_tensors
        grad_input1, grad_input2, grad_weight, grad_bias = None, None, None, None
        output_dtype = grad_output.dtype
        if ctx.needs_input_grad[0]:
            grad_input1 = torch.einsum(
                "...o,oij,...j->...i",
                grad_output,
                weight.to(output_dtype),
                input2.to(output_dtype),
            ).to(ctx.input1_dtype)
        if ctx.needs_input_grad[1]:
            grad_input2 = torch.einsum(
                "...o,...i,oij->...j",
                grad_output,
                input1.to(output_dtype),
                weight.to(output_dtype),
            ).to(ctx.input2_dtype)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.einsum(
                "...o,...i,...j->oij",
                grad_output,
                input1.to(output_dtype),
                input2.to(output_dtype),
            ).to(ctx.weight_dtype)
        if ctx.needs_input_grad[3]:
            if grad_output.dim() > 1:
                grad_bias = grad_output.flatten(0, -2).sum(dim=0)
            else:
                grad_bias = grad_output
            grad_bias = grad_bias.to(ctx.bias_dtype)
        return grad_input1, grad_input2, grad_weight, grad_bias


def bilinear(input1, input2, weight, bias=None):
    return BilinearFunction.apply(input1, input2, weight, bias)


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
