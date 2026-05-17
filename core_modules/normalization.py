import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, aggregate_dim, normalized_shape, eps):
        input_var = torch.sqrt(
            input.var(dim=aggregate_dim, correction=0, keepdim=True) + eps
        )
        input_normed = (input - input.mean(dim=aggregate_dim, keepdim=True)) / input_var
        out = input_normed * weight if weight is not None else input_normed
        if bias is not None:
            out = out + bias

        if any(ctx.needs_input_grad):
            ctx.save_for_backward(input_normed, input_var, weight)
            ctx.aggregate_dim = aggregate_dim
            ctx.batch_dim = tuple(range(input.dim() - len(normalized_shape)))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_normed, input_var, weight) = ctx.saved_tensors
        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_output * input_normed).sum(ctx.batch_dim)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(ctx.batch_dim)

        grad_input = grad_output * weight if weight is not None else grad_output
        grad_input = (
            grad_input
            - torch.mean(grad_input, dim=ctx.aggregate_dim, keepdim=True)
            - torch.mean(grad_input * input_normed, dim=ctx.aggregate_dim, keepdim=True)
            * input_normed
        ) / input_var
        return grad_input, grad_weight, grad_bias, None, None, None


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            self.normalized_shape = torch.Size(normalized_shape)
        elif isinstance(normalized_shape, torch.Size):
            self.normalized_shape = normalized_shape
        else:
            raise TypeError(
                f"normalized_shape must be int, list, or torch.Size, got {type(normalized_shape)}"
            )
        self.aggregate_dim = tuple(range(-len(self.normalized_shape), 0))
        self.eps = float(eps)
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(normalized_shape, device=device, dtype=dtype)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(normalized_shape, device=device, dtype=dtype)
                )

    def forward(self, input):
        return LayerNormFunction.apply(
            input,
            self.weight,
            self.bias,
            self.aggregate_dim,
            self.normalized_shape,
            self.eps,
        )
