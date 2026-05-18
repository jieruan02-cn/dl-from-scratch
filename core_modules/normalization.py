import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps):
        agg_dim = tuple(range(-len(normalized_shape), 0))
        var, mean = torch.var_mean(input, dim=agg_dim, correction=0, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        if weight is None:
            out = (input - mean) * rstd
        else:
            out = (input - mean) * rstd * weight
        if bias is not None:
            out = out + bias

        if any(ctx.needs_input_grad):
            # save input instead of normalized input to reduce persistent
            # saved-for-backward state (held across entire forward pass until that
            # layer's backward fires), input is typically needed anyway due to residual
            # in transformer, so mean + rstd is cheaper than normed_input (N) + rstd
            ctx.save_for_backward(input, mean, rstd, weight)
            ctx.agg_dim = agg_dim
            ctx.batch_dim = tuple(range(input.dim() - len(normalized_shape)))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, mean, rstd, weight = ctx.saved_tensors
        normed_input = (input - mean) * rstd
        grad_input, grad_weight, grad_bias = None, None, None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output if weight is None else grad_output * weight
            grad_input = (
                grad_input
                - torch.mean(grad_input, dim=ctx.agg_dim, keepdim=True)
                - torch.mean(grad_input * normed_input, dim=ctx.agg_dim, keepdim=True)
                * normed_input
            ) * rstd
        if ctx.needs_input_grad[2]:
            grad_weight = (grad_output * normed_input).sum(ctx.batch_dim)
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(ctx.batch_dim)

        return grad_input, None, grad_weight, grad_bias, None


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    return LayerNormFunction.apply(input, normalized_shape, weight, bias, eps)


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
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, eps):
        agg_dim = tuple(range(-len(normalized_shape), 0))
        rms_sq = torch.mean(input * input, dim=agg_dim, keepdim=True)
        default_eps = torch.finfo(torch.promote_types(input.dtype, torch.float32)).eps
        rstd = torch.rsqrt(rms_sq + default_eps if eps is None else rms_sq + eps)
        out = input * rstd if weight is None else input * rstd * weight

        if any(ctx.needs_input_grad):
            ctx.save_for_backward(input, rstd, weight)
            ctx.agg_dim = agg_dim
            ctx.batch_dim = tuple(range(input.dim() - len(normalized_shape)))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, rstd, weight = ctx.saved_tensors
        normed_input = input * rstd
        grad_input, grad_weight = None, None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output if weight is None else grad_output * weight
            grad_input = (
                grad_input
                - torch.mean(grad_input * normed_input, dim=ctx.agg_dim, keepdim=True)
                * normed_input
            ) * rstd
        if ctx.needs_input_grad[2]:
            grad_weight = (grad_output * normed_input).sum(dim=ctx.batch_dim)
        return grad_input, None, grad_weight, None


def rms_norm(input, normalized_shape, weight=None, eps=None):
    return RMSNormFunction.apply(input, normalized_shape, weight, eps)


class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=None,
        elementwise_affine=True,
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
        self.eps = eps if eps is None else float(eps)
        self.weight = None
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=device, dtype=dtype)
            )

    def forward(self, input):
        return rms_norm(input, self.normalized_shape, self.weight, self.eps)


class GroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_groups, weight, bias, eps):
        N, C, F = input.shape[0], input.shape[1], input.shape[2:]
        input_view = input.view((N, num_groups, C // num_groups) + F)
        agg_dim = tuple(range(2, input_view.dim()))
        var, mean = torch.var_mean(input_view, dim=agg_dim, correction=0, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        affine_shape = (num_groups, C // num_groups) + (1,) * (input_view.dim() - 3)
        if weight is None:
            out = (input_view - mean) * rstd
        else:
            out = (input_view - mean) * rstd * weight.view(affine_shape)
        if bias is not None:
            out = out + bias.view(affine_shape)

        if any(ctx.needs_input_grad):
            ctx.save_for_backward(input_view, mean, rstd, weight)
            ctx.agg_dim = agg_dim
            ctx.affine_shape = affine_shape
        return out.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, mean, rstd, weight = ctx.saved_tensors
        normed_input = (input - mean) * rstd
        grad_input, grad_weight, grad_bias = None, None, None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.view_as(input)
            if weight is not None:
                grad_input = grad_input * weight.view(ctx.affine_shape)
            grad_input = (
                grad_input
                - torch.mean(grad_input, dim=ctx.agg_dim, keepdim=True)
                - torch.mean(grad_input * normed_input, dim=ctx.agg_dim, keepdim=True)
                * normed_input
            ) * rstd
            grad_input = grad_input.view_as(grad_output)

        batch_dim = (0,) + tuple(range(2, grad_output.dim()))
        if ctx.needs_input_grad[2]:
            grad_weight = (grad_output * normed_input.view_as(grad_output)).sum(
                dim=batch_dim
            )
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(dim=batch_dim)
        return grad_input, None, grad_weight, grad_bias, None


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    return GroupNormFunction.apply(input, num_groups, weight, bias, eps)


class GroupNorm(nn.Module):
    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-05,
        affine=True,
        device=None,
        dtype=None,
        *,
        bias=True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = float(eps)
        self.weight, self.bias = None, None
        if affine:
            self.weight = nn.Parameter(
                torch.ones(num_channels, device=device, dtype=dtype)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(num_channels, device=device, dtype=dtype)
                )

    def forward(self, input):
        return group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
