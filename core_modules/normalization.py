import torch
import torch.nn as nn
import torch.distributed as dist


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
            ctx.batch_dims = tuple(range(input.dim() - len(normalized_shape)))
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
            grad_weight = (grad_output * normed_input).sum(ctx.batch_dims)
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(ctx.batch_dims)

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
            ctx.batch_dims = tuple(range(input.dim() - len(normalized_shape)))
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
            grad_weight = (grad_output * normed_input).sum(dim=ctx.batch_dims)
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
        # use reshape instead of view as view will break on non-contiguous input, e.g.
        # transpose or permute, reshape will work always and copy only when needed.
        input_view = input.reshape((N, num_groups, C // num_groups) + F)
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
        return out.reshape_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, mean, rstd, weight = ctx.saved_tensors
        normed_input = (input - mean) * rstd
        grad_input, grad_weight, grad_bias = None, None, None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.reshape_as(input)
            if weight is not None:
                grad_input = grad_input * weight.view(ctx.affine_shape)
            grad_input = (
                grad_input
                - torch.mean(grad_input, dim=ctx.agg_dim, keepdim=True)
                - torch.mean(grad_input * normed_input, dim=ctx.agg_dim, keepdim=True)
                * normed_input
            ) * rstd
            grad_input = grad_input.reshape_as(grad_output)

        batch_dims = (0,) + tuple(range(2, grad_output.dim()))
        if ctx.needs_input_grad[2]:
            grad_weight = (grad_output * normed_input.reshape_as(grad_output)).sum(
                dim=batch_dims
            )
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(dim=batch_dims)
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


class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    # the training argument is needed to distinguish BN.eval() and BN.train(), for most
    # modules, their eval() and train() behaves the same, and backward() shouldn't run
    # for eval() model. One exception is BN and Dropout, thus the trainning flag is
    # needed to distinguish the backward run, as mu and var are fixed in BN.eval().
    def forward(ctx, input, mean, var, weight, bias, training, eps):
        rstd = torch.rsqrt(var + eps)
        out = (input - mean) * rstd
        affine_shape = (input.shape[1],) + (1,) * (input.dim() - 2)
        if weight is not None:
            out = out * weight.reshape(affine_shape)
        if bias is not None:
            out = out + bias.reshape(affine_shape)

        if any(ctx.needs_input_grad):
            ctx.eps = eps
            ctx.affine_shape = affine_shape
            ctx.save_for_backward(input, mean, rstd, weight)
            ctx.training = training
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = None, None, None

        if any(ctx.needs_input_grad):
            input, mean, rstd, weight = ctx.saved_tensors
            normed_input = (input - mean) * rstd
            reduce_dims = (0,) + tuple(range(2, grad_output.dim()))

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * rstd
            if weight is not None:
                grad_input = grad_input * weight.reshape(ctx.affine_shape)
            if ctx.training:
                # dim = reduce_dims as average is over not just the batch dim, for 3D/4D/5D,
                # all remaining dimensions are essentially treated as one.
                grad_input = (
                    grad_input
                    - torch.mean(grad_input, dim=reduce_dims, keepdim=True)
                    - torch.mean(
                        grad_input * normed_input, dim=reduce_dims, keepdim=True
                    )
                    * normed_input
                )

        if ctx.needs_input_grad[3]:
            grad_weight = (grad_output * normed_input).sum(dim=reduce_dims)
        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=reduce_dims)
        return grad_input, None, None, grad_weight, grad_bias, None, None


def batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):

    if training or running_mean is None or running_var is None:
        reduce_dims = (0,) + tuple(range(2, input.dim()))
        var, mean = torch.var_mean(input, dim=reduce_dims, correction=0)
        if running_mean is not None:
            # Calling detach() to cut running stat out of the graph, as var, mean comes
            # from input which requires_grad = True.
            running_mean.mul_(1 - momentum).add_(mean.detach(), alpha=momentum)
        if running_var is not None:
            # Avoid recompute unbiased_var.
            unbiased_var = var / (1.0 - 1.0 / (input.numel() // input.shape[1]))
            running_var.mul_(1 - momentum).add_(unbiased_var.detach(), alpha=momentum)
    else:
        mean, var = running_mean, running_var
    affine_shape = (input.shape[1],) + (1,) * (input.dim() - 2)
    return BatchNormFunction.apply(
        input,
        mean.reshape(affine_shape),
        var.reshape(affine_shape),
        weight,
        bias,
        training,
        eps,
    )


class _BatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
        *,
        bias=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight, self.bias = None, None
        config = {"device": device, "dtype": dtype}
        if affine:
            self.weight = nn.Parameter(torch.ones((num_features,), **config))
            if bias:
                self.bias = nn.Parameter(torch.zeros((num_features,), **config))

        # register_buffer ensures the tensor moves with the modules and saved in the
        # state_dict, which simple tensor construction don't. neither are tracked by
        # optimizer. mainly used for fixed, non-learnable parameters, e.g. positional
        # encodings, BN running stat. directly define tensor is for throwaway tensors
        # inside the forward pass.
        self.register_buffer(
            "running_mean",
            torch.zeros((num_features,), **config) if track_running_stats else None,
        )
        # var initialized to 1.0 to avoid exploading variance.
        self.register_buffer(
            "running_var",
            torch.ones((num_features,), **config) if track_running_stats else None,
        )

    def forward(self, input):
        self._check_input_dim(input)
        return batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(f"Expect 2D/3D input, got {input.dim()}D input.")


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"Expect 4D input, got {input.dim()}D input.")


class BatchNorm3d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f"Expect 5D input, got {input.dim()}D input.")


class SyncBatchNorm(_BatchNorm):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        process_group=None,
        device=None,
        dtype=None,
        *,
        bias=True,
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
            bias=bias,
        )
        self.process_group = process_group

    def forward(self, input):
        if input.dim() < 3:
            raise ValueError(f"Expect at least 3D input, got {input.dim()}D input.")
        if self.training or self.running_mean is None or self.running_var is None:
            reduce_dims = (0,) + tuple(range(2, input.dim()))
            sum = torch.sum(input, dim=reduce_dims)
            dist.all_reduce(sum, dist.ReduceOp.SUM, self.process_group)
            square_sum = torch.sum(input * input, dim=reduce_dims)
            dist.all_reduce(square_sum, dist.ReduceOp.SUM, self.process_group)
            N = torch.tensor([input.numel() // input.shape[1]], device=input.device)
            dist.all_reduce(N, dist.ReduceOp.SUM, self.process_group)

            mean = sum / N
            var = square_sum / N - mean * mean
            if self.running_mean is not None:
                self.running_mean.mul_(1 - self.momentum).add_(
                    mean.detach(), alpha=self.momentum
                )
            if self.running_var is not None:
                unbiased_var = var * N / (N - 1)
                self.running_var.mul_(1 - self.momentum).add_(
                    unbiased_var.detach(), alpha=self.momentum
                )
        else:
            mean, var = self.running_mean, self.running_var
        affine_shape = (input.shape[1],) + (1,) * (input.dim() - 2)
        return BatchNormFunction.apply(
            input,
            mean.reshape(affine_shape),
            var.reshape(affine_shape),
            self.weight,
            self.bias,
            self.training,
            self.eps,
        )
