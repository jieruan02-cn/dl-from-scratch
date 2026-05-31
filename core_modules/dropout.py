import torch
import torch.nn as nn
from activation import SELUFunction


def get_dropout_mask_shape(input, dim):
    if dim == 0:
        return input.shape

    assert input.dim() in (dim + 1, dim + 2)
    return input.shape[:-dim] + (1,) * dim


class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p, training, inplace, dim):
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"dropout probability has to be in [0, 1.0), got {p}")

        mask = None
        if training:
            out = input if inplace else input.clone()
            # Use boolean mask to save memory. Using rng state saves more but computing
            # random number for large tensor is computationally expensive and unclear
            # trade-off.
            mask_shape = get_dropout_mask_shape(input, dim)
            mask = torch.rand(mask_shape, device=input.device) > p
            out.mul_(mask).div_(1 - p)
        else:
            out = input

        if inplace:
            ctx.mark_dirty(input)
        if ctx.needs_input_grad[0]:
            ctx.p = p
            ctx.training = training
            ctx.save_for_backward(mask)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        if ctx.training:
            grad_input = grad_output * mask
            grad_input.div_(1 - ctx.p)
        else:
            grad_input = grad_output
        return grad_input, None, None, None, None


def dropout(input, p=0.5, training=True, inplace=False):
    return DropoutFunction.apply(input, p, training, inplace, 0)


def dropout1d(input, p=0.5, training=True, inplace=False):
    return DropoutFunction.apply(input, p, training, inplace, 1)


def dropout2d(input, p=0.5, training=True, inplace=False):
    return DropoutFunction.apply(input, p, training, inplace, 2)


def dropout3d(input, p=0.5, training=True, inplace=False):
    return DropoutFunction.apply(input, p, training, inplace, 3)


class Dropout(nn.Module):
    _fn = staticmethod(dropout)

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return self._fn(input, self.p, self.training, self.inplace)


class Dropout1d(Dropout):
    _fn = staticmethod(dropout1d)


class Dropout2d(Dropout):
    _fn = staticmethod(dropout2d)


class Dropout3d(Dropout):
    _fn = staticmethod(dropout3d)


class AlphaDropoutFunction(torch.autograd.Function):
    ALPHA = -SELUFunction.ALPHA * SELUFunction.SCALE

    @staticmethod
    def forward(ctx, input, p, training, inplace, mask_shape):
        mask, a = None, None
        if training:
            out = input if inplace else input.clone()
            mask = torch.rand(mask_shape, device=input.device) > p
            out.sub_(AlphaDropoutFunction.ALPHA).mul_(mask).add_(
                AlphaDropoutFunction.ALPHA
            )

            a = (1 - p + AlphaDropoutFunction.ALPHA**2 * p * (1 - p)) ** (-0.5)
            b = -p * AlphaDropoutFunction.ALPHA * a
            out.mul_(a).add_(b)
        else:
            out = input

        if inplace:
            ctx.mark_dirty(input)
        if ctx.needs_input_grad[0]:
            ctx.a = a
            ctx.training = training
            ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.training:
            (mask,) = ctx.saved_tensors
            grad_input = grad_output * mask * ctx.a
        else:
            grad_input = grad_output
        return grad_input, None, None, None


def alpha_dropout(input, p=0.5, training=False, inplace=False):
    return AlphaDropoutFunction.apply(input, p, training, inplace, input.shape)


def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    assert input.dim() in (4, 5)
    mask_shape = input.shape[:-3] + (1, 1, 1)
    return AlphaDropoutFunction.apply(input, p, training, inplace, mask_shape)


class AlphaDropout(nn.Module):
    _fn = staticmethod(alpha_dropout)

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return self._fn(input, self.p, self.training, self.inplace)


class FeatureAlphaDropout(AlphaDropout):
    _fn = staticmethod(feature_alpha_dropout)
