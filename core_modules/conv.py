import math
import torch
import torch.nn as nn


def _canonical_tuple(input, dim):
    return (input,) * dim if isinstance(input, int) else input


def _canonical_dilated_kernel_size(kernel_size, dilation):
    return tuple((k - 1) * d + 1 for k, d in zip(kernel_size, dilation))


def _canonical_padding(padding, dim, stride, size):
    if isinstance(padding, int):
        return (padding,) * dim
    elif isinstance(padding, tuple):
        return padding
    elif isinstance(padding, str):
        if padding == "valid":
            return (0,) * dim
        elif padding == "same":
            assert all(x == 1 for x in stride) and all(s % 2 != 0 for s in size)
            return tuple((s - 1) // 2 for s in size)
        else:
            raise ValueError(f"Expect padding to be valid or same, got {padding}")
    else:
        raise TypeError(f"Expect padding of type int/tuple/str, got {type(padding)}")


# TODO(jieruan): Eliminate im2col entirely via shift-and-add
def conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    B, dim = input.size(0), input.dim() - 2
    C_out, C_in_per_group, *kernel_size = weight.shape
    # view first to avoid one memory copy after unfold.
    input = input.view(B, groups, C_in_per_group, *input.shape[2:])

    stride = _canonical_tuple(stride, dim)
    dilation = _canonical_tuple(dilation, dim)
    size = _canonical_dilated_kernel_size(kernel_size, dilation)
    padding = _canonical_padding(padding, dim, stride, size)
    if any(p != 0 for p in padding):
        pad = tuple(e for pair in reversed(list(zip(padding, padding))) for e in pair)
        input = nn.functional.pad(input, pad)
    for i in range(dim):
        input = input.unfold(dimension=i + 3, size=size[i], step=stride[i])
    # use slice instead of range
    indexing_tuple = tuple([slice(None)] * (3 + dim)) + tuple(
        slice(0, size[i], dilation[i]) for i in range(dim)
    )
    input = input[indexing_tuple]
    out_feature_shape = input.shape[3 : (3 + dim)]

    input = input.permute(
        0, 1, *range(3, 3 + dim), 2, *range(3 + dim, 3 + 2 * dim)
    ).reshape(B, groups, math.prod(out_feature_shape), -1)
    out = input @ weight.view(groups, C_out // groups, -1).transpose(1, 2)
    out = out.transpose(-1, -2).reshape(B, C_out, *out_feature_shape)
    if bias is not None:
        out = out + bias.view((1, C_out) + (1,) * dim)
    return out


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return conv(input, weight, bias, stride, padding, dilation, groups)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return conv(input, weight, bias, stride, padding, dilation, groups)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return conv(input, weight, bias, stride, padding, dilation, groups)


class ConvBase(nn.Module):
    dim = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                f"Expect in_channels % groups == 0 and out_channels % groups == 0, got in_channels = {in_channels}, out_channels ={out_channels}, groups = {groups}"
            )
        if padding_mode not in ("zeros", "reflect", "replicate", "circular"):
            raise ValueError(
                f"Expect padding_mode to be zeros/reflect/replicate/circular, got {padding_mode}"
            )
        if isinstance(kernel_size, tuple) and self.dim != len(kernel_size):
            raise ValueError(
                f"Expect len(kernel_size) == dim, got kernel_size = {kernel_size}, dim = {self.dim}"
            )

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _canonical_tuple(kernel_size, self.dim)
        self.stride = _canonical_tuple(stride, self.dim)
        self.dilation = _canonical_tuple(dilation, self.dim)
        size = _canonical_dilated_kernel_size(self.kernel_size, self.dilation)
        self.padding = _canonical_padding(padding, self.dim, self.stride, size)
        self.groups = groups
        self.padding_mode = padding_mode

        config = {"device": device, "dtype": dtype}
        weight_shape = (out_channels, in_channels // groups) + self.kernel_size
        self.weight = nn.Parameter(torch.empty(weight_shape, **config))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **config))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, input):
        has_batch = input.dim() == 2 + self.dim
        input = input if has_batch else input.unsqueeze(0)
        if any(p != 0 for p in self.padding):
            pad = reversed(list(zip(self.padding, self.padding)))
            pad = tuple(item for pair in pad for item in pair)
            mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
            input = nn.functional.pad(input, pad, mode)

        out = conv(
            input,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )
        out = out if has_batch else out.squeeze(0)
        return out

    def reset_parameters(self):
        kernel_prod = math.prod(self.kernel_size)
        bound = math.sqrt(self.groups / (self.in_channels * kernel_prod))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)


class Conv1d(ConvBase):
    dim = 1


class Conv2d(ConvBase):
    dim = 2


class Conv3d(ConvBase):
    dim = 3
