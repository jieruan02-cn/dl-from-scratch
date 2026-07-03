import math
import torch
import torch.nn as nn


def _canonical_padding1d(padding, stride, window_size):
    if isinstance(padding, int):
        return padding
    elif isinstance(padding, tuple):
        return padding[0]
    elif isinstance(padding, str):
        if padding == "valid":
            return 0
        elif padding == "same":
            assert stride == 1 and (window_size - 1) % 2 == 0
            return (window_size - 1) // 2
        else:
            raise ValueError(f"Expect padding to be valid or same, got {padding}")
    else:
        raise TypeError(f"Expect padding of type int/tuple/str, got {type(padding)}")


def _canonical_padding2d(padding, stride_H, stride_W, size_H, size_W):
    if isinstance(padding, int):
        return padding, padding
    elif isinstance(padding, tuple):
        return padding
    elif isinstance(padding, str):
        if padding == "valid":
            return 0, 0
        elif padding == "same":
            assert stride_H == 1 and stride_W == 1
            assert (size_H - 1) % 2 == 0 and (size_W - 1) % 2 == 0
            return (size_H - 1) // 2, (size_W - 1) // 2
        else:
            raise ValueError(f"Expect padding to be valid or same, got {padding}")
    else:
        raise TypeError(f"Expect padding of type int/tuple/str, got {type(padding)}")


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    B, C_in, L_in = input.shape
    C_out, kernel_size = weight.size(0), weight.size(-1)
    input = input.view(B, groups, C_in // groups, L_in)
    window_size = (kernel_size - 1) * dilation + 1
    padding = _canonical_padding1d(padding, stride, window_size)
    if padding != 0:
        input = nn.functional.pad(input, (padding, padding), mode="constant", value=0)
    # need to use tensor.unfold as nn.Unfold or nn.functional.unfold only support 4D.
    input = input.unfold(-1, window_size, stride).transpose(-2, -3)
    input = input[:, :, :, :, 0:window_size:dilation].reshape(input.shape[:3] + (-1,))

    out = input @ weight.view(groups, C_out // groups, -1).transpose(1, 2)
    out = out.transpose(2, 3).reshape(B, C_out, -1)
    if bias is not None:
        out = out + bias[None, :, None]
    return out


class Conv1d(nn.Module):
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
        if in_channels % groups != 0:
            raise ValueError(
                f"Expect in_channels % groups == 0, got in_channels = {in_channels}, groups = {groups}"
            )
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        config = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels // groups, kernel_size), **config)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels,), **config))
        self.reset_parameters()

    def forward(self, input):
        has_batch = input.dim() == 3
        input = input if has_batch else input.unsqueeze(0)
        padding = _canonical_padding1d(
            self.padding, self.stride, (self.kernel_size - 1) * self.dilation + 1
        )
        if padding != 0:
            assert self.padding_mode in ("zeros", "reflect", "replicate", "circular")
            padding_mode = (
                "constant" if self.padding_mode == "zeros" else self.padding_mode
            )
            input = nn.functional.pad(input, (padding, padding), padding_mode)

        out = conv1d(
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
        bound = math.sqrt(self.groups / (self.in_channels * self.kernel_size))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    B, C_in, H, W = input.shape
    C_out, C_in_per_group, kernel_H, kernel_W = weight.shape
    dilation_H, dilation_W = (
        (dilation, dilation) if isinstance(dilation, int) else dilation
    )
    size_H, size_W = (kernel_H - 1) * dilation_H + 1, (kernel_W - 1) * dilation_W + 1
    stride_H, stride_W = (stride, stride) if isinstance(stride, int) else stride
    padding_H, padding_W = _canonical_padding2d(
        padding, stride_H, stride_W, size_H, size_W
    )
    if padding_H != 0 or padding_W != 0:
        input = nn.functional.pad(input, (padding_W, padding_W, padding_H, padding_H))

    input = input.view(B, groups, C_in_per_group, *input.shape[2:])
    input = input.unfold(3, size_H, stride_H).unfold(4, size_W, stride_W)
    input = input[:, :, :, :, :, 0:size_H:dilation_H, 0:size_W:dilation_W]
    L_out_H, L_out_W = input.shape[3:5]
    input = input.permute(0, 1, 3, 4, 2, 5, 6).reshape(B, groups, L_out_H * L_out_W, -1)

    out = input @ weight.view(groups, C_out // groups, -1).transpose(1, 2)
    out = out.transpose(-1, -2).reshape(B, C_out, L_out_H, L_out_W)
    if bias is not None:
        out = out + bias[None, :, None, None]
    return out


class Conv2d(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
