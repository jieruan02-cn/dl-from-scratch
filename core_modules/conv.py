import math
import torch
import torch.nn as nn


def conv1d_input_reshape(input):
    return input


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    (B, C_in, L), C_out, kernel_size = input.shape, weight.size(0), weight.size(-1)
    if groups > 1:
        input = input.view(B, C_in // groups, groups, L).sum(dim=2)
    input = input.unfold
    out = input @ weight.view(C_out, -1).transpose(0, 1)
    out = out.transpose(1, 2)
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
        out = conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        out = out if has_batch else out.squeeze(0)
        return out

    def reset_parameters(self):
        bound = math.sqrt(self.groups / (self.in_channels * self.kernel_size))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
