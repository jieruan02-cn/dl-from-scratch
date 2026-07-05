import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_modules.conv import (
    conv,
    conv1d,
    conv2d,
    conv3d,
    Conv1d,
    Conv2d,
    Conv3d,
)

# Per-dim references and our implementations.
F_CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
NN_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
OUR_FUNC = {1: conv1d, 2: conv2d, 3: conv3d}
OUR_MOD = {1: Conv1d, 2: Conv2d, 3: Conv3d}

# Distinct spatial sizes per axis so axis-order mistakes are caught.
SPATIAL = {1: (16,), 2: (14, 16), 3: (8, 10, 12)}

CIN, COUT = 4, 6
ATOL = 1e-5


def _kernel(dim, k):
    return k if isinstance(k, tuple) else (k,) * dim


def _make(dim, groups, bias, kernel=3, batch=2, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, CIN, *SPATIAL[dim])
    w = torch.randn(COUT, CIN // groups, *_kernel(dim, kernel))
    b = torch.randn(COUT) if bias else None
    return x, w, b


# --------------------------------------------------------------------------
# Functional: conv / conv1d / conv2d / conv3d  vs  F.conv{1,2,3}d
# --------------------------------------------------------------------------

# Scalar configs apply uniformly across all three dims.
FUNC_CONFIGS = [
    dict(),
    dict(stride=2),
    dict(padding=1),
    dict(padding="same"),
    dict(padding="valid"),
    dict(dilation=2),
    dict(groups=2),
    dict(bias=False),
    dict(stride=2, padding=1, dilation=2, groups=2),
]


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("cfg", FUNC_CONFIGS)
def test_functional_matches_torch(dim, cfg):
    cfg = dict(cfg)
    groups = cfg.pop("groups", 1)
    bias = cfg.pop("bias", True)
    x, w, b = _make(dim, groups, bias)
    out = OUR_FUNC[dim](x, w, bias=b, groups=groups, **cfg)
    ref = F_CONV[dim](x, w, bias=b, groups=groups, **cfg)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=ATOL)


@pytest.mark.parametrize(
    "dim,cfg",
    [
        (2, dict(stride=(2, 1))),
        (2, dict(padding=(1, 2))),
        (2, dict(dilation=(2, 1))),
        (2, dict(kernel=(3, 5))),
        (3, dict(stride=(2, 1, 2))),
        (3, dict(kernel=(3, 5, 3))),
    ],
)
def test_functional_per_axis_params(dim, cfg):
    cfg = dict(cfg)
    kernel = cfg.pop("kernel", 3)
    x, w, b = _make(dim, groups=1, bias=True, kernel=kernel)
    out = OUR_FUNC[dim](x, w, bias=b, **cfg)
    ref = F_CONV[dim](x, w, bias=b, **cfg)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=ATOL)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_delegators_match_generic(dim):
    x, w, b = _make(dim, groups=1, bias=True)
    assert torch.equal(OUR_FUNC[dim](x, w, bias=b), conv(x, w, bias=b))


# --------------------------------------------------------------------------
# Module: Conv{1,2,3}d  vs  nn.Conv{1,2,3}d  (weights copied over)
# --------------------------------------------------------------------------

MODULE_CONFIGS = [
    dict(),
    dict(stride=2),
    dict(padding=1),
    dict(padding="same"),
    dict(padding="valid"),
    dict(dilation=2),
    dict(groups=2),
    dict(bias=False),
    dict(padding=2, padding_mode="reflect"),
    dict(padding=2, padding_mode="replicate"),
    dict(padding=2, padding_mode="circular"),
]


def _copy_weights(ours, ref):
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        if ours.bias is not None:
            ref.bias.copy_(ours.bias)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("cfg", MODULE_CONFIGS)
def test_module_matches_torch(dim, cfg):
    torch.manual_seed(0)
    ours = OUR_MOD[dim](CIN, COUT, 3, **cfg)
    ref = NN_CONV[dim](CIN, COUT, 3, **cfg)
    _copy_weights(ours, ref)
    x = torch.randn(2, CIN, *SPATIAL[dim])
    out, ref_out = ours(x), ref(x)
    assert out.shape == ref_out.shape
    assert torch.allclose(out, ref_out, atol=ATOL)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate", "circular"])
def test_module_unbatched(dim, padding_mode):
    torch.manual_seed(0)
    ours = OUR_MOD[dim](CIN, COUT, 3, padding=2, padding_mode=padding_mode)
    ref = NN_CONV[dim](CIN, COUT, 3, padding=2, padding_mode=padding_mode)
    _copy_weights(ours, ref)
    x = torch.randn(CIN, *SPATIAL[dim])  # no batch dim
    out, ref_out = ours(x), ref(x)
    assert out.dim() == 1 + dim  # stays unbatched
    assert out.shape == ref_out.shape
    assert torch.allclose(out, ref_out, atol=ATOL)


@pytest.mark.parametrize(
    "dim,kernel", [(2, (3, 5)), (3, (3, 5, 3))]
)
def test_module_nonsquare_kernel(dim, kernel):
    torch.manual_seed(0)
    ours = OUR_MOD[dim](CIN, COUT, kernel, groups=2)
    ref = NN_CONV[dim](CIN, COUT, kernel, groups=2)
    _copy_weights(ours, ref)
    x = torch.randn(2, CIN, *SPATIAL[dim])
    assert torch.allclose(ours(x), ref(x), atol=ATOL)


def test_module_no_bias_registers_none():
    assert Conv2d(CIN, COUT, 3, bias=False).bias is None


# --------------------------------------------------------------------------
# Validation guards
# --------------------------------------------------------------------------

def test_raises_in_channels_not_divisible_by_groups():
    with pytest.raises(ValueError):
        Conv2d(5, 6, 3, groups=2)


def test_raises_out_channels_not_divisible_by_groups():
    with pytest.raises(ValueError):
        Conv2d(4, 5, 3, groups=2)


def test_raises_bad_padding_mode():
    with pytest.raises(ValueError):
        Conv2d(4, 6, 3, padding_mode="bogus")


def test_raises_kernel_size_wrong_length():
    with pytest.raises(ValueError):
        Conv2d(4, 6, (3, 3, 3))  # 3 entries for a 2d conv


def test_same_padding_requires_stride_one():
    with pytest.raises(AssertionError):
        Conv2d(4, 6, 3, stride=2, padding="same")


def test_same_padding_requires_odd_window():
    with pytest.raises(AssertionError):
        Conv2d(4, 6, 4, padding="same")  # even kernel -> even window


# --------------------------------------------------------------------------
# Autograd smoke test
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_backward_populates_grads(dim):
    torch.manual_seed(0)
    m = OUR_MOD[dim](CIN, COUT, 3, padding=1)
    x = torch.randn(2, CIN, *SPATIAL[dim], requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert m.weight.grad is not None
