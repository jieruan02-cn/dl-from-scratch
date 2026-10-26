import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# loss.py imports its siblings as `from activation import ...`, so core_modules/
# itself has to be on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "core_modules"))

from loss import (  # noqa: E402
    GaussianNLLLoss,
    PoissonNLLLoss,
    gaussian_nll_loss,
    poisson_nll_loss,
)


def _grads(fn, input, target, var, **kwargs):
    input = input.clone().requires_grad_()
    var = var.clone().requires_grad_()
    out = fn(input, target, var, **kwargs)
    out.sum().backward()
    return out.detach(), input.grad, var.grad


def _assert_matches_torch(input, target, var, **kwargs):
    out, grad_input, grad_var = _grads(gaussian_nll_loss, input, target, var, **kwargs)
    ref, ref_grad_input, ref_grad_var = _grads(
        F.gaussian_nll_loss, input, target, var, **kwargs
    )
    assert out.shape == ref.shape
    assert torch.allclose(out, ref)
    assert torch.allclose(grad_input, ref_grad_input)
    assert grad_var is not None
    assert torch.allclose(grad_var, ref_grad_var)


# ── matches torch ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_gaussian_nll_heteroscedastic_matches_torch(reduction, full):
    torch.manual_seed(0)
    input = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    var = torch.rand(4, 3, dtype=torch.float64) + 0.1
    _assert_matches_torch(input, target, var, reduction=reduction, full=full)


@pytest.mark.parametrize(
    "input_shape, var_shape",
    [
        ((4, 3), (4, 1)),  # homoscedastic, explicit trailing 1
        ((4, 3), (4,)),  # one dim short, N != D
        ((3, 3), (3,)),  # one dim short, N == D: broadcasting alone would be wrong
        ((2, 4, 3), (2, 4)),  # one dim short, 3-D input
    ],
)
def test_gaussian_nll_homoscedastic_var_shapes(input_shape, var_shape):
    torch.manual_seed(0)
    input = torch.randn(*input_shape, dtype=torch.float64)
    target = torch.randn(*input_shape, dtype=torch.float64)
    var = torch.rand(*var_shape, dtype=torch.float64) + 0.1
    _assert_matches_torch(input, target, var, reduction="none")


def test_gaussian_nll_float_var_matches_torch():
    torch.manual_seed(0)
    input = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    out = gaussian_nll_loss(input, target, 0.1, reduction="none")
    ref = F.gaussian_nll_loss(input, target, 0.1, reduction="none")
    assert torch.allclose(out, ref)


def test_gaussian_nll_float_var_keeps_input_dtype():
    input = torch.randn(4, 3)
    target = torch.randn(4, 3)
    assert gaussian_nll_loss(input, target, 0.5).dtype == torch.float32


# ── clamp: straight-through gradient, caller's var untouched ─────────────────


def test_gaussian_nll_var_below_eps_gradient_passes_through():
    # A tracked clamp would give zero gradient here; torch's clamp is
    # straight-through, so the gradient is that of the unclamped formula.
    torch.manual_seed(0)
    input = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    var = torch.tensor([0.0, 1e-9, 0.5], dtype=torch.float64).expand(4, 3).clone()
    _assert_matches_torch(input, target, var, reduction="none")


def test_gaussian_nll_does_not_mutate_var():
    input = torch.randn(4, 3)
    target = torch.randn(4, 3)
    var = torch.tensor([0.0, 1e-9, 0.5]).expand(4, 3).clone()
    original = var.clone()
    gaussian_nll_loss(input, target, var)
    assert torch.equal(var, original)


def test_gaussian_nll_backward_through_exp_head():
    # exp saves its output for backward, so clamping that output in place
    # would trip autograd's version counter.
    input = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    log_var = torch.full((4, 3), -20.0, dtype=torch.float64, requires_grad=True)
    gaussian_nll_loss(input, target, torch.exp(log_var)).backward()
    assert log_var.grad is not None
    assert torch.all(log_var.grad != 0)


def test_gaussian_nll_gradcheck():
    torch.manual_seed(0)
    input = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 3, dtype=torch.float64)
    var = (torch.rand(4, 3, dtype=torch.float64) + 0.1).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda i, v: gaussian_nll_loss(i, target, v), (input, var)
    )


# ── invalid inputs ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("var", [-0.1, torch.tensor([[0.5, -0.1, 0.5]] * 4)])
def test_gaussian_nll_negative_var_raises(var):
    with pytest.raises(ValueError):
        gaussian_nll_loss(torch.randn(4, 3), torch.randn(4, 3), var)


def test_gaussian_nll_invalid_reduction_raises():
    with pytest.raises(ValueError):
        gaussian_nll_loss(torch.randn(4, 3), torch.randn(4, 3), 0.5, reduction="avg")


@pytest.mark.xfail(
    strict=True,
    reason="var shape is only checked by ndim; torch raises for these shapes",
)
@pytest.mark.parametrize("var_shape", [(4, 3, 1), (2, 3)])
def test_gaussian_nll_wrong_var_shape_raises(var_shape):
    with pytest.raises(ValueError):
        gaussian_nll_loss(torch.randn(4, 3), torch.randn(4, 3), torch.ones(var_shape))


# ── module ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_gaussian_nll_module_matches_torch(reduction, full):
    torch.manual_seed(0)
    input = torch.randn(4, 3)
    target = torch.randn(4, 3)
    var = torch.rand(4, 3) + 0.1
    out = GaussianNLLLoss(full=full, reduction=reduction)(input, target, var)
    ref = nn.GaussianNLLLoss(full=full, reduction=reduction)(input, target, var)
    assert torch.allclose(out, ref)


# ── poisson nll: matches torch ────────────────────────────────────────────────


def _poisson_grads(fn, input, target, **kwargs):
    input = input.clone().requires_grad_()
    out = fn(input, target, **kwargs)
    out.sum().backward()
    return out.detach(), input.grad


def _assert_poisson_matches_torch(input, target, **kwargs):
    out, grad = _poisson_grads(poisson_nll_loss, input, target, **kwargs)
    ref, ref_grad = _poisson_grads(F.poisson_nll_loss, input, target, **kwargs)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref)
    assert torch.allclose(grad, ref_grad)


def _counts(*shape):
    # Poisson samples with rate 2 include plenty of 0s and 1s.
    return torch.poisson(torch.full(shape, 2.0, dtype=torch.float64))


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("log_input", [True, False])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_poisson_nll_matches_torch(reduction, log_input, full):
    torch.manual_seed(0)
    if log_input:
        input = torch.randn(4, 3, dtype=torch.float64)
    else:
        input = torch.rand(4, 3, dtype=torch.float64) + 0.1
    target = _counts(4, 3)
    target[0, :2] = torch.tensor([0.0, 1.0])  # full=True must skip these
    _assert_poisson_matches_torch(
        input, target, log_input=log_input, full=full, reduction=reduction
    )


def test_poisson_nll_unbatched_and_broadcast_target():
    torch.manual_seed(0)
    _assert_poisson_matches_torch(torch.randn(5, dtype=torch.float64), _counts(5))
    _assert_poisson_matches_torch(
        torch.randn(4, 3, dtype=torch.float64), _counts(3), reduction="none"
    )


# ── poisson nll: full, eps ────────────────────────────────────────────────────


def test_poisson_nll_full_stirling_only_for_target_above_one():
    # log(y!) is exactly 0 for y = 0 and y = 1, and Stirling's approximation is
    # poor below 1, so torch adds the term only where target > 1.
    input = torch.zeros(5, dtype=torch.float64)
    target = torch.tensor([0.0, 0.5, 1.0, 1.5, 3.0], dtype=torch.float64)
    base = poisson_nll_loss(input, target, reduction="none")
    out = poisson_nll_loss(input, target, full=True, reduction="none")
    assert torch.equal(out[:3], base[:3])
    assert torch.all(out[3:] > base[3:])
    _assert_poisson_matches_torch(input, target, full=True, reduction="none")


def test_poisson_nll_full_zero_target_is_finite():
    input = torch.randn(4, 3)
    target = torch.zeros(4, 3)
    assert torch.isfinite(poisson_nll_loss(input, target, full=True))


def test_poisson_nll_rate_input_below_eps():
    # With log_input=False the loss takes log(input + eps), so input = 0 stays
    # finite and its gradient matches torch.
    input = torch.tensor([0.0, 1e-12, 1.0], dtype=torch.float64)
    target = torch.ones(3, dtype=torch.float64)
    out, grad = _poisson_grads(
        poisson_nll_loss, input, target, log_input=False, reduction="none"
    )
    assert torch.all(torch.isfinite(out)) and torch.all(torch.isfinite(grad))
    _assert_poisson_matches_torch(input, target, log_input=False, reduction="none")


@pytest.mark.parametrize("log_input", [True, False])
def test_poisson_nll_gradcheck(log_input):
    torch.manual_seed(0)
    if log_input:
        input = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    else:
        input = (torch.rand(4, 3, dtype=torch.float64) + 0.1).requires_grad_()
    target = _counts(4, 3)
    assert torch.autograd.gradcheck(
        lambda i: poisson_nll_loss(i, target, log_input=log_input, full=True),
        (input,),
    )


def test_poisson_nll_invalid_reduction_raises():
    with pytest.raises(ValueError):
        poisson_nll_loss(torch.randn(4, 3), _counts(4, 3), reduction="avg")


# ── poisson nll: module ───────────────────────────────────────────────────────


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("log_input", [True, False])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_poisson_nll_module_matches_torch(reduction, log_input, full):
    torch.manual_seed(0)
    input = torch.rand(4, 3) + 0.1
    target = torch.poisson(torch.full((4, 3), 2.0))
    kwargs = dict(log_input=log_input, full=full, reduction=reduction)
    out = PoissonNLLLoss(**kwargs)(input, target)
    ref = nn.PoissonNLLLoss(**kwargs)(input, target)
    assert torch.allclose(out, ref)
