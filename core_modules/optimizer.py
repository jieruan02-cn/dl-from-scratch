from typing import cast

import torch
from collections import defaultdict


class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        params = list(params)
        if isinstance(params[0], torch.Tensor):
            self.param_groups = [{"params": params}]
        else:
            self.param_groups = params
        for group in self.param_groups:
            # group = group | defaults is incorrect as it only rebinds the reference
            for k, v in defaults.items():
                group.setdefault(k, v)
        self.state = defaultdict(dict)

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        raise NotImplementedError

    def state_dict(self):
        return self.state


def _to_scalar(x: float | torch.Tensor):
    if isinstance(x, torch.Tensor) and x.dim() != 0:
        return x.squeeze()
    else:
        return x


def _single_tensor_sgd(
    params,
    grads,
    momentum_buffer_list,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
):
    lr = _to_scalar(lr)
    for i, param in enumerate(params):
        grad = -grads[i] if maximize else grads[i]

        if weight_decay != 0.0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0.0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = grad.detach().clone()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf
        param.add_(grad, alpha=-lr)


def _multi_tensor_sgd(
    params,
    grads,
    momentum_buffer_list,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
):
    if len(params) == 0:
        return
    lr = _to_scalar(lr)

    grouped_tensors = torch.optim.Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=True
    )
    for (
        device_params,
        device_grads,
        device_momentum_buffer_list,
    ), indices in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if weight_decay != 0.0:
            # Reuse the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(
                    device_grads, device_params, alpha=weight_decay
                )

        if momentum != 0:
            bufs: list[torch.Tensor] = []

            all_states_with_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[
                            indices[i]
                        ] = device_grads[i].detach().clone()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)
                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs

        if isinstance(lr, torch.Tensor):
            grads_x_lr = torch._foreach_mul(device_grads, -lr)
            torch._foreach_add_(device_params, grads_x_lr)
        else:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)


def sgd(
    params,
    d_p_list,
    momentum_buffer_list,
    has_sparse_grad=False,
    foreach=None,
    fused=None,
    grad_scale=None,
    found_inf=None,
    *,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
):
    if fused is not None:
        raise ValueError("fused SGD not supported yet.")
    if foreach is None:
        func = _single_tensor_sgd
    else:
        func = _multi_tensor_sgd

    # grad_scale and found_inf is only for fused implementation.
    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr,
        dampening,
        nesterov,
        maximize,
    )


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=False,
        foreach=None,
        differentiable=False,
        fused=None,
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        has_sparse_grad = False
        for group in self.param_groups:
            with torch.set_grad_enabled(group["differentiable"]):
                params_with_grad = []
                d_p_list = []
                momentum_buffer_list = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    momentum_buffer_list.append(self.state[p].get("b", None))
                sgd(
                    params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    has_sparse_grad,
                    foreach=group["foreach"],
                    fused=group["fused"],
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    lr=group["lr"],
                    dampening=group["dampening"],
                    nesterov=group["nesterov"],
                    maximize=group["maximize"],
                )

                for p, buf in zip(params_with_grad, momentum_buffer_list):
                    if buf is not None:
                        self.state[p]["b"] = buf

        return loss


def _single_tensor_rmsprop(
    params,
    grads,
    square_avgs,
    grad_avgs,
    momentum_buffer_list,
    state_steps,
    maximize,
    differentiable,
    capturable,
    has_complex,
    lr,
    alpha,
    eps,
    weight_decay,
    momentum,
    centered,
):
    pass


def _multi_tensor_rmsprop(
    params,
    grads,
    square_avgs,
    grad_avgs,
    momentum_buffer_list,
    state_steps,
    maximize,
    differentiable,
    capturable,
    has_complex,
    lr,
    alpha,
    eps,
    weight_decay,
    momentum,
    centered,
):
    pass


def rmsprop(
    params,
    grads,
    square_avgs,
    grad_avgs,
    momentum_buffer_list,
    state_steps,
    foreach=None,
    maximize=False,
    differentiable=False,
    capturable=False,
    has_complex=False,
    *,
    lr,
    alpha,
    eps,
    weight_decay,
    momentum,
    centered,
):
    if foreach is None:
        func = _single_tensor_rmsprop
    else:
        func = _multi_tensor_rmsprop

    func(
        params,
        grads,
        square_avgs,
        grad_avgs,
        momentum_buffer_list,
        state_steps,
        maximize,
        differentiable,
        capturable,
        has_complex,
        lr,
        alpha,
        eps,
        weight_decay,
        momentum,
        centered,
    )


class RMSprop(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        alpha=0.99,
        eps=1e-08,
        weight_decay=0,
        momentum=0,
        centered=False,
        capturable=False,
        foreach=None,
        maximize=False,
        differentiable=False,
    ):
        defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "centered": centered,
            "capturable": capturable,
            "foreach": foreach,
            "maximize": maximize,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            with torch.set_grad_enabled(group["differentiable"]):
                params_with_grad = []
                grads = []
                square_avgs = []
                grad_avgs = []
                momentum_buffer_list = []
                state_steps = []
                has_complex = False
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if not has_complex and torch.is_complex(p):
                        has_complex = True
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]

                rmsprop(
                    params_with_grad,
                    grads,
                    square_avgs,
                    grad_avgs,
                    momentum_buffer_list,
                    state_steps,
                    foreach=group["foreach"],
                    maximize=group["maximize"],
                    differentiable=group["differentiable"],
                    capturable=group["capturable"],
                    has_complex=has_complex,
                    lr=group["lr"],
                    alpha=group["alpha"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    centered=group["centered"],
                )

        return loss
