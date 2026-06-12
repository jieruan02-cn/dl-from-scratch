import math

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
        # First call is non-inplace to avoid corrupt p.grad, the autograd-owned memory.
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
        foreach = False

    if foreach:
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

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
    maximize,
    lr,
    alpha,
    eps,
    weight_decay,
    momentum,
    centered,
):
    lr = _to_scalar(lr)

    for i, param in enumerate(params):
        # detach() is wrong here because if differentiable is False, then no need for it,
        # if differentiable is True, it violates user intent; clone() is not needed as no
        # inplace modification of grad.
        grad = -grads[i] if maximize else grads[i]

        if weight_decay != 0.0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg = square_avgs[i]
        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        local_square_avg = square_avg
        if centered:
            grad_avg = grad_avgs[i]
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            local_square_avg = square_avg - grad_avg * grad_avg

        avg = torch.sqrt(local_square_avg).add_(eps)
        if momentum > 0.0:
            buf = momentum_buffer_list[i]
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr)
        else:
            param.addcdiv_(grad, avg, value=-lr)


def _multi_tensor_rmsprop(
    params,
    grads,
    square_avgs,
    grad_avgs,
    momentum_buffer_list,
    maximize,
    lr,
    alpha,
    eps,
    weight_decay,
    momentum,
    centered,
):
    lr = _to_scalar(lr)

    tensor_lists = [params, grads, square_avgs]
    if centered:
        tensor_lists.append(grad_avgs)
    if momentum > 0.0:
        tensor_lists.append(momentum_buffer_list)
    grouped_tensors = torch.optim.Optimizer._group_tensors_by_device_and_dtype(
        tensor_lists
    )

    for (
        device_params,
        device_grads,
        device_square_avgs,
        *rest,
    ), _ in grouped_tensors.values():
        index = 0
        if centered:
            device_grad_avgs = rest[index]
            index += 1
        if momentum > 0.0:
            device_momentum_buffer_list = rest[index]

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if weight_decay != 0.0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(
                    device_grads, device_params, alpha=weight_decay
                )

        torch._foreach_mul_(device_square_avgs, alpha)
        torch._foreach_addcmul_(
            device_square_avgs, device_grads, device_grads, 1 - alpha
        )

        local_square_avgs = device_square_avgs
        if centered:
            torch._foreach_mul_(device_grad_avgs, alpha)
            torch._foreach_add_(device_grad_avgs, device_grads, alpha=1 - alpha)
            local_square_avgs = torch._foreach_sub(
                device_square_avgs,
                torch._foreach_mul(device_grad_avgs, device_grad_avgs),
            )

        avg = torch._foreach_add_(torch._foreach_sqrt(local_square_avgs), eps)
        if momentum > 0.0:
            torch._foreach_mul_(device_momentum_buffer_list, momentum)
            torch._foreach_addcdiv_(device_momentum_buffer_list, device_grads, avg)
            torch._foreach_add_(device_params, device_momentum_buffer_list, alpha=-lr)
        else:
            torch._foreach_addcdiv_(device_params, device_grads, avg, -lr)


def rmsprop(
    params,
    grads,
    square_avgs,
    grad_avgs,
    momentum_buffer_list,
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
        foreach = False

    if foreach:
        func = _multi_tensor_rmsprop
    else:
        func = _single_tensor_rmsprop

    func(
        params,
        grads,
        square_avgs,
        grad_avgs,
        momentum_buffer_list,
        maximize,
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
                has_complex = False
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if not has_complex and torch.is_complex(p):
                        has_complex = True
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state["square_avg"] = torch.zeros_like(p)
                        if group["centered"]:
                            state["grad_avg"] = torch.zeros_like(p)
                        if group["momentum"]:
                            state["momentum_buffer"] = torch.zeros_like(p)
                    square_avgs.append(state["square_avg"])
                    grad_avgs.append(state["grad_avg"] if group["centered"] else None)
                    momentum_buffer_list.append(
                        state["momentum_buffer"] if group["momentum"] else None
                    )

                rmsprop(
                    params_with_grad,
                    grads,
                    square_avgs,
                    grad_avgs,
                    momentum_buffer_list,
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

                # Redundant if all mutates in place as list holds the reference to state.
                for p, square_avg, grad_avg, momentum_buf in zip(
                    params_with_grad,
                    square_avgs,
                    grad_avgs,
                    momentum_buffer_list,
                ):
                    state = self.state[p]
                    if square_avg is not None:
                        state["square_avg"] = square_avg
                    if grad_avg is not None:
                        state["grad_avg"] = grad_avg
                    if momentum_buf is not None:
                        state["momentum_buffer"] = momentum_buf

        return loss


def _single_tensor_adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    decoupled_weight_decay,
    amsgrad,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
    maximize,
):
    lr = _to_scalar(lr)
    for i, param in enumerate(params):
        grad = -grads[i] if maximize else grads[i]
        if weight_decay != 0.0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        t = state_steps[i].add_(1).item()
        exp_avg = exp_avgs[i]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]
            max_exp_avg_sq.clamp_(min=exp_avg_sq)
            normed_exp_avg_sq = max_exp_avg_sq.div(1 - beta2**t)
        else:
            normed_exp_avg_sq = exp_avg_sq.div(1 - beta2**t)
        normed_exp_avg_sq.sqrt_().add_(eps)

        # ALU compute optimization to divide the (1 - beta1**t) first
        param.addcdiv_(exp_avg, normed_exp_avg_sq, value=-lr / (1 - beta1**t))


def _multi_tensor_adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    decoupled_weight_decay,
    amsgrad,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
    maximize,
):
    lr = _to_scalar(lr)
    tensor_lists = [params, grads, exp_avgs, exp_avg_sqs]
    # Passing list of None will raise in _group_tensors_by_device_and_dtype, so append
    # conditionally.
    if amsgrad:
        tensor_lists.append(max_exp_avg_sqs)
    grouped_tensors = torch.optim.Optimizer._group_tensors_by_device_and_dtype(
        tensor_lists, with_indices=True
    )
    for (
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        *rest,
    ), indices in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if weight_decay != 0.0:
            if decoupled_weight_decay:
                torch._foreach_mul_(device_params, 1 - lr * weight_decay)
            elif maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(
                    device_grads, device_params, alpha=weight_decay
                )

        torch._foreach_mul_(device_exp_avgs, beta1)
        torch._foreach_add_(device_exp_avgs, device_grads, alpha=1 - beta1)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, 1 - beta2
        )

        step_sizes = []
        bias_correction2_sqrts = []
        for i in indices:
            step_t = state_steps[i]
            step_t.add_(1.0)
            t = step_t.item()
            step_sizes.append(-lr / (1 - beta1**t))
            bias_correction2_sqrts.append(math.sqrt(1 - beta2**t))
        if amsgrad:
            device_max_exp_avg_sqs = rest[0]
            torch._foreach_clamp_min_(device_max_exp_avg_sqs, device_exp_avg_sqs)
            normed_exp_avg_sqs = torch._foreach_sqrt(device_max_exp_avg_sqs)
        else:
            normed_exp_avg_sqs = torch._foreach_sqrt(device_exp_avg_sqs)
        torch._foreach_div_(normed_exp_avg_sqs, bias_correction2_sqrts)
        torch._foreach_add_(normed_exp_avg_sqs, eps)

        torch._foreach_addcdiv_(
            device_params, device_exp_avgs, normed_exp_avg_sqs, step_sizes
        )


def adam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    foreach=None,
    capturable=False,
    differentiable=False,
    fused=None,
    grad_scale=None,
    found_inf=None,
    has_complex=False,
    decoupled_weight_decay=False,
    *,
    amsgrad,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
    maximize,
):
    if fused is not None:
        raise ValueError("fused adam not supported yet.")
    if foreach is None:
        foreach = False

    if foreach:
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    return func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        decoupled_weight_decay,
        amsgrad,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
        maximize,
    )


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
        *,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
        decoupled_weight_decay=False,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "foreach": foreach,
            "maximize": maximize,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": fused,
            "decoupled_weight_decay": decoupled_weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            with torch.set_grad_enabled(group["differentiable"]):
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []
                grad_scale, has_complex, found_inf = None, False, False
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if not has_complex and torch.is_complex(p):
                        has_complex = True
                    if not found_inf and torch.isinf(p.grad).any():
                        found_inf = True
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avgs"] = torch.zeros_like(p)
                        state["exp_avg_sqs"] = torch.zeros_like(p)
                        state["max_exp_avg_sqs"] = (
                            torch.zeros_like(p) if group["amsgrad"] else None
                        )
                        # FORCE CPU: This guarantees .item() will be instantaneous
                        state["state_steps"] = torch.tensor(0.0)

                    exp_avgs.append(state["exp_avgs"])
                    exp_avg_sqs.append(state["exp_avg_sqs"])
                    max_exp_avg_sqs.append(state["max_exp_avg_sqs"])
                    state_steps.append(state["state_steps"])

                adam(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    foreach=group["foreach"],
                    capturable=group["capturable"],
                    differentiable=group["differentiable"],
                    fused=group["fused"],
                    grad_scale=grad_scale,
                    has_complex=has_complex,
                    decoupled_weight_decay=group["decoupled_weight_decay"],
                    amsgrad=group["amsgrad"],
                    beta1=group["betas"][0],
                    beta2=group["betas"][1],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                )

        return loss


def adamw(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    foreach=None,
    capturable=False,
    differentiable=False,
    fused=None,
    grad_scale=None,
    found_inf=None,
    has_complex=False,
    *,
    amsgrad,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
    maximize,
):
    return adam(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        foreach=foreach,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
        decoupled_weight_decay=True,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )


class AdamW(Adam):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
        *,
        maximize=False,
        foreach=None,
        capturable=False,
        differentiable=False,
        fused=None,
    ):
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=True,
        )


def _single_tensor_adagrad(
    params, grads, state_sums, state_steps, lr, weight_decay, lr_decay, eps, maximize
):
    lr = _to_scalar(lr)
    for param, grad, state_sum, state_step in zip(
        params, grads, state_sums, state_steps
    ):
        if maximize:
            grad = -grad
        if weight_decay != 0.0:
            if maximize:
                grad.add_(param, alpha=weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)
        state_sum.addcmul_(grad, grad)

        step_t = state_step.add_(1).item()
        step_size = -lr / (1 + (step_t - 1) * lr_decay)
        param.addcdiv_(grad, state_sum.sqrt().add_(eps), value=step_size)


def _multi_tensor_adagrad(
    params, grads, state_sums, state_steps, lr, weight_decay, lr_decay, eps, maximize
):
    lr = _to_scalar(lr)
    group_tensors = torch.optim.Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums], with_indices=True
    )
    for (
        device_params,
        device_grads,
        device_state_sums,
    ), indices in group_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)
        if weight_decay != 0.0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(
                    device_grads, device_params, alpha=weight_decay
                )
        torch._foreach_addcmul_(device_state_sums, device_grads, device_grads)

        step_sizes = []
        for i in indices:
            step_t = state_steps[i]
            step_t.add_(1.0)
            step_sizes.append(-lr / (1 + (step_t.item() - 1) * lr_decay))
        denom = torch._foreach_sqrt(device_state_sums)
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(device_params, device_grads, denom, step_sizes)


def adagrad(
    params,
    grads,
    state_sums,
    state_steps,
    fused=None,
    grad_scale=None,
    found_inf=None,
    has_sparse_grad=False,
    foreach=None,
    differentiable=False,
    has_complex=False,
    *,
    lr,
    weight_decay,
    lr_decay,
    eps,
    maximize,
):
    if fused is not None:
        raise ValueError("fused adagrad not supported yet.")
    if foreach is None:
        foreach = False

    if foreach:
        func = _multi_tensor_adagrad
    else:
        func = _single_tensor_adagrad

    func(
        params,
        grads,
        state_sums,
        state_steps,
        lr,
        weight_decay,
        lr_decay,
        eps,
        maximize,
    )


class Adagrad(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach=None,
        *,
        maximize=False,
        differentiable=False,
        fused=None,
    ):
        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "eps": eps,
            "foreach": foreach,
            "maximize": maximize,
            "differentiable": differentiable,
            "fused": fused,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            with torch.set_grad_enabled(group["differentiable"]):
                params_with_grad = []
                grads = []
                state_sums = []
                state_steps = []
                grad_scale, found_inf, has_sparse_grad, has_complex = (
                    None,
                    None,
                    False,
                    False,
                )
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if torch.isinf(p.grad).any():
                        found_inf = True
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    if torch.is_complex(p):
                        has_complex = True
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]

                    if len(state) == 0:
                        state["state_sums"] = torch.full_like(
                            p, group["initial_accumulator_value"]
                        )
                        state["state_steps"] = torch.tensor(0.0)
                    state_sums.append(state["state_sums"])
                    state_steps.append(state["state_steps"])

                adagrad(
                    params_with_grad,
                    grads,
                    state_sums,
                    state_steps,
                    group["fused"],
                    grad_scale,
                    found_inf,
                    has_sparse_grad,
                    group["foreach"],
                    group["differentiable"],
                    has_complex,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    lr_decay=group["lr_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                )

        return loss


def rms(x):
    return x.pow(2).mean().sqrt_()


class Adafactor(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta2_decay=-0.8,
        eps=(None, 0.001),
        d=1.0,
        weight_decay=0.0,
        *,
        foreach=None,
        maximize=False,
    ):
        defaults = {
            "lr": lr,
            "beta2_decay": beta2_decay,
            "eps": eps,
            "d": d,
            "weight_decay": weight_decay,
            "foreach": foreach,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = _to_scalar(group["lr"])
            beta2_decay = group["beta2_decay"]
            eps1, eps2 = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if eps1 is None:
                    eps1 = torch.finfo(p.dtype).eps

                config = {"device": p.device, "dtype": p.dtype}
                state = self.state[p]
                if len(state) == 0:
                    state["state_step"] = torch.tensor(0.0)
                    if p.dim() > 1:
                        state["row_factor"] = torch.zeros(
                            p.shape[:-2] + (1, p.shape[-1]), **config
                        )
                        state["col_factor"] = torch.zeros(p.shape[:-1] + (1,), **config)
                    else:
                        state["second_moment"] = torch.zeros_like(p)

                state_step = state["state_step"]
                if p.dim() > 1:
                    row_factor = state["row_factor"]
                    col_factor = state["col_factor"]
                else:
                    second_moment = state["second_moment"]

                step_t = state_step.add_(1.0).item()
                beta2, rho = 1 - step_t**beta2_decay, min(lr, 1 / math.sqrt(step_t))
                alpha = rho * rms(p).clamp(min=eps2)

                grad = -p.grad if maximize else p.grad.clone()
                p.mul_(1 - lr * weight_decay)
                if p.dim() > 1:
                    grad_prod = grad * grad
                    row_factor.mul_(beta2).add_(
                        grad_prod.sum(dim=-2, keepdim=True), alpha=1 - beta2
                    )
                    col_factor.mul_(beta2).add_(
                        grad_prod.sum(dim=-1, keepdim=True), alpha=1 - beta2
                    )
                    grad.div_(row_factor.sqrt().clamp_(min=eps1))
                    grad.div_(col_factor.sqrt().clamp_(min=eps1))
                    grad.mul_(row_factor.sum(-1, keepdim=True).clamp_(min=eps1).sqrt_())
                else:
                    second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    grad.div_(second_moment.sqrt().clamp_(min=eps1))

                # calling .item() on the rms(grad) tensor will force a CPU-GPU
                # synchronization, need to work on tensor space
                grad.mul_(-alpha / (rms(grad) / group["d"]).clamp_(min=1.0))
                p.add_(grad)

        return loss


def newton_schulz(a, b, c, eps, k, x):
    # x = x.to(torch.bfloat16) in real impl for speed, here keep fp32 for precision.
    x = x / torch.linalg.matrix_norm(x, ord="fro").clamp(min=eps)
    for _ in range(k):
        if x.shape[-2] > x.shape[-1]:
            cov = x.T @ x
            x = x @ torch.addmm(cov, cov, cov, beta=b, alpha=c).add_(a)
        else:
            cov = x @ x.T
            x = torch.addmm(cov, cov, cov, beta=b, alpha=c).add_(a) @ x
    return x


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-07,
        ns_steps=5,
        adjust_lr_fn=None,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        if any(p.dim() != 2 for group in self.param_groups for p in group["params"]):
            raise ValueError("only accept param of 2D.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            a, b, c = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(param)
                momentum_buf = state["momentum_buffer"]
                momentum_buf.mul_(momentum).add_(param.grad)
                if nesterov:
                    nesterov_momentum = momentum_buf.mul(momentum).add_(param.grad)
                else:
                    nesterov_momentum = momentum_buf
                orthogonal_momentum = newton_schulz(
                    a, b, c, eps, ns_steps, nesterov_momentum
                )

                lr = _to_scalar(group["lr"])
                param.mul_(1 - lr * weight_decay)

                if adjust_lr_fn is not None:
                    lr = adjust_lr_fn(lr, param.shape)
                else:
                    lr = lr * math.sqrt(max(1.0, param.shape[-2] / param.shape[-1]))
                param.add_(orthogonal_momentum, alpha=-lr)

        return loss
