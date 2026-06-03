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
    for i, (p, d_p, momentum_buf) in enumerate(
        zip(params, d_p_list, momentum_buffer_list)
    ):
        g = -d_p if maximize else d_p
        if weight_decay != 0.0:
            g = g + weight_decay * p
        if momentum != 0.0:
            if momentum_buf is None:
                b = g.clone()
                momentum_buffer_list[i] = b
            else:
                momentum_buf.mul_(momentum).add_(g, alpha=1 - dampening)
                b = momentum_buf
            if nesterov:
                g = g + momentum * b
            else:
                g = b
        p.sub_(lr * g)


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

        for group in self.param_groups:
            with torch.set_grad_enabled(group["differentiable"]):
                params_with_grad = []
                d_p_list = []
                momentum_buffer_list = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    momentum_buffer_list.append(self.state[p].get("b", None))
                sgd(
                    params_with_grad,
                    d_p_list,
                    momentum_buffer_list,
                    foreach=group["foreach"],
                    fused=group["fused"],
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    lr=group["lr"],
                    dampening=group["dampening"],
                    nesterov=group["nesterov"],
                    maximize=group["maximize"],
                )

                for p, buf in zip(group["params"], momentum_buffer_list):
                    if buf is not None:
                        self.state[p]["b"] = buf

        return loss
