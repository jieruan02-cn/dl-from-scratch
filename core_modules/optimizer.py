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
