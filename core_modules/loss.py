import torch
import torch.nn as nn
from activation import logsigmoid, sigmoid, log_softmax, softmax


def mse_loss(input, target, reduction="mean", weight=None):
    diff = input - target
    out = diff * diff
    if weight is not None:
        out = out * weight
    if reduction == "none":
        return out
    elif reduction == "mean":
        return torch.mean(out)
    elif reduction == "sum":
        return torch.sum(out)
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return mse_loss(input, target, self.reduction)


def l1_loss(input, target, reduction="mean", weight=None):
    out = torch.abs(input - target)
    if weight is not None:
        out = out * weight
    if reduction == "none":
        return out
    elif reduction == "mean":
        return torch.mean(out)
    elif reduction == "sum":
        return torch.sum(out)
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class L1Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return l1_loss(input, target, self.reduction)


# Use customized backward as default backward() will take multiple unnecessary derivative
# on indicator function and increase number of ops.
class HuberLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, target, reduction="mean", delta=1.0, weight=None):
        diff = input - target
        abs_diff = torch.abs(diff)
        out = torch.where(
            abs_diff < delta,
            0.5 * abs_diff * abs_diff,
            delta * (abs_diff - 0.5 * delta),
        )
        if weight is not None:
            out = out * weight

        if reduction == "none":
            return out
        elif reduction == "mean":
            return torch.mean(out)
        elif reduction == "sum":
            return torch.sum(out)
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def setup_context(ctx, inputs, output):
        if any(ctx.needs_input_grad):
            input, target, reduction, delta, weight = inputs
            ctx.reduction = reduction
            ctx.delta = delta
            # A tensor T is alive in memory at backward time iff any one of the following holds a reference:
            # 1. Python user scope — pred = model(x) binds pred as a local.
            # 2. An upstream op's save_for_backward — keeps T alive via its SavedVariable.
            # 3. Our own save_for_backward — the choice we're making.
            # If (1) or (2) already holds, then (3) is free. Otherwise (3) costs T.numel() * dtype_size
            #
            # target is typically saved by 1), thus saving input, target usually wins in
            # memory as if 1) or 2) happens to input (when people do pred = model(x)),
            # but saving diff is clearer in code and predictable in memory usage.
            ctx.save_for_backward(input, target, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, weight = ctx.saved_tensors
        # Use clamp instead of where to save ops more,
        # Option 1: grad_output * torch.where(abs_diff < ctx.delta, diff, torch.sign(diff) * ctx.delta) - 6 kernels
        # Option 2: grad_output * diff.clamp(-ctx.delta, ctx.delta) - 2 kernels as clamp is fused into 1.
        diff = input - target
        grad_input = grad_output * diff.clamp(-ctx.delta, ctx.delta)
        if weight is not None:
            grad_input.mul_(weight)
        if ctx.reduction == "mean":
            grad_input.div_(diff.numel())

        grad_target = -grad_input if ctx.needs_input_grad[1] else None
        return grad_input, grad_target, None, None, None


def huber_loss(input, target, reduction="mean", delta=1.0, weight=None):
    return HuberLossFunction.apply(input, target, reduction, delta, weight)


class HuberLoss(nn.Module):
    def __init__(self, reduction="mean", delta=1.0):
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, input, target):
        return huber_loss(input, target, self.reduction, self.delta)


def smooth_l1_loss(input, target, reduction="mean", beta=1.0):
    if beta == 0:
        return l1_loss(input, target, reduction)
    return huber_loss(input, target, reduction, beta) / beta


class SmoothL1Loss(nn.Module):
    def __init__(self, reduction="mean", beta=1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, input, target):
        return smooth_l1_loss(input, target, self.reduction, self.beta)


def _post_process(grad, weight, reduction):
    if grad is None:
        return grad
    if weight is not None:
        grad.mul_(weight)
    if reduction == "mean":
        grad.div_(grad.numel())
    return grad


class BCELossFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, target, weight, reduction):
        eps = max(torch.finfo(input.dtype).tiny, 1e-100)
        out = -target * torch.log(input.clamp(min=eps)) - (1 - target) * torch.log(
            (1 - input).clamp(min=eps)
        )
        if weight is not None:
            out = out * weight

        if reduction == "none":
            return out
        elif reduction == "mean":
            return torch.mean(out)
        elif reduction == "sum":
            return torch.sum(out)
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def setup_context(ctx, inputs, output):
        if any(ctx.needs_input_grad):
            input, target, weight, reduction = inputs
            ctx.reduction = reduction
            ctx.save_for_backward(input, target, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, weight = ctx.saved_tensors

        # 2. Clamp/gradient inconsistency at the boundary (subtle, possibly intentional).
        eps = max(torch.finfo(input.dtype).tiny, 1e-100)
        clamp_input = input.clamp(min=eps)
        clamp_1minput = (1 - input).clamp(min=eps)
        grad_input = (
            grad_output * (clamp_input - target) / (clamp_input * clamp_1minput)
        )
        grad_input = _post_process(grad_input, weight, ctx.reduction)

        grad_target = None
        if ctx.needs_input_grad[1]:
            grad_target = grad_output * torch.log(clamp_1minput / clamp_input)
            grad_target = _post_process(grad_target, weight, ctx.reduction)
        return grad_input, grad_target, None, None


def binary_cross_entropy(input, target, weight=None, reduction="mean"):
    return BCELossFunction.apply(input, target, weight, reduction)


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return binary_cross_entropy(input, target, self.weight, self.reduction)


class BCEWithLogitsLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, target, weight, reduction, pos_weight):
        out = input * (1 - target)
        log_sig = logsigmoid(input)
        if pos_weight is None:
            out = out - log_sig
        else:
            out = out - (1 - target + pos_weight * target) * log_sig
        if weight is not None:
            out = out * weight

        if reduction == "none":
            return out
        elif reduction == "mean":
            return torch.mean(out)
        elif reduction == "sum":
            return torch.sum(out)
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def setup_context(ctx, inputs, output):
        if any(ctx.needs_input_grad):
            input, target, weight, reduction, pos_weight = inputs
            ctx.reduction = reduction
            ctx.save_for_backward(input, target, weight, pos_weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, weight, pos_weight = ctx.saved_tensors

        if pos_weight is None:
            grad_input = grad_output * (sigmoid(input) - target)
            grad_target = -grad_output * input if ctx.needs_input_grad[1] else None
        else:
            scale = 1 - target + pos_weight * target
            grad_input = grad_output * (scale * sigmoid(input) - pos_weight * target)
            grad_target = (
                grad_output * ((1 - pos_weight) * logsigmoid(input) - input)
                if ctx.needs_input_grad[1]
                else None
            )
        grad_input = _post_process(grad_input, weight, ctx.reduction)
        grad_target = _post_process(grad_target, weight, ctx.reduction)
        return grad_input, grad_target, None, None, None


def binary_cross_entropy_with_logits(
    input, target, weight=None, reduction="mean", pos_weight=None
):
    return BCEWithLogitsLossFunction.apply(input, target, weight, reduction, pos_weight)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, reduction="mean", pos_weight=None):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input, target):
        return binary_cross_entropy_with_logits(
            input, target, self.weight, self.reduction, self.pos_weight
        )


class NLLLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, target, weight, ignore_index, reduction):
        dim = 0 if input.dim() == 1 else 1
        clamped_target = target.clamp(0, input.shape[dim] - 1)
        out = -input.gather(dim, clamped_target.unsqueeze(dim)).squeeze(dim)
        target_weight = (target != ignore_index).to(input.dtype)
        if weight is not None:
            target_weight = target_weight * weight[clamped_target]
        out.mul_(target_weight)

        if reduction == "none":
            return out
        elif reduction == "mean":
            return torch.sum(out) / torch.sum(target_weight)
        elif reduction == "sum":
            return torch.sum(out)
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def setup_context(ctx, inputs, output):
        if ctx.needs_input_grad[0]:
            input, target, weight, ignore_index, reduction = inputs
            ctx.dim = 0 if input.dim() == 1 else 1
            ctx.input_shape = input.shape
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.num_classes = input.shape[ctx.dim]
            ctx.save_for_backward(target, weight)

    @staticmethod
    def backward(ctx, grad_output):
        target, weight = ctx.saved_tensors
        clamped_target = target.clamp(0, ctx.num_classes - 1)
        target_weight = (target != ctx.ignore_index).to(grad_output.dtype)
        if weight is not None:
            target_weight = target_weight * weight[clamped_target]
        grad_input = torch.zeros(
            ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        grad_input.scatter_(
            ctx.dim,
            index=clamped_target.unsqueeze(ctx.dim),
            src=-(grad_output * target_weight).unsqueeze(ctx.dim),
        )
        if ctx.reduction == "mean":
            grad_input.div_(torch.sum(target_weight))
        return grad_input, None, None, None, None


def nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean"):
    return NLLLossFunction.apply(input, target, weight, ignore_index, reduction)


class NLLLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        return nll_loss(input, target, self.weight, self.ignore_index, self.reduction)


def _get_weight_shape(input):
    dim = 0 if input.dim() == 1 else 1
    weight_shape = [1] * input.dim()
    weight_shape[dim] = input.shape[dim]
    return tuple(weight_shape)


def _compute_weight_sum(input, target, weight, ignore_index, num_classes):
    weight_sum = input.numel() // num_classes
    if target.dim() < input.dim():
        mask = target != ignore_index
        if weight is None:
            weight_sum = mask.sum()
        else:
            weight_sum = torch.sum(weight[target.clamp(0, num_classes - 1)] * mask)
    return weight_sum


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, target, weight, ignore_index, reduction, label_smoothing):
        dim = 0 if input.dim() == 1 else 1
        num_classes = input.shape[dim]
        weight_shape = _get_weight_shape(input)
        logp = log_softmax(input, dim)
        if target.dim() < input.dim():
            out = NLLLossFunction.apply(logp, target, weight, ignore_index, "none")
            if label_smoothing > 0.0:
                out.mul_(1 - label_smoothing)
                scale = label_smoothing / num_classes
                sample_mask = (target != ignore_index).to(logp.dtype)
                if weight is None:
                    out.sub_(scale * torch.sum(logp, dim) * sample_mask)
                else:
                    weight_view = weight.reshape(weight_shape)
                    out.sub_(scale * torch.sum(logp * weight_view, dim) * sample_mask)
        else:
            if weight is None:
                out = torch.sum(-logp * target, dim)
            else:
                out = torch.sum(-logp * target * weight.reshape(weight_shape), dim)
            if label_smoothing > 0.0:
                scale = label_smoothing / num_classes
                if weight is None:
                    w_logp = torch.sum(logp, dim)
                else:
                    w_logp = torch.sum(logp * weight.reshape(weight_shape), dim)
                out.mul_(1 - label_smoothing).sub_(scale * w_logp)

        if reduction == "none":
            return out
        elif reduction == "mean":
            weight_sum = _compute_weight_sum(
                input, target, weight, ignore_index, num_classes
            )
            return torch.sum(out) / weight_sum
        elif reduction == "sum":
            return torch.sum(out)
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def setup_context(ctx, inputs, output):
        if any(ctx.needs_input_grad):
            input, target, weight, ignore_index, reduction, label_smoothing = inputs
            ctx.dim = 0 if input.dim() == 1 else 1
            ctx.num_classes = input.shape[ctx.dim]
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.label_smoothing = label_smoothing
            ctx.weight_sum = _compute_weight_sum(
                input, target, weight, ignore_index, ctx.num_classes
            )
            ctx.save_for_backward(input, target, weight)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reduction == "none":
            grad_output = grad_output.unsqueeze(ctx.dim)

        input, target, weight = ctx.saved_tensors
        weight_shape = _get_weight_shape(input)
        scale = ctx.label_smoothing / ctx.num_classes
        y = torch.full_like(input, scale)
        if weight is not None:
            y.mul_(weight.reshape(weight_shape))

        if target.dim() < input.dim():
            valid_mask = (target != ctx.ignore_index).to(input.dtype)
            y.mul_(valid_mask.unsqueeze(ctx.dim))

            clamped_target = target.clamp(0, ctx.num_classes - 1)
            target_delta = (1 - ctx.label_smoothing) * valid_mask
            if weight is not None:
                target_delta *= weight[clamped_target]
            y.scatter_add_(
                ctx.dim,
                index=clamped_target.unsqueeze(ctx.dim),
                src=target_delta.unsqueeze(ctx.dim),
            )
        else:
            target_y = (1 - ctx.label_smoothing) * target
            if weight is not None:
                target_y.mul_(weight.reshape(weight_shape))
            y.add_(target_y)

        grad_input = grad_output * (
            softmax(input, ctx.dim) * torch.sum(y, ctx.dim, keepdim=True) - y
        )

        grad_target = None
        if ctx.needs_input_grad[1]:
            grad_target = (
                (ctx.label_smoothing - 1) * grad_output * log_softmax(input, ctx.dim)
            )
            if weight is not None:
                grad_target.mul_(weight.reshape(weight_shape))

        if ctx.reduction == "mean":
            grad_input.div_(ctx.weight_sum)
            if grad_target is not None:
                grad_target.div_(ctx.weight_sum)
        return grad_input, grad_target, None, None, None, None


def cross_entropy(
    input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0
):
    return CrossEntropyFunction.apply(
        input, target, weight, ignore_index, reduction, label_smoothing
    )


class CrossEntropyLoss(nn.Module):
    def __init__(
        self, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return cross_entropy(
            input,
            target,
            self.weight,
            self.ignore_index,
            self.reduction,
            self.label_smoothing,
        )


class KLDivFunction(torch.autograd.Function):
    @staticmethod
    def forward(input, target, reduction, log_target):
        if log_target:
            out = target.exp() * (target - input)
        else:
            out = torch.xlogy(target, target) - target * input

        if reduction == "mean":
            return out.mean()
        elif reduction == "batchmean":
            return out.sum() / input.size(0)
        elif reduction == "sum":
            return out.sum()
        elif reduction == "none":
            return out
        else:
            raise ValueError(
                f"Expect reduction to be none/mean/sum/batchmean, got {reduction}."
            )

    @staticmethod
    def setup_context(ctx, inputs, output):
        if any(ctx.needs_input_grad):
            input, target, reduction, log_target = inputs
            ctx.reduction = reduction
            ctx.log_target = log_target
            if ctx.needs_input_grad[1]:
                ctx.save_for_backward(input, target)
            else:
                ctx.save_for_backward(None, target)

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            if ctx.log_target:
                grad_input = -grad_output * target.exp()
            else:
                grad_input = -grad_output * target

        grad_target = None
        if ctx.needs_input_grad[1]:
            if ctx.log_target:
                grad_target = grad_output * target.exp() * (1 + target - input)
            else:
                grad_target = grad_output * (1 + target.log() - input)

        if ctx.reduction in ("mean", "batchmean"):
            denom = target.numel() if ctx.reduction == "mean" else target.size(0)
            if grad_input is not None:
                grad_input.div_(denom)
            if grad_target is not None:
                grad_target.div_(denom)

        return grad_input, grad_target, None, None


def kl_div(input, target, reduction="mean", log_target=False):
    return KLDivFunction.apply(input, target, reduction, log_target)


class KLDivLoss(nn.Module):
    def __init__(self, reduction="mean", log_target=False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input, target):
        return kl_div(input, target, self.reduction, self.log_target)


class CosineSimilarityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, dim, eps):
        norm1 = torch.linalg.vector_norm(x1, dim=dim, keepdim=True)
        norm2 = torch.linalg.vector_norm(x2, dim=dim, keepdim=True)
        out = (x1 * x2).sum(dim=dim) / (norm1 * norm2).clamp(min=eps).squeeze(dim)
        if any(ctx.needs_input_grad):
            ctx.dim = dim
            ctx.eps = eps
            # Recomputing activations to save memory (Gradient Checkpointing) is a
            # standard tactic, but you should only do it for massive tensors. norm1,
            # norm2 are usually (Batch, ), small to save
            ctx.save_for_backward(x1, x2, norm1, norm2, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, norm1, norm2, out = ctx.saved_tensors

        mask = norm1 * norm2 > ctx.eps
        norm_prod = (norm1 * norm2).clamp(min=ctx.eps)
        cos_sim = out.unsqueeze(ctx.dim)

        grad_x1 = None
        if ctx.needs_input_grad[0]:
            # Clamp norms used in division branch so background computation never
            # executes a divide-by-zero, even if that branch is ultimately masked out.
            norm1_sqr = torch.where(mask, norm1 * norm1, torch.ones_like(norm1))
            grad_x1 = grad_output.unsqueeze(ctx.dim) * torch.where(
                mask, x2 / norm_prod - cos_sim * x1 / norm1_sqr, x2 / ctx.eps
            )
        grad_x2 = None
        if ctx.needs_input_grad[1]:
            norm2_sqr = torch.where(mask, norm2 * norm2, torch.ones_like(norm2))
            grad_x2 = grad_output.unsqueeze(ctx.dim) * torch.where(
                mask, x1 / norm_prod - cos_sim * x2 / norm2_sqr, x1 / ctx.eps
            )

        return grad_x1, grad_x2, None, None


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    return CosineSimilarityFunction.apply(x1, x2, dim, eps)


class CosineSimilarity(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return cosine_similarity(x1, x2, self.dim, self.eps)


def cosine_embedding_loss(input1, input2, target, margin=0.0, reduction="mean"):
    # The compose-over-custom-Function decision is right here. Worth being
    # explicit about why, so you don't second-guess and rewrite it later: the
    # only mass-tensors are x1, x2 of shape (N, D), and those are already
    # saved by the inner CosineSimilarityFunction. Everything downstream —
    # cos_sim, the where-mask, the clamped output — is shape (N,). A custom
    # outer Function would save the same (N, D) tensors plus the same tiny
    # (N,) tensors, with extra code. No win. Custom Functions earn their
    # complexity when you can collapse intermediates that the autograd tape
    # would otherwise hold — there's nothing to collapse here.
    cos_sim = cosine_similarity(input1, input2, dim=-1)
    out = torch.where(target == 1, 1 - cos_sim, (cos_sim - margin).clamp(min=0.0))
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    elif reduction == "none":
        return out
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1, input2, target):
        return cosine_embedding_loss(
            input1, input2, target, self.margin, self.reduction
        )


def hinge_embedding_loss(input, target, margin=1.0, reduction="mean"):
    out = torch.where(target == 1, input, (margin - input).clamp(min=0.0))
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    elif reduction == "none":
        return out
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class HingeEmbeddingLoss(nn.Module):
    def __init__(self, margin=1.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input, target):
        return hinge_embedding_loss(input, target, self.margin, self.reduction)


# class SoftMarginLossFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(input, target, reduction="mean"):
#         prod = input * target
#         mask = prod > 0.0
#         out = torch.where(
#             mask, torch.log1p(torch.exp(-prod)), torch.log1p(torch.exp(prod)) - prod
#         )

#         if reduction == "mean":
#             return out.mean()
#         elif reduction == "sum":
#             return out.sum()
#         elif reduction == "none":
#             return out
#         else:
#             raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         if ctx.needs_input_grad[0]:
#             input, target, reduction = inputs
#             ctx.reduction = reduction
#             ctx.save_for_backward(input, target)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, target = ctx.saved_tensors
#         grad_input = grad_output * target * (sigmoid(input * target) - 1)
#         if ctx.reduction == "mean":
#             grad_input.div_(input.numel())
#         return grad_input, None, None


def soft_margin_loss(input, target, reduction="mean"):
    # we don't use SoftMarginLossFunction.apply because there's no memory saved and only
    # 1-2 kernels saved, the customized version saves input + target, while input *
    # target saves target and logsigmoid saves input or ouput.
    out = -logsigmoid(input * target)
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    elif reduction == "none":
        return out
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class SoftMarginLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return soft_margin_loss(input, target, self.reduction)


def margin_ranking_loss(input1, input2, target, margin=0.0, reduction="mean"):
    out = (margin - target * (input1 - input2)).clamp(min=0.0)
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    elif reduction == "none":
        return out
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class MarginRankingLoss(nn.Module):
    def __init__(self, margin=0.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1, input2, target):
        return margin_ranking_loss(input1, input2, target, self.margin, self.reduction)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    return torch.linalg.vector_norm(x1 - x2 + eps, ord=p, dim=-1, keepdim=keepdim)


class PairwiseDistance(nn.Module):
    def __init__(self, p=2.0, eps=1e-06, keepdim=False):
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1, x2):
        return pairwise_distance(x1, x2, self.p, self.eps, self.keepdim)


def triplet_margin_with_distance_loss(
    anchor,
    positive,
    negative,
    distance_function=None,
    margin=1.0,
    swap=False,
    reduction="mean",
):
    if distance_function is None:
        distance_function = pairwise_distance
    d_ap = distance_function(anchor, positive)
    d_an = distance_function(anchor, negative)
    if swap:
        d_pn = distance_function(positive, negative)
        out = (d_ap - torch.minimum(d_an, d_pn) + margin).clamp(min=0.0)
    else:
        out = (d_ap - d_an + margin).clamp(min=0.0)
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    elif reduction == "none":
        return out
    else:
        raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")


class TripletMarginWithDistanceLoss(nn.Module):
    def __init__(
        self, *, distance_function=None, margin=1.0, swap=False, reduction="mean"
    ):
        super().__init__()
        self.distance_function = distance_function
        self.margin = margin
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        return triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            self.distance_function,
            self.margin,
            self.swap,
            self.reduction,
        )


def triplet_margin_loss(
    anchor,
    positive,
    negative,
    margin=1.0,
    p=2.0,
    eps=1e-06,
    swap=False,
    reduction="mean",
):
    distance_function = PairwiseDistance(p=p, eps=eps)
    return triplet_margin_with_distance_loss(
        anchor, positive, negative, distance_function, margin, swap, reduction
    )


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0, p=2.0, eps=1e-06, swap=False, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        return triplet_margin_loss(
            anchor,
            positive,
            negative,
            self.margin,
            self.p,
            self.eps,
            self.swap,
            self.reduction,
        )


class MultiMarginLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, p, margin, weight, reduction):
        target_view = target.unsqueeze(-1)
        error = input - torch.gather(input, dim=-1, index=target_view)
        compensate_src = torch.full(
            target_view.shape, -margin, device=input.device, dtype=input.dtype
        )
        error.add_(margin).scatter_add_(-1, target_view, compensate_src)
        out = error.clamp(min=0.0)
        if ctx.needs_input_grad[0]:
            ctx.p = p
            ctx.margin = margin
            ctx.reduction = reduction
            ctx.num_classes = input.size(-1)
            ctx.save_for_backward(
                out if p == 2 else None, error > 0.0, target_view, weight
            )

        if p == 2:
            out = out * out
        out = torch.sum(out, dim=-1) / input.size(-1)

        if weight is not None:
            out.mul_(weight[target])

        if reduction == "mean":
            return out.mean()
        elif reduction == "sum":
            return out.sum()
        elif reduction == "none":
            return out
        else:
            raise ValueError(f"Expect reduction to be none/mean/sum, got {reduction}.")

    @staticmethod
    def backward(ctx, grad_output):
        out, mask, target_view, weight = ctx.saved_tensors
        if ctx.p == 2:
            out.mul_(2.0)
            neg_out_sum = -torch.sum(out, dim=-1, keepdim=True)
            grad_input = torch.scatter_add(out, -1, target_view, neg_out_sum)
        else:
            grad_input = mask.to(grad_output.dtype)
            grad_input.scatter_add_(
                -1, target_view, -grad_input.sum(dim=-1, keepdim=True)
            )
        if weight is not None:
            grad_input.mul_(weight[target_view])
        if ctx.reduction == "mean" and mask.dim() > 0:
            grad_input.div_(mask.size(0))

        grad_input.mul_(
            grad_output.unsqueeze(-1) if ctx.reduction == "none" else grad_output
        ).div_(ctx.num_classes)
        return grad_input, None, None, None, None, None


def multi_margin_loss(input, target, p=1.0, margin=1.0, weight=None, reduction="mean"):
    return MultiMarginLossFunction.apply(input, target, p, margin, weight, reduction)


class MultiMarginLoss(nn.Module):
    def __init__(self, p=1, margin=1.0, weight=None, reduction="mean"):
        super().__init__()
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return multi_margin_loss(
            input, target, self.p, self.margin, self.weight, self.reduction
        )


class MultiLabelMarginLossFunction(torch.autograd.Function):
    pass


def multilabel_margin_loss(input, target, reduction="mean"):
    return MultiLabelMarginLossFunction.apply(input, target, reduction)


class MultiLabelMarginLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return multilabel_margin_loss(input, target, self.reduction)


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    # dynamic programming.
    pass


class CTCLoss(nn.Module):
    def __init__(self, blank=0, reduction="mean", zero_infinity=False):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )
