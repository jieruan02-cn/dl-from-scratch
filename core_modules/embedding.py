import torch
import torch.nn as nn


# Lessons:
# 1. in-place op on leaf variable that requires grad (Parameter) cause runtime error generally for two reasons:
#   1) doing in-place op confuse torch computation graph as such node is not a leaf if assigned but parameter has
#   to be a leaf so optimizer knows what to update; 2)leaf variables are meant to be updated by optimizer after
#   .backward(). torch optimizer use no_grad for such update exactly.
# 2. requires_grad_(False) and fill_(0) are only allowed under torch.no_grad() as above, but their behavior differ
#   requires_grad_ only modify the slice view, the fill_ modifies the underlying data, so
#   self.weight[padding_idx].requires_grad_(False) doesn't work, it is a no op silently, which can be test using assert
#   self.weight[padding_idx].requires_grad is False, which will fail, because requires_grad is parameter level data, it
#   only gets update if we call self.weight.requires_grad_(False)
# 3. use register_buffer whenever you have a tensor that is not a parameter (doesn't need gradients) but is still
#   a part of the model's state that needs to stay on the same device as your weights. Common examples include
#   attention masks in Transformers or the running mean and variance in BatchNorm layers.
# 4. scale_grad_by_freq in PyTorch's context they use the last batch's stat instead of running stat.
class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(weight, x, padding_idx, scale_grad_by_freq):
        return weight[x]

    @staticmethod
    def setup_context(ctx, inputs, output):
        weight, x, padding_idx, scale_grad_by_freq = inputs
        ctx.embedding_shape = weight.shape
        ctx.save_for_backward(x)
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.padding_idx = padding_idx

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        unique_x, unique_x_inverse, *rest = torch.unique(
            x, return_inverse=True, return_counts=ctx.scale_grad_by_freq
        )
        values = torch.zeros(
            (unique_x.numel(), ctx.embedding_shape[1]),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        values.index_add_(
            0,
            unique_x_inverse.flatten(),
            grad_output.reshape(-1, ctx.embedding_shape[1]),
        )
        if ctx.scale_grad_by_freq:
            values /= rest[0].unsqueeze(-1).to(grad_output.dtype)
        if ctx.padding_idx is not None:
            values[unique_x == ctx.padding_idx] = 0
        grad_weight = torch.sparse_coo_tensor(
            indices=unique_x.unsqueeze(0),
            values=values,
            size=ctx.embedding_shape,
            check_invariants=True,
        )
        return grad_weight, None, None, None


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        if _weight is not None:
            self.weight = nn.Parameter(_weight, requires_grad=not _freeze)
            assert _weight.shape == (num_embeddings, embedding_dim)
        else:
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                requires_grad=not _freeze,
            )
            nn.init.normal_(self.weight)

        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)
            if self.weight.requires_grad and not sparse:
                self.weight.register_hook(self._zero_pad_grad)

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                unique_indices = torch.unique(x)
                norms = torch.linalg.vector_norm(
                    self.weight[unique_indices], ord=self.norm_type, dim=-1
                )
                mask = norms > self.max_norm
                self.weight[unique_indices[mask]] *= (
                    self.max_norm / norms[mask]
                ).unsqueeze(-1)
        if self.sparse:
            out = EmbeddingFunction.apply(
                self.weight, x, self.padding_idx, self.scale_grad_by_freq
            )
        else:
            out = self.weight[x]

        if self.weight.requires_grad and self.scale_grad_by_freq and not self.sparse:
            freq = torch.bincount(x.flatten(), minlength=self.num_embeddings)
            out.register_hook(
                lambda grad: grad / freq[x].clamp(min=1).to(grad.dtype).unsqueeze(-1)
            )
        return out

    def _zero_pad_grad(self, grad):
        grad = grad.clone()
        grad[self.padding_idx] = 0
        return grad
