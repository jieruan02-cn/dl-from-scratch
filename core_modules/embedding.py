import math
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
    def forward(
        weight, input, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    ):
        if max_norm is not None:
            # no need for "with torch.no_grad():" as customized Function forward is in no grad.
            unique_indices = torch.unique(input)
            norms = torch.linalg.vector_norm(
                weight[unique_indices], ord=norm_type, dim=-1
            )
            mask = norms > max_norm
            weight[unique_indices[mask]] *= (max_norm / norms[mask]).unsqueeze(-1)

        # Cannot run register_hook on customized Function as the autograd is disable and
        # out doesn't require grad. Also the customized backward handle it alread, no
        # need to re-divide.
        return weight[input]

    @staticmethod
    def setup_context(ctx, inputs, output):
        weight, input, padding_idx, _, _, scale_grad_by_freq, sparse = inputs
        ctx.embedding_shape = weight.shape
        ctx.save_for_backward(input)
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.padding_idx = padding_idx
        ctx.sparse = sparse

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        unique_x, unique_x_inverse, *rest = torch.unique(
            input, return_inverse=True, return_counts=ctx.scale_grad_by_freq
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

        if ctx.sparse:
            grad_weight = torch.sparse_coo_tensor(
                indices=unique_x.unsqueeze(0),
                values=values,
                size=ctx.embedding_shape,
                check_invariants=True,
            )
        else:
            grad_weight = torch.zeros(
                ctx.embedding_shape, device=grad_output.device, dtype=grad_output.dtype
            )
            grad_weight[unique_x] = values
        return grad_weight, None, None, None, None, None, None


def embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return EmbeddingFunction.apply(
        weight, input, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    )


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
        if padding_idx is not None and padding_idx < 0:
            self.padding_idx += num_embeddings
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
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)
            # no need to register hook again as the zero grad is done in Customized grad.

    def forward(self, input):
        return embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class EmbeddingBagFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        input,
        weight,
        offsets,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weight,
        include_last_offset,
        padding_idx,
    ):
        if max_norm is not None:
            unique_indices = torch.unique(input)
            norms = torch.linalg.vector_norm(
                weight[unique_indices], ord=norm_type, dim=-1
            )
            mask = norms > max_norm
            weight[unique_indices[mask]] *= (max_norm / norms[mask]).unsqueeze(-1)

        # out = torch.zeros(input.shape[:-1] + (weight.size(-1),), device=)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


def embedding_bag(
    input,
    weight,
    offsets=None,
    max_norm=None,
    norm_type=2,
    scale_grad_by_freq=False,
    mode="mean",
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=None,
):
    return EmbeddingBagFunction.apply(
        input,
        weight,
        offsets,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


class EmbeddingBag(Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        mode="mean",
        sparse=False,
        _weight=None,
        include_last_offset=False,
        padding_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            True,
            device,
            dtype,
        )
        self.mode = mode
        self.include_last_offset = include_last_offset

    def forward(self, input, offsets=None, per_sample_weights=None):
        return embedding_bag(
            input,
            self.weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, in_features, max_window, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.max_window = max_window

        config = {"device": device, "dtype": torch.float32}
        denom_tensor = torch.arange(0, in_features, **config).unsqueeze(0)
        denom_tensor[:, 1:in_features:2] -= 1
        denom_tensor.mul_(math.log(10000) / in_features).exp_()
        positional_encoding = (
            torch.arange(0, max_window, **config).unsqueeze(-1) / denom_tensor
        )
        positional_encoding[:, 0:in_features:2].sin_()
        positional_encoding[:, 1:in_features:2].cos_()
        if dtype is not None:
            positional_encoding = positional_encoding.to(dtype)
        # Set persistent=False to save memory in checkpoint as it won't enter state_dict.
        self.register_buffer("positional_encoding", positional_encoding, False)

    def forward(self, input):
        # broadcastable
        return input + self.positional_encoding[0 : input.size(-2), :]
