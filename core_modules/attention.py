import math
import torch
from activation import softmax
from dropout import dropout


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    attn_score = (query @ key.mT).mul_(scale)

    if attn_mask is None:
        if is_causal:
            attn_mask = torch.tril(
                torch.ones(query.size(-2), key.size(-2)),
                dtype=torch.bool,
                device=attn_score.device,
            )
            attn_mask.reshape((1,) * (query.dim() - 2) + attn_mask.shape)
            attn_score = torch.where(attn_mask, attn_score, -torch.inf)
    elif attn_mask.dtype == torch.bool:
        attn_score = torch.where(attn_mask, attn_score, -torch.inf)
    else:
        attn_score.add_(attn_mask)

    attn_score = softmax(attn_score)
    if dropout_p != 0.0:
        attn_score = dropout(attn_score)
    return attn_score @ value
