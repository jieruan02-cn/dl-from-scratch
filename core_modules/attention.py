import math
import torch
import torch.nn as nn
from activation import softmax
from dropout import dropout


# TODO(jieruan): write the customized FlashAttention algorithm for learning and peak
# memory gain. To get the speed gain, I need to write CUDA C++ to ensure the transient
# matrix lives in on-chip SRAM instead of HBM.
def scaled_dot_product_attention_core(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    attn_score = (query @ key.mT).mul_(scale)

    if attn_mask is None:
        if is_causal:
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), device=query.device, dtype=torch.bool
            ).tril_()
            attn_mask = attn_mask.reshape((1,) * (query.dim() - 2) + attn_mask.shape)
            attn_score = torch.where(attn_mask, attn_score, -torch.inf)
    elif attn_mask.dtype == torch.bool:
        attn_score = torch.where(attn_mask, attn_score, -torch.inf)
    else:
        attn_score.add_(attn_mask)

    attn_score = softmax(attn_score, dim=-1)
    if dropout_p != 0.0:
        attn_score = dropout(attn_score, p=dropout_p)
    return attn_score @ value


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
    if enable_gqa and key.size(-3) > 1 and query.size(-3) > key.size(-3):
        num_group = query.size(-3) // key.size(-3)
        query_view, key_view, value_view = (
            query.view(query.shape[:-3] + (key.size(-3), num_group) + query.shape[-2:]),
            key.unsqueeze(-3),
            value.unsqueeze(-3),
        )
        out = scaled_dot_product_attention_core(
            query_view, key_view, value_view, attn_mask, dropout_p, is_causal, scale
        )
        out = out.view(query.shape[:-1] + (value.shape[-1],))
    else:
        out = scaled_dot_product_attention_core(
            query, key, value, attn_mask, dropout_p, is_causal, scale
        )
    return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        super().__init__()

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        pass
