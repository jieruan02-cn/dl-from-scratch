import math
import torch
import torch.nn as nn
from activation import softmax
from dropout import dropout
from linear import linear


# TODO(jieruan): write the customized FlashAttention algorithm for learning and peak
# memory gain. To get the speed gain, I need to write CUDA C++ to ensure the transient
# matrix lives in on-chip SRAM instead of HBM.
def scaled_dot_product_attention_core(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    need_weights=False,
):
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    attn_weights = (query @ key.mT).mul_(scale)

    if is_causal:
        assert attn_mask is None
        attn_mask = torch.ones(
            query.size(-2), key.size(-2), device=query.device, dtype=torch.bool
        ).tril_()
        attn_mask = attn_mask.reshape((1,) * (query.dim() - 2) + attn_mask.shape)
        attn_weights = torch.where(attn_mask, attn_weights, -torch.inf)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weights = torch.where(attn_mask, attn_weights, -torch.inf)
        else:
            attn_weights.add_(attn_mask)

    attn_weights = softmax(attn_weights, dim=-1)
    if dropout_p != 0.0:
        attn_weights = dropout(attn_weights, p=dropout_p)
    return attn_weights @ value, attn_weights if need_weights else None


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
        out, _ = scaled_dot_product_attention_core(
            query_view, key_view, value_view, attn_mask, dropout_p, is_causal, scale
        )
        out = out.view(query.shape[:-1] + (value.shape[-1],))
    else:
        out, _ = scaled_dot_product_attention_core(
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
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.num_heads = num_heads
        self.dropout = dropout
        self.add_zeron_attn = add_zero_attn
        self.batch_first = batch_first
        self.embed_dim = embed_dim

        config = {"device": device, "dtype": dtype}
        kdim, vdim = (
            embed_dim if kdim is None else kdim,
            embed_dim if vdim is None else vdim,
        )
        self.weight_query = nn.Parameter(torch.empty(embed_dim, embed_dim, **config))
        self.weight_key = nn.Parameter(torch.empty(embed_dim, kdim, **config))
        self.weight_value = nn.Parameter(torch.empty(embed_dim, vdim, **config))
        self.weight_out = nn.Parameter(torch.empty(embed_dim, embed_dim, **config))
        if self.bias:
            self.bias_query = nn.Parameter(torch.empty(embed_dim, **config))
            self.bias_key = nn.Parameter(torch.empty(embed_dim, **config))
            self.bias_value = nn.Parameter(torch.empty(embed_dim, **config))
            self.bias_out = nn.Parameter(torch.empty(embed_dim, **config))
        else:
            self.register_parameter("bias_query", None)
            self.register_parameter("bias_key", None)
            self.register_parameter("bias_value", None)
            self.register_parameter("bias_out", None)
        if self.add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(embed_dim, **config))
            self.bias_v = nn.Parameter(torch.empty(embed_dim, **config))
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.reset_parameters()

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

        query = linear(query, self.weight_query, self.bias_query)
        key = linear(key, self.weight_key, self.bias_key)
        value = linear(value, self.weight_value, self.bias_value)

        is_unbatched = query.dim() == 2
        if is_unbatched:
            attn_out_shape = (1, query.size(0), self.embed_dim)
        elif self.batch_first:
            attn_out_shape = query.shape[:-1] + (self.embed_dim,)
        else:
            attn_out_shape = (query.size(1), query.size(0), self.embed_dim)

        if is_causal:
            assert attn_mask is None
            L, S = (
                query.size(1) if self.batch_first else query.size(0),
                key.size(1) if self.batch_first else key.size(0),
            )
            attn_mask = torch.ones(L, S, device=query.device, dtype=torch.bool)
            attn_mask = attn_mask.triu_(diagonal=1)
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.view(
                    (attn_mask.size(0) // self.num_heads, self.num_heads)
                    + attn_mask.shape[-2:]
                )
            if attn_mask.dtype == torch.bool:
                attn_mask = ~attn_mask
        if key_padding_mask is not None:
            if key_padding_mask.dim() > 1:
                key_padding_mask = key_padding_mask[:, None, None, :]
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask & ~key_padding_mask
                else:
                    attn_mask = attn_mask + key_padding_mask
            elif key_padding_mask.dtype == torch.bool:
                attn_mask = ~key_padding_mask
            else:
                attn_mask = key_padding_mask
        query, key, value = self._normalize_shape(
            query, key, value, is_unbatched, self.batch_first
        )

        if self.add_bias_kv:
            view_shape = (1, self.num_heads, 1, -1)
            expand_shape = (key.size(0), -1, -1, -1)
            key = torch.cat(
                [key, self.bias_k.view(view_shape).expand(expand_shape)], dim=2
            )
            value = torch.cat(
                [value, self.bias_v.view(view_shape).expand(expand_shape)], dim=2
            )
            if attn_mask is not None:
                pad_value = True if attn_mask.dtype == torch.bool else 0.0
                attn_mask = nn.functional.pad(attn_mask, (0, 1), value=pad_value)

        out, attn_weights = scaled_dot_product_attention_core(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
            need_weights=need_weights,
        )

        out = self._unnormalize_shape(
            out, is_unbatched, self.batch_first, attn_out_shape
        )
        out = linear(out, self.weight_out, self.bias_out)
        if attn_weights is not None:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            if is_unbatched:
                attn_weights = attn_weights.squeeze(0)

        return out, attn_weights

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_query)
        nn.init.xavier_uniform_(self.weight_key)
        nn.init.xavier_uniform_(self.weight_value)
        nn.init.xavier_uniform_(self.weight_out)
        if self.bias:
            nn.init.zeros_(self.bias_query)
            nn.init.zeros_(self.bias_key)
            nn.init.zeros_(self.bias_value)
            nn.init.zeros_(self.bias_out)
        if self.add_bias_kv:
            nn.init.zeros_(self.bias_k)
            nn.init.zeros_(self.bias_v)

    def _normalize_shape(self, query, key, value, is_unbatched, batch_first):
        if is_unbatched:
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
        elif not batch_first:
            query, key, value = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )
        kv_shape = key.shape[:-1] + (self.num_heads, -1)
        query = query.view(query.shape[:-1] + (self.num_heads, -1)).transpose(1, 2)
        key = key.view(kv_shape).transpose(1, 2)
        value = value.view(kv_shape).transpose(1, 2)
        return query, key, value

    def _unnormalize_shape(self, out, is_unbatched, batch_first, attn_out_shape):
        out = out.transpose(1, 2).reshape(attn_out_shape)
        if is_unbatched:
            out = out.squeeze(0)
        elif not batch_first:
            out = out.transpose(0, 1)
        return out
