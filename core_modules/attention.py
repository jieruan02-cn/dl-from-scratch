import copy
import math
import torch
import torch.nn as nn
from activation import softmax, relu
from dropout import dropout, Dropout
from linear import linear, Linear
from normalization import LayerNorm


def _canonical_mask(mask, device=None, dtype=None):
    if mask.dtype == torch.bool:
        mask = torch.zeros_like(
            mask, device=device, dtype=torch.float32 if dtype is None else dtype
        ).masked_fill_(mask, -torch.inf)
    return mask


def _causal_mask(shape, device=None, dtype=None):
    return torch.full(shape, -torch.inf, device=device, dtype=dtype).triu_(diagonal=1)


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
    config = {"device": query.device, "dtype": query.dtype}
    if is_causal:
        assert attn_mask is None
        attn_mask = _causal_mask((query.size(-2), key.size(-2)), **config)
    elif attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = _canonical_mask(~attn_mask, **config)

    attn_weights = query @ key.mT
    attn_weights.mul_(1.0 / math.sqrt(query.size(-1)) if scale is None else scale)
    if attn_mask is not None:
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
        self.add_zero_attn = add_zero_attn
        self.batch_first = batch_first
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        self.head_dim = embed_dim // num_heads

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

        config = {"device": query.device, "dtype": query.dtype}
        if is_causal:
            assert attn_mask is None
            attn_mask = _causal_mask(
                (query.size(0), key.size(0))
                if is_unbatched or not self.batch_first
                else (query.size(1), key.size(1)),
                **config,
            )
        elif attn_mask is not None:
            attn_mask = _canonical_mask(attn_mask, **config)
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.view(
                    (attn_mask.shape[0] // self.num_heads, self.num_heads)
                    + attn_mask.shape[-2:]
                )
        if key_padding_mask is not None:
            key_padding_mask = _canonical_mask(key_padding_mask, **config)
            if key_padding_mask.dim() > 1:
                key_padding_mask = key_padding_mask[:, None, None, :]
            attn_mask = (
                key_padding_mask if attn_mask is None else attn_mask + key_padding_mask
            )
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
                attn_mask = nn.functional.pad(attn_mask, (0, 1))
        if self.add_zero_attn:
            key = nn.functional.pad(key, (0, 0, 0, 1))
            value = nn.functional.pad(value, (0, 0, 0, 1))
            if attn_mask is not None:
                attn_mask = nn.functional.pad(attn_mask, (0, 1))

        out, attn_weights = scaled_dot_product_attention_core(
            query,
            key,
            value,
            attn_mask=attn_mask,
            # need to disable dropout when inference
            dropout_p=self.dropout if self.training else 0.0,
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


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=relu,
        layer_norm_eps=1e-05,
        batch_first=False,
        norm_first=False,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.multi_head_attn = MultiheadAttention(
            d_model, nhead, dropout, bias, batch_first=batch_first, **config
        )
        self.dropout = Dropout(p=dropout)
        self.layer_norm1 = LayerNorm(d_model, layer_norm_eps, bias=bias, **config)
        self.ffn = nn.Sequential(
            Linear(d_model, dim_feedforward, bias, **config),
            Lambda(activation),
            Dropout(p=dropout),
            Linear(dim_feedforward, d_model, bias, **config),
            Dropout(p=dropout),
        )
        self.layer_norm2 = LayerNorm(d_model, layer_norm_eps, bias=bias, **config)
        self.norm_first = norm_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if self.norm_first:
            mha_input = self.layer_norm1(src)
            mha_out, _ = self.multi_head_attn(
                query=mha_input,
                key=mha_input,
                value=mha_input,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                attn_mask=src_mask,
                is_causal=is_causal,
            )
            out = src + self.dropout(mha_out)
            out = out + self.ffn(self.layer_norm2(out))
        else:
            out, _ = self.multi_head_attn(
                query=src,
                key=src,
                value=src,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                attn_mask=src_mask,
                is_causal=is_causal,
            )
            out = self.layer_norm1(src + self.dropout(out))
            out = self.layer_norm2(out + self.ffn(out))

        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=relu,
        layer_norm_eps=1e-05,
        batch_first=False,
        norm_first=False,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        config = {"device": device, "dtype": dtype}
        self.multi_head_attn = MultiheadAttention(
            d_model, nhead, dropout, bias, batch_first=batch_first, **config
        )
        self.layer_norm1 = LayerNorm(d_model, layer_norm_eps, bias=bias, **config)
        self.dropout1 = Dropout(p=dropout)

        self.multi_head_cross_attn = MultiheadAttention(
            d_model, nhead, dropout, bias, batch_first=batch_first, **config
        )
        self.layer_norm2 = LayerNorm(d_model, layer_norm_eps, bias=bias, **config)
        self.dropout2 = Dropout(p=dropout)

        self.ffn = nn.Sequential(
            Linear(d_model, dim_feedforward, bias, **config),
            Lambda(activation),
            Dropout(p=dropout),
            Linear(dim_feedforward, d_model, bias, **config),
            Dropout(p=dropout),
        )
        self.layer_norm3 = LayerNorm(d_model, layer_norm_eps, bias=bias, **config)
        self.norm_first = norm_first

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        if self.norm_first:
            mha_input = self.layer_norm1(tgt)
            mha_output, _ = self.multi_head_attn(
                query=mha_input,
                key=mha_input,
                value=mha_input,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                attn_mask=tgt_mask,
                average_attn_weights=False,
                is_causal=tgt_is_causal,
            )
            out = tgt + self.dropout1(mha_output)

            cross_attn_output, _ = self.multi_head_cross_attn(
                query=self.layer_norm2(out),
                key=memory,
                value=memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                attn_mask=memory_mask,
                average_attn_weights=False,
                is_causal=memory_is_causal,
            )
            out = out + self.dropout2(cross_attn_output)
            out = out + self.ffn(self.layer_norm3(out))
        else:
            out, _ = self.multi_head_attn(
                query=tgt,
                key=tgt,
                value=tgt,
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,
                attn_mask=tgt_mask,
                average_attn_weights=False,
                is_causal=tgt_is_causal,
            )
            out = self.layer_norm1(tgt + self.dropout1(out))

            cross_attn_output, _ = self.multi_head_cross_attn(
                query=out,
                key=memory,
                value=memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
                attn_mask=memory_mask,
                average_attn_weights=False,
                is_causal=memory_is_causal,
            )
            out = self.layer_norm2(out + self.dropout2(cross_attn_output))
            out = self.layer_norm3(out + self.ffn(out))

        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ):
        super().__init__()
        # this will make each layer's initial weights the same, factory version might be
        # practically better.
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        out = src
        for layer in self.layers:
            out = layer(
                out,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        if self.norm is not None:
            out = self.norm(out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer() for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=None,
        memory_is_causal=None,
    ):
        pass
