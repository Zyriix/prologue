import math
from math import pi
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from torch import einsum, broadcast_tensors, Tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast
from torch import nn
from einops import rearrange, repeat, reduce
import einops
import numpy as np
import os
import copy
import warnings
from utils import print0
from itertools import chain

# Well Inited Linear
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    if mode == 'default':         return np.sqrt(1 / fan_in) * (torch.rand(*shape) * 2 - 1)  # nn.Linear default
    if mode == 'trunc_normal':    return torch.nn.init.trunc_normal_(torch.empty(*shape), std=0.02)
    if mode == 'uniform': return torch.rand() * 2 - 1
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='trunc_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class AdaRMSNorm(nn.Module):
    def __init__(self,dim, cond_dim=None, elementwise_affine=True, cond_based_affine=True, centering=False, eps=1e-6):
        super(AdaRMSNorm, self).__init__()        
        self.dim = dim
        self.eps = eps 
        self.cond_based_affine = cond_based_affine      
        self.elementwise_affine = elementwise_affine
        self.scale  = dim ** 0.5
        assert not(cond_dim is None and cond_based_affine), 'cond_dim must be provided if cond_based_affine is True'
        if elementwise_affine:
            if cond_based_affine and cond_dim is not None:
                self.affine = Linear(cond_dim, dim, init_weight=1e-5)
            else:
                self.weight = nn.Parameter(torch.zeros(self.dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x, cond_emb=None):   
        with torch.amp.autocast('cuda',enabled=False):
            output = F.normalize(x.float(), dim=(-1)) * self.scale
        if self.elementwise_affine:
            weight = self.affine(cond_emb).unsqueeze(1) if self.cond_based_affine and cond_emb is not None else self.weight
            output = output.mul(1. + weight.float())
        return output.type_as(x)

class AdaLN(nn.Module):
    def __init__(self,dim, cond_dim=None, elementwise_affine=True, cond_based_affine=True, bias=True, eps=1e-6):
        super(AdaLN, self).__init__()        
        self.norm = nn.LayerNorm(dim, elementwise_affine=elementwise_affine and not cond_based_affine, eps=eps)  
        self.cond_based_affine = cond_based_affine
        self.bias = bias
        assert not(cond_dim is None and cond_based_affine), 'cond_dim must be provided if cond_based_affine is True'
        self.affine = Linear(cond_dim,  2 * dim if bias else dim, init_weight=1e-5) if cond_based_affine and cond_dim is not None else None      
    def forward(self, x, cond_emb=None):  
        x = self.norm(x)
        if self.cond_based_affine:
            if self.bias:
                shift, scale = self.affine(cond_emb).unsqueeze(1).chunk(2, dim=-1)
                x = x.mul(1. + scale).add_(shift)
            else:
                scale = self.affine(cond_emb).unsqueeze(1)
                x = x.mul(1. + scale)
        return x


class FluxRopeEMB(nn.Module):
    def __init__(self, theta, axes_dim: list[int]):
        super().__init__()
        # theta can be a scalar (applied to all axes) or a list (per-axis theta)
        if isinstance(theta, (int, float)):
            self.theta = [theta] * len(axes_dim)
        else:
            assert len(theta) == len(axes_dim), \
                f"rope theta list length {len(theta)} must match axes_dim length {len(axes_dim)}"
            self.theta = list(theta)
        self.axes_dim = axes_dim

    @autocast('cuda',enabled=False)
    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta[i]) for i in range(len(self.axes_dim))],
            dim=-3,
        )

        return emb.unsqueeze(1)

@autocast('cuda',enabled=False)
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    # Accept both (n,) and (b, n) positions.
    pos=pos.float()
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def apply_rope(x, freqs_cis):
    """Apply RoPE to ``x`` ``[B, H, L, D]`` using ``freqs_cis`` ``[*, 1, S, D/2, 2, 2]``."""
    if freqs_cis is None:
        raise ValueError("freqs_cis is None but RoPE is enabled. Did you forget to pass q_pe/k_pe?")

    b, h, l, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"RoPE requires last dim even, got D={d}")
    x_dtype = x.dtype
    x_float = x.float().reshape(b, h, l, d // 2, 2)
    freqs = freqs_cis.to(device=x.device, dtype=x_float.dtype)
    # Matrix-vector multiply using columns: y = M[...,0]*x0 + M[...,1]*x1
    x0 = x_float[..., 0]
    x1 = x_float[..., 1]
    x_out = freqs[..., 0] * x0.unsqueeze(-1) + freqs[..., 1] * x1.unsqueeze(-1)  # (..., 2)
    return x_out.reshape(b, h, l, d).to(dtype=x_dtype)


# DropPath & FFN
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., ffn_type='geglu', out_value=1.0, ffn_bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.ffn_type = ffn_type
        if self.ffn_type == 'geglu':
            self.fc1 = Linear(in_features, 2*hidden_features, bias=ffn_bias)
            self.act = nn.GELU(approximate='tanh')
            self.fc2 = Linear(hidden_features, out_features, bias=ffn_bias, init_weight=out_value)
        elif self.ffn_type == 'ffn':
            self.fc1 = Linear(in_features, hidden_features, bias=ffn_bias)
            self.act = nn.GELU(approximate='tanh')
            self.fc2 = Linear(hidden_features, out_features, bias=ffn_bias, init_weight=out_value)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.ffn_type == 'geglu':
            gate, value = self.fc1(x).chunk(2, dim=-1)
            gated = self.act(value) * gate
            return self.drop(self.fc2(gated))
        elif self.ffn_type == 'ffn':
            return self.drop(self.fc2(self.act(self.fc1(x))))
    

class Attention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_norm=False, 
        rope=False, out_value=1.0, attn_out_bias=True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_norm = attn_norm
        self.rope = rope
        
        self.q_norm = AdaRMSNorm(self.head_dim,cond_based_affine=False) if self.attn_norm else None
        self.k_norm = AdaRMSNorm(self.head_dim,cond_based_affine=False) if self.attn_norm else None
       
        
        self.to_q = Linear(embed_dim, embed_dim, bias=False)
        self.to_kv = Linear(embed_dim, embed_dim * 2, bias=False)

        self.proj = Linear(embed_dim, embed_dim, bias=attn_out_bias, init_weight=out_value)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop = attn_drop    
    
    def forward(self, x, context_emb=None, causal=False, attn_bias=None, cache_kv=False, past_kvs=None,pe=None,ctx_pe=None, return_attn=False):
        B, L, C = x.shape

        q = self.to_q(x) 
        q = einops.rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        
        kv = self.to_kv(x) if context_emb is None else self.to_kv(context_emb)
        k, v = einops.rearrange(kv, 'b l (k h d) -> k b h l d', k=2, h=self.num_heads)
        q = self.q_norm(q) if self.q_norm is not None else q
        k = self.k_norm(k) if self.k_norm is not None else k

        if self.rope:
            q = apply_rope(q, pe)
            if context_emb is not None and ctx_pe is not None:
                k = apply_rope(k, ctx_pe)
            else:
                k = apply_rope(k, pe)

        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(1).expand(B, self.num_heads, L, L)

        if past_kvs is not None:
            past_keys, past_values = past_kvs
            k = torch.cat((past_keys, k), dim=2)
            v = torch.cat((past_values, v), dim=2)
    
        if return_attn:
            scale = self.head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            if causal and past_kvs is None and attn_bias is None:
                L_q, L_k = q.size(2), k.size(2)
                causal_mask = torch.triu(torch.ones(L_q, L_k, device=q.device, dtype=torch.bool), diagonal=1)
                attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            if attn_bias is not None:
                attn_weights = attn_weights + attn_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            oup = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, L, -1)
        else:
            attn_weights = None
            dropout_p = self.attn_drop if self.training else 0.0
            oup = F.scaled_dot_product_attention(
                query=q, 
                key=k, 
                value=v, 
                is_causal=causal and past_kvs is None and attn_bias is None,
                dropout_p=dropout_p,
                attn_mask=attn_bias
            ).transpose(1, 2).reshape(B, L, -1)

        out = self.proj_drop(self.proj(oup))
        if return_attn:
            return (out, (k, v), attn_weights) if cache_kv else (out, attn_weights)
        else:
            return (out, (k, v)) if cache_kv else out

class TransformerLayer(nn.Module):
    def __init__(
        self, block_idx, embed_dim, cond_dim,
        num_heads, mlp_ratio=8/3, drop=0., attn_drop=0., drop_path=0., attn_norm=False, cross_attn=False, rope=False,
        ffn_type='geglu', norm_layer=AdaRMSNorm, cond_based_affine=True, ffn_bias=True, attn_out_bias=True
    ):
        super(TransformerLayer, self).__init__()
        self.block_idx = block_idx
        self.cond_based_affine = cond_based_affine
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        out_value = 1.0 if cond_based_affine else 1e-5

        self.adaln_1 = norm_layer(embed_dim, cond_dim, cond_based_affine=cond_based_affine)
        self.attn1 = Attention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_norm=attn_norm, rope=rope, out_value=out_value, attn_out_bias=attn_out_bias)
        self.gate1 = Linear(cond_dim, embed_dim, bias=ffn_bias, init_weight=1e-5) if cond_based_affine else None

        if cross_attn:
            self.has_cross_attn = True
            self.adaln_2 = norm_layer(embed_dim, cond_dim, cond_based_affine=cond_based_affine)
            self.adaln_ctx = norm_layer(embed_dim, cond_dim, cond_based_affine=cond_based_affine)

            self.attn2 = Attention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_norm=attn_norm, rope=rope, out_value=out_value, attn_out_bias=attn_out_bias)
            self.gate2 = Linear(cond_dim, embed_dim, bias=ffn_bias, init_weight=1e-5) if cond_based_affine else None
        else:
            self.has_cross_attn = False

        self.adaln_mlp = norm_layer(embed_dim, cond_dim, cond_based_affine=cond_based_affine)
        mlp_ratio = eval(mlp_ratio) if isinstance(mlp_ratio, str) else mlp_ratio
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, ffn_type=ffn_type, out_value=out_value, ffn_bias=ffn_bias)
        self.gate_mlp = Linear(cond_dim, embed_dim, bias=ffn_bias, init_weight=1e-5) if cond_based_affine else None
        
    def forward(self, x,context_emb=None,  cond_emb=None, causal=False, attn_mask=None, cache_kv=False, past_kvs=None,pe=None,ctx_pe=None, return_attn=False):
        self_attn_bias = attn_mask if (attn_mask is not None and not self.has_cross_attn) else None
        self_causal = causal if self_attn_bias is None else False
        attn_out1 = self.attn1(self.adaln_1(x, cond_emb), context_emb=None,causal=self_causal, attn_bias=self_attn_bias, cache_kv=cache_kv, past_kvs=past_kvs,pe=pe,ctx_pe=None, return_attn=return_attn)
        if return_attn:
            if cache_kv:
                attn_out1, new_kvs, layer_attn_weights = attn_out1
            else:
                attn_out1, layer_attn_weights = attn_out1
        elif cache_kv:
            attn_out1, new_kvs = attn_out1[0], attn_out1[1]
        gate1 = self.gate1(cond_emb).unsqueeze(1) if self.gate1 is not None else 1.0
        x = x + gate1 * self.drop_path(attn_out1)
        if context_emb is not None and self.has_cross_attn:
            gate2 = self.gate2(cond_emb).unsqueeze(1) if self.gate2 is not None else 1.0
            x = x + gate2 * self.drop_path(self.attn2(self.adaln_2(x, cond_emb), context_emb=self.adaln_ctx(context_emb,cond_emb), causal=False, attn_bias=attn_mask,pe=pe,ctx_pe=ctx_pe))
        gate_mlp = self.gate_mlp(cond_emb).unsqueeze(1) if self.gate_mlp is not None else 1.0
        x = x + gate_mlp * self.drop_path(self.ffn(self.adaln_mlp(x, cond_emb)))
    
        if return_attn:
            return (x, new_kvs, layer_attn_weights) if cache_kv else (x, layer_attn_weights)
        elif cache_kv:
            return x, new_kvs
        else:
            return x

class Transformer(nn.Module):
    def __init__(
        self, 
        layer_num, 
        input_dim, dim, output_dim, max_seq_len, heads,
        cond_input_dim, cond_dim, 
        context_type='none', ctx_input_dim=None, ctx_max_seq_len=None, 
        emb_dropout=0., drop=0., attn_drop=0., 
        rope=False, abs_pos=False, 
        attn_norm=False,  causal=False, 
        zero_out=False, noise_query = False, 
        ffn_type='geglu', # ffn or geglu
        norm_layer='RMSNorm', # LayerNorm or RMSNorm
        mlp_ratio=8/3,
        rope_theta=2000,
        rope_ctx_theta=10000,
        rope_axes_dim=[32,32,32],
        rope_input_type='l',
        rope_hybrid_l_len=0,
        rope_ctx_type='none',
        out_norm=True,
        out_layer=True,
        out_act=False,
        input_bias=True,
        input_layer=True,
        use_label=True,
        ffn_bias=True,
        out_bias=True,
        attn_out_bias=True,
        bos_zero_rope=False,
        **params
    ):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_query = noise_query
        self.cond_input_dim = cond_input_dim
        self.dim = dim
        self.causal = causal
        self.bos_zero_rope = bos_zero_rope
        if bos_zero_rope:
            print0("[Transformer] bos_zero_rope=True: BOS token will use zero RoPE position (pos=0)")
        self.concat_causal = (context_type == 'concat' and causal)
        if self.concat_causal:
            self.causal = False
            _ctx_len = ctx_max_seq_len
            _total = max_seq_len + _ctx_len
            _mask = torch.zeros(1, _total, _total)
            _mask[:, :_ctx_len, _ctx_len:] = float('-inf')
            _mask[:, _ctx_len:, _ctx_len:].masked_fill_(
                torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
                float('-inf'))
            self.register_buffer('concat_causal_mask', _mask)
        self.cross_attn = context_type=='cross_attn'
        print0(norm_layer)
        norm_layer = AdaRMSNorm if norm_layer=="RMSNorm" else AdaLN
        
        self.use_label = use_label
        self.layers = nn.ModuleList([
            TransformerLayer(block_idx=i, embed_dim=dim, 
            cond_dim=cond_dim, num_heads=heads, mlp_ratio=mlp_ratio, 
            drop=drop, attn_drop=attn_drop, attn_norm=attn_norm, rope=rope, cross_attn=self.cross_attn, 
            ffn_type=ffn_type, norm_layer=norm_layer, cond_based_affine=self.use_label, ffn_bias=ffn_bias, attn_out_bias=attn_out_bias)
            for i in range(layer_num)
        ])

        out_weight = 1e-5 if zero_out else 1
        scale = np.sqrt(dim)
        self.rope_input_type = rope_input_type
        self.rope_hybrid_l_len = rope_hybrid_l_len
        self.rope_ctx_type = rope_ctx_type

        self.out_norm = norm_layer(dim, cond_dim, cond_based_affine=self.use_label) if out_norm else None
        self.out_activate = nn.GELU(approximate='tanh') if out_act else None
        self.out = Linear(dim, output_dim, bias=out_bias, init_weight=out_weight) if out_layer else None

        if self.use_label:
            self.cond_proj = nn.Sequential(Linear(cond_input_dim, cond_dim), norm_layer(cond_dim, cond_based_affine=False), nn.GELU(approximate='tanh')) 
        else:
            self.cond_proj = None 
        self.max_seq_len = max_seq_len
        self.context_type = context_type # "concat, cross_attn, none"
        if abs_pos:
            if context_type == 'none':
                self.abs_pos = nn.Embedding(max_seq_len, dim)
                nn.init.trunc_normal_(self.abs_pos.weight, std=0.02)
            elif context_type == 'concat':
                self.abs_pos = nn.Embedding(max_seq_len + ctx_max_seq_len, dim)
                nn.init.trunc_normal_(self.abs_pos.weight, std=0.02)
            elif context_type == 'cross_attn':
                self.abs_pos = nn.Embedding(max_seq_len, dim)
                self.ctx_pos = nn.Embedding(ctx_max_seq_len, dim)
                nn.init.trunc_normal_(self.abs_pos.weight, std=0.02)
                nn.init.trunc_normal_(self.ctx_pos.weight, std=0.02)
            else:
                raise ValueError(f"Invalid context_type: {context_type}")
        else:
            self.abs_pos = None
        
        if rope:
            self.rope_emb = FluxRopeEMB(theta=rope_theta, axes_dim=rope_axes_dim)
            if context_type != 'none':
                self.ctx_rope_emb = FluxRopeEMB(theta=rope_ctx_theta, axes_dim=rope_axes_dim)
        else:
            self.rope_emb = None
            self.ctx_rope_emb = None
        
        self.input_layer = Linear(input_dim, dim, bias=input_bias) if input_layer else  nn.Identity()
        if context_type != 'none':
            self.learnable_query = nn.Parameter(torch.empty(1, max_seq_len, dim))
            nn.init.trunc_normal_(self.learnable_query, std=0.02)
        self.emb_dropout = nn.Dropout(p=emb_dropout) if emb_dropout > 0 else nn.Identity()

    def prepare_pos_ids(self, x_len, z_len, past_len,device):
        if self.context_type == 'none':
            if self.rope_input_type == 'hw':
                w = int(np.sqrt(self.max_seq_len))
                idx = torch.arange(past_len, past_len + x_len, device=device)
                x_ids = ctx_ids = torch.stack([
                    (idx // w).float(),
                    (idx % w).float(),
                    torch.zeros(x_len, device=device, dtype=torch.float32),
                ], dim=1)
            elif self.rope_input_type == 'l':
                l = x_len
                x_coords = {
                    "l": torch.arange(past_len, past_len + l, device=device, dtype=torch.float32),
                    "h": torch.arange(1, device=device, dtype=torch.float32),
                    "w": torch.arange(1, device=device, dtype=torch.float32),
                }
                x_ids = ctx_ids = torch.cartesian_prod( x_coords["l"], x_coords["h"], x_coords["w"])
            elif self.rope_input_type == 'hybrid':
                l_len = self.rope_hybrid_l_len
                hw_total = self.max_seq_len - l_len
                h = w = int(np.sqrt(hw_total))
                end_pos = past_len + x_len
                all_ids = []
                sem_start = past_len
                sem_end = min(end_pos, l_len)
                if sem_start < sem_end:
                    n = sem_end - sem_start
                    all_ids.append(torch.stack([
                        torch.arange(sem_start, sem_end, device=device, dtype=torch.float32),
                        torch.zeros(n, device=device, dtype=torch.float32),
                        torch.zeros(n, device=device, dtype=torch.float32),
                    ], dim=1))
                vis_start = max(past_len, l_len)
                if vis_start < end_pos:
                    n = end_pos - vis_start
                    offset = vis_start - l_len
                    idx = torch.arange(offset, offset + n, device=device)
                    all_ids.append(torch.stack([
                        torch.zeros(n, device=device, dtype=torch.float32),
                        (idx // w).float(),
                        (idx % w).float(),
                    ], dim=1))
                x_ids = ctx_ids = torch.cat(all_ids, dim=0)
        elif self.context_type == 'concat':
            if self.rope_input_type == 'hw':
                h = w = int(np.sqrt(z_len))
                x_coords = {
                    "h": torch.arange(h, device=device, dtype=torch.float32),
                    "w": torch.arange(w, device=device, dtype=torch.float32),
                    "l": torch.arange(1, device=device, dtype=torch.float32),
                }
            elif self.rope_input_type == 'l':
                l = z_len
                x_coords = {
                    "h": torch.arange(1, device=device, dtype=torch.float32),
                    "w": torch.arange(1, device=device, dtype=torch.float32),
                    "l": torch.arange(l, device=device, dtype=torch.float32),
                }
            if self.rope_ctx_type == 'hw':
                h = w = int(np.sqrt(x_len))
                ctx_coords = {
                    "h": torch.arange(h, device=device, dtype=torch.float32),
                    "w": torch.arange(w, device=device, dtype=torch.float32),
                    "l": torch.arange(1, device=device, dtype=torch.float32),
                }
            elif self.rope_ctx_type == 'l':
                l = x_len
                ctx_coords = {
                    "h": torch.arange(1, device=device, dtype=torch.float32),
                    "w": torch.arange(1, device=device, dtype=torch.float32),
                    "l": torch.arange(l, device=device, dtype=torch.float32),
                }
            x_ids = torch.cartesian_prod(x_coords["h"], x_coords["w"], x_coords["l"])
            ctx_ids = torch.cartesian_prod(ctx_coords["h"], ctx_coords["w"], ctx_coords["l"])
        elif self.context_type == 'cross_attn':
            if self.rope_input_type == 'hw':
                h = w = int(np.sqrt(z_len))
                x_coords = {
                    "h": torch.arange(h, device=device, dtype=torch.float32),
                    "w": torch.arange(w, device=device, dtype=torch.float32),
                    "l": torch.arange(1, device=device, dtype=torch.float32),
                }
            elif self.rope_input_type == 'l':
                l = z_len
                x_coords = {
                    "h": torch.arange(1, device=device, dtype=torch.float32),
                    "w": torch.arange(1, device=device, dtype=torch.float32),
                    "l": torch.arange(l, device=device, dtype=torch.float32),
                }
            if self.rope_ctx_type == 'hw':
                h = w = int(np.sqrt(x_len))
                ctx_coords = {
                    "h": torch.arange(h, device=device, dtype=torch.float32),
                    "w": torch.arange(w, device=device, dtype=torch.float32),
                    "l": torch.arange(1, device=device, dtype=torch.float32),
                }
            elif self.rope_ctx_type == 'l':
                l = x_len
                ctx_coords = {
                    "h": torch.arange(1, device=device, dtype=torch.float32),
                    "w": torch.arange(1, device=device, dtype=torch.float32),
                    "l": torch.arange(l, device=device, dtype=torch.float32),
                }
            x_ids = torch.cartesian_prod(x_coords["h"], x_coords["w"], x_coords["l"])
            ctx_ids = torch.cartesian_prod(ctx_coords["h"], ctx_coords["w"], ctx_coords["l"])
        return x_ids, ctx_ids
    def forward(self, x, condition, attn_mask=None, cache_kv=False, past_kvs=None, return_attn=False):
        x_len = x.size(1)
        z_len = self.max_seq_len if self.context_type != 'none' else 0
        if self.context_type == 'none':
            x, context_emb = self.input_layer(x), None
        elif self.context_type == 'concat':
            learnable_query = self.learnable_query.expand(x.size(0), -1, -1).to(x.device)
            x = self.input_layer(x)
            x = torch.cat((x, learnable_query), dim=1)
            x, context_emb = x, None
        else:             
            learnable_query = self.learnable_query.expand(x.size(0), -1, -1).to(x.device)
            x, context_emb = learnable_query, self.input_layer(x)
        
        x = self.emb_dropout(x)

        cond_emb = self.cond_proj[0](condition) if (self.cond_proj is not None and condition is not None) else None
        cond_emb = self.emb_dropout(cond_emb) if cond_emb is not None else None
        cond_emb = self.cond_proj[1](cond_emb) if cond_emb is not None else None
        cond_emb = self.cond_proj[2](cond_emb) if cond_emb is not None else None

        if self.abs_pos is not None:
            if self.context_type == 'none':
                pos_ids = torch.arange(x_len, device=x.device)
                if past_kvs is not None:
                    pos_ids = pos_ids + past_kvs[0][0].shape[2]
                x = x + self.abs_pos(pos_ids)
            elif self.context_type == 'concat':
                pos_ids = torch.arange(x_len+z_len, device=x.device)
                x = x + self.abs_pos(pos_ids)
            elif self.context_type == 'cross_attn':
                pos_ids = torch.arange(x_len, device=x.device)
                x = x + self.abs_pos(pos_ids)
                if context_emb is not None and self.ctx_pos is not None:
                    context_emb = context_emb + self.ctx_pos(torch.arange(z_len, device=context_emb.device))
        
        pe = ctx_pe = None
        if self.rope_emb is not None:
            past_len = 0
            if past_kvs is not None:
                past_len = past_kvs[0][0].shape[2]
            x_ids, ctx_ids = self.prepare_pos_ids(x_len, z_len, past_len, x.device)
            pe = self.rope_emb(x_ids)
            if self.context_type != 'none':
                ctx_pe = self.ctx_rope_emb(ctx_ids)
            if self.context_type == 'concat':
                pe = torch.cat([ctx_pe, pe], dim=2)
                ctx_pe = None
                
            if self.bos_zero_rope  and past_len == 0:
                pe = pe.clone()
                pe[:, :, 0, :, 0, 0] = 0 # cos = 0
                pe[:, :, 0, :, 0, 1] = 0 # -sin = 0
                pe[:, :, 0, :, 1, 0] = 0 # sin = 0
                pe[:, :, 0, :, 1, 1] = 0 # cos = 0
    
        attn_mask = self.concat_causal_mask if self.concat_causal else attn_mask

        kv_caches = []
        all_attn_maps = [] if return_attn else None
        for i,layer in enumerate(self.layers):
            layer_output = layer(x, context_emb, cond_emb, self.causal, attn_mask, cache_kv, past_kvs[i] if past_kvs is not None else None, pe=pe, ctx_pe=ctx_pe, return_attn=return_attn)
            if return_attn:
                if cache_kv:
                    x, new_kvs, layer_attn = layer_output
                    kv_caches.append(new_kvs)
                else:
                    x, layer_attn = layer_output
                all_attn_maps.append(layer_attn)
            elif cache_kv:
                x, new_kvs = layer_output[0], layer_output[1]
                kv_caches.append(new_kvs)
            else:
                x = layer_output
                
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            if cond_emb is not None:
                cond_emb = cond_emb.float()
            if self.out_norm is not None:
                x = self.out_norm(x, cond_emb)
            if self.out_activate is not None:   
                x = self.out_activate(x)
            if self.out is not None:
                x = self.out(x)
        
        if return_attn:
            if cache_kv:
                return x, kv_caches, all_attn_maps
            else:
                return x, all_attn_maps
        elif cache_kv:
            return x, kv_caches
        else:
            return x


def insert_eos_token(idx, z_len, eos_id):
    """Insert eos_id at position z_len in idx, returning a tensor with length +1."""
    return torch.cat([idx[:, :z_len],
                      torch.full((idx.shape[0], 1), eos_id, device=idx.device, dtype=idx.dtype),
                      idx[:, z_len:]], dim=1)


@dataclass
class VQLossDetail:
    quant_loss: Tensor
    entropy_loss: Tensor
    sample_entropy: Tensor
    batch_entropy: Tensor
    l2norm_z: Tensor
    l2norm_code: Tensor

    @staticmethod
    def zero(device):
        z = torch.zeros((), device=device)
        return VQLossDetail(z, z, z, z, z, z)

    @staticmethod
    def from_tuple(t):
        return VQLossDetail(*t)


@dataclass
class EncoderOutput:
    quant: Tensor
    indices: Tensor
    one_hot: Tensor
    semantic_vq_loss: VQLossDetail
    visual_vq_loss: VQLossDetail
    semantic_quant: Tensor
    visual_quant: Tensor
    semantic_indices: Tensor
    visual_indices: Tensor
    semantic_one_hot: Tensor = None


@dataclass
class AROutput:
    logits: Tensor
    semantic_logits: Tensor
    visual_logits: Tensor


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.prologue = config.get("Prologue", False)
        self.share_semantic_encoder = config.get("share_semantic_encoder", True)
        self.share_semantic_codebook = config.get("share_semantic_codebook", False)
        self.z_len = config["z_len"]
        self.x_len = config["x_len"]
        if self.prologue:
            if self.share_semantic_encoder and config["Encoder"].get("context_type") != "concat":
                warnings.warn(f"Prologue with shared backbone requires Encoder context_type='concat', got '{config['Encoder'].get('context_type')}'. Forcing to 'concat'.")
                config["Encoder"]["context_type"] = "concat"
            
        self.enc = Transformer(**config["Encoder"])
        self.quantizer = PrologueQuantizer(**config["Quantizer"])

        if self.prologue and not self.share_semantic_codebook:
            self.semantic_quantizer = PrologueQuantizer(**config["SemanticQuantizer"])

        if self.prologue and not self.share_semantic_encoder:
            self.semantic_input_type = config.get("semantic_input_type", "encoder_output")
            self.semantic_enc = Transformer(**config["SemanticEncoder"])

        if not self.prologue or self.share_semantic_codebook:
            self._forward_impl = self._forward_simple
            self._encode_idx_impl = self._encode_idx_simple
        elif self.share_semantic_encoder:
            self._forward_impl = self._forward_split_codebook
            self._encode_idx_impl = self._encode_idx_split_codebook
        else:
            self._forward_impl = self._forward_separate_enc
            self._encode_idx_impl = self._encode_idx_separate_enc

    def forward(self, x, labels, training=False) -> EncoderOutput:
        return self._forward_impl(x, labels, training)

    def encode_idx(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        return self._encode_idx_impl(x, labels)

    # ---- strategy: simple (no Prologue, or Prologue with shared codebook) ----

    def _forward_simple(self, x, labels, training):
        h = self.enc(x, labels)
        if self.enc.context_type == 'concat':
            h = h[:, -self.enc.max_seq_len:]
        quant, idx, one_hot, vqloss = self.quantizer(h, labels, training=training)
        vl = VQLossDetail.from_tuple(vqloss) if isinstance(vqloss, tuple) else VQLossDetail(vqloss, *([torch.zeros((), device=quant.device)] * 5))
        return EncoderOutput(
            quant=quant, indices=idx, one_hot=one_hot,
            semantic_vq_loss=None, visual_vq_loss=vl,
            semantic_quant=None, visual_quant=quant,
            semantic_indices=None, visual_indices=idx,
            semantic_one_hot=None,

        )

    def _encode_idx_simple(self, x, labels):
        z = self.enc(x, labels)
        if self.enc.context_type == 'concat':
            z = z[:, -self.enc.max_seq_len:]
        return self.quantizer.encode(z, labels)

    # ---- strategy: split codebook (Prologue, shared encoder, separate codebooks) ----

    def _forward_split_codebook(self, x, labels, training):
        h = self.enc(x, labels)
        h_v, h_s = h[:, :self.x_len, :], h[:, self.x_len:, :]
        quant_v, idx_v, oh_v, loss_v = self.quantizer(h_v, labels, training=training)
        quant_s, idx_s, oh_s, loss_s = self.semantic_quantizer(h_s, labels, training=training)
        raw_oh_s = oh_s
        idx = torch.cat([idx_s, idx_v], dim=1)
        if oh_s is not None and oh_v is not None:
            max_cb = max(oh_s.shape[-1], oh_v.shape[-1])
            if oh_s.shape[-1] < max_cb:
                oh_s = F.pad(oh_s, (0, max_cb - oh_s.shape[-1]))
            if oh_v.shape[-1] < max_cb:
                oh_v = F.pad(oh_v, (0, max_cb - oh_v.shape[-1]))
            one_hot = torch.cat([oh_s, oh_v], dim=1)
        else:
            one_hot = None
        vl_s = VQLossDetail.from_tuple(loss_s) if isinstance(loss_s, tuple) else VQLossDetail(loss_s, *([torch.zeros((), device=quant_v.device)] * 5))
        vl_v = VQLossDetail.from_tuple(loss_v) if isinstance(loss_v, tuple) else VQLossDetail(loss_v, *([torch.zeros((), device=quant_v.device)] * 5))
        return EncoderOutput(
            quant=quant_v, indices=idx, one_hot=one_hot,
            semantic_vq_loss=vl_s, visual_vq_loss=vl_v,
            semantic_quant=quant_s, visual_quant=quant_v,
            semantic_indices=idx_s, visual_indices=idx_v,
            semantic_one_hot=raw_oh_s,
        )

    def _encode_idx_split_codebook(self, x, labels):
        z = self.enc(x, labels)
        z_v, z_s = z[:, :self.x_len, :], z[:, self.x_len:, :]
        idx_v = self.quantizer.encode(z_v, labels)
        idx_s = self.semantic_quantizer.encode(z_s, labels)
        return torch.cat([idx_s, idx_v], dim=1)

    # ---- strategy: separate encoder (Prologue-Post, independent backbone) ----

    def _forward_separate_enc(self, x, labels, training):
        h = self.enc(x, labels)
        vq_out = self.quantizer(h, labels, training=training,
                                return_continuous=(self.semantic_input_type == "pre_quant"))
        if self.semantic_input_type == "pre_quant":
            quant_v, idx_v, oh_v, loss_v, z_continuous = vq_out
            sem_input = z_continuous
        else:
            quant_v, idx_v, oh_v, loss_v = vq_out
            sem_input = h if self.semantic_input_type == "encoder_output" else x

        h_s = self.semantic_enc(sem_input, labels)
        if self.semantic_enc.context_type == 'concat':
            h_s = h_s[:, -self.semantic_enc.max_seq_len:]
        quant_s, idx_s, oh_s, loss_s = self.semantic_quantizer(h_s, labels, training=training)
        raw_oh_s = oh_s

        idx = torch.cat([idx_s, idx_v], dim=1)
        if oh_s is not None and oh_v is not None:
            max_cb = max(oh_s.shape[-1], oh_v.shape[-1])
            if oh_s.shape[-1] < max_cb:
                oh_s = F.pad(oh_s, (0, max_cb - oh_s.shape[-1]))
            if oh_v.shape[-1] < max_cb:
                oh_v = F.pad(oh_v, (0, max_cb - oh_v.shape[-1]))
            one_hot = torch.cat([oh_s, oh_v], dim=1)
        else:
            one_hot = None
        vl_s = VQLossDetail.from_tuple(loss_s) if isinstance(loss_s, tuple) else VQLossDetail(loss_s, *([torch.zeros((), device=quant_v.device)] * 5))
        vl_v = VQLossDetail.from_tuple(loss_v) if isinstance(loss_v, tuple) else VQLossDetail(loss_v, *([torch.zeros((), device=quant_v.device)] * 5))
        return EncoderOutput(
            quant=quant_v, indices=idx, one_hot=one_hot,
            semantic_vq_loss=vl_s, visual_vq_loss=vl_v,
            semantic_quant=quant_s, visual_quant=quant_v,
            semantic_indices=idx_s, visual_indices=idx_v,
            semantic_one_hot=raw_oh_s,
        )

    def _encode_idx_separate_enc(self, x, labels):
        h = self.enc(x, labels)
        idx_v = self.quantizer.encode(h, labels)
        if self.semantic_input_type == "encoder_output":
            sem_input = h
        elif self.semantic_input_type == "image_patch":
            sem_input = x
        else:
            vq_out = self.quantizer(h, labels, training=False, return_continuous=True)
            sem_input = vq_out[4]
        h_s = self.semantic_enc(sem_input, labels)
        if self.semantic_enc.context_type == 'concat':
            h_s = h_s[:, -self.semantic_enc.max_seq_len:]
        idx_s = self.semantic_quantizer.encode(h_s, labels)
        return torch.cat([idx_s, idx_v], dim=1)

    # ---- helper properties & methods ----

    @property
    def has_separate_semantic(self):
        return self.prologue and not self.share_semantic_encoder

    @property
    def visual_modules(self):
        return ["enc", "quantizer"]

    @property
    def semantic_modules(self):
        return ["semantic_enc", "semantic_quantizer"] if self.has_separate_semantic else []

    @property
    def total_token_len(self):
        if self.prologue and not self.share_semantic_codebook:
            return self.z_len + self.x_len
        return self.z_len

    def get_visual_codes(self, indices, labels):
        if self.prologue and not self.share_semantic_codebook:
            visual_ids = indices[:, -self.x_len:]
            return self.quantizer.get_codes_w_indices(visual_ids, labels)
        return self.quantizer.get_codes_w_indices(indices, labels)

    def visual_parameters(self):
        return chain(self.enc.parameters(), self.quantizer.parameters())

    def semantic_parameters(self):
        return chain(self.semantic_enc.parameters(), self.semantic_quantizer.parameters())


class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dec = Transformer(**config["Decoder"])
        self.is_concat = config["Decoder"].get("context_type", "none") == "concat"
        self.output_len = config["Decoder"]["max_seq_len"]

    def forward(self, quant, labels):
        x_hat = self.dec(quant, labels)
        if self.is_concat:
            x_hat = x_hat[:, -self.output_len:]
        return x_hat


class ARModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ar_config = config["ARModel"]

        self.conditional_injection = ar_config.get("conditional_injection", "llamagen")  # dit | llamagen
        # ste_ar_embedding: use one_hot @ W_emb for the prologue prefix so grads flow into the encoder.
        self.ste_ar_embedding = ar_config.get("ste_ar_embedding", False)
        self.num_classes = ar_config.cond_input_dim
        print0("conditional_injection: ", self.conditional_injection)
        print0("ste_ar_embedding: ", self.ste_ar_embedding)
        print0("num_classes: ", self.num_classes)

        prologue = config.get("Prologue", False) and not config.get("share_semantic_codebook", False)
        self.z_len = int(config.get("z_len", 0)) if prologue else 0

        vis_cb_size = int(config["Quantizer"]["codebook_size"])
        sem_cb_size = int(config["SemanticQuantizer"]["codebook_size"]) if prologue else 0
        self.ar_vocab_size = vis_cb_size + sem_cb_size
        self.semantic_offset = vis_cb_size
        self.semantic_codebook_size = sem_cb_size
        self.visual_codebook_size = vis_cb_size

        use_eos = bool(config.get("use_eos", False))
        self.eos_len = 1 if (use_eos and self.z_len > 0) else 0
        if self.eos_len > 0:
            self.eos_token_id = self.ar_vocab_size
            self.ar_vocab_size += 1
            ar_config['max_seq_len'] = int(ar_config['max_seq_len']) + 1
            if 'rope_hybrid_l_len' in ar_config and int(ar_config['rope_hybrid_l_len']) > 0:
                ar_config['rope_hybrid_l_len'] = int(ar_config['rope_hybrid_l_len']) + 1
            print0(f"use_eos: eos_token_id={self.eos_token_id}, max_seq_len→{ar_config['max_seq_len']}")
        else:
            self.eos_token_id = -1

        print0(f"AR vocab: {self.ar_vocab_size}, semantic_offset: {self.semantic_offset}")

        bos_num = self.num_classes if self.conditional_injection == "llamagen" else 1
        self.bos_emb = nn.Embedding(bos_num, ar_config["dim"])
        nn.init.trunc_normal_(self.bos_emb.weight, std=0.02)

        self.semantic_emb = nn.Embedding(self.ar_vocab_size, ar_config["dim"])
        nn.init.trunc_normal_(self.semantic_emb.weight, std=0.02)
        self.tied_embedding = ar_config.get("tied_embedding", False)
        if self.tied_embedding:
            ar_config['out_layer'] = False
            print0("tied_embedding: True, output layer disabled in Transformer")

        ar_config['output_dim'] = self.ar_vocab_size
        self.ar_model = Transformer(**ar_config)

        self.temperature = ar_config["temperature"]
        self.max_length = ar_config["max_seq_len"]

        uncond_idx = self.num_classes - 1
        self.register_buffer(
            'uncond_ar_labels',
            F.one_hot(torch.tensor([uncond_idx], dtype=torch.long), num_classes=self.num_classes).float()
        )

        self.register_buffer('logit_mask', None, persistent=False)

    def forward(self, idx, labels, temperature=None, semantic_one_hot=None, return_attn=False) -> AROutput:
        bz = idx.shape[0]

        if self.conditional_injection == "llamagen":
            bos = self.bos_emb(torch.argmax(labels, dim=1)).unsqueeze(1)
            labels = self.uncond_ar_labels.expand(bz, -1).to(device=labels.device, dtype=labels.dtype)
        else:
            bos = self.bos_emb(torch.zeros(bz, device=idx.device, dtype=torch.long)).unsqueeze(1)

        # STE: prefix embedding via one_hot @ W_emb so grads reach the encoder.
        if self.ste_ar_embedding and semantic_one_hot is not None and self.z_len > 0:
            sem_weight = self.semantic_emb.weight[self.semantic_offset:self.semantic_offset + self.semantic_codebook_size]
            sem_token_emb = semantic_one_hot @ sem_weight
            rest_token_emb = self.semantic_emb(idx[:, self.z_len:-1])
            token_emb = torch.cat([sem_token_emb, rest_token_emb], dim=1)
        elif self.ste_ar_embedding and semantic_one_hot is not None:
            weight = self.semantic_emb.weight[:self.visual_codebook_size]
            token_emb = semantic_one_hot[:, :-1] @ weight
        else:
            token_emb = self.semantic_emb(idx[:, :-1])

        shift_input = torch.cat([bos, token_emb], dim=1)
        result = self.ar_model(shift_input, labels, return_attn=return_attn)
        if return_attn:
            out, all_attn_maps = result
        else:
            out = result
        logits = F.linear(out, self.semantic_emb.weight) if self.tied_embedding else out
        
        if temperature is not None:
            logits = logits / temperature
        elif self.temperature is not None:
            logits = logits / self.temperature

        if self.z_len > 0:
            semantic_logits = logits[:, :self.z_len]
            visual_logits = logits[:, self.z_len + self.eos_len:]
        else:
            semantic_logits = logits
            visual_logits = logits
        ar_output = AROutput(logits=logits, semantic_logits=semantic_logits, visual_logits=visual_logits)
        if return_attn:
            return ar_output, all_attn_maps
        return ar_output

    def set_logit_mask(self, mask):
        """Install per-step ``logit_mask`` buffer; pads EOS row/slot at ``z_len`` when ``use_eos``."""
        if self.eos_len > 0:
            if mask is not None:
                mask = F.pad(mask, (0, 1), value=float('-inf'))
                eos_row = torch.full((1, self.ar_vocab_size), float('-inf'))
                eos_row[0, self.eos_token_id] = 0.
                mask = torch.cat([mask[:self.z_len], eos_row, mask[self.z_len:]], dim=0)
            else:
                mask = torch.zeros(self.max_length, self.ar_vocab_size)
                mask[:, self.eos_token_id] = float('-inf')
                mask[self.z_len] = float('-inf')
                mask[self.z_len, self.eos_token_id] = 0.
        self.register_buffer('logit_mask', mask, persistent=False)

    @torch.no_grad()
    @torch._dynamo.disable
    def sampling(self, bz, class_label=None, temperature=1.0, topK=None, topP=None, cfg=16.0, cfg_schedule='cosine', cfg_power=2.75, cache_kv=False,
                 semantic_cfg_schedule=None, semantic_cfg_scale=None, semantic_cfg_power=None, semantic_cfg_start=0.0,
                 visual_cfg_schedule=None, visual_cfg_scale=None, visual_cfg_power=None, visual_cfg_start=1.0,
                 semantic_temperature=None):
        cfg = 0. if class_label is None else cfg

        use_segmented = (semantic_cfg_schedule is not None or visual_cfg_schedule is not None)
        use_cfg = cfg > 0.
        if use_segmented:
            _ss = semantic_cfg_scale if semantic_cfg_scale is not None else cfg
            _vs = visual_cfg_scale if visual_cfg_scale is not None else cfg
            use_cfg = _ss > 0. or _vs > 0.

        uncond_idx = int(self.ar_model.cond_input_dim) - 1
        device = self.bos_emb.weight.device
        
        if self.conditional_injection == "llamagen":
            cond_bos = self.bos_emb(torch.argmax(class_label, dim=1)).unsqueeze(1)  # [B, 1, D]
            uncond_bos = self.bos_emb(torch.full((bz,), uncond_idx, device=device, dtype=torch.long)).unsqueeze(1)
            ar_labels = self.uncond_ar_labels.expand(bz, -1).to(device=device)  # [B, num_classes]
            uncond_labels = self.uncond_ar_labels.expand(bz, -1).to(device=device)
        else:
            cond_bos = self.bos_emb(torch.zeros(bz, device=device, dtype=torch.long)).unsqueeze(1)
            uncond_bos = cond_bos 
            ar_labels = class_label
            uncond_labels = self.uncond_ar_labels.expand(bz, -1).to(device=device)

        quant_input = torch.cat([cond_bos, uncond_bos], dim=0) if use_cfg else cond_bos
        ar_labels = torch.cat([ar_labels, uncond_labels], dim=0) if use_cfg else ar_labels
        quant_output = []
        past_kvs = None
        
        for step in range(self.max_length):
            # CFG
            if use_cfg:
                ar_out = self.ar_model(quant_input, ar_labels, cache_kv=cache_kv, past_kvs=past_kvs)
                if cache_kv:
                    hidden_all, past_kvs = ar_out
                else:
                    hidden_all = ar_out
                
                if self.tied_embedding:
                    hidden_all = F.linear(hidden_all[:, -1:], self.semantic_emb.weight)

                logits_all = hidden_all[:, -1]
                logits, uncond_logits = logits_all.chunk(2, dim=0)
                
                is_semantic_step = (self.z_len > 0 and step < self.z_len)

                if use_segmented:
                    if is_semantic_step:
                        seg_schedule = semantic_cfg_schedule or 'constant'
                        seg_scale = semantic_cfg_scale if semantic_cfg_scale is not None else cfg
                        seg_power = semantic_cfg_power if semantic_cfg_power is not None else cfg_power
                        seg_start = semantic_cfg_start
                        seg_t = step / self.z_len if self.z_len > 0 else 0.0
                    else:
                        seg_schedule = visual_cfg_schedule or 'constant'
                        seg_scale = visual_cfg_scale if visual_cfg_scale is not None else cfg
                        seg_power = visual_cfg_power if visual_cfg_power is not None else cfg_power
                        seg_start = visual_cfg_start
                        visual_start = self.z_len
                        visual_len = self.max_length - visual_start
                        seg_t = (step - visual_start) / visual_len if visual_len > 0 else 0.0

                    if seg_schedule == 'constant':
                        cfg_scale = seg_scale
                    elif seg_schedule == 'linear':
                        cfg_scale = seg_start + (seg_scale - seg_start) * seg_t
                    elif seg_schedule == 'cosine':
                        shape = (1 - math.cos(
                            (seg_t ** seg_power) * math.pi)) * 0.5
                        cfg_scale = seg_start + (seg_scale - seg_start) * shape
                    else:
                        raise ValueError(f"Invalid segmented cfg schedule: {seg_schedule}")
                elif cfg_schedule == 'constant':
                    cfg_scale = cfg
                elif cfg_schedule == 'linear':
                    cfg_scale = 1.0 * (1-step/self.max_length) + cfg * (step/self.max_length)
                elif cfg_schedule == 'cosine':
                    cfg_scale = (1 - math.cos(
                        ((step / self.max_length) ** cfg_power) * math.pi)) * 1/2
                    cfg_scale = (cfg - 1) * cfg_scale + 1
                else:
                    raise ValueError(f"Invalid cfg_schedule: {cfg_schedule}")
                
                logits = cfg_scale * logits + (1 - cfg_scale) * uncond_logits
            else:
                ar_out = self.ar_model(quant_input, ar_labels, cache_kv=cache_kv, past_kvs=past_kvs)
                if cache_kv:
                    hidden, past_kvs = ar_out
                else:
                    hidden = ar_out
                if self.tied_embedding:
                    hidden = F.linear(hidden[:, -1:], self.semantic_emb.weight)
                logits = hidden[:, -1]

            is_semantic_step_temp = (self.z_len > 0 and step < self.z_len)
            t = semantic_temperature if (semantic_temperature is not None and is_semantic_step_temp) else temperature
            logits = logits / t

            if self.logit_mask is not None:
                logits = logits + self.logit_mask[step]

            # Top-K filtering
            if topK is not None and topK > 0.:
                top_logits, top_indices = logits.topk(topK, dim=-1)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(dim=-1, index=top_indices, src=top_logits)
            
            # Top-P (nucleus) filtering
            if topP is not None and 0. < topP < 1.:
                sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                probs_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = probs_sum > topP
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = float('-inf')
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            
            # Sample the next token
            with torch.amp.autocast("cuda", enabled=False):
                next_idx = torch.multinomial(F.softmax(logits.float(), dim=-1), 1)
            next_idx = next_idx.to(dtype=torch.long)
            quant_output.append(next_idx)
        
            # Feed the new token back as the next input embedding
            next_emb = self.semantic_emb(next_idx)

            if use_cfg:
                next_emb = torch.cat([next_emb, next_emb], dim=0)  # [2B, 1, D]
            
            if not cache_kv:
                quant_input = torch.cat((quant_input, next_emb), dim=1)
            else:
                quant_input = next_emb
        quant_output = torch.cat(quant_output, dim=1)
        return quant_output

# deterministic Hard-Gumbel Softmax Quantizer   
class L2Normalize(nn.Module):
    """Wrapper for F.normalize to make it a proper nn.Module"""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x, element_wise=False):
        if not element_wise:
            return F.normalize(x, dim=self.dim)
        else:
            dtype_one = torch.Tensor([1.0]).to(x.device).to(x.dtype)
            return torch.where(x>0, dtype_one, -dtype_one)

class PrologueQuantizer(nn.Module):
    """VQ with optional STE variants (``VQ`` / ``LFQ`` / ``ProbQ``) and per-position ``pos_select_mask``."""

    def __init__(self, codebook_size, dim, z_dim, cond_dim,
                codebook_init='trunc_normal',
                z_norm_type='none',  # AdaLN, AdaRMSNorm, FixLN, FixRMSNorm, l2, none
                temperature=1.0,
                frozen_codebook=False, codebook_proj='none',
                ste='ProbQ',
                length=256,
                vq_beta=0.25, use_quant_loss=False, use_entropy_loss=False, monitor_metrics=True,
                 **params):
        super().__init__()
        self.monitor_metrics = monitor_metrics
        self.codebook_size = codebook_size  # codebook size (K)
        self.dim = dim                       # embedding dimension (D)
        self.z_dim = z_dim
        self.length = int(length)

        self.proj_in = Linear(dim, z_dim)
        elementwise_affine = True
        cond_based_affine = False
        self.cond_proj = None
        if 'Ada' in z_norm_type:
            self.cond_proj = nn.Sequential(Linear(1001, cond_dim), AdaRMSNorm(cond_dim, cond_based_affine=False), nn.GELU(approximate='tanh'))
            elementwise_affine = True
            cond_based_affine = True
        if 'Fix' in z_norm_type:
            cond_based_affine = False
            elementwise_affine = False
        self.z_norm_type = z_norm_type
        self.z_norm = nn.Identity()
        if 'l2' in z_norm_type:
            self.z_norm = L2Normalize(dim=-1)
        elif 'RMSNorm' in z_norm_type:
            self.z_norm = AdaRMSNorm(z_dim, cond_dim, elementwise_affine=elementwise_affine, cond_based_affine=cond_based_affine)
        elif 'LN' in z_norm_type:
            self.z_norm = AdaLN(z_dim, cond_dim, elementwise_affine=elementwise_affine, cond_based_affine=cond_based_affine)
        if 'LFQ' in z_norm_type:
            self.z_norm = L2Normalize(dim=-1, element_wise=True)

        self.codebook_init = codebook_init
        if self.codebook_init == 'LFQ':
            self.register_buffer("indices_map", 2**torch.arange(z_dim).view(1, 1, -1))  # B L D
            self.register_buffer("embedding", self.indices_to_emb(torch.arange(codebook_size).view(-1, 1, 1)).view(-1, z_dim))  # N D
        else:
            self.embedding = nn.Embedding(codebook_size, z_dim)
            if codebook_init == 'trunc_normal':
                torch.nn.init.trunc_normal_(self.embedding.weight, std=0.02)
            elif codebook_init == 'uniform':
                self.embedding.weight.data.uniform_(-np.sqrt(1 / codebook_size), np.sqrt(1 / codebook_size))
            elif codebook_init == 'simvq':
                self.embedding.weight.data.normal_(mean=0, std=self.z_dim**-0.5)
        if frozen_codebook:
            self.embedding.requires_grad_(False)

        self.codebook_proj = nn.Identity()
        if codebook_proj == 'linear':
            self.codebook_proj = Linear(z_dim, z_dim)
        elif codebook_proj == 'norm_linear':
            self.codebook_proj = nn.Sequential(AdaRMSNorm(z_dim, elementwise_affine=True, cond_based_affine=False), Linear(z_dim, z_dim))
        elif codebook_proj == 'linear_norm':
            self.codebook_proj = nn.Sequential(Linear(z_dim, z_dim), AdaRMSNorm(z_dim, elementwise_affine=True, cond_based_affine=False))
        elif codebook_proj == 'mlp':
            self.codebook_proj = nn.Sequential(Linear(z_dim, dim), nn.GELU(approximate='tanh'), Linear(dim, z_dim))
        elif codebook_proj == 'mlp_with_norm':
            self.codebook_proj = nn.Sequential(Linear(z_dim, dim), AdaRMSNorm(dim, elementwise_affine=True, cond_based_affine=False), nn.GELU(approximate='tanh'), Linear(dim, z_dim))
        elif codebook_proj == 'l2':
            self.codebook_proj = L2Normalize(dim=-1)

        self.ste = ste
        self.vq_beta = vq_beta
        self.use_quant_loss = use_quant_loss
        self.use_entropy_loss = use_entropy_loss
        self.temperature = temperature
        print0(f"z_norm_type: {z_norm_type}")
        print0(f"codebook_proj: {codebook_proj}")
        print0(f"codebook_init: {codebook_init}")
        print0(f"frozen_codebook: {frozen_codebook}")
        print0(f"temperature: {temperature}")
        print0(f"ste: {ste}")

        # All-zero pos_select_mask; AR sampling composes a global mask (see utils.build_ar_logit_mask).
        mask_bool = torch.ones(self.length, codebook_size, dtype=torch.bool)
        self.register_buffer('pos_select_mask', torch.where(mask_bool, 0., float('-inf')))

    def indices_to_emb(self, indices):
        return ((indices.int() & self.indices_map) != 0).float() * 2. - 1.

    @property
    def codebook(self):
        if self.codebook_init == 'LFQ':
            return self.embedding
        else:
            return self.embedding.weight

    def _get_pos_mask(self, L):
        mask = self.pos_select_mask
        return mask[:L] if L < mask.shape[0] else mask

    @autocast("cuda", enabled=False)
    def forward(self, x, labels=None, training=False, log_usage=False, indices=None, return_continuous=False):
        B, L, D = x.shape
        x = x.float()
        labels = self.cond_proj(labels.float()) if self.cond_proj is not None and labels is not None else labels

        z = self.proj_in(x)
        z_normed = self.z_norm(z, labels) if 'Ada' in self.z_norm_type else self.z_norm(z)

        codebook = self.codebook
        codebook_normed = self.codebook_proj(codebook)

        logits = torch.einsum('bld,nd->bln', z_normed, codebook_normed)
        prob = F.softmax(logits / self.temperature, dim=-1)

        pos_mask = self._get_pos_mask(L)
        indices = torch.argmax(prob + pos_mask, dim=-1) if indices is None else indices

        one_hot_ng = F.one_hot(indices, self.codebook_size).view(B, L, -1).to(z.device).to(z.dtype)
        one_hot = prob + (one_hot_ng - prob).detach()

        if self.ste == 'VQ':
            quant = codebook_normed[indices]
            quant_ste = z_normed + (quant - z_normed).detach()
        elif self.ste == 'LFQ':
            quant = codebook[indices]
            quant_ste = z + (quant - z).detach()
        elif self.ste == 'ProbQ':
            quant_ste = torch.einsum('bln,nd->bld', one_hot, codebook_normed)

        quant_loss = torch.tensor(0., device=z.device, dtype=z.dtype)
        if self.monitor_metrics or (training and self.use_quant_loss):
            if self.ste == 'VQ':
                quant_loss = self.vq_beta * (quant.detach() - z_normed).pow(2).mean() + (quant - z_normed.detach()).pow(2).mean()
            elif self.ste == 'LFQ':
                quant_loss = self.vq_beta * torch.mean((quant.detach() - z)**2) + torch.mean((quant - z.detach())**2)
            elif self.ste == 'ProbQ':
                quant_ng = torch.einsum('bln,nd->bld', one_hot_ng, codebook_normed)
                quant_loss = torch.mean((quant_ste - z_normed)**2) + self.vq_beta * torch.mean((quant_ng.detach() - z_normed)**2) + torch.mean((quant_ng - z_normed.detach())**2)
            quant_loss = quant_loss.detach() if not (training and self.use_quant_loss) else quant_loss

        sample_entropy = torch.tensor(0., device=z.device, dtype=z.dtype)
        batch_entropy = torch.tensor(0., device=z.device, dtype=z.dtype)
        entropy_loss = torch.tensor(0., device=z.device, dtype=z.dtype)
        if self.monitor_metrics or (training and self.use_entropy_loss):
            masked_logits = logits + pos_mask
            sample_entropy, batch_entropy, entropy_loss = compute_entropy_loss(logits=masked_logits.reshape(-1, self.codebook_size))
            sample_entropy = sample_entropy.detach() if not (training and self.use_entropy_loss) else sample_entropy
            batch_entropy = batch_entropy.detach() if not (training and self.use_entropy_loss) else batch_entropy
            entropy_loss = entropy_loss.detach() if not (training and self.use_entropy_loss) else entropy_loss

        l2norm_code = torch.tensor(0., device=z.device, dtype=z.dtype)
        l2norm_z = torch.tensor(0., device=z.device, dtype=z.dtype)
        if self.monitor_metrics:
            l2norm_code = torch.norm(codebook_normed, p=2, dim=-1).mean().detach()
            l2norm_z = torch.norm(z_normed, p=2, dim=-1).mean().detach()

        loss_tuple = (quant_loss, entropy_loss, sample_entropy, batch_entropy, l2norm_z, l2norm_code)
        if return_continuous:
            return quant_ste, indices, one_hot, loss_tuple, z_normed
        return quant_ste, indices, one_hot, loss_tuple

    @autocast("cuda", enabled=False)
    def encode(self, x: torch.Tensor, labels=None):
        B, L, D = x.shape
        x = x.float()
        labels = self.cond_proj(labels.float()) if self.cond_proj is not None and labels is not None else labels

        z = self.proj_in(x)
        z_normed = self.z_norm(z, labels) if 'Ada' in self.z_norm_type else self.z_norm(z)

        codebook = self.codebook
        codebook_normed = self.codebook_proj(codebook)
        logits = torch.einsum('bld,nd->bln', z_normed, codebook_normed)
        indices = torch.argmax(logits + self._get_pos_mask(L), dim=-1)
        return indices

    @autocast("cuda", enabled=False)
    def get_codes_w_indices(self, indices, labels=None, **params):
        if self.codebook_init == 'LFQ':
            codes = self.indices_to_emb(indices)
        else:
            codes = self.embedding(indices)
            codes = self.codebook_proj(codes)
        return codes


def compute_entropy_loss(
    logits,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """Entropy loss on logits (affinities over the last dim); from MAGVIT (Yu et al., 2024)."""
    with torch.amp.autocast("cuda", enabled=False):
        probs = F.softmax(logits.float() / temperature, -1)
        log_probs = F.log_softmax(logits.float() / temperature + eps, -1)
    probs = probs.to(logits.dtype)
    log_probs = log_probs.to(logits.dtype)

    avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))
    
    sample_entropy = -torch.sum(torch.nan_to_num(probs * log_probs, nan=0.0), -1)
    sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss
