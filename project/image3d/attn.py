import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import todos
import pdb


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=8, input_dim=3):
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.num_freqs = num_freqs
        self.out_dim = input_dim * (self.num_freqs * 2 + 1)  # 51

    def forward(self, x):
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)


class MLP(nn.Module):
    def __init__(self, width, expand_ratio = 4, output_width = None):
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, heads=16, n_data=512, width=1024):
        super().__init__()
        self.heads = heads
        self.q_norm = nn.LayerNorm(width // heads, elementwise_affine=True, eps=1e-6)
        self.k_norm = nn.LayerNorm(width // heads, elementwise_affine=True, eps=1e-6)

    def forward(self, q, kv):
        # tensor [q] size: [1, 8000, 1024], min: -8.914062, max: 8.632812, mean: -0.044021
        # tensor [kv] size: [1, 512, 2048], min: -8.289062, max: 7.117188, mean: -0.008912
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2

        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # q, k, v is tuple: len = 3
        #     tensor [item] size: [1, 8000, 16, 64], min: -8.539062, max: 7.707031, mean: 0.039068
        #     tensor [item] size: [1, 512, 16, 64], min: -9.210938, max: 9.054688, mean: -0.000345
        #     tensor [item] size: [1, 512, 16, 64], min: -7.964844, max: 7.097656, mean: 0.005364
        q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d", h=self.heads), (q, k, v))
        # q, k, v is tuple: len = 3
        #     tensor [item] size: [1, 16, 8000, 64], min: -8.539062, max: 7.707031, mean: 0.039068
        #     tensor [item] size: [1, 16, 512, 64], min: -9.210938, max: 9.054688, mean: -0.000345
        #     tensor [item] size: [1, 16, 512, 64], min: -7.964844, max: 7.097656, mean: 0.005364
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        # tensor [out] size: [1, 8000, 1024], min: -7.214844, max: 5.949219, mean: 0.02275

        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(self, width=1024, heads=16, n_data=512, data_width=1024):
        super().__init__()
        self.c_q = nn.Linear(width, width, bias=False)
        self.c_kv = nn.Linear(data_width, width * 2, bias=False)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(heads=heads, n_data=n_data, width=width)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, n_data=512, width=1024, heads=16, mlp_expand_ratio=4, data_width=1024):
        super().__init__()
        self.attn = MultiheadCrossAttention(n_data=n_data, width=width, heads=heads, data_width=data_width)
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = nn.LayerNorm(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x, data):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, heads = 16, n_ctx = 512, width=1024):
        super().__init__()
        self.heads = heads
        self.q_norm = nn.LayerNorm(width // heads, elementwise_affine=True, eps=1e-6)
        self.k_norm = nn.LayerNorm(width // heads, elementwise_affine=True, eps=1e-6)

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        # q, k, v is tuple: len = 3
        #     tensor [item] size: [1, 512, 16, 64], min: -6.835938, max: 6.566406, mean: -0.002259
        #     tensor [item] size: [1, 512, 16, 64], min: -6.929688, max: 7.53125, mean: 0.000862
        #     tensor [item] size: [1, 512, 16, 64], min: -3.498047, max: 3.671875, mean: -0.000247
        q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d", h=self.heads), (q, k, v))
        # q, k, v is tuple: len = 3
        #     tensor [item] size: [1, 16, 512, 64], min: -6.835938, max: 6.566406, mean: -0.002259
        #     tensor [item] size: [1, 16, 512, 64], min: -6.929688, max: 7.53125, mean: 0.000862
        #     tensor [item] size: [1, 16, 512, 64], min: -3.498047, max: 3.671875, mean: -0.000247
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx = 512, width = 1024, heads = 16):
        super().__init__()
        self.c_qkv = nn.Linear(width, width * 3, bias=False)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, width=width)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_ctx = 512, width = 1024, heads = 16):
        super().__init__()
        self.attn = MultiheadAttention(n_ctx=n_ctx, width=width, heads=heads)
        self.ln_1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width)
        self.ln_2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
        n_ctx = 512,
        width = 1024,
        layers = 16,
        heads = 16,
    ):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(n_ctx=n_ctx, width=width, heads=heads) for _ in range(layers)
            ]
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(self, num_latents=512, out_channels=1, width=1024, heads=16, mlp_expand_ratio=4):
        super().__init__()
        self.fourier_embedder = FourierEmbedder()
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents, width=width, mlp_expand_ratio=mlp_expand_ratio, heads=heads
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)

    def forward(self, queries=None, latents=None):
        # tensor [queries] size: [1, 8000, 3], min: -1.009766, max: 1.009766, mean: -0.658704
        # tensor [latents] size: [1, 512, 1024], min: -369.75, max: 36.4375, mean: 0.016268
        query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        x = self.cross_attn_decoder(query_embeddings, latents)
        x = self.ln_post(x)
        occ = self.output_proj(x)

        # tensor [occ] size: [1, 8000, 1], min: -1.000977, max: -0.999023, mean: -0.999919

        return occ
