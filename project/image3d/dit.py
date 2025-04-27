import os
import math
import torch
from torch import nn

from einops import rearrange
import todos
import pdb


def attention(q, k, v):
    x = nn.functional.scaled_dot_product_attention(q, k, v)
    # tensor [x] size: [2, 16, 1882, 64], min: -7.289062, max: 8.179688, mean: 0.010608
    x = rearrange(x, "B H L D -> B L (H D)")
    # tensor [x] size: [2, 1882, 1024], min: -7.289062, max: 8.179688, mean: 0.010608
    return x


def timestep_embedding(t, dim=256, max_period=10000, time_factor: float = 1000.0):
    assert time_factor == 1000

    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
    # ============================================================
    # math.log(max_period) === 6.907755278982137
    # math.log(max_period)/half === 0.053966838117047944
    # ============================================================
    # freqs.size() -- [128]
    # freqs
    # tensor([1.000000, 0.947464, 0.897687, 0.850526, 0.805842, 0.763506, 0.723394,
    #         0.685390, 0.649382, 0.615265, 0.582942, 0.552316, 0.523299, 0.495807,
    #         0.469759, 0.445079, 0.421697, 0.399542, 0.378552, 0.358664, 0.339821,
    #         0.321968, 0.305053, 0.289026, 0.273842, 0.259455, 0.245824, 0.232910,
    #         0.220673, 0.209080, 0.198096, 0.187688, 0.177828, 0.168485, 0.159634,
    #         0.151247, 0.143301, 0.135773, 0.128640, 0.121881, 0.115478, 0.109411,
    #         0.103663, 0.098217, 0.093057, 0.088168, 0.083536, 0.079148, 0.074989,
    #         0.071050, 0.067317, 0.063780, 0.060430, 0.057255, 0.054247, 0.051397,
    #         0.048697, 0.046138, 0.043714, 0.041418, 0.039242, 0.037180, 0.035227,
    #         0.033376, 0.031623, 0.029961, 0.028387, 0.026896, 0.025483, 0.024144,
    #         0.022876, 0.021674, 0.020535, 0.019456, 0.018434, 0.017466, 0.016548,
    #         0.015679, 0.014855, 0.014075, 0.013335, 0.012635, 0.011971, 0.011342,
    #         0.010746, 0.010182, 0.009647, 0.009140, 0.008660, 0.008205, 0.007774,
    #         0.007365, 0.006978, 0.006612, 0.006264, 0.005935, 0.005623, 0.005328,
    #         0.005048, 0.004783, 0.004532, 0.004294, 0.004068, 0.003854, 0.003652,
    #         0.003460, 0.003278, 0.003106, 0.002943, 0.002788, 0.002642, 0.002503,
    #         0.002371, 0.002247, 0.002129, 0.002017, 0.001911, 0.001811, 0.001715,
    #         0.001625, 0.001540, 0.001459, 0.001382, 0.001310, 0.001241, 0.001176,
    #         0.001114, 0.001055], device='cuda:0')
    # pp t.size() -- [2], t[:, None].size() -- [2, 1]
    args = t[:, None].float() * freqs[None]
    # args.size() -- [2, 128]
    # torch.cos(args).size() -- [2, 128]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: # False
        pdb.set_trace()
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):  # True
        embedding = embedding.to(t)

    # tensor [embedding] size: [2, 256], min: 0.0, max: 1.0, mean: 0.5
    return embedding


class GELU(nn.Module):
    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return nn.functional.gelu(x.contiguous(), approximate=self.approximate)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=1024):
        super().__init__()
        assert in_dim == 256
        assert hidden_dim == 1024

        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads = 8):
        super().__init__()
        # self.num_heads = num_heads
        # head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # !!!!!!!!!!!!!!! useless, place holder ...

        # qkv = self.qkv(x)

        # todos.debug.output_var("qkv", qkv)
        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # todos.debug.output_var("q, k, v", (q, k, v))

        # q, k = self.norm(q, k, v)
        # x = attention(q, k, v)
        # x = self.proj(x)
        return x


class SingleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.multiplier = 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)
        return out[0], out[1], out[2] # shift, scale, gate


class DoubleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.multiplier = 6
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)
        # len(out) == 6, ==> out[0].size() -- [2, 1, 1024]
        return out[0], out[1], out[2], out[3], out[4], out[5] # shift, scale, gate; shift, scale2, gate2

class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        assert hidden_size == 1024
        assert num_heads == 16
        assert mlp_ratio == 4

        mlp_hidden_dim = int(hidden_size * mlp_ratio) # 4096 ?
        self.num_heads = num_heads
        self.img_mod = DoubleModulation(hidden_size)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = DoubleModulation(hidden_size)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img, txt, vec):
        #shift, scale, gate
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = self.img_mod(vec)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        img_qkv = self.img_attn.qkv(img_modulated)

        # self.num_heads -- 16
        # tensor [img_qkv] size: [2, 512, 3072], min: -6.292969, max: 5.761719, mean: 0.01594
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # img_q, img_k, img_v is tuple: len = 3
        #     tensor [item] size: [2, 16, 512, 64], min: -24.125, max: 23.796875, mean: 0.132935
        #     tensor [item] size: [2, 16, 512, 64], min: -28.15625, max: 28.09375, mean: 0.04367
        #     tensor [item] size: [2, 16, 512, 64], min: -21.53125, max: 24.40625, mean: 0.032035

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        # tensor [txt_qkv] size: [2, 512, 3072], min: -6.292969, max: 5.761719, mean: 0.01594

        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # txt_q, txt_k, txt_v is tuple: len = 3
        #     tensor [item] size: [2, 16, 1370, 64], min: -22.9375, max: 20.359375, mean: 0.026992
        #     tensor [item] size: [2, 16, 1370, 64], min: -14.015625, max: 28.53125, mean: -0.0224
        #     tensor [item] size: [2, 16, 1370, 64], min: -13.148438, max: 13.695312, mean: 0.007059
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp((1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift)

        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp((1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = GELU(approximate="tanh")
        self.modulation = SingleModulation(hidden_size)

    def forward(self, x, vec):
        mod_shift, mod_scale, mod_gate = self.modulation(vec)

        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        # tensor [qkv] size: [2, 1882, 3072], min: -20.578125, max: 22.84375, mean: 0.003015
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)  # self.num_heads == 16
        # q, k, v is tuple: len = 3
        #     tensor [item] size: [2, 16, 1882, 64], min: -12.992188, max: 14.09375, mean: -0.001616
        #     tensor [item] size: [2, 16, 1882, 64], min: -17.484375, max: 22.84375, mean: 0.001019
        #     tensor [item] size: [2, 16, 1882, 64], min: -20.578125, max: 20.78125, mean: 0.009641
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class Hunyuan3DDiT(nn.Module):
    """
      target: hy3dgen.shapegen.models.Hunyuan3DDiT
      params:
        in_channels: 64
        context_in_dim: 1536
        hidden_size: 1024
        mlp_ratio: 4.0
        num_heads: 16
        depth: 8
        depth_single_blocks: 16
        axes_dim: [ 64 ]
        theta: 10000
        qkv_bias: true
        guidance_embed: false    
    """

    def __init__(
        self,
        in_channels=64,
        context_in_dim=1536,
        hidden_size=1024,
        mlp_ratio=4.0,
        num_heads=16,
        depth=8,  # 16
        depth_single_blocks=16,  # 32
        time_factor=1000,
    ):
        super().__init__()
        self.time_factor = time_factor
        self.latent_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        self.cond_in = nn.Linear(context_in_dim, hidden_size)

        # len(self.double_blocks) === 8
        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]  # 8 ?
        )

        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_single_blocks)]
        )
        self.final_layer = LastLayer(hidden_size, 1, in_channels)

        self.load_weights()

    def forward(self, x, t, cond):
        # todos.debug.output_var("x, t, cond", (x, t, cond))

        # tensor [x] size: [2, 512, 64], min: -4.179688, max: 4.238281, mean: 0.005243
        # tensor [t] size: [2], min: 0.0, max: 0.0, mean: 0.0
        # tensor [cond] size: [2, 1370, 1536], min: -15.28125, max: 14.375, mean: -0.009306
        # --------------------------------------------------------------------------------
        latent = self.latent_in(x)
        # tensor [latent] size: [2, 512, 1024], min: -6.734375, max: 6.480469, mean: -0.002356

        # self.time_factor --- 1000
        # timestep_embedding(t, 256, self.time_factor).size() -- [2, 256]
        vec = self.time_in(timestep_embedding(t, 256, self.time_factor).to(dtype=latent.dtype))
        # tensor [vec] size: [2, 1024], min: -0.274414, max: 5.097656, mean: 0.028905

        cond = self.cond_in(cond)
        # tensor [cond] size: [2, 1370, 1024], min: -143.375, max: 155.125, mean: -0.016774

        for block in self.double_blocks:  # len(self.double_blocks) === 8
            latent, cond = block(img=latent, txt=cond, vec=vec)

        # tensor [cond] size: [2, 1370, 1024], min: -263.0, max: 3560.0, mean: 0.430395
        # tensor [latent] size: [2, 512, 1024], min: -93.375, max: 106.9375, mean: -0.064714
        latent = torch.cat((cond, latent), 1)
        # tensor [latent] size: [2, 1882, 1024], min: -263.0, max: 3560.0, mean: 0.2957

        for block in self.single_blocks:  # len(self.single_blocks) === 16
            latent = block(latent, vec=vec)

        # tensor [latent] size: [2, 1882, 1024], min: -207.5, max: 3752.0, mean: 0.260497
        latent = latent[:, cond.shape[1] :, ...]
        # tensor [latent] size: [2, 512, 1024], min: -64.3125, max: 203.875, mean: -0.013699

        latent = self.final_layer(latent, vec)
        # tensor [latent] size: [2, 512, 64], min: -4.257812, max: 4.207031, mean: -0.006032

        return latent

    def load_weights(self, model_path="models/image3d_dit.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)
