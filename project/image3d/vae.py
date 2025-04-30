"""SDXL 1.0 Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import todos
# import ggml_engine
import pdb

class AutoencoderKL(nn.Module):
    def __init__(self, latent_channels = 4):
        super().__init__()
        self.scaling_factor = 0.18215
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.load_weights()

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # tensor [moments] size: [1, 8, 128, 128], min: -27.390625, max: 26.421875, mean: -10.786383

        # Create DiagonalGaussianDistribution
        meanvar, logvar = torch.chunk(moments, 2, dim=1)
        # meanvar.size() -- [1, 4, 128, 128]
        # logvar.size()  -- [1, 4, 128, 128]
        logvar = torch.clamp(logvar, -30.0, 20.0)
        stdvar = torch.exp(0.5 * logvar)
        output = meanvar + stdvar * torch.randn(meanvar.shape).to(device=x.device)

        #tensor [output] size: [1, 4, 128, 128], min: -16.906248, max: 26.421879, mean: 0.430516
        return output * self.scaling_factor


    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x):
        # uselesss, placeholder ...
        z = self.encode(x)
        return self.decode(z)

    def load_weights(self, model_path="models/image3d_vae.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)

class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_downsample: bool = True,
    ):
        super().__init__()
        # assert in_channels == 128 or ...
        # assert out_channels == 128 or ...
        assert num_layers == 2
        assert resnet_groups == 32
        # assert add_downsample == True or ...

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample: # True
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels)])
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states

# -----------------------------------------------------------------------------
class Attention(nn.Module):
    r"""
        base on AttnProcessor2_0
        in_channels,
        heads=in_channels // attention_head_dim,
        dim_head=attention_head_dim,
        bias=True,
    """

    def __init__(self,
        query_dim: int,
        heads: int = 1,
        dim_head: int = 512,
        bias: bool = True,
        norm_num_groups = 32,
    ):
        super().__init__()
        assert query_dim == 512
        assert heads == 1
        assert dim_head == 512
        assert bias == True
        assert norm_num_groups == 32

        self.heads = heads
        self.scale = dim_head**-0.5

        self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=1e-6, affine=True)
        self.to_q = nn.Linear(query_dim, dim_head * heads, bias=bias)

        # only relevant for the `AddedKVProcessor` classes
        self.to_k = nn.Linear(query_dim, dim_head * heads, bias=bias)
        self.to_v = nn.Linear(query_dim, dim_head * heads, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(dim_head * heads, query_dim, bias=True))
        self.to_out.append(nn.Dropout(0.0))

    def forward(self, hidden_states):
        residual = hidden_states

        assert hidden_states.ndim == 4

        # input_ndim = hidden_states.ndim # 4
        # if input_ndim == 4:
        B, C, H, W = hidden_states.shape
        hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        head_dim = key.shape[-1] // self.heads
        query = query.view(B, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(B, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(B, -1, self.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(B, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        # if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)
        hidden_states = hidden_states + residual

        return hidden_states

# -----------------------------------------------------------------------------
class UNetMidBlock2D(nn.Module):
    def __init__(self,
        in_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        attention_head_dim: int = 512,
    ):
        super().__init__()
        assert in_channels == 512
        assert num_layers == 1
        assert resnet_groups == 32
        assert attention_head_dim == 512

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            attentions.append(
                Attention(
                    in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                )
            )

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        # pdb.set_trace()

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states)

        return hidden_states

# -----------------------------------------------------------------------------
class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()
        # assert num_layers == 1 or ...
        assert resnet_groups == 32

        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(
                out_channels, 
                # use_conv=True, out_channels=out_channels,
            )])
        else:
            self.upsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states



# !!! -----------------------------------------------------------------------------
def nonlinearity(x):
    return x * torch.sigmoid(x)  # nonlinearity, F.silu

# !!! -----------------------------------------------------------------------------
def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# !!! -----------------------------------------------------------------------------
class Downsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x

# !!! -----------------------------------------------------------------------------
class Upsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


# !!! -----------------------------------------------------------------------------
class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        assert out_channels is not None

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:  # To support torch.jit.script
            self.conv_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)  # nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)  # nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h


class Encoder(nn.Module):
    def __init__(self,
        in_channels = 3,
        out_channels = 4,
        down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        block_out_channels = [128, 256, 512, 512],
        layers_per_block = 2,
        norm_num_groups = 32,
    ):
        super().__init__()
        assert in_channels == 3
        assert out_channels == 4
        assert layers_per_block == 2
        assert norm_num_groups == 32

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = DownEncoderBlock2D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_groups=norm_num_groups,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels # 8
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        # pdb.set_trace()

    def forward(self, sample):
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    def __init__(self,
        in_channels = 4,
        out_channels = 3,
        up_block_types = ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels = [128, 256, 512, 512],
        layers_per_block = 2,
        norm_num_groups = 32,
    ):
        super().__init__()
        assert in_channels == 4
        assert out_channels == 3
        assert layers_per_block == 2
        assert norm_num_groups == 32

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                num_layers=layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_groups=norm_num_groups,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        # pdb.set_trace()

    def forward(self, sample):
        sample = self.conv_in(sample)
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

def torch_nn_arange(x):
    if x.dim() == 2:
        B, C = x.size()
        a = torch.arange(x.nelement())/x.nelement()
        a = a.to(x.device)
        return a.view(B, C)

    if x.dim() == 3:
        B, C, HW = x.size()
        a = torch.arange(x.nelement())/x.nelement()
        a = a.to(x.device)
        return a.view(B, C, HW)

    B, C, H, W = x.size()
    a = torch.arange(x.nelement())/x.nelement()
    a = a.to(x.device)
    return a.view(B, C, H, W)


if __name__ == "__main__":
    model = AutoencoderKL()
    model.eval()

    x = torch.randn(6, 4, 64, 64)
    x = torch_nn_arange(x)
    with torch.no_grad():
        y = model.decode(x)
    todos.debug.output_var("y1", y)
    # expect tensor [y1] size: [6, 3, 512, 512], min: -0.549043, max: 0.309814, mean: -0.072002

    x = torch.randn(1, 3, 1024, 1024)
    x = torch_nn_arange(x)
    with torch.no_grad():
        y = model.encode(x)
    todos.debug.output_var("y2", y)
    # expect tensor [y2] size: [1, 4, 128, 128], min: -10.59375, max: 7.34375, mean: 0.331158
