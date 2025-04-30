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

def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads = None,
    resnet_groups = None,
    cross_attention_dim = None,
    downsample_padding = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm = None,
    attention_head_dim = None,
    downsample_type = None,
    dropout: float = 0.0,
):
    # If attn head dim is not defined, we default it to the number of heads
    # down_block_type ---- 'DownEncoderBlock2D'

    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownEncoderBlock2D": # True
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads = None,
    resnet_groups = None,
    cross_attention_dim = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm = None,
    attention_head_dim = None,
    upsample_type = None,
    dropout: float = 0.0,
) -> nn.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
        temb = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # attn = Attention(
        #   (group_norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        #   (to_q): Linear(in_features=512, out_features=512, bias=True)
        #   (to_k): Linear(in_features=512, out_features=512, bias=True)
        #   (to_v): Linear(in_features=512, out_features=512, bias=True)
        #   (to_out): ModuleList(
        #     (0): Linear(in_features=512, out_features=512, bias=True)
        #     (1): Dropout(p=0.0, inplace=False)
        #   )
        # )
        # encoder_hidden_states = None
        # attention_mask = None
        # temb = None
        # args = ()
        # kwargs = {}

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None: # False
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim # 4

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None: # False
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None: # True
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None: # True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None: # False
            query = attn.norm_q(query)
        if attn.norm_k is not None: # False
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection: # True
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor # attn.rescale_output_factor === 1

        return hidden_states

##################################################################################
# DownEncoderBlock2D, UNetMidBlock2D, UpDecoderBlock2D
##################################################################################

class AutoencoderKL(nn.Module):
    def __init__(self,
        in_channels = 3,
        out_channels = 3,
        down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types = ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels = [128, 256, 512, 512],
        layers_per_block = 2,
        act_fn = "silu",
        latent_channels = 4,
        norm_num_groups = 32,
    ):
        super().__init__()
        self.scaling_factor = 0.18215

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        # self.load_weights()

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
        return output


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

# -----------------------------------------------------------------------------
class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        # in_channels = 128
        # out_channels = 128
        # dropout = 0.0
        # num_layers = 2
        # resnet_eps = 1e-06
        # resnet_time_scale_shift = 'default'
        # resnet_act_fn = 'silu'
        # resnet_groups = 32
        # resnet_pre_norm = True
        # output_scale_factor = 1.0
        # add_downsample = True
        # downsample_padding = 0

        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    # temb_channels=None,
                    # eps=resnet_eps,
                    # groups=resnet_groups,
                    # dropout=dropout,
                    # time_embedding_norm=resnet_time_scale_shift,
                    # non_linearity=resnet_act_fn,
                    # output_scale_factor=output_scale_factor,
                    # pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample: # True
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        # use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states

# -----------------------------------------------------------------------------
class Attention(nn.Module):
    r"""
        in_channels,
        heads=in_channels // attention_head_dim,
        dim_head=attention_head_dim,
        eps=resnet_eps,
        norm_num_groups=attn_groups,
        bias=True,
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim = None,
        heads: int = 8,
        kv_heads = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm = None,
        norm_num_groups = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor = None,
        out_dim: int = None,
    ):
        super().__init__()
        # query_dim = 512
        # cross_attention_dim = None
        # heads = 1
        # kv_heads = None
        # dim_head = 512
        # dropout = 0.0
        # bias = True
        # qk_norm = None
        # norm_num_groups = 32
        # out_bias = True
        # scale_qk = True
        # eps = 1e-06
        # processor = <diffusers.models.attention_processor.AttnProcessor2_0 object at 0x7fdfc59ce030>
        # out_dim = None

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = out_dim // dim_head if out_dim is not None else heads


        if norm_num_groups is not None: # True
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        # only relevant for the `AddedKVProcessor` classes
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))


        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None: # False
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)
        #pdb.set_trace()




    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states = None,
        attention_mask = None,
        **cross_attention_kwargs,
    ):
        r"""
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )



# -----------------------------------------------------------------------------
class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        # in_channels = 512
        # temb_channels = None
        # dropout = 0.0
        # num_layers = 1
        # resnet_eps = 1e-06
        # resnet_time_scale_shift = 'default'
        # resnet_act_fn = 'silu'
        # resnet_groups = 32
        # attn_groups = None
        # resnet_pre_norm = True
        # add_attention = True
        # attention_head_dim = 512
        # output_scale_factor = 1

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None: # True
            attn_groups = resnet_groups

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                # temb_channels=temb_channels,
                # eps=resnet_eps,
                # groups=resnet_groups,
                # dropout=dropout,
                # time_embedding_norm=resnet_time_scale_shift,
                # non_linearity=resnet_act_fn,
                # output_scale_factor=output_scale_factor,
                # pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            if self.add_attention: # True ????
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        bias=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    # temb_channels=temb_channels,
                    # eps=resnet_eps,
                    # groups=resnet_groups,
                    # dropout=dropout,
                    # time_embedding_norm=resnet_time_scale_shift,
                    # non_linearity=resnet_act_fn,
                    # output_scale_factor=output_scale_factor,
                    # pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False
        #pdb.set_trace()

    def forward(self, hidden_states: torch.Tensor, temb = None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states

# -----------------------------------------------------------------------------
class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels = None,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    # temb_channels=temb_channels,
                    # eps=resnet_eps,
                    # groups=resnet_groups,
                    # dropout=dropout,
                    # time_embedding_norm=resnet_time_scale_shift,
                    # non_linearity=resnet_act_fn,
                    # output_scale_factor=output_scale_factor,
                    # pre_norm=resnet_pre_norm,
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

        self.resolution_idx = resolution_idx

    def forward(self, hidden_states: torch.Tensor, temb = None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states



# ==========================================================================================================================
def nonlinearity(x):
    return x * torch.sigmoid(x)  # nonlinearity, F.silu


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# -----------------------------------------------------------------------------
class Downsample2D(nn.Module):
    '''
        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class Upsample2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class ResnetBlock2D(nn.Module):
    '''
        in_channels=in_channels,
        out_channels=out_channels,

        temb_channels=None,
        eps=resnet_eps,
        groups=resnet_groups,
        dropout=dropout,
        time_embedding_norm=resnet_time_scale_shift,
        non_linearity=resnet_act_fn,
        output_scale_factor=output_scale_factor,
        pre_norm=resnet_pre_norm,    
    '''
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
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
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types = ("DownEncoderBlock2D",),
        block_out_channels = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
    ):
        super().__init__()
        # in_channels = 3
        # out_channels = 4
        # down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D']
        # block_out_channels = [128, 256, 512, 512]
        # layers_per_block = 2
        # norm_num_groups = 32
        # act_fn = 'silu'
        # double_z = True

        self.layers_per_block = layers_per_block

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
            # DownEncoderBlock2D
            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=True,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels # 8
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        # pdb.set_trace()

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

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
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types = ("UpDecoderBlock2D",),
        block_out_channels = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
    ):
        super().__init__()
        # in_channels = 4
        # out_channels = 3
        # up_block_types = ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D']
        # block_out_channels = [128, 256, 512, 512]
        # layers_per_block = 2
        # norm_num_groups = 32
        # act_fn = 'silu'
        # norm_type = 'group'

        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None
        assert temb_channels == None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            # UpDecoderBlock2D
            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        # pdb.set_trace()

    def forward(self, sample, latent_embeds = None):
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample




if __name__ == "__main__":
    model = AutoencoderKL()

    print(model)

    # model.cuda()
    # input = torch.randn(1, 3, 1024, 1024).cuda()
    # with torch.no_grad():
    #     output = model(input)

    # todos.debug.output_var("output", output)

    # class_name = model.__class__.__name__
    # model = torch.jit.script(model)
    # print(f"torch.jit.script({class_name}) OK !")
