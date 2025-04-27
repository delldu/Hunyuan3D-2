import os
import torch
from torch import nn

import todos
import pdb

# ----------------------------------------
class Dinov2Embeddings(nn.Module):
    """
    Construct the CLS token, mask token, position and patch embeddings.
    """
    def __init__(self, hidden_size=1536, patch_size=14):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))  # [1, 1, 1536]
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_size))  # [1, 1536]
        self.patch_embeddings = Dinov2PatchEmbeddings()
        num_patches = self.patch_embeddings.num_patches  # 1369
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))  # [1, 1370, 1536]
        self.patch_size = patch_size  # 14
        # self = Dinov2Embeddings(
        #   (patch_embeddings): Dinov2PatchEmbeddings(
        #     (projection): Conv2d(3, 1536, kernel_size=(14, 14), stride=(14, 14))
        #   )
        #   (dropout): Dropout(p=0.0, inplace=False)
        # )

    def interpolate_pos_encoding(self, embeddings, height: int, width: int):
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1  # 1369

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        pdb.set_trace()
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions ** 0.5)

        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        todos.debug.output_var("patch_pos_embed 1", patch_pos_embed)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        todos.debug.output_var("patch_pos_embed 2", patch_pos_embed)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32), size=(new_height, new_width), mode="bicubic", align_corners=False
        ).to(dtype=patch_pos_embed.dtype)

        todos.debug.output_var("patch_pos_embed 3", patch_pos_embed)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        todos.debug.output_var("patch_pos_embed 4", patch_pos_embed)

        pdb.set_trace()
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.size()  # size() -- [1, 3, 518, 518]
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, H, W)
        return embeddings

# ----------------------------------------
class Dinov2PatchEmbeddings(nn.Module):
    def __init__(self, hidden_size=1536, image_size=518, patch_size=14, num_channels=3):
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # self.patch_size = patch_size
        self.num_channels = num_channels # 3
        self.num_patches = num_patches # Keep !!!

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        #     (projection): Conv2d(3, 1536, kernel_size=(14, 14), stride=(14, 14))

    def forward(self, pixel_values):
        # tensor [pixel_values] size: [1, 3, 518, 518], min: -2.117904, max: 2.64, mean: -1.124644
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


def sdpa_attention_forward(query, key, value, scaling=0.125):
    # scaling = 0.125
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, scale=scaling, is_causal=False
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


class Dinov2SelfAttention(nn.Module):
    def __init__(self, hidden_size=1536, num_attention_heads=24):
        super().__init__()
        self.num_attention_heads = num_attention_heads # 24
        self.attention_head_size = int(hidden_size / num_attention_heads) # 64
        self.all_head_size = num_attention_heads * self.attention_head_size # 1536
        self.scaling = self.attention_head_size ** -0.5 # 0.125

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=True)

    def transpose_for_scores(self, x):
        # tensor [x] size: [1, 1370, 1536], min: -8.190102, max: 8.054783, mean: 0.01069
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # tensor [x] size: [1, 1370, 24, 64], min: -8.190102, max: 8.054783, mean: 0.01069
        return x.permute(0, 2, 1, 3) # [1, 24, 1370, 64]

    def forward(self, hidden_states):
        # tensor [hidden_states] size: [1, 1370, 1536], min: -29.074425, max: 17.105459, mean: -0.011517
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # tensor [key_layer] size: [1, 24, 1370, 64], min: -8.190102, max: 8.054783, mean: 0.01069

        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        context_layer = sdpa_attention_forward(query_layer, key_layer, value_layer, scaling=self.scaling)
        # context_layer.size() -- [1, 1370, 24, 64]

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # new_context_layer_shape -- torch.Size([1, 1370, 1536])

        context_layer = context_layer.reshape(new_context_layer_shape)
        return context_layer

# ----------------------------------------
class Dinov2SelfOutput(nn.Module):
    def __init__(self, hidden_size=1536):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    # def forward(self, hidden_states, input_tensor):
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states

# ----------------------------------------
class Dinov2Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Dinov2SelfAttention()
        self.output = Dinov2SelfOutput()

    def forward(self, hidden_states):
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs) #, hidden_states)
        return attention_output


# ----------------------------------------
class Dinov2LayerScale(nn.Module):
    def __init__(self, hidden_size=1536, layerscale_value=1.0):
        super().__init__()
        self.lambda1 = nn.Parameter(layerscale_value * torch.ones(hidden_size))

    def forward(self, hidden_state):
        return hidden_state * self.lambda1


# -------------------------------------
class Dinov2SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size=1536, mlp_ratio=4):
        super().__init__()
        assert hidden_size == 1536
        assert mlp_ratio == 4
        
        in_features = out_features = hidden_size
        hidden_features = int(hidden_size * mlp_ratio) # 6144
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8 # 4096
        assert hidden_features == 4096
        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state):
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


# -----------------------------------
class Dinov2Layer(nn.Module):
    def __init__(self, hidden_size=1536):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = Dinov2Attention()
        self.layer_scale1 = Dinov2LayerScale()
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.mlp = Dinov2SwiGLUFFN()
        self.layer_scale2 = Dinov2LayerScale()

    def forward(self, hidden_states):
        attention_output = self.attention(self.norm1(hidden_states))
        attention_output = self.layer_scale1(attention_output)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = layer_output + hidden_states
        return layer_output


# ----------------------------------------------------------
class Dinov2Encoder(nn.Module):
    def __init__(self, num_hidden_layers=40):
        super().__init__()
        self.layer = nn.ModuleList([Dinov2Layer() for _ in range(num_hidden_layers)])

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
        return hidden_states  # last_hidden_state

# -----------------------------------------------------------------------------------
class Dinov2Model(nn.Module):
    ##########################################################################################
    # main_image_encoder:
    #   #type: DinoImageEncoder # dino giant
    #   kwargs:
    #     config:
    #       attention_probs_dropout_prob: 0.0
    #       drop_path_rate: 0.0
    #       hidden_act: gelu
    #       hidden_dropout_prob: 0.0
    #       hidden_size: 1536
    #       image_size: 518
    #       initializer_range: 0.02
    #       layer_norm_eps: 1.e-6
    #       layerscale_value: 1.0
    #       mlp_ratio: 4
    #       model_type: dinov2
    #       num_attention_heads: 24
    #       num_channels: 3
    #       num_hidden_layers: 40
    #       patch_size: 14
    #       qkv_bias: true
    #       torch_dtype: float32
    #       use_swiglu_ffn: true
    #     image_size: 518
    ##########################################################################################
    def __init__(self, hidden_size=1536, num_hidden_layers=40):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.embeddings = Dinov2Embeddings()
        self.encoder = Dinov2Encoder()
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.load_weights()

    def forward(self, pixel_values):
        # tensor [pixel_values] size: [1, 3, 518, 518], min: -2.099609, max: 2.638672, mean: 1.449731
        embedding_output = self.embeddings(pixel_values)
        encoder_output = self.encoder(embedding_output)
        last_hidden_state = self.layernorm(encoder_output)

        return last_hidden_state

    def load_weights(self, model_path="models/image3d_dinov2.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)
