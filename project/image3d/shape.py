import os
import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision import transforms

from .dinov2 import Dinov2Model
from .dit import Hunyuan3DDiT
from .attn import FourierEmbedder, Transformer, CrossAttentionDecoder

from tqdm import tqdm
from einops import repeat

import todos
import pdb

def dense_grid(res, box_m=1.01):
    assert res == 384
    x = torch.linspace(-box_m, box_m, res + 1)
    y = torch.linspace(-box_m, box_m, res + 1)
    z = torch.linspace(-box_m, box_m, res + 1)
    xs, ys, zs = torch.meshgrid(x, y, z, indexing = "ij")
    xyz_grid = torch.stack((xs, ys, zs), dim = -1)

    return xyz_grid # size() -- [385, 385, 385, 3]

class ShapeVAE(nn.Module):
    """
    target: hy3dgen.shapegen.models.ShapeVAE
    params:
      num_latents: 512
      embed_dim: 64
      num_freqs: 8
      include_pi: false
      heads: 16
      width: 1024
      num_decoder_layers: 16
      qkv_bias: false
      qk_norm: true
      scale_factor: 1.0188137142395404
      geo_decoder_mlp_expand_ratio: 4
      geo_decoder_downsample_ratio: 1
      geo_decoder_ln_post: true
    """
    def __init__(self,
        num_latents = 512,
        embed_dim = 64,
        width = 1024,
        heads = 16,
        num_decoder_layers = 16,
        scale_factor = 1.0188137142395404,
    ):
        super().__init__()
        self.post_kl = nn.Linear(embed_dim, width)
        # self.post_kl -- Linear(in_features=64, out_features=1024, bias=True)

        self.transformer = Transformer(n_ctx=num_latents, width=width, layers=num_decoder_layers, heads=heads)
        self.geo_decoder = CrossAttentionDecoder(
            out_channels=1, num_latents=num_latents, mlp_expand_ratio=4, width=width, heads=heads
        )

        self.scale_factor = scale_factor  # 1.0188137142395404
        self.latent_shape = (num_latents, embed_dim)  # (512, 64)

        self.load_weights()

    def forward(self, latents):
        # 1. latents decode
        latents = latents/self.scale_factor 
        # tensor [latents] size: [1, 512, 64], min: -4.003906, max: 3.90625, mean: 0.018309
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        # tensor [latents] size: [1, 512, 1024], min: -374.5, max: 37.09375, mean: 0.019848

        # 2. latents to 3d volume
        grid_res = 384
        num_chunks = 8000
        batch_size = latents.shape[0] # 1
        xyz_samples = dense_grid(384)
        xyz_samples = xyz_samples.view(-1, 3).to(latents.device)
        # tensor [xyz_samples] size: [57066625, 3], min: -1.009766, max: 1.009766, mean: 0.0

        batch_logits = []
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc=f"Volume Decoding"):
            chunk_queries = xyz_samples[start: start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = self.geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)
        # len(batch_logits) -- 7134
        # batch_logits[0].size() -- [1, 8000, 1]
        grid_logits = torch.cat(batch_logits, dim=1) # torch.cat(batch_logits, dim=1).size() -- [1, 57066625, 1]
        # grid_size --[385, 385, 385]
        grid_logits = grid_logits.view((batch_size, grid_res + 1, grid_res + 1, grid_res + 1)).float()
        # 385*385*385 === 57066625
        # tensor [grid_logits] size: [1, 385, 385, 385], min: -1.082031, max: 1.067383, mean: -0.787309
        return grid_logits


    def load_weights(self, model_path="models/image3d_shapevae.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)



def image_transform(image):
    image_size = 518
    T = transforms.Compose(
        [
            transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(image_size), # 518
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return T(image)

class ShapeGenerator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.dinov2_model = Dinov2Model()
        self.dit_model = Hunyuan3DDiT()
        self.vae_model = ShapeVAE()

    def forward(self, image):
        image = F.interpolate(image, size=(512, 512), mode="bilinear", align_corners=True)

        # tensor [image] size: [1, 3, 512, 512], min: 0.0, max: 1.0, mean: 0.8434
        image = image_transform(image)
        # tensor [inputs] size: [1, 3, 518, 518], min: -2.119141, max: 2.638672, mean: 1.745283

        self.dinov2_model.to(self.device)
        dinov2_output = self.dinov2_model(image)
        # todos.debug.output_var("dinov2_output", dinov2_output)
        self.dinov2_model.cpu()

        dit_condition = torch.cat((dinov2_output, torch.zeros_like(dinov2_output)), dim = 0)
        # tensor [dit_condition] size: [2, 1370, 1536], min: -16.389088, max: 15.987875, mean: -0.009674
        
        latents = torch.randn(1, 512, 64).to(image.device)

        num_inference_steps = 50
        step_scale = 1.0/(num_inference_steps - 1.0)
        sigmas = torch.linspace(0, 1, num_inference_steps).to(image.device) # [0.0, 1.0]

        pbar = tqdm(total=num_inference_steps, desc="Diffusion Sampling:")
        guidance_scale = 5.0

        self.dit_model.to(self.device)
        for i in range(num_inference_steps):
            pbar.update(1)
            latent_model_input = torch.cat([latents] * 2, dim=0) # size() -- [2, 512, 64]


            timestep = sigmas[i].expand(2)
            noise_pred = self.dit_model(latent_model_input, timestep, dit_condition) # xxxx_9999
            # tensor [noise_pred] size: [2, 512, 64], min: -3.826172, max: 3.9375, mean: 0.000697

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            if (i < num_inference_steps - 1):
                latents = latents + step_scale * noise_pred

        self.dit_model.cpu()

        self.vae_model.to(self.device)
        grid_logits = self.vae_model(latents)
        self.vae_model.cpu()

        return grid_logits # [1, 385, 385, 385]
