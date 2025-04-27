import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .dinov2 import Dinov2Model
from .dit import Hunyuan3DDiT
from .vae import ShapeVAE
from tqdm import tqdm

import todos
import pdb


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
        self.dinov2_model.cpu()

        # todos.debug.output_var("dinov2_output", dinov2_output)

        dit_condition = torch.cat((dinov2_output, torch.zeros_like(dinov2_output)), dim = 0)
        # tensor [dit_condition] size: [2, 1370, 1536], min: -16.389088, max: 15.987875, mean: -0.009674
        
        latents = torch.randn(1, 512, 64).to(image.device)
        # todos.debug.output_var("latents", latents)

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

        # todos.debug.output_var("latents", latents)

        self.vae_model.to(self.device)
        grid_logits = self.vae_model(latents)
        self.vae_model.cpu()

        return grid_logits
