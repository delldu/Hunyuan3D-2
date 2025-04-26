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

# self.normal = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False)
# x = self.normal(x)

# def image_center(images, border_ratio = 0.15):
#     B, C, H, W = image.size()
#     assert C == 4
#     mask = image[:, 3:4, :, :]
#     coords = torch.nonzero(mask.view(H, W), as_tuple=True)
#     x1, y1, x2, y2 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
#     crop_image = image[:, :, y1:y2, x1:x2]

#     # add border ...
#     pad_size = int(max(y2 - y1, x2 - x1) * border_ratio)//2
#     p4d = (pad_size, pad_size, pad_size, pad_size)
#     pad_image = F.pad(crop_image, p4d)

#     # Scale to 512...
#     B2, C2, H2, W2 = pad_image.size()
#     scale = 512/max(H2, W2)
#     H2 = int(scale * H2)
#     W2 = int(scale * W2)
#     scale_image = F.interpolate(pad_image, size=(H2, W2))

#     # Center padding ...
#     pad_left = (512 - W2)//2
#     pad_right = 512 - W2 - pad_left
#     pad_top = (512 - H2)//2
#     pad_bottom = 512 - H2 - pad_top
#     p4d = (pad_left, pad_right, pad_top, pad_bottom) # left,right, top, bottom
#     pad_image = F.pad(scale_image, p4d)

#     return pad_image

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

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov2_model = Dinov2Model()
        self.dit_model = Hunyuan3DDiT()
        self.vae_model = ShapeVAE()


    def forward(self, image):
        image = image_transform(image)
        dinov2_output = self.dinov2_model(image)
        todos.debug.output_var("dinov2_output", dinov2_output)

        dit_condition = torch.cat((dinov2_output, torch.zeros_like(dinov2_output)), dim = 0)
        todos.debug.output_var("dit_condition", dit_condition)

        latents = torch.randn(1, 512, 64).to(image.device)
        todos.debug.output_var("latents", latents)

        num_inference_steps = 50
        scale = 1.0/(num_inference_steps - 1.0)
        sigmas = torch.linspace(0, 1, num_inference_steps).to(image.device) # [0.0, 1.0]

        pbar = tqdm(total=num_inference_steps, desc="Diffusion Sampling:")
        guidance_scale = 5.0

        for i in range(num_inference_steps):
            pbar.update(1)
            latent_model_input = torch.cat([latents] * 2, dim=0) # size() -- [2, 512, 64]


            # # NOTE: we assume model get timesteps ranged from 0 to 1
            # timestep = t.expand(latent_model_input.shape[0]).to(
            #     latents.dtype) / self.scheduler.num_train_timesteps
            # # timestep -- tensor([0.040802, 0.040802], device='cuda:0', dtype=torch.float16)
            timestep = sigmas[i].expand(2)
            print(timestep)


            # # tensor [latent_model_input] size: [2, 512, 64], min: -3.921875, max: 3.830078, mean: -0.000997
            # # tensor [timestep] size: [2], min: 0.0, max: 0.0, mean: 0.0
            # # cond is dict:
            # #     tensor [main] size: [2, 1370, 1536], min: -15.28125, max: 14.375, mean: -0.009306
            # # self.model -- Hunyuan3DDiT
            noise_pred = self.dit_model(latent_model_input, timestep, dit_condition) # xxxx_9999
            # tensor [noise_pred] size: [2, 512, 64], min: -3.826172, max: 3.9375, mean: 0.000697

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)

            # # guidance_scale == 5.0
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # # # compute the previous noisy sample x_t -> x_t-1
            # # outputs = self.scheduler.step(noise_pred, t, latents) # xxxx_9999
            # # # outputs is dict:
            # # # tensor [prev_sample] size: [1, 512, 64], min: -3.839844, max: 3.755859, mean: -0.000487
            # # latents = outputs.prev_sample

            if (i < num_inference_steps - 1):
                latents = latents + scale * noise_pred

        todos.debug.output_var("latents", latents)
        mesh = self.vae_model(latents)

        return mesh


if __name__ == "__main__":
    image = todos.data.load_rgba_tensor("../images/demo.png")
    image = image_center(image)
    # image = image * 2.0 - 1.0 # ==> convert image from [0.0, 1.0] ==> [-1.0, 1.0]
    image = image_transform(image[:, 0:3, :, :])

    todos.debug.output_var("image", image)

    dinov2_model = Dinov2Model()
    dinov2_model.eval()
    with torch.no_grad():
        dinov2_output = dinov2_model(image[:, 0:3, :, :])
    todos.debug.output_var("dinov2_output", dinov2_output)

    dit_model = Hunyuan3DDiT()
    dit_model.eval()

    dit_condition = torch.cat((dinov2_output, torch.zeros_like(dinov2_output)), dim = 0)
    todos.debug.output_var("dit_condition", dit_condition)

    latents = torch.randn(1, 512, 64)
    todos.debug.output_var("latents", latents)

    num_inference_steps = 50
    scale = 1.0/(num_inference_steps - 1.0)
    sigmas = torch.linspace(0, 1, num_inference_steps) # [0.0, 1.0]

    from tqdm import tqdm
    pbar = tqdm(total=num_inference_steps, desc="Diffusion Sampling:")
    guidance_scale = 5.0

    for i in range(num_inference_steps):
        pbar.update(1)
        latent_model_input = torch.cat([latents] * 2, dim=0) # size() -- [2, 512, 64]


        # # NOTE: we assume model get timesteps ranged from 0 to 1
        # timestep = t.expand(latent_model_input.shape[0]).to(
        #     latents.dtype) / self.scheduler.num_train_timesteps
        # # timestep -- tensor([0.040802, 0.040802], device='cuda:0', dtype=torch.float16)
        timestep = sigmas[i].expand(2)
        print(timestep)


        # # tensor [latent_model_input] size: [2, 512, 64], min: -3.921875, max: 3.830078, mean: -0.000997
        # # tensor [timestep] size: [2], min: 0.0, max: 0.0, mean: 0.0
        # # cond is dict:
        # #     tensor [main] size: [2, 1370, 1536], min: -15.28125, max: 14.375, mean: -0.009306
        # # self.model -- Hunyuan3DDiT
        with torch.no_grad():
            noise_pred = dit_model(latent_model_input, timestep, dit_condition) # xxxx_9999
        # tensor [noise_pred] size: [2, 512, 64], min: -3.826172, max: 3.9375, mean: 0.000697

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)

        # # guidance_scale == 5.0
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # # # compute the previous noisy sample x_t -> x_t-1
        # # outputs = self.scheduler.step(noise_pred, t, latents) # xxxx_9999
        # # # outputs is dict:
        # # # tensor [prev_sample] size: [1, 512, 64], min: -3.839844, max: 3.755859, mean: -0.000487
        # # latents = outputs.prev_sample

        if (i < num_inference_steps - 1):
            latents = latents + scale * noise_pred

    todos.debug.output_var("latents", latents)