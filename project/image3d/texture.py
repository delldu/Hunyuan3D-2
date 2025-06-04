import torch
import torch.nn as nn
import numpy as np
import todos
import pdb


class TextureGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, mask):
    	return image



# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


class EulerAncestralDiscreteScheduler:
    """
    Ancestral sampling with Euler method steps.
    """
    def __init__(
        self,
        num_train_timesteps = 1000,
        beta_start = 0.00085,
        beta_end = 0.012,
    ):
        self.num_train_timesteps = num_train_timesteps
        # pdb.set_trace()

        # this schedule is very specific to the latent diffusion model.
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        # tensor [betas] size: [1000], min: 0.00085, max: 0.012, mean: 0.005349

        betas = rescale_zero_terminal_snr(betas)
        # betas -- tensor([    0.000850,     0.000917,     0.000922,  ...,     0.557568,
        #     0.751132,     1.000000])


        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0) # !!!!!!!!!!!!!!!!!!!!!!!!!!

        # Close to 0 without being 0 so first sigma is not inf
        # FP16 smallest positive subnormal works well here
        self.alphas_cumprod[-1] = 2**-24

        # # # self.alphas_cumprod.size() -- [1000]
        # sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        # sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        # self.sigmas = torch.from_numpy(sigmas) # self.sigmas.size() -- [1001], [4096.0, ..., 0.0]


    def scale_model_input(self, sample_1, timestep_i) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
        """
        # tensor [sample_1] size: [12, 4, 64, 64], min: -20960.0, max: 17488.0, mean: 3.652174
        # tensor [timestep] size: [], min: 999.0, max: 999.0, mean: 999.0
        sigma = self.sigmas[timestep_i]
        sample_1 = sample_1 / ((sigma**2 + 1) ** 0.5)

        return sample_1

    def set_timesteps(self, num_inference_steps):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        """

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps / self.num_inference_steps # ==> 33.333333333333336


        # timesteps = (np.arange(self.num_train_timesteps, 0, -step_ratio)).round().astype(np.float32)
        # timesteps -= 1
        timesteps = (np.arange(self.num_train_timesteps - 1, 0, -step_ratio)).round().astype(np.float32)
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        pdb.set_trace()

        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)


    def step(self, sample_1, timestep_i, sample_2):
        sigma = self.sigmas[timestep_i]

        # 1. compute predicted original sample_1 (x_0) from sigma-scaled predicted noise
        # * c_out + input * c_skip
        pred_original_sample = sample_2 * (-sigma / (sigma**2 + 1) ** 0.5) + (sample_1 / (sigma**2 + 1))

        sigma_from = self.sigmas[timestep_i]
        sigma_to = self.sigmas[timestep_i + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5 # === sigma_to**2/sigma_from ?

        # 2. Convert to an ODE derivative
        derivative = (sample_1 - pred_original_sample) / sigma
        dt = sigma_down - sigma
        print(f"{timestep_i}: {dt} ...")

        prev_sample = sample_1 + derivative * dt
        noise = torch.randn(sample_2.shape)
        # tensor [noise] size: [6, 4, 64, 64], min: -4.523438, max: 4.144531, mean: -0.000119

        prev_sample = prev_sample + noise * sigma_up
        # tensor [prev_sample] size: [6, 4, 64, 64], min: -288.5, max: 262.5, mean: 0.10095

        return (prev_sample, pred_original_sample)

    def __len__(self):
        return self.num_train_timesteps


if __name__ == "__main__":
    s = EulerAncestralDiscreteScheduler()
    s.set_timesteps(30)

    sample_1 = torch.randn(6, 4, 64, 64)
    timestep_i = 1 # 996.0
    sample_2 = torch.randn(6, 4, 64, 64)

    x1, x2 = s.step(sample_1, timestep_i, sample_2)

    # for timestep_i in range(30):
    #     x1, x2 = s.step(sample_1, timestep_i, sample_2)

    pdb.set_trace()