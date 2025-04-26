import os
import torch
import torch.nn as nn
from .attn import FourierEmbedder, Transformer, CrossAttentionDecoder
import numpy as np
from skimage import measure
import trimesh

# from .surface_extractors import MCSurfaceExtractor
# from .volume_decoders import VanillaVolumeDecoder
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


def create_mesh(grid_logit):
    '''Create mesh from grid logit'''

    mesh_v, mesh_f, normals, _ = measure.marching_cubes(
        grid_logit.cpu().numpy(),
        0.0, # mc_level
        method="lewiner"
    )
    # array [mesh_v] shape: (327988, 3), min: 1.8486219644546509, max: 382.1461181640625, mean: 184.00257873535156
    # array [mesh_f] shape: (655980, 3), min: 0, max: 327987, mean: 163994.911803
    # array [normals] shape: (327988, 3), min: -1.0, max: 1.0, mean: 0.005313000176101923
    grid_size = [385, 385, 385]
    bbox_min = np.array([-1.01, -1.01, -1.01])
    bbox_size = np.array([2.02,  2.02,  2.02])
    mesh_v = mesh_v / grid_size * bbox_size + bbox_min

    mesh_f = mesh_f[:, ::-1] # !!!! [0, 1, 2] ==> [2, 1, 0] !!!
    mesh = trimesh.Trimesh(mesh_v, mesh_f)
    # mesh ...
    return mesh # mesh.export("xxxx.glb")

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
    def __init__(
        self,
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

        self.transformer = Transformer(
            n_ctx=num_latents, width=width, layers=num_decoder_layers, heads=heads
        )
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
        xyz_samples, grid_size = dense_grid(384)
        xyz_samples = xyz_samples.view(-1, 3).to(latents.device)
        # tensor [xyz_samples] size: [57066625, 3], min: -1.009766, max: 1.009766, mean: 0.0

        batch_logits = []
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc=f"Volume Decoding"):
            chunk_queries = xyz_samples[start: start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = self.geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)  # ggml_cat xxxx_????
        # len(batch_logits) -- 7134
        # batch_logits[0].size() -- [1, 8000, 1]
        grid_logits = torch.cat(batch_logits, dim=1) # torch.cat(batch_logits, dim=1).size() -- [1, 57066625, 1]
        # grid_size --[385, 385, 385]
        grid_logits = grid_logits.view((batch_size, grid_res + 1, grid_res + 1, grid_res + 1)).float()
        # 385*385*385 === 57066625
        # tensor [grid_logits] size: [1, 385, 385, 385], min: -1.082031, max: 1.067383, mean: -0.787309
        return create_mesh(grid_logits[0])


    def load_weights(self, model_path="models/image3d_shapevae.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)
