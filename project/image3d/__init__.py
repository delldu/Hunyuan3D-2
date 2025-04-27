"""Image 3D Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2025(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 09 Apr 2025 10:36:34 AM CST
# ***
# ************************************************************************************/
#
__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

import numpy as np
from skimage import measure
import trimesh
from . import shape

import todos
import pdb

def image_center(image, border_ratio = 0.15):
    B, C, H, W = image.size()
    assert C == 4

    # 1) ============================================================================
    mask = image[:, 3:4, :, :]
    coords = torch.nonzero(mask.view(H, W), as_tuple=True)
    x1, y1, x2, y2 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
    h = y2 - y1
    w = x2 - x1
    if h == 0 or w == 0:
        raise ValueError('input image is empty')
    scale = max(H, W) * (1.0 - border_ratio) / max(h, w) # 0.9550561797752809
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (max(H, W) - w2) // 2
    x2_max = x2_min + w2
    y2_min = (max(H, W) - h2) // 2
    y2_max = y2_min + h2

    # 2) ============================================================================
    crop_image = image[:, :, y1:y2, x1:x2]
    new_image = torch.zeros(B, C, max(H, W), max(H, W))
    new_image[:, :, y2_min:y2_max, x2_min:x2_max] = \
        F.interpolate(crop_image, size=(h2, w2), mode="bilinear", align_corners=True)

    new_bg = torch.ones(B, 3, max(H, W), max(H, W))
    new_mask = new_image[:, 3:4, :, :]
    new_image = new_image[:, 0:3, :, :] * new_mask + new_bg * (1.0 - new_mask)

    return new_image, new_mask

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

import cv2
from PIL import Image
from einops import repeat, rearrange

def array_to_tensor(np_array):
    image_pt = torch.tensor(np_array).float()
    image_pt = image_pt / 255 * 2 - 1

    # tensor [image_pt] size: [512, 512, 3], min: -1.0, max: 1.0, mean: 0.68679
    image_pt = rearrange(image_pt, "h w c -> c h w")
    # tensor [image_pt] size: [3, 512, 512], min: -1.0, max: 1.0, mean: 0.68679

    image_pts = repeat(image_pt, "c h w -> b c h w", b=1)
    return image_pts

class ImageProcessorV2:
    def __init__(self, size=512, border_ratio=0.15):
        self.size = size
        self.border_ratio = border_ratio

    @staticmethod
    def recenter(image, border_ratio: float = 0.15):
        """ recenter an image to leave some empty space at the image border.
        """
        # array [image] shape: (500, 500, 4), min: 0, max: 255, mean: 62.681185
        if image.shape[-1] == 4: # True
            mask = image[..., 3]
        else:
            mask = np.ones_like(image[..., 0:1]) * 255
            image = np.concatenate([image, mask], axis=-1)
            mask = mask[..., 0]


        H, W, C = image.shape

        size = max(H, W)
        result = np.zeros((size, size, C), dtype=np.uint8)

        coords = np.nonzero(mask)
        # (Pdb) coords -- (array([ 24,  24,  24, ..., 469, 469, 469]), array([130, 131, 132, ..., 310, 311, 312]))

        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        if h == 0 or w == 0:
            raise ValueError('input image is empty')
        desired_size = int(size * (1 - border_ratio)) # desired_size
        scale = desired_size / max(h, w) # 0.9550561797752809
        h2 = int(h * scale)
        w2 = int(w * scale)

        x2_min = (size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (size - w2) // 2
        y2_max = y2_min + w2

        # x_min,x_max, y_min,y_max -- 24, 469, 108, 389

        #  x2_min,x2_max -- (37, 462), y2_min,y2_max -- (116, 384)
        result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2),
                                                          interpolation=cv2.INTER_AREA)

        # array [result] shape: (500, 500, 4), min: 0, max: 255, mean: 57.086336
        bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255

        mask = result[..., 3:].astype(np.float32) / 255
        result = result[..., :3] * mask + bg * (1 - mask)

        mask = mask * 255
        result = result.clip(0, 255).astype(np.uint8)
        mask = mask.clip(0, 255).astype(np.uint8)

        return result, mask

    def load_image(self, image, border_ratio=0.15, to_tensor=True):
        # image --- PIL.Image.Image
        # border_ratio = 0.15
        # to_tensor = True

        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            image, mask = self.recenter(image, border_ratio=border_ratio)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image): # True
            image = image.convert("RGBA")
            image = np.asarray(image)
            image, mask = self.recenter(image, border_ratio=border_ratio)

        # self.size == 512
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = mask[..., np.newaxis]

        if to_tensor: # True
            image = array_to_tensor(image)
            mask = array_to_tensor(mask)

        # tensor [image] size: [1, 3, 512, 512], min: -1.0, max: 1.0, mean: 0.68679
        # tensor [mask] size: [1, 1, 512, 512], min: -1.0, max: 1.0, mean: -0.32582
        return image, mask

    def __call__(self, image, border_ratio=0.15, to_tensor=True, **kwargs):
        # border_ratio = 0.15
        # to_tensor = True
        # kwargs = {}
        if self.border_ratio is not None: # self.border_ratio == 0.15
            border_ratio = self.border_ratio
        image, mask = self.load_image(image, border_ratio=border_ratio, to_tensor=to_tensor)
        outputs = {
            'image': image,
            'mask': mask
        }
        # outputs is dict:
        #     tensor [image] size: [1, 3, 512, 512], min: -1.0, max: 1.0, mean: 0.68679
        #     tensor [mask] size: [1, 1, 512, 512], min: -1.0, max: 1.0, mean: -0.32582
        return outputs


def get_shape_model():
    """Create model."""

    # model = vae.ShapeVAE()
    # model = dit.Hunyuan3DDiT()
    device = todos.model.get_device()    
    model = shape.ShapeGenerator(device)
    # model = model.to(device)
    model.eval()

    if "cpu" in str(device.type):
        model.float()

    print(f"Running on {device} ...")
    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image3d.torch"):
    #     model.save("output/image3d.torch")
    # torch.save(model.state_dict(), "/tmp/image3d.pth")

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    model, device = get_shape_model()
    # load files
    input_filenames = todos.data.load_files(input_files)

    image_process = ImageProcessorV2()

    # start predict
    progress_bar = tqdm(total=len(input_filenames))
    for filename in input_filenames:
        progress_bar.update(1)

        input_tensor = todos.data.load_rgba_tensor(filename)
        input_image, input_mask = image_center(input_tensor)
        input_image = input_image.to(device)

        # image = Image.open(filename).convert("RGBA")
        # input_tensor = image_process(image).pop("image").to(device)
        # input_tensor = (input_tensor + 1.0)/2.0

        # model = model.half()
        with torch.no_grad():
            grid_logits = model(input_image)

        # output_file = f"{output_dir}/{os.path.basename(filename)}"
        obj_filename = os.path.basename(filename)
        obj_filename = obj_filename.replace(".jpg", ".obj")
        obj_filename = obj_filename.replace(".png", ".obj")
        output_file = f"{output_dir}/{obj_filename}"
        output_mesh = create_mesh(grid_logits[0])
        output_mesh.export(output_file)
