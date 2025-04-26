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

import todos
from . import shape

import pdb

def image_center(image, border_ratio = 0.15):
    B, C, H, W = image.size()
    assert C == 4
    mask = image[:, 3:4, :, :]
    coords = torch.nonzero(mask.view(H, W), as_tuple=True)
    x1, y1, x2, y2 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
    crop_image = image[:, :, y1:y2, x1:x2]

    # add border ...
    pad_size = int(max(y2 - y1, x2 - x1) * border_ratio)//2
    p4d = (pad_size, pad_size, pad_size, pad_size)
    pad_image = F.pad(crop_image, p4d)

    # Scale to 512...
    B2, C2, H2, W2 = pad_image.size()
    scale = 512/max(H2, W2)
    H2 = int(scale * H2)
    W2 = int(scale * W2)
    scale_image = F.interpolate(pad_image, size=(H2, W2))

    # Center padding ...
    pad_left = (512 - W2)//2
    pad_right = 512 - W2 - pad_left
    pad_top = (512 - H2)//2
    pad_bottom = 512 - H2 - pad_top
    p4d = (pad_left, pad_right, pad_top, pad_bottom) # left,right, top, bottom
    pad_image = F.pad(scale_image, p4d)

    return pad_image


def get_shape_model():
    """Create model."""

    # model = vae.ShapeVAE()
    # model = dit.Hunyuan3DDiT()
    device = todos.model.get_device()    
    model = shape.Generator(device)
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
    # load model
    model, device = get_shape_model()
    # print(model)

    # load files
    input_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(input_filenames))
    for filename in input_filenames:
        progress_bar.update(1)

        input_tensor = todos.data.load_rgba_tensor(filename)
        input_tensor = image_center(input_tensor)[:, 0:3, :, :].to(device)

        # model = model.half()
        with torch.no_grad():
            output_mesh = model(input_tensor)

        # output_file = f"{output_dir}/{os.path.basename(filename)}"

        obj_filename = os.path.basename(filename)
        obj_filename = obj_filename.replace(".jpg", ".obj")
        obj_filename = obj_filename.replace(".png", ".obj")
        output_file = f"{output_dir}/{obj_filename}"

        output_mesh.export(output_file)
