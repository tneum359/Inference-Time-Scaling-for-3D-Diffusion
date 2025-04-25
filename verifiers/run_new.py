import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from torchvision.io import read_image
import glob
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_dir', type=str, help='Path to the directory created by scaling.py, containing the 6 generated views.')
parser.add_argument('--output_path', type=str, default='outputs_reconstruction/', help='Base directory for reconstruction outputs.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale parameter for standard cameras.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance for video.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load reconstruction model
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

# make output directories
object_name = os.path.basename(os.path.normpath(args.input_dir))
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

###############################################################################
# Stage 1: Load Pre-generated multiview images from scaling.py output
###############################################################################

print(f"Loading images from: {args.input_dir}")

# Find generated image files (excluding the input image)
image_files = sorted(glob.glob(os.path.join(args.input_dir, "prompt@*.png")))
if len(image_files) != 6:
    print(f"Warning: Expected 6 images matching 'prompt@*.png' in {args.input_dir}, but found {len(image_files)}. Reconstruction might fail.")
    # Attempt to find any 6 pngs excluding input_image.png as a fallback
    all_pngs = sorted(glob.glob(os.path.join(args.input_dir, "*.png"))) 
    image_files = [f for f in all_pngs if os.path.basename(f) != 'input_image.png']
    if len(image_files) != 6:
         raise FileNotFoundError(f"Could not find exactly 6 generated PNG images in {args.input_dir}")
    print(f"Fallback: Found {len(image_files)} PNGs (excluding input_image.png), proceeding with these.")

# Load and stack images
loaded_images = []
for img_file in image_files:
    img = read_image(img_file)
    img = img[:3, :, :]
    img = img / 255.0
    loaded_images.append(img)

images_tensor = torch.stack(loaded_images)
images_tensor = images_tensor.unsqueeze(0)

print(f"Loaded images tensor with shape: {images_tensor.shape}")

# Store in a structure similar to the old 'outputs' for compatibility with Stage 2 loop
outputs = [{'name': object_name, 'images': images_tensor}]

###############################################################################
# Stage 2: Reconstruction. (Using loaded images)
###############################################################################

# Use standard cameras expected by the reconstruction model
input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating mesh for {name} ...')

    images = sample['images'].to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )
        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")

        # get video
        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1, 
                M=120, 
                radius=args.distance, 
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)
            
            frames = render_frames(
                model, 
                planes, 
                render_cameras=render_cameras, 
                render_size=render_size, 
                chunk_size=chunk_size, 
                is_flexicubes=IS_FLEXICUBES,
            )

            save_video(
                frames,
                video_path_idx,
                fps=30,
            )
            print(f"Video saved to {video_path_idx}")