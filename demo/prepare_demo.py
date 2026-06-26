#!/usr/bin/env python3
"""
Generate degraded versions of your own images for FlowSteer.

Pre-generated degraded images for the two included samples are already in
demo/degraded/ — you do not need to run this script to use the quick demo.

Use this script if you want to add your own images:
  1. Place your clean .jpg or .png images in demo/inputs/
  2. Run from the repo root:
         python demo/prepare_demo.py
  3. Degraded versions will appear in:
         demo/degraded/superres_4x/   (4x downsampled then pixel-upsampled)
         demo/degraded/colorized/     (grayscale measurement)
         demo/degraded/denoised/      (additive Gaussian noise, sigma=50)
         demo/degraded/deblurred/     (Gaussian blur)
  4. Update the --source_img_dir path and prompts in src/batch_run.sh, or
     call src/edit_image.py directly with your new degraded image.
"""
import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from flux.ddnm_degrads import (
    set_operator,
    PatchUpsampleEmbeds,
    get_super_resolution_scales,
    sigma as NOISE_SIGMA,
)

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(DEMO_DIR, "inputs")
DEGRADED_DIR = os.path.join(DEMO_DIR, "degraded")

TASKS = {
    "super resolution": "superres_4x",
    "colorization":     "colorized",
    "denoising":        "denoised",
    "deblurring":       "deblurred",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

for subdir in TASKS.values():
    os.makedirs(os.path.join(DEGRADED_DIR, subdir), exist_ok=True)

image_files = [f for f in os.listdir(INPUTS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not image_files:
    print(f"No images found in {INPUTS_DIR}.")
    print("Add .jpg or .png images to demo/inputs/ and re-run this script.")
    raise SystemExit(1)

transform = T.ToTensor()

for fname in image_files:
    print(f"\nProcessing: {fname}")
    img = Image.open(os.path.join(INPUTS_DIR, fname)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    for task, subdir in TASKS.items():
        A = set_operator(x.shape, task)
        degraded = A(x)

        if task == "denoising":
            noise_std = NOISE_SIGMA / 255.0
            noise = torch.randn_like(x) * noise_std
            degraded = torch.clamp(x + noise, 0.0, 1.0)

        if task == "super resolution":
            sh, sw = get_super_resolution_scales(x.shape[2], x.shape[3])
            degraded = PatchUpsampleEmbeds(degraded, sh, sw)

        out_path = os.path.join(DEGRADED_DIR, subdir, fname)
        TF.to_pil_image(torch.clamp(degraded.squeeze(0), 0.0, 1.0).cpu()).save(out_path, quality=95)
        print(f"  [{task:20s}] -> {os.path.relpath(out_path)}")

print(f"\nDone. {len(image_files)} image(s) processed across {len(TASKS)} tasks.")
