import os
from PIL import Image
import math
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ------------- Degradation mode default -------------------
IR_mode = "super resolution"  # Options: "colorization", "super resolution", "denoising", "deblurring"

# Used for super resolution. Change these here and edit_image.py will stay in sync.
scale_h = 4
scale_w = 4


def get_super_resolution_scales(height=None, width=None):
    sr_scale_h, sr_scale_w = scale_h, scale_w
    if height is not None and width is not None and height >= width:
        sr_scale_h, sr_scale_w = sr_scale_w, sr_scale_h
    return sr_scale_h, sr_scale_w


# Degradation-specific lambda schedule defaults used by sampling.py.
# start/step/end can be fractions of the denoising loop, where 0 is the first
# step and 1 is the last step.
DEFAULT_LAMBDA_SCHEDULE = {
    "kind": "step",
    "start": 0.50,
    "step": 0.70,
    "end": 0.85,
    "level_hi": 1.0,
    "level_lo": 0.5,
    "final_pad": 3,
}

LAMBDA_SCHEDULES = {
    "super resolution": dict(DEFAULT_LAMBDA_SCHEDULE),
    "colorization": dict(DEFAULT_LAMBDA_SCHEDULE),
    "inpainting": dict(DEFAULT_LAMBDA_SCHEDULE),
    "denoising": dict(DEFAULT_LAMBDA_SCHEDULE),
    "deblurring": dict(DEFAULT_LAMBDA_SCHEDULE),
    "old photo restoration": dict(DEFAULT_LAMBDA_SCHEDULE),
}


def get_lambda_schedule_config(IR_mode, overrides=None):
    config = dict(LAMBDA_SCHEDULES.get(IR_mode, DEFAULT_LAMBDA_SCHEDULE))
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    return config

# Used when IR_mode == "deblurring"
blur_sigma   = 2      # std dev of Gaussian PSF (in pixels)
kernel_size  = 11       # odd number, e.g., 11/15/21
wiener_lambda = 1e-1    # Tikhonov/Wiener regularizer for pinv(A); try 1e-4 .. 1e-2

# Used when IR_mode == "denoising"
sigma = 50             # noise level in [0,255] intensity units;

device = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------


# ---------- Degradation Operators -------------
#https://github.com/wyhuai/DDNM

def color2gray(x):
# Convert RGB image to grayscale by averaging the channels,but repeat it to maintain 3 channel structure.
    coef = 1/3
    gray = x[:, 0, :, :] * coef + x[:, 1, :, :] * coef + x[:, 2, :, :] * coef
    return gray.unsqueeze(1).repeat(1, 3, 1, 1)

def PatchUpsample(x, scale):
    n, c, h, w = x.shape
    x = x.view(n, c, h, 1, w, 1).expand(-1, -1, -1, scale, -1, scale)
    return x.contiguous().view(n, c, h * scale, w * scale)

def PatchUpsampleEmbeds(x, scale_h, scale_w):
    n, c, h, w = x.shape
    # Embedding dimensions change into [B, Cenc, Henc, Wenc]. So we need new scale_h and scale_w for embeddings.
    x = x.view(n, c, h, 1, w, 1).expand(-1, -1, -1, scale_h, -1, scale_w)
    return x.contiguous().view(n, c, h * scale_h, w * scale_w)

def gray2color(x):
    # Input shape: [N, 3, H, W] where all 3 channels are the same
    # First reduce to single channel
    x = x[:, 0, :, :]  # shape: [N, H, W]
    
    coef = 1/3
    # Reverse the averaging operation by scaling up
    base = coef**2 + coef**2 + coef**2  # sum of squared coefficients
    restored = x * coef / base  # scale back to plausible RGB value
    
    # Stack it back to 3 channels
    return torch.stack([restored, restored, restored], dim=1)  # shape: [N, 3, H, W]

def gaussian_2d_kernel(sigma: float, kernel_size: int, device="cpu", dtype=torch.float32):
    """
    Returns a (kernel_size x kernel_size) normalized 2D Gaussian kernel.
    kernel_size must be odd.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd."
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()
    return k  # shape [K, K]

def _fft_filter_from_kernel(k2d: torch.Tensor, h: int, w: int, c: int, device, dtype):
    """
    Center the small kernel into an (h,w) canvas (per-channel), roll to align,
    and return its FFT. Uses a safe real dtype for FFT (float32 if bf16/half).
    """
    # Use a dtype supported by torch.fft
    if dtype in (torch.float16, torch.bfloat16):
        safe_dtype = torch.float32
    else:
        safe_dtype = dtype

    ks = k2d.shape[-1]
    filt = torch.zeros((1, c, h, w), device=device, dtype=safe_dtype)
    filt[..., :ks, :ks] = k2d.to(device=device, dtype=safe_dtype)
    filt = torch.roll(filt, shifts=(-(ks - 1)//2, -(ks - 1)//2), dims=(2, 3))
    return torch.fft.fft2(filt)  # complex64 if safe_dtype=float32
    

def set_operator(img_shape, IR_mode):
    if IR_mode == "colorization":
        _, _, h, w = img_shape
        return color2gray

    elif IR_mode == "inpainting":
        _, _, h, w = img_shape
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        return lambda z: z * mask

    elif IR_mode == "super resolution":
        _, _, h, w = img_shape
        sr_scale_h, sr_scale_w = get_super_resolution_scales(h, w)
        down = torch.nn.AdaptiveAvgPool2d((h // sr_scale_h, w // sr_scale_w))
        # #maintain input image size
        # up = torch.nn.Upsample(size=(h, w), mode="bilinear", align_corners=False)
        return lambda z: down(z)
    
    if IR_mode == "super resolution embeds":
        _, _, h, w = img_shape
        #  Embedding dimensions change into [B, Cenc, Henc, Wenc]. So we need new scale_h and scale_w for embeddings.
        down = torch.nn.AdaptiveAvgPool2d((h // scale_h, w // scale_w))
        return lambda z: down(z)
    
    # ---- NEW: Denoising ----
    elif IR_mode == "denoising":
        # Identity measurement: y = x
        return lambda z: z

    # ---- NEW: Deblurring (circular conv via FFT) ----
    elif IR_mode == "deblurring":
        _, c, h, w = img_shape
        k2d = gaussian_2d_kernel(blur_sigma, kernel_size, device="cpu", dtype=torch.float32)

        def A_blur(z: torch.Tensor):
            # Do FFT math in float32; cast back to original dtype at the end
            orig_dtype = z.dtype
            z32 = z.to(torch.float32)
            H_hat = _fft_filter_from_kernel(k2d, z32.shape[-2], z32.shape[-1], z32.shape[1],
                                            device=z32.device, dtype=z32.dtype)
            out32 = torch.fft.ifft2(torch.fft.fft2(z32) * H_hat).real
            return out32.to(orig_dtype)

        return A_blur

    elif IR_mode == "old photo restoration":
        _, _, h, w = img_shape
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        A1 = lambda z: z * mask
        A2 = color2gray
        sr_scale_h, sr_scale_w = get_super_resolution_scales(h, w)
        A3 = torch.nn.AdaptiveAvgPool2d((h // sr_scale_h, w // sr_scale_w))
        return lambda z: A3(A2(A1(z)))

    else:
        raise ValueError(f"Unknown IR_mode: {IR_mode}")
    

def set_pinv_operator(img_shape, IR_mode):
    _, _, h, w = img_shape
    if IR_mode == "colorization":
        return gray2color

    elif IR_mode == "inpainting":
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        return lambda z: z * mask

    elif IR_mode == "super resolution":
        # The input to Ap is the downsampled image, so we upsample with PatchUpsample
        sr_scale_h, sr_scale_w = get_super_resolution_scales(h, w)
        return lambda z: PatchUpsampleEmbeds(z, sr_scale_h, sr_scale_w)
    
    elif IR_mode == "super resolution embeds":
        # The input to Ap is the downsampled image, so we upsample with PatchUpsample
        return lambda z: PatchUpsampleEmbeds(z, scale_h, scale_w)
    
    # ---- NEW: Denoising ----
    elif IR_mode == "denoising":
        # Pseudoinverse of Identity is Identity (both left and right inverse)
        return lambda z: z

    # ---- NEW: Deblurring pinv(A) via Wiener/Tikhonov deconvolution ----
    elif IR_mode == "deblurring":
        _, c, h, w = img_shape
        k2d = gaussian_2d_kernel(blur_sigma, kernel_size, device="cpu", dtype=torch.float32)

        def A_pinv_blur(y: torch.Tensor):
            orig_dtype = y.dtype
            y32 = y.to(torch.float32)
            H_hat = _fft_filter_from_kernel(k2d, y32.shape[-2], y32.shape[-1], y32.shape[1],
                                            device=y32.device, dtype=y32.dtype)
            Y_hat = torch.fft.fft2(y32)
            denom = (H_hat.conj() * H_hat).real + float(wiener_lambda)  # regularized
            H_pinv = H_hat.conj() / denom
            x32 = torch.fft.ifft2(Y_hat * H_pinv).real
            return x32.to(orig_dtype)

        return A_pinv_blur

    elif IR_mode == "old photo restoration":
        sr_scale_h, sr_scale_w = get_super_resolution_scales(h, w)
        mask = torch.ones(img_shape, device=device)
        mask[:, :, h//4:h*3//4, w//4:w*3//4] = 0
        A1p = lambda z: z * mask
        A2p = gray2color
        A3p = lambda z: PatchUpsampleEmbeds(z, sr_scale_h, sr_scale_w)
        return lambda z: A1p(A2p(A3p(z)))

    else:
        raise ValueError(f"Unknown IR_mode: {IR_mode}")
# -----------------------------------------------



# ------------ DDNM pipeline function to be imported to the edit.py script -----------
def ddnm_simple(xt, y, z, t, lambda_t=1, IR_mode="super resolution"):
# Refer: https://arxiv.org/pdf/2212.00490
# https://github.com/wyhuai/DDNM

    x0t = xt

    A = set_operator(x0t.shape, IR_mode)
    Ap = set_pinv_operator(x0t.shape, IR_mode)   
    
    y0_hat = A(x0t)
    yres = y - y0_hat
    xres = Ap(yres)
    DDNM_xt= x0t + lambda_t*xres
    
    return DDNM_xt

def ddnm_flow(x0, y, v, t, lambda_t=1, IR_mode="super resolution"):
    """
    v is the velocity at time t, x0 is the estimate for x0 (the clean image estimate given xt)
    """
    # Refer: PNP flow https://arxiv.org/pdf/2410.02423

    # Expected value of x0, given xt
    x0t = x0

    A = set_operator(x0t.shape, IR_mode)
    Ap = set_pinv_operator(x0t.shape, IR_mode)   
    
    y0_hat = A(x0t)
    yres = y - y0_hat
    xres = Ap(yres)
    DDNM_x0t= x0t + lambda_t*xres

    # DDNM_xt = DDNM_x0t
    DDNM_xt = DDNM_x0t + t*v
    
    return DDNM_xt



# --------- Image Processing Pipeline to create the degraded images dataset----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a single degradation type from clean images")
    parser.add_argument("--input_dir", default="../demo/inputs",
                        help="Folder of clean input images")
    parser.add_argument("--output_dir", default="../demo/degraded/superres_4x",
                        help="Folder to save degraded output images")
    parser.add_argument("--ir_mode", default="super resolution",
                        choices=["super resolution", "colorization", "denoising", "deblurring"],
                        help="Degradation type to apply")
    _args = parser.parse_args()
    data_folder = _args.input_dir
    output_folder = _args.output_dir
    _ir_mode = _args.ir_mode

    print(f"Using device: {device}")
    print(f"Degradation mode: {_ir_mode}")
    os.makedirs(output_folder, exist_ok=True)

    original_y_folder = None
    if _ir_mode == "super resolution":
        original_y_folder = f"{output_folder}_original_y"
        os.makedirs(original_y_folder, exist_ok=True)

    transform = T.ToTensor()
    image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(("jpg", "png", "jpeg"))]

    for fname in image_files:
        path = os.path.join(data_folder, fname)
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        A = set_operator(img_tensor.shape, _ir_mode)
        degraded = A(img_tensor)

        if _ir_mode == "denoising":
            noise_std = torch.as_tensor(sigma, dtype=img_tensor.dtype, device=img_tensor.device) / 255.0
            noise = torch.randn_like(img_tensor) * noise_std
            degraded = torch.clamp(img_tensor + noise, 0.0, 1.0)

        if _ir_mode == "super resolution":
            y_lr_img = TF.to_pil_image(torch.clamp(degraded.squeeze(0), 0, 1).cpu())
            y_lr_img.save(os.path.join(original_y_folder, fname))
            sr_scale_h, sr_scale_w = get_super_resolution_scales(img_tensor.shape[2], img_tensor.shape[3])
            degraded = PatchUpsampleEmbeds(degraded, sr_scale_h, sr_scale_w)

        TF.to_pil_image(torch.clamp(degraded.squeeze(0), 0, 1).cpu()).save(
            os.path.join(output_folder, fname)
        )

    if _ir_mode == "super resolution":
        print(f"Saved {len(image_files)} LR measurements to {original_y_folder}/")
    print(f"Saved {len(image_files)} degraded images to {output_folder}/")
