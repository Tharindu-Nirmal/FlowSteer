import math
from typing import Callable

import torch
from torch.distributions import Beta
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder

from .ddnm_degrads import ddnm_simple, get_lambda_schedule_config
from .lambda_schdules import make_lambda_schedule, make_lambda_ramp_schedule, make_lambda_step_schedule
from .util import (load_ae)
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


# Reorder the function definition to be used in denoise.
def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

# --- helpers for debugging---
def tensor_chw_neg1to1_to_pil(x_chw: torch.Tensor) -> Image.Image:
    """
    x_chw: torch.Tensor with shape (C,H,W), values in [-1,1]
    returns: PIL RGB image
    """
    x = x_chw.detach().float().clamp(-1, 1)
    x = (x + 1.0) * 127.5  # to [0,255]
    x = x.round().clamp(0, 255).to(torch.uint8)
    x = rearrange(x, "c h w -> h w c").cpu().numpy()
    return Image.fromarray(x, mode="RGB")

def save_image_grid_with_labels(images, labels, out_path, cols=6, pad=8, caption_h=22, bg=(255,255,255)):
    """
    images: list of PIL Images (all same size)
    labels: list of strings (same length as images)
    cols:   number of columns in the grid
    pad:    padding between tiles (px)
    caption_h: reserved height under each tile for the label
    """
    assert len(images) == len(labels) and len(images) > 0
    w, h = images[0].size
    n = len(images)
    rows = math.ceil(n / cols)

    grid_w = pad + cols*(w + pad)
    grid_h = pad + rows*(h + caption_h + pad)

    canvas = Image.new("RGB", (grid_w, grid_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    for idx, (im, text) in enumerate(zip(images, labels)):
        r = idx // cols
        c = idx % cols
        x0 = pad + c*(w + pad)
        y0 = pad + r*(h + caption_h + pad)

        canvas.paste(im, (x0, y0))

        # textbbox gives (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        tx = x0 + (w - tw)//2
        ty = y0 + h + (caption_h - th)//2
        draw.text((tx, ty), text, fill=(0,0,0), font=font)

    canvas.save(out_path, quality=95, subsampling=0)


def make_degradation_lambda_schedule(timesteps, degradation_type, overrides=None):
    schedule_config = get_lambda_schedule_config(degradation_type, overrides)
    schedule_kind = schedule_config.pop("kind", "step")

    if schedule_kind == "step":
        return make_lambda_step_schedule(timesteps=timesteps, **schedule_config)
    if schedule_kind == "ramp":
        return make_lambda_ramp_schedule(timesteps=timesteps, **schedule_config)
    if schedule_kind == "beta":
        return make_lambda_schedule(timesteps=timesteps, **schedule_config)

    raise ValueError(f"Unknown lambda schedule kind: {schedule_kind}")


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    y: Tensor,
    z: Tensor,
    degradation_type: str,
    width,
    height,
    inverse,
    info,
    name: str = "flux-dev",
    offload: bool = False,
    guidance: float = 4.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    # edits for ddnm update: The order here is for going from noise to image.
    lambda_log: list[float] = []
    t_log: list[float] = []

    edit_count = 15 # last steps to do the edit
    final_pad = 3 # last steps to skip ddnm

    torch_device = torch.device(device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    
    #having the z latent in image space for ddnm update
    z = unpack(z, height, width) #[B,C,H,W]
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        z = ae.decode(z)

    #"Inverting" in this code is the processing of converting the image into noise.
    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    lambda_sched = make_degradation_lambda_schedule(
        timesteps,
        degradation_type,
        overrides=info.get("lambda_schedule_overrides"),
    )

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    frames, labels = [], []

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]
        lambda_t = float(lambda_sched[i].item())
        info['lambda_t'] = lambda_t
        lambda_log.append(lambda_t)
        t_log.append(float(info['t']))


        #vhat_(ti) in algorithm 1 of the paper
        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        

        #Z_(ti + delta ti)
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True

        #vhat_(ti + delta ti) in algorithm 1 of the paper
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        #Calculating acceleration (the derivate of velocity).
        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)

        #Second order update for the Latent.
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

        # Z_(1) from the PnP-Flow approach:
        img_clean_hat = img - (t_curr * pred)

        #ddnm update in image space. y should be in image space.
        if (not(inverse) and (lambda_t > 0.0)):
            # decode
            print(f"Performing DDNM update at step {i} with t={info['t']:.4f}")
            img = unpack(img, height, width) #[B,C,H,W]
            img_clean_hat = unpack(img_clean_hat, height, width) #[B,C,H,W]
            vt = unpack(pred, height, width) #[B,C,H,W]
            
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                img = ae.decode(img)
                img_clean_hat = ae.decode(img_clean_hat)
                vt = ae.decode(vt)
            

            t = info['t']

            img = ddnm_simple(img, y, z, t, lambda_t=lambda_t, IR_mode=degradation_type) # both y and img are in [B,C,H,W]

            # The only relevant part from the encode() function
            img = ae.encode(img.to()).to(torch.bfloat16)
            # print("encoded img shape:", img.shape) #[1,16,40,60]

            # The only relevant part in the prepare() function
            img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # Debugging: Convert to image space and save as of this point
        img_debug = unpack(img, height, width) #[B,C,H,W]
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            img_debug = ae.decode(img_debug)   #[B,C,H,W], ~[-1,1]
        
        # bring into PIL format and save
        x_debug = tensor_chw_neg1to1_to_pil(img_debug[0])
        frames.append(x_debug)
        labels.append(f"itr{i}_t={info['t']}")

    # After the loop, save one grid image with captions
    save_image_grid_with_labels(frames, labels,out_path=f"debug_grid_{'img2noise' if inverse else 'noise2img'}.png",
                                cols=6, pad=8, caption_h=22)
    

    # --- Save timesteps and lambda schedule in one image ---
    try:
        xs = list(range(len(t_log)))

        fig, ax1 = plt.subplots(figsize=(9,4.5))
        # Left axis: time t
        ax1.plot(xs, t_log, marker='o', linewidth=1.5)
        ax1.set_xlabel('Step index (i)')
        ax1.set_ylabel('Timestep t', rotation=90)
        ax1.grid(True, linestyle='--', alpha=0.4)

        # Right axis: lambda_t
        ax2 = ax1.twinx()
        ax2.plot(xs, lambda_log, linestyle='--', linewidth=1.8)
        ax2.set_ylabel('λ_t', rotation=90)

        title_dir = 'img→noise' if inverse else 'noise→img'
        fig.suptitle(f'Time & λ Schedule ({title_dir}) | steps={len(xs)}  | peak λ={max(lambda_log):.3f}')
        fig.tight_layout()
        fig.savefig(f"t_and_lambda_schedule_{'img2noise' if inverse else 'noise2img'}.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print('Schedule plot failed:', e)

    return img, info


