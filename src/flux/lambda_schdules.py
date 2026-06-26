import torch

def make_lambda_schedule(timesteps: list[float],*,
    tail_frac: float = 0.22,    # e.g., last 22% of steps carry nonzeros
    peak_at: float = 0.90,      # mode inside the *windowed* [0,1]
    rise_smooth: float = 100.0, # bigger => gentler/slower rise (affects 'a')
    drop_sharp: float = 10.0,   # bigger => sharper fall (affects 'b')
    pre_peak_atten: float = 2.5,  # >1 narrows the peak; <1 fattens it
    power: float = 1.6,         # gamma for extra attenuation before peak
    floor: float = 0.0,         # min λ (0 keeps truly off before the window)
    final_pad: int = 1,         # force last N steps to zero (stability)
) -> torch.Tensor:
    import torch
    from torch.distributions import Beta

    T = len(timesteps) - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if T <= 0:
        return torch.zeros(0, device=device, dtype=dtype)

    # Iteration-aligned progress: u=0 at i=0, u=1 at i=T-1 for *current loop order*
    u = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)

    # ---- Window the last tail_frac of steps only ----
    tail_frac = float(min(max(tail_frac, 1e-3), 0.99))
    u_start = 1.0 - tail_frac
    in_tail = (u >= u_start).float()
    u_tail = torch.zeros_like(u)
    denom = (1.0 - u_start)
    u_tail[in_tail.bool()] = (u[in_tail.bool()] - u_start) / denom  # [u_start,1]→[0,1]

    # ---- Beta shape on the windowed coordinate ----
    b = max(drop_sharp, 2.0)
    a_raw = (1.0 + peak_at * (b - 2.0)) / max(1e-3, (1.0 - peak_at))
    a = max(2.0, a_raw + max(0.0, rise_smooth - 1.0))

    dist = Beta(torch.tensor(a, device=device, dtype=dtype),
                torch.tensor(b, device=device, dtype=dtype))
    pdf = dist.log_prob(torch.clamp(u_tail, 1e-5, 1.0 - 1e-5)).exp()
    pdf = pdf / (pdf.max() + 1e-8)

    # Pre-peak attenuation + optional narrowing
    if pre_peak_atten != 1.0:
        pdf = pdf * torch.pow(torch.clamp(u_tail, 0.0, 1.0), pre_peak_atten)
    if power != 1.0:
        pdf = torch.pow(pdf, power)
        pdf = pdf / (pdf.max() + 1e-8)

    # Apply window, hard-pad last N steps (in current loop order)
    lam = pdf * in_tail
    if final_pad > 0 and final_pad < T:
        lam[-final_pad:] = 0.0

    # Renormalize after masking/padding so peak==1 (unless all zero)
    m = lam.max()
    if m > 0:
        lam = lam / m
    else:
        # fallback: put a spike at the last kept index
        last_kept = max(0, T - final_pad - 1)
        lam[last_kept] = 1.0

    # Optional floor
    if floor > 0:
        lam = torch.clamp(lam, min=floor, max=1.0)

    return lam


def _to_idx(x, T):
    """Accept absolute index (int) or fraction in (0,1]; clamp to [0, T-1]."""
    if isinstance(x, float):
        i = int(round(x * (T-1)))
    else:
        i = int(x)
    return max(0, min(T-1, i))

def make_lambda_ramp_schedule(
    timesteps: list[float],
    *,
    startramp,          # int index or fraction of (T-1), e.g., 0.85
    endramp,            # int index or fraction, must be > startramp
    final_pad: int = 1, # zero the last N steps of the loop
) -> torch.Tensor:
    T = len(timesteps) - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = torch.zeros(T, device=device, dtype=torch.float32)
    if T <= 0:
        return lam

    s = _to_idx(startramp, T)
    e = _to_idx(endramp,   T)
    if e <= s:
        e = min(T-1, s+1)  # ensure at least 1 step of ramp

    # linear ramp from 0 at s to 1 at e
    span = max(1, e - s)
    vals = torch.linspace(0.0, 1.0, span + 1, device=device)  # inclusive end
    lam[s:e+1] = vals

    # hard pad last N steps
    if final_pad > 0 and final_pad < T:
        lam[-final_pad:] = 0.0

    # if peak got padded away, shift it left by one
    if lam.max() <= 0 and T - final_pad - 1 >= 0:
        lam[T - final_pad - 1] = 1.0
    return lam

def make_lambda_step_schedule(
    timesteps: list[float],
    *,
    start,              # int index or fraction
    step,               # boundary where it drops from hi to lo
    end,                # end of activity window (exclusive)
    level_hi: float = 1.0,
    level_lo: float = 0.5,
    final_pad: int = 1,
) -> torch.Tensor:
    T = len(timesteps) - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = torch.zeros(T, device=device, dtype=torch.float32)
    if T <= 0:
        return lam

    i0 = _to_idx(start, T)
    i1 = _to_idx(step,  T)
    i2 = _to_idx(end,   T) + 1  # make end exclusive

    # order & clamp
    i0, i1 = min(i0, i1), max(i0, i1)
    i1, i2 = min(i1, i2), max(i1, i2)
    i0 = max(0, min(T, i0))
    i1 = max(0, min(T, i1))
    i2 = max(0, min(T, i2))

    if i0 < i1:
        lam[i0:i1] = level_hi
    if i1 < i2:
        lam[i1:i2] = level_lo

    if final_pad > 0 and final_pad < T:
        lam[-final_pad:] = 0.0

    # keep a peak=1 if padding nuked it
    if lam.max() <= 0 and T - final_pad - 1 >= 0:
        lam[T - final_pad - 1] = 1.0
    else:
        # normalize only if we accidentally exceeded 1
        lam = lam / max(1.0, float(lam.max()))
    return lam
