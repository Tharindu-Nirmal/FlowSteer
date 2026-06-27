<div align="center">

# FlowSteer: Conditioning Flow Field for Consistent Image Restoration

> **FlowSteer: Conditioning Flow Field for Consistent Image Restoration** <br>
> [Tharindu Wickremasinghe](https://tharindu-nirmal.github.io/), [Chenyang Qi](https://scholar.google.com/citations?user=qNweIR4AAAAJ&hl=en), [Harshana Weligampola](https://harshana95.github.io/), [Zhengzhong Tu](https://vztu.github.io/), [Stanley H. Chan](https://engineering.purdue.edu/ChanGroup/stanleychan.html)<br>
> Purdue University &nbsp;·&nbsp; HKUST &nbsp;·&nbsp; Texas A&M University

[![Arxiv](https://img.shields.io/badge/arXiv-2512.08125-b31b1b.svg?style=for-the-badge&logo=arxiv)](https://www.arxiv.org/abs/2512.08125)
[![Project Page](https://img.shields.io/badge/Project-Page-green?style=for-the-badge)](https://tharindu-nirmal.github.io/FlowSteer/)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg?style=for-the-badge)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wickremasinghe_FlowSteer_Conditioning_Flow_Field_for_Consistent_Image_Restoration_CVPRF_2026_paper.pdf)
[![FlowSteer Dataset](https://img.shields.io/badge/FlowSteer100-Dataset-FF4F1D.svg?style=for-the-badge&logo=Huggingface)](https://huggingface.co/datasets/lwickrem/FlowSteer100)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)

</div>

---

This repository is the official implementation of [FlowSteer](https://www.arxiv.org/abs/2512.08125). FlowSteer is an operator-aware conditioning method that enables flow-based generative models (FLUX) to perform **zero-shot image restoration** — super-resolution, deblurring, denoising, and colorization — without retraining or task-specific adapters. 
<div align="center">
  <img src="docs/static/images/Website_Method_1.png" width="90%" alt="FlowSteer Method Overview">
</div>

FlowSteer injects a measurement prior along the sampling trajectory at each step, steering the flow toward clean images that are consistent with the degraded observation.

<div align="center">
  <img src="docs/static/images/Website_Method_2.png" width="90%" alt="FlowSteer Method Details">
</div>

For interactive visual results and comparisons, see the **[project page](https://tharindu-nirmal.github.io/FlowSteer/)**.

---

## 🔥 Latest News!
* [June 26, 2026]: Code is released on github.
* [Feb 21, 2026]: Flow Steer is accepted for CVPR 2026 - Findings track! 
* [Dec 10, 2025]: Released the [Project page](https://tharindu-nirmal.github.io/FlowSteer/).
* [Dec 9, 2025]: Paper available on [arXiv](https://www.arxiv.org/abs/2512.08125).

---

## Setup

### 1. Create the environment

```bash
conda create -n flowsteer python=3.10
conda activate flowsteer
pip install -r requirements-full.txt
pip install -e .        # installs the local flux package
```

### 2. Download the FLUX.1-dev model

Model weights are downloaded automatically from Hugging Face on the first run. Before running:

1. Accept the [FLUX.1-dev license](https://huggingface.co/black-forest-labs/FLUX.1-dev) on Hugging Face.
2. Log in to the Hub:

```bash
huggingface-cli login
```

If you already have the weights stored locally, point to them with environment variables instead:

```bash
export FLUX_DEV=/path/to/flux1-dev.safetensors
export AE=/path/to/ae.safetensors
```

---

## Quick Demo

Two sample images and their pre-generated degraded versions are included in `demo/`. Run all four restoration tasks on one of them directly:

```bash
cd src
bash batch_run.sh
```

Results are saved to `demo/outputs/`.

---

## Restore Your Own Image

**Step 1** — generate degraded versions of your clean image.

Place your image(s) in `demo/inputs/` and run:

```bash
python demo/prepare_demo.py
```

This writes four degraded versions per image to `demo/degraded/` (super-resolution, colorization, denoising, deblurring).

**Step 2** — run FlowSteer on a degraded image:

```bash
cd src
python edit_image.py \
    --source_prompt "A low resolution image of a cat." \
    --target_prompt "A high resolution image of a cat. Sharp fur details, photorealistic." \
    --degradation "super resolution" \
    --guidance 4 \
    --inject 5 \
    --source_img_dir ../demo/degraded/superres_4x/your_image.jpg \
    --output_dir ../demo/outputs/ \
    --num_steps 30 --name flux-dev --offload
```

### Supported Tasks

| Task | `--degradation` value |
|------|-----------------------|
| Super-resolution (4×) | `super resolution` |
| Deblurring | `deblurring` |
| Denoising | `denoising` |
| Colorization | `colorization` |

### Key Parameters

| Parameter | Typical range | Description |
|-----------|--------------|-------------|
| `--inject` | 4 – 9 | FlowSteer conditioning steps — higher values enforce the measurement prior more strongly |
| `--guidance` | 3 – 5 | Classifier-free guidance scale |
| `--num_steps` | 25 – 30 | Total flow steps |
| `--lambda_start` / `--lambda_step` / `--lambda_end` | 0.0 – 1.0 | Fraction of denoising steps over which the DDNM correction is active. **These should be tuned per degradation task** — the values in `batch_run.sh` provide task-specific starting points for super-resolution, colorization, denoising, and deblurring |

---

## Disclaimer

This project is released for academic use. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.

---

## ❣️ Acknowledgement

We thank [RF-Edit](https://rf-solver-edit.github.io/) for their work and codebase, and [SeeU](https://github.com/pandayuanyu/SeeU) for the project page template.

---

## 🌟 Citation

If you find this work helpful, please star this repo and cite our paper.

```bibtex
@article{Tharindu2025_FlowSteer,
  title={{FlowSteer}: Conditioning Flow Field for Consistent Image Restoration},
  author={Wickremasinghe, Tharindu and Qi, Chenyang and Weligampola, Harshana and Tu, Zhengzhong and Chan, Stanley H.},
  journal={arXiv preprint arXiv:2512.08125},
  year={2025}
}
```

---

## ✉️ Contact

If you have any questions or comments, feel free to contact us at tharindu.ncr@gmail.com. Suggestions and collaborations are also highly welcome!
