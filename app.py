#!/usr/bin/env python3

# Gradio UI: browse real VisA anomalies vs saved VAE PNGs, decode random z or
# manual first-dim sliders with per-class ConvVAE checkpoints, and mu-space interpolation
# between two uploads (same decode path as latent_viz.run_interpolation).

# Paths default to repo-relative constants; override with CLI flags or env
# DEFECT_GALLERY_* so Quest vs laptop need no code edits.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import gradio as gr
from torchvision.utils import make_grid

from latent_viz import best_checkpoint, load_vae
from model_vae import choose_data_root, get_vae_eval_transform

# Defaults (override with argparse / environment below).
ARTIFACTS_VAE_DEFAULT = Path("artifacts/vae")
VAE_SYNTH_DIR_DEFAULT = Path("data/generated/vae_defects")
_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}  # (resolved .pt path, device str)


def _device_from_flag(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def list_object_names(artifacts_root: Path) -> list[str]:
    if not artifacts_root.is_dir():
        return []
    return sorted(p.name for p in artifacts_root.iterdir() if p.is_dir() and p.name != "plots")


def real_anomaly_paths(data_root: Path, obj: str) -> list[Path]:
    d = data_root / obj / "Data" / "Images" / "Anomaly"
    if not d.is_dir():
        return []
    out: list[Path] = []
    for pattern in ("*.JPG", "*.jpg", "*.jpeg", "*.png", "*.PNG"):
        out.extend(d.glob(pattern))
    return sorted({p.resolve() for p in out})


def synth_paths(root: Path, obj: str) -> list[Path]:
    d = root / obj
    if not d.is_dir():
        return []
    out: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.PNG", "*.JPEG"):
        out.extend(d.glob(pattern))
    return sorted(out)


def tensor01_chw_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)


def get_cached_vae(obj: str, checkpoint_str: str, artifacts_root: Path, device: torch.device):
    ckpt_path = Path(checkpoint_str).expanduser() if checkpoint_str.strip() else None
    if ckpt_path is None or not ckpt_path.is_file():
        cand = best_checkpoint(artifacts_root / obj)
        if cand is None:
            raise FileNotFoundError(f"No checkpoint for object {obj!r} under {artifacts_root}")
        ckpt_path = cand
    key = (str(ckpt_path.resolve()), str(device))
    # Cache avoids reloading large checkpoints on every UI click.
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_vae(ckpt_path, device)
    return _MODEL_CACHE[key], ckpt_path


def gallery_preview(
    source: str,
    obj: str,
    index: float,
    artifacts_root: Path,
    vae_synth_dir: Path,
):
    if not obj:
        return None, "_Pick an object._", gr.update(maximum=0, value=0)
    try:
        data_root = choose_data_root()
    except FileNotFoundError:
        data_root = None

    if source == "Real (VisA anomalies)":
        # Real path uses raw VisA tree to let users inspect the original defects.
        paths = real_anomaly_paths(data_root, obj) if data_root else []
        ckpt_note = ""
    elif source == "VAE generations":
        # Synth path mirrors sample_vae/model_vae output folders.
        paths = synth_paths(vae_synth_dir, obj)
        bc = best_checkpoint(artifacts_root / obj)
        ckpt_note = f"\n\nBest checkpoint (for Decode / Interpolate): `{bc.name}`" if bc else ""
    n = len(paths)
    if n == 0:
        return None, f"No images found for **{obj}** in this source.{ckpt_note}", gr.update(maximum=0, value=0)
    max_i = n - 1
    i = int(np.clip(round(index), 0, max_i))
    p = paths[i]
    try:
        img = Image.open(p).convert("RGB")
    except OSError:
        return None, f"Could not open `{p}`", gr.update(maximum=max_i, value=i)
    cap = (
        f"**{p.name}**  \n`{p.as_posix()}`  \n{i + 1} / {n}  \nSource: {source}"
        + (ckpt_note.replace("\n\n", "\n") if ckpt_note else "")
    )
    return img, cap, gr.update(maximum=max_i, value=i)


def decode_random_z(
    obj: str,
    checkpoint_override: str,
    seed: float,
    artifacts_root: Path,
    device_torch: torch.device,
):
    if not obj:
        return None, "_Pick an object._"
    try:
        model, ckpt_used = get_cached_vae(obj, checkpoint_override.strip(), artifacts_root, device_torch)
    except Exception as e:
        return None, f"**Error:** {e}"
    # Seed makes random samples reproducible for demos/report screenshots.
    torch.manual_seed(int(seed))
    z = torch.randn(1, model.latent_dim, device=device_torch)
    with torch.no_grad():
        out = model.decode(z).clamp(0, 1)
    pil = tensor01_chw_to_pil(out[0])
    cap = (
        f"**Random z ~ N(0, I)**  \nCheckpoint: `{ckpt_used.name}`  \n`{ckpt_used.as_posix()}`  \n"
        f"latent_dim={model.latent_dim}, seed={int(seed)}"
    )
    return pil, cap


def decode_slider_z(
    obj: str,
    checkpoint_override: str,
    seed: float,
    z0: float,
    z1: float,
    z2: float,
    z3: float,
    artifacts_root: Path,
    device_torch: torch.device,
):
    if not obj:
        return None, "_Pick an object._"
    try:
        model, ckpt_used = get_cached_vae(obj, checkpoint_override.strip(), artifacts_root, device_torch)
    except Exception as e:
        return None, f"**Error:** {e}"
    # Start from a seeded random base latent so slider deltas are visible.
    g = torch.Generator(device=device_torch).manual_seed(int(seed))
    z = torch.randn(1, model.latent_dim, generator=g, device=device_torch)
    z[0, 0] += z0
    if model.latent_dim > 1:
        z[0, 1] += z1
    if model.latent_dim > 2:
        z[0, 2] += z2
    if model.latent_dim > 3:
        z[0, 3] += z3
    with torch.no_grad():
        out = model.decode(z).clamp(0, 1)
    pil = tensor01_chw_to_pil(out[0])
    cap = (
        f"**Manual z deltas** (applied to random base z, seed={int(seed)})  \n"
        f"Checkpoint: `{ckpt_used.name}`  \n`{ckpt_used.as_posix()}`"
    )
    return pil, cap


def interpolate_ui(
    img_a: Image.Image | None,
    img_b: Image.Image | None,
    obj: str,
    checkpoint_override: str,
    steps: float,
    artifacts_root: Path,
    device_torch: torch.device,
):
    if not obj:
        return None, "_Pick an object._"
    if img_a is None or img_b is None:
        return None, "Upload two images."
    try:
        model, ckpt_used = get_cached_vae(obj, checkpoint_override.strip(), artifacts_root, device_torch)
    except Exception as e:
        return None, f"**Error:** {e}"
    tfm = get_vae_eval_transform()
    # Reuse VAE eval transform so UI inputs match training/eval preprocessing.
    x1 = tfm(img_a).unsqueeze(0).to(device_torch)
    x2 = tfm(img_b).unsqueeze(0).to(device_torch)
    n_steps = int(np.clip(round(steps), 3, 25))
    with torch.no_grad():
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
    frames: list[torch.Tensor] = []
    with torch.no_grad():
        for t in torch.linspace(0, 1, n_steps, device=device_torch):
            mu = (1 - t) * mu1 + t * mu2
            frames.append(model.decode(mu).clamp(0, 1))
    strip = torch.cat(frames, dim=0)
    grid = make_grid(strip, nrow=n_steps)
    pil = tensor01_chw_to_pil(grid)
    cap = f"**Interpolation in μ space** ({n_steps} steps)  \nCheckpoint: `{ckpt_used.name}`"
    return pil, cap


def build_app(
    artifacts_root: Path,
    vae_synth_dir: Path,
    device_flag: str,
):
    device_torch = _device_from_flag(device_flag)
    objects_ = list_object_names(artifacts_root)
    if not objects_:
        objects_ = ["(no folders in artifacts/vae)"]

    def wrap_gallery(src, obj, idx):
        return gallery_preview(src, obj, idx, artifacts_root, vae_synth_dir)

    def wrap_rand(obj, ck, seed):
        return decode_random_z(obj, ck, seed, artifacts_root, device_torch)

    def wrap_slide(obj, ck, seed, z0, z1, z2, z3):
        return decode_slider_z(obj, ck, seed, z0, z1, z2, z3, artifacts_root, device_torch)

    def wrap_interp(a, b, obj, ck, st):
        return interpolate_ui(a, b, obj, ck, st, artifacts_root, device_torch)

    # Single-page tabs keep this lightweight while still covering browse/decode/interpolate.
    with gr.Blocks(title="Defect sample gallery") as demo:
        gr.Markdown(
            "### VisA defect gallery + VAE decode  \n"
            "Browse **Real** images and **VAE** synth PNGs. "
            "Decode uses the same `[0,1]` VAE transforms as `model_vae` / `latent_viz`."
        )
        with gr.Tabs():
            with gr.Tab("Gallery"):
                src = gr.Radio(
                    label="Source",
                    choices=["Real (VisA anomalies)", "VAE generations"],
                    value="Real (VisA anomalies)",
                )
                obj_g = gr.Dropdown(label="Object class", choices=objects_, value=objects_[0])
                idx = gr.Slider(label="Image index", minimum=0, maximum=100, value=0, step=1)
                gal_img = gr.Image(label="Preview", type="pil")
                gal_cap = gr.Markdown()
                for comp in (src, obj_g, idx):
                    comp.change(wrap_gallery, [src, obj_g, idx], [gal_img, gal_cap, idx])

            with gr.Tab("Random z & sliders"):
                obj_d = gr.Dropdown(label="Object class", choices=objects_, value=objects_[0])
                ck_d = gr.Textbox(label="Checkpoint path (optional)", placeholder="Leave empty for best_val_recon")
                seed = gr.Number(label="RNG seed", value=42, precision=0)
                btn_r = gr.Button("Sample random z")
                out_d = gr.Image(label="Decoded", type="pil")
                cap_d = gr.Markdown()
                btn_r.click(wrap_rand, [obj_d, ck_d, seed], [out_d, cap_d])
                gr.Markdown("**First four latent dimensions** (deltas on top of random base z)")
                z0 = gr.Slider(label="z[0]", minimum=-3.0, maximum=3.0, value=0.0, step=0.05)
                z1 = gr.Slider(label="z[1]", minimum=-3.0, maximum=3.0, value=0.0, step=0.05)
                z2 = gr.Slider(label="z[2]", minimum=-3.0, maximum=3.0, value=0.0, step=0.05)
                z3 = gr.Slider(label="z[3]", minimum=-3.0, maximum=3.0, value=0.0, step=0.05)
                btn_s = gr.Button("Decode from sliders")
                btn_s.click(wrap_slide, [obj_d, ck_d, seed, z0, z1, z2, z3], [out_d, cap_d])

            with gr.Tab("Interpolation (μ)"):
                obj_i = gr.Dropdown(label="Object class", choices=objects_, value=objects_[0])
                ck_i = gr.Textbox(label="Checkpoint path (optional)", placeholder="Leave empty for best_val_recon")
                im_a = gr.Image(label="Image A", type="pil")
                im_b = gr.Image(label="Image B", type="pil")
                st = gr.Slider(label="Steps", minimum=3, maximum=25, value=11, step=1)
                btn_i = gr.Button("Interpolate & decode")
                out_i = gr.Image(label="Grid", type="pil")
                cap_i = gr.Markdown()
                btn_i.click(wrap_interp, [im_a, im_b, obj_i, ck_i, st], [out_i, cap_i])

        gr.Markdown(
            f"_artifact root:_ `{artifacts_root}` · "
            f"_VAE synth dir:_ `{vae_synth_dir}` · "
            f"_device:_ `{device_torch}`"
        )
        demo.load(fn=wrap_gallery, inputs=[src, obj_g, idx], outputs=[gal_img, gal_cap, idx])
    return demo


def env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


def main() -> None:
    p = argparse.ArgumentParser(description="Gradio gallery for VisA + VAE samples + decode.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Gradio public link.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--artifacts-root", type=Path, default=None, help="Per-class VAE checkpoints.")
    p.add_argument("--vae-synth-dir", type=Path, default=None, help="data/generated/vae_defects/<object>/")
    args = p.parse_args()

    ar = args.artifacts_root or env_path("DEFECT_GALLERY_ARTIFACTS_VAE", ARTIFACTS_VAE_DEFAULT)
    vs = args.vae_synth_dir or env_path("DEFECT_GALLERY_VAE_SYNTH", VAE_SYNTH_DIR_DEFAULT)
    demo = build_app(ar.expanduser().resolve(), vs.expanduser().resolve(), args.device)
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
