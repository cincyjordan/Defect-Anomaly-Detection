#!/usr/bin/env python3

# t-SNE and latent-space interpolation for per-object ConvVAE checkpoints (same [0,1] eval
# transforms as model_vae: Resize + ToTensor).

# mu: the encoder maps each image x to a diagonal Gaussian q(z|x); mu is the mean of that
# distribution (same shape as latent_dim). Training samples z = mu + sigma*noise for the
# decoder; for analysis we use mu alone, a deterministic summary of where the image
# sits in latent space, so we can compare images and run t-SNE without sampling noise.

# Reads data/processed/vae/split_assignments.csv (written by anomalydetect.py). Each VisA
# object has its own trained encoder under artifacts/vae/<object>/vae_trial_*.pt

# Outputs: by default outputs/latent_viz/<object>_tsne_<split|label>.png (point color from
# color-by). Interpolation: interpolate saves a single-row torchvision grid PNG.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Headless backend keeps plotting stable on SSH / servers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torchvision.utils import make_grid, save_image

from model_vae import AnomalyImageDataset, ConvVAE, choose_data_root, get_vae_eval_transform

VAE_SPLIT_CSV = Path("data/processed/vae/split_assignments.csv")


def load_vae(checkpoint_path: Path, device: torch.device) -> ConvVAE:
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload.get("config") or {}
    latent_dim = int(cfg.get("latent_dim", 128))
    m = ConvVAE(latent_dim=latent_dim).to(device)
    m.load_state_dict(payload["model_state_dict"])
    m.eval()
    return m


def best_checkpoint(class_dir: Path) -> Path | None:
    best_score = float("inf")
    best_p: Path | None = None
    # Pick by validation reconstruction so CLI defaults are reproducible.
    for ckpt in sorted(class_dir.glob("vae_trial_*.pt")):
        try:
            payload = torch.load(ckpt, map_location="cpu")
            s = float(payload.get("best_val_recon", float("inf")))
        except Exception:
            continue
        if s < best_score:
            best_score = s
            best_p = ckpt
    return best_p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE latent t-SNE and interpolation utilities.")
    p.add_argument("--object", type=str, default="", help="VisA object/class name (e.g. candle).")
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt for that object.")
    p.add_argument(
        "--all-objects",
        action="store_true",
        help="Run t-SNE for every class folder under --artifacts-root (best checkpoint each).",
    )
    p.add_argument("--artifacts-root", type=Path, default=Path("artifacts/vae"))
    p.add_argument("--split-csv", type=Path, default=VAE_SPLIT_CSV)
    p.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated: train,val,test")
    p.add_argument("--max-samples", type=int, default=400, help="Max images per object for t-SNE.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/latent_viz"))
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument(
        "--interpolate",
        nargs=2,
        metavar=("IMAGE_A", "IMAGE_B"),
        default=None,
        help="Two image paths; saves a latent interpolation grid (--checkpoint required).",
    )
    p.add_argument("--steps", type=int, default=11, help="Interpolation steps including endpoints.")
    p.add_argument("--interp-output", type=Path, default=Path("outputs/latent_viz/interp_grid.png"))
    p.add_argument(
        "--color-by",
        choices=("split", "label"),
        default="split",
        help="t-SNE point color: data split, or VisA anomaly label (type).",
    )
    return p.parse_args()


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_flag)

# Build (object name, checkpoint path) pairs for the t-SNE loop.
# With --all-objects: one entry per subfolder of artifacts/vae (skips "plots"), best
# checkpoint by lowest best_val_recon. Otherwise: single --object, using --checkpoint
# if valid else best trial under artifacts/vae/<object>/.
def collect_tsne_jobs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    if args.all_objects:
        if not args.artifacts_root.is_dir():
            raise FileNotFoundError(f"Missing artifacts root: {args.artifacts_root}")
        for class_dir in sorted(p for p in args.artifacts_root.iterdir() if p.is_dir() and p.name != "plots"):
            ckpt = best_checkpoint(class_dir)
            if ckpt is None:
                print(f"[{class_dir.name}] no checkpoint found, skip")
                continue
            jobs.append((class_dir.name, ckpt))
        return jobs

    if not args.object:
        raise SystemExit("Provide --object NAME, or use --all-objects")
    ckpt = args.checkpoint
    if ckpt is None or not ckpt.is_file():
        cand = best_checkpoint(args.artifacts_root / args.object)
        if cand is None:
            raise FileNotFoundError(f"No checkpoint for {args.object}; pass --checkpoint")
        ckpt = cand
    return [(args.object, ckpt)]

# Linear interpolation between encoder means: mu_t = (1-t)*mu_a + t*mu_b, decode each
# step, concat reconstructions, torchvision.make_grid in one row -> --interp-output.
def run_interpolation(args: argparse.Namespace, device: torch.device) -> None:
    if args.checkpoint is None or not Path(args.checkpoint).is_file():
        raise SystemExit("--interpolate requires --checkpoint to a valid .pt file")
    ckpt = Path(args.checkpoint)
    model = load_vae(ckpt, device)
    tfm = get_vae_eval_transform()
    p1, p2 = Path(args.interpolate[0]), Path(args.interpolate[1])
    if not p1.is_file() or not p2.is_file():
        raise FileNotFoundError(f"Interp paths must exist: {p1}, {p2}")
    x1 = tfm(Image.open(p1).convert("RGB")).unsqueeze(0).to(device)
    x2 = tfm(Image.open(p2).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        # Interpolate in mu (deterministic latent) instead of sampled z to avoid extra noise.
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
    frames: list[torch.Tensor] = []
    with torch.no_grad():
        for t in torch.linspace(0, 1, args.steps, device=device):
            mu = (1 - t) * mu1 + t * mu2
            frames.append(model.decode(mu).clamp(0, 1))
    strip = torch.cat(frames, dim=0)
    out = Path(args.interp_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(strip, nrow=args.steps)
    save_image(grid, str(out))
    print(f"Wrote interpolation grid ({args.steps} steps) to {out.resolve()}")

# Load VAE split CSV; for each job, subset rows by object and --splits, cap at
# --max-samples, encode batched mu, sklearn TSNE -> 2D, matplotlib scatter colored
# by --color-by, save outputs/latent_viz/<object>_tsne_<split|label>.png (default paths).
def run_tsne_visualization(args: argparse.Namespace, device: torch.device) -> None:
    if not args.split_csv.is_file():
        raise FileNotFoundError(f"Missing VAE split CSV: {args.split_csv}")
    frame = pd.read_csv(args.split_csv)
    want_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    jobs = collect_tsne_jobs(args)
    data_root = choose_data_root()
    color_key = "split" if args.color_by == "split" else "label"

    for obj, ckpt in jobs:
        sub = frame[frame["object"].astype(str) == obj].copy()
        sub = sub[sub["split"].astype(str).isin(want_splits)]
        if sub.empty:
            print(f"[{obj}] no rows for splits {want_splits}, skip")
            continue
        if len(sub) > args.max_samples:
            # Cap runtime and visual clutter; t-SNE gets slow/noisy with very large N.
            sub = sub.sample(n=args.max_samples, random_state=42).sort_index()
        sub = sub.reset_index(drop=True)

        model = load_vae(ckpt, device)
        ds = AnomalyImageDataset(sub, data_root, get_vae_eval_transform())
        mus: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(ds), args.batch_size):
                end = min(start + args.batch_size, len(ds))
                batch_list = [ds[i][0] for i in range(start, end)]
                batch = torch.stack(batch_list, dim=0).to(device)
                mu, _ = model.encode(batch)
                mus.append(mu.cpu().numpy())
        if not mus:
            Z = np.zeros((0, model.latent_dim))
        else:
            Z = np.concatenate(mus, axis=0)

        labels = [str(sub.iloc[i][color_key]) for i in range(len(sub))]
        title = f"t-SNE of mu ({args.color_by}) | {obj} | n={len(labels)} | {ckpt.name}"
        safe_obj = obj.replace("/", "_")
        png = args.out_dir / f"{safe_obj}_tsne_{args.color_by}.png"

        if Z.shape[0] < 3:
            print(f"Skip t-SNE ({title}): need >=3 points, got {Z.shape[0]}")
            continue
        # Adaptive perplexity keeps settings sane for both tiny and moderate sample counts.
        perplexity = min(30, max(5, Z.shape[0] // 4))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto")
        Z2 = tsne.fit_transform(Z)
        uniq = sorted(set(labels))
        base = plt.colormaps["tab10"] if hasattr(plt, "colormaps") else plt.cm.get_cmap("tab10")
        color_map = {u: base(i % 10) for i, u in enumerate(uniq)}
        plt.figure(figsize=(8, 6))
        for lab in uniq:
            mask = np.array([lab == ell for ell in labels])
            plt.scatter(Z2[mask, 0], Z2[mask, 1], s=18, alpha=0.75, label=lab, color=color_map[lab])
        plt.title(title)
        plt.legend(loc="best", fontsize=8, markerscale=1.5)
        plt.tight_layout()
        png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png, dpi=150)
        plt.close()
        print(f"Wrote {png}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.interpolate is not None:
        run_interpolation(args, device)
    else:
        run_tsne_visualization(args, device)


if __name__ == "__main__":
    main()
