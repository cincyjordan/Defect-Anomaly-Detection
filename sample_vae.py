#!/usr/bin/env python3
"""
Decode random latent vectors with a trained ConvVAE and save RGB images for inspection.

Synth files default to data/generated/vae_defects/ so they match build_classifier_augmented_split.

After full model_vae.py training (per-object), PNGs already land there as `<object>_best_sample_*.png`.
Use this script for ad-hoc sampling from any single checkpoint (e.g. artifacts/vae/candle/vae_trial_000.pt),
or run once across all class folders under artifacts/vae.

Run (from repo root):

  python sample_vae.py
  python sample_vae.py --checkpoint artifacts/vae/candle/vae_trial_000.pt --num 200 --seed 123
  python sample_vae.py --all-classes
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import torch

from model_vae import save_decoder_random_pngs


def pick_best_checkpoint_per_class(artifacts_root: Path) -> dict[str, Path]:
    best: dict[str, tuple[float, Path]] = {}
    for class_dir in sorted(p for p in artifacts_root.iterdir() if p.is_dir() and p.name != "plots"):
        for ckpt in sorted(class_dir.glob("vae_trial_*.pt")):
            try:
                payload = torch.load(ckpt, map_location="cpu")
                score = float(payload.get("best_val_recon", float("inf")))
            except Exception:
                continue
            current = best.get(class_dir.name)
            if current is None or score < current[0]:
                best[class_dir.name] = (score, ckpt)
    return {k: v[1] for k, v in best.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Save VAE random samples as PNG.")
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Scan artifacts/vae/<class>/ and sample from each class's best checkpoint.",
    )
    parser.add_argument(
        "--artifacts-root",
        dest="artifacts_root",
        type=Path,
        default=Path("artifacts/vae"),
        help="Root containing per-class checkpoint folders (used with --all-classes).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/vae/candle/vae_trial_000.pt"),
        help="Trained ConvVAE checkpoint (expects config + latent_dim).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("data/generated/vae_defects"),
        help="Directory for PNG outputs.",
    )
    parser.add_argument("--num", type=int, default=200, help="Number of random samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.all_classes:
        if not args.artifacts_root.is_dir():
            raise FileNotFoundError(f"Missing artifacts root: {args.artifacts_root}")
        picks = pick_best_checkpoint_per_class(args.artifacts_root)
        if not picks:
            raise FileNotFoundError(f"No class checkpoints found under {args.artifacts_root}")
        for class_name, ckpt in picks.items():
            out_dir = args.output_dir / class_name
            stem_prefix = Path(ckpt).stem.replace(".", "_")
            class_seed = (args.seed + zlib.adler32(class_name.encode())) & 0xFFFFFFFF
            save_decoder_random_pngs(
                ckpt,
                out_dir,
                num=args.num,
                seed=class_seed,
                stem_prefix=stem_prefix,
                device=device,
            )
            print(f"[{class_name}] wrote {args.num} PNG(s) to {out_dir.resolve()} from {ckpt}")
        return

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    stem_prefix = Path(args.checkpoint).stem.replace(".", "_")
    save_decoder_random_pngs(
        args.checkpoint,
        args.output_dir,
        num=args.num,
        seed=args.seed,
        stem_prefix=stem_prefix,
        device=device,
    )
    print(f"Wrote {args.num} PNG(s) under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
