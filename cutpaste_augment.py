#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter

from anomalydetect import IMAGE_SIZE


def make_cutpaste_image(
    rng: random.Random,
    bg_path: Path,
    src_path: Path,
    mask_path: Path,
    min_scale: float,
    max_scale: float,
) -> Image.Image:
    bg = Image.open(bg_path).convert("RGB").resize(IMAGE_SIZE)
    w, h = bg.size

    src = Image.open(src_path).convert("RGB").resize(IMAGE_SIZE)
    mask = Image.open(mask_path).convert("L").resize(IMAGE_SIZE, resample=Image.Resampling.NEAREST)
    patch_mask = mask.point(lambda v: 255 if v >= 16 else 0, mode="L")
    bbox = patch_mask.getbbox()
    if bbox is None:
        return bg
    patch = src.crop(bbox)
    patch_mask = patch_mask.crop(bbox)
    patch_w, patch_h = patch.size

    scale = rng.uniform(min_scale, max_scale)
    new_w = max(8, min(int(round(patch_w * scale)), w))
    new_h = max(8, min(int(round(patch_h * scale)), h))
    patch = patch.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    patch_mask = patch_mask.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    if new_w >= w or new_h >= h:
        return bg
    dx = rng.randint(0, w - new_w)
    dy = rng.randint(0, h - new_h)

    # Feather the defect boundary to reduce copy-paste seams.
    blur_radius = max(1, min(new_w, new_h) // 16)
    patch_mask = patch_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    out = bg.copy()
    out.paste(patch, (dx, dy), mask=patch_mask)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CutPaste anomaly generator.")
    parser.add_argument(
        "--object",
        type=str,
        default="all",
        help="VisA object/category to process, or 'all' for every object in the split CSV.",
    )
    parser.add_argument("--num", type=int, default=400, help="Number of synthetic images per object.")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("data/processed/classifier/split_assignments.csv"),
        help="Base classifier split CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/cutpaste_defects"),
        help="Root directory to save generated images; object subfolders are created under it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-scale", type=float, default=0.8, help="Min scale for masked defect patch.")
    parser.add_argument("--max-scale", type=float, default=1.2, help="Max scale for masked defect patch.")
    args = parser.parse_args()

    if args.num <= 0:
        raise ValueError("--num must be > 0")
    if not args.split_csv.is_file():
        raise FileNotFoundError(f"Missing split CSV: {args.split_csv}")

    raw_root = Path("data/raw/visa-anomaly-detection")
    if not raw_root.is_dir():
        raw_root = Path("data/raw")
    if not raw_root.is_dir():
        raise FileNotFoundError("Could not locate VisA data root under data/raw")

    def resolve_path(rel_path: str) -> Path:
        rel = rel_path.replace("\\", "/").lstrip("/")
        if rel.startswith("generated/"):
            return Path("data") / rel
        return raw_root / rel

    df = pd.read_csv(args.split_csv)
    objects_all = sorted(str(x) for x in df["object"].dropna().astype(str).unique())
    if str(args.object).lower() == "all":
        target_objects = objects_all
    else:
        target_objects = [str(args.object)]

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for obj in target_objects:
        normals = df[
            (df["split"] == "train")
            & (df["object"].astype(str) == obj)
            & (df["binary_label"].astype(str) == "normal")
        ].copy()
        anomalies = df[
            (df["split"] == "train")
            & (df["object"].astype(str) == obj)
            & (df["binary_label"].astype(str) == "anomaly")
        ].copy()

        if normals.empty or anomalies.empty:
            print(
                f"Skipping object={obj!r}: need train normal + anomaly rows "
                f"(normals={len(normals)}, anomalies={len(anomalies)})."
            )
            continue

        normal_paths = [resolve_path(str(x)) for x in normals["image"].tolist()]
        normal_paths = [p for p in normal_paths if p.is_file()]
        if not normal_paths:
            print(f"Skipping object={obj!r}: no valid normal image files resolved.")
            continue

        anomaly_pairs: list[tuple[Path, Path]] = []
        for _, row in anomalies.iterrows():
            image_rel = str(row["image"])
            mask_rel = str(row.get("mask", "")).strip()
            if not mask_rel:
                continue
            img_path = resolve_path(image_rel)
            mask_path = resolve_path(mask_rel)
            if img_path.is_file() and mask_path.is_file():
                anomaly_pairs.append((img_path, mask_path))
        if not anomaly_pairs:
            print(f"Skipping object={obj!r}: no anomaly image/mask pairs resolved.")
            continue

        out_dir_obj = args.output_dir / obj
        out_dir_obj.mkdir(parents=True, exist_ok=True)

        for i in range(args.num):
            bg_path = rng.choice(normal_paths)
            src_path, mask_path = rng.choice(anomaly_pairs)
            out = make_cutpaste_image(
                rng,
                bg_path=bg_path,
                src_path=src_path,
                mask_path=mask_path,
                min_scale=float(args.min_scale),
                max_scale=float(args.max_scale),
            )
            out_path = out_dir_obj / f"cutpaste_{obj}_sample_{i:03d}.png"
            out.save(out_path)
            total_written += 1

        print(f"Wrote {args.num} CutPaste image(s) for object={obj!r} to {out_dir_obj.resolve()}")

    print(f"Done. Wrote {total_written} total CutPaste image(s) under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

