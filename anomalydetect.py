"""
VisA: integrity cleanup (remove bad .jpgs + update image_anno.csv), counts,
then stratified train/val/test 60/20/20 by binary class (normal vs anomaly).
Writes split_assignments.csv and copies each image under data/processed/by_split/{train,val,test}/…

Run: python anomalydetect.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DOWNLOADED_FOLDER = "visa-anomaly-detection"
OUTPUT_DIR = Path("data/raw")
SPLIT_OUT = Path("data/processed/split_assignments.csv")
BY_SPLIT_ROOT = Path("data/processed/by_split")
RANDOM_STATE = 42


if __name__ == "__main__":
    out = OUTPUT_DIR.resolve()
    data_root: Path | None = None
    for candidate in (out, out / DOWNLOADED_FOLDER):
        if not candidate.is_dir():
            continue
        for child in candidate.iterdir():
            if child.is_dir() and (child / "image_anno.csv").is_file():
                data_root = candidate
                break
        if data_root is not None:
            break

    if data_root is None:
        print(
            f"No dataset found under {out} or {out / DOWNLOADED_FOLDER} "
            "(expected a category folder with image_anno.csv).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    parts: list[pd.DataFrame] = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        anno = d / "image_anno.csv"
        if not anno.is_file():
            continue
        part = pd.read_csv(anno)
        part = part.copy()
        part["object"] = d.name
        parts.append(part)

    if not parts:
        print(f"No image_anno.csv files under {data_root}.", file=sys.stderr)
        raise SystemExit(1)

    df = pd.concat(parts, ignore_index=True)
    df["binary_label"] = df["label"].map(
        lambda x: "normal" if str(x).strip().lower() == "normal" else "anomaly"
    )

    print(f"Data root: {data_root}\n")

    print("Integrity (.jpg paths in manifests)")
    bad: list[tuple[str, str]] = []
    for rel in df["image"].astype(str):
        rel_clean = rel.replace("\\", "/").lstrip("/")
        path = (data_root / rel_clean).resolve()
        if not path.is_file():
            bad.append((rel, "missing"))
            continue
        if path.suffix.lower() != ".jpg":
            bad.append((rel, f"not .jpg: {path.suffix!r}"))
            continue
        try:
            with Image.open(path) as im:
                im.load()
        except Exception as e:
            bad.append((rel, str(e)))

    print(f"Checked {len(df)} paths. Bad: {len(bad)}")
    for rel, err in bad:
        print(f"  {rel} -> {err}")

    bad_rels = {rel for rel, _ in bad}
    for rel in bad_rels:
        rel_clean = rel.replace("\\", "/").lstrip("/")
        fp = (data_root / rel_clean).resolve()
        if fp.is_file():
            fp.unlink()
            print(f"Removed file: {fp}")

    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        anno = d / "image_anno.csv"
        if not anno.is_file():
            continue
        sub = pd.read_csv(anno)
        before = len(sub)
        sub = sub[~sub["image"].astype(str).isin(bad_rels)]
        if len(sub) < before:
            sub.to_csv(anno, index=False)
            print(f"Updated {anno.name} in {d.name}: {before} -> {len(sub)} rows")

    if bad_rels:
        df = df[~df["image"].isin(bad_rels)].reset_index(drop=True)

    print("\nCounts (after cleanup; image_anno per category)")
    print("Global:")
    for k, v in df.groupby("binary_label", dropna=False).size().items():
        print(f"  {k}: {int(v)}")
    print("\nBy category x class:")
    print(
        df.groupby(["object", "binary_label"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nTrain / val / test (60% / 20% / 20%, stratified by normal vs anomaly)")
    n = len(df)
    if n == 0:
        print("No rows left after cleanup; cannot split.", file=sys.stderr)
        raise SystemExit(1)
    vc = df["binary_label"].value_counts()
    if vc.min() < 2:
        print(
            "Need at least 2 samples per class to stratify; adjust data or split manually.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df["binary_label"],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["binary_label"],
        random_state=RANDOM_STATE,
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(SPLIT_OUT, index=False)
    print(f"Wrote {SPLIT_OUT} ({len(split_df)} rows)")
    print(
        split_df.groupby(["split", "binary_label"]).size().unstack(fill_value=0).to_string()
    )
    print(
        "\nFractions of total:",
        f"train {len(train_df) / n:.3f}, val {len(val_df) / n:.3f}, test {len(test_df) / n:.3f}",
    )

    # Folder layout: same relative paths as under data/raw, grouped under train/val/test (full copies).
    if BY_SPLIT_ROOT.exists():
        shutil.rmtree(BY_SPLIT_ROOT)
    BY_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for _, row in split_df.iterrows():
        rel = str(row["image"]).replace("\\", "/").lstrip("/")
        src = (data_root / rel).resolve()
        if not src.is_file():
            continue
        dst = BY_SPLIT_ROOT / str(row["split"]) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1
    print(f"\nFolder copy: {BY_SPLIT_ROOT}/{{train,val,test}}/… ({n_copied} files)")
