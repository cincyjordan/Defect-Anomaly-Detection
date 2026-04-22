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
from torchvision import transforms

DOWNLOADED_FOLDER = "visa-anomaly-detection"
OUTPUT_DIR = Path("data/raw")
CLASSIFIER_ROOT = Path("data/processed/classifier")
CLASSIFIER_SPLIT_OUT = CLASSIFIER_ROOT / "split_assignments.csv"
CLASSIFIER_BY_SPLIT_ROOT = CLASSIFIER_ROOT / "by_split"
VAE_ROOT = Path("data/processed/vae")
VAE_SPLIT_OUT = VAE_ROOT / "split_assignments.csv"
VAE_BY_SPLIT_ROOT = VAE_ROOT / "by_split"
RANDOM_STATE = 42

# Shared preprocessing settings for BOTH VAE and classifier inputs.
IMAGE_SIZE = (256, 256)
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def get_eval_transform() -> transforms.Compose:
    """Deterministic preprocessing for val/test/inference."""
    # Keep evaluation deterministic so metrics are comparable run-to-run.
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def get_train_transform(use_augmentation: bool = False) -> transforms.Compose:
    """
    Train preprocessing with the same size/normalization config.
    Optional augmentation is deliberately mild.
    """
    # Always enforce one size first so model input shape is fixed.
    steps: list[transforms.Transform] = [transforms.Resize(IMAGE_SIZE)]
    if use_augmentation:
        # Optional, mild augmentation only.
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    return transforms.Compose(steps)

if __name__ == "__main__":
    force_rebuild = "--force-rebuild" in sys.argv
    out = OUTPUT_DIR.resolve()
    data_root: Path | None = None
    # Support either layout:
    # 1) data/raw/<category>/...
    # 2) data/raw/visa-anomaly-detection/<category>/...
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

    # Build one unified manifest from each category's image_anno.csv.
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

    # Integrity pass: identify missing/unreadable rows before any counting/splitting.
    print("Integrity (.jpg paths in manifests)")
    bad: list[tuple[str, str]] = []
    for row in df.itertuples(index=False):
        rel = str(row.image)
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
    # Remove bad files from disk when present.
    for rel in bad_rels:
        rel_clean = rel.replace("\\", "/").lstrip("/")
        fp = (data_root / rel_clean).resolve()
        if fp.is_file():
            fp.unlink()
            print(f"Removed file: {fp}")

    # Keep manifests in sync by dropping rows for removed/bad paths.
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
    # All downstream stats are computed after cleanup.
    print("\nCounts (after cleanup; image_anno per category)")
    print("Global:")
    global_counts = df.groupby("binary_label", dropna=False).size()
    for k, v in global_counts.items():
        share = (v / len(df)) * 100 if len(df) else 0.0
        print(f"  {k}: {int(v)} ({share:.2f}%)")
    print("\nBy category x class:")
    by_object_class = df.groupby(["object", "binary_label"]).size().unstack(fill_value=0)
    print(by_object_class.to_string())
    print("\nBy category x class (% within category):")
    print((by_object_class.div(by_object_class.sum(axis=1), axis=0) * 100).round(2).to_string())

    # If processed splits already exist, run quality checks only.
    # Avoids rebuilding the splits and copying the images when not necessary.
    classifier_exists = CLASSIFIER_SPLIT_OUT.is_file()
    vae_exists = VAE_SPLIT_OUT.is_file()
    if classifier_exists and vae_exists and not force_rebuild:
        print(
            "\nProcessed split CSVs already exist. "
            "Skipping rebuild/copy and running quality checks only. "
            "Use --force-rebuild to regenerate."
        )

        # Classifier split checks from existing CSV.
        split_df = pd.read_csv(CLASSIFIER_SPLIT_OUT)
        print(f"\nLoaded {CLASSIFIER_SPLIT_OUT} ({len(split_df)} rows)")
        split_class_counts = split_df.groupby(["split", "binary_label"]).size().unstack(fill_value=0)
        print(split_class_counts.to_string())
        print("\nSplit x class (% within split):")
        print((split_class_counts.div(split_class_counts.sum(axis=1), axis=0) * 100).round(2).to_string())
        print("\nSplit quality checks")
        print("By split x object x class:")
        split_obj = (
            split_df.groupby(["split", "object", "binary_label"]).size().unstack(fill_value=0)
        )
        print(split_obj.to_string())
        min_anomaly_per_split_obj = (
            split_df[split_df["binary_label"] == "anomaly"].groupby(["split", "object"]).size()
        )
        low_support = min_anomaly_per_split_obj[min_anomaly_per_split_obj < 5]
        if len(low_support) > 0:
            print("\nWARNING: low anomaly support (<5) in split/object cells:")
            for (sp, obj), cnt in low_support.items():
                print(f"  {sp} / {obj}: {int(cnt)}")
        else:
            print("\nNo split/object anomaly cells below 5 samples.")

        # VAE split checks from existing CSV.
        vae_split_df = pd.read_csv(VAE_SPLIT_OUT)
        print(f"\nLoaded {VAE_SPLIT_OUT} ({len(vae_split_df)} rows)")
        print(vae_split_df.groupby("split").size().to_string())
        print("\nVAE split quality checks")
        vae_split_obj = vae_split_df.groupby(["split", "object"]).size()
        print(vae_split_obj.to_string())
        low_support_vae = vae_split_obj[vae_split_obj < 5]
        if len(low_support_vae) > 0:
            print("\nWARNING: low anomaly support (<5) in VAE split/object cells:")
            for (sp, obj), cnt in low_support_vae.items():
                print(f"  {sp} / {obj}: {int(cnt)}")
        else:
            print("\nNo VAE split/object anomaly cells below 5 samples.")
        raise SystemExit(0)

    # Two-stage split gives exact 60/20/20 while preserving class proportions.
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
    CLASSIFIER_SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(CLASSIFIER_SPLIT_OUT, index=False)
    print(f"Wrote {CLASSIFIER_SPLIT_OUT} ({len(split_df)} rows)")
    split_class_counts = split_df.groupby(["split", "binary_label"]).size().unstack(fill_value=0)
    print(split_class_counts.to_string())
    print("\nSplit x class (% within split):")
    print((split_class_counts.div(split_class_counts.sum(axis=1), axis=0) * 100).round(2).to_string())
    print(
        "\nFractions of total:",
        f"train {len(train_df) / n:.3f}, val {len(val_df) / n:.3f}, test {len(test_df) / n:.3f}",
    )
    print("\nSplit quality checks")
    print("By split x object x class:")
    split_obj = (
        split_df.groupby(["split", "object", "binary_label"]).size().unstack(fill_value=0)
    )
    print(split_obj.to_string())
    min_anomaly_per_split_obj = (
        split_df[split_df["binary_label"] == "anomaly"].groupby(["split", "object"]).size()
    )
    low_support = min_anomaly_per_split_obj[min_anomaly_per_split_obj < 5]
    if len(low_support) > 0:
        print("\nWARNING: low anomaly support (<5) in split/object cells:")
        for (sp, obj), cnt in low_support.items():
            print(f"  {sp} / {obj}: {int(cnt)}")
    else:
        print("\nNo split/object anomaly cells below 5 samples.")

    # Mirror classifier split folders for tooling that expects directory-based datasets.
    if CLASSIFIER_BY_SPLIT_ROOT.exists():
        shutil.rmtree(CLASSIFIER_BY_SPLIT_ROOT)
    CLASSIFIER_BY_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for _, row in split_df.iterrows():
        rel = str(row["image"]).replace("\\", "/").lstrip("/")
        src = (data_root / rel).resolve()
        if not src.is_file():
            continue
        dst = CLASSIFIER_BY_SPLIT_ROOT / str(row["split"]) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1
    print(
        f"\nClassifier folder copy: {CLASSIFIER_BY_SPLIT_ROOT}/{{train,val,test}}/… "
        f"({n_copied} files)"
    )

    # VAE split (anomaly-only)
    vae_df = df[df["binary_label"] == "anomaly"].copy()
    print("\nVAE split (anomaly-only train/val/test: 60% / 20% / 20%)")
    n_vae = len(vae_df)
    if n_vae == 0:
        print("No anomaly rows available; skipping VAE split.", file=sys.stderr)
    else:
        vae_train_df, vae_temp_df = train_test_split(
            vae_df,
            test_size=0.4,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        vae_val_df, vae_test_df = train_test_split(
            vae_temp_df,
            test_size=0.5,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        vae_train_df = vae_train_df.copy()
        vae_val_df = vae_val_df.copy()
        vae_test_df = vae_test_df.copy()
        vae_train_df["split"] = "train"
        vae_val_df["split"] = "val"
        vae_test_df["split"] = "test"

        vae_split_df = pd.concat([vae_train_df, vae_val_df, vae_test_df], ignore_index=True)
        VAE_SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
        vae_split_df.to_csv(VAE_SPLIT_OUT, index=False)
        print(f"Wrote {VAE_SPLIT_OUT} ({len(vae_split_df)} rows)")
        print(vae_split_df.groupby("split").size().to_string())
        print(
            "Fractions of total:",
            f"train {len(vae_train_df) / n_vae:.3f}, "
            f"val {len(vae_val_df) / n_vae:.3f}, "
            f"test {len(vae_test_df) / n_vae:.3f}",
        )

        # Same split-quality style checks for VAE rows.
        print("\nVAE split quality checks")
        vae_split_obj = vae_split_df.groupby(["split", "object"]).size()
        print(vae_split_obj.to_string())
        low_support_vae = vae_split_obj[vae_split_obj < 5]
        if len(low_support_vae) > 0:
            print("\nWARNING: low anomaly support (<5) in VAE split/object cells:")
            for (sp, obj), cnt in low_support_vae.items():
                print(f"  {sp} / {obj}: {int(cnt)}")
        else:
            print("\nNo VAE split/object anomaly cells below 5 samples.")

        if VAE_BY_SPLIT_ROOT.exists():
            shutil.rmtree(VAE_BY_SPLIT_ROOT)
        VAE_BY_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
        n_vae_copied = 0
        for _, row in vae_split_df.iterrows():
            rel = str(row["image"]).replace("\\", "/").lstrip("/")
            src = (data_root / rel).resolve()
            if not src.is_file():
                continue
            dst = VAE_BY_SPLIT_ROOT / str(row["split"]) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_vae_copied += 1
        print(
            f"\nVAE folder copy: {VAE_BY_SPLIT_ROOT}/{{train,val,test}}/… "
            f"({n_vae_copied} files)"
        )
