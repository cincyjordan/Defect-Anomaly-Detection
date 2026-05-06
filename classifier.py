from __future__ import annotations

from typing import Any

# Baseline binary classifier: VisA normal vs anomaly.

# Resume + Slurm: if a job hits TIME LIMIT mid-search, rerun with determinism preserved, e.g.:
#   rm -f artifacts/classifier/baseline_trial_007.pt   # optional, if that trial did not finish
#   python classifier.py --start-trial 7
# Ask for enough walltime on GPU (sbatch/srun): e.g. #SBATCH --time=08:00:00 (cluster-specific).

# Purpose: Establish a benchmark before adding VAE-generated synthetic defects. Same
# preprocessing as anomalydetect.py ensures fair comparison later when augmenting train.

# Data: data/processed/classifier/split_assignments.csv (train/val/test rows) produced by
# anomalydetect.py; image paths resolve under data/raw (see choose_data_root). Augmented CSV
# from anomalydetect --build-augmented-classifier adds synthetic train rows whose paths live
# under data/generated/... (resolved by resolve_classifier_image_path).

# Training: Random hyperparameter search (N_TRIALS x EPOCHS per trial). Each trial saves
# best weights by lowest validation BCE loss under artifacts/classifier/. Test metrics
# include PR-AUC / ROC-AUC because accuracy alone is misleading under class imbalance.

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

# Shared crop size and transforms with anomalydetect / VAE so all models see comparable pixels.
from anomalydetect import IMAGE_SIZE, get_eval_transform, get_train_transform

CLASSIFIER_SPLIT_CSV = Path("data/processed/classifier/split_assignments.csv")
RAW_ROOT = Path("data/raw")
DATA_ROOT = Path("data")
CHECKPOINT_DIR = Path("artifacts/classifier")

# Controls repeatability of weight init and the random-search sampler.
SEED = 42
EPOCHS = 20
# DataLoader prefetch workers (0 = load in main process only).
NUM_WORKERS = 2
N_TRIALS = 10

# Random-search space: log-uniform LR; discrete choices for batch size, dropout, augmentation.
LR_LOG10_RANGE = (-4.3, -2.7)  # ~5e-5 to ~2e-3
BATCH_SIZE_CHOICES = [16, 32, 64]
DROPOUT_CHOICES = [0.1, 0.2, 0.3, 0.4]
USE_AUG_CHOICES = [False, True]


def sample_classifier_trial_cfg(rng: random.Random) -> dict:
    """One random-search draw; must stay in sync across runs for reproducible resumes."""
    return {
        "lr": 10 ** rng.uniform(*LR_LOG10_RANGE),
        "batch_size": rng.choice(BATCH_SIZE_CHOICES),
        "dropout": rng.choice(DROPOUT_CHOICES),
        "use_aug": rng.choice(USE_AUG_CHOICES),
    }


def best_overall_from_saved_trials(checkpoint_dir: Path, start_exclusive: int) -> dict[str, Any]:
    """Load best_val_loss (+ path, config if present) from completed checkpoints 000 .. start_exclusive-1."""
    best_overall: dict[str, Any] = {
        "val_loss": float("inf"),
        "trial": None,
        "path": None,
        "config": None,
    }
    for trial_idx in range(start_exclusive):
        path = checkpoint_dir / f"baseline_trial_{trial_idx:03d}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Resume needs prior checkpoint for trial {trial_idx:03d}: {path} "
                f"(finish earlier trials first, or lower --start-trial)."
            )
        saved = torch.load(path, map_location="cpu")
        vloss = float(saved["best_val_loss"])
        cfg = saved.get("config") if isinstance(saved.get("config"), dict) else {}
        candidate = {"val_loss": vloss, "trial": trial_idx, "path": path, "config": cfg or None}
        if vloss < best_overall["val_loss"]:
            best_overall = candidate
    return best_overall


# PyTorch Dataset wrapping one split DataFrame: loads RGB, applies transform, returns label.
class ClassifierImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        # CSV paths are posix-style relative to RAW root; normalize Windows slashes if present.
        rel = str(row["image"]).replace("\\", "/").lstrip("/")
        path = resolve_classifier_image_path(rel, self.data_root)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        # BCEWithLogitsLoss expects float targets 0.0 (normal) or 1.0 (anomaly).
        y = 1.0 if str(row["binary_label"]).strip().lower() == "anomaly" else 0.0
        return x, torch.tensor(y, dtype=torch.float32)


# Lightweight CNN: stride-2 convs downsample, global average pool to a fixed vector, then
# linear logits. Dropout is tuned per trial to reduce overfitting given dataset imbalance.
class BaselineCNN(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        # Shape (N,) raw logits - sigmoid applied only for metrics / interpretation.
        return self.classifier(x).squeeze(1)


# Normalize where VisA JPEGs live: repo may unzip as data/raw/... or data/raw/visa-anomaly-detection/... .
def choose_data_root() -> Path:
    for p in (RAW_ROOT, RAW_ROOT / "visa-anomaly-detection"):
        if p.is_dir():
            return p
    raise FileNotFoundError("Could not locate data root under data/raw")


def resolve_classifier_image_path(rel: str, raw_root: Path) -> Path:
    """VisA-relative paths resolve under raw_root; VAE synth paths use CSV prefix generated/… under data/."""
    rel_clean = str(rel).replace("\\", "/").lstrip("/")
    if rel_clean.startswith("generated/"):
        return (DATA_ROOT / rel_clean).resolve()
    return (raw_root / rel_clean).resolve()


# Train + validate each epoch. Checkpoint criterion: lowest validation loss (BCE).
# Training accuracy is printed for intuition but not used for model selection.
def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    best_path: Path,
) -> float:
    best_val_loss = float("inf")
    # Numerically stable binary classification loss for logits + float {0,1} targets.
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train(True)
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            # Fixed 0.5 probability threshold for binary predictions.
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += int((preds == y).sum().item())
            train_total += int(y.numel())
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        train_acc = train_correct / max(train_total, 1)

        model.train(False)
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += float(loss.item())

                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += int((preds == y).sum().item())
                val_total += int(y.numel())
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        val_acc = val_correct / max(val_total, 1)
        print(
            f"  epoch {epoch:03d} | "
            f"train loss {train_loss:.6f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.6f}, acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Minimal checkpoint here; run() re-saves with full trial config after training ends.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {},
                },
                best_path,
            )

    return best_val_loss


# Held-out evaluation: same BCE as training for a comparable "test loss", plus ranking metrics
# (PR-AUC, ROC-AUC) that reflect rare-positive performance. ROC-AUC is undefined if test has
# only one class (edge case when filtering or tiny splits).
def test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.train(False)

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    test_batches = 0
    y_chunks: list[np.ndarray] = []
    prob_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            test_loss_sum += float(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            test_correct += int((preds == y).sum().item())
            test_total += int(y.numel())
            test_batches += 1

            y_chunks.append(y.detach().float().cpu().numpy())
            prob_chunks.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_chunks).astype(np.int64)
    y_prob = np.concatenate(prob_chunks).astype(np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    # Report metrics focused on imbalance performance.
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    pr_auc = float(average_precision_score(y_true, y_prob))
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))

    # Fix label order so tn/fp/fn/tp always mean normal-vs-anomaly the same way.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "loss": test_loss_sum / max(test_batches, 1),
        "acc": test_correct / max(test_total, 1),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def run(
    start_trial: int = 0,
    end_trial_exclusive: int | None = None,
    split_assignments: Path | None = None,
) -> None:
    # End-to-end: load splits -> for each sampled hyperparam set, train with early-like
    # selection via best val loss checkpoint -> evaluate best weights on test -> track which
    # trial minimized validation loss across the whole search.
    global_end = N_TRIALS if end_trial_exclusive is None else end_trial_exclusive
    if start_trial < 0 or global_end > N_TRIALS or start_trial >= global_end:
        raise ValueError(
            f"Need 0 <= start_trial < end_trial_exclusive <= N_TRIALS ({N_TRIALS}); "
            f"got start_trial={start_trial}, end_trial_exclusive={global_end}"
        )

    split_csv = split_assignments if split_assignments is not None else CLASSIFIER_SPLIT_CSV
    if not split_csv.is_file():
        raise FileNotFoundError(f"Missing classifier split CSV: {split_csv}")

    frame = pd.read_csv(split_csv)
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Classifier train/val/test splits are empty.")

    data_root = choose_data_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for _ in range(start_trial):
        sample_classifier_trial_cfg(rng)

    if start_trial == 0:
        best_overall: dict[str, Any] = {
            "val_loss": float("inf"),
            "trial": None,
            "path": None,
            "config": None,
        }
    else:
        best_overall = best_overall_from_saved_trials(CHECKPOINT_DIR, start_trial)

    print(
        f"Baseline classifier random search: trials [{start_trial:03d}, {global_end:03d}), "
        f"full search has N_TRIALS={N_TRIALS}, epochs={EPOCHS}, device={device}\n"
        f"split CSV: {split_csv}"
    )

    for trial_idx in range(start_trial, global_end):
        cfg = sample_classifier_trial_cfg(rng)
        print(f"\nTrial {trial_idx:03d}")
        print(
            f"cfg: lr={cfg['lr']:.6g}, batch_size={cfg['batch_size']}, "
            f"dropout={cfg['dropout']:.2f}, use_aug={cfg['use_aug']}"
        )

        # Train may use jitter/flip/etc.; val/test always deterministic eval preprocessing.
        train_ds = ClassifierImageDataset(train_df, data_root, get_train_transform(cfg["use_aug"]))
        val_ds = ClassifierImageDataset(val_df, data_root, get_eval_transform())
        test_ds = ClassifierImageDataset(test_df, data_root, get_eval_transform())
        # pin_memory speeds host -> GPU copies when CUDA is available.
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        model = BaselineCNN(dropout=cfg["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        best_path = CHECKPOINT_DIR / f"baseline_trial_{trial_idx:03d}.pt"
        best_val_loss = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=EPOCHS,
            best_path=best_path,
        )

        saved = torch.load(best_path, map_location="cpu")
        saved["config"] = {
            **cfg,
            "seed": SEED,
            "epochs": EPOCHS,
            "num_workers": NUM_WORKERS,
            "trial_index": trial_idx,
        }
        torch.save(saved, best_path)

        # Reload best val checkpoint into a fresh module so test reflects saved weights, not
        # whatever the last training epoch left in memory.
        best_model = BaselineCNN(dropout=cfg["dropout"]).to(device)
        best_model.load_state_dict(saved["model_state_dict"])
        test_stats = test(best_model, test_loader, device)

        print(f"best val loss (trial {trial_idx:03d}): {best_val_loss:.6f}")
        print(
            f"test loss {test_stats['loss']:.6f}, acc {test_stats['acc']:.4f} | "
            f"precision {test_stats['precision']:.4f}, recall {test_stats['recall']:.4f}, "
            f"f1 {test_stats['f1']:.4f} | PR-AUC {test_stats['pr_auc']:.4f}",
            end="",
        )
        if np.isfinite(test_stats["roc_auc"]):
            print(f", ROC-AUC {test_stats['roc_auc']:.4f}")
        else:
            print(" (ROC-AUC n/a: single class in test)")
        print(
            f"confusion (tn fp / fn tp): {test_stats['tn']} {test_stats['fp']} / "
            f"{test_stats['fn']} {test_stats['tp']}"
        )
        print(f"checkpoint: {best_path}")

        if best_val_loss < float(best_overall["val_loss"]):
            best_overall = {"val_loss": best_val_loss, "trial": trial_idx, "path": best_path, "config": cfg}

    print("\nRandom search complete.")
    print(f"Best trial: {best_overall['trial']}")
    print(f"Best val loss: {best_overall['val_loss']:.6f}")
    print(f"Best checkpoint: {best_overall['path']}")
    print(f"Best config: {best_overall['config']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline classifier with random hyperparameter search.",
    )
    parser.add_argument(
        "--start-trial",
        type=int,
        default=0,
        help="First trial index to run [0, N_TRIALS). RNG is advanced as if trials before this ran.",
    )
    parser.add_argument(
        "--end-trial-exclusive",
        type=int,
        default=None,
        help=f"Exclusive end trial index (default: {N_TRIALS}). Example: "
        "`--start-trial 7` runs trials 007..009 when N_TRIALS=10.",
    )
    parser.add_argument(
        "--split-assignments",
        type=Path,
        default=CLASSIFIER_SPLIT_CSV,
        help="Classifier split CSV (default: baseline). Use data/processed/classifier_augmented/"
        "split_assignments.csv for VAE-augmented train set.",
    )
    args = parser.parse_args()
    run(
        start_trial=args.start_trial,
        end_trial_exclusive=args.end_trial_exclusive,
        split_assignments=args.split_assignments,
    )


if __name__ == "__main__":
    main()
