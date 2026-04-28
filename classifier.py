from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from anomalydetect import IMAGE_SIZE, get_eval_transform, get_train_transform

CLASSIFIER_SPLIT_CSV = Path("data/processed/classifier/split_assignments.csv")
RAW_ROOT = Path("data/raw")
CHECKPOINT_DIR = Path("artifacts/classifier")

SEED = 42
EPOCHS = 20
NUM_WORKERS = 2
N_TRIALS = 10

LR_LOG10_RANGE = (-4.3, -2.7)  # ~5e-5 to ~2e-3
BATCH_SIZE_CHOICES = [16, 32, 64]
DROPOUT_CHOICES = [0.1, 0.2, 0.3, 0.4]
USE_AUG_CHOICES = [False, True]


class ClassifierImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        rel = str(row["image"]).replace("\\", "/").lstrip("/")
        path = self.data_root / rel
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = 1.0 if str(row["binary_label"]).strip().lower() == "anomaly" else 0.0
        return x, torch.tensor(y, dtype=torch.float32)


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
        return self.classifier(x).squeeze(1)


def choose_data_root() -> Path:
    for p in (RAW_ROOT, RAW_ROOT / "visa-anomaly-detection"):
        if p.is_dir():
            return p
    raise FileNotFoundError("Could not locate data root under data/raw")


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    best_path: Path,
) -> float:
    best_val_loss = float("inf")
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": {},
                },
                best_path,
            )

    return best_val_loss


def test_loop(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.train(False)

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    test_batches = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            test_loss_sum += float(loss.item())

            preds = (torch.sigmoid(logits) >= 0.5).float()
            test_correct += int((preds == y).sum().item())
            test_total += int(y.numel())
            test_batches += 1

    return {
        "loss": test_loss_sum / max(test_batches, 1),
        "acc": test_correct / max(test_total, 1),
    }


def run():
    if not CLASSIFIER_SPLIT_CSV.is_file():
        raise FileNotFoundError(f"Missing classifier split CSV: {CLASSIFIER_SPLIT_CSV}")

    frame = pd.read_csv(CLASSIFIER_SPLIT_CSV)
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
    best_overall = {"val_loss": float("inf"), "trial": None, "path": None, "config": None}
    print(f"Baseline classifier random search: trials={N_TRIALS}, epochs={EPOCHS}, device={device}")

    for trial_idx in range(N_TRIALS):
        cfg = {
            "lr": 10 ** rng.uniform(*LR_LOG10_RANGE),
            "batch_size": rng.choice(BATCH_SIZE_CHOICES),
            "dropout": rng.choice(DROPOUT_CHOICES),
            "use_aug": rng.choice(USE_AUG_CHOICES),
        }
        print(f"\nTrial {trial_idx:03d}")
        print(
            f"cfg: lr={cfg['lr']:.6g}, batch_size={cfg['batch_size']}, "
            f"dropout={cfg['dropout']:.2f}, use_aug={cfg['use_aug']}"
        )

        train_ds = ClassifierImageDataset(train_df, data_root, get_train_transform(cfg["use_aug"]))
        val_ds = ClassifierImageDataset(val_df, data_root, get_eval_transform())
        test_ds = ClassifierImageDataset(test_df, data_root, get_eval_transform())
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
        best_val_loss = train_loop(
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

        best_model = BaselineCNN(dropout=cfg["dropout"]).to(device)
        best_model.load_state_dict(saved["model_state_dict"])
        test_stats = test_loop(best_model, test_loader, device)

        print(f"best val loss (trial {trial_idx:03d}): {best_val_loss:.6f}")
        print(f"test loss {test_stats['loss']:.6f}, test acc {test_stats['acc']:.4f}")
        print(f"checkpoint: {best_path}")

        if best_val_loss < best_overall["val_loss"]:
            best_overall = {"val_loss": best_val_loss, "trial": trial_idx, "path": best_path, "config": cfg}

    print("\nRandom search complete.")
    print(f"Best trial: {best_overall['trial']}")
    print(f"Best val loss: {best_overall['val_loss']:.6f}")
    print(f"Best checkpoint: {best_overall['path']}")
    print(f"Best config: {best_overall['config']}")


def main():
    run()


if __name__ == "__main__":
    main()
