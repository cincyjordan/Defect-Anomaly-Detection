from __future__ import annotations

# ConvVAE trainer for VisA *defect-only* images.

# The downstream classifier will be trained on normal + real defect images (and later
# synthetic defects). This script models the distribution of real defects only, so the VAE
# can later sample new defect-looking images for minority-class augmentation.

# Data layout: anomalydetect.py writes data/processed/vae/split_assignments.csv and
# mirrored folders; this file reads that CSV. Image paths are relative to data/raw (or
# data/raw/visa-anomaly-detection - see choose_data_root).
# Training: Random search over hyperparameters (N_TRIALS). Each trial trains for EPOCHS,
# checkpoints the best validation reconstruction MSE per trial under artifacts/vae/.
# Optional test split is evaluated once per trial using the best checkpoint weights.
# Preconditions: preprocessing (resize/normalize/aug) matches anomalydetect.get_*_transform
# so VAE inputs stay consistent with the rest of the project.

import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from anomalydetect import IMAGE_SIZE, get_eval_transform, get_train_transform

VAE_SPLIT_CSV = Path("data/processed/vae/split_assignments.csv")
RAW_ROOT = Path("data/raw")
CHECKPOINT_DIR = Path("artifacts/vae")

# Reproducibility for sampled hyperparameters and PyTorch RNG.
SEED = 42
EPOCHS = 25
NUM_WORKERS = 2
N_TRIALS = 12

# Random-search space: each trial draws one config. Log-uniform for lr/beta is standard
# because good values often span orders of magnitude.
LR_LOG10_RANGE = (-4.3, -2.5)  # ~5e-5 to ~3e-3
BATCH_SIZE_CHOICES = [16, 32, 64]
LATENT_DIM_CHOICES = [32, 64, 128, 256]
BETA_LOG10_RANGE = (-1.0, 0.6)  # ~0.1 to ~4
USE_AUG_CHOICES = [False, True]


# Dataset: rows come from split CSV (image column). We resolve paths against data_root
# because the repo may live on another machine - only relative paths are portable.
class AnomalyImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        rel = str(self.frame.iloc[idx]["image"]).replace("\\", "/").lstrip("/")
        path = self.data_root / rel
        img = Image.open(path)
        x = self.transform(img)
        # Return rel for debugging only; training loops ignore the string.
        return x, rel


# Conv VAE: encoder downsamples by 2^4 = 16; decoder upsamples symmetrically. Latent is
# diagonal Gaussian (mu, logvar); reparameterization enables backprop through sampling.
# Decoder ends in Sigmoid so outputs are bounded (loss is still MSE vs the transformed target x).
class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        h, w = IMAGE_SIZE[0] // 16, IMAGE_SIZE[1] // 16
        self._enc_feat_shape = (256, h, w)
        feat_dim = 256 * h * w
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, feat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps, sigma = exp(0.5 * logvar); eps ~ N(0,I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(z.shape[0], *self._enc_feat_shape)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# Kaggle extract sometimes uses data/raw or data/raw/visa-anomaly-detection/.
def choose_data_root() -> Path:
    for p in (RAW_ROOT, RAW_ROOT / "visa-anomaly-detection"):
        if p.is_dir():
            return p
    raise FileNotFoundError("Could not locate data root under data/raw")


# One epoch = full train pass + full val pass. Objective: reconstruction MSE + beta * KL.
# beta trades off reconstruction fidelity vs a standard-normal latent prior (beta-VAE style).
def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    beta: float,
    epochs: int,
    best_path: Path,
) -> float:
    # Checkpoint on val reconstruction loss (not combined loss) because we mainly care how well images
    # reconstruct; KL can dominate the summed loss depending on beta.
    best_val_recon = float("inf")

    for epoch in range(1, epochs + 1):
        model.train(True)
        train_total_loss = 0.0
        train_total_recon = 0.0
        train_total_kld = 0.0
        train_batches = 0
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            # Same-scale MSE between input batch x and reconstruction (both model tensors).
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            # Analytic KL for q(z|x) vs N(0,I) when q is diagonal Gaussian.
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kld

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_total_loss += float(loss.item())
            train_total_recon += float(recon_loss.item())
            train_total_kld += float(kld.item())
            train_batches += 1

        train_stats = {
            "loss": train_total_loss / max(train_batches, 1),
            "recon": train_total_recon / max(train_batches, 1),
            "kld": train_total_kld / max(train_batches, 1),
        }

        model.train(False)
        # Validation: no dropout / no grad; same loss terms as train for logging.
        val_total_loss = 0.0
        val_total_recon = 0.0
        val_total_kld = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                recon_loss = F.mse_loss(recon, x, reduction="mean")
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kld

                val_total_loss += float(loss.item())
                val_total_recon += float(recon_loss.item())
                val_total_kld += float(kld.item())
                val_batches += 1

        val_stats = {
            "loss": val_total_loss / max(val_batches, 1),
            "recon": val_total_recon / max(val_batches, 1),
            "kld": val_total_kld / max(val_batches, 1),
        }

        print(
            f"  epoch {epoch:03d} | "
            f"train loss {train_stats['loss']:.6f} (recon {train_stats['recon']:.6f}, kld {train_stats['kld']:.6f}) | "
            f"val loss {val_stats['loss']:.6f} (recon {val_stats['recon']:.6f}, kld {val_stats['kld']:.6f})"
        )

        if val_stats["recon"] < best_val_recon:
            best_val_recon = val_stats["recon"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    # Filled after training in run(); placeholder keeps checkpoint shape stable.
                    "config": {},
                    "best_val_recon": best_val_recon,
                },
                best_path,
            )

    return best_val_recon


# Held-out metrics only; no gradients. Uses same beta as training trial for comparable loss scale.
def test(model: nn.Module, test_loader: DataLoader, device: torch.device, beta: float) -> dict[str, float]:
    model.train(False)
    test_total_loss = 0.0
    test_total_recon = 0.0
    test_total_kld = 0.0
    test_batches = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kld
            test_total_loss += float(loss.item())
            test_total_recon += float(recon_loss.item())
            test_total_kld += float(kld.item())
            test_batches += 1
    return {
        "loss": test_total_loss / max(test_batches, 1),
        "recon": test_total_recon / max(test_batches, 1),
        "kld": test_total_kld / max(test_batches, 1),
    }


def run():
    # Load splits produced by anomalydetect (VAE-specific stratified split on anomalies).
    if not VAE_SPLIT_CSV.is_file():
        raise FileNotFoundError(f"Missing VAE split CSV: {VAE_SPLIT_CSV}")

    frame = pd.read_csv(VAE_SPLIT_CSV)
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()
    if train_df.empty or val_df.empty:
        raise RuntimeError("VAE train/val splits are empty.")


    data_root = choose_data_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    best_overall = {"val_recon": float("inf"), "trial": None, "path": None, "config": None}
    print(f"Random search: trials={N_TRIALS}, epochs={EPOCHS}, device={device}")

    for trial_idx in range(N_TRIALS):
        # Fresh model + optimizer per trial; no weight carry-over between random configs.
        cfg = {
            "lr": 10 ** rng.uniform(*LR_LOG10_RANGE),
            "batch_size": rng.choice(BATCH_SIZE_CHOICES),
            "latent_dim": rng.choice(LATENT_DIM_CHOICES),
            "beta": 10 ** rng.uniform(*BETA_LOG10_RANGE),
            "use_aug": rng.choice(USE_AUG_CHOICES),
        }
        print(f"\nTrial {trial_idx:03d}")
        print(
            f"cfg: lr={cfg['lr']:.6g}, batch_size={cfg['batch_size']}, "
            f"latent_dim={cfg['latent_dim']}, beta={cfg['beta']:.4f}, use_aug={cfg['use_aug']}"
        )

        # Train: optional mild aug; val/test: fixed transform so metrics are comparable.
        train_ds = AnomalyImageDataset(train_df, data_root, get_train_transform(cfg["use_aug"]))
        val_ds = AnomalyImageDataset(val_df, data_root, get_eval_transform())
        test_ds = AnomalyImageDataset(test_df, data_root, get_eval_transform()) if not test_df.empty else None
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),  # speeds host->device copies when CUDA is used
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = (
            DataLoader(
                test_ds,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
            if test_ds is not None
            else None
        )

        model = ConvVAE(latent_dim=cfg["latent_dim"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        best_path = CHECKPOINT_DIR / f"vae_trial_{trial_idx:03d}.pt"
        best_val_recon = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            beta=cfg["beta"],
            epochs=EPOCHS,
            best_path=best_path,
        )

        saved = torch.load(best_path, map_location="cpu")
        # Persist full trial metadata next to best weights so you know how each checkpoint was trained.
        saved["config"] = {**cfg, "seed": SEED, "epochs": EPOCHS, "num_workers": NUM_WORKERS, "trial_index": trial_idx}
        torch.save(saved, best_path)

        print(f"best val recon (trial {trial_idx:03d}): {best_val_recon:.6f}")
        print(f"checkpoint: {best_path}")

        if best_val_recon < best_overall["val_recon"]:
            best_overall = {"val_recon": best_val_recon, "trial": trial_idx, "path": best_path, "config": cfg}

        if test_loader is not None:
            # Rebuild a clean module and load best weights (optimizer state is not needed for eval).
            best_model = ConvVAE(latent_dim=cfg["latent_dim"]).to(device)
            best_model.load_state_dict(saved["model_state_dict"])
            test_stats = test(best_model, test_loader, device, cfg["beta"])
            print(
                f"test loss {test_stats['loss']:.6f} "
                f"(recon {test_stats['recon']:.6f}, kld {test_stats['kld']:.6f})"
            )
        else:
            print("No test split found; skipped test loop.")

    print("\nRandom search complete.")
    print(f"Best trial: {best_overall['trial']}")
    print(f"Best val recon loss: {best_overall['val_recon']:.6f}")
    print(f"Best checkpoint: {best_overall['path']}")
    print(f"Best config: {best_overall['config']}")


if __name__ == "__main__":
    run()
