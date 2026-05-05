from __future__ import annotations

# ConvVAE trainer for VisA *defect-only* images.

# The downstream classifier will be trained on normal + real defect images (and later
# synthetic defects). This script models the distribution of real defects only, so the VAE
# can later sample new defect-looking images for minority-class augmentation.

# Data layout: anomalydetect.py writes data/processed/vae/split_assignments.csv and
# mirrored folders; this file reads that CSV. Image paths are relative to data/raw (or
# data/raw/visa-anomaly-detection - see choose_data_root).
# Training: each VisA object (category) trains its own VAE only on anomalies for that
# object - avoids one global manifold averaging unrelated products together.
# Random search (N_TRIALS) runs per object; checkpoints land under artifacts/vae/<object>/.
# Per-trial curves: artifacts/vae/plots/<object>/…
#
# After all objects finish, random decoder samples from each object's best trial are saved
# under data/generated/vae_defects/ as <object>_best_sample_*.png (no-post-samples to skip).
# Optional test split is evaluated once per trial using the best checkpoint weights.
# Preconditions: preprocessing (resize/normalize/aug) matches anomalydetect.get_*_transform
# so VAE inputs stay consistent with the rest of the project.

import random
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from anomalydetect import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, get_eval_transform, get_train_transform

VAE_SPLIT_CSV = Path("data/processed/vae/split_assignments.csv")
RAW_ROOT = Path("data/raw")
CHECKPOINT_DIR = Path("artifacts/vae")
# Post-training decoder samples per object best checkpoint (for inspection + classifier CSV).
SYNTH_IMAGE_DIR_DEFAULT = Path("data/generated/vae_defects")
POST_TRAIN_NUM_SAMPLES = 10

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


def denormalized_to_rgb01(batch: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet-style normalize used across this project (see anomalydetect transforms)."""
    mean = batch.new_tensor(NORMALIZE_MEAN).view(1, -1, 1, 1)
    std = batch.new_tensor(NORMALIZE_STD).view(1, -1, 1, 1)
    return torch.clamp(batch * std + mean, 0.0, 1.0)


def save_decoder_random_pngs(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    num: int,
    seed: int,
    stem_prefix: str,
    device: torch.device | None = None,
) -> None:
    """Load best checkpoint, decode num random priors z ~ N(0,I), save PNGs under output_dir."""
    if num <= 0:
        return
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for sampling: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload.get("config") or {}
    latent_dim = int(cfg.get("latent_dim", 128))
    if latent_dim <= 0:
        raise ValueError("Checkpoint missing valid config['latent_dim'].")

    torch.manual_seed(int(seed))

    model = ConvVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem_prefix_safe = stem_prefix.replace(".", "_")
    with torch.no_grad():
        for i in range(num):
            z = torch.randn(1, latent_dim, device=device)
            recon = model.decode(z)
            rgb = denormalized_to_rgb01(recon)
            out_path = output_dir / f"{stem_prefix_safe}_sample_{i:03d}.png"
            save_image(rgb, out_path)


def effective_batch_size(requested_bs: int, dataset_len: int) -> int:
    return max(1, min(requested_bs, dataset_len))


def save_vae_trial_curves(
    history: dict[str, list[float]],
    out_path: Path,
    trial_idx: int,
    beta: float,
    object_label: str | None = None,
) -> None:
    """Write train vs val recon / KL / total loss for one random-search trial (headless-safe)."""
    epochs = history["epoch"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(epochs, history["train_recon"], label="train", lw=2)
    axes[0].plot(epochs, history["val_recon"], label="val", lw=2)
    axes[0].set_ylabel("recon MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_kld"], label="train", lw=2)
    axes[1].plot(epochs, history["val_kld"], label="val", lw=2)
    axes[1].set_ylabel("KL (q(z|x) vs N(0,I))")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_loss"], label="train", lw=2)
    axes[2].plot(epochs, history["val_loss"], label="val", lw=2)
    axes[2].set_ylabel("total loss\n(recon + beta*KL)")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    pref = f"{object_label} — " if object_label else ""
    fig.suptitle(f"{pref}VAE trial {trial_idx:03d} (beta={beta:.4g})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
) -> tuple[float, dict[str, list[float]]]:
    # Checkpoint on val reconstruction loss (not combined loss) because we mainly care how well images
    # reconstruct; KL can dominate the summed loss depending on beta.
    best_val_recon = float("inf")
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_recon": [],
        "train_kld": [],
        "val_loss": [],
        "val_recon": [],
        "val_kld": [],
    }

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

        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_stats["loss"])
        history["train_recon"].append(train_stats["recon"])
        history["train_kld"].append(train_stats["kld"])
        history["val_loss"].append(val_stats["loss"])
        history["val_recon"].append(val_stats["recon"])
        history["val_kld"].append(val_stats["kld"])

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

    return best_val_recon, history


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


def run(
    *,
    synth_image_dir: Path = SYNTH_IMAGE_DIR_DEFAULT,
    post_train_num_samples: int = POST_TRAIN_NUM_SAMPLES,
) -> None:
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
    plots_dir = CHECKPOINT_DIR / "plots"
    rng = random.Random(SEED)
    objects_sorted = sorted(str(x) for x in train_df["object"].dropna().unique())

    print(
        f"Per-object random search: objects={len(objects_sorted)}, trials={N_TRIALS}, "
        f"epochs={EPOCHS}, device={device}"
    )

    per_object_best: list[dict[str, object]] = []

    for obj in objects_sorted:
        train_o = train_df[train_df["object"].astype(str) == obj].copy()
        val_o = val_df[val_df["object"].astype(str) == obj].copy()
        test_o = test_df[test_df["object"].astype(str) == obj].copy()

        if train_o.empty:
            print(f"\nSkipping object={obj}: no train anomalies.")
            continue
        if val_o.empty:
            print(f"\nSkipping object={obj}: no val anomalies (needed for reconstruction checkpoint).")
            continue

        obj_ckpt_root = CHECKPOINT_DIR / obj
        obj_ckpt_root.mkdir(parents=True, exist_ok=True)
        plots_o = plots_dir / obj

        print(f"\n======== Object: {obj} | train {len(train_o)} | val {len(val_o)} | test {len(test_o)} ==========")

        best_for_obj = {
            "val_recon": float("inf"),
            "trial": None,
            "path": None,
            "config": None,
        }

        for trial_idx in range(N_TRIALS):
            cfg = {
                "lr": 10 ** rng.uniform(*LR_LOG10_RANGE),
                "batch_size": rng.choice(BATCH_SIZE_CHOICES),
                "latent_dim": rng.choice(LATENT_DIM_CHOICES),
                "beta": 10 ** rng.uniform(*BETA_LOG10_RANGE),
                "use_aug": rng.choice(USE_AUG_CHOICES),
            }
            bs_train = effective_batch_size(cfg["batch_size"], len(train_o))
            bs_val = effective_batch_size(cfg["batch_size"], len(val_o))
            bs_test = effective_batch_size(cfg["batch_size"], len(test_o))

            print(f"\n  Trial {trial_idx:03d} ({obj})")
            print(
                f"  cfg: lr={cfg['lr']:.6g}, batch_size(train/val/test)={bs_train}/{bs_val}/{bs_test}, "
                f"latent_dim={cfg['latent_dim']}, beta={cfg['beta']:.4f}, use_aug={cfg['use_aug']}"
            )

            train_ds = AnomalyImageDataset(train_o, data_root, get_train_transform(cfg["use_aug"]))
            val_ds = AnomalyImageDataset(val_o, data_root, get_eval_transform())
            test_ds = AnomalyImageDataset(test_o, data_root, get_eval_transform()) if not test_o.empty else None
            train_loader = DataLoader(
                train_ds,
                batch_size=bs_train,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=bs_val,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
            test_loader = (
                DataLoader(
                    test_ds,
                    batch_size=bs_test,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=torch.cuda.is_available(),
                )
                if test_ds is not None
                else None
            )

            model = ConvVAE(latent_dim=cfg["latent_dim"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            best_path = obj_ckpt_root / f"vae_trial_{trial_idx:03d}.pt"
            best_val_recon, history = train(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                beta=cfg["beta"],
                epochs=EPOCHS,
                best_path=best_path,
            )

            curve_path = plots_o / f"vae_trial_{trial_idx:03d}.png"
            save_vae_trial_curves(history, curve_path, trial_idx, cfg["beta"], object_label=obj)

            saved = torch.load(best_path, map_location="cpu")
            saved["config"] = {
                **cfg,
                "object": obj,
                "seed": SEED,
                "epochs": EPOCHS,
                "num_workers": NUM_WORKERS,
                "trial_index": trial_idx,
            }
            torch.save(saved, best_path)

            print(f"  best val recon: {best_val_recon:.6f}")
            print(f"  checkpoint: {best_path}")
            print(f"  training curves: {curve_path}")

            if best_val_recon < float(best_for_obj["val_recon"]):
                best_for_obj = {
                    "val_recon": best_val_recon,
                    "trial": trial_idx,
                    "path": best_path,
                    "config": cfg.copy(),
                }

            if test_loader is not None:
                best_model = ConvVAE(latent_dim=cfg["latent_dim"]).to(device)
                best_model.load_state_dict(saved["model_state_dict"])
                test_stats = test(best_model, test_loader, device, cfg["beta"])
                print(
                    f"  test loss {test_stats['loss']:.6f} "
                    f"(recon {test_stats['recon']:.6f}, kld {test_stats['kld']:.6f})"
                )
            else:
                print("  No test split for this object; skipped test.")

        print(
            f"\n>>> {obj} BEST : trial={best_for_obj['trial']} | "
            f"val recon={best_for_obj['val_recon']:.6f} | checkpoint={best_for_obj['path']}"
        )

        per_object_best.append(best_for_obj | {"object": obj})

    print("\nPer-object random search complete.")
    for row in per_object_best:
        print(
            f"  object={row['object']!r} | best_trial={row['trial']} | "
            f"val_recon={row['val_recon']:.6f} | {row['path']}"
        )

    if post_train_num_samples > 0 and per_object_best:
        synth_image_dir = synth_image_dir.resolve()
        synth_image_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\nSaving decoder samples ({post_train_num_samples} each) under {synth_image_dir.resolve()} ..."
        )
        for row in per_object_best:
            if row["path"] is None:
                continue
            obj = str(row["object"])
            obj_seed = (SEED + zlib.adler32(obj.encode())) & 0xFFFFFFFF
            save_decoder_random_pngs(
                Path(row["path"]),
                synth_image_dir,
                num=post_train_num_samples,
                seed=obj_seed,
                stem_prefix=f"{obj}_best",
                device=device,
            )
        print("Done.")

    elif post_train_num_samples <= 0:
        print("\nSkipping post-training decoder PNGs (--no-post-samples).")


if __name__ == "__main__":
    import sys

    n_post = POST_TRAIN_NUM_SAMPLES
    if "--no-post-samples" in sys.argv:
        n_post = 0
    run(post_train_num_samples=n_post)
