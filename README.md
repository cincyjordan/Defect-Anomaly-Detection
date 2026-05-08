## **Name:** Generating industrial defects for anomaly detection (VisA).

**Overview:** Combat class imbalance (~9.6k normal vs ~1.2k defect images globally) so a downstream CNN does not plateau on nominal accuracy while missing defects. A convolutional VAE trains on defect crops only, per VisA category, and generated (and CutPaste) anomalies augment minority-class classifier training alongside real labels.

## **Extra Criteria:**

**Metrics tracking:** visually compare loss vs reconstruction runs Epoch curves as: VAE reconstruction/KL/total grids; classifier baseline vs augmented BCE & accuracy panels. Static plots substituted for weights-and-biases dashboards for reproducibility. 

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/classifier_baseline_vs_augmented_loss_grid.png' width = '800'>

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/classifier_histogram_grid.png' width = '800'>

**Classifier compare:** Classifier loss/accuracy (baseline vs. synthetic augmentation)
The baseline (left) trains steadily but slowly, with train and val tracking closely, low overfitting but modest gains. The augmented version (right) learns faster and reaches higher accuracy (~92-93% train), but there's a widening gap between train and val loss, suggesting the synthetic data helps but introduces some distribution mismatch.


**Hyperparameter tuning:** LR, latent size, batch search Random search in `**model_vae.py`** (latent dim, beta, LR, augmentation) and `**classifier.py**` (LR, batch, dropout); best runs selected on val reconstruction / val BCE respectively. Mosaic VAE curves show cross-category variability. 

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/vae_outputs_one_sample_per_class.png' width = '800'>

**VAE decoder samples results:**
The reconstructions are recognizable but blurry, typical of VAEs. Simple-shaped objects (candle, cashew, macaroni) reconstruct more crisply. More complex PCB boards show more blur and loss of fine detail, which makes sense given the higher spatial complexity. Overall the VAE has captured class-level structure successfully.

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/vae_best_loss_grid.png' width = '800'>

All classes converge cleanly within ~5-10 epochs. Reconstruction loss and total loss drop sharply then flatten. KL loss is very small across the board. Train and val curves stay close together, indicating no significant overfitting, the VAEs trained well across all 12 classes.

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/vae_histogram_grid.png' width = '800'>

**Candle Weights/Biases**
Parameters are tightly zero-centered with very small sigmoid values throughout, a sign of a well-regularized model. The fc_logvar layers are especially tight, which is expected in a VAE bottleneck. No dead neurons or saturated weights are evident.

## **Evaluation Metrics:**

### **Classifier test summary (baseline trial)**

### Metrics table

| Metric | Value |
|---|---:|
| Accuracy | 0.8937 |
| Precision | 0.6471 |
| Recall | 0.0917 |
| F1 score | 0.1606 |
| PR-AUC | 0.3575 |
| ROC-AUC | 0.7804 |

### Confusion matrix

|  | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| **Actual Normal** | 1912 (TN) | 12 (FP) |
| **Actual Anomaly** | 218 (FN) | 22 (TP) |

**Interpretation:** the model keeps false positives low (12), but misses many true anomalies (218 FN), which explains high overall accuracy yet low recall on the defect class.

## **Classifier with synthetic data summary**

### Metrics table

| Metric | Value |
|---|---:|
| Accuracy | 0.8965 |
| Precision | 0.7857 |
| Recall | 0.0917 |
| F1 score | 0.1642 |
| PR-AUC | 0.3786 |
| ROC-AUC | 0.7764 |

### Confusion matrix

|  | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| **Actual Normal** | 1918 (TN) | 6 (FP) |
| **Actual Anomaly** | 218 (FN) | 22 (TP) |

**Interpretation:** synthetic-data training further reduces false positives (12 -> 6) and improves precision/PR-AUC, but recall remains unchanged because true positives stay at 22 while false negatives remain high (218).

## **Latent space exploration:** t‑SNE, interpolations | **`latent_viz.py`**

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/latent_viz_grid.png' width = '800'>

**`latent_viz_grid`**: Most classes show reasonable mixing of train/val/test splits in latent space, meaning the VAE generalizes well. A few classes (pcb2, pcb3, pipe_fryum) show extreme axis scales with outliers far from the main cluster, likely a few anomalous samples or numerical instability in the latent encoding for those categories.

**Gallery GUI:** browse / sample latent defects **`app.py`** (Gradio).

## **Difficulties & fixes**

**Problem:** Per-class anomalies are few relative to normals once stratified; the VAE and classifier risk under-covering textured failure modes, hurting recall-style behavior.

**Mitigation:** Per-object defect-only VAE so manifolds are not averaged across unrelated products. `cutpaste_augment.py`: composites real anomaly patches (mask-guided) onto normal backs to grow cheap, label-consistent defects during training-heavy phases without poisoning held-out splits. CutPaste-fed paths plug into augmentation flows noted in `**README`** Branch B alongside decoder samples.

 


### Step 1: Environment

1. Use Python 3.10+ with a fresh virtualenv (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **`requirements.txt` pins CUDA 12.8 PyTorch.** If you are on CPU-only Linux/macOS or a different CUDA version, reinstall `torch` and `torchvision` from https://pytorch.org for your setup, then leave the rest of the requirements as-is.

### Step 2: Download VisA into `data/raw`

1. On Kaggle: Account -> API -> Create New Token.
2. Create a file named `.env` in the project root with:

   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```

   `load_data.py` reads `.env` and writes `kaggle.json`; you should not paste keys into scripts.

3. Download and unpack the dataset:

   ```bash
   python load_data.py
   ```

   Expected layout: JPEGs plus `image_anno.csv` under `data/raw/visa-anomaly-detection/` (or an equivalent subtree that passes the script checks).

### Step 3: Build stratified splits (required for everything downstream)

This cleans bad images if needed and writes CSVs plus mirrored folders for the classifier train/val/test and VAE anomaly-only splits.

```bash
python anomalydetect.py
```

Key outputs:

- `data/processed/classifier/split_assignments.csv` - normal vs anomaly, all splits.
- `data/processed/vae/split_assignments.csv` - anomaly rows stratified per VisA **object**.
- Copies under `data/processed/*/by_split/...`.

Re-run **`anomalydetect.py`** whenever you regenerate synthetic images that should appear in classifier training (below).

---

### Branch A: Baseline classifier only (no VAE)

Use this branch if you only want normal-vs-anomaly performance on real data.

```bash
python classifier.py
```

Writes checkpoints under `artifacts/classifier/` and per-trial curve PNGs (see script header for resume flags `--start-trial` / `--end-trial-exclusive`).

---

### Branch B: Full pipeline (VAE -> synthetic defects -> augmented classifier)

#### Step B1: CutPaste synthetic defects

If you maintain CutPaste outputs under `data/generated/cutpaste_defects/`, **`model_vae.py` can pick them up** as extra anomaly training rows.

```bash
python cutpaste_augment.py
```

(See `--help` in that file for knobs; layouts must match paths the VAE CSV expects.)

#### Step B2: Train per-object VAEs

Long random search across objects; checkpoints per class:

```bash
python model_vae.py
```

Add `python model_vae.py --no-post-samples` on slow machines if you want to skip the final PNG dump into `data/generated/vae_defects/`.

Outputs include:

- `artifacts/vae/<object>/vae_trial_*.pt`
- Loss curves under `artifacts/vae/plots/<object>/`.

#### Step B3: Extra VAE decoding samples (optional)

If you need more PNGs under `data/generated/vae_defects/`:

```bash
python sample_vae.py --all-classes
```

(or point `--checkpoint` at a single `.pt`).

#### Step B4: Rebuild classifier augmented split CSV + folders

Adds synthetic anomaly paths to classifier **train** only (val/test unchanged).

```bash
python anomalydetect.py --build-augmented-classifier --synthetic-dir data/generated/vae_defects
```

If CutPaste PNGs alone are sufficient for your augment path, omit `--synthetic-dir` per `anomalydetect.py` docs.

Produces:

- `data/processed/classifier_augmented/split_assignments.csv`

#### Step B5: train classifier on augmented CSV

```bash
python classifier.py --split-assignments data/processed/classifier_augmented/split_assignments.csv
```

**Note:** Default checkpoint names like `baseline_trial_006.pt` are shared across CSVs. Copy a `.pt` to a backup path before overwriting if you compare baseline vs augmented.

---

### Step 4: Figures and exploratory UI (after you have checkpoints)

Approximate dependency order:

1. Latent summaries (needs `artifacts/vae/<class>/*.pt` and VAE CSV):

   ```bash
   python latent_viz.py --all-objects
   ```

   Default PNG folder: `outputs/latent_viz/`.

2. Optional grids (VAE PNG montages, latent PNG montages, histograms; optional classifier log comparison):

   ```bash
   python aggregate_viz_grids.py --classifier-loss-use-embedded-trial6
   ```

   Omit `--classifier-loss-use-embedded-trial6` if you will pass `--classifier-loss-log-baseline`/`-augmented` or two checkpoints with embedded `history` (run `aggregate_viz_grids.py --help`).

3. Gradio gallery (needs trained VAE checkpoints):

   ```bash
   python app.py
   ```

   Use `python app.py --help` (and optionally `DEFECT_GALLERY_*` env vars) for artifact directories on another machine.

---

### Quick sanity checklist

| You want | Prerequisites |
|---------|----------------|
| Baseline classifier | Steps 1–3, then Branch A |
| VAE augmented classifier | Steps 1–3, Branch B through B5 |
| t-SNE / interp plots | Step B2 (or checkpoints from elsewhere) + latent_viz CSV |
| `aggregate_viz_grids.py` grids | Populate `vae_best_plots_per_class/` best loss PNGs; run latent_viz for `latent_viz_grid.png` |

If something imports `image_anno.csv` or splits and fails missing file, rerun from **Step 2** then **Step 3**.
