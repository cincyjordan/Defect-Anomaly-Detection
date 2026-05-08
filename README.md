## **Name:** Generating industrial defects for anomaly detection (VisA).

**Overview:** Combat class imbalance (~9.6k normal vs ~1.2k defect images globally) so a downstream CNN does not plateau on nominal accuracy while missing defects. A convolutional VAE trains on defect crops only, per VisA category, and generated (and CutPaste) anomalies augment minority-class classifier training alongside real labels.

## **Extra Criteria:**

**Metrics tracking:** visually compare loss vs reconstruction runs Epoch curves as **PNG summaries** (`outputs/viz_grids/`): VAE recon/KL/total grids; classifier baseline vs augmented BCE & accuracy panels. Static plots substituted for weights-and-biases dashboards for reproducibility. 

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/classifier_baseline_vs_augmented_loss_grid.png' width = '800'>

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/classifier_histogram_grid.png' width = '800'>

**Classifier compare:** often lower train BCE after augmentation versus **messier validation** if synthetic defects do not match test defect appearances; interpret your PNG for alignment of both curves before claiming generalization gains.


**Hyperparameter tuning:** LR, latent size, batch search Random search in `**model_vae.py`** (latent dim, beta, LR, augmentation) and `**classifier.py**` (LR, batch, dropout); best runs selected on val reconstruction / val BCE respectively. Mosaic VAE curves show cross-category variability. 

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/vae_best_loss_grid.png' width = '800'>

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/vae_histogram_grid.png' width = '800'>

The **VAE mosaic** hints which objects converge smoothly (steady val recon) versus which linger with higher KL trade-offs—informing where **CutPaste** or more decoder samples warrant budget.

**Latent space exploration:** t‑SNE, interpolations | **`latent_viz.py`**

<img src='https://raw.githubusercontent.com/cincyjordan/Defect-Anomaly-Detection/refs/heads/main/outputs/viz_grids/latent_viz_grid.png' width = '800'>

 **`latent_viz_grid`**: separation/noise suggests how hard linear decision surfaces become in \mu space before the pooling CNN. Histograms sanity-check scale spread without asserting optimality.

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
