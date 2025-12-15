Here is a complete `README.md` you can paste into the repo (you can tweak dataset paths / wording as you like):

````markdown
# Deep Learning Coursework – Endoscapes Detection & CVS

This repository contains my COMP0220 Deep Learning coursework code for surgical image understanding on the **Endoscapes2023** dataset (Camma Lab). It implements:

- A **Faster R-CNN (ResNet-50 + FPN)** object detector for surgical tools and key anatomical structures.
- A **ResNet-50 multi-label classifier** for the three **Critical View of Safety (CVS)** criteria (C1, C2, C3).
- Config-driven training, evaluation, and prediction scripts for both **baseline** and **tuned** models.

---

## 1. Repository Structure

Main files:

- `config.toml` – baseline configuration (paths, hyper-parameters for detector + CVS).
- `config_tuned.toml` – tuned configuration with more aggressive training settings.
- `config_utils.py` – helpers for loading TOML configs and nested values.

Detection:

- `train_detector.py` / `train_detector_tuned.py` – train baseline / tuned Faster R-CNN.
- `evaluate_detector.py` / `evaluate_detector_tuned.py` – evaluate trained detectors (COCO mAP).
- `predict_one.py` – run detector on a single frame and visualise predictions vs ground truth.
- `predict_all.py` – batch prediction / comparison using `predict_comparison.toml`.
- `predict_one.toml`, `predict_comparison.toml` – configs for prediction scripts.

CVS classification:

- `train_cvs.py` / `train_cvs_tuned.py` – train baseline / tuned CVS ResNet-50 classifier.
- `evaluate_cvs.py` / `evaluate_cvs_tuned.py` – evaluate classifiers (per-criterion metrics).

Utilities:

- `inspect_data.ipynb` – quick data sanity checks (COCO boxes + CVS CSV).
- (Model checkpoints and logs are written under `runs/…` at training time.)

---

## 2. Dataset & Expected Layout

This code assumes the **Endoscapes2023** dataset has been downloaded separately and unpacked to a folder, e.g.:

```text
endoscapes/
  train/
    annotation_coco.json
    001_1234.jpg
    001_1266.jpg
    ...
  val/
    annotation_coco.json
    ...
  test/
    annotation_coco.json
    ...
  all_metadata.csv          # CVS labels (vid, frame, C1, C2, C3, etc.)
````

In `config.toml` and `config_tuned.toml`, set:

```toml
[paths]
data_root = "C:/path/to/endoscapes"
runs_dir  = "runs"
```

so that `data_root` points to this directory on your machine.

> Note: The dataset itself is **not** included in this repository. You must obtain it from the official Endoscapes / CAMMA distribution subject to their license and ethics constraints.

---

## 3. Environment Setup

1. Create and activate a virtual environment (example: Windows PowerShell):

   ```bash
   python -m venv endoscapes_venv
   .\endoscapes_venv\Scripts\activate
   ```

2. Install dependencies (typical set):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install pycocotools opencv-python pandas matplotlib tomli
   ```

   Adjust the CUDA / PyTorch index-url according to your hardware and OS.

3. (Optional) Install Jupyter if you want to run the notebook:

   ```bash
   pip install jupyterlab
   ```

---

## 4. Inspecting the Data

Before training, you can sanity-check that images, COCO annotations and CVS CSV line up.

1. Open `inspect_data.ipynb` in Jupyter.
2. Set `DATA_ROOT` inside the notebook to your Endoscapes path.
3. Run all cells:

   * It will draw an example image with GT bounding boxes from `annotation_coco.json`.
   * It will print CVS columns and show a few rows that match files in the chosen split.

---

## 5. Training – Baseline Models

All scripts are config-driven. By default they read from `config.toml`.

### 5.1 Detector (Faster R-CNN ResNet-50 FPN)

From the repo root:

```bash
python train_detector.py --config config.toml
```

What this does:

* Loads a COCO-pretrained Faster R-CNN (ResNet-50 + FPN).
* Replaces the classifier head with Endoscapes classes.
* Trains on `train_split` / validates on `val_split` from `[detector.data]` in `config.toml`.
* Uses SGD with hyper-parameters from `[detector.train]`.
* Writes checkpoints and a `history.jsonl` to `detector.train.out_dir` (e.g. `runs/detector_baseline`):

  * `detector_best.pth` – best val mAP.
  * `detector_last.pth` – last epoch model.
  * `meta.json` – category mappings + image size settings.

### 5.2 CVS Classifier (ResNet-50 Multi-label)

```bash
python train_cvs.py --config config.toml
```

This script:

* Loads an ImageNet-pretrained ResNet-50.
* Replaces the 1000-class head with a 3-output head for C1, C2, C3.
* Builds a dataset from `all_metadata.csv`, filtered to `train_split` / `val_split` and keyframes if configured.
* Trains with BCEWithLogitsLoss and hyper-parameters from `[cvs.train]`.
* Writes checkpoints and logs to `cvs.train.out_dir` (e.g. `runs/cvs_baseline`):

  * `cvs_best.pth`, `cvs_last.pth`, `meta.json`, `history.jsonl`.

Each epoch logs train loss, val loss, macro-F1 and per-criterion metrics.

---

## 6. Training – Tuned Models

Tuned configurations are defined in `config_tuned.toml`. They typically adjust:

* Learning rates, epochs, batch sizes.
* Detection image size / CVS thresholds.
* Output directories (e.g. `runs/detector_tuned_*`, `runs/cvs_tuned_*`).

Run:

```bash
python train_detector_tuned.py --config config_tuned.toml
python train_cvs_tuned.py --config config_tuned.toml
```

Baseline and tuned models write to **different `out_dir`s**, so you can compare them without overwriting checkpoints.

---

## 7. Evaluation

### 7.1 Detector mAP (COCO)

Baseline:

```bash
python evaluate_detector.py --config config.toml
```

Tuned:

```bash
python evaluate_detector_tuned.py --config config_tuned.toml
```

These scripts:

* Load the specified checkpoint and `meta.json`.
* Run COCO mAP evaluation on the split specified in `[eval.detector]` (usually `test`).
* Print a JSON summary with at least `{"split": "...", "map": ...}`.

### 7.2 CVS Metrics (C1, C2, C3)

Baseline:

```bash
python evaluate_cvs.py --config config.toml
```

Tuned:

```bash
python evaluate_cvs_tuned.py --config config_tuned.toml
```

They:

* Load the classifier checkpoint + `meta.json`.
* Evaluate on the configured split in `[eval.cvs]`.
* Use a probability threshold (e.g. `threshold = 0.5` or 0.65) from the config.
* Print a JSON object with:

  * `macro_f1`
  * Per-criterion precision, recall, F1 and TP/FP/FN counts.

---

## 8. Prediction & Visualisation

### 8.1 Single-image Detector Demo

Configure `predict_one.toml` with:

* `data_root`, `split`, `image` (filename or full path),
* `ckpt`, `meta`, `score_thr`, `topk`, `save_dir`.

Then run:

```bash
python predict_one.py --config predict_one.toml
```

It will:

* Load the detector and run inference on the chosen frame.
* Optionally load COCO GT annotations for that frame.
* Save two PNGs in `save_dir`:

  * `*_GT.png` – ground-truth boxes per class.
  * `*_PRED.png` – predicted boxes, coloured per class, with scores.

### 8.2 Batch Comparison (e.g. baseline vs tuned)

Use `predict_all.py` with `predict_comparison.toml` pointing to:

* A list of images or a split.
* Paths for multiple checkpoints (e.g. baseline vs tuned).
* Output folder for side-by-side visualisations.

Run:

```bash
python predict_all.py --config predict_comparison.toml
```

---

## 9. Logs & Analysis

Each training run appends a `history.jsonl` in its `out_dir`. Typical entries:

* Detector: `{"epoch": 3, "train_loss": ..., "val_map": ...}`
* CVS: `{"epoch": 4, "train_loss": ..., "val_loss": ..., "val_macro_f1": ..., "per_criterion": [...]}`

You can load this file into a Jupyter notebook and plot:

* Training vs validation loss.
* Validation mAP over epochs.
* Per-criterion F1 over epochs for CVS.

---

## 10. Notes

* This code is written for a **single-GPU** environment (e.g. RTX 4060). Multi-GPU training is not implemented.
* Mixed precision (AMP) can be enabled/disabled via the `amp` flags in `config.toml` / `config_tuned.toml`.
* All paths in the configs are Windows-style by default (e.g. `C:/Users/...`). Adjust them if you run on Linux or macOS.

---

## 11. Acknowledgements

* **Endoscapes2023 dataset** – provided by the CAMMA group; used here for educational purposes only.
* **PyTorch**, **TorchVision**, and **pycocotools** – for deep learning and COCO evaluation.

```
::contentReference[oaicite:0]{index=0}
```
