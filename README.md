```markdown
# Endoscapes2023 Coursework Pipeline (Baseline vs. Tuned)

This repo implements a comparative study on the Endoscapes dataset using two distinct modeling approaches:

### 1. Baseline Models (Control Group)
* **Detector:** Faster R-CNN with **ResNet-50** backbone. Trained with constant LR.
* **CVS Classifier:** **Single-Frame** ResNet-50 Classifier.

### 2. Tuned Models (Experimental Group)
* **Detector:** Faster R-CNN with **ResNet-101** backbone + Learning Rate Scheduler + "Info" fix.
* **CVS Classifier:** **Temporal Model** (CNN + LSTM) processing 5-frame video sequences with ResNet-101 backbone.

Everything is configurable via `config.toml` (Baseline) and `config_tuned.toml` (Tuned).

---

## Folder layout (expected)

Your Endoscapes root folder should look like this:

```

endoscapes/
all\_metadata.csv
train/
annotation\_coco.json
\*.jpg
val/
annotation\_coco.json
\*.jpg
test/
annotation\_coco.json
\*.jpg

````

---

## Files in this project

**Configuration**
* `config.toml` — Configuration for Baseline experiments.
* `config_tuned.toml` — Configuration for Tuned experiments (ResNet-101, Temporal, Schedulers).
* `config_utils.py` — Helper for reading TOML configs.

**Training Scripts**
* `train_detector.py` — Train Baseline Detector (ResNet-50).
* `train_detector_tuned.py` — Train Tuned Detector (ResNet-101 + Scheduler).
* `train_cvs.py` — Train Baseline CVS (Single Frame).
* `train_cvs_tuned.py` — Train Tuned CVS (Temporal LSTM).

**Evaluation & Visualization**
* `evaluate_detector.py` — Evaluate Baseline Detector (mAP).
* `evaluate_detector_tuned.py` — Evaluate Tuned Detector (ResNet-101 support).
* `evaluate_cvs.py` — Evaluate Baseline CVS.
* `evaluate_cvs_tuned.py` — Evaluate Tuned CVS (Sequence support).
* `predict_detector_one.py` — Visualize predictions (Box + CVS) vs. Ground Truth for a single image.

---

## 0) Environment setup (Windows / venv)

From your coursework folder:

```powershell
python -m venv endoscapes_venv
.\endoscapes_venv\Scripts\activate
python -m pip install --upgrade pip
````

**Important:** Install the **GPU (CUDA)** version of PyTorch. Do **not** use the CPU version.

```powershell
# Example for CUDA 12.1 (Adjust if you have a different CUDA version)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

Install remaining dependencies:

```powershell
pip install numpy pandas pillow matplotlib pycocotools
```

-----

## 1\) Train the Models

### A. Baseline (ResNet-50 / Single Frame)

```powershell
# Detector
python train_detector.py --config config.toml

# CVS
python train_cvs.py --config config.toml
```

**Outputs:** `runs/detector_baseline2` and `runs/cvs_baseline2`

### B. Tuned (ResNet-101 / Temporal LSTM)

```powershell
# Detector (ResNet-101 + Scheduler)
python train_detector_tuned.py --config config_tuned.toml

# CVS (Temporal Sequence Length = 5)
python train_cvs_tuned.py --config config_tuned.toml
```

**Outputs:** `runs/detector_tuned_r101` and `runs/cvs_tuned_temporal`

-----

## 2\) Evaluate the Models

Use the specific script corresponding to the model type (Baseline vs. Tuned).

### Baseline Evaluation

```powershell
# Detector (mAP)
python evaluate_detector.py --config config.toml

# CVS (Macro F1)
python evaluate_cvs.py --config config.toml
```

### Tuned Evaluation

*Note: Must use `_tuned.py` scripts to handle ResNet-101 and Temporal Data loading.*

```powershell
# Detector (mAP)
python evaluate_detector_tuned.py --config config_tuned.toml

# CVS (Macro F1 on Video Sequences)
python evaluate_cvs_tuned.py --config config_tuned.toml
```

-----

## 3\) Visualize Predictions (Qualitative Analysis)

You can pick a random annotated image to see how well your models perform visually.

**To visualize Baseline models:**

```powershell
python predict_detector_one.py --config config.toml --pick_annotated
```

**To visualize Tuned models:**

```powershell
python predict_detector_one.py --config config_tuned.toml --pick_annotated
```

This will display:

1.  **Ground Truth:** Actual Bounding Boxes + Actual CVS Labels (from CSV).
2.  **Prediction:** Predicted Boxes + Predicted CVS Probabilities.

-----

## Recommended Workflow for Report

1.  **Train Baseline:** Run `train_detector.py` and `train_cvs.py`.
2.  **Train Tuned:** Run `train_detector_tuned.py` and `train_cvs_tuned.py`.
3.  **Plot Training Curves:** Use the `history.jsonl` files generated in the `runs/` folders to plot Loss vs. Epoch for both models.
4.  **Compare Metrics:** Run the evaluation scripts and compare `mAP` (Detector) and `Macro F1` (CVS).
5.  **Visual Evidence:** Use `predict_detector_one.py` to save example images where the Tuned model performs better than the Baseline (e.g., detecting a tool that the Baseline missed, or correctly classifying a CVS criteria).

<!-- end list -->

```
```