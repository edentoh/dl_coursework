```md
# Endoscapes2023 Coursework Pipeline (Detector + CVS)

This repo trains **two models** on the Endoscapes dataset:

1) **Object Detection** (6 classes): Faster R-CNN (torchvision) trained on COCO-style bounding boxes.  
2) **CVS Classification** (C1/C2/C3): ResNet-50 multi-label classifier trained from `all_metadata.csv` using `vid` + `frame`.

Everything is configurable via `config.toml`.

---

## Folder layout (expected)

Your Endoscapes root folder should look like this:

```

endoscapes/
all_metadata.csv
train/
annotation_coco.json
*.jpg
val/
annotation_coco.json
*.jpg
test/
annotation_coco.json
*.jpg

````

Notes:
- The detector uses `train/annotation_coco.json`, `val/annotation_coco.json`, etc.
- The CVS model uses `all_metadata.csv` which contains `vid`, `frame`, `C1`, `C2`, `C3`, and usually `is_ds_keyframe`.

---

## Files in this project

- `config.toml` — central config (paths, epochs, batch sizes, ckpt/meta paths for evaluation)
- `config_utils.py` — helper for reading TOML config
- `train_detector.py` — train Faster R-CNN detector
- `evaluate_detector.py` — evaluate detector and print COCO mAP
- `train_cvs.py` — train CVS classifier (C1/C2/C3)
- `evaluate_cvs.py` — evaluate CVS classifier

---

## 0) Environment setup (Windows / venv)

From your coursework folder:

```powershell
python -m venv endoscapes_venv
.\endoscapes_venv\Scripts\activate
python -m pip install --upgrade pip
````

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Install remaining dependencies:

```powershell
pip install numpy pandas pillow matplotlib pycocotools
```

---

## 1) Configure `config.toml`

Example important keys:

```toml
[paths]
data_root = "C:/Users/TUF/COMP0220_Deep_Learning/coursework/endoscapes"

[detector.train]
epochs = 40
batch_size = 2
num_workers = 4
out_dir = "runs/detector"

[cvs.train]
epochs = 20
batch_size = 32
num_workers = 4
out_dir = "runs/cvs"

[eval.detector]
split = "test"
ckpt  = "runs/detector/detector_best.pth"
meta  = "runs/detector/meta.json"

[eval.cvs]
split = "test"
ckpt  = "runs/cvs/cvs_best.pth"
meta  = "runs/cvs/meta.json"
```

Important:

* Use forward slashes in Windows paths inside TOML: `C:/Users/...`
* `eval.detector.ckpt` and `eval.cvs.ckpt` must exist before evaluation.

---

## 2) Train the detector (Faster R-CNN)

Run:

```powershell
python train_detector.py --config config.toml
```

Outputs saved to:

* `runs/detector/detector_last.pth`
* `runs/detector/detector_best.pth`
* `runs/detector/history.jsonl`
* `runs/detector/meta.json`

### What is val mAP?

The training prints `val_map`, which is COCO **mAP@[0.5:0.95]** (the standard COCO metric). Higher is better.

---

## 3) Evaluate the detector

If you set `[eval.detector]` in `config.toml`, just run:

```powershell
python evaluate_detector.py --config config.toml
```

It prints something like:

```json
{
  "split": "test",
  "ckpt": "runs/detector/detector_best.pth",
  "map": 0.3160
}
```

---

## 4) Train CVS (C1/C2/C3)

This trains a **multi-label classifier** outputting 3 logits for C1/C2/C3.

Run:

```powershell
python train_cvs.py --config config.toml
```

Key points:

* The CSV uses `vid` + `frame`. The code matches images by generating filename stem:

  * default pattern: `"{vid}_{frame}"` → like `166_13700.jpg`
* If your filenames differ, change:

  * `cvs.data.filename_pattern` in `config.toml`

Optional (recommended):

* `cvs.data.keyframes_only=true` uses only CSV rows where `is_ds_keyframe == TRUE`.

Outputs:

* `runs/cvs/cvs_last.pth`
* `runs/cvs/cvs_best.pth`
* `runs/cvs/history.jsonl`
* `runs/cvs/meta.json`

### Why do I see 3 precision/recall/F1 values?

Because CVS has **3 separate binary labels** (C1,C2,C3). The script prints metrics per criterion + macro average.

---

## 5) Evaluate CVS

If you set `[eval.cvs]` in `config.toml`, run:

```powershell
python evaluate_cvs.py --config config.toml
```

Example output:

```json
{
  "split": "test",
  "macro_f1": 0.48,
  "per_criterion": [...]
}
```

---

## CLI Overrides (optional)

Any setting can be overridden without editing TOML. Examples:

Detector:

```powershell
python train_detector.py --config config.toml --epochs 10 --lr 0.001
```

CVS:

```powershell
python train_cvs.py --config config.toml --batch_size 16 --epochs 5
```

CVS filename pattern override:

```powershell
python train_cvs.py --config config.toml --filename_pattern "{vid}_{frame}"
```


## What checkpoints are (.pth)?

`.pth` files store **model weights** (parameters).
To use them:

* build the same architecture
* call `load_state_dict(torch.load(...))`

---

## Recommended workflow

1. Train detector
2. Evaluate detector on test
3. Train CVS
4. Evaluate CVS on test
5. For report: include training curves (`history.jsonl`) and final metrics

---

## Final quick commands

```powershell
# activate venv
.\endoscapes_venv\Scripts\activate

# train detector
python train_detector.py --config config.toml

# eval detector
python evaluate_detector.py --config config.toml

# train cvs
python train_cvs.py --config config.toml

# eval cvs
python evaluate_cvs.py --config config.toml

# predict using detector
python predict_detector_one.py --config config.toml
```

```
::contentReference[oaicite:0]{index=0}
```
