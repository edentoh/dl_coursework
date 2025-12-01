import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from config_utils import load_toml, deep_get


def build_existing_file_index(split_dir: Path) -> Dict[str, str]:
    files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.jpeg")) + list(split_dir.glob("*.png"))
    idx: Dict[str, str] = {}
    for f in files:
        idx[f.name.lower()] = f.name
        idx[f.stem.lower()] = f.name
    return idx


def make_stem(pattern: str, vid: int, frame: int) -> str:
    return pattern.format(vid=int(vid), frame=int(frame))


def map_row_to_existing_filename(row: pd.Series, existing_idx: Dict[str, str], pattern: str) -> Optional[str]:
    try:
        stem = make_stem(pattern, row["vid"], row["frame"]).strip()
    except Exception:
        return None

    key = stem.lower()
    if key in existing_idx:
        return existing_idx[key]

    try:
        stem2 = make_stem(pattern, int(float(row["vid"])), int(float(row["frame"]))).strip()
        key2 = stem2.lower()
        if key2 in existing_idx:
            return existing_idx[key2]
    except Exception:
        pass

    return None


class EndoscapesCVSDataset(Dataset):
    def __init__(self, split_dir: Path, df: pd.DataFrame, image_size: int, filename_pattern: str):
        self.split_dir = split_dir
        self.tfms = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

        existing_idx = build_existing_file_index(split_dir)
        df = df.copy()
        df["_mapped_file"] = df.apply(lambda r: map_row_to_existing_filename(r, existing_idx, filename_pattern), axis=1)
        df = df[df["_mapped_file"].notna()].copy()

        for c in ["C1", "C2", "C3"]:
            if c not in df.columns:
                raise RuntimeError(f"CSV missing required column '{c}'. Found: {list(df.columns)}")
        df = df.dropna(subset=["C1", "C2", "C3"]).copy()

        self.files = df["_mapped_file"].astype(str).tolist()
        self.labels = df[["C1", "C2", "C3"]].astype(float).values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.split_dir / self.files[idx]).convert("RGB")
        x = self.tfms(img)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def compute_metrics(logits: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    y_pred = (probs >= threshold).int()
    y_bin = (y_true >= 0.5).int()

    eps = 1e-9
    per = []
    for k in range(3):
        tp = int(((y_pred[:, k] == 1) & (y_bin[:, k] == 1)).sum().item())
        fp = int(((y_pred[:, k] == 1) & (y_bin[:, k] == 0)).sum().item())
        fn = int(((y_pred[:, k] == 0) & (y_bin[:, k] == 1)).sum().item())
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        per.append({"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    macro_f1 = float(np.mean([d["f1"] for d in per]))
    return macro_f1, per


def build_model(out_dim: int = 3):
    from torchvision.models import resnet50

    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
    except Exception:
        weights = "DEFAULT"

    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    return model


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    macro_f1, per = compute_metrics(logits, y, threshold=threshold)
    return macro_f1, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")

    # optional overrides
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--metadata_csv", type=str, default=None)
    p.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--meta", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    cfg_toml = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg_toml = load_toml(cfg_path)

    data_root = args.data_root or deep_get(cfg_toml, "paths.data_root", None)
    if data_root is None:
        raise RuntimeError("Missing data_root. Provide --data_root or set [paths].data_root in config.toml")

    # get defaults from config
    split = args.split or deep_get(cfg_toml, "eval.cvs.split", "test")
    ckpt = args.ckpt or deep_get(cfg_toml, "eval.cvs.ckpt", None)
    meta_path = args.meta or deep_get(cfg_toml, "eval.cvs.meta", None)
    num_workers = args.num_workers if args.num_workers is not None else int(deep_get(cfg_toml, "eval.cvs.num_workers", 4))
    batch_size = args.batch_size if args.batch_size is not None else int(deep_get(cfg_toml, "eval.cvs.batch_size", 64))
    threshold = args.threshold if args.threshold is not None else float(deep_get(cfg_toml, "eval.cvs.threshold", 0.5))

    if ckpt is None:
        raise RuntimeError("Missing CVS checkpoint. Set [eval.cvs].ckpt in config.toml or pass --ckpt")
    if meta_path is None:
        raise RuntimeError("Missing CVS meta.json. Set [eval.cvs].meta in config.toml or pass --meta")

    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    filename_pattern = meta.get("filename_pattern", "{vid}_{frame}")
    keyframes_only = bool(meta.get("keyframes_only", False))
    image_size = int(meta.get("image_size", 224))
    metadata_csv_name = args.metadata_csv or meta.get("metadata_csv") or deep_get(cfg_toml, "cvs.data.metadata_csv", "all_metadata.csv")

    df = pd.read_csv(Path(data_root) / metadata_csv_name)
    if keyframes_only:
        if "is_ds_keyframe" not in df.columns:
            raise RuntimeError("meta.json says keyframes_only=true but CSV has no is_ds_keyframe column.")
        df = df[df["is_ds_keyframe"].astype(str).str.upper().isin(["TRUE", "1", "YES"])].copy()

    split_dir = Path(data_root) / split
    ds = EndoscapesCVSDataset(split_dir, df, image_size=image_size, filename_pattern=filename_pattern)

    if len(ds) == 0:
        existing = sorted([p.name for p in (list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))])[:10]
        sample_rows = df[["vid", "frame"]].head(10).to_dict("records") if ("vid" in df.columns and "frame" in df.columns) else []
        sample_stems = [filename_pattern.format(vid=r["vid"], frame=r["frame"]) for r in sample_rows[:10]]
        raise RuntimeError(
            "No CVS-labeled frames matched this split folder.\n\n"
            f"Split folder: {split_dir}\n"
            f"filename_pattern: {filename_pattern}\n"
            f"keyframes_only: {keyframes_only}\n"
            f"Example images in folder: {existing}\n"
            f"Example generated stems: {sample_stems}\n"
        )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(out_dim=3)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)

    macro_f1, per = evaluate(model, loader, device, threshold=threshold)

    out = {
        "split": split,
        "ckpt": ckpt,
        "meta": meta_path,
        "n": len(ds),
        "threshold": float(threshold),
        "macro_f1": float(macro_f1),
        "per_criterion": per,
        "image_size": image_size,
        "filename_pattern": filename_pattern,
        "keyframes_only": keyframes_only,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
