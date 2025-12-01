import argparse
import json
import random
from dataclasses import dataclass
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Cfg:
    data_root: Path
    metadata_csv: Path
    out_dir: Path
    train_split: str
    val_split: str
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    amp: bool
    freeze_backbone: bool
    seed: int
    image_size: int
    device: str
    keyframes_only: bool
    filename_pattern: str
    threshold: float
    config_path: str


def build_existing_file_index(split_dir: Path) -> Dict[str, str]:
    """key: lowercase stem or lowercase full name -> actual filename"""
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

    # handle float-like "13700.0"
    try:
        stem2 = make_stem(pattern, int(float(row["vid"])), int(float(row["frame"]))).strip()
        key2 = stem2.lower()
        if key2 in existing_idx:
            return existing_idx[key2]
    except Exception:
        pass

    return None


class EndoscapesCVSDataset(Dataset):
    """
    CVS dataset from all_metadata.csv that contains columns:
      vid, frame, C1, C2, C3, is_ds_keyframe (optional filter)
    Images assumed like: {vid}_{frame}.jpg (pattern controls stem)
    """

    def __init__(self, split_dir: Path, df: pd.DataFrame, image_size: int, train: bool, pattern: str):
        self.split_dir = split_dir
        self.pattern = pattern

        existing_idx = build_existing_file_index(split_dir)

        df = df.copy()
        df["_mapped_file"] = df.apply(lambda r: map_row_to_existing_filename(r, existing_idx, pattern), axis=1)
        df = df[df["_mapped_file"].notna()].copy()

        for c in ["C1", "C2", "C3"]:
            if c not in df.columns:
                raise RuntimeError(f"CSV missing required column '{c}'. Found columns: {list(df.columns)}")

        df = df.dropna(subset=["C1", "C2", "C3"]).copy()

        self.files = df["_mapped_file"].astype(str).tolist()
        self.labels = df[["C1", "C2", "C3"]].astype(float).values

        if train:
            self.tfms = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                ]
            )
        else:
            self.tfms = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        img = Image.open(self.split_dir / fn).convert("RGB")
        x = self.tfms(img)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def build_model(out_dim: int = 3, freeze_backbone: bool = False):
    from torchvision.models import resnet50

    try:
        from torchvision.models import ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT
    except Exception:
        weights = "DEFAULT"

    model = resnet50(weights=weights)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    return model


@torch.no_grad()
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


def train_one_epoch(model, loader, optimizer, device, scaler, loss_fn):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        total += float(loss.detach().cpu())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, threshold: float):
    model.eval()
    total = 0.0
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total += float(loss.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    macro_f1, per = compute_metrics(logits, y, threshold=threshold)
    return total / max(1, len(loader)), macro_f1, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")

    # overrides
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--metadata_csv", type=str, default=None)
    p.add_argument("--train_split", type=str, default=None)
    p.add_argument("--val_split", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--filename_pattern", type=str, default=None)
    p.add_argument("--threshold", type=float, default=None)

    # flags
    p.add_argument("--amp", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--keyframes_only", action="store_true")

    args = p.parse_args()

    cfg_toml = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg_toml = load_toml(cfg_path)

    data_root = args.data_root or deep_get(cfg_toml, "paths.data_root", None)
    if data_root is None:
        raise RuntimeError("Missing data_root. Provide --data_root or set [paths].data_root in config.toml")

    metadata_csv_name = args.metadata_csv or deep_get(cfg_toml, "cvs.data.metadata_csv", "all_metadata.csv")
    train_split = args.train_split or deep_get(cfg_toml, "cvs.data.train_split", "train")
    val_split = args.val_split or deep_get(cfg_toml, "cvs.data.val_split", "val")
    out_dir = args.out_dir or deep_get(cfg_toml, "cvs.train.out_dir", "runs/cvs")

    filename_pattern = args.filename_pattern or deep_get(cfg_toml, "cvs.data.filename_pattern", "{vid}_{frame}")
    keyframes_only_cfg = bool(deep_get(cfg_toml, "cvs.data.keyframes_only", False))
    keyframes_only = bool(args.keyframes_only) or keyframes_only_cfg

    epochs = args.epochs if args.epochs is not None else int(deep_get(cfg_toml, "cvs.train.epochs", 15))
    batch_size = args.batch_size if args.batch_size is not None else int(deep_get(cfg_toml, "cvs.train.batch_size", 32))
    num_workers = args.num_workers if args.num_workers is not None else int(deep_get(cfg_toml, "cvs.train.num_workers", 4))
    lr = args.lr if args.lr is not None else float(deep_get(cfg_toml, "cvs.train.lr", 3e-4))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(deep_get(cfg_toml, "cvs.train.weight_decay", 1e-4))
    seed = args.seed if args.seed is not None else int(deep_get(cfg_toml, "cvs.train.seed", 42))
    image_size = args.image_size if args.image_size is not None else int(deep_get(cfg_toml, "cvs.train.image_size", 224))
    threshold = args.threshold if args.threshold is not None else float(deep_get(cfg_toml, "eval.cvs.threshold", 0.5))

    amp = bool(args.amp) or bool(deep_get(cfg_toml, "cvs.train.amp", False))
    freeze_backbone = bool(args.freeze_backbone) or bool(deep_get(cfg_toml, "cvs.train.freeze_backbone", False))

    cfg = Cfg(
        data_root=Path(data_root),
        metadata_csv=Path(data_root) / metadata_csv_name,
        out_dir=Path(out_dir),
        train_split=train_split,
        val_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        weight_decay=weight_decay,
        amp=amp,
        freeze_backbone=freeze_backbone,
        seed=seed,
        image_size=image_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        keyframes_only=keyframes_only,
        filename_pattern=filename_pattern,
        threshold=threshold,
        config_path=str(cfg_path),
    )

    seed_everything(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.metadata_csv.exists():
        raise FileNotFoundError(f"Missing {cfg.metadata_csv}")

    df = pd.read_csv(cfg.metadata_csv)
    for col in ["vid", "frame", "C1", "C2", "C3"]:
        if col not in df.columns:
            raise RuntimeError(f"CSV missing '{col}'. Columns found: {list(df.columns)}")

    if cfg.keyframes_only:
        if "is_ds_keyframe" not in df.columns:
            raise RuntimeError("You used keyframes_only but CSV has no 'is_ds_keyframe' column.")
        df = df[df["is_ds_keyframe"].astype(str).str.upper().isin(["TRUE", "1", "YES"])].copy()

    train_dir = cfg.data_root / cfg.train_split
    val_dir = cfg.data_root / cfg.val_split

    train_ds = EndoscapesCVSDataset(train_dir, df, cfg.image_size, train=True, pattern=cfg.filename_pattern)
    val_ds = EndoscapesCVSDataset(val_dir, df, cfg.image_size, train=False, pattern=cfg.filename_pattern)

    if len(train_ds) == 0:
        existing = sorted([p.name for p in (list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")))])[:10]
        sample_rows = df[["vid", "frame"]].head(10).to_dict("records")
        sample_stems = [cfg.filename_pattern.format(vid=r["vid"], frame=r["frame"]) for r in sample_rows]
        raise RuntimeError(
            "No CVS rows matched your TRAIN folder.\n\n"
            f"Train folder: {train_dir}\n"
            f"Pattern used: {cfg.filename_pattern}  (expects files like: {cfg.filename_pattern}.jpg)\n"
            f"Example train images: {existing}\n"
            f"Example generated stems from CSV: {sample_stems}\n"
        )

    # Print split stats
    def stats(name, ds: EndoscapesCVSDataset):
        y = ds.labels
        pos = (y >= 0.5).sum(axis=0)
        print(f"{name}: n={len(ds)} | positives [C1,C2,C3]={pos.tolist()}")

    stats("TRAIN", train_ds)
    stats("VAL  ", val_ds)

    model = build_model(out_dim=3, freeze_backbone=cfg.freeze_backbone)
    device = torch.device(cfg.device)
    model.to(device)

    # imbalance weights from training
    y_train = torch.tensor(train_ds.labels, dtype=torch.float32)
    y_bin = (y_train >= 0.5).float()
    pos = y_bin.sum(dim=0)
    neg = y_bin.shape[0] - pos
    pos_weight = (neg / (pos + 1e-8)).clamp(1.0, 50.0).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and cfg.device == "cuda") else None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    best_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, loss_fn)
        va_loss, macro_f1, per = evaluate(model, val_loader, device, loss_fn, threshold=cfg.threshold)

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_macro_f1": macro_f1, "per_criterion": per}
        print(json.dumps(row))

        # neat print
        print(
            f"\nEpoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | macroF1={macro_f1:.4f} (thr={cfg.threshold})"
        )
        for name, d in zip(["C1", "C2", "C3"], per):
            print(
                f"  {name}: P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
                f"(TP={d['tp']} FP={d['fp']} FN={d['fn']})"
            )
        print("")

        torch.save(model.state_dict(), cfg.out_dir / "cvs_last.pth")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), cfg.out_dir / "cvs_best.pth")

        with open(cfg.out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    meta = {
        "config_path": cfg.config_path,
        "data_root": str(cfg.data_root),
        "metadata_csv": cfg.metadata_csv.name,
        "train_split": cfg.train_split,
        "val_split": cfg.val_split,
        "out_dir": str(cfg.out_dir),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "amp": cfg.amp,
        "freeze_backbone": cfg.freeze_backbone,
        "seed": cfg.seed,
        "image_size": cfg.image_size,
        "filename_pattern": cfg.filename_pattern,
        "keyframes_only": cfg.keyframes_only,
        "threshold": cfg.threshold,
        "label_cols": ["C1", "C2", "C3"],
        "id_cols": ["vid", "frame"],
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "best_val_macro_f1": float(best_f1),
    }
    (cfg.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Best val macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
