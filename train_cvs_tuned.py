# train_cvs_tuned.py
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights 

from config_utils import load_toml, deep_get


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    seq_len: int 


# ---------------------------------------------------------
# 1. Temporal Dataset
# ---------------------------------------------------------
def build_existing_file_index(split_dir: Path) -> Dict[str, str]:
    files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.jpeg")) + list(split_dir.glob("*.png"))
    idx: Dict[str, str] = {}
    for f in files:
        idx[f.name.lower()] = f.name
        idx[f.stem.lower()] = f.name
    return idx

def make_stem(pattern: str, vid: int, frame: int) -> str:
    return pattern.format(vid=int(vid), frame=int(frame))

def map_row_to_existing_filename(row, existing_idx: Dict[str, str], pattern: str) -> Optional[str]:
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

class TemporalCVSDataset(Dataset):
    def __init__(self, split_dir: Path, df: pd.DataFrame, image_size: int, train: bool, pattern: str, seq_len: int = 5):
        self.split_dir = split_dir
        self.seq_len = seq_len
        self.pattern = pattern
        self.existing_idx = build_existing_file_index(split_dir)

        df = df.copy()
        df["_mapped_file"] = df.apply(lambda r: map_row_to_existing_filename(r, self.existing_idx, pattern), axis=1)
        valid_mask = df["_mapped_file"].notna() & df[["C1", "C2", "C3"]].notna().all(axis=1)
        self.df = df[valid_mask].copy()

        self.groups = dict(list(self.df.groupby("vid")))
        self.samples = self.df.to_dict('records')

        # Use ImageNet Normalization
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if train:
            self.tfms = T.Compose([
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                norm
            ])
        else:
            self.tfms = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                norm
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target_row = self.samples[idx]
        vid = target_row['vid']
        current_frame = target_row['frame']
        
        vid_df = self.groups[vid].sort_values("frame")
        available_frames = vid_df[vid_df['frame'] <= current_frame]
        
        if len(available_frames) >= self.seq_len:
            seq_rows = available_frames.iloc[-self.seq_len:]
        else:
            pad_count = self.seq_len - len(available_frames)
            first_row = available_frames.iloc[0:1]
            seq_rows = pd.concat([first_row] * pad_count + [available_frames])
            
        images = []
        for _, r in seq_rows.iterrows():
            fn = r['_mapped_file']
            path = self.split_dir / fn
            if not path.exists():
                img = Image.new('RGB', (224, 224))
            else:
                img = Image.open(path).convert("RGB")
            images.append(self.tfms(img))
            
        x_seq = torch.stack(images, dim=0) # (Seq_Len, C, H, W)
        labels = [target_row['C1'], target_row['C2'], target_row['C3']]
        y = torch.tensor(labels, dtype=torch.float32)
        return x_seq, y


# ---------------------------------------------------------
# 2. Temporal Model (ResNet50 + LSTM)
# ---------------------------------------------------------
class TemporalCVSModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256, dropout=0.5, bidirectional=True, freeze_backbone=False):
        super().__init__()
        
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
                
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = resnet.fc.in_features 
        
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=self.feat_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=self.bidirectional
        )
        
        self.dropout = nn.Dropout(p=dropout) 
        
        fc_input_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x_flat) 
        feats = feats.flatten(1)               
        feats = feats.view(B, T, -1)           
        
        lstm_out, _ = self.lstm(feats)
        last_out = lstm_out[:, -1, :]          
        
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        return logits


# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
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
        per.append({"precision": prec, "recall": rec, "f1": f1})
    macro_f1 = float(np.mean([d["f1"] for d in per]))
    return macro_f1, per


def train_one_epoch(model, loader, optimizer, device, scaler, loss_fn):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(x)
            loss = loss_fn(logits, y)
            
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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


# --- WRAPPER FUNCTION TO RUN ONE EXPERIMENT ---
def run_experiment(exp_cfg, common_cfg):
    print("="*60)
    print(f"STARTING EXPERIMENT: Hidden Dim = {exp_cfg['hidden_dim']}")
    print("="*60)
    
    # Merge configs
    # Create specific output directory
    out_dir = Path(common_cfg.out_dir) / f"hidden_{exp_cfg['hidden_dim']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = Cfg(
        data_root=common_cfg.data_root,
        metadata_csv=common_cfg.metadata_csv,
        out_dir=out_dir, # Use the specific subfolder
        train_split=common_cfg.train_split,
        val_split=common_cfg.val_split,
        epochs=common_cfg.epochs,
        batch_size=common_cfg.batch_size,
        num_workers=4,
        lr=common_cfg.lr,
        weight_decay=common_cfg.weight_decay,
        amp=True,
        freeze_backbone=False,
        seed=42,
        image_size=common_cfg.image_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        keyframes_only=True,
        filename_pattern="{vid}_{frame}",
        threshold=0.6,
        config_path=common_cfg.config_path,
        seq_len=common_cfg.seq_len,
    )

    seed_everything(cfg.seed)
    
    df = pd.read_csv(cfg.metadata_csv)
    if cfg.keyframes_only:
        df = df[df["is_ds_keyframe"].astype(str).str.upper().isin(["TRUE", "1", "YES"])].copy()

    train_dir = cfg.data_root / cfg.train_split
    val_dir = cfg.data_root / cfg.val_split

    train_ds = TemporalCVSDataset(train_dir, df, cfg.image_size, train=True, pattern=cfg.filename_pattern, seq_len=cfg.seq_len)
    val_ds = TemporalCVSDataset(val_dir, df, cfg.image_size, train=False, pattern=cfg.filename_pattern, seq_len=cfg.seq_len)

    # Use the Experiment-Specific Hidden Dim
    HIDDEN_DIM = exp_cfg['hidden_dim']
    DROPOUT_P = 0.7
    BIDIRECTIONAL = True
    POS_WEIGHT_VALS = [4.0, 4.0, 4.5] 
    
    model = TemporalCVSModel(
        num_classes=3, 
        hidden_dim=HIDDEN_DIM, 
        dropout=DROPOUT_P, 
        bidirectional=BIDIRECTIONAL,
        freeze_backbone=cfg.freeze_backbone
    )
    model.to(cfg.device)

    pos_weight = torch.tensor(POS_WEIGHT_VALS).to(cfg.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    milestones = [int(cfg.epochs*0.6), int(cfg.epochs*0.8)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and cfg.device == "cuda") else None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    best_f1 = -1.0
    (cfg.out_dir / "history.jsonl").write_text("", encoding="utf-8")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device=cfg.device, scaler=scaler, loss_fn=loss_fn)
        scheduler.step()
        
        va_loss, macro_f1, per = evaluate(model, val_loader, device=cfg.device, loss_fn=loss_fn, threshold=cfg.threshold)
        curr_lr = optimizer.param_groups[0]["lr"]

        print(f"[HDim={HIDDEN_DIM}] Epoch {epoch:02d} | Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Val F1: {macro_f1:.4f}")
        
        row = {
            "epoch": epoch, 
            "train_loss": tr_loss, 
            "val_loss": va_loss, 
            "val_macro_f1": macro_f1,
            "lr": curr_lr
        }
        with open(cfg.out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), cfg.out_dir / "cvs_tuned_best.pth")
            
            meta = {
                "config_path": cfg.config_path,
                "data_root": str(cfg.data_root),
                "metadata_csv": cfg.metadata_csv.name,
                "train_split": cfg.train_split,
                "val_split": cfg.val_split,
                "out_dir": str(cfg.out_dir),
                
                "image_size": cfg.image_size,
                "seq_len": cfg.seq_len,
                "filename_pattern": cfg.filename_pattern,
                "keyframes_only": cfg.keyframes_only,
                
                "backbone": "resnet50",
                "model_type": "temporal_lstm_bidirectional",
                "hidden_dim": HIDDEN_DIM,  # Correctly saved per run
                "dropout": DROPOUT_P,
                "bidirectional": BIDIRECTIONAL,
                
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "pos_weight": POS_WEIGHT_VALS,
                "seed": cfg.seed,
                "best_val_macro_f1": float(best_f1),
            }
            (cfg.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            
    print(f"Finished Exp (HDim={HIDDEN_DIM}). Best F1: {best_f1:.4f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config_tuned.toml")
    args, unknown = p.parse_known_args()

    cfg_toml = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg_toml = load_toml(cfg_path)

    data_root = deep_get(cfg_toml, "paths.data_root", None)
    if data_root is None:
         raise RuntimeError("Missing data_root in config")
    
    metadata_csv_name = deep_get(cfg_toml, "cvs.data.metadata_csv", "all_metadata.csv")
    
    # We will use this base dir and append subfolders in run_experiment
    base_out_dir = deep_get(cfg_toml, "cvs.train.out_dir", "runs/cvs_tuned")
    
    seq_len = int(deep_get(cfg_toml, "cvs.data.seq_len", 5))
    epochs = int(deep_get(cfg_toml, "cvs.train.epochs", 25))
    batch_size = int(deep_get(cfg_toml, "cvs.train.batch_size", 8)) 
    lr = float(deep_get(cfg_toml, "cvs.train.lr", 1e-4))
    weight_decay = float(deep_get(cfg_toml, "cvs.train.weight_decay", 1e-4))
    image_size = int(deep_get(cfg_toml, "cvs.train.image_size", 224))
    
    # Common Config Object
    common_cfg = Cfg(
        data_root=Path(data_root),
        metadata_csv=Path(data_root) / metadata_csv_name,
        out_dir=Path(base_out_dir), 
        train_split=deep_get(cfg_toml, "cvs.data.train_split", "train"),
        val_split=deep_get(cfg_toml, "cvs.data.val_split", "val"),
        epochs=epochs,
        batch_size=batch_size,
        num_workers=4,
        lr=lr,
        weight_decay=weight_decay,
        amp=True,
        freeze_backbone=False,
        seed=42,
        image_size=image_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        keyframes_only=True,
        filename_pattern="{vid}_{frame}",
        threshold=0.6,
        config_path=str(cfg_path),
        seq_len=seq_len,
    )

    # --- DEFINE THE EXPERIMENTS HERE ---
    experiments = [
        {"hidden_dim": 512},
        {"hidden_dim": 128},

    ]

    print(f"Found {len(experiments)} experiments to run sequentially.")
    
    for i, exp in enumerate(experiments):
        print(f"\n>>> Running Experiment {i+1}/{len(experiments)}: {exp}")
        run_experiment(exp, common_cfg)
        
    print("\nALL EXPERIMENTS COMPLETED.")

if __name__ == "__main__":
    main()