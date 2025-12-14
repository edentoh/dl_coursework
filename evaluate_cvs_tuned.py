import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
# Use standard ResNet50
from torchvision.models import resnet50, resnet101, resnet18

from config_utils import load_toml, deep_get

# ---------------------------------------------------------
# Utilities & Dataset
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
    def __init__(self, split_dir: Path, df: pd.DataFrame, image_size: int, pattern: str, seq_len: int):
        self.split_dir = split_dir
        self.seq_len = seq_len
        self.pattern = pattern
        self.existing_idx = build_existing_file_index(split_dir)
        
        df = df.copy()
        df["_mapped_file"] = df.apply(lambda r: map_row_to_existing_filename(r, self.existing_idx, pattern), axis=1)
        self.df = df[df["_mapped_file"].notna() & df[["C1","C2","C3"]].notna().all(axis=1)].copy()
        
        self.groups = dict(list(self.df.groupby("vid")))
        self.samples = self.df.to_dict('records')
        
        self.tfms = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        vid_df = self.groups[row['vid']].sort_values("frame")
        frames = vid_df[vid_df['frame'] <= row['frame']]
        
        if len(frames) >= self.seq_len:
            seq = frames.iloc[-self.seq_len:]
        else:
            pad = self.seq_len - len(frames)
            seq = pd.concat([frames.iloc[0:1]] * pad + [frames])
            
        imgs = []
        for _, r in seq.iterrows():
            img = Image.open(self.split_dir / r["_mapped_file"]).convert("RGB")
            imgs.append(self.tfms(img))
            
        x = torch.stack(imgs, dim=0)
        y = torch.tensor([row['C1'], row['C2'], row['C3']], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------
# Dynamic Model Building (Matches Training)
# ---------------------------------------------------------
class TemporalCVSModel(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=3, hidden_dim=256, bidirectional=True):
        super().__init__()
        
        # Load correct backbone
        if "resnet101" in backbone_name:
            resnet = resnet101(weights=None)
        elif "resnet18" in backbone_name:
            resnet = resnet18(weights=None)
        else:
            resnet = resnet50(weights=None)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = resnet.fc.in_features
        
        self.lstm = nn.LSTM(
            input_size=feat_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        # Inference only needs structure, dropout p doesn't matter (it's off in eval)
        self.dropout = nn.Dropout(p=0.5) 
        
        fc_input = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.feature_extractor(x).flatten(1).view(B, T, -1)
        
        lstm_out, _ = self.lstm(feats)
        last_out = lstm_out[:, -1, :]
        
        # Dropout is identity in eval(), but good to keep structure
        last_out = self.dropout(last_out)
        
        return self.fc(last_out)


def compute_metrics(logits, y_true, threshold):
    probs = torch.sigmoid(logits)
    y_pred = (probs >= threshold).int()
    y_true = (y_true >= 0.5).int()
    per = []
    for k in range(3):
        tp = ((y_pred[:, k]==1) & (y_true[:, k]==1)).sum().item()
        fp = ((y_pred[:, k]==1) & (y_true[:, k]==0)).sum().item()
        fn = ((y_pred[:, k]==0) & (y_true[:, k]==1)).sum().item()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        per.append({"precision": prec, "recall": rec, "f1": f1})
    macro = np.mean([d["f1"] for d in per])
    return macro, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_tuned.toml")
    p.add_argument("--save_path", default=None, help="Path to append results JSON")
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    
    cfg = load_toml(args.config)
    data_root = Path(deep_get(cfg, "paths.data_root"))
    
    # 1. Load Eval Config
    split = deep_get(cfg, "eval.cvs.split", "test")
    ckpt = deep_get(cfg, "eval.cvs.ckpt")
    meta_path = deep_get(cfg, "eval.cvs.meta")
    threshold = args.threshold if args.threshold is not None else float(deep_get(cfg, "eval.cvs.threshold", 0.6))
    
    if ckpt is None or meta_path is None:
        raise RuntimeError("Missing ckpt or meta path in config [eval.cvs]")

    # 2. Load Training Metadata
    print(f"Loading metadata from: {meta_path}")
    train_meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    
    # Extract structural params
    backbone_name = train_meta.get("backbone", "resnet50")
    hidden_dim = int(train_meta.get("hidden_dim", 256))
    bidirectional = bool(train_meta.get("bidirectional", True)) # Default to True based on your latest train
    seq_len = int(train_meta.get("seq_len", 5))
    image_size = int(train_meta.get("image_size", 224))
    
    print(f"Model Configuration: {backbone_name} | Seq: {seq_len} | Bidirectional: {bidirectional}")

    # 3. Setup Dataset
    csv_path = data_root / train_meta.get("metadata_csv", "all_metadata.csv")
    df = pd.read_csv(csv_path)
    if train_meta.get("keyframes_only"):
        df = df[df["is_ds_keyframe"].astype(str).str.upper().isin(["TRUE","1","YES"])]
        
    ds = TemporalCVSDataset(data_root / split, df, image_size, 
                            train_meta.get("filename_pattern", "{vid}_{frame}"), seq_len)
    
    batch_size = int(deep_get(cfg, "eval.cvs.batch_size", 8))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
    
    # 4. Build & Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalCVSModel(backbone_name=backbone_name, 
                             hidden_dim=hidden_dim, 
                             bidirectional=bidirectional)
    
    print(f"Loading checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    
    # 5. Run Evaluation
    all_logits, all_y = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_logits.append(logits.detach().cpu()) 
        all_y.append(y.detach().cpu())
        
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y)
    
    macro, per = compute_metrics(logits, y_true, threshold)
    
    # 6. Construct Final Output (Merging Evaluation + Training Meta)
    result = {
        # Evaluation Settings
        "eval_split": split,
        "eval_threshold": threshold,
        "eval_macro_f1": macro,
        "eval_per_criterion": per,
        "eval_checkpoint": ckpt,
        
        # Merged Training Metadata (Everything from meta.json)
        "training_meta": train_meta
    }
    
    # Print clean summary
    print("-" * 40)
    print(f"Eval Split: {split} | Threshold: {threshold}")
    print(f"Macro F1:   {macro:.4f}")
    print("-" * 40)
    
    # 7. Save / Append to File
    if args.save_path:
        sp = Path(args.save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        
        history = []
        if sp.exists():
            try:
                history = json.loads(sp.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = [history]
            except:
                history = []
        
        history.append(result)
        
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Result appended to: {sp}")
    else:
        # Just print JSON to stdout if no file specified
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()