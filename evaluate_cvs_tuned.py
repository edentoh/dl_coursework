import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import resnet101

from config_utils import load_toml, deep_get

# -------------------------
# Re-use Temporal Logic
# -------------------------
def build_existing_file_index(split_dir: Path) -> Dict[str, str]:
    files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
    idx = {}
    for f in files:
        idx[f.name.lower()] = f.name
        idx[f.stem.lower()] = f.name
    return idx

def map_row_to_filename(row, idx, pattern):
    try:
        stem = pattern.format(vid=int(row["vid"]), frame=int(row["frame"])).strip().lower()
        return idx.get(stem)
    except: return None

class TemporalCVSDataset(Dataset):
    def __init__(self, split_dir: Path, df: pd.DataFrame, image_size: int, pattern: str, seq_len: int):
        self.split_dir = split_dir
        self.seq_len = seq_len
        self.idx = build_existing_file_index(split_dir)
        
        df = df.copy()
        df["_file"] = df.apply(lambda r: map_row_to_filename(r, self.idx, pattern), axis=1)
        self.df = df[df["_file"].notna() & df[["C1","C2","C3"]].notna().all(axis=1)].copy()
        
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
            img = Image.open(self.split_dir / r["_file"]).convert("RGB")
            imgs.append(self.tfms(img))
            
        x = torch.stack(imgs, dim=0)
        y = torch.tensor([row['C1'], row['C2'], row['C3']], dtype=torch.float32)
        return x, y

class TemporalCVSModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256):
        super().__init__()
        resnet = resnet101(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(resnet.fc.in_features, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.feature_extractor(x).flatten(1).view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        return self.fc(lstm_out[:, -1, :])

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

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_tuned.toml")
    args = p.parse_args()
    
    cfg = load_toml(args.config)
    data_root = Path(deep_get(cfg, "paths.data_root"))
    
    split = deep_get(cfg, "eval.cvs.split", "test")
    ckpt = deep_get(cfg, "eval.cvs.ckpt")
    meta_path = deep_get(cfg, "eval.cvs.meta")
    threshold = float(deep_get(cfg, "eval.cvs.threshold", 0.6))
    
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    seq_len = int(meta.get("seq_len", 5))
    
    # Dataset
    csv_path = data_root / meta.get("metadata_csv", "all_metadata.csv")
    df = pd.read_csv(csv_path)
    if meta.get("keyframes_only"):
        df = df[df["is_ds_keyframe"].astype(str).str.upper().isin(["TRUE","1","YES"])]
        
    ds = TemporalCVSDataset(data_root / split, df, meta.get("image_size", 224), 
                            meta.get("filename_pattern", "{vid}_{frame}"), seq_len)
    
    loader = DataLoader(ds, batch_size=8, num_workers=4)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalCVSModel()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    model.eval()
    
    all_logits, all_y = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_y.append(y)
        
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y)
    
    macro, per = compute_metrics(logits, y_true, threshold)
    
    print(json.dumps({
        "split": split,
        "macro_f1": macro,
        "per_criterion": per,
        "threshold": threshold,
        "model": "Temporal (LSTM)"
    }, indent=2))

if __name__ == "__main__":
    main()