import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50, resnet101

from config_utils import load_toml, deep_get


# ---------------------------------------------------------
# Model Builders (Updated for Tuned Support)
# ---------------------------------------------------------
def build_detector_model(num_classes: int, backbone_name: str = "resnet50", min_size: int = 640, max_size: int = 1024):
    if backbone_name == "resnet101":
        print(f"Building Faster R-CNN with {backbone_name}...")
        backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=None)
        model = FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size)
    else:
        # Default ResNet-50
        print(f"Building Faster R-CNN with {backbone_name}...")
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        except:
            weights = "DEFAULT"
        model = fasterrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

class TemporalCVSModel(nn.Module):
    """Same class as in train_cvs_tuned.py"""
    def __init__(self, num_classes=3, hidden_dim=256):
        super().__init__()
        resnet = resnet101(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(resnet.fc.in_features, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x).flatten(1).view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        return self.fc(lstm_out[:, -1, :])

def build_cvs_model(model_type: str = "simple", out_dim: int = 3):
    if "temporal" in model_type or "lstm" in model_type:
        print(f"Building Temporal CVS Model ({model_type})...")
        return TemporalCVSModel(num_classes=out_dim)
    else:
        print("Building Baseline CVS Model (ResNet50)...")
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_dim)
        return model


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def find_image_id_by_filename(coco: COCO, filename: str) -> Optional[int]:
    for img in coco.dataset.get("images", []):
        if img.get("file_name") == filename:
            return int(img["id"])
    return None


def get_cvs_gt(filename: str, csv_path: Path) -> Tuple[str, List[int]]:
    if not csv_path.exists():
        return "CSV Not Found", []
    try:
        stem = Path(filename).stem
        parts = re.split(r'[_\-]', stem)
        if len(parts) < 2: return "Filename Parse Error", []
        vid, frame = int(parts[0]), int(parts[1])

        df = pd.read_csv(csv_path)
        row = df[(df['vid'] == vid) & (df['frame'] == frame)]
        if len(row) == 0: return "No CSV Entry", []
        
        row = row.iloc[0]
        c1, c2, c3 = int(row['C1']), int(row['C2']), int(row['C3'])
        return f"C1:{c1}  C2:{c2}  C3:{c3}", [c1, c2, c3]
    except Exception as e:
        return f"Lookup Error: {e}", []


def build_class_color_map(class_names: Dict[int, str]) -> Dict[int, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    ids = sorted(class_names.keys())
    colors = {}
    for i, cid in enumerate(ids):
        colors[cid] = cmap(i % 20)
    return colors


def draw_boxes(ax, boxes_xyxy, label_texts, class_ids, class_colors, linestyle, linewidth=2.2, text_alpha=0.75):
    for (x1, y1, x2, y2), txt, cid in zip(boxes_xyxy, label_texts, class_ids):
        color = class_colors.get(cid, (1, 1, 1, 1))
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=linewidth, linestyle=linestyle, edgecolor=color)
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 2), txt, fontsize=9, color="black", bbox=dict(facecolor=color, alpha=text_alpha, edgecolor="none", pad=2))


def draw_cvs_info(ax, title_prefix, filename, info_text, bg_color="black"):
    full_text = f"{title_prefix} | {filename}\n{info_text}"
    ax.text(10, 10, full_text, fontsize=12, color="white", verticalalignment='top', bbox=dict(facecolor=bg_color, alpha=0.7, edgecolor="none", pad=6))


def _as_path_or_none(x: Optional[str]) -> Optional[Path]:
    if x is None: return None
    x = str(x).strip()
    if x == "" or x.lower() in ["none", "null"]: return None
    return Path(x)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml", help="Path to config.toml")
    p.add_argument("--data_root", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--det_ckpt", default=None)
    p.add_argument("--det_meta", default=None)
    p.add_argument("--cvs_ckpt", default=None)
    p.add_argument("--cvs_meta", default=None)
    p.add_argument("--metadata_csv", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--pick_annotated", action="store_true")
    p.add_argument("--score_thr", type=float, default=None)
    p.add_argument("--save_dir", default=None)
    args = p.parse_args()

    cfg_toml = {}
    if Path(args.config).exists():
        cfg_toml = load_toml(args.config)

    data_root = Path(args.data_root or deep_get(cfg_toml, "paths.data_root"))
    split = args.split or deep_get(cfg_toml, "predict.detector.split", "test")
    split_dir = data_root / split
    ann_path = split_dir / deep_get(cfg_toml, "predict.detector.ann_name", "annotation_coco.json")
    csv_path = data_root / (args.metadata_csv or deep_get(cfg_toml, "cvs.data.metadata_csv", "all_metadata.csv"))

    # Configs
    det_ckpt = args.det_ckpt or deep_get(cfg_toml, "eval.detector.ckpt")
    det_meta_path = args.det_meta or deep_get(cfg_toml, "eval.detector.meta")
    cvs_ckpt = args.cvs_ckpt or deep_get(cfg_toml, "eval.cvs.ckpt")
    cvs_meta_path = args.cvs_meta or deep_get(cfg_toml, "eval.cvs.meta")
    
    score_thr = args.score_thr if args.score_thr is not None else float(deep_get(cfg_toml, "predict.detector.score_thr", 0.5))
    save_dir = args.save_dir or deep_get(cfg_toml, "predict.detector.save_dir", "runs/predict_vis")

    # Load Annotations
    coco = COCO(str(ann_path))
    cat_ids = sorted(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    label_to_cat_id = {i + 1: cid for i, cid in enumerate(cat_ids)}
    class_colors = build_class_color_map(cat_id_to_name)

    # Select Image
    cfg_image = deep_get(cfg_toml, "predict.detector.image", "")
    image_arg = args.image if args.image else (cfg_image if str(cfg_image).strip() != "" else None)
    pick_annotated = args.pick_annotated or bool(deep_get(cfg_toml, "predict.detector.pick_annotated", False))

    if pick_annotated:
        img_id = int(np.random.choice(coco.getImgIds()))
        info = coco.loadImgs([img_id])[0]
        img_path = split_dir / info["file_name"]
    else:
        if not image_arg: raise RuntimeError("Provide --image or use --pick_annotated")
        img_path = split_dir / image_arg if not Path(image_arg).exists() else Path(image_arg)
        if not img_path.exists(): raise FileNotFoundError(f"Image not found: {img_path}")
        img_id = find_image_id_by_filename(coco, img_path.name)

    print(f"Processing: {img_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- LOAD DETECTOR ----
    if det_ckpt:
        backbone_name = "resnet50"
        if det_meta_path and Path(det_meta_path).exists():
            dm = json.loads(Path(det_meta_path).read_text(encoding='utf-8'))
            backbone_name = dm.get("backbone", "resnet50") # Check meta for resnet101
        
        det_model = build_detector_model(len(cat_ids)+1, backbone_name=backbone_name)
        det_model.load_state_dict(torch.load(det_ckpt, map_location="cpu", weights_only=True))
        det_model.to(device)
        det_model.eval()
    else:
        raise RuntimeError("No detector checkpoint found.")

    # ---- LOAD CVS ----
    cvs_model = None
    cvs_type = "simple"
    seq_len = 1
    if cvs_ckpt:
        if cvs_meta_path and Path(cvs_meta_path).exists():
            cm = json.loads(Path(cvs_meta_path).read_text(encoding='utf-8'))
            cvs_type = cm.get("model_type", "simple")
            seq_len = cm.get("seq_len", 1)
        
        cvs_model = build_cvs_model(model_type=cvs_type, out_dim=3)
        cvs_model.load_state_dict(torch.load(cvs_ckpt, map_location="cpu", weights_only=True))
        cvs_model.to(device)
        cvs_model.eval()

    # ---- PREDICT ----
    pil_img = Image.open(img_path).convert("RGB")
    
    # 1. Detector Prediction
    img_t = T.ToTensor()(pil_img).to(device)
    with torch.no_grad():
        det_out = det_model([img_t])[0]

    boxes = det_out["boxes"].cpu()
    scores = det_out["scores"].cpu()
    labels = det_out["labels"].cpu()
    keep = scores >= score_thr
    pred_boxes = boxes[keep].numpy().tolist()
    pred_scores = scores[keep].tolist()
    pred_labels = labels[keep].tolist()
    
    pred_box_texts = [f"{cat_id_to_name.get(label_to_cat_id.get(int(l)), '?')} {s:.2f}" for l, s in zip(pred_labels, pred_scores)]
    pred_box_cids = [label_to_cat_id.get(int(l)) for l in pred_labels]

    # 2. CVS Prediction
    cvs_pred_str = "CVS Model Not Loaded"
    if cvs_model:
        # Preprocess
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        cvs_t = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])(pil_img).to(device)
        
        # Handle Temporal Input
        if "temporal" in cvs_type or "lstm" in cvs_type:
            # We only have 1 image, but model needs [B, T, C, H, W]
            # We replicate the image T times to mimic a static video
            cvs_input = cvs_t.unsqueeze(0).unsqueeze(0).repeat(1, seq_len, 1, 1, 1) # [1, 5, 3, 224, 224]
        else:
            # Simple model [B, C, H, W]
            cvs_input = cvs_t.unsqueeze(0)

        with torch.no_grad():
            logits = cvs_model(cvs_input)[0]
            probs = torch.sigmoid(logits)
        
        p = probs.cpu().tolist()
        preds = [1 if x >= 0.5 else 0 for x in p]
        cvs_pred_str = f"C1:{preds[0]} ({p[0]:.2f})  C2:{preds[1]} ({p[1]:.2f})  C3:{preds[2]} ({p[2]:.2f})"

    # ---- VISUALIZE ----
    gt_boxes, gt_box_texts, gt_box_cids = [], [], []
    if img_id is not None:
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            x, y, w, h = a["bbox"]
            gt_boxes.append([x, y, x+w, y+h])
            cid = a["category_id"]
            gt_box_cids.append(cid)
            gt_box_texts.append(cat_id_to_name.get(cid, str(cid)))
    
    gt_cvs_str, _ = get_cvs_gt(img_path.name, csv_path)
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)

    print("\n--- PREDICTION SUMMARY ---")
    print(f"Image       : {img_path.name}")
    print(f"Model Det   : {backbone_name if det_ckpt else 'None'}")
    print(f"Model CVS   : {cvs_type if cvs_ckpt else 'None'}")
    print(f"GT Boxes    : {n_gt}")
    print(f"Pred Boxes  : {n_pred} (thr={score_thr})")
    print(f"GT CVS      : {gt_cvs_str}")
    print(f"Pred CVS    : {cvs_pred_str}")
    print("--------------------------")

    save_dir_p = Path(save_dir) if save_dir else None
    if save_dir_p: save_dir_p.mkdir(parents=True, exist_ok=True)

    fig1 = plt.figure(figsize=(12, 8))
    plt.imshow(pil_img)
    ax1 = plt.gca()
    if gt_boxes: draw_boxes(ax1, gt_boxes, gt_box_texts, gt_box_cids, class_colors, "--")
    draw_cvs_info(ax1, "GROUND TRUTH", img_path.name, f"CVS: {gt_cvs_str}\nBoxes: {n_gt}", "darkgreen")
    plt.axis("off"); plt.tight_layout()
    if save_dir_p: plt.savefig(save_dir_p / f"{img_path.stem}_GT.png", dpi=150, bbox_inches='tight')

    fig2 = plt.figure(figsize=(12, 8))
    plt.imshow(pil_img)
    ax2 = plt.gca()
    if pred_boxes: draw_boxes(ax2, pred_boxes, pred_box_texts, pred_box_cids, class_colors, "-")
    draw_cvs_info(ax2, "PREDICTION", img_path.name, f"CVS: {cvs_pred_str}\nBoxes: {n_pred}", "darkblue")
    plt.axis("off"); plt.tight_layout()
    if save_dir_p: plt.savefig(save_dir_p / f"{img_path.stem}_PRED.png", dpi=150, bbox_inches='tight')
    print(f"Saved visualizations to: {save_dir_p}")
    plt.show()

if __name__ == "__main__":
    main()