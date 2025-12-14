import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import resnet50, resnet101
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config_utils import load_toml, deep_get


# ---------------------------------------------------------
# Dynamic Model Builders (Reads from Meta)
# ---------------------------------------------------------
def build_detector_model(num_classes: int, meta: Dict[str, Any]):
    """Builds detector based on training metadata."""
    backbone_name = meta.get("backbone", "resnet50")
    min_size = int(meta.get("img_min_size", 800))
    max_size = int(meta.get("img_max_size", 1333))
    
    print(f"Building Detector -> {backbone_name} | Res: {min_size}/{max_size}")
    
    if "resnet101" in backbone_name:
        backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=None)
    else:
        backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None)
        
    model = FasterRCNN(backbone, num_classes=num_classes, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


class TemporalCVSModel(nn.Module):
    """Universal Temporal Model matching train_cvs_tuned.py"""
    def __init__(self, meta: Dict[str, Any], num_classes=3):
        super().__init__()
        
        backbone_name = meta.get("backbone", "resnet50")
        hidden_dim = int(meta.get("hidden_dim", 256))
        bidirectional = bool(meta.get("bidirectional", True))
        
        if "resnet101" in backbone_name:
            resnet = resnet101(weights=None)
        else:
            resnet = resnet50(weights=None)
            
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = resnet.fc.in_features

        self.lstm = nn.LSTM(
            input_size=self.feat_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        fc_input = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x_flat).flatten(1)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


def build_cvs_model(meta: Dict[str, Any], out_dim: int = 3):
    model_type = meta.get("model_type", "simple")
    
    if "temporal" in model_type or "lstm" in model_type:
        return TemporalCVSModel(meta, num_classes=out_dim)
    else:
        from torchvision.models import resnet50
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


# --- RESTORED ORIGINAL BOX DRAWING (Labels ON boxes) ---
def draw_boxes(ax, boxes_xyxy, label_texts, class_ids, class_colors, linestyle, linewidth=2.2, text_alpha=0.75):
    """Draws boxes and places labels directly on top of them."""
    for (x1, y1, x2, y2), txt, cid in zip(boxes_xyxy, label_texts, class_ids):
        color = class_colors.get(cid, (1, 1, 1, 1))
        
        # Draw Rectangle
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=linewidth, linestyle=linestyle, edgecolor=color)
        ax.add_patch(rect)
        
        # Draw Label (On the box)
        ax.text(x1, max(0, y1 - 2), txt, fontsize=9, color="black", 
                bbox=dict(facecolor=color, alpha=text_alpha, edgecolor="none", pad=2))


def run_detector_inference(model, pil_img, device, score_thr, cat_id_to_name, label_to_cat_id):
    img_t = T.ToTensor()(pil_img).to(device)
    with torch.no_grad():
        det_out = model([img_t])[0]

    boxes = det_out["boxes"].cpu()
    scores = det_out["scores"].cpu()
    labels = det_out["labels"].cpu()
    
    keep = scores >= score_thr
    pred_boxes = boxes[keep].numpy().tolist()
    pred_scores = scores[keep].tolist()
    pred_labels = labels[keep].tolist()
    
    pred_box_texts = [f"{cat_id_to_name.get(label_to_cat_id.get(int(l)), '?')} {s:.2f}" for l, s in zip(pred_labels, pred_scores)]
    pred_box_cids = [label_to_cat_id.get(int(l)) for l in pred_labels]
    
    return pred_boxes, pred_box_texts, pred_box_cids


def run_cvs_inference(model, meta, pil_img, device, threshold):
    if model is None: return "Not Loaded"

    model_type = meta.get("model_type", "simple")
    image_size = int(meta.get("image_size", 224))
    
    is_baseline = "baseline" in model_type or "simple" in model_type
    
    transforms_list = [
        T.Resize((image_size, image_size)), 
        T.ToTensor()
    ]
    
    if not is_baseline: 
         norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         transforms_list.append(norm)

    cvs_t = T.Compose(transforms_list)(pil_img).to(device)
    
    if "temporal" in model_type or "lstm" in model_type:
        seq_len = int(meta.get("seq_len", 1))
        cvs_input = cvs_t.unsqueeze(0).unsqueeze(0).repeat(1, seq_len, 1, 1, 1) 
    else:
        cvs_input = cvs_t.unsqueeze(0)

    with torch.no_grad():
        logits = model(cvs_input)
        if logits.dim() > 2: logits = logits[:, -1, :] 
        probs = torch.sigmoid(logits).squeeze(0)
    
    p = probs.cpu().tolist()
    preds = [1 if x >= threshold else 0 for x in p]
    return f"C1:{preds[0]} ({p[0]:.2f})  C2:{preds[1]} ({p[1]:.2f})  C3:{preds[2]} ({p[2]:.2f})"


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="predict_comparison.toml", help="Path to config TOML")
    args = p.parse_args()

    cfg_toml = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg_toml = load_toml(cfg_path)
    else:
        print(f"Warning: Config file {cfg_path} not found. Using defaults.")

    data_root = Path(deep_get(cfg_toml, "paths.data_root"))
    
    split = deep_get(cfg_toml, "comparison.split", "test")
    split_dir = data_root / split
    ann_path = split_dir / "annotation_coco.json"
    csv_path = data_root / deep_get(cfg_toml, "comparison.metadata_csv", "all_metadata.csv")

    score_thr = float(deep_get(cfg_toml, "comparison.score_thr", 0.5))
    cvs_thr = float(deep_get(cfg_toml, "comparison.cvs_thr", 0.5))
    save_dir = deep_get(cfg_toml, "comparison.save_dir", "runs/comparison_vis")
    
    pick_annotated = bool(deep_get(cfg_toml, "comparison.pick_annotated", True))
    image_name = deep_get(cfg_toml, "comparison.image", "")

    if not ann_path.exists(): raise RuntimeError(f"Annotation file not found: {ann_path}")
    coco = COCO(str(ann_path))
    cat_ids = sorted(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    label_to_cat_id = {i + 1: cid for i, cid in enumerate(cat_ids)}
    class_colors = build_class_color_map(cat_id_to_name)
    num_classes = len(cat_ids) + 1

    if pick_annotated:
        print("Selecting random annotated image...")
        img_id = int(np.random.choice(coco.getImgIds()))
        info = coco.loadImgs([img_id])[0]
        img_path = split_dir / info["file_name"]
    elif image_name:
        img_path = split_dir / image_name
        if not img_path.exists(): raise FileNotFoundError(f"Image not found: {img_path}")
        img_id = find_image_id_by_filename(coco, img_path.name)
    else:
        raise RuntimeError("No image selected! Set 'pick_annotated = true' or provide 'image' in toml.")

    print(f"Processing: {img_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- LOAD MODELS ---
    models = {"baseline": {}, "tuned": {}}
    
    def load_pair(key, det_ckpt, det_meta_f, cvs_ckpt, cvs_meta_f):
        if det_ckpt and Path(det_ckpt).exists():
            meta = {}
            if det_meta_f and Path(det_meta_f).exists():
                meta = json.loads(Path(det_meta_f).read_text(encoding='utf-8'))
            model = build_detector_model(num_classes, meta)
            model.load_state_dict(torch.load(det_ckpt, map_location="cpu", weights_only=True))
            model.to(device).eval()
            models[key]["det"] = model
            models[key]["det_name"] = meta.get("backbone", "resnet50")
        else:
            print(f"Skipping {key} Detector (Not found)")
            models[key]["det"] = None

        if cvs_ckpt and Path(cvs_ckpt).exists():
            meta = {}
            if cvs_meta_f and Path(cvs_meta_f).exists():
                meta = json.loads(Path(cvs_meta_f).read_text(encoding='utf-8'))
            model = build_cvs_model(meta, out_dim=3)
            model.load_state_dict(torch.load(cvs_ckpt, map_location="cpu", weights_only=True))
            model.to(device).eval()
            models[key]["cvs"] = model
            models[key]["cvs_meta"] = meta
        else:
            print(f"Skipping {key} CVS (Not found)")
            models[key]["cvs"] = None

    load_pair("baseline", 
              deep_get(cfg_toml, "baseline.det_ckpt"), 
              deep_get(cfg_toml, "baseline.det_meta"), 
              deep_get(cfg_toml, "baseline.cvs_ckpt"), 
              deep_get(cfg_toml, "baseline.cvs_meta"))
              
    load_pair("tuned", 
              deep_get(cfg_toml, "tuned.det_ckpt"), 
              deep_get(cfg_toml, "tuned.det_meta"), 
              deep_get(cfg_toml, "tuned.cvs_ckpt"), 
              deep_get(cfg_toml, "tuned.cvs_meta"))

    # --- INFERENCE ---
    pil_img = Image.open(img_path).convert("RGB")
    
    base_res = {}
    if models["baseline"]["det"]:
        base_res["boxes"] = run_detector_inference(models["baseline"]["det"], pil_img, device, score_thr, cat_id_to_name, label_to_cat_id)
    if models["baseline"]["cvs"]:
        base_res["cvs_str"] = run_cvs_inference(models["baseline"]["cvs"], models["baseline"]["cvs_meta"], pil_img, device, cvs_thr)
    else:
        base_res["cvs_str"] = "N/A"

    tuned_res = {}
    if models["tuned"]["det"]:
        tuned_res["boxes"] = run_detector_inference(models["tuned"]["det"], pil_img, device, score_thr, cat_id_to_name, label_to_cat_id)
    if models["tuned"]["cvs"]:
        tuned_res["cvs_str"] = run_cvs_inference(models["tuned"]["cvs"], models["tuned"]["cvs_meta"], pil_img, device, cvs_thr)
    else:
        tuned_res["cvs_str"] = "N/A"

    gt_boxes, gt_box_texts, gt_box_cids = [], [], []
    if img_id is not None:
        for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            x, y, w, h = a["bbox"]
            gt_boxes.append([x, y, x+w, y+h])
            cid = a["category_id"]
            gt_box_cids.append(cid)
            gt_box_texts.append(cat_id_to_name.get(cid, str(cid)))
    gt_cvs_str, _ = get_cvs_gt(img_path.name, csv_path)

    # --- VISUALIZATION ---
    # Increased height (figsize) to make room for text at the bottom
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    # 1. Ground Truth
    axes[0].imshow(pil_img)
    if gt_boxes: draw_boxes(axes[0], gt_boxes, gt_box_texts, gt_box_cids, class_colors, "--")
    axes[0].set_title(f"GROUND TRUTH\n{img_path.name}", fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel(f"GT CVS:\n{gt_cvs_str}", fontsize=13, fontweight='bold', color='darkgreen')
    axes[0].set_xticks([]); axes[0].set_yticks([]) # Hide ticks but keep xlabel

    # 2. Baseline
    axes[1].imshow(pil_img)
    if "boxes" in base_res and base_res["boxes"]:
        draw_boxes(axes[1], base_res["boxes"][0], base_res["boxes"][1], base_res["boxes"][2], class_colors, "-")
    axes[1].set_title("BASELINE MODEL", fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel(f"PRED CVS:\n{base_res['cvs_str']}", fontsize=13, fontweight='bold', color='darkblue')
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # 3. Tuned
    axes[2].imshow(pil_img)
    if "boxes" in tuned_res and tuned_res["boxes"]:
        draw_boxes(axes[2], tuned_res["boxes"][0], tuned_res["boxes"][1], tuned_res["boxes"][2], class_colors, "-")
    axes[2].set_title("TUNED MODEL", fontsize=14, fontweight='bold', pad=10)
    axes[2].set_xlabel(f"PRED CVS:\n{tuned_res['cvs_str']}", fontsize=13, fontweight='bold', color='purple')
    axes[2].set_xticks([]); axes[2].set_yticks([])

    plt.tight_layout()
    
    save_dir_p = Path(save_dir)
    save_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = save_dir_p / f"compare_{img_path.stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {out_path}")
    
    # --- POP OUT THE WINDOW ---
    plt.show()

if __name__ == "__main__":
    main()