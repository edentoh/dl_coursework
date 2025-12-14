import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50, resnet101
from pycocotools.coco import COCO

from matplotlib.patches import FancyBboxPatch
import textwrap

from config_utils import load_toml, deep_get

# ---------------------------------------------------------
# 1. Model Builders
# ---------------------------------------------------------
def build_detector(num_classes: int, meta: Dict[str, Any]):
    backbone_name = meta.get("backbone", "resnet50")
    min_size = int(meta.get("img_min_size", 800))
    max_size = int(meta.get("img_max_size", 1333))
    
    print(f"Loading Detector: {backbone_name} (Res: {min_size}-{max_size})")
    
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
        
        print(f"Loading CVS Model: {backbone_name} + LSTM")
        
        if "resnet101" in backbone_name:
            resnet = resnet101(weights=None)
        else:
            resnet = resnet50(weights=None)
            
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = resnet.fc.in_features
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=bidirectional)
        fc_input = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x_flat).flatten(1)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        return self.fc(lstm_out[:, -1, :])

def build_cvs(meta: Dict[str, Any], out_dim: int = 3):
    model_type = meta.get("model_type", "simple")
    if "temporal" in model_type or "lstm" in model_type:
        return TemporalCVSModel(meta, num_classes=out_dim)
    else:
        print("Loading CVS Model: Baseline ResNet50")
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, out_dim)
        return model

# ---------------------------------------------------------
# 2. LLM Text Generator
# ---------------------------------------------------------
def generate_llm_text(det_results, cvs_probs, cat_id_to_name):
    """Translates numerical outputs into a surgical report prompt."""
    det_lines = []
    if len(det_results["scores"]) > 0:
        for cid, score in zip(det_results["labels"], det_results["scores"]):
            name = cat_id_to_name.get(cid, f"Object_{cid}")
            det_lines.append(f"- {name} (Confidence: {score:.1%})")
        det_summary = "Detected Anatomical Structures:\n" + "\n".join(det_lines)
    else:
        det_summary = "No specific anatomical structures or tools were detected in this view."

    criteria_names = [
        "Hepatocystic Triangle Cleared",
        "Liver Bed Separated",
        "Two Structures Only"
    ]
    
    cvs_lines = []
    for i, prob in enumerate(cvs_probs):
        status = "SATISFIED" if prob > 0.7 else "NOT SATISFIED"
        confidence_desc = "High" if prob > 0.8 or prob < 0.2 else "Moderate"
        cvs_lines.append(f"Criterion {i+1} ({criteria_names[i]}): {status} ({prob:.1%} probability)")
    
    cvs_summary = "Critical View of Safety (CVS) Assessment:\n" + "\n".join(cvs_lines)

    llm_prompt = (
        "You are an AI Surgical Assistant. Analyze the following computer vision report from a laparoscopic cholecystectomy:\n\n"
        f"{det_summary}\n\n"
        f"{cvs_summary}\n\n"
        "TASK: Provide a concise 2-sentence warning or confirmation to the surgeon regarding safety."
    )
    return llm_prompt

# ---------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------
def draw_results(img, det_res, cvs_probs, cat_id_to_name, save_path):
    # 2-row layout: image on top, CVS panel below (prevents overlap/clipping)
    fig, (ax_img, ax_txt) = plt.subplots(
        2, 1,
        figsize=(12, 11),
        gridspec_kw={"height_ratios": [12, 2]},
        constrained_layout=True
    )

    # --- Image panel ---
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("Surgical AI Prediction", fontsize=16, pad=12)

    # 1) Draw detector boxes
    boxes = det_res.get("boxes", [])
    labels = det_res.get("labels", [])
    scores = det_res.get("scores", [])

    if len(boxes) > 0:
        cmap = plt.get_cmap("tab10")
        for i, box in enumerate(boxes):
            cid = int(labels[i]) if i < len(labels) else -1
            score = float(scores[i]) if i < len(scores) else 0.0
            label = cat_id_to_name.get(cid, str(cid))
            color = cmap(cid % 10) if cid >= 0 else (1, 1, 1, 1)

            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, linewidth=2.5, edgecolor=color
            )
            ax_img.add_patch(rect)

            # Put label INSIDE the box near the top-left
            text_str = f"{label}: {score:.2f}"
            ax_img.text(
                x1, y1 + 12, text_str,
                color="white", fontsize=10, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=2),
                va="top"
            )

    # --- CVS panel ---
    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)

    ax_txt.text(
        0.5, 0.88, "CRITICAL VIEW OF SAFETY PROBABILITIES",
        ha="center", va="center", fontsize=14, fontweight="bold"
    )

    # One panel box
    panel = FancyBboxPatch(
        (0.03, 0.12), 0.94, 0.62,
        boxstyle="round,pad=0.02",
        linewidth=1.5, edgecolor="black", facecolor="white", alpha=0.95,
        transform=ax_txt.transAxes
    )
    ax_txt.add_patch(panel)

    c_colors = ["darkgreen" if p > 0.5 else "darkred" for p in cvs_probs[:3]]

    lines = [
        f"1. Hepatocystic triangle cleared of all fat and fibrous tissue: {cvs_probs[0]:.0%}",
        f"2. Lower third of the gallbladder separated from the liver bed: {cvs_probs[1]:.0%}",
        f"3. Only two structures seen entering the gallbladder (duct and artery): {cvs_probs[2]:.0%}",
    ]

    # Wrap long lines
    wrapped = [textwrap.fill(s, width=120) for s in lines]

    # Place from top to bottom
    y = 0.70
    dy = 0.22
    for s, col in zip(wrapped, c_colors):
        ax_txt.text(
            0.06, y, s,
            ha="left", va="top",
            fontsize=12, fontweight="bold",
            color=col,
            transform=ax_txt.transAxes
        )
        y -= dy

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")
    plt.show()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="predict_one.toml")
    args = p.parse_args()
    
    cfg = load_toml(args.config)
    data_root = Path(deep_get(cfg, "paths.data_root"))
    
    # 1. Image Selection Logic
    ann_file = data_root / deep_get(cfg, "prediction.annotation_file")
    if not ann_file.exists(): raise FileNotFoundError(f"Annotation file missing: {ann_file}")
    
    coco = COCO(str(ann_file))
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    
    pick_random = deep_get(cfg, "prediction.pick_random", False)
    
    if pick_random:
        img_ids = coco.getImgIds()
        rand_id = random.choice(img_ids)
        img_info = coco.loadImgs([rand_id])[0]
        # Try finding image in typical folders
        possible_folders = ["test", "train", "val", "."]
        found = False
        for folder in possible_folders:
            img_path = data_root / folder / img_info["file_name"]
            if img_path.exists():
                found = True
                break
        if not found:
             # Just use data_root if structure unknown
             img_path = data_root / img_info["file_name"]
    else:
        img_path = data_root / deep_get(cfg, "prediction.image")
    
    if not img_path.exists():
         # Last ditch effort: search recursively
         found_list = list(data_root.rglob(img_path.name))
         if found_list: img_path = found_list[0]
         else: raise FileNotFoundError(f"Could not find image: {img_path}")

    print(f"\nProcessing Image: {img_path.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Models
    det_meta = json.loads(Path(deep_get(cfg, "prediction.det_meta")).read_text())
    det_model = build_detector(len(cat_id_to_name)+1, det_meta)
    det_model.load_state_dict(torch.load(deep_get(cfg, "prediction.det_ckpt"), map_location="cpu", weights_only=True))
    det_model.to(device).eval()
    
    cvs_meta = json.loads(Path(deep_get(cfg, "prediction.cvs_meta")).read_text())
    cvs_model = build_cvs(cvs_meta)
    cvs_model.load_state_dict(torch.load(deep_get(cfg, "prediction.cvs_ckpt"), map_location="cpu", weights_only=True))
    cvs_model.to(device).eval()

    # 3. Process Image
    pil_img = Image.open(img_path).convert("RGB")
    
    # Detector Input
    det_t = T.ToTensor()(pil_img).to(device)
    
    # CVS Input (Read size from meta!)
    cvs_img_size = int(cvs_meta.get("image_size", 224)) # <--- FIXED
    print(f"CVS Input Size from Meta: {cvs_img_size}x{cvs_img_size}")

    cvs_tfms = [T.Resize((cvs_img_size, cvs_img_size)), T.ToTensor()]
    
    # Handle Normalization if tuned
    is_tuned_cvs = "baseline" not in cvs_meta.get("model_type", "simple")
    if is_tuned_cvs:
        cvs_tfms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    cvs_t = T.Compose(cvs_tfms)(pil_img).to(device)
    
    # Handle Temporal seq_len
    if "temporal" in cvs_meta.get("model_type", ""):
        seq_len = int(cvs_meta.get("seq_len", 1))
        cvs_t = cvs_t.unsqueeze(0).unsqueeze(0).repeat(1, seq_len, 1, 1, 1)
    else:
        cvs_t = cvs_t.unsqueeze(0)

    # 4. Inference
    with torch.no_grad():
        det_out = det_model([det_t])[0]
        cvs_logits = cvs_model(cvs_t)
        if cvs_logits.dim() > 2: cvs_logits = cvs_logits[:, -1, :]
        cvs_probs = torch.sigmoid(cvs_logits).cpu().squeeze().tolist()

    # 5. Filter Results
    det_thr = deep_get(cfg, "prediction.det_thr", 0.5)
    keep = det_out["scores"] > det_thr
    
    final_res = {
        "boxes": det_out["boxes"][keep].cpu().numpy(),
        "scores": det_out["scores"][keep].cpu().numpy(),
        "labels": det_out["labels"][keep].cpu().numpy()
    }

    # 6. Generate Text Outputs
    llm_prompt = generate_llm_text(final_res, cvs_probs, cat_id_to_name)
    
    print("\n" + "="*60)
    print(" GENERATED TEXT OUTPUT FOR LLM")
    print("="*60)
    print(llm_prompt)
    print("="*60 + "\n")

    # 7. Visualize
    save_dir = Path(deep_get(cfg, "prediction.save_dir"))
    save_dir.mkdir(parents=True, exist_ok=True)
    draw_results(pil_img, final_res, cvs_probs, cat_id_to_name, save_dir / f"pred_{img_path.stem}.png")

if __name__ == "__main__":
    main()