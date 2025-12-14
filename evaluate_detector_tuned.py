import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from config_utils import load_toml, deep_get


def collate_fn(batch):
    return tuple(zip(*batch))

def _clamp_(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return x.clamp(min=lo, max=hi)

def sanitize_target_xyxy(target: Dict[str, Any], image_w: int, image_h: int, min_size: float = 1.0) -> Dict[str, Any]:
    boxes = target["boxes"]
    if boxes.numel() == 0: return target
    x1 = boxes[:, 0].clamp(0, image_w - 1)
    y1 = boxes[:, 1].clamp(0, image_h - 1)
    x2 = boxes[:, 2].clamp(0, image_w - 1)
    y2 = boxes[:, 3].clamp(0, image_h - 1)
    
    keep = ((x2 - x1) >= min_size) & ((y2 - y1) >= min_size)
    if keep.sum() == 0:
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0,), dtype=torch.int64)
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        return target
        
    target["boxes"] = torch.stack([x1, y1, x2, y2], dim=1)[keep]
    target["labels"] = target["labels"][keep]
    target["area"] = target["area"][keep]
    target["iscrowd"] = target["iscrowd"][keep]
    return target

class EndoscapesCocoDetection(Dataset):
    def __init__(self, images_dir: Path, ann_path: Path, fixed_label_map: Optional[Dict] = None):
        self.images_dir = images_dir
        self.coco = COCO(str(ann_path))
        self.img_ids = sorted(self.coco.getImgIds())
        
        # --- FIX: FORCE CONSISTENT MAPPING ---
        if fixed_label_map:
            # Reconstruct map from meta.json (keys are usually strings in JSON)
            self.cat_id_to_label = {int(k): int(v) for k, v in fixed_label_map.items()}
            self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}
            print(f"Loaded Fixed Label Map (Size {len(self.cat_id_to_label)})")
        else:
            # Fallback (Dangerous for Test sets)
            print("WARNING: Auto-generating label map. This may cause ID mismatches on Test sets!")
            cat_ids = sorted(self.coco.getCatIds())
            self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
            self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        img_np = np.array(img)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        _, H, W = img_t.shape
        
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0: continue
            
            # Use strict mapping check
            cat_id = a["category_id"]
            if cat_id not in self.cat_id_to_label:
                continue # Skip unknown classes (unlikely in Test, but safe)
                
            boxes.append([x, y, w, h])
            labels.append(self.cat_id_to_label[cat_id])
            areas.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        if not boxes:
            target = {"boxes": torch.zeros((0,4)), "labels": torch.zeros((0,), dtype=torch.int64),
                      "image_id": torch.tensor([img_id]), "area": torch.zeros((0,)), "iscrowd": torch.zeros((0,))}
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            # xywh -> xyxy
            boxes_t[:, 2] += boxes_t[:, 0]
            boxes_t[:, 3] += boxes_t[:, 1]
            target = {
                "boxes": boxes_t, "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([img_id]), "area": torch.tensor(areas),
                "iscrowd": torch.tensor(iscrowd)
            }
        target = sanitize_target_xyxy(target, W, H)
        return img_t, target

def build_model_from_meta(num_classes: int, train_meta: Dict):
    backbone_name = train_meta.get("backbone", "resnet50")
    min_size = int(train_meta.get("img_min_size", 800))
    max_size = int(train_meta.get("img_max_size", 1333))
    
    print(f"Building Model: {backbone_name} | Min: {min_size} Max: {max_size}")

    anchor_sizes = train_meta.get("anchor_sizes", None)
    aspect_ratios = train_meta.get("aspect_ratios", None)
    rpn_anchor_generator = None
    
    if anchor_sizes is not None and aspect_ratios is not None:
        print(f"-> Restoring Custom Anchors: {anchor_sizes}")
        if isinstance(anchor_sizes[0], (int, float)):
             anchor_sizes = tuple((s,) for s in anchor_sizes)
        else:
             anchor_sizes = tuple(tuple(s) for s in anchor_sizes)
        if isinstance(aspect_ratios[0], (int, float)):
             aspect_ratios = tuple((r,) for r in aspect_ratios)
        else:
             aspect_ratios = tuple(tuple(r) for r in aspect_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    if "resnet101" in backbone_name:
        backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=None)
        model = FasterRCNN(backbone, num_classes=num_classes, 
                           min_size=min_size, max_size=max_size,
                           rpn_anchor_generator=rpn_anchor_generator)
        return model

    model = fasterrcnn_resnet50_fpn(weights=None, min_size=min_size, max_size=max_size,
                                    rpn_anchor_generator=rpn_anchor_generator)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

@torch.no_grad()
def coco_eval_map(model, dataset, device, batch_size, num_workers):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    results = []
    
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            boxes = out["boxes"].cpu()
            scores = out["scores"].cpu()
            labels = out["labels"].cpu()
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            
            for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                # Use label_to_cat_id to reverse the model's prediction back to COCO ID
                cat_id = dataset.label_to_cat_id.get(int(l))
                if cat_id:
                    results.append({"image_id": img_id, "category_id": cat_id, "bbox": b, "score": s})
    
    if not results: return {"map": 0.0, "stats": []}
    
    if "info" not in dataset.coco.dataset: dataset.coco.dataset["info"] = {"description": "Endoscapes"}
    if "licenses" not in dataset.coco.dataset: dataset.coco.dataset["licenses"] = []

    coco_dt = dataset.coco.loadRes(results)
    coco_eval = COCOeval(dataset.coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return {
        "map": float(coco_eval.stats[0]),
        "map_50": float(coco_eval.stats[1]),
        "map_75": float(coco_eval.stats[2]),
        "map_small": float(coco_eval.stats[3]),
        "map_medium": float(coco_eval.stats[4]),
        "map_large": float(coco_eval.stats[5])
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config_tuned.toml")
    p.add_argument("--save_path", type=str, default=None)
    args = p.parse_args()

    cfg = load_toml(args.config)
    data_root = Path(deep_get(cfg, "paths.data_root"))
    
    split = deep_get(cfg, "eval.detector.split", "test")
    ann_name = deep_get(cfg, "eval.detector.ann_name", "annotation_coco.json")
    ckpt = deep_get(cfg, "eval.detector.ckpt")
    meta_path = deep_get(cfg, "eval.detector.meta")
    
    print(f"Loading metadata from: {meta_path}")
    train_meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    
    # --- CRITICAL FIX: Extract Label Map from Training Meta ---
    # JSON stores keys as strings "1": 1. We must convert them if needed.
    fixed_map = train_meta.get("cat_id_to_label", None)
    
    # Pass fixed_map to Dataset so it matches training EXACTLY
    ds = EndoscapesCocoDetection(
        data_root / split, 
        data_root / split / ann_name,
        fixed_label_map=fixed_map 
    )
    
    # Use num_classes from the Dataset (which now matches Training)
    num_classes = len(ds.cat_id_to_label) + 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_meta(num_classes, train_meta)
    
    print(f"Loading checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    
    batch_size = int(deep_get(cfg, "eval.detector.batch_size", 2))
    metrics = coco_eval_map(model, ds, device, batch_size=batch_size, num_workers=2)
    
    result = {
        "eval_split": split,
        "eval_map": metrics["map"],
        "eval_map_50": metrics["map_50"],
        "eval_map_75": metrics["map_75"],
        "eval_map_small": metrics["map_small"],
        "eval_map_medium": metrics["map_medium"],
        "eval_map_large": metrics["map_large"],
        "eval_checkpoint": ckpt,
        "training_meta": train_meta
    }

    print("-" * 40)
    print(f"Eval Split: {split} | Backbone: {train_meta.get('backbone')}")
    print(f"mAP (0.5:0.95): {metrics['map']:.4f}")
    print(f"mAP (Small):    {metrics['map_small']:.4f}")
    print("-" * 40)

    if args.save_path:
        sp = Path(args.save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if sp.exists():
            try:
                history = json.loads(sp.read_text(encoding="utf-8"))
                if not isinstance(history, list): history = [history]
            except: history = []
        history.append(result)
        with open(sp, "w", encoding="utf-8") as f: json.dump(history, f, indent=2)
        print(f"Result appended to: {sp}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()