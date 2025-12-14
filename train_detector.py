import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from config_utils import load_toml, deep_get

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    return tuple(zip(*batch))

def _clamp_(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return x.clamp(min=lo, max=hi)

def sanitize_target_xyxy(target: Dict[str, Any], image_w: int, image_h: int, min_size: float = 1.0) -> Dict[str, Any]:
    boxes = target["boxes"]
    if boxes.numel() == 0: return target

    x1 = _clamp_(boxes[:, 0], 0, image_w - 1)
    y1 = _clamp_(boxes[:, 1], 0, image_h - 1)
    x2 = _clamp_(boxes[:, 2], 0, image_w - 1)
    y2 = _clamp_(boxes[:, 3], 0, image_h - 1)

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

# ---------------------------------------------------------
# Augmentations
# ---------------------------------------------------------
class RandomHorizontalFlipDet:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, target):
        if random.random() > self.p: return img, target
        _, _, w = img.shape
        img = torch.flip(img, [2])
        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes[:, [0, 2]] = w - 1 - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return img, target

class ColorJitterSimple:
    def __init__(self, brightness=0.15, contrast=0.15):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img, target):
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - mean) * c + mean
        img = img * b
        img = img.clamp(0, 1)
        return img, target

class ComposeDet:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img, target):
        for t in self.transforms: img, target = t(img, target)
        return img, target

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class EndoscapesCocoDetection(Dataset):
    def __init__(self, images_dir: Path, ann_path: Path, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(str(ann_path))
        self.img_ids = sorted(self.coco.getImgIds())
        self.transforms = transforms
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        _, H, W = img_t.shape
        
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0: continue
            boxes.append([x, y, w, h])
            labels.append(self.cat_id_to_label[a["category_id"]])
            areas.append(a.get("area", w*h))
            iscrowd.append(a.get("iscrowd", 0))

        if not boxes:
            target = {"boxes": torch.zeros((0,4)), "labels": torch.zeros((0,), dtype=torch.int64),
                      "image_id": torch.tensor([img_id]), "area": torch.zeros((0,)), "iscrowd": torch.zeros((0,))}
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            # xywh -> xyxy
            boxes_t[:, 2] += boxes_t[:, 0]
            boxes_t[:, 3] += boxes_t[:, 1]
            target = {
                "boxes": boxes_t,
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
                "area": torch.tensor(areas),
                "iscrowd": torch.tensor(iscrowd)
            }

        target = sanitize_target_xyxy(target, W, H)
        if self.transforms:
            img_t, target = self.transforms(img_t, target)
            _, H2, W2 = img_t.shape
            target = sanitize_target_xyxy(target, W2, H2)
        return img_t, target

# ---------------------------------------------------------
# Training / Eval Loops
# ---------------------------------------------------------
def train_one_epoch(model, optimizer, loader, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_coco(model, dataset, device, num_workers=2):
    model.eval()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=num_workers)
    results = []
    
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        
        for out, tgt in zip(outputs, targets):
            img_id = tgt["image_id"].item()
            boxes = out["boxes"].cpu()
            scores = out["scores"].cpu()
            labels = out["labels"].cpu()
            
            # Convert XYXY (Model output) -> XYWH (COCO format)
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            
            for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                cat_id = dataset.label_to_cat_id.get(int(l))
                if cat_id:
                    results.append({
                        "image_id": img_id, 
                        "category_id": cat_id, 
                        "bbox": b, 
                        "score": s
                    })
    
    if not results: return 0.0
    
    if "info" not in dataset.coco.dataset:
        dataset.coco.dataset["info"] = {"description": "Endoscapes"}
    if "licenses" not in dataset.coco.dataset:
        dataset.coco.dataset["licenses"] = []

    coco_dt = dataset.coco.loadRes(results)
    coco_eval = COCOeval(dataset.coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0] # mAP 0.5:0.95

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_tuned.toml")
    args = p.parse_args()
    
    cfg = load_toml(args.config)
    # Output dir for this specific baseline experiment
    out_dir = Path(deep_get(cfg, "detector.train.out_dir", "runs/detector_cls_init"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Configuration ---
    epochs = int(deep_get(cfg, "detector.train.epochs", 30))
    lr = float(deep_get(cfg, "detector.train.lr", 0.005))
    batch_size = int(deep_get(cfg, "detector.train.batch_size", 4))
    weight_decay = float(deep_get(cfg, "detector.train.weight_decay", 0.0001))
    momentum = float(deep_get(cfg, "detector.train.momentum", 0.9))
    seed = int(deep_get(cfg, "detector.train.seed", 42))
    amp = bool(deep_get(cfg, "detector.train.amp", True))
    num_workers = int(deep_get(cfg, "detector.train.num_workers", 2))

    img_min_size = int(deep_get(cfg, "detector.model.img_min_size", 1200))
    img_max_size = int(deep_get(cfg, "detector.model.img_max_size", 1600))
    
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    data_root = Path(deep_get(cfg, "paths.data_root"))
    
    train_tfms = ComposeDet([
        RandomHorizontalFlipDet(p=0.5),
        ColorJitterSimple(brightness=0.15, contrast=0.15) 
    ])

    train_ds = EndoscapesCocoDetection(
        data_root/"train", 
        data_root/"train"/"annotation_coco.json", 
        transforms=train_tfms
    )
    val_ds = EndoscapesCocoDetection(data_root/"val", data_root/"val"/"annotation_coco.json")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    
    # --- MODEL BUILDING (CLASSIFICATION BACKBONE ONLY) ---
    print("Building Faster R-CNN (Initialised with ImageNet Backbone only)...")
    
    # 1. Load Backbone with ImageNet Weights
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights='DEFAULT')
    
    # 2. Build Faster R-CNN (Heads will be Randomly Initialized)
    num_classes = len(train_ds.cat_id_to_label) + 1 
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       min_size=img_min_size, 
                       max_size=img_max_size)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    milestones = [int(epochs*0.7), int(epochs*0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None

    print("Starting training...")
    
    best_map = 0.0
    (out_dir / "history.jsonl").write_text("", encoding="utf-8")

    for epoch in range(1, epochs+1):
        loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        mAP = evaluate_coco(model, val_ds, device, num_workers=num_workers)
        
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val mAP: {mAP:.4f} | LR: {curr_lr:.6f}")
        
        row = {"epoch": epoch, "train_loss": loss, "val_map": mAP, "lr": curr_lr}
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), out_dir / "detector_best.pth")
            
            # --- META SAVING ---
            meta = {
                "config_path": args.config,
                "data_root": str(data_root),
                "out_dir": str(out_dir),
                
                "backbone": "resnet50_imagenet_only", # Explicitly tagging this experiment
                "img_min_size": img_min_size,
                "img_max_size": img_max_size,
                "num_classes_including_bg": num_classes,
                
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "best_val_map": float(best_map),
                "cat_id_to_label": train_ds.cat_id_to_label, 
                "label_to_cat_id": train_ds.label_to_cat_id,
            }
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                
    print(f"Finished. Best CLS-Init mAP: {best_map:.4f}")
    print(f"Saved detailed meta.json to {out_dir}")

if __name__ == "__main__":
    main()