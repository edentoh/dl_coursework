# train_detector.py
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert

from config_utils import load_toml, deep_get


# -------------------------
# Utilities
# -------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def _clamp_(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return x.clamp(min=lo, max=hi)


def sanitize_target_xyxy(
    target: Dict[str, Any],
    image_w: int,
    image_h: int,
    min_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Ensures:
      - boxes in xyxy are within image bounds
      - x1<=x2, y1<=y2
      - filters boxes with non-positive (or too small) width/height

    TorchVision detectors REQUIRE positive width/height, otherwise they assert.
    """
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return target

    x1 = _clamp_(boxes[:, 0], 0, image_w - 1)
    y1 = _clamp_(boxes[:, 1], 0, image_h - 1)
    x2 = _clamp_(boxes[:, 2], 0, image_w - 1)
    y2 = _clamp_(boxes[:, 3], 0, image_h - 1)

    x_min = torch.minimum(x1, x2)
    x_max = torch.maximum(x1, x2)
    y_min = torch.minimum(y1, y2)
    y_max = torch.maximum(y1, y2)

    fixed = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    w = fixed[:, 2] - fixed[:, 0]
    h = fixed[:, 3] - fixed[:, 1]
    keep = (w >= min_size) & (h >= min_size)

    if keep.sum().item() == 0:
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0,), dtype=torch.int64)
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        return target

    target["boxes"] = fixed[keep]
    target["labels"] = target["labels"][keep]
    target["area"] = target["area"][keep]
    target["iscrowd"] = target["iscrowd"][keep]
    return target


# -------------------------
# Simple detection transforms
# -------------------------
class RandomHorizontalFlipDet:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        _, h, w = img.shape
        img = torch.flip(img, dims=[2])

        boxes = target["boxes"]
        if boxes.numel() > 0:
            x1 = boxes[:, 0]
            x2 = boxes[:, 2]
            boxes = boxes.clone()
            boxes[:, 0] = (w - 1) - x2
            boxes[:, 2] = (w - 1) - x1
            target["boxes"] = boxes

        return img, target


class ColorJitterSimple:
    """Lightweight brightness/contrast jitter. Expects img in [0,1]."""

    def __init__(self, brightness=0.15, contrast=0.15):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: torch.Tensor, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - mean) * c + mean
        img = img * b
        img = img.clamp(0, 1)
        return img, target


class ComposeDet:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


# -------------------------
# Dataset
# -------------------------
class EndoscapesCocoDetection(Dataset):
    """
    COCO detection dataset (boxes). Returns:
      image: FloatTensor [C,H,W], range [0,1]
      target: dict(boxes [N,4] xyxy, labels [N], image_id, area, iscrowd)
    """

    def __init__(self, images_dir: Path, ann_path: Path, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(str(ann_path))
        self.img_ids = sorted(self.coco.getImgIds())
        self.transforms = transforms

        # Map COCO category_id -> contiguous 1..K (0 reserved for background)
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        img_path = self.images_dir / file_name

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        _, H, W = img_t.shape

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes_xywh = []
        labels = []
        areas = []
        iscrowd = []

        for a in anns:
            if "bbox" not in a:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes_xywh.append([x, y, w, h])
            labels.append(self.cat_id_to_label[a["category_id"]])
            areas.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))

        if len(boxes_xywh) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_xywh_t = torch.tensor(boxes_xywh, dtype=torch.float32)
            boxes = box_convert(boxes_xywh_t, in_fmt="xywh", out_fmt="xyxy")
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        # sanitize BEFORE transforms
        target = sanitize_target_xyxy(target, image_w=W, image_h=H, min_size=1.0)

        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)
            _, H2, W2 = img_t.shape
            target = sanitize_target_xyxy(target, image_w=W2, image_h=H2, min_size=1.0)

        return img_t, target


# -------------------------
# COCO evaluation
# -------------------------
@torch.no_grad()
def coco_eval_map(model, dataset: EndoscapesCocoDetection, device: torch.device, batch_size: int, num_workers: int):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    coco_gt = dataset.coco

    results = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            if len(out["boxes"]) == 0:
                continue

            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            xywh = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh").tolist()

            for b, s, l in zip(xywh, scores.tolist(), labels.tolist()):
                cat_id = dataset.label_to_cat_id.get(int(l), None)
                if cat_id is None:
                    continue
                if not (np.isfinite(b[0]) and np.isfinite(b[1]) and np.isfinite(b[2]) and np.isfinite(b[3]) and np.isfinite(s)):
                    continue
                if b[2] <= 0 or b[3] <= 0:
                    continue

                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cat_id),
                        "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "score": float(s),
                    }
                )

    if len(results) == 0:
        return {"map": 0.0}

    # pycocotools.loadRes can expect these keys
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {"description": "Endoscapes COCO annotations"}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"map": float(coco_eval.stats[0])}  # mAP@[.5:.95]


# -------------------------
# Model + training loop
# -------------------------
def build_fasterrcnn(num_classes: int, img_min_size: int, img_max_size: int, freeze_backbone: bool):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    except Exception:
        weights = "DEFAULT"

    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=img_min_size, max_size=img_max_size)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        for p in model.backbone.body.parameters():
            p.requires_grad = False

    return model


def train_one_epoch(model, optimizer, loader, device, scaler=None, use_amp: bool = False):
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


@dataclass
class DetConfig:
    data_root: Path
    train_split: str
    val_split: str
    ann_name: str
    out_dir: Path
    epochs: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    momentum: float
    amp: bool
    img_min_size: int
    img_max_size: int
    freeze_backbone: bool
    seed: int
    device: str
    config_path: str


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml", help="TOML config file")

    # allow CLI overrides; if None, we read from config/defaults
    p.add_argument("--data_root", type=str, default=None, help="Path to endoscapes/ root")
    p.add_argument("--train_split", type=str, default=None)
    p.add_argument("--val_split", type=str, default=None)
    p.add_argument("--ann_name", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--img_min_size", type=int, default=None)
    p.add_argument("--img_max_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    # flags (if present => True). If not present, we may still enable from config.
    p.add_argument("--amp", action="store_true", help="Enable AMP (mixed precision)")
    p.add_argument("--freeze_backbone", action="store_true", help="Train ROI head only (freeze backbone)")

    args = p.parse_args()

    # Load config if it exists, otherwise use empty dict
    cfg_toml = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg_toml = load_toml(cfg_path)

    # Paths / defaults from config.toml
    data_root = args.data_root or deep_get(cfg_toml, "paths.data_root", None)
    if data_root is None:
        raise RuntimeError(
            "data_root not provided.\n"
            "Fix: either pass --data_root <PATH> or set [paths].data_root in config.toml"
        )

    train_split = args.train_split or deep_get(cfg_toml, "detector.data.train_split", "train")
    val_split = args.val_split or deep_get(cfg_toml, "detector.data.val_split", "val")
    ann_name = args.ann_name or deep_get(cfg_toml, "detector.data.ann_name", "annotation_coco.json")

    out_dir = args.out_dir or deep_get(cfg_toml, "detector.train.out_dir", "runs/detector")
    epochs = args.epochs if args.epochs is not None else int(deep_get(cfg_toml, "detector.train.epochs", 15))
    batch_size = args.batch_size if args.batch_size is not None else int(deep_get(cfg_toml, "detector.train.batch_size", 2))
    num_workers = args.num_workers if args.num_workers is not None else int(deep_get(cfg_toml, "detector.train.num_workers", 4))
    lr = args.lr if args.lr is not None else float(deep_get(cfg_toml, "detector.train.lr", 0.005))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(deep_get(cfg_toml, "detector.train.weight_decay", 1e-4))
    momentum = args.momentum if args.momentum is not None else float(deep_get(cfg_toml, "detector.train.momentum", 0.9))

    img_min_size = args.img_min_size if args.img_min_size is not None else int(deep_get(cfg_toml, "detector.model.img_min_size", 640))
    img_max_size = args.img_max_size if args.img_max_size is not None else int(deep_get(cfg_toml, "detector.model.img_max_size", 1024))
    seed = args.seed if args.seed is not None else int(deep_get(cfg_toml, "detector.train.seed", 42))

    # flags: CLI can force them on; otherwise config can enable them
    amp = bool(args.amp) or bool(deep_get(cfg_toml, "detector.train.amp", False))
    freeze_backbone = bool(args.freeze_backbone) or bool(deep_get(cfg_toml, "detector.train.freeze_backbone", False))

    cfg = DetConfig(
        data_root=Path(data_root),
        train_split=train_split,
        val_split=val_split,
        ann_name=ann_name,
        out_dir=Path(out_dir),
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        amp=amp,
        img_min_size=img_min_size,
        img_max_size=img_max_size,
        freeze_backbone=freeze_backbone,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config_path=str(cfg_path),
    )

    seed_everything(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = cfg.data_root / cfg.train_split
    val_dir = cfg.data_root / cfg.val_split
    train_ann = train_dir / cfg.ann_name
    val_ann = val_dir / cfg.ann_name

    if not train_ann.exists():
        raise FileNotFoundError(f"Missing {train_ann}")
    if not val_ann.exists():
        raise FileNotFoundError(f"Missing {val_ann}")

    train_tfms = ComposeDet([RandomHorizontalFlipDet(p=0.5), ColorJitterSimple(0.15, 0.15)])
    train_ds = EndoscapesCocoDetection(train_dir, train_ann, transforms=train_tfms)
    val_ds = EndoscapesCocoDetection(val_dir, val_ann, transforms=None)

    num_classes = 1 + len(train_ds.cat_id_to_label)  # background + K

    model = build_fasterrcnn(
        num_classes=num_classes,
        img_min_size=cfg.img_min_size,
        img_max_size=cfg.img_max_size,
        freeze_backbone=cfg.freeze_backbone,
    )
    device = torch.device(cfg.device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and cfg.device == "cuda") else None

    best_map = -1.0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, scaler=scaler, use_amp=(scaler is not None))
        val_metrics = coco_eval_map(model, val_ds, device, batch_size=1, num_workers=cfg.num_workers)
        mAP = val_metrics["map"]

        row = {"epoch": epoch, "train_loss": loss, "val_map": mAP}
        print(json.dumps(row))

        torch.save(model.state_dict(), cfg.out_dir / "detector_last.pth")
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), cfg.out_dir / "detector_best.pth")

        with open(cfg.out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    print(f"Done. Best val mAP: {best_map:.4f}")

    meta = {
        "config_path": cfg.config_path,
        "data_root": str(cfg.data_root),
        "train_split": cfg.train_split,
        "val_split": cfg.val_split,
        "ann_name": cfg.ann_name,
        "out_dir": str(cfg.out_dir),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "momentum": cfg.momentum,
        "amp": cfg.amp,
        "img_min_size": cfg.img_min_size,
        "img_max_size": cfg.img_max_size,
        "freeze_backbone": cfg.freeze_backbone,
        "seed": cfg.seed,
        "device": cfg.device,
        "num_classes_including_bg": num_classes,
        "cat_id_to_label": train_ds.cat_id_to_label,
        "label_to_cat_id": train_ds.label_to_cat_id,
    }
    (cfg.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()