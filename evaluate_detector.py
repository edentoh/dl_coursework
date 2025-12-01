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
from torchvision.ops import box_convert

from config_utils import load_toml, deep_get


def collate_fn(batch):
    return tuple(zip(*batch))


def _clamp_(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return x.clamp(min=lo, max=hi)


def sanitize_target_xyxy(target: Dict[str, Any], image_w: int, image_h: int, min_size: float = 1.0) -> Dict[str, Any]:
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


class EndoscapesCocoDetection(Dataset):
    def __init__(self, images_dir: Path, ann_path: Path):
        self.images_dir = images_dir
        self.coco = COCO(str(ann_path))
        self.img_ids = sorted(self.coco.getImgIds())

        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img_path = self.images_dir / info["file_name"]

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        _, H, W = img_t.shape

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes_xywh, labels, areas, iscrowd = [], [], [], []
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
            boxes = box_convert(boxes_xywh_t, "xywh", "xyxy")
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
        target = sanitize_target_xyxy(target, image_w=W, image_h=H, min_size=1.0)
        return img_t, target


def build_model(num_classes: int, min_size: int, max_size: int):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    except Exception:
        weights = "DEFAULT"

    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def coco_eval_map(model, dataset: EndoscapesCocoDetection, device: torch.device, batch_size: int, num_workers: int):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
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

            xywh = box_convert(boxes, "xyxy", "xywh").tolist()
            for b, s, l in zip(xywh, scores.tolist(), labels.tolist()):
                cat_id = dataset.label_to_cat_id.get(int(l), None)
                if cat_id is None:
                    continue
                if not (np.isfinite(b[0]) and np.isfinite(b[1]) and np.isfinite(b[2]) and np.isfinite(b[3]) and np.isfinite(s)):
                    continue
                if b[2] <= 0 or b[3] <= 0:
                    continue
                results.append(
                    {"image_id": img_id, "category_id": int(cat_id), "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "score": float(s)}
                )

    if len(results) == 0:
        return {"map": 0.0}

    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {"description": "Endoscapes COCO annotations"}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"map": float(coco_eval.stats[0])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")

    # optional overrides
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    p.add_argument("--ann_name", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--meta", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    args = p.parse_args()

    cfg_toml = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg_toml = load_toml(cfg_path)

    data_root = args.data_root or deep_get(cfg_toml, "paths.data_root", None)
    if data_root is None:
        raise RuntimeError("Missing data_root. Provide --data_root or set [paths].data_root in config.toml")

    split = args.split or deep_get(cfg_toml, "eval.detector.split", deep_get(cfg_toml, "detector.data.test_split", "test"))
    ann_name = args.ann_name or deep_get(cfg_toml, "eval.detector.ann_name", deep_get(cfg_toml, "detector.data.ann_name", "annotation_coco.json"))
    ckpt = args.ckpt or deep_get(cfg_toml, "eval.detector.ckpt", None)
    meta_path = args.meta or deep_get(cfg_toml, "eval.detector.meta", None)
    num_workers = args.num_workers if args.num_workers is not None else int(deep_get(cfg_toml, "eval.detector.num_workers", 4))
    batch_size = args.batch_size if args.batch_size is not None else int(deep_get(cfg_toml, "eval.detector.batch_size", 1))

    if ckpt is None:
        raise RuntimeError("Missing detector checkpoint. Set [eval.detector].ckpt in config.toml or pass --ckpt")

    # image size from meta if available
    min_size, max_size = 640, 1024
    if meta_path and Path(meta_path).exists():
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        min_size = int(meta.get("img_min_size", min_size))
        max_size = int(meta.get("img_max_size", max_size))

    split_dir = Path(data_root) / split
    ann_path = split_dir / ann_name
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing {ann_path}")

    ds = EndoscapesCocoDetection(split_dir, ann_path)
    num_classes = 1 + len(ds.cat_id_to_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, min_size=min_size, max_size=max_size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)

    metrics = coco_eval_map(model, ds, device, batch_size=batch_size, num_workers=num_workers)
    print(json.dumps({"split": split, "ckpt": ckpt, **metrics}, indent=2))


if __name__ == "__main__":
    main()
