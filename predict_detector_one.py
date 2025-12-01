import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config_utils import load_toml, deep_get


def build_model(num_classes: int, min_size: int = 640, max_size: int = 1024):
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    except Exception:
        weights = "DEFAULT"

    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def find_image_id_by_filename(coco: COCO, filename: str) -> Optional[int]:
    for img in coco.dataset.get("images", []):
        if img.get("file_name") == filename:
            return int(img["id"])
    return None


def build_class_color_map(class_names: Dict[int, str]) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Returns dict: coco_category_id -> RGBA color.
    Uses matplotlib tab20 palette for distinct colors.
    """
    cmap = plt.get_cmap("tab20")
    ids = sorted(class_names.keys())
    colors = {}
    for i, cid in enumerate(ids):
        colors[cid] = cmap(i % 20)  # RGBA tuple
    return colors


def draw_boxes(
    ax,
    boxes_xyxy: List[List[float]],
    label_texts: List[str],
    class_ids: List[int],
    class_colors: Dict[int, Tuple[float, float, float, float]],
    linestyle: str,
    linewidth: float = 2.2,
    text_alpha: float = 0.75,
):
    for (x1, y1, x2, y2), txt, cid in zip(boxes_xyxy, label_texts, class_ids):
        color = class_colors.get(cid, (1, 1, 1, 1))
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=linewidth,
            linestyle=linestyle,
            edgecolor=color,
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            max(0, y1 - 2),
            txt,
            fontsize=9,
            color="black",
            bbox=dict(facecolor=color, alpha=text_alpha, edgecolor="none", pad=2),
        )


def _as_path_or_none(x: Optional[str]) -> Optional[Path]:
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in ["none", "null"]:
        return None
    return Path(x)


def main():
    p = argparse.ArgumentParser()

    # Config
    p.add_argument("--config", type=str, default="config.toml", help="Path to config.toml")

    # Optional overrides (if not provided, read from config)
    p.add_argument("--data_root", default=None, help=".../endoscapes (overrides [paths].data_root)")
    p.add_argument("--split", default=None, choices=["train", "val", "test"])
    p.add_argument("--ann_name", default=None)

    p.add_argument("--ckpt", default=None, help="Detector .pth (overrides [eval.detector].ckpt)")
    p.add_argument("--meta", default=None, help="Detector meta.json (overrides [eval.detector].meta)")

    p.add_argument("--image", default=None, help="Full path OR filename inside split folder")
    p.add_argument("--pick_annotated", action="store_true", help="Pick an annotated COCO image")
    p.add_argument("--score_thr", type=float, default=None)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--save_dir", default=None)

    args = p.parse_args()

    # ---- load config ----
    cfg_toml = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg_toml = load_toml(cfg_path)

    # ---- resolve parameters (CLI overrides config) ----
    data_root = args.data_root or deep_get(cfg_toml, "paths.data_root", None)
    if data_root is None:
        raise RuntimeError("Missing data_root. Provide --data_root or set [paths].data_root in config.toml")

    split = args.split or deep_get(cfg_toml, "predict.detector.split", deep_get(cfg_toml, "eval.detector.split", "test"))
    ann_name = args.ann_name or deep_get(cfg_toml, "predict.detector.ann_name", deep_get(cfg_toml, "eval.detector.ann_name", "annotation_coco.json"))

    ckpt = args.ckpt or deep_get(cfg_toml, "eval.detector.ckpt", None)
    meta = args.meta or deep_get(cfg_toml, "eval.detector.meta", None)

    # Predict parameters
    cfg_image = deep_get(cfg_toml, "predict.detector.image", "")
    image_arg = args.image if args.image is not None else (cfg_image if str(cfg_image).strip() != "" else None)

    cfg_pick_annotated = bool(deep_get(cfg_toml, "predict.detector.pick_annotated", False))
    pick_annotated = bool(args.pick_annotated) or cfg_pick_annotated

    score_thr = args.score_thr if args.score_thr is not None else float(deep_get(cfg_toml, "predict.detector.score_thr", 0.5))
    topk = args.topk if args.topk is not None else int(deep_get(cfg_toml, "predict.detector.topk", 50))
    save_dir = args.save_dir if args.save_dir is not None else deep_get(cfg_toml, "predict.detector.save_dir", None)
    save_dir = str(save_dir).strip() if save_dir not in [None, ""] else None

    if ckpt is None:
        raise RuntimeError("Missing detector checkpoint. Set [eval.detector].ckpt in config.toml or pass --ckpt")

    data_root = Path(data_root)
    split_dir = data_root / split
    ann_path = split_dir / ann_name
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    coco = COCO(str(ann_path))
    cat_ids_sorted = sorted(coco.getCatIds())
    num_classes = 1 + len(cat_ids_sorted)

    # Category names (COCO category_id -> name)
    cats = coco.loadCats(cat_ids_sorted)
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in cats}

    # Our training mapped contiguous labels 1..K from cat_ids_sorted
    label_to_cat_id = {i + 1: cid for i, cid in enumerate(cat_ids_sorted)}

    class_colors = build_class_color_map(cat_id_to_name)

    # ---- choose image ----
    if pick_annotated:
        img_id = int(np.random.choice(coco.getImgIds()))
        info = coco.loadImgs([img_id])[0]
        img_path = split_dir / info["file_name"]
        gt_available = True
    else:
        if image_arg is None:
            raise RuntimeError("Provide --image (or set [predict.detector].image) OR enable pick_annotated=true")

        img_path = Path(image_arg)
        if not img_path.exists():
            img_path = split_dir / str(image_arg)

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_arg}")

        img_id = find_image_id_by_filename(coco, img_path.name)
        gt_available = img_id is not None

    # ---- model sizes: prefer meta.json, fallback to config detector.model ----
    min_size = int(deep_get(cfg_toml, "detector.model.img_min_size", 640))
    max_size = int(deep_get(cfg_toml, "detector.model.img_max_size", 1024))
    meta_path = _as_path_or_none(meta)
    if meta_path is not None and meta_path.exists():
        meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
        min_size = int(meta_obj.get("img_min_size", min_size))
        max_size = int(meta_obj.get("img_max_size", max_size))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, min_size=min_size, max_size=max_size)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    model.eval()

    # ---- load image ----
    img = Image.open(img_path).convert("RGB")
    x = T.ToTensor()(img).to(device)

    # ---- predict ----
    with torch.no_grad():
        out = model([x])[0]

    pred_boxes_t = out["boxes"].detach().cpu()
    pred_labels_t = out["labels"].detach().cpu()
    pred_scores_t = out["scores"].detach().cpu()

    keep = pred_scores_t >= score_thr
    pred_boxes_t = pred_boxes_t[keep][:topk]
    pred_labels_t = pred_labels_t[keep][:topk]
    pred_scores_t = pred_scores_t[keep][:topk]

    pred_cat_ids = [label_to_cat_id.get(int(l), -1) for l in pred_labels_t.tolist()]
    pred_boxes = pred_boxes_t.numpy().tolist()

    pred_texts = []
    for cid, sc in zip(pred_cat_ids, pred_scores_t.tolist()):
        name = cat_id_to_name.get(cid, f"cid{cid}")
        pred_texts.append(f"{name} {sc:.2f}")

    # ---- GT ----
    gt_boxes, gt_texts, gt_cat_ids = [], [], []
    if gt_available:
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            if "bbox" not in a:
                continue
            x0, y0, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            cid = int(a["category_id"])
            gt_boxes.append([x0, y0, x0 + w, y0 + h])
            gt_cat_ids.append(cid)
            gt_texts.append(cat_id_to_name.get(cid, str(cid)))

    # ---- save folder ----
    save_dir_p = Path(save_dir) if save_dir else None
    if save_dir_p is not None:
        save_dir_p.mkdir(parents=True, exist_ok=True)

    # ---- FIG 1: GT ONLY ----
    fig1 = plt.figure(figsize=(12, 7))
    plt.imshow(img)
    ax1 = plt.gca()
    if gt_available and len(gt_boxes) > 0:
        draw_boxes(ax1, gt_boxes, gt_texts, gt_cat_ids, class_colors, linestyle="--", linewidth=2.5, text_alpha=0.7)
        plt.title(f"GT ONLY — {img_path.name} (GT boxes: {len(gt_boxes)})")
    else:
        plt.title(f"GT ONLY — {img_path.name} (No GT available for this frame)")
        ax1.text(
            10, 20,
            "No GT available: image not referenced by annotation_coco.json",
            fontsize=11,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=4),
            color="white",
        )
    plt.axis("off")
    plt.tight_layout()
    if save_dir_p is not None:
        gt_out = save_dir_p / f"{img_path.stem}_GT.png"
        fig1.savefig(gt_out, dpi=200, bbox_inches="tight")
        print(f"Saved: {gt_out}")

    # ---- FIG 2: PRED ONLY ----
    fig2 = plt.figure(figsize=(12, 7))
    plt.imshow(img)
    ax2 = plt.gca()
    if len(pred_boxes) > 0:
        draw_boxes(ax2, pred_boxes, pred_texts, pred_cat_ids, class_colors, linestyle="-", linewidth=2.5, text_alpha=0.7)
        plt.title(f"PRED ONLY — {img_path.name} (Pred boxes: {len(pred_boxes)} | thr={score_thr})")
    else:
        plt.title(f"PRED ONLY — {img_path.name} (No predictions ≥ {score_thr})")
        ax2.text(
            10, 20,
            f"No predictions above threshold {score_thr}",
            fontsize=11,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=4),
            color="white",
        )
    plt.axis("off")
    plt.tight_layout()
    if save_dir_p is not None:
        pred_out = save_dir_p / f"{img_path.stem}_PRED.png"
        fig2.savefig(pred_out, dpi=200, bbox_inches="tight")
        print(f"Saved: {pred_out}")

    # Console summary
    print("\n--- SUMMARY ---")
    print("Image:", img_path)
    print("Split:", split)
    print("CKPT :", ckpt)
    print("Meta :", meta if meta else "(none)")
    print("Score thr:", score_thr, "| topk:", topk)
    print("GT boxes:", len(gt_boxes), "| Pred boxes:", len(pred_boxes))

    plt.show()


if __name__ == "__main__":
    main()