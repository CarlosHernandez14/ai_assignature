#!/usr/bin/env python3
"""
ant_detect_crop.py

Detect ants (or other class) with YOLO + optional SAHI sliced inference,
crop each detection and resize the crops to fixed size.

Usage examples:
  python ant_detect_crop.py --source ./images --out_dir ./ant_crops --model yolov8n.pt --use_sahi --slice 512 --overlap 0.2 --target_class ant --crop_size 100

  python ant_detect_crop.py --source ./images --out_dir ./ant_crops --model yolov8n.pt --target_class ant --crop_size 128x128 --yolo_conf 0.35

Requirements:
  pip install ultralytics sahi pillow tqdm numpy pandas
"""

import argparse
from pathlib import Path
import os
import csv
from PIL import Image
import math
import numpy as np
from tqdm import tqdm

def parse_size_arg(s):
    if s is None:
        return None
    s = str(s)
    if "x" in s:
        a,b = s.lower().split("x")
        return (int(a), int(b))
    else:
        v = int(s)
        return (v, v)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def pad_bbox(x1,y1,x2,y2, img_w, img_h, pad_frac):
    # pad_frac is fraction of box width/height to expand on each side
    w = x2 - x1
    h = y2 - y1
    pad_w = w * pad_frac
    pad_h = h * pad_frac
    nx1 = max(0, int(round(x1 - pad_w)))
    ny1 = max(0, int(round(y1 - pad_h)))
    nx2 = min(img_w, int(round(x2 + pad_w)))
    ny2 = min(img_h, int(round(y2 + pad_h)))
    return nx1, ny1, nx2, ny2

def save_crop(img_path, bbox, out_path, crop_size=None):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        img_w, img_h = im.size
        x1,y1,x2,y2 = bbox
        # clamp
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(img_w, int(round(x2)))
        y2 = min(img_h, int(round(y2)))
        if x2 <= x1 or y2 <= y1:
            return False
        crop = im.crop((x1,y1,x2,y2))
        if crop_size:
            crop = crop.resize(crop_size, Image.BICUBIC)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, quality=95)
    return True

def run_sahi_prediction(image_path, detection_model, slice_h, slice_w, overlap_h, overlap_w, conf_th):
    # returns SAHI PredictionResult (see SAHI docs)
    from sahi.predict import get_sliced_prediction
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
    )
    # Note: result.object_prediction_list holds ObjectPrediction objects (bbox, score, category)
    return result

def run_plain_yolo_prediction(image_path, yolov8_model, conf_th):
    # uses ultralytics YOLO model (model(path) -> results)
    res = yolov8_model.predict(source=str(image_path), conf=conf_th, imgsz=640, verbose=False)
    # res is list; take first
    r = res[0]
    object_list = []
    boxes = getattr(r, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        # iterate boxes
        try:
            xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            clsids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)
            names = r.names
            for xy, cf, cid in zip(xyxy_arr, confs, clsids):
                x1,y1,x2,y2 = map(float, xy)
                object_list.append({"bbox":(x1,y1,x2,y2), "score":float(cf), "category": names[int(cid)]})
        except Exception:
            # fallback: iterate boxes
            for b in boxes:
                try:
                    xy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else np.array(b.xyxy[0])
                    c = float(b.conf.cpu().numpy()) if hasattr(b.conf, "cpu") else float(b.conf)
                    cid = int(b.cls.cpu().numpy()) if hasattr(b.cls, "cpu") else int(b.cls)
                    name = r.names[cid] if hasattr(r, "names") else str(cid)
                    x1,y1,x2,y2 = map(float, xy)
                    object_list.append({"bbox":(x1,y1,x2,y2), "score":c, "category": name})
                except Exception:
                    continue
    return object_list

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", "-s", required=True, help="Folder with images (or single image)")
    p.add_argument("--out_dir", "-o", required=True, help="Output base dir")
    p.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model path (Ultralytics) or detect weights")
    p.add_argument("--use_sahi", action="store_true", help="Use SAHI sliced inference (recommended for ants/small objects)")
    p.add_argument("--slice", type=int, default=512, help="SAHI slice height & width (square) in px")
    p.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio (0.0-0.5)")
    p.add_argument("--yolo_conf", type=float, default=0.35, help="Confidence threshold for detections")
    p.add_argument("--target_class", default="ant", help="Category name to keep (e.g. 'ant'). For custom models use the class name used in training.")
    p.add_argument("--crop_size", default="100", help="Resize crops to this size (single int or WxH like 128x128). If omitted, saves raw crops.")
    p.add_argument("--single_largest", action="store_true", help="If multiple detections per image, only save the largest per image")
    p.add_argument("--pad_frac", type=float, default=0.1, help="Pad bbox by this fraction of bbox size before cropping (0.1 = 10%)")
    p.add_argument("--yolo_device", default=None, help="Device for YOLO/SAHI (e.g. 'cuda:0' or 'cpu')")
    p.add_argument("--min_score", type=float, default=0.25, help="Minimum detection score to accept (post SAHI or YOLO)")
    args = p.parse_args()

    src = Path(args.source)
    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    ensure_dir(crops_dir)
    csv_path = out_dir / "results.csv"

    # Gather images (support single image or dir)
    if src.is_dir():
        img_files = sorted([p for p in src.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tiff")])
    else:
        img_files = [src]

    crop_size = parse_size_arg(args.crop_size) if args.crop_size else None

    # Try to import SAHI if requested
    detection_model = None
    yolomodel = None
    if args.use_sahi:
        try:
            from sahi import AutoDetectionModel
            # create detection_model
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=args.model,
                confidence_threshold=args.yolo_conf,
                device=args.yolo_device or "cpu",
            )
        except Exception as e:
            print("Error loading SAHI AutoDetectionModel:", e)
            print("Make sure 'sahi' is installed and model path is correct.")
            return
    else:
        # load ultralytics raw model
        try:
            from ultralytics.models.yolo import YOLO
            yolomodel = YOLO(args.model)
            if args.yolo_device:
                try:
                    yolomodel.model.to(args.yolo_device)
                except Exception:
                    pass
        except Exception as e:
            print("Error loading YOLO model:", e)
            return

    rows = []
    for img_path in tqdm(img_files, desc="Images"):
        img_path = str(img_path)
        detections = []
        try:
            if args.use_sahi:
                result = run_sahi_prediction(img_path, detection_model, args.slice, args.slice, args.overlap, args.overlap, args.yolo_conf)
                # result.object_prediction_list -> iterate
                for pred in result.object_prediction_list:
                    # pred has bbox (minx,miny,maxx,maxy), score, category
                    bbox = pred.bbox
                    # SAHI provides bbox properties minx/miny/maxx/maxy
                    x1 = bbox.minx
                    y1 = bbox.miny
                    x2 = bbox.maxx
                    y2 = bbox.maxy
                    score = float(pred.score.value) if hasattr(pred, "score") and hasattr(pred.score, "value") else float(pred.score)
                    category = pred.category.name if hasattr(pred, "category") else getattr(pred, "category", None)
                    detections.append({"bbox":(x1,y1,x2,y2), "score":score, "category":category})
            else:
                preds = run_plain_yolo_prediction(img_path, yolomodel, args.yolo_conf)
                for p in preds:
                    detections.append(p)
        except Exception as e:
            tqdm.write(f"Prediction failed for {img_path}: {e}")
            continue

        # Filter by class and min_score
        filtered = [d for d in detections if (d.get("category") is not None and str(d.get("category")).lower() == args.target_class.lower() and d.get("score",0.0) >= args.min_score)]
        # Optionally take only largest bbox
        if args.single_largest and len(filtered) > 1:
            areas = [( (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]) ) for d in filtered]
            idx = int(np.argmax(areas))
            filtered = [filtered[idx]]

        saved_paths = []
        # Load image size once
        if len(filtered) > 0:
            with Image.open(img_path) as tmpim:
                img_w, img_h = tmpim.size
        for i, d in enumerate(filtered):
            x1,y1,x2,y2 = d["bbox"]
            # pad
            x1p,y1p,x2p,y2p = pad_bbox(x1,y1,x2,y2, img_w, img_h, args.pad_frac)
            base = Path(img_path).stem
            out_name = f"{base}__{args.target_class}__{i}.jpg"
            out_path = crops_dir / out_name
            ok = save_crop(img_path, (x1p,y1p,x2p,y2p), out_path, crop_size)
            if ok:
                saved_paths.append(str(out_path))

        rows.append({
            "source": img_path,
            "n_detections": len(detections),
            "n_filtered": len(filtered),
            "saved_crops": ";".join(saved_paths),
        })

    # write CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print("Done. Results CSV:", csv_path)
    print("Saved crops under:", crops_dir)

if __name__ == "__main__":
    main()
