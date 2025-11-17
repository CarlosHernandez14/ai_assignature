#!/usr/bin/env python3
"""
clean_images_pipeline_with_crops.py

Usage examples:
  # Keep full images (no cropping)
  python clean_images_pipeline_with_crops.py -i ./images -t dog --model both -o ./cleaned_output

  # Crop YOLO detections and resize crops to 224x224
  python clean_images_pipeline_with_crops.py -i ./images -t dog --model both --yolo_resize 224 --yolo_single_crop -o ./cleaned_output

  # Crop YOLO detections and resize to 128x64
  python clean_images_pipeline_with_crops.py -i ./images -t ladybug --model yolo --yolo_resize 128x64 -o ./cleaned_output
"""

import argparse
import logging
from pathlib import Path
import os
import shutil
import csv
from PIL import Image
import numpy as np

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def is_readable_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def ensure_dirs(out_base):
    base = Path(out_base)
    keep = base / "keep"
    discard = base / "discard"
    uncertain = base / "uncertain"
    crops = base / "crops"   # if cropping mode used, we may put crops here (subfolders kept too)
    for d in (keep, discard, uncertain, crops):
        d.mkdir(parents=True, exist_ok=True)
    return keep, discard, uncertain, crops

def parse_resize_arg(val):
    """Accept '224' or '224x224' or '128x64' -> returns (w,h) ints"""
    if val is None:
        return None
    s = str(val)
    if "x" in s:
        parts = s.lower().split("x")
        if len(parts) != 2:
            raise ValueError("yolo_resize must be int or WIDTHxHEIGHT")
        return (int(parts[0]), int(parts[1]))
    else:
        v = int(s)
        return (v, v)

def yolo_predict_batch(paths, model_name="yolov8n.pt", imgsz=640, conf=0.35, device=None):
    """
    Return a dict: path -> list of detections, where each detection is dict:
      {'name': str, 'conf': float, 'xyxy': (x1,y1,x2,y2)}
    """
    try:
        from ultralytics.models.yolo import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics package required for YOLO. Install via `pip install ultralytics`") from e

    model = YOLO(model_name)
    if device:
        try:
            model.model.to(device)
        except Exception:
            pass

    results_map = {}
    for p in paths:
        detections = []
        try:
            res = model(str(p), imgsz=imgsz, conf=conf, verbose=False)  # res is a list
            r = res[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                # Each box has xyxy, conf, cls
                # Some ultralytics versions provide boxes.xyxy, boxes.conf, boxes.cls; handle possible tensor/np variations
                try:
                    xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
                    clsids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)
                except Exception:
                    # fallback: iterate each box object
                    for b in boxes:
                        try:
                            xy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else np.array(b.xyxy[0])
                            c = float(b.conf.cpu().numpy()) if hasattr(b.conf, "cpu") else float(b.conf)
                            cid = int(b.cls.cpu().numpy()) if hasattr(b.cls, "cpu") else int(b.cls)
                            name = r.names[cid] if hasattr(r, "names") else str(cid)
                            detections.append({'name': name, 'conf': c, 'xyxy': (float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3]))})
                        except Exception:
                            continue
                    results_map[str(p)] = detections
                    continue

                for xy, cf, cid in zip(xyxy_arr, confs, clsids):
                    cname = r.names[int(cid)] if hasattr(r, "names") else str(int(cid))
                    x1, y1, x2, y2 = map(float, xy)
                    detections.append({'name': cname, 'conf': float(cf), 'xyxy': (x1, y1, x2, y2)})
        except Exception as e:
            logging.warning(f"YOLO failed on {p}: {e}")
        results_map[str(p)] = detections
    return results_map

# CLIP helper (unchanged from previous script)
def clip_score_image(path, prompts):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        raise RuntimeError("transformers + torch required for CLIP. Install via `pip install transformers torch`") from e

    if not hasattr(clip_score_image, "model"):
        clip_score_image.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_score_image.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(path).convert("RGB")
    inputs = clip_score_image.proc(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = clip_score_image.model(**inputs)
        logits_per_image = out.logits_per_image
        probs = torch.softmax(logits_per_image, dim=1)[0].cpu().numpy()
    return probs.tolist()

def decide(yolo_dets, clip_scores, prompts, target_prompt, clip_thresh, yolo_target_names=set(), include_terms=None, exclude_terms=None, yolo_conf=0.35):
    include_terms = include_terms or []
    exclude_terms = exclude_terms or []

    # YOLO decision
    if yolo_dets:
        for det in yolo_dets:
            name = det.get('name')
            conf = det.get('conf', 0.0)
            if name in yolo_target_names and conf >= yolo_conf:
                return "keep", {"by": "yolo", "yolo_top": name, "yolo_conf": conf, "yolo_det": det}
    # CLIP decision
    if clip_scores is not None:
        try:
            idx = prompts.index(target_prompt)
        except ValueError:
            idx = 0
        target_score = clip_scores[idx]
        exclude_hits = []
        for ex in exclude_terms:
            if ex in prompts:
                ex_idx = prompts.index(ex)
                if clip_scores[ex_idx] >= clip_thresh:
                    exclude_hits.append((ex, clip_scores[ex_idx]))
        if target_score >= clip_thresh and not exclude_hits:
            return "keep", {"by": "clip", "clip_score": float(target_score)}
        elif target_score >= clip_thresh and exclude_hits:
            return "uncertain", {"by": "clip_excluded", "clip_score": float(target_score), "exclude_hits": exclude_hits}
        else:
            return "uncertain", {"by": "low_clip", "clip_score": float(target_score)}
    return "uncertain", {"by": "no_info"}

def crop_and_save(original_path, dets, save_dir, resize_to=None, single_largest=False, prefix=""):
    """
    dets: list of detection dicts with 'xyxy' and 'name' and 'conf'
    resize_to: (w,h) tuple or None
    If single_largest=True, only save largest det (by area)
    Returns list of saved file paths.
    """
    saved = []
    if not dets:
        return saved

    try:
        img = Image.open(original_path).convert("RGB")
    except Exception as e:
        logging.warning(f"Cannot open for cropping: {original_path}: {e}")
        return saved

    img_w, img_h = img.size

    if single_largest and len(dets) > 1:
        # compute areas and select largest
        areas = []
        for d in dets:
            x1,y1,x2,y2 = d['xyxy']
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            areas.append(w*h)
        idx = int(np.argmax(areas))
        dets = [dets[idx]]

    for idx, d in enumerate(dets):
        x1,y1,x2,y2 = d['xyxy']
        # clamp to image bounds and int
        x1i = max(0, int(round(x1)))
        y1i = max(0, int(round(y1)))
        x2i = min(img_w, int(round(x2)))
        y2i = min(img_h, int(round(y2)))
        if x2i <= x1i or y2i <= y1i:
            logging.debug(f"Skipping empty crop for {original_path}: {d['xyxy']}")
            continue
        try:
            crop = img.crop((x1i, y1i, x2i, y2i))
            if resize_to:
                crop = crop.resize(resize_to, Image.BICUBIC)
            # save with suffix indicating crop index and class
            base = Path(original_path).stem
            ext = ".jpg"
            cname = d.get('name','obj')
            out_name = f"{prefix}{base}__{cname}__{idx}{ext}"
            out_path = save_dir / out_name
            crop.save(out_path, quality=95)
            saved.append(str(out_path))
        except Exception as e:
            logging.warning(f"Failed to crop/save {original_path}: {e}")
    return saved

def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Clean images pipeline with YOLO cropping/resizing")
    parser.add_argument("--input_dir", "-i", required=True, help="Folder with images")
    parser.add_argument("--target_class", "-t", required=True, help="Target class name, e.g. dog, cat, turtle, ant, ladybug")
    parser.add_argument("--model", choices=("yolo", "clip", "both"), default="both", help="Which model(s) to use for filtering")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="YOLO model path or name (ultralytics)")
    parser.add_argument("--yolo_conf", type=float, default=0.35, help="YOLO confidence threshold for detection")
    parser.add_argument("--imgsz", type=int, default=640, help="Resize images to this size for YOLO inference")
    parser.add_argument("--clip_thresh", type=float, default=0.40, help="CLIP softmax threshold to accept target")
    parser.add_argument("--dedup", action="store_true", help="Enable perceptual deduplication")
    parser.add_argument("--phash_thresh", type=int, default=4, help="Hamming threshold for phash dedup")
    parser.add_argument("--output_dir", "-o", default="output", help="Base output directory")
    parser.add_argument("--auto_discard", action="store_true", help="Automatically move UNCERTAIN to discard (not recommended)")
    parser.add_argument("--include_terms", nargs="*", default=[], help="Extra include textual terms for CLIP prompts")
    parser.add_argument("--exclude_terms", nargs="*", default=[], help="Terms to consider excluding (e.g. wolf)")
    parser.add_argument("--device", default=None, help="torch device for YOLO/CLIP (e.g. 'cuda:0' or 'cpu')")
    # new args for cropping
    parser.add_argument("--yolo_resize", default=None, help="If provided, crop YOLO-detected regions and resize to this size. Accepts single int or WIDTHxHEIGHT (e.g. 224 or 128x128)")
    parser.add_argument("--yolo_single_crop", action="store_true", help="If set, save only the single largest YOLO crop per image (otherwise save all detections)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    keep_dir, discard_dir, uncertain_dir, crops_dir = ensure_dirs(args.output_dir)

    # gather images
    all_files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    logging.info(f"Found {len(all_files)} image files")
    readable = [str(p) for p in all_files if is_readable_image(p)]
    if len(readable) < len(all_files):
        logging.info(f"Skipped {len(all_files)-len(readable)} unreadable files")

    # optional dedup (same as previous behavior, but minimal implementation)
    if args.dedup:
        logging.info("Running deduplication (phash)...")
        try:
            import imagehash
            from PIL import Image as PILImage
            hashes = []
            kept = []
            for p in readable:
                try:
                    h = imagehash.phash(PILImage.open(p))
                except Exception:
                    continue
                is_dup = False
                for existing in hashes:
                    if abs(h - existing) <= args.phash_thresh:
                        is_dup = True
                        break
                if not is_dup:
                    hashes.append(h)
                    kept.append(p)
            readable = kept
            logging.info(f"{len(readable)} images remain after dedup")
        except Exception as e:
            logging.warning("imagehash not available or failed; skipping dedup")

    # CLIP prompt setup
    target_prompt = f"a photo of a {args.target_class}"
    prompts = [target_prompt] + [f"a photo of a {x}" for x in args.include_terms + args.exclude_terms]

    # YOLO inference if requested
    yolo_results = {}
    if args.model in ("yolo", "both"):
        logging.info("Running YOLO inference (this may take time)...")
        yolo_results = yolo_predict_batch(readable, model_name=args.yolo_model, imgsz=args.imgsz, conf=args.yolo_conf, device=args.device)

    rows = []
    resize_tuple = parse_resize_arg(args.yolo_resize)
    for p in readable:
        pstr = str(p)
        yolo_dets = yolo_results.get(pstr, []) if yolo_results else []
        clip_scores = None
        if args.model in ("clip", "both"):
            try:
                clip_scores = clip_score_image(pstr, prompts)
            except Exception as e:
                logging.warning(f"CLIP failed for {pstr}: {e}")
                clip_scores = None

        yolo_target_names = {args.target_class} if args.target_class in ("dog","cat") else {args.target_class}
        decision, meta = decide(yolo_dets, clip_scores, prompts, target_prompt, args.clip_thresh, yolo_target_names, include_terms=args.include_terms, exclude_terms=args.exclude_terms, yolo_conf=args.yolo_conf)

        # If cropping requested and we have yolo detections, save crops
        saved_files = []
        if resize_tuple and yolo_dets:
            # filter detections to those matching target class and >= conf threshold
            targ_dets = [d for d in yolo_dets if d.get('name') in yolo_target_names and d.get('conf',0.0) >= args.yolo_conf]
            if targ_dets:
                # create per-class subfolder under crops_dir
                class_dir = crops_dir / args.target_class
                class_dir.mkdir(parents=True, exist_ok=True)
                saved_files = crop_and_save(pstr, targ_dets, class_dir, resize_to=resize_tuple, single_largest=args.yolo_single_crop, prefix="")
                if saved_files:
                    # Mark decision keep if we successfully saved crops
                    decision = "keep"
            else:
                # no matching detections; fallthrough to previous logic (uncertain/discard)
                pass

        # If not cropping mode OR no crops produced, copy the whole image to respective folder
        if not resize_tuple or not saved_files:
            if decision == "keep":
                shutil.copy(pstr, keep_dir / Path(pstr).name)
            elif decision == "uncertain":
                dest = discard_dir if args.auto_discard else uncertain_dir
                shutil.copy(pstr, dest / Path(pstr).name)
            else:
                shutil.copy(pstr, discard_dir / Path(pstr).name)
        else:
            # crops were saved; optionally copy originals to 'keep/orig' for traceability
            orig_saved_dir = keep_dir / "orig"
            orig_saved_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(pstr, orig_saved_dir / Path(pstr).name)

        rows.append({
            "path": pstr,
            "decision": decision,
            "yolo_detections": ";".join([f"{d['name']}:{d['conf']:.3f}" for d in yolo_dets]) if yolo_dets else "",
            "clip_scores": ";".join([f"{prompts[i]}:{(clip_scores[i] if clip_scores is not None else 'NA')}" for i in range(len(prompts))]) if clip_scores is not None else "",
            "meta": str(meta),
            "saved_crops": ";".join(saved_files) if saved_files else ""
        })

    csv_path = Path(args.output_dir) / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path","decision","yolo_detections","clip_scores","meta","saved_crops"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    logging.info(f"Done. Results CSV: {csv_path}")
    logging.info(f"Kept (full images): {len(list((Path(args.output_dir)/'keep').glob('*')))}")
    logging.info(f"Discarded: {len(list((Path(args.output_dir)/'discard').glob('*')))}")
    logging.info(f"Uncertain: {len(list((Path(args.output_dir)/'uncertain').glob('*')))}")
    logging.info(f"Crops saved under: {Path(args.output_dir)/'crops'}")

if __name__ == "__main__":
    main()
