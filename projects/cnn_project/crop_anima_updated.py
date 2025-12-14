#!/usr/bin/env python3
"""
cut_animal.py - robust inference -> crop pipeline for Roboflow/inference-style models.

New features in this version:
 - Much more tolerant parsing of `model.infer()` results (various bbox formats).
 - --debug-dump: when parsing fails for the first image, save the raw `results`
   to disk as JSON (and a small text summary) so you can paste it here for further
   customization.
 - Improved logging to show which parsing path was used.
 - Keeps all original CLI options and behavior (padding, min-area, resize, dry-run).

Usage example (same as before):
 python cut_animal.py -i datasets/ant -o datasets/ant/crop --model-id ants-species-detection-actual/5 \
   --inference-config '{"confidence": 0.7, "iou":0.5}' --size 100 100 --min-confidence 0.3 --prefix crop --verbose --debug-dump
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv, find_dotenv

# Import your get_model helper (same as you used before)
from inference import get_model

load_dotenv(find_dotenv())


def parse_args():
    parser = argparse.ArgumentParser(description="Crop detections from images and resize them.")
    parser.add_argument("--input", "-i", required=True,
                        help="Input folder or glob (folder will be scanned for images).")
    parser.add_argument("--output", "-o", required=True,
                        help="Output folder where crops will be saved.")
    parser.add_argument("--model-id", required=True, help="Model id used by get_model().")
    parser.add_argument("--api-key", default=None,
                        help="API key for remote model. If omitted, will use ROBOFLOW_API_KEY env var from .env or env.")
    parser.add_argument("--inference-config", default='{}',
                        help="JSON string or path to JSON file with inference configuration (e.g. '{\"confidence\":0.7}').")
    parser.add_argument("--size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=[100, 100],
                        help="Target size (width height) to which crops will be resized. Default: 100 100.")
    parser.add_argument("--pattern", default="*.jpg,*.jpeg,*.png",
                        help="Comma-separated image file patterns to look for (default: jpg,jpeg,png).")
    parser.add_argument("--recursive", action="store_true", help="Search input folder recursively.")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="If available from detector, discard detections below this confidence.")
    parser.add_argument("--padding", type=float, default=0.0,
                        help="Padding around bbox as fraction of bbox size (e.g. 0.1 -> expand each side by 10%).")
    parser.add_argument("--max-crops-per-image", type=int, default=0,
                        help="Max crops to extract per image (0 => unlimited).")
    parser.add_argument("--start-index", type=int, default=1, help="Start index for output filenames.")
    parser.add_argument("--prefix", default="", help="Prefix to add to output filenames.")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg", help="Output image format.")
    parser.add_argument("--min-area", type=int, default=16,
                        help="Minimum area (pixels) of crop to accept (default 16).")
    parser.add_argument("--min-width", type=int, default=2, help="Minimum crop width in pixels.")
    parser.add_argument("--min-height", type=int, default=2, help="Minimum crop height in pixels.")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except writing files.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--debug-dump", action="store_true",
                        help="When parsing fails for the first image, dump raw results to ./debug_inference_dump.json for inspection.")
    return parser.parse_args()


def load_inference_config(cfg_arg):
    try:
        cfg_path = Path(cfg_arg)
        if cfg_path.exists() and cfg_path.is_file():
            return json.loads(cfg_path.read_text())
        else:
            return json.loads(cfg_arg)
    except Exception as e:
        logging.error("Failed to parse inference config. Provide a valid JSON string or path. Error: %s", e)
        return {}


def discover_images(input_path: str, patterns: str, recursive: bool):
    p = Path(input_path)
    patterns_list = [pat.strip() for pat in patterns.split(",") if pat.strip()]
    found = []
    if p.is_file():
        found.append(p)
        return found
    if recursive:
        for pat in patterns_list:
            for f in p.rglob(pat):
                if f.is_file():
                    found.append(f)
    else:
        for pat in patterns_list:
            for f in p.glob(pat):
                if f.is_file():
                    found.append(f)
    return sorted(found)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_bbox(x_min, y_min, x_max, y_max, padding, img_w, img_h):
    if padding <= 0.0:
        return x_min, y_min, x_max, y_max
    w = x_max - x_min
    h = y_max - y_min
    pad_w = int(round(w * padding))
    pad_h = int(round(h * padding))
    nx_min = clamp(x_min - pad_w, 0, img_w - 1)
    ny_min = clamp(y_min - pad_h, 0, img_h - 1)
    nx_max = clamp(x_max + pad_w, 0, img_w - 1)
    ny_max = clamp(y_max + pad_h, 0, img_h - 1)
    return nx_min, ny_min, nx_max, ny_max


def try_get_confidences(detections):
    for attr in ("conf", "confidence", "scores", "probability", "score"):
        vals = getattr(detections, attr, None)
        if vals is not None:
            return np.asarray(vals)
    try:
        if isinstance(detections, dict) and "confidence" in detections:
            return np.asarray(detections["confidence"])
    except Exception:
        pass
    return None


def _is_normalized(vals, img_w, img_h):
    try:
        x, y, w, h = vals
        return 0 < x <= 1 and 0 < y <= 1 and 0 < w <= 1 and 0 < h <= 1
    except Exception:
        return False


def _is_normalized_xyxy(vals, img_w, img_h):
    try:
        x1, y1, x2, y2 = vals
        return 0 < x1 <= 1 and 0 < y1 <= 1 and 0 < x2 <= 1 and 0 < y2 <= 1
    except Exception:
        return False


def parse_inference_results(results, img_w: int, img_h: int):
    """
    Try many common shapes and return (xyxy ndarray Nx4, confidences ndarray or None, class_ids ndarray or None)
    If nothing matched -> return (None, None, None)
    """
    # helper to coerce single-element wraps
    if isinstance(results, (list, tuple)) and len(results) == 1:
        results = results[0]

    preds = None

    # common containers
    if isinstance(results, dict):
        # prefer keys that commonly hold predictions
        for k in ("predictions", "preds", "detections", "outputs", "results"):
            if k in results:
                preds = results[k]
                break
        # sometimes the model returns a list of boxes directly under 'predictions'
        if preds is None and (("bbox" in results and isinstance(results["bbox"], list)) or ("boxes" in results and isinstance(results["boxes"], list))):
            preds = results.get("boxes") or results.get("bbox")
    elif isinstance(results, (list, tuple)):
        preds = results

    # if nothing, try to inspect attributes (object-like)
    if preds is None and hasattr(results, "__dict__"):
        # try to extract a .predictions attribute (e.g. SDK objects)
        preds = getattr(results, "predictions", None) or getattr(results, "preds", None)

    if preds is None:
        # nothing recognizable
        return None, None, None

    # Normalize preds to list
    try:
        preds = list(preds)
    except Exception:
        preds = [preds]

    boxes = []
    confs = []
    cls = []

    for p in preds:
        # If prediction is simple numeric list/tuple of 4 -> try to interpret
        if isinstance(p, (list, tuple)) and len(p) >= 4 and all(isinstance(x, (int, float)) for x in p[:4]):
            a, b, c, d = [float(x) for x in p[:4]]
            # could be (x,y,w,h) normalized or absolute, or (x1,y1,x2,y2)
            if _is_normalized((a, b, c, d), img_w, img_h):
                # treat as xywh normalized
                x = a * img_w; y = b * img_h; w = c * img_w; h = d * img_h
                x1 = x - w / 2; y1 = y - h / 2; x2 = x + w / 2; y2 = y + h / 2
            elif _is_normalized_xyxy((a, b, c, d), img_w, img_h):
                x1 = a * img_w; y1 = b * img_h; x2 = c * img_w; y2 = d * img_h
            else:
                # assume absolute xyxy if c > a and d > b
                if c > a and d > b:
                    x1, y1, x2, y2 = a, b, c, d
                else:
                    # assume xywh absolute
                    x, y, w, h = a, b, c, d
                    x1 = x - w / 2; y1 = y - h / 2; x2 = x + w / 2; y2 = y + h / 2
            boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
            confs.append(0.0)
            cls.append(None)
            continue

        if not isinstance(p, dict):
            logging.debug("Skipping non-dict prediction: %s", type(p))
            continue

        # Roboflow style: x,y,width,height
        if all(k in p for k in ("x", "y", "width", "height")):
            try:
                x = float(p["x"]); y = float(p["y"]); w = float(p["width"]); h = float(p["height"])
                if _is_normalized((x, y, w, h), img_w, img_h):
                    x *= img_w; y *= img_h; w *= img_w; h *= img_h
                x1 = x - w / 2; y1 = y - h / 2; x2 = x + w / 2; y2 = y + h / 2
                boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
                confs.append(float(p.get("confidence", p.get("score", 0.0))))
                cls.append(p.get("class", p.get("label")))
                continue
            except Exception:
                pass

        # 'bbox' key: list [x1,y1,x2,y2] or dict {xmin:..}
        if "bbox" in p:
            bbox = p["bbox"]
            try:
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                    if _is_normalized_xyxy((x1, y1, x2, y2), img_w, img_h):
                        x1 *= img_w; y1 *= img_h; x2 *= img_w; y2 *= img_h
                    boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
                    confs.append(float(p.get("confidence", p.get("score", 0.0))))
                    cls.append(p.get("class", p.get("label")))
                    continue
                elif isinstance(bbox, dict):
                    # try keys: xmin,xmax or left,top,width,height inside bbox dict
                    lower = {k.lower(): v for k, v in bbox.items()}
                    if all(k in lower for k in ("xmin", "ymin", "xmax", "ymax")):
                        x1 = float(lower["xmin"]); y1 = float(lower["ymin"]); x2 = float(lower["xmax"]); y2 = float(lower["ymax"])
                        if _is_normalized_xyxy((x1, y1, x2, y2), img_w, img_h):
                            x1 *= img_w; y1 *= img_h; x2 *= img_w; y2 *= img_h
                        boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
                        confs.append(float(p.get("confidence", p.get("score", 0.0))))
                        cls.append(p.get("class", p.get("label")))
                        continue
            except Exception:
                pass

        # keys x_min,y_min,x_max,y_max or xmin,ymin,xmax,ymax
        for keys in (("x_min", "y_min", "x_max", "y_max"), ("xmin", "ymin", "xmax", "ymax")):
            if all(k in p for k in keys):
                try:
                    x1 = float(p[keys[0]]); y1 = float(p[keys[1]]); x2 = float(p[keys[2]]); y2 = float(p[keys[3]])
                    if _is_normalized_xyxy((x1, y1, x2, y2), img_w, img_h):
                        x1 *= img_w; y1 *= img_h; x2 *= img_w; y2 *= img_h
                    boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
                    confs.append(float(p.get("confidence", p.get("score", 0.0))))
                    cls.append(p.get("class", p.get("label")))
                    break
                except Exception:
                    pass

        # left/top/width/height
        if all(k in p for k in ("left", "top", "width", "height")):
            try:
                left = float(p["left"]); top = float(p["top"]); w = float(p["width"]); h = float(p["height"])
                if 0 < left <= 1 and 0 < top <= 1 and 0 < w <= 1 and 0 < h <= 1:
                    left *= img_w; top *= img_h; w *= img_w; h *= img_h
                x1 = left; y1 = top; x2 = left + w; y2 = top + h
                boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
                confs.append(float(p.get("confidence", p.get("score", 0.0))))
                cls.append(p.get("class", p.get("label")))
                continue
            except Exception:
                pass

        # fallback: sometimes prediction contains an inner dict with 'box' or 'bbox' keys
        for inner_key in ("box", "bounds", "rectangle"):
            if inner_key in p and isinstance(p[inner_key], dict):
                b = p[inner_key]
                # try same patterns as above
                if all(k in b for k in ("x", "y", "width", "height")):
                    try:
                        x = float(b["x"]); y = float(b["y"]); w = float(b["width"]); h = float(b["height"])
                        if _is_normalized((x, y, w, h), img_w, img_h):
                            x *= img_w; y *= img_h; w *= img_w; h *= img_h
                        x1 = x - w / 2; y1 = y - h / 2; x2 = x + w / 2; y2 = y + h / 2
                        boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
                        confs.append(float(p.get("confidence", p.get("score", 0.0))))
                        cls.append(p.get("class", p.get("label")))
                        continue
                    except Exception:
                        pass

        # if none matched, debug log
        logging.debug("Unrecognized prediction dict keys: %s", list(p.keys()))

    if not boxes:
        return None, None, None

    xyxy = np.asarray(boxes, dtype=int)
    conf_arr = np.asarray(confs, dtype=float) if confs else None
    class_arr = np.asarray(cls) if cls else None
    return xyxy, conf_arr, class_arr


def build_detections_from_parsed(xyxy, confs, class_ids):
    # supervision.Detections expects xyxy as ndarray, optionally confidence and class_id arrays
    return sv.Detections(xyxy=xyxy, confidence=(confs if confs is not None else None),
                         class_id=(class_ids if class_ids is not None else None))


def dump_debug(results, img_path: Path, dump_path: Path):
    try:
        # Provide a safe serializable representation; if not JSON-serializable, use repr
        def safe(obj):
            try:
                json.dumps(obj)
                return obj
            except Exception:
                try:
                    # try converting numpy arrays
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                except Exception:
                    pass
                return repr(obj)
        payload = {
            "image": str(img_path),
            "summary": {
                "type": str(type(results)),
                "repr_first_1000": repr(results)[:1000]
            },
            "results_raw": safe(results)
        }
        dump_path.write_text(json.dumps(payload, indent=2))
        logging.warning("Wrote debug dump to %s", dump_path)
    except Exception as e:
        logging.exception("Failed to write debug dump: %s", e)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        logging.warning("No API key provided and ROBOFLOW_API_KEY env var was not set. Model load may fail.")

    inference_cfg = load_inference_config(args.inference_config)

    input_path = args.input
    output_path = Path(args.output)
    if not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    logging.info("Output folder: %s%s", output_path, " (dry-run)" if args.dry_run else "")

    images = discover_images(input_path, args.pattern, args.recursive)
    if not images:
        logging.warning("No images found in %s (patterns=%s). Exiting.", input_path, args.pattern)
        return

    logging.info("Found %d images to process.", len(images))

    logging.info("Loading model id=%s ...", args.model_id)
    try:
        model = get_model(model_id=args.model_id, api_key=api_key)
    except Exception as e:
        logging.exception("Failed to load model with get_model(). Error: %s", e)
        logging.error("If using the 'inference' package, consider installing extras: pip install 'inference[transformers]' 'inference[sam]' etc.")
        return

    total_crops = 0
    file_index = args.start_index
    target_w, target_h = int(args.size[0]), int(args.size[1])

    # debug dump path (only used when --debug-dump)
    dump_path = Path("./debug_inference_dump.json")
    dumped = False

    for img_path in images:
        img_path = Path(img_path)
        logging.info("Processing image: %s", img_path)
        frame = cv2.imread(str(img_path))
        if frame is None:
            logging.warning("Could not read image %s. Skipping.", img_path)
            continue
        img_h, img_w = frame.shape[:2]

        # Run inference
        try:
            results = model.infer(frame, inference_configuration=inference_cfg)
            # sometimes model returns [dict-with-predictions]
            if isinstance(results, (list, tuple)) and len(results) > 0 and isinstance(results[0], dict) and "predictions" in results[0]:
                results = results[0]
        except Exception as e:
            logging.exception("Model inference failed for %s: %s", img_path, e)
            continue

        detections = None
        # try library converter first
        try:
            detections = sv.Detections.from_inference(results)
            logging.debug("Used sv.Detections.from_inference() for %s", img_path)
        except Exception as e:
            logging.debug("sv.Detections.from_inference() failed for %s: %s", img_path, e)
            # fallback parsing
            try:
                xyxy, confs, class_ids = parse_inference_results(results, img_w, img_h)
                if xyxy is None:
                    logging.warning("Could not parse inference results for %s. Skipping image.", img_path)
                    # debug dump once if requested
                    if args.debug_dump and not dumped:
                        dump_debug(results, img_path, dump_path)
                        dumped = True
                    continue
                detections = build_detections_from_parsed(xyxy, confs, class_ids)
                logging.debug("Built sv.Detections from parsed results for %s (num=%d).", img_path, len(xyxy))
            except Exception as e2:
                logging.exception("Parsing fallback failed for %s: %s", img_path, e2)
                if args.debug_dump and not dumped:
                    dump_debug(results, img_path, dump_path)
                    dumped = True
                continue

        # now proceed as before
        xyxy = getattr(detections, "xyxy", None)
        if xyxy is None:
            logging.warning("detections.xyxy not found for %s. Skipping.", img_path)
            continue

        confidences = try_get_confidences(detections)

        crops_this_image = 0
        for det_idx, bbox in enumerate(xyxy):
            try:
                x_min, y_min, x_max, y_max = [int(round(float(v))) for v in bbox[:4]]
            except Exception as e:
                logging.debug("Invalid bbox format for %s index %s: %s. Skipping.", img_path, det_idx, e)
                continue

            if confidences is not None and args.min_confidence is not None:
                if det_idx < len(confidences) and confidences[det_idx] < args.min_confidence:
                    logging.debug("Skipping bbox %d due to low confidence %.3f", det_idx, confidences[det_idx])
                    continue

            x_min_e, y_min_e, x_max_e, y_max_e = expand_bbox(x_min, y_min, x_max, y_max, args.padding, img_w, img_h)

            w = x_max_e - x_min_e
            h = y_max_e - y_min_e
            area = w * h
            if area < args.min_area or w < args.min_width or h < args.min_height:
                logging.debug("Skipping small crop (w=%d h=%d area=%d) for %s", w, h, area, img_path)
                continue

            cropped = frame[y_min_e:y_max_e, x_min_e:x_max_e]
            if cropped.size == 0:
                logging.debug("Zero-size crop for bbox %d on %s. Skipping.", det_idx, img_path)
                continue

            try:
                resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
            except Exception as e:
                logging.exception("Resize failed for %s bbox %d: %s", img_path, det_idx, e)
                continue

            base = img_path.stem
            prefix_part = f"{args.prefix}_" if args.prefix else ""
            out_name = f"{prefix_part}{file_index}_{base}_{det_idx}.{args.format}"
            out_path = output_path / out_name

            if args.dry_run:
                logging.info("[DRY RUN] Would save crop to %s (w=%d h=%d)", out_path, w, h)
            else:
                try:
                    success = cv2.imwrite(str(out_path), resized)
                    if not success:
                        logging.warning("cv2.imwrite failed for %s", out_path)
                except Exception as e:
                    logging.exception("Failed to write crop to %s: %s", out_path, e)
                    continue

            total_crops += 1
            file_index += 1
            crops_this_image += 1

            if args.max_crops_per_image > 0 and crops_this_image >= args.max_crops_per_image:
                logging.debug("Reached max crops per image (%d) for %s", args.max_crops_per_image, img_path)
                break

    logging.info("Processing complete. Total crops saved: %d", total_crops)


if __name__ == "__main__":
    main()
