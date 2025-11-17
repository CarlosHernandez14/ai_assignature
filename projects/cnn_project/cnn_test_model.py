# predict_folder.py
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

# Allowed image extensions
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

def discover_image_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                files.append(os.path.join(root, fn))
    files.sort()
    return files

def load_and_preprocess_image(path, img_size):
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im = im.resize(img_size, Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0
    return arr

def infer_class_names_from_labels_dir(labels_dir):
    subdirs = [d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))]
    subdirs.sort()
    return subdirs

def annotate_and_save_image(image_array, text, out_path):
    img = Image.fromarray((image_array * 255).astype("uint8"))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()
    text_w, text_h = draw.textsize(text, font=font)
    rect = (0, 0, text_w + 8, text_h + 6)
    draw.rectangle(rect, fill=(0, 0, 0, 150))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    img.save(out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True, help="Path to saved Keras model (.h5)")
    p.add_argument("--images", "-i", required=True, help="Folder with images to predict (recursively searched)")
    p.add_argument("--labels-dir", "-l", required=False,
                   help="(optional) dataset root used for training; used to infer class names in sorted order")
    p.add_argument("--img-size", type=int, default=100, help="Resize size (square). Default=100")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction")
    p.add_argument("--out-csv", type=str, default=None,
                   help="(optional) CSV output file path. If not provided, predictions are printed only.")
    p.add_argument("--save-annotated", action="store_true",
                   help="Save annotated images with predicted label in folder annotated/")
    args = p.parse_args()

    # Load model
    print("Loading model:", args.model)
    model = tf.keras.models.load_model(args.model)
    print("Model loaded.")

    # Class names
    class_names = None
    if args.labels_dir:
        if not os.path.isdir(args.labels_dir):
            raise SystemExit(f"--labels-dir provided but not a directory: {args.labels_dir}")
        class_names = infer_class_names_from_labels_dir(args.labels_dir)
        print("Inferred class names (index -> name):")
        for idx, name in enumerate(class_names):
            print(f"  {idx}: {name}")
    else:
        print("No --labels-dir provided. Output will show class indices only.")

    # Discover images
    img_paths = discover_image_files(args.images)
    if not img_paths:
        raise SystemExit("No images found in folder: " + args.images)
    print(f"Found {len(img_paths)} images under {args.images}")

    # Prepare output
    results = []
    batch = []
    batch_paths = []
    imgs_size = (args.img_size, args.img_size)

    out_annot_dir = None
    if args.save_annotated:
        out_annot_dir = os.path.join(os.getcwd(), "annotated")
        os.makedirs(out_annot_dir, exist_ok=True)

    # Iterate and predict in batches
    for path in tqdm(img_paths, desc="Predicting"):
        try:
            arr = load_and_preprocess_image(path, imgs_size)
        except Exception as e:
            print(f"Skipped {path} (error reading): {e}")
            continue
        batch.append(arr)
        batch_paths.append(path)

        if len(batch) >= args.batch_size:
            X = np.stack(batch, axis=0)
            preds = model.predict(X, verbose=0)
            top_idxs = np.argmax(preds, axis=1)
            top_probs = np.max(preds, axis=1)
            for pth, idx, prob, imarr in zip(batch_paths, top_idxs, top_probs, X):
                label = class_names[idx] if class_names else str(idx)
                print(f"{os.path.basename(pth)} -> {label} ({prob:.4f})")
                results.append({"filename": pth, "pred_index": int(idx), "pred_label": label, "prob": float(prob)})
                if out_annot_dir:
                    save_path = os.path.join(out_annot_dir, os.path.basename(pth))
                    annotate_and_save_image(imarr, f"{label} {prob:.3f}", save_path)
            batch = []
            batch_paths = []

    # Final partial batch
    if batch:
        X = np.stack(batch, axis=0)
        preds = model.predict(X, verbose=0)
        top_idxs = np.argmax(preds, axis=1)
        top_probs = np.max(preds, axis=1)
        for pth, idx, prob, imarr in zip(batch_paths, top_idxs, top_probs, X):
            label = class_names[idx] if class_names else str(idx)
            print(f"{os.path.basename(pth)} -> {label} ({prob:.4f})")
            results.append({"filename": pth, "pred_index": int(idx), "pred_label": label, "prob": float(prob)})
            if out_annot_dir:
                save_path = os.path.join(out_annot_dir, os.path.basename(pth))
                annotate_and_save_image(imarr, f"{label} {prob:.3f}", save_path)

    # Save CSV only if user provided --out-csv
    if args.out_csv:
        df = pd.DataFrame(results)
        df.to_csv(args.out_csv, index=False)
        print("Saved predictions to:", args.out_csv)
    else:
        print("No --out-csv provided; predictions were printed to console only.")

    if out_annot_dir:
        print("Annotated images saved to:", out_annot_dir)
    print("Done.")

if __name__ == "__main__":
    main()
