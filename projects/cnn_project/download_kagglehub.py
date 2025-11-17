#!/usr/bin/env python3
"""
Downloads a KaggleHub dataset (using kagglehub.dataset_download),
optionally resizes images to a fixed square size and saves them all into one folder.

Behavior:
 - If --size is provided: images are resized to SIZE x SIZE and saved as JPEG.
 - If --size is NOT provided: images are copied as-is into the output folder (original extensions preserved).

Usage examples:
  # Resize to 100x100 (same as original behavior)
  python download_resize_optional_size.py --dataset jessicali9530/stanford-dogs-dataset --out_dir ./data/stanford_100 --size 100 --max_images 5000 --shuffle

  # Do NOT resize: just copy the original images to out_dir
  python download_resize_optional_size.py --dataset jessicali9530/stanford-dogs-dataset --out_dir ./data/stanford_orig --max_images 5000 --shuffle

Requires:
    pip install kagglehub pillow tqdm
"""
import argparse
import os
import zipfile
import tempfile
import shutil
import random
from pathlib import Path
from typing import Optional, List

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import kagglehub  # assumes kagglehub is configured on your machine

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


def download_dataset(dataset_id: str) -> Path:
    print(f"Downloading dataset '{dataset_id}' using kagglehub...")
    path = kagglehub.dataset_download(dataset_id)
    p = Path(path)
    print("Downloaded ->", p)
    return p


def extract_if_zip(p: Path, extract_to: Optional[Path] = None) -> Path:
    if p.is_dir():
        return p
    if zipfile.is_zipfile(p):
        dest = extract_to or Path(tempfile.mkdtemp(prefix="dataset_extract_"))
        print(f"Extracting zip {p} -> {dest}")
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(dest)
        return dest
    return p.parent


def find_image_files(root: Path) -> List[Path]:
    files = []
    for file in root.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMG_EXTS:
            files.append(file)
    return files


def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def save_resized(image_path: Path, out_path: Path, size: int):
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im = im.resize((size, size), resample=Image.LANCZOS) # type:ignore
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, format="JPEG", quality=90)
    except UnidentifiedImageError:
        raise
    except Exception as ex:
        raise


def copy_original(image_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use copy2 to preserve metadata if desired
    shutil.copy2(image_path, out_path)


def main():
    parser = argparse.ArgumentParser(description="Download dataset from kagglehub, optionally resize images and collect them in one folder.")
    parser.add_argument("--dataset", "-d", default="jessicali9530/stanford-dogs-dataset", help="KaggleHub dataset id (owner/dataset).")
    parser.add_argument("--out_dir", "-o", required=True, help="Output directory to store images.")
    parser.add_argument("--max_images", "-m", type=int, default=0, help="Maximum number of images to save (0 = all).")
    parser.add_argument("--size", "-s", type=int, default=None, help="If provided, resize images to SIZE x SIZE (square) and save as JPEG. If omitted, images are copied as-is.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle image ordering before selecting max_images.")
    parser.add_argument("--skip_existing", action="store_true", help="If true, skip saving if an output filename already exists.")
    parser.add_argument("--prefix", default="img", help="Prefix for output files (default 'img').")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    # 1) Download
    downloaded = None
    try:
        downloaded = download_dataset(args.dataset)
    except Exception as e:
        print("ERROR: dataset_download failed:", e)
        return

    # 2) Extract if needed
    extracted_dir = None
    temp_extract_dir = None
    try:
        if zipfile.is_zipfile(downloaded):
            temp_extract_dir = Path(tempfile.mkdtemp(prefix="kagglehub_extracted_"))
            extracted_dir = extract_if_zip(downloaded, temp_extract_dir)
        else:
            extracted_dir = extract_if_zip(Path(downloaded))
    except Exception as e:
        print("ERROR during extraction:", e)
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return

    # 3) Find all image files recursively
    print("Scanning for image files under:", extracted_dir)
    image_files = find_image_files(Path(extracted_dir))
    print(f"Found {len(image_files)} image files.")

    if len(image_files) == 0:
        print("No images found. Exiting.")
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return

    # 4) Optionally shuffle and limit
    if args.shuffle:
        random.shuffle(image_files)

    max_images = args.max_images if args.max_images and args.max_images > 0 else None
    if max_images:
        image_files = image_files[:max_images]

    # 5) Resize (if size provided) OR copy originals
    failures = 0
    saved = 0
    total = len(image_files)
    pad_width = max(6, len(str(total)))  # zero-pad width for filenames
    resizing = args.size is not None

    mode_desc = f"Resizing to {args.size}x{args.size} and saving as JPEG" if resizing else "Copying originals without resizing"
    print(f"{mode_desc} into {out_dir} ...")

    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images", unit="img")):
        if resizing:
            out_name = f"{args.prefix}_{idx:0{pad_width}d}.jpg"
            out_path = out_dir / out_name
        else:
            # preserve original extension
            ext = img_path.suffix.lower()
            out_name = f"{args.prefix}_{idx:0{pad_width}d}{ext}"
            out_path = out_dir / out_name

        if out_path.exists() and args.skip_existing:
            continue

        try:
            if resizing:
                save_resized(img_path, out_path, args.size)
            else:
                # copy original file as-is
                copy_original(img_path, out_path)
            saved += 1
        except UnidentifiedImageError:
            failures += 1
        except Exception as e:
            failures += 1
            tqdm.write(f"Failed to process {img_path}: {e}")

    print(f"Done. Saved: {saved}. Failures/errors: {failures}.")

    # 6) cleanup extracted temp dir if used
    if temp_extract_dir and Path(temp_extract_dir).exists():
        try:
            shutil.rmtree(temp_extract_dir)
        except Exception:
            pass

    # Count images in out_dir with any of the accepted extensions (including jpg)
    out_count = sum(1 for _ in out_dir.rglob("*") if _.is_file() and _.suffix.lower() in IMG_EXTS.union({".jpg"}))
    print("Output directory contains:", out_count, "images.")


if __name__ == "__main__":
    main()
