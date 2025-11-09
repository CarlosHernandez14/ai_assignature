#!/usr/bin/env python3
"""
Downloads a KaggleHub dataset (using kagglehub.dataset_download),
resizes images to a fixed square size and saves them all into one folder.

Usage example:
python download_resize.py \
    --dataset jessicali9530/stanford-dogs-dataset \
    --out_dir ./data/stanford_100 \
    --max_images 5000 \
    --size 100 \
    --shuffle

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
    """
    Calls kagglehub.dataset_download and returns a Path to the downloaded resource.
    This may be a directory or a zip file depending on kagglehub behaviour.
    """
    print(f"Downloading dataset '{dataset_id}' using kagglehub...")
    path = kagglehub.dataset_download(dataset_id)  # keeps your original call
    p = Path(path)
    print("Downloaded ->", p)
    return p


def extract_if_zip(p: Path, extract_to: Optional[Path] = None) -> Path:
    """
    If p is a zip file, extract it and return the folder where files are extracted.
    If p is already a directory, return p unchanged.
    """
    if p.is_dir():
        return p
    if zipfile.is_zipfile(p):
        dest = extract_to or Path(tempfile.mkdtemp(prefix="dataset_extract_"))
        print(f"Extracting zip {p} -> {dest}")
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(dest)
        return dest
    # not a dir or zip -> return parent (some dataset_download implementations may return a file)
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
            # Convert to RGB to avoid issues with PNG palettes / transparency
            im = im.convert("RGB")
            # Resize using a good resampling filter
            im = im.resize((size, size), resample=Image.LANCZOS)  # type: ignore
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Save as JPEG for uniformity — you can change format if you want
            im.save(out_path, format="JPEG", quality=90)
    except UnidentifiedImageError:
        raise
    except Exception as ex:
        raise


def main():
    parser = argparse.ArgumentParser(description="Download dataset from kagglehub, resize images and collect them in one folder.")
    parser.add_argument("--dataset", "-d", default="jessicali9530/stanford-dogs-dataset", help="KaggleHub dataset id (owner/dataset).")
    parser.add_argument("--out_dir", "-o", required=True, help="Output directory to store resized images.")
    parser.add_argument("--max_images", "-m", type=int, default=0, help="Maximum number of images to save (0 = all).")
    parser.add_argument("--size", "-s", type=int, default=100, help="Size to resize images to (square). Default 100.")
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
            # If dataset_download returned a directory path or a file containing folders
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

    # 5) Resize and save all into out_dir
    failures = 0
    saved = 0
    total = len(image_files)
    pad_width = max(6, len(str(total)))  # zero-pad width for filenames

    print(f"Resizing to {args.size}x{args.size} and saving into {out_dir} ...")
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images", unit="img")):
        out_name = f"{args.prefix}_{idx:0{pad_width}d}.jpg"
        out_path = out_dir / out_name
        if out_path.exists() and args.skip_existing:
            # skip and continue
            continue
        try:
            save_resized(img_path, out_path, args.size)
            saved += 1
        except UnidentifiedImageError:
            failures += 1
            # corrupted or unreadable image: skip
        except Exception as e:
            failures += 1
            # print but continue
            tqdm.write(f"Failed to process {img_path}: {e}")

    print(f"Done. Saved: {saved}. Failures/skipped due to errors: {failures}.")

    # 6) cleanup extracted temp dir if used
    if temp_extract_dir and Path(temp_extract_dir).exists():
        try:
            shutil.rmtree(temp_extract_dir)
        except Exception:
            pass

    print("Output directory contains:", len(list(out_dir.glob("*.jpg"))), "images.")


if __name__ == "__main__":
    main()
