#!/usr/bin/env python3
"""
download_and_resize_inat_resume_indexed.py

Resume-capable iNaturalist downloader.

Behavior:
- If --size is omitted (None) the script will NOT resize images; it saves the original bytes.
- Script will attempt to fetch a larger variant when the chosen URL appears to be a square/thumbnail
  (e.g. contains "square" or "_square") by trying common replacements to "large".
"""
import argparse
import os
import time
import csv
import requests
import re
import glob
from io import BytesIO
from PIL import Image

API_URL = "https://api.inaturalist.org/v1/observations"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--taxon", type=int, default=47336)
    p.add_argument("--out_dir", type=str, required=True)
    # If size is omitted (None), do NOT resize and save original files
    p.add_argument("--size", type=int, default=None, help="If omitted, images are downloaded at original size (no resize).")
    p.add_argument("--method", choices=["pad","crop"], default="pad")
    p.add_argument("--bg", nargs=3, type=int, default=[0,0,0])
    p.add_argument("--licenses", nargs="+", default=["cc0","cc-by","cc-by-sa"])
    p.add_argument("--quality", choices=["research","any"], default="research")
    p.add_argument("--per_page", type=int, default=200)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--min_accept_size", type=int, default=64)
    p.add_argument("--sleep_image", type=float, default=0.12)
    p.add_argument("--sleep_page", type=float, default=0.6)
    p.add_argument("--start_id_below", type=int, default=0,
                   help="Optional start id_below. If 0, script will infer from manifest (resume).")
    p.add_argument("--start_index", type=int, default=0,
                   help="Sequential filename start index. If 0 auto-detect next free index in resized dir.")
    return p.parse_args()

# ---------- helpers ----------
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def best_photo_url(photo):
    # keep original preferred ordering
    for k in ("original_url","large_url","medium_url","url","square_url"):
        v = photo.get(k)
        if v: return v
    return None

def try_upscale_url(url):
    """
    Try common replacements that produce a larger image variant when the URL contains
    'square' or thumbnail patterns. Return upgraded url or None.
    """
    if not url:
        return None
    # Quick list of replacements to try (ordered)
    reps = [
        ("square","large"),
        ("_square","_large"),
        ("/square/","/large/"),
        ("/thumb/","/large/"),  # some APIs use /thumb/
        ("=square","=large"),
        ("-square","-large"),
    ]
    for a,b in reps:
        if a in url:
            candidate = url.replace(a,b)
            if candidate != url:
                return candidate
    return None

def download_bytes(url, timeout=20):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.content
        return None
    except Exception:
        return None

def image_max_dim_from_bytes(bts):
    try:
        im = Image.open(BytesIO(bts))
        mx = max(im.size)
        im.close()
        return mx
    except Exception:
        return 0

def get_image_info_from_bytes(bts):
    """
    Return (width, height, format_string). format_string can be None.
    """
    try:
        im = Image.open(BytesIO(bts))
        w,h = im.size
        fmt = im.format  # e.g. 'JPEG', 'PNG', or None
        im.close()
        return w,h,fmt
    except Exception:
        return None,None,None

def resize_pad_image_bytes(bts, size, bg_rgb):
    im = Image.open(BytesIO(bts)).convert("RGB")
    w,h = im.size
    scale = size / max(w,h)
    new_w = max(1, int(round(w * scale))); new_h = max(1, int(round(h * scale)))
    im_resized = im.resize((new_w,new_h), Image.LANCZOS) # type: ignore
    canvas = Image.new("RGB", (size,size), tuple(bg_rgb))
    canvas.paste(im_resized, ((size-new_w)//2, (size-new_h)//2))
    im.close()
    return canvas

def resize_crop_image_bytes(bts, size):
    im = Image.open(BytesIO(bts)).convert("RGB")
    w,h = im.size
    if w==h:
        im_c = im
    elif w>h:
        left=(w-h)//2; im_c = im.crop((left,0,left+h,h))
    else:
        top=(h-w)//2; im_c = im.crop((0,top,w,top+w))
    out = im_c.resize((size,size), Image.LANCZOS) # type: ignore
    im.close()
    return out

# read manifest to get processed keys and infer starting id_below
def read_manifest(manifest_path):
    processed = set()
    last_ids = []
    if not os.path.exists(manifest_path):
        return processed, None
    with open(manifest_path, newline='', encoding='utf8') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            try:
                oid = int(r.get("obs_id") or 0)
                pid = int(r.get("photo_id") or 0)
            except:
                oid = pid = 0
            if oid and pid:
                processed.add((oid,pid))
                last_ids.append(oid)
    min_id = min(last_ids) if last_ids else None
    return processed, min_id

def append_manifest_row(manifest_path, rowdict, fieldnames):
    write_header = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header: writer.writeheader()
        writer.writerow(rowdict)

# find next free sequential index in resized_dir
def detect_next_index(resized_dir, provided_start_index=0, pad_digits=6):
    existing = sorted(os.listdir(resized_dir)) if os.path.exists(resized_dir) else []
    num_pattern = re.compile(r'(\d{1,})\.(?:jpg|jpeg|png|webp)$', re.IGNORECASE)
    found_nums = []
    for f in existing:
        m = num_pattern.match(f)
        if m:
            try:
                found_nums.append(int(m.group(1)))
            except:
                pass
    max_found = max(found_nums) if found_nums else 0
    if provided_start_index and provided_start_index > 0:
        start = provided_start_index if provided_start_index > max_found else max_found + 1
    else:
        start = max_found + 1
    return start

def safe_get_next_filename(resized_dir, idx, digits=6, preferred_ext="jpg"):
    while True:
        base = f"{idx:0{digits}d}"
        matches = glob.glob(os.path.join(resized_dir, base + ".*"))
        if not matches:
            name = f"{base}.{preferred_ext}"
            return idx, name
        idx += 1

def choose_extension_from_format(fmt):
    if not fmt:
        return "jpg"
    fmt = fmt.upper()
    if fmt in ("JPEG","JPG"):
        return "jpg"
    if fmt == "PNG":
        return "png"
    if fmt == "WEBP":
        return "webp"
    return "jpg"

def save_original_bytes_to_path(bts, out_path, image_format):
    if image_format:
        try:
            with open(out_path, "wb") as fh:
                fh.write(bts)
            return True
        except Exception:
            pass
    try:
        im = Image.open(BytesIO(bts)).convert("RGB")
        im.save(out_path, quality=95)
        im.close()
        return True
    except Exception:
        return False

def main():
    args = parse_args()
    args.licenses = [l.lower() for l in args.licenses]
    ensure_dir(args.out_dir)
    resized_dir = os.path.join(args.out_dir, "resized")
    ensure_dir(resized_dir)
    manifest = os.path.join(args.out_dir, "manifest.csv")
    processed_set, min_obs_in_manifest = read_manifest(manifest)

    fieldnames = ["out_filename","obs_id","photo_id","taxon_id","taxon_name",
                  "orig_url","final_url","orig_max_dim","final_size","license","status","note"]

    id_below = args.start_id_below if args.start_id_below else (min_obs_in_manifest or 0)
    if id_below:
        print("Resuming with id_below =", id_below)
    else:
        print("No id_below provided; starting from newest obs.")

    file_index = detect_next_index(resized_dir, provided_start_index=args.start_index, pad_digits=6)
    print(f"Starting sequential filenames from index {file_index} (files in {resized_dir} will not be overwritten).")

    downloaded = 0
    session = requests.Session()
    consecutive_empty = 0

    while True:
        params = {"taxon_id": args.taxon, "per_page": args.per_page, "photos": "true"}
        if args.quality == "research": params["quality_grade"] = "research"
        if id_below:
            params["id_below"] = id_below

        try:
            resp = session.get(API_URL, params=params, timeout=30)
        except Exception as e:
            print("Request error:", e); time.sleep(5); continue

        if resp.status_code == 429:
            print("429 rate limit — sleeping 30s")
            time.sleep(30)
            continue
        if resp.status_code == 403:
            print("403 error (window too large?) — reduce per_page or use id_below properly. Resp:", resp.text)
            break
        if resp.status_code != 200:
            print("HTTP", resp.status_code, resp.text); break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            print("No more results from API. Exiting.")
            break

        obs_ids = [r.get("id") for r in results if isinstance(r.get("id"), int)]
        if not obs_ids:
            print("No obs ids in results — stopping.")
            break
        next_id_below = min(obs_ids)

        new_found_in_batch = 0
        for obs in results:
            obs_id = obs.get("id")
            taxon = obs.get("taxon") or {}
            taxon_id = taxon.get("id")
            taxon_name = taxon.get("name")
            obs_photos = obs.get("observation_photos") or obs.get("photos") or []
            for op in obs_photos:
                photo = op.get("photo") if isinstance(op, dict) and op.get("photo") else op
                if not isinstance(photo, dict):
                    continue
                photo_id = photo.get("id")
                if (obs_id, photo_id) in processed_set:
                    continue
                license_code = (photo.get("license_code") or obs.get("license_code") or "").lower()
                if not license_code or license_code not in args.licenses:
                    append_manifest_row(manifest, {
                        "out_filename":None,"obs_id":obs_id,"photo_id":photo_id,"taxon_id":taxon_id,"taxon_name":taxon_name,
                        "orig_url":photo.get("url"),"final_url":None,"orig_max_dim":None,"final_size":None,"license":license_code,
                        "status":"skipped-license","note":"license not allowed"
                    }, fieldnames)
                    processed_set.add((obs_id,photo_id))
                    continue

                url = best_photo_url(photo)
                if not url:
                    append_manifest_row(manifest, {
                        "out_filename":None,"obs_id":obs_id,"photo_id":photo_id,"taxon_id":taxon_id,"taxon_name":taxon_name,
                        "orig_url":None,"final_url":None,"orig_max_dim":None,"final_size":None,"license":license_code,
                        "status":"no_url","note":"no url fields"
                    }, fieldnames)
                    processed_set.add((obs_id,photo_id))
                    continue

                # download the chosen url
                bts = download_bytes(url)
                final_url = url
                orig_w, orig_h, orig_fmt = (None, None, None)
                orig_max = 0
                if bts:
                    ow,oh,ofmt = get_image_info_from_bytes(bts)
                    if ow and oh:
                        orig_w, orig_h, orig_fmt = ow, oh, ofmt
                        orig_max = max(ow,oh)

                # If the chosen URL appears to be a square thumbnail (contains 'square' or '_square')
                # attempt to fetch a larger variant by common replacements — use it if it's larger.
                # This is the important change to avoid saving tiny thumbnails like 75x75 when larger variants exist.
                if bts and ("square" in url or "_square" in url or "/thumb/" in url):
                    alt = try_upscale_url(url)
                    if alt:
                        alt_bts = download_bytes(alt)
                        if alt_bts:
                            alt_ow,alt_oh,alt_fmt = get_image_info_from_bytes(alt_bts)
                            alt_max = max(alt_ow or 0, alt_oh or 0)
                            if alt_max > orig_max:
                                bts = alt_bts
                                final_url = alt
                                orig_w,orig_h,orig_fmt = alt_ow,alt_oh,alt_fmt
                                orig_max = alt_max

                # If not found or orig_max smaller than min_accept_size, keep previous behavior: try upscale anyway
                if not bts or orig_max < args.min_accept_size:
                    alt = try_upscale_url(url)
                    if alt:
                        alt_bts = download_bytes(alt)
                        alt_ow, alt_oh, alt_fmt = (None,None,None)
                        alt_max = 0
                        if alt_bts:
                            alt_ow,alt_oh,alt_fmt = get_image_info_from_bytes(alt_bts)
                            alt_max = max(alt_ow or 0, alt_oh or 0)
                        if alt_bts and alt_max >= orig_max:
                            bts = alt_bts; final_url = alt
                            orig_w, orig_h, orig_fmt = alt_ow, alt_oh, alt_fmt
                            orig_max = alt_max

                if not bts:
                    append_manifest_row(manifest, {
                        "out_filename":None,"obs_id":obs_id,"photo_id":photo_id,"taxon_id":taxon_id,"taxon_name":taxon_name,
                        "orig_url":url,"final_url":final_url,"orig_max_dim":None,"final_size":None,"license":license_code,
                        "status":"download_failed","note":"could not download"
                    }, fieldnames)
                    processed_set.add((obs_id,photo_id))
                    continue

                if orig_max < args.min_accept_size:
                    append_manifest_row(manifest, {
                        "out_filename":None,"obs_id":obs_id,"photo_id":photo_id,"taxon_id":taxon_id,"taxon_name":taxon_name,
                        "orig_url":url,"final_url":final_url,"orig_max_dim":orig_max,"final_size":None,"license":license_code,
                        "status":"skipped-small","note":"below min size"
                    }, fieldnames)
                    processed_set.add((obs_id,photo_id))
                    continue

                # If size is provided: resize. If size is None: save original bytes.
                if args.size:
                    try:
                        if args.method == "pad":
                            out_im = resize_pad_image_bytes(bts, args.size, args.bg)
                        else:
                            out_im = resize_crop_image_bytes(bts, args.size)
                    except Exception as e:
                        append_manifest_row(manifest, {
                            "out_filename":None,"obs_id":obs_id,"photo_id":photo_id,"taxon_id":taxon_id,"taxon_name":taxon_name,
                            "orig_url":url,"final_url":final_url,"orig_max_dim":orig_max,"final_size":None,"license":license_code,
                            "status":"resize_error","note":str(e)
                        }, fieldnames)
                        processed_set.add((obs_id,photo_id))
                        continue

                    file_index, out_fname = safe_get_next_filename(resized_dir, file_index, digits=6, preferred_ext="jpg")
                    out_path = os.path.join(resized_dir, out_fname)

                    try:
                        out_im.save(out_path, quality=95)
                    except Exception as e:
                        append_manifest_row(manifest, {
                            "out_filename":None,"obs_id":obs_id,"photo_id":photo_id,"taxon_id":taxon_id,"taxon_name":taxon_name,
                            "orig_url":url,"final_url":final_url,"orig_max_dim":orig_max,"final_size":None,"license":license_code,
                            "status":"save_error","note":str(e)
                        }, fieldnames)
                        processed_set.add((obs_id,photo_id))
                        file_index += 1
                        continue

                    final_size_str = f"{args.size}x{args.size}"

                else:
                    # No resize: save original bytes and preserve extension when possible
                    ext = choose_extension_from_format(orig_fmt)
                    file_index, out_fname = safe_get_next_filename(resized_dir, file_index, digits=6, preferred_ext=ext)
                    out_path = os.path.join(resized_dir, out_fname)

                    success = save_original_bytes_to_path(bts, out_path, orig_fmt)
                    if not success:
                        append_manifest_row(manifest, {
                            "out_filename":None,"obs_id":obs_id,"photo_id":photo_id,"taxon_id":taxon_id,"taxon_name":taxon_name,
                            "orig_url":url,"final_url":final_url,"orig_max_dim":orig_max,"final_size":None,"license":license_code,
                            "status":"save_error","note":"could not save original bytes"
                        }, fieldnames)
                        processed_set.add((obs_id,photo_id))
                        file_index += 1
                        continue

                    final_size_str = f"{orig_w}x{orig_h}" if orig_w and orig_h else None

                append_manifest_row(manifest, {
                    "out_filename":out_fname,"obs_id":obs_id,"photo_id":photo_id,"taxon_id":taxon_id,"taxon_name":taxon_name,
                    "orig_url":url,"final_url":final_url,"orig_max_dim":orig_max,"final_size":final_size_str,
                    "license":license_code,"status":"downloaded_resized" if args.size else "downloaded_original","note":""
                }, fieldnames)

                processed_set.add((obs_id,photo_id))
                downloaded += 1
                new_found_in_batch += 1
                file_index += 1

                if args.max_images and downloaded >= args.max_images:
                    print("Reached requested max_images:", args.max_images)
                    return

                time.sleep(args.sleep_image)

        id_below = next_id_below
        if new_found_in_batch == 0:
            consecutive_empty += 1
            if consecutive_empty >= 5:
                print("No new items in 5 consecutive batches — stopping.")
                break
        else:
            consecutive_empty = 0

        time.sleep(args.sleep_page)

    print("Complete. Resized/images in:", resized_dir)
    print("Manifest:", manifest)

if __name__ == "__main__":
    main()
