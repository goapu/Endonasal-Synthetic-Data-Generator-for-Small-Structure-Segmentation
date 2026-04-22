# -*- coding: utf-8 -*-
"""
Nasal Small-Structure Data Generator
Author: Dilip Goswami, Berlin, Germany

Generates a high-quality synthetic segmentation dataset for nasal endoscopy
by combining object-centered cropping, strict quality control, class-aware
sampling, realistic endoscopic augmentation, and duplicate rejection.

Highlights:
- Targets ~1,300 high-quality samples
- Preserves near-complete object visibility
- Rejects blurry / low-value crops
- Simulates realistic endoscopic imaging conditions
- Produces balanced outputs for small and rare classes
"""

import os
import io
import csv
import sys
import json
import time
import base64
import random
import argparse
import hashlib
import statistics
import unicodedata
import re
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw

DEFAULT_SRC = "/Volumes/T9/Annotations Endonasal AR supervisely/data/Training"
DEFAULT_DST = "/Volumes/T9/Annotations Endonasal AR supervisely/data/SYNTH_GoldStandard"

# ==========================================================
# Precision Targets (~1,300 total)
# ==========================================================
TARGET_CLASSES = [
    "Arc choanal",
    "Opercule du cornet moyen",
    "Cornet inférieur",
    "Cornet moyen",
    "Septum",
]

# Heavier weighting is assigned to small / rare classes.
# Common classes are intentionally limited to reduce imbalance.
JOB_PLAN = {
    "Arc choanal": 450,
    "Opercule du cornet moyen": 400,
    "Cornet moyen": 250,
    "Cornet inférieur": 120,
    "Septum": 100,
}

OUT_SIZE = 512
FG_DECAY = 0.01
MISS_STREAK_BEFORE_ADAPT = 10
MAX_CROPS_PER_IMAGE = 8
REQUIRE_FULLY_INSIDE = True
MIN_OBJ_COVERAGE = 0.98
SEED = 999
USE_DEDUP = True

# Strict sharpness rejection to avoid weak supervision.
BLUR_THRESHOLD = 120.0


# ==========================================================
# Naming Manager
# ==========================================================
class NamingManager:
    """Manage unique left/right output filenames."""

    def __init__(self, dst_root: Path, real_filenames: set):
        self.dst_root = dst_root
        self.real_filenames = real_filenames
        self.next_left = 10001
        self.next_right = 10000
        self._scan_existing_destination()

    def _scan_existing_destination(self):
        """Resume numbering from existing destination folders."""
        left_img_dir = (
            self.dst_root
            / "P100_blockframes_subselection_supervisely"
            / "LEFTNOSE_B1"
            / "img"
        )
        if left_img_dir.exists():
            max_odd = 10001 - 2
            for p in left_img_dir.glob("*.jpg"):
                if p.stem.isdigit():
                    val = int(p.stem)
                    if val % 2 != 0:
                        max_odd = max(max_odd, val)
            if max_odd >= 10001:
                self.next_left = max_odd + 2

        right_img_dir = (
            self.dst_root
            / "P100_blockframes_subselection_supervisely"
            / "RIGHTNOSE_B1"
            / "img"
        )
        if right_img_dir.exists():
            max_even = 10000 - 2
            for p in right_img_dir.glob("*.jpg"):
                if p.stem.isdigit():
                    val = int(p.stem)
                    if val % 2 == 0:
                        max_even = max(max_even, val)
            if max_even >= 10000:
                self.next_right = max_even + 2

    def get_name(self, side: str) -> str:
        """Return the next valid filename for the requested side."""
        side = side.lower().strip()
        if side == "left":
            while True:
                candidate = str(self.next_left)
                if candidate not in self.real_filenames:
                    self.next_left += 2
                    return candidate
                self.next_left += 2
        else:
            while True:
                candidate = str(self.next_right)
                if candidate not in self.real_filenames:
                    self.next_right += 2
                    return candidate
                self.next_right += 2


# ==========================================================
# Utilities
# ==========================================================
def fold(s: str) -> str:
    """Lowercase and remove accents for robust label matching."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", (s or "").lower().strip())
        if not unicodedata.combining(c)
    )


ALIASES = {
    "nasal septum": "septum",
    "septal": "septum",
    "septum nasal": "septum",
    "septa": "septum",
    "inferieur": "inférieur",
}

FUZZY_PATTERNS = [
    (re.compile(r"\bsept(?:um|al|a)\b"), "septum"),
    (re.compile(r"\bsept(?:al)?\s*spur\b"), "septum"),
    (re.compile(r"\bspina\s+septi\b"), "septum"),
    (re.compile(r"\bopercule\b.*\bcornet\b.*\bmoyen\b"), "opercule du cornet moyen"),
    (re.compile(r"\bcornet\b.*\binf(?:erieur|érieur)\b"), "cornet inférieur"),
    (re.compile(r"\bcornet\b.*\bmoyen\b"), "cornet moyen"),
    (re.compile(r"\barc\b.*\bchoanal\b"), "arc choanal"),
]


def canonicalize(folded: str, enable_fuzzy=True, extra_alias=None):
    """Map free-form labels to canonical class names."""
    if extra_alias and folded in extra_alias:
        return extra_alias[folded]
    if folded in ALIASES:
        return ALIASES[folded]
    if enable_fuzzy:
        for rx, canon in FUZZY_PATTERNS:
            if rx.search(folded):
                return canon
    return folded


# Minimum crop size per class to preserve useful detail.
PER_CLASS = {
    "arc choanal": {
        "start_fg": 0.65,
        "min_fg": 0.55,
        "aug_per_crop": 5,
        "min_side": 220,
        "tries": 50,
    },
    "opercule du cornet moyen": {
        "start_fg": 0.62,
        "min_fg": 0.50,
        "aug_per_crop": 6,
        "min_side": 210,
        "tries": 50,
    },
    "cornet inferieur": {
        "start_fg": 0.56,
        "min_fg": 0.45,
        "aug_per_crop": 2,
        "min_side": 180,
        "tries": 20,
    },
    "cornet inférieur": {
        "start_fg": 0.56,
        "min_fg": 0.45,
        "aug_per_crop": 2,
        "min_side": 180,
        "tries": 20,
    },
    "cornet moyen": {
        "start_fg": 0.60,
        "min_fg": 0.48,
        "aug_per_crop": 3,
        "min_side": 200,
        "tries": 30,
    },
    "septum": {
        "start_fg": 0.50,
        "min_fg": 0.40,
        "aug_per_crop": 1,
        "min_side": 180,
        "tries": 20,
    },
}


def _mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)



def _save_json(path: Path, data: dict):
    _mkdir(path.parent)
    json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)



def _read_json(path: Path):
    return json.load(open(path, "r", encoding="utf-8"))



def _downscaled_md5(pil_img, size=64):
    g = pil_img.convert("L").resize((size, size), resample=Image.BILINEAR)
    return hashlib.md5(g.tobytes()).hexdigest()



def _mask_md5_from_shapes(shapes, size_xy):
    """Hash a rasterized version of annotation geometry for deduplication."""
    w, h = size_xy
    m = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(m)

    for s in shapes:
        try:
            if s["geometryType"] == "polygon" and s.get("points"):
                pts = [
                    (int(round(x)), int(round(y)))
                    for x, y in s["points"]
                    if isinstance(x, (int, float)) and isinstance(y, (int, float))
                ]
                if len(pts) >= 3:
                    dr.polygon(pts, fill=255, outline=255)

            elif s["geometryType"] == "rectangle" and s.get("points"):
                (x0, y0), (x1, y1) = s["points"]
                dr.rectangle(
                    [
                        (int(round(x0)), int(round(y0))),
                        (int(round(x1)), int(round(y1))),
                    ],
                    fill=255,
                    outline=255,
                )

            elif s["geometryType"] == "bitmap" and s.get("bitmap", {}).get("data"):
                bm = s["bitmap"]
                msrc = Image.open(io.BytesIO(base64.b64decode(bm["data"]))).convert("L")
                ox, oy = bm.get("origin", [0, 0])
                m.paste(msrc, (int(round(ox)), int(round(oy))), msrc)
        except Exception:
            pass

    m = m.resize((64, 64), Image.NEAREST)
    return hashlib.md5(m.tobytes()).hexdigest()



def clamp(v, lo, hi):
    return max(lo, min(hi, v))



def bbox_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])



def sup_parse_image_ann(ann_path: Path):
    """Parse Supervisely-style annotation JSON."""
    data = _read_json(ann_path)
    img_h = int(data.get("size", {}).get("height") or data.get("imgHeight") or 0)
    img_w = int(data.get("size", {}).get("width") or data.get("imgWidth") or 0)
    tags = [t.get("name") if isinstance(t, dict) else str(t) for t in data.get("tags", [])]
    objs = []

    for obj in data.get("objects", data.get("labels", [])):
        cls = obj.get("classTitle") or obj.get("class_name") or obj.get("title")
        if not cls:
            continue

        geom = obj.get("geometryType") or obj.get("shape") or obj.get("type")
        item = {
            "classTitle": cls,
            "geometryType": None,
            "points": None,
            "bbox": None,
            "shape": None,
        }

        if geom in ("polygon", "polyline") or obj.get("points"):
            pts = obj.get("points", {}).get("exterior") or obj.get("polygon")
            if pts and isinstance(pts, list):
                item["geometryType"] = "polygon"
                clean = []
                for p in pts:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        try:
                            clean.append([float(p[0]), float(p[1])])
                        except Exception:
                            pass
                item["points"] = clean
                xs = [p[0] for p in clean]
                ys = [p[1] for p in clean]
                if xs and ys:
                    item["bbox"] = [
                        max(0, min(xs)),
                        max(0, min(ys)),
                        min(img_w, max(xs)),
                        min(img_h, max(ys)),
                    ]

        elif geom == "rectangle" or obj.get("rectangle"):
            r = obj.get("rectangle")
            if r and isinstance(r, list) and len(r) == 2:
                item["geometryType"] = "rectangle"
                item["points"] = [
                    [float(r[0][0]), float(r[0][1])],
                    [float(r[1][0]), float(r[1][1])],
                ]
                item["bbox"] = [int(r[0][0]), int(r[0][1]), int(r[1][0]), int(r[1][1])]

        elif geom == "bitmap" or obj.get("bitmap"):
            bm = obj.get("bitmap")
            if bm and bm.get("data"):
                item["geometryType"] = "bitmap"
                item["bitmap"] = bm
                b = obj.get("bbox") or [0, 0, img_w, img_h]
                item["bbox"] = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

        else:
            b = obj.get("bbox")
            if b:
                item["geometryType"] = "rectangle"
                item["bbox"] = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

        if item.get("bbox"):
            objs.append(item)

    return img_h, img_w, objs, tags



def pair_image_for_ann(img_dir: Path, ann_path: Path):
    """Find the image corresponding to a given annotation file."""
    stem = ann_path.stem.replace(".jpg", "").replace(".png", "")
    first = stem
    for ext in (".jpg", ".png", ".jpeg", ".tif", ".bmp"):
        c = img_dir / (first + ext)
        if c.exists():
            return c
    hits = list(img_dir.glob(first + ".*"))
    return hits[0] if hits else None



def rasterize_fg_mask_within_crop(E, crop):
    """Rasterize all objects within a crop to estimate foreground density."""
    x0, y0, x1, y1 = crop
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    mask = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(mask)

    for o in E["objects"]:
        try:
            if o["geometryType"] == "polygon" and o.get("points"):
                pts = []
                for p in o["points"]:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        x, y = p
                        pts.append((int(round(x - x0)), int(round(y - y0))))
                if len(pts) >= 3:
                    dr.polygon(pts, outline=255, fill=255)

            elif o["geometryType"] == "rectangle" and o.get("points"):
                (rx0, ry0), (rx1, ry1) = o["points"]
                dr.rectangle(
                    [
                        (int(round(rx0 - x0)), int(round(ry0 - y0))),
                        (int(round(rx1 - x0)), int(round(ry1 - y0))),
                    ],
                    outline=255,
                    fill=255,
                )

            elif o["geometryType"] == "bitmap" and o.get("bitmap"):
                bm = o["bitmap"]
                if bm and bm.get("data"):
                    m = Image.open(io.BytesIO(base64.b64decode(bm["data"]))).convert("L")
                    off = bm.get("origin", [0, 0])
                    ox = int(round(off[0] - x0))
                    oy = int(round(off[1] - y0))
                    mask.paste(m, (ox, oy), m)
        except Exception:
            pass

    return mask



def crop_box_around(bbox, W, H, scale_range=(1.6, 3.2), min_side=150):
    """Create a context-preserving square crop around an object bbox."""
    bx0, by0, bx1, by1 = bbox
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    k = random.uniform(scale_range[0], scale_range[1])
    jitter = int(0.10 * max(bw, bh))
    cx = int((bx0 + bx1) / 2) + random.randint(-jitter, jitter)
    cy = int((by0 + by1) / 2) + random.randint(-jitter, jitter)
    side = int(max(min_side, k * max(bw, bh)))
    x0 = clamp(cx - side // 2, 0, W - 1)
    y0 = clamp(cy - side // 2, 0, H - 1)
    x1 = clamp(x0 + side, 1, W)
    y1 = clamp(y0 + side, 1, H)
    return [x0, y0, x1, y1]


# ==========================================================
# Quality Filter
# ==========================================================
def is_blurry(pil_img: Image.Image, threshold: float) -> bool:
    """Reject low-sharpness samples using edge-variance estimation."""
    g = pil_img.convert("L")
    e = g.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(e, dtype=np.float32)
    var = float(arr.var())
    return var < threshold


# ==========================================================
# Augmentation
# ==========================================================
def endoscopic_augment(img: Image.Image) -> Image.Image:
    """Simulate endoscopic noise, contrast, and brightness variation."""
    aug = img.copy()

    # 1. Subtle Gaussian sensor noise.
    if random.random() < 0.5:
        np_img = np.array(aug)
        noise_level = random.uniform(3, 10)
        noise = np.random.normal(0, noise_level, np_img.shape).astype(np.int16)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        aug = Image.fromarray(np_img)

    # 2. Local contrast adjustment.
    if random.random() < 0.7:
        factor = random.uniform(0.9, 1.2)
        aug = ImageEnhance.Contrast(aug).enhance(factor)

    # 3. Brightness drift from moving light source.
    if random.random() < 0.6:
        factor = random.uniform(0.85, 1.15)
        aug = ImageEnhance.Brightness(aug).enhance(factor)

    return aug



def transform_shapes(shapes, w, h, hflip=False, vflip=False, rot=0, out_size=None):
    """Apply image-space transforms to annotation geometry."""

    def tf_point(x, y, W, H):
        if hflip:
            x = (W - 1) - x
        if vflip:
            y = (H - 1) - y
        if rot % 360 == 90:
            x, y = y, (W - 1) - x
            W, H = H, W
        elif rot % 360 == 180:
            x, y = (W - 1) - x, (H - 1) - y
        elif rot % 360 == 270:
            x, y = (H - 1) - y, x
            W, H = H, W
        return x, y, W, H

    W0, H0 = w, h
    new = []

    for s in shapes:
        t = {"classTitle": s["classTitle"], "geometryType": s["geometryType"]}

        if s["geometryType"] == "polygon" and s.get("points"):
            pts = []
            W, H = W0, H0
            for (x, y) in s["points"]:
                x, y, W, H = tf_point(float(x), float(y), W, H)
                pts.append([x, y])
            t["points"] = pts

        elif s["geometryType"] == "rectangle" and s.get("points"):
            (x0, y0), (x1, y1) = s["points"]
            W, H = W0, H0
            p = []
            for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                x, y, W, H = tf_point(float(x), float(y), W, H)
                p.append([x, y])
            xs = [q[0] for q in p]
            ys = [q[1] for q in p]
            t["points"] = [[min(xs), min(ys)], [max(xs), max(ys)]]

        elif s["geometryType"] == "bitmap" and s.get("bitmap", {}).get("data"):
            bm = dict(s["bitmap"])
            ox, oy = bm.get("origin", [0, 0])
            W, H = W0, H0
            x, y, W, H = tf_point(float(ox), float(oy), W, H)
            bm["origin"] = [x, y]
            t["bitmap"] = bm

        else:
            continue

        new.append(t)

    if out_size:
        sx = out_size / float(W)
        sy = out_size / float(H)
        for s in new:
            if s["geometryType"] == "polygon" and s.get("points"):
                s["points"] = [[x * sx, y * sy] for (x, y) in s["points"]]
            elif s["geometryType"] == "rectangle" and s.get("points"):
                (x0, y0), (x1, y1) = s["points"]
                s["points"] = [[x0 * sx, y0 * sy], [x1 * sx, y1 * sy]]
            elif s["geometryType"] == "bitmap" and s.get("bitmap", {}).get("origin") is not None:
                x, y = s["bitmap"]["origin"]
                s["bitmap"]["origin"] = [x * sx, y * sy]

    return new



def side_from_path(path: Path, tags):
    """Infer left/right side from path or annotation tags."""
    s = str(path)
    if "LEFT" in s.upper():
        return "left"
    if "RIGHT" in s.upper():
        return "right"
    for t in tags or []:
        u = (t or "").lower()
        if "left" in u:
            return "left"
        if "right" in u:
            return "right"
    return "left"


# ==========================================================
# Dataset Scanning
# ==========================================================
def scan_dataset(src_root: Path, classes_whitelist=None, enable_fuzzy=True, extra_alias=None):
    """Scan dataset leaves and normalize annotation class labels."""
    leaves = []
    for p in src_root.rglob("img"):
        if p.is_dir() and (p.parent / "ann").exists():
            leaves.append(p.parent)

    if not leaves:
        print("No leaves with img/ and ann/ found.")
        sys.exit(1)

    entries = []
    counts_norm = Counter()
    real_filenames = set()

    for leaf in leaves:
        img_dir = leaf / "img"
        ann_dir = leaf / "ann"

        for ann_path in sorted([p for p in ann_dir.glob("*.json") if p.is_file()]):
            img_path = pair_image_for_ann(img_dir, ann_path)
            if img_path is None:
                continue

            real_filenames.add(img_path.stem)

            try:
                img_h, img_w, objs, tags = sup_parse_image_ann(ann_path)
            except Exception:
                continue

            if not objs:
                continue

            for o in objs:
                folded = fold(o.get("classTitle") or "")
                folded = canonicalize(
                    folded,
                    enable_fuzzy=enable_fuzzy,
                    extra_alias=extra_alias,
                )
                o["classTitleNorm"] = folded
                counts_norm[folded] += 1

            entries.append(
                {
                    "img_path": str(img_path),
                    "img_w": img_w,
                    "img_h": img_h,
                    "objects": objs,
                    "tags": tags,
                }
            )

    if classes_whitelist:
        wl = set(canonicalize(fold(c)) for c in classes_whitelist)
        counts_norm = Counter({k: v for k, v in counts_norm.items() if k in wl})

    return entries, counts_norm, real_filenames


# ==========================================================
# Generation Engine
# ==========================================================
def generator_for_class(canon_name, target_count, entries, dst_root, registry, cfg, name_manager):
    """Generate target samples for a single canonical class."""
    cnorm = canon_name
    start_fg = cfg.get("start_fg", 0.60)
    min_fg = cfg.get("min_fg", 0.46)
    tries = cfg.get("tries", 20)
    min_side = cfg.get("min_side", 160)
    pool = []

    # 1. Build candidate pool.
    for idx, E in enumerate(entries):
        if E["img_w"] <= 0 or E["img_h"] <= 0:
            continue
        for j, obj in enumerate(E["objects"]):
            if obj.get("classTitleNorm") != cnorm:
                continue
            pool.append((idx, j))

    if not pool:
        return 0

    random.shuffle(pool)

    proj_root = dst_root / "P100_blockframes_subselection_supervisely"
    left_root = proj_root / "LEFTNOSE_B1"
    right_root = proj_root / "RIGHTNOSE_B1"

    for root in (left_root, right_root):
        _mkdir(root / "img")
        _mkdir(root / "ann")
        _mkdir(root / "img_info")

    pair_hashes = registry.setdefault("pair_hashes", set())

    def save_aug(img_pil, shapes, side, hflip=False, vflip=False, rot=0, augment_photo=False):
        """Save one transformed crop + annotation pair if valid and unique."""
        w, h = img_pil.size

        # Apply optional appearance augmentation.
        aug_photo = endoscopic_augment(img_pil) if augment_photo else img_pil.copy()

        aug = aug_photo
        if hflip:
            aug = ImageOps.mirror(aug)
        if vflip:
            aug = ImageOps.flip(aug)
        if rot:
            aug = aug.rotate(rot, expand=True)

        aug_w, aug_h = aug.size
        shapes_tr = transform_shapes(shapes, w=w, h=h, hflip=hflip, vflip=vflip, rot=rot, out_size=None)

        # Resize to standard output resolution.
        aug_resized = aug.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
        shapes_rs = transform_shapes(
            shapes_tr,
            w=aug_w,
            h=aug_h,
            hflip=False,
            vflip=False,
            rot=0,
            out_size=OUT_SIZE,
        )

        # Reject blurry outputs.
        if is_blurry(aug_resized, BLUR_THRESHOLD):
            return False

        ih = _downscaled_md5(aug_resized, size=64)
        mh = _mask_md5_from_shapes(shapes_rs, (OUT_SIZE, OUT_SIZE))
        ph = ih + "|" + mh

        if USE_DEDUP and ph in pair_hashes:
            return False

        base_name = name_manager.get_name(side)
        root = left_root if side == "left" else right_root

        (root / "img" / f"{base_name}.jpg").parent.mkdir(parents=True, exist_ok=True)
        aug_resized.save(root / "img" / f"{base_name}.jpg", quality=100, subsampling=0)
        _save_json(
            root / "ann" / f"{base_name}.json",
            {"size": {"height": OUT_SIZE, "width": OUT_SIZE}, "objects": shapes_rs},
        )
        pair_hashes.add(ph)
        return True

    produced = 0
    dyn = float(start_fg)
    attempts = 0
    it = iter(pool)
    empty_epochs = 0

    while produced < target_count:
        try:
            idx, j = next(it)
        except StopIteration:
            empty_epochs += 1
            if empty_epochs >= 5:
                break
            if dyn > min_fg:
                dyn = max(min_fg, dyn - FG_DECAY)
            random.shuffle(pool)
            it = iter(pool)
            continue

        E = entries[idx]
        try:
            full_img = Image.open(E["img_path"]).convert("RGB")
        except Exception:
            continue

        bbox = E["objects"][j].get("bbox") or [0, 0, E["img_w"], E["img_h"]]

        # Preserve surrounding tissue context.
        cur_lo, cur_hi = 1.4, 2.8
        t = 0

        while t < (tries + 2) and produced < target_count:
            t += 1
            attempts += 1
            crop = crop_box_around(
                bbox,
                E["img_w"],
                E["img_h"],
                scale_range=(cur_lo, cur_hi),
                min_side=min_side,
            )
            x0, y0, x1, y1 = crop

            # Reject crops that truncate the target structure.
            bx0, by0, bx1, by1 = bbox
            visible_w = max(0, min(x1, bx1) - max(x0, bx0))
            visible_h = max(0, min(y1, by1) - max(y0, by0))
            vis_area = visible_w * visible_h
            full_area = max(1, bbox_area(bbox))
            cov = vis_area / float(full_area)

            if REQUIRE_FULLY_INSIDE and cov < 0.98:
                continue

            # Estimate foreground density inside the crop.
            fg_mask = rasterize_fg_mask_within_crop(E, crop)
            fg_arr = np.asarray(fg_mask, dtype=np.uint8)
            fg_ratio = float((fg_arr > 0).sum()) / float(max(1, fg_arr.size))
            if fg_ratio < dyn:
                dyn = max(min_fg, dyn - FG_DECAY)
                continue

            # Extract only target-class shapes relative to crop origin.
            shapes = []
            for s in E["objects"]:
                if s.get("classTitleNorm") != cnorm:
                    continue

                if s["geometryType"] == "polygon" and s.get("points"):
                    pts = [[p[0] - x0, p[1] - y0] for p in s["points"] if len(p) == 2]
                    if len(pts) >= 3:
                        shapes.append(
                            {
                                "classTitle": s["classTitle"],
                                "geometryType": "polygon",
                                "points": pts,
                            }
                        )

                elif s["geometryType"] == "rectangle" and s.get("points"):
                    (rx0, ry0), (rx1, ry1) = s["points"]
                    shapes.append(
                        {
                            "classTitle": s["classTitle"],
                            "geometryType": "rectangle",
                            "points": [[rx0 - x0, ry0 - y0], [rx1 - x0, ry1 - y0]],
                        }
                    )

                elif s["geometryType"] == "bitmap" and s.get("bitmap"):
                    bm = dict(s["bitmap"])
                    ox, oy = bm.get("origin", [0, 0])
                    bm["origin"] = [ox - x0, oy - y0]
                    shapes.append(
                        {
                            "classTitle": s["classTitle"],
                            "geometryType": "bitmap",
                            "bitmap": bm,
                        }
                    )

            if not shapes:
                continue

            side = side_from_path(Path(E["img_path"]), E.get("tags", []))
            crop_img = full_img.crop((x0, y0, x1, y1))

            # 1. Save the clean crop.
            if save_aug(crop_img, shapes, side, augment_photo=False):
                produced += 1

            # 2. Save augmented variants.
            for _ in range(cfg.get("aug_per_crop", 3)):
                if produced >= target_count:
                    break
                hflip = random.random() < 0.5
                vflip = random.random() < 0.3
                rot = random.choice([0, 90, 180, 270])
                if save_aug(crop_img, shapes, side, hflip, vflip, rot, augment_photo=True):
                    produced += 1

            if attempts % 500 == 0:
                print(f"[{cnorm}] progress: {produced}/{target_count}")

    return produced


# ==========================================================
# Main
# ==========================================================
def main():
    ap = argparse.ArgumentParser("Gold Standard Data Generator v4.0")
    ap.add_argument("--src", type=str, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=str, default=DEFAULT_DST)
    ap.add_argument("--no-fuzzy", action="store_true")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    proj_root = dst_root / "P100_blockframes_subselection_supervisely"
    _mkdir(proj_root)

    # Initialize deduplication registry.
    reg_path = proj_root / "registry.json"
    registry = {"pair_hashes": set()}
    if USE_DEDUP and reg_path.exists():
        try:
            registry["pair_hashes"] = set(_read_json(reg_path).get("pair_hashes", []))
        except Exception:
            pass

    print("[info] Scanning dataset ...")
    entries, counts_norm, real_filenames = scan_dataset(
        src_root,
        classes_whitelist=TARGET_CLASSES,
        enable_fuzzy=(not args.no_fuzzy),
    )
    name_manager = NamingManager(dst_root, real_filenames)

    plan = {
        canonicalize(fold(k), enable_fuzzy=(not args.no_fuzzy)): v
        for k, v in JOB_PLAN.items()
    }
    per_class_counts = {}

    for canon, target in plan.items():
        cfg = PER_CLASS.get(canon, {})
        print(f"\n[run] Class '{canon}': target {target} (Quality Mode)")
        n = generator_for_class(canon, int(target), entries, dst_root, registry, cfg, name_manager)
        per_class_counts[canon] = int(n)
        try:
            _save_json(reg_path, {"pair_hashes": list(registry["pair_hashes"])})
        except Exception:
            pass

    summary = {"total": sum(per_class_counts.values()), "per_class": per_class_counts}
    _save_json(proj_root / "dataset_summary.json", summary)

    print("\n=== DONE (Gold Standard v4.0) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
