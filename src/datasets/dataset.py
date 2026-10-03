#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py — Người 1: Dataset & EDA
Đề tài: Multi-Scale Features, Attention Gates và Boundary-Aware Loss
        cho phân vùng khuyết tật bề mặt nhỏ (KolektorSDD2 + Magnetic Tile)

Dữ liệu đầu vào (tải từ DatasetNinja, định dạng Supervisely):

    kolektorsdd2-DatasetNinja/                 magnetic-tile-surface-defect-DatasetNinja/
    ├── meta.json                              ├── meta.json
    ├── train/{img, ann}                       └── ds/{img, ann}      (KHÔNG chia sẵn train/test)
    └── test/{img, ann}

  * Ảnh `img/<tên>.<ext>` đi cặp với `ann/<tên>.<ext>.json`.
  * Ann JSON có `size {height, width}` và `objects[]`; mỗi object là một `bitmap`
    (chuỗi base64 của PNG nén zlib + `origin [x, y]`). `objects == []` => ảnh sạch (negative).
  * KSDD2 có 1 lớp ("defect"); Magnetic Tile có 5 lớp (blowhole, break, crack, fray, uneven).

Đầu ra (data_segmentation/):

    data_segmentation/
    ├── train/{ann, img, masks_png}
    ├── val/{ann, img, masks_png}
    └── test/{ann, img, masks_png}

  * `ann/`        : bản sao ann JSON gốc (đổi tên theo sample_id).
  * `img/`        : ảnh gốc, đổi tên theo sample_id (`ksdd2_<id>` / `mt_<id>`).
  * `masks_png/`  : mask nhị phân 0/255 (L-mode), cùng kích thước ảnh, giải mã từ ann.

Các thành phần chính:
  1. Giải mã mask Supervisely (bitmap + polygon)     -> `ann_to_mask`
  2. Kiểm toán cặp ảnh–ann–mask                       -> `scan_dataset`
  3. Phân loại khuyết tật Small / Medium / Large      -> `classify_size`
  4. Chia Train/Val/Test phân tầng                    -> `assign_splits`
  5. Xuất dataset + kiểm tra lại sau khi xuất        -> `export_dataset`, `verify_export`
  6. Augmentation đồng bộ ảnh+mask (online)           -> `CoupledAugmentor`
  7. Boundary map cho Boundary-Aware Loss             -> `extract_boundary`
  8. PyTorch Dataset                                  -> `SurfaceDefectDataset`
"""

from __future__ import annotations

import base64
import io
import json
import random
import shutil
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    HAS_TORCH = True
except ImportError:  # vẫn chạy được EDA khi chưa cài torch
    torch = None
    _TorchDataset = object
    HAS_TORCH = False


# ==============================================================================
# 0. HẰNG SỐ
# ==============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")

# Định nghĩa Small / Medium / Large theo DIỆN TÍCH KHUYẾT TẬT (pixel², tổng các vùng lỗi trong ảnh).
#   Small  : area <  1 024 px²
#   Medium : 1 024 <= area <= 4 096 px²
#   Large  : area >  4 096 px²
# Có thể đổi tại đây; EDA (biểu đồ ECDF) cho thấy các ngưỡng này nằm ở đâu trên phân phối thực.
SMALL_MAX_PX = 1024
MEDIUM_MAX_PX = 4096
SIZE_ORDER = ["Small", "Medium", "Large"]
NEGATIVE_LABEL = "Negative"


def classify_size(area_px: int) -> str:
    """Phân loại quy mô khuyết tật theo tổng diện tích (px²)."""
    if area_px <= 0:
        return NEGATIVE_LABEL
    if area_px < SMALL_MAX_PX:
        return "Small"
    if area_px <= MEDIUM_MAX_PX:
        return "Medium"
    return "Large"


# ==============================================================================
# 1. GIẢI MÃ ANN SUPERVISELY -> MASK NHỊ PHÂN
# ==============================================================================

def decode_bitmap(b64_data: str) -> np.ndarray:
    """
    Giải mã `bitmap.data` của Supervisely: base64 -> zlib -> PNG -> mảng bool (h, w).
    (Đã kiểm chứng trên mẫu thật: PNG mode 'P' với giá trị {0, 1}.)
    """
    raw = zlib.decompress(base64.b64decode(b64_data))
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode in ("RGBA", "LA"):
            arr = np.array(im.getchannel("A"))
        else:
            arr = np.array(im.convert("L"))
    return arr > 0


def _polygon_to_mask(points: dict, h: int, w: int) -> np.ndarray:
    """Polygon Supervisely: {"exterior": [[x, y], ...], "interior": [[[x, y], ...], ...]}."""
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    ext = points.get("exterior", [])
    if len(ext) >= 3:
        draw.polygon([tuple(p) for p in ext], fill=1)
    for hole in points.get("interior", []):
        if len(hole) >= 3:
            draw.polygon([tuple(p) for p in hole], fill=0)
    return np.array(canvas, dtype=bool)


def ann_to_mask(ann: dict, img_w: Optional[int] = None, img_h: Optional[int] = None
                ) -> Tuple[np.ndarray, dict]:
    """
    Dựng mask nhị phân (uint8 0/255) kích thước (h, w) từ một ann Supervisely.

    Trả về (mask, info) với info:
        classes       : list tên lớp xuất hiện (đã loại trùng, giữ thứ tự)
        class_pixels  : {tên lớp: số pixel mask của lớp đó} (object chồng nhau bị đếm lặp)
        num_objects   : số object trong ann
        issues        : list mã lỗi (BITMAP_OUT_OF_BOUNDS, EMPTY_OBJECT_MASK, UNSUPPORTED_GEOMETRY, BAD_BITMAP)
    Kích thước canvas lấy từ `ann["size"]` (nếu thiếu thì dùng img_w/img_h).
    """
    size = ann.get("size", {})
    h = int(size.get("height", img_h or 0))
    w = int(size.get("width", img_w or 0))
    canvas = np.zeros((h, w), dtype=bool)
    info = {"classes": [], "class_pixels": {}, "num_objects": 0, "issues": []}

    for obj in ann.get("objects", []):
        info["num_objects"] += 1
        title = obj.get("classTitle", "defect")
        gtype = obj.get("geometryType")
        obj_mask = np.zeros((h, w), dtype=bool)

        if gtype == "bitmap":
            try:
                bm = decode_bitmap(obj["bitmap"]["data"])
            except Exception:
                info["issues"].append("BAD_BITMAP")
                continue
            ox, oy = obj["bitmap"].get("origin", [0, 0])
            bh, bw = bm.shape
            y1, x1 = min(h, oy + bh), min(w, ox + bw)
            if ox < 0 or oy < 0 or oy + bh > h or ox + bw > w:
                info["issues"].append("BITMAP_OUT_OF_BOUNDS")
            if ox >= 0 and oy >= 0 and y1 > oy and x1 > ox:
                obj_mask[oy:y1, ox:x1] = bm[: y1 - oy, : x1 - ox]
        elif gtype == "polygon":
            obj_mask = _polygon_to_mask(obj.get("points", {}), h, w)
        else:
            info["issues"].append("UNSUPPORTED_GEOMETRY")
            continue

        px = int(obj_mask.sum())
        if px == 0:
            info["issues"].append("EMPTY_OBJECT_MASK")
        if title not in info["classes"]:
            info["classes"].append(title)
        info["class_pixels"][title] = info["class_pixels"].get(title, 0) + px
        canvas |= obj_mask

    info["issues"] = sorted(set(info["issues"]))
    return canvas.astype(np.uint8) * 255, info


def ann_path_for(img_path: Path) -> Path:
    """`img/<tên>.<ext>` -> `ann/<tên>.<ext>.json` (quy ước Supervisely)."""
    return img_path.parent.parent / "ann" / (img_path.name + ".json")


# ==============================================================================
# 2. BOUNDARY MAP + THỐNG KÊ MASK
# ==============================================================================

def extract_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    Bản đồ ranh giới cho Boundary-Aware Loss: morphological gradient = dilate(M) - erode(M).
    Trả về uint8 0/255 cùng kích thước mask.
    """
    m = (mask > 0).astype(np.uint8) * 255
    if not m.any():
        return np.zeros_like(m)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * thickness + 1, 2 * thickness + 1))
    return cv2.subtract(cv2.dilate(m, k), cv2.erode(m, k))


def analyze_mask(mask: np.ndarray) -> dict:
    """Diện tích, số vùng liên thông (8-neighbour), vùng lớn nhất, chu vi, bbox của mask."""
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())
    if area == 0:
        return dict(area_px=0, num_components=0, largest_component_px=0,
                    component_areas=[], perimeter_px=0.0, bbox=None)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comp_areas = sorted((int(a) for a in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    ys, xs = np.where(binary)
    return dict(area_px=area, num_components=len(comp_areas), largest_component_px=comp_areas[0],
                component_areas=comp_areas, perimeter_px=round(perimeter, 2),
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))


# ==============================================================================
# 3. CẤU TRÚC DỮ LIỆU + KIỂM TOÁN
# ==============================================================================

@dataclass
class DatasetSpec:
    name: str                      # "KolektorSDD2" | "MagneticTile"
    prefix: str                    # "ksdd2" | "mt"
    root: Path
    subsets: Dict[str, Path]       # {split gốc: thư mục chứa img/ và ann/}
    known_classes: List[str] = field(default_factory=list)


def make_ksdd2_spec(root: Path) -> DatasetSpec:
    root = Path(root)
    return DatasetSpec("KolektorSDD2", "ksdd2", root,
                       {s: root / s for s in ("train", "test") if (root / s / "img").is_dir()},
                       _read_meta_classes(root))


def make_mt_spec(root: Path) -> DatasetSpec:
    root = Path(root)
    subsets = {}
    for cand in (root / "ds", root):
        if (cand / "img").is_dir():
            subsets["all"] = cand      # Magnetic Tile gốc không chia train/test
            break
    return DatasetSpec("MagneticTile", "mt", root, subsets, _read_meta_classes(root))


def _read_meta_classes(root: Path) -> List[str]:
    try:
        with open(Path(root) / "meta.json", encoding="utf-8") as f:
            return [c["title"] for c in json.load(f).get("classes", [])]
    except Exception:
        return []


@dataclass
class SampleRecord:
    sample_id: str
    source: str
    orig_split: str                # split gốc trong dataset ("train" | "test" | "all")
    img_path: Path
    ann_path: Path
    width: int
    height: int
    ann_width: int
    ann_height: int
    channels: int
    aspect_ratio: float
    is_positive: bool
    classes: List[str]
    primary_class: str             # lớp có nhiều pixel nhất ("none" nếu ảnh sạch)
    class_pixels: Dict[str, int]
    num_objects: int
    area_px: int
    rel_area_pct: float
    num_components: int
    largest_component_px: int
    component_areas: List[int]
    size_category: str             # Small | Medium | Large | Negative
    perimeter_px: float
    bbox: Optional[Tuple[int, int, int, int]]
    issues: List[str]
    split: str = ""                # train | val | test (gán bởi assign_splits)

    @property
    def integrity_status(self) -> str:
        return "OK" if not self.issues else ";".join(self.issues)


def scan_dataset(spec: DatasetSpec) -> Tuple[List[SampleRecord], dict]:
    """
    Kiểm toán toàn bộ một dataset: ghép ảnh–ann, giải mã mask, đo diện tích, kiểm tra toàn vẹn.

    Trả về (records, report). `report` gồm:
        n_images, n_ann_files, missing_ann, orphan_ann, corrupt_images, bad_ann_json, id_collisions
    Mẫu thiếu ann / ảnh hỏng / ann hỏng bị loại khỏi `records` (vì không có nhãn đáng tin).
    Các lỗi còn lại (SIZE_MISMATCH, BITMAP_OUT_OF_BOUNDS, ...) vẫn giữ mẫu và ghi vào `issues`.
    """
    records: List[SampleRecord] = []
    report = dict(n_images=0, n_ann_files=0, missing_ann=[], orphan_ann=[],
                  corrupt_images=[], bad_ann_json=[], id_collisions=0)
    seen_ids = set()

    for orig_split, sub_root in spec.subsets.items():
        img_dir, ann_dir = sub_root / "img", sub_root / "ann"
        img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        ann_files = sorted(ann_dir.glob("*.json")) if ann_dir.is_dir() else []
        report["n_images"] += len(img_files)
        report["n_ann_files"] += len(ann_files)

        img_names = {p.name for p in img_files}
        report["orphan_ann"] += [str(a.relative_to(spec.root)) for a in ann_files
                                 if a.name[:-5] not in img_names]

        for img_path in img_files:
            ann_path = ann_dir / (img_path.name + ".json")
            if not ann_path.exists():
                report["missing_ann"].append(str(img_path.relative_to(spec.root)))
                continue
            try:
                with Image.open(img_path) as im:
                    im.load()
                    w, h = im.size
                    channels = len(im.getbands())
            except Exception:
                report["corrupt_images"].append(str(img_path.relative_to(spec.root)))
                continue
            try:
                with open(ann_path, encoding="utf-8") as f:
                    ann = json.load(f)
                mask, info = ann_to_mask(ann, w, h)
            except Exception:
                report["bad_ann_json"].append(str(ann_path.relative_to(spec.root)))
                continue

            issues = list(info["issues"])
            aw, ah = int(ann["size"]["width"]), int(ann["size"]["height"])
            if (aw, ah) != (w, h):
                issues.append("SIZE_MISMATCH")
            for c in info["classes"]:
                if spec.known_classes and c not in spec.known_classes:
                    issues.append("UNKNOWN_CLASS")
            if info["num_objects"] > 0 and not mask.any():
                issues.append("ALL_OBJECTS_EMPTY")

            st = analyze_mask(mask)
            sample_id = f"{spec.prefix}_{img_path.stem}"
            if sample_id in seen_ids:
                report["id_collisions"] += 1
                sample_id = f"{spec.prefix}_{orig_split}_{img_path.stem}"
            seen_ids.add(sample_id)

            primary = (max(info["class_pixels"], key=info["class_pixels"].get)
                       if info["class_pixels"] and st["area_px"] > 0 else "none")
            records.append(SampleRecord(
                sample_id=sample_id, source=spec.name, orig_split=orig_split,
                img_path=img_path, ann_path=ann_path, width=w, height=h,
                ann_width=aw, ann_height=ah, channels=channels,
                aspect_ratio=round(w / h, 4) if h else 1.0,
                is_positive=st["area_px"] > 0, classes=info["classes"], primary_class=primary,
                class_pixels=info["class_pixels"], num_objects=info["num_objects"],
                area_px=st["area_px"], rel_area_pct=round(100.0 * st["area_px"] / (w * h), 4),
                num_components=st["num_components"], largest_component_px=st["largest_component_px"],
                component_areas=st["component_areas"], size_category=classify_size(st["area_px"]),
                perimeter_px=st["perimeter_px"], bbox=st["bbox"],
                issues=sorted(set(issues)),
            ))
    return records, report


def records_to_dataframe(records: Sequence[SampleRecord], base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Bảng thống kê -> dataset_statistics.csv (đường dẫn quy về tương đối so với base_dir)."""
    def rel(p: Path) -> str:
        try:
            return p.relative_to(base_dir).as_posix() if base_dir else p.as_posix()
        except ValueError:
            return p.as_posix()

    rows = []
    for r in records:
        rows.append({
            "sample_id": r.sample_id, "dataset_source": r.source,
            "orig_split": r.orig_split, "split": r.split,
            "img_path": rel(r.img_path), "ann_path": rel(r.ann_path),
            "width": r.width, "height": r.height, "ann_width": r.ann_width, "ann_height": r.ann_height,
            "channels": r.channels, "aspect_ratio": r.aspect_ratio,
            "is_positive": r.is_positive, "num_objects": r.num_objects,
            "classes": "|".join(r.classes) if r.classes else "none", "primary_class": r.primary_class,
            "defect_area_px": r.area_px, "relative_defect_area_pct": r.rel_area_pct,
            "num_components": r.num_components, "largest_component_px": r.largest_component_px,
            "component_areas": ";".join(map(str, r.component_areas)),
            "size_category": r.size_category, "boundary_perimeter_px": r.perimeter_px,
            "bbox_x1": r.bbox[0] if r.bbox else "", "bbox_y1": r.bbox[1] if r.bbox else "",
            "bbox_x2": r.bbox[2] if r.bbox else "", "bbox_y2": r.bbox[3] if r.bbox else "",
            "integrity_status": r.integrity_status,
        })
    return pd.DataFrame(rows)


def components_dataframe(records: Sequence[SampleRecord]) -> pd.DataFrame:
    """Mỗi dòng = một vùng lỗi liên thông (dùng cho phân tích kích thước ở mức từng khuyết tật)."""
    rows = [{"sample_id": r.sample_id, "dataset_source": r.source, "component_area_px": a,
             "size_category": classify_size(a)}
            for r in records for a in r.component_areas]
    return pd.DataFrame(rows, columns=["sample_id", "dataset_source", "component_area_px", "size_category"])


# ==============================================================================
# 4. KIỂM TRA TRÙNG LẶP GẦN GIỐNG (dHash) — phát hiện rò rỉ dữ liệu giữa các split
# ==============================================================================

def dhash(path: Path, hash_size: int = 16) -> np.ndarray:
    with Image.open(path) as im:
        g = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def find_near_duplicates(records: Sequence[SampleRecord], max_hamming: int = 2) -> List[Tuple[str, str, int]]:
    """
    Tìm cặp ảnh gần trùng (khoảng cách Hamming dHash-256 <= max_hamming) TRONG CÙNG nguồn.
    Chỉ báo cáo, không tự xóa. Trả về list (id_a, id_b, khoảng cách).
    """
    pairs: List[Tuple[str, str, int]] = []
    for src in sorted({r.source for r in records}):
        recs = [r for r in records if r.source == src]
        H = np.stack([dhash(r.img_path) for r in recs]).astype(np.float32)
        agree = H @ H.T + (1 - H) @ (1 - H).T
        dist = H.shape[1] - agree
        iu, ju = np.where(np.triu(dist <= max_hamming, k=1))
        pairs += [(recs[i].sample_id, recs[j].sample_id, int(dist[i, j])) for i, j in zip(iu, ju)]
    return pairs


# ==============================================================================
# 5. CHIA TRAIN / VAL / TEST PHÂN TẦNG
# ==============================================================================

def _stratified_assign(recs: List[SampleRecord], fractions: Dict[str, float], rng: random.Random) -> None:
    """
    Chia phân tầng theo (nguồn, positive/negative, nhóm kích thước, lớp chính).
    Phần dư sau khi làm tròn được phân ngẫu nhiên theo xác suất tỷ lệ với phần lẻ,
    nên tỷ lệ toàn cục vẫn bám sát mục tiêu dù có nhiều nhóm nhỏ.
    """
    total = sum(fractions.values())
    fr = {k: v / total for k, v in fractions.items()}
    groups: Dict[tuple, List[SampleRecord]] = {}
    for r in recs:
        groups.setdefault((r.source, r.is_positive, r.size_category, r.primary_class), []).append(r)

    for key in sorted(groups, key=str):
        grp = sorted(groups[key], key=lambda r: r.sample_id)
        rng.shuffle(grp)
        n = len(grp)
        exact = {k: n * f for k, f in fr.items()}
        count = {k: int(v) for k, v in exact.items()}
        rest = n - sum(count.values())
        remain = {k: exact[k] - count[k] for k in fr}
        for _ in range(rest):
            ks = [k for k in fr if remain[k] > 0] or list(fr)
            pick = rng.choices(ks, weights=[remain[k] if remain[k] > 0 else 1 for k in ks])[0]
            count[pick] += 1
            remain[pick] = 0
        i = 0
        for k in fr:
            for r in grp[i:i + count[k]]:
                r.split = k
            i += count[k]


def assign_splits(records: Sequence[SampleRecord], train_r: float = 0.70, val_r: float = 0.15,
                  test_r: float = 0.15, seed: int = 42, respect_official: bool = True) -> None:
    """
    Gán `record.split` ∈ {train, val, test}.
      * respect_official=True : giữ nguyên tập test chính thức của dataset (KSDD2),
        tách val từ tập train chính thức với tỷ lệ val_r/(train_r+val_r).
      * Dataset không có split gốc (Magnetic Tile: orig_split == "all") hoặc respect_official=False
        => chia phân tầng 3 phần theo train_r/val_r/test_r.
    """
    rng = random.Random(seed)
    train_val, three_way = [], []
    for r in records:
        if respect_official and r.orig_split == "test":
            r.split = "test"
        elif respect_official and r.orig_split == "train":
            train_val.append(r)
        else:
            three_way.append(r)
    _stratified_assign(train_val, {"train": train_r, "val": val_r}, rng)
    _stratified_assign(three_way, {"train": train_r, "val": val_r, "test": test_r}, rng)


# ==============================================================================
# 6. XUẤT DATASET + KIỂM TRA SAU KHI XUẤT
# ==============================================================================

def export_dataset(records: Sequence[SampleRecord], out_dir: Path, include_negatives: bool = False,
                   overwrite: bool = False) -> Dict[str, int]:
    """
    Xuất data_segmentation/{train,val,test}/{img, ann, masks_png}.
    include_negatives=False (mặc định): chỉ xuất ảnh CÓ khuyết tật (bài toán segmentation).
    Thư mục đích đã có dữ liệu mà overwrite=False => báo lỗi (không xóa nhầm dữ liệu của bạn).
    Chỉ xóa 3 thư mục train/val/test bên trong out_dir khi overwrite=True.
    """
    out_dir = Path(out_dir)
    for sp in SPLITS:
        d = out_dir / sp
        if d.exists() and any(d.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{d} đã có dữ liệu. Dùng overwrite=True (--overwrite) để ghi đè.")
            shutil.rmtree(d)
        for sub in ("img", "ann", "masks_png"):
            (d / sub).mkdir(parents=True, exist_ok=True)

    counts = {sp: 0 for sp in SPLITS}
    for r in records:
        if not r.split or (not r.is_positive and not include_negatives):
            continue
        base = out_dir / r.split
        ext = r.img_path.suffix.lower()
        shutil.copy2(r.img_path, base / "img" / f"{r.sample_id}{ext}")
        shutil.copy2(r.ann_path, base / "ann" / f"{r.sample_id}{ext}.json")
        with open(r.ann_path, encoding="utf-8") as f:
            mask, _ = ann_to_mask(json.load(f), r.width, r.height)
        if mask.shape != (r.height, r.width):          # ann lệch kích thước ảnh -> ép về cỡ ảnh
            mask = cv2.resize(mask, (r.width, r.height), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(mask, mode="L").save(base / "masks_png" / f"{r.sample_id}.png")
        counts[r.split] += 1
    return counts


def verify_export(out_dir: Path) -> dict:
    """
    Kiểm tra data_segmentation/: mỗi ảnh có đủ ann + mask; mask cùng kích thước ảnh;
    mask chỉ gồm {0, 255}; mask PNG khớp với mask giải mã lại từ ann.
    """
    out_dir = Path(out_dir)
    res = {"n_checked": 0, "problems": []}
    for sp in SPLITS:
        img_dir = out_dir / sp / "img"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
            res["n_checked"] += 1
            tag = f"{sp}/{img_path.name}"
            ann_p = ann_path_for(img_path)
            mask_p = out_dir / sp / "masks_png" / f"{img_path.stem}.png"
            if not ann_p.exists():
                res["problems"].append((tag, "MISSING_ANN"))
            if not mask_p.exists():
                res["problems"].append((tag, "MISSING_MASK"))
                continue
            with Image.open(img_path) as im:
                size = im.size
            m = np.array(Image.open(mask_p))
            if (m.shape[1], m.shape[0]) != size:
                res["problems"].append((tag, f"SIZE_MISMATCH img={size} mask={(m.shape[1], m.shape[0])}"))
            if not set(np.unique(m)).issubset({0, 255}):
                res["problems"].append((tag, "MASK_NOT_BINARY"))
            if ann_p.exists():
                with open(ann_p, encoding="utf-8") as f:
                    ref, _ = ann_to_mask(json.load(f))
                if ref.shape == m.shape and not np.array_equal(ref, m):
                    res["problems"].append((tag, "MASK_PNG_DIFFERS_FROM_ANN"))
    return res


# ==============================================================================
# 7. AUGMENTATION ĐỒNG BỘ ẢNH + MASK (ONLINE, CHỈ DÙNG CHO TẬP TRAIN)
# ==============================================================================

class CoupledAugmentor:
    """
    Biến đổi hình học áp dụng CÙNG LÚC lên ảnh và mask (mask dùng nội suy nearest => vẫn nhị phân),
    biến đổi quang học (độ sáng/tương phản) chỉ áp dụng lên ảnh.

    Dùng online trong DataLoader (mỗi epoch ra biến thể khác nhau) nên không phình dung lượng đĩa
    và không làm rò rỉ ảnh đã augment sang val/test.
    Đầu vào/ra: image uint8 (H, W, 3), mask uint8 (H, W) với giá trị 0/255.
    """

    def __init__(self, p_hflip=0.5, p_vflip=0.5, p_rot90=0.5, p_affine=0.5,
                 max_rotate_deg=15.0, scale_range=(0.9, 1.1), max_shift=0.05,
                 p_photometric=0.5, brightness=0.2, contrast=0.2, seed: Optional[int] = None):
        self.p_hflip, self.p_vflip, self.p_rot90, self.p_affine = p_hflip, p_vflip, p_rot90, p_affine
        self.max_rotate_deg, self.scale_range, self.max_shift = max_rotate_deg, scale_range, max_shift
        self.p_photometric, self.brightness, self.contrast = p_photometric, brightness, contrast
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = self.rng
        if r.random() < self.p_hflip:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if r.random() < self.p_vflip:
            image, mask = image[::-1], mask[::-1]
        if r.random() < self.p_rot90:
            k = int(r.integers(1, 4))
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)

        if r.random() < self.p_affine:
            h, w = mask.shape
            ang = r.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            sc = r.uniform(*self.scale_range)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            M[0, 2] += r.uniform(-self.max_shift, self.max_shift) * w
            M[1, 2] += r.uniform(-self.max_shift, self.max_shift) * h
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if r.random() < self.p_photometric:
            a = 1.0 + r.uniform(-self.contrast, self.contrast)
            b = r.uniform(-self.brightness, self.brightness) * 255.0
            image = np.clip(image.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return image, mask


# ==============================================================================
# 8. PYTORCH DATASET
# ==============================================================================

class SurfaceDefectDataset(_TorchDataset):
    """
    Đọc data_segmentation/<split>/{img, masks_png}.
    Mỗi mẫu trả về dict:
        image    : float32 (3, H, W) trong [0, 1]
        mask     : float32 (1, H, W) ∈ {0, 1}
        boundary : float32 (1, H, W) ∈ {0, 1}  (cho Boundary-Aware Loss)
        sample_id, is_positive
    Là torch.Tensor nếu đã cài PyTorch, ngược lại là numpy.ndarray.

    target_size = (H, W): bắt buộc nếu batch_size > 1 vì ảnh các nguồn khác kích thước.
    augment=True chỉ nên bật cho split "train".
    sources: lọc theo tiền tố, ví dụ ("ksdd2",) hoặc ("mt",).
    """

    def __init__(self, root_dir: Path, split: str = "train", target_size: Optional[Tuple[int, int]] = None,
                 augment: bool = False, sources: Optional[Sequence[str]] = None,
                 boundary_thickness: int = 1, augmentor: Optional[CoupledAugmentor] = None):
        assert split in SPLITS, f"split phải thuộc {SPLITS}"
        self.dir = Path(root_dir) / split
        files = sorted(p for p in (self.dir / "img").iterdir() if p.suffix.lower() in IMG_EXTS)
        if sources:
            files = [p for p in files if any(p.stem.startswith(s + "_") for s in sources)]
        self.files = files
        self.target_size = target_size
        self.boundary_thickness = boundary_thickness
        self.augmentor = (augmentor or CoupledAugmentor()) if augment else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        p = self.files[idx]
        img = np.array(Image.open(p).convert("RGB"))
        mask = np.array(Image.open(self.dir / "masks_png" / f"{p.stem}.png").convert("L"))
        if self.augmentor:                      # augment ở kích thước gốc (rot90 có thể đổi H<->W)...
            img, mask = self.augmentor(img, mask)
        if self.target_size:                    # ...rồi mới resize => kích thước đầu ra luôn cố định
            th, tw = self.target_size
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        boundary = (extract_boundary(mask, self.boundary_thickness) > 0).astype(np.float32)
        out = {
            "image": (img.astype(np.float32) / 255.0).transpose(2, 0, 1),
            "mask": (mask > 127).astype(np.float32)[None],
            "boundary": boundary[None],
            "sample_id": p.stem,
            "is_positive": bool((mask > 127).any()),
        }
        if HAS_TORCH:
            for k in ("image", "mask", "boundary"):
                out[k] = torch.from_numpy(np.ascontiguousarray(out[k]))
        return out
