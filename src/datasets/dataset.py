#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: dataset.py
Đề tài: Nghiên cứu về Multi-Scale Features, Attention Gates và Boundary-Aware Loss
        cho bài toán phân vùng khuyết tật bề mặt nhỏ trên KolektorSDD2
Phân hệ: Người 1 — Dataset & EDA (SEMANTIC SEGMENTATION - DEFECT ONLY PIPELINE)

Quy chuẩn xử lý:
1. LỌC 100% MẪU CÓ DEFECT:
   - Bài toán là Semantic Segmentation, loại bỏ hoàn toàn ảnh sạch (Negative).
   - Chỉ giữ lại ảnh chứa khuyết tật thực sự (has_defect == True).
2. MASK ĐỊNH DẠNG LABELME JSON:
   - Mask được lưu trữ ở định dạng chuẩn LabelMe JSON (.json).
   - File JSON chứa polygons (points: [[x, y], ...]), label: "defect", imageHeight, imageWidth, v.v.
   - Hỗ trợ lọc mẫu dựa trên metadata trong file JSON (kiểm tra shapes có chứa nhãn "defect").
   - Kèm thư mục masks_png/ phục vụ nạp trực tiếp vào PyTorch DataLoader tốc độ cao.
3. ĐẶT TÊN ĐỒNG NHẤT VỚI TIỀN TỐ NGUỒN:
   - KolektorSDD2: `ksdd2_defect_0001.png` kèm `ksdd2_defect_0001.json`
   - Magnetic Tile: `mt_<defect_type>_defect_0001.png` kèm `mt_<defect_type>_defect_0001.json`
4. HỖ TRỢ BOUNDARY-AWARE LOSS:
   - Trích xuất bản đồ ranh giới/đường biên (Boundary Edge Map) trực tiếp từ mask.
5. PHÂN LOẠI KHUYẾT TẬT NHỎ (SMALL DEFECTS):
   - Small: < 1,024 px² (< 0.5% diện tích)
   - Medium: 1,024 ~ 4,096 px²
   - Large: > 4,096 px²
6. STRATIFIED SPLIT:
   - Phân chia phân tầng 70% Train - 15% Val - 15% Test trên tập khuyết tật.
7. COUPLED AUGMENTATION:
   - Tăng cường đồng bộ (Ảnh + Mask) trên tập Train.
"""

import os
import json
import shutil
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass


# ==============================================================================
# 1. ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU
# ==============================================================================

@dataclass
class DefectPolygonShape:
    label: str
    points: List[List[float]]   # [[x1, y1], [x2, y2], ...]
    area_pixels: int
    category: str               # 'Small', 'Medium', 'Large'


@dataclass
class DefectSegmentationInfo:
    total_area_pixels: int
    relative_area_pct: float
    category: str
    num_components: int
    boundary_perimeter_px: float
    shapes: List[DefectPolygonShape] = field(default_factory=list)


@dataclass
class SampleRecord:
    sample_id: str
    dataset_source: str          # 'KolektorSDD2' hoặc 'MagneticTile'
    image_path: Path
    mask_path: Optional[Path]
    width: int
    height: int
    channels: int
    aspect_ratio: float
    has_defect: bool
    defect_info: DefectSegmentationInfo
    image_hash: Optional[np.ndarray] = None
    split: str = "train"
    integrity_status: str = "OK"


# ==============================================================================
# 2. XỬ LÝ MASK LABELME JSON (POLYGON CONVERTER & METADATA AUDITOR)
# ==============================================================================

class LabelMeJsonConverter:
    """
    Chuyển đổi hai chiều giữa Mask nhị phân (Binary Mask) và định dạng LabelMe JSON.
    Hỗ trợ kiểm tra metadata trong file JSON để xác định mẫu có lỗi (Defect).
    """
    @staticmethod
    def mask_to_polygons(mask_np: np.ndarray, min_area: int = 4) -> List[List[List[float]]]:
        """
        Trích xuất các tọa độ đa giác (Polygon points) từ ảnh mask nhị phân.
        """
        binary = (mask_np > 127).astype(np.uint8)
        if np.sum(binary) == 0:
            return []

        polygons = []
        if HAS_CV2:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                # Giảm độ răng cưa polygon bằng approxPolyDP
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = approx.reshape(-1, 2)
                if len(pts) >= 3:
                    polygons.append([[round(float(x), 2), round(float(y), 2)] for x, y in pts])
        else:
            from scipy import ndimage
            labeled, n_features = ndimage.label(binary)
            for i in range(1, n_features + 1):
                comp = (labeled == i)
                area = np.sum(comp)
                if area < min_area:
                    continue
                rows = np.any(comp, axis=1)
                cols = np.any(comp, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                polygons.append([
                    [float(xmin), float(ymin)], [float(xmax), float(ymin)],
                    [float(xmax), float(ymax)], [float(xmin), float(ymax)]
                ])

        return polygons

    @classmethod
    def create_labelme_json(cls, image_filename: str, img_w: int, img_h: int,
                            polygons: List[List[List[float]]], label: str = "defect") -> dict:
        """
        Tạo cấu trúc dict chuẩn định dạng LabelMe JSON.
        """
        shapes = []
        for poly in polygons:
            shapes.append({
                "label": label,
                "points": poly,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            })

        return {
            "version": "5.2.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_filename,
            "imageData": None,
            "imageHeight": int(img_h),
            "imageWidth": int(img_w)
        }

    @staticmethod
    def json_to_mask(json_data: dict, img_w: int, img_h: int) -> np.ndarray:
        """
        Tái tạo mặt nạ nhị phân numpy array (0 và 255) từ dữ liệu LabelMe JSON.
        """
        mask_img = Image.new('L', (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask_img)

        for shape in json_data.get("shapes", []):
            pts = shape.get("points", [])
            if len(pts) >= 3:
                flat_pts = [(p[0], p[1]) for p in pts]
                draw.polygon(flat_pts, outline=255, fill=255)

        return np.array(mask_img, dtype=np.uint8)

    @classmethod
    def is_defect_json(cls, json_path: Path) -> bool:
        """
        Kiểm tra metadata trong file JSON: trả về True nếu có ít nhất 1 shape mang nhãn 'defect'.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for s in shapes:
                lbl = str(s.get("label", "")).lower()
                pts = s.get("points", [])
                if "defect" in lbl and len(pts) >= 3:
                    return True
            return False
        except Exception:
            return False


# ==============================================================================
# 3. BOUNDARY EXTRACTOR (HỖ TRỢ BOUNDARY-AWARE LOSS)
# ==============================================================================

class BoundaryExtractor:
    @staticmethod
    def extract_boundary(mask_binary: np.ndarray, thickness: int = 1) -> np.ndarray:
        """
        Trích xuất bản đồ ranh giới/đường biên: Boundary = Dilation(M) - Erosion(M).
        """
        mask_u8 = (mask_binary > 0).astype(np.uint8) * 255
        if np.sum(mask_u8) == 0:
            return np.zeros_like(mask_u8)

        if HAS_CV2:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness * 2 + 1, thickness * 2 + 1))
            dilated = cv2.dilate(mask_u8, kernel)
            eroded = cv2.erode(mask_u8, kernel)
            return cv2.subtract(dilated, eroded)
        else:
            from scipy import ndimage
            struct = ndimage.generate_binary_structure(2, 1)
            dilated = ndimage.binary_dilation(mask_u8 > 0, structure=struct, iterations=thickness)
            eroded = ndimage.binary_erosion(mask_u8 > 0, structure=struct, iterations=thickness)
            return (dilated ^ eroded).astype(np.uint8) * 255

    @staticmethod
    def calculate_perimeter(mask_binary: np.ndarray) -> float:
        if np.sum(mask_binary) == 0:
            return 0.0
        if HAS_CV2:
            contours, _ = cv2.findContours((mask_binary > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return float(sum(cv2.arcLength(c, True) for c in contours))
        else:
            boundary = BoundaryExtractor.extract_boundary(mask_binary)
            return float(np.count_nonzero(boundary > 0))


# ==============================================================================
# 4. DEFECT SIZE ANALYZER (ĐỊNH NGHĨA KHUYẾT TẬT NHỎ CHO SEGMENTATION)
# ==============================================================================

class DefectSizeAnalyzer:
    SMALL_THRESH_PX = 1024       # < 1,024 px² (hoặc < 0.5% diện tích)
    MEDIUM_THRESH_PX = 4096      # 1,024 ~ 4,096 px²

    @classmethod
    def classify_defect(cls, area_px: int) -> str:
        if area_px == 0:
            return "None"
        if area_px < cls.SMALL_THRESH_PX:
            return "Small"
        elif area_px <= cls.MEDIUM_THRESH_PX:
            return "Medium"
        else:
            return "Large"

    @classmethod
    def analyze_mask(cls, mask_np: np.ndarray, img_w: int, img_h: int) -> DefectSegmentationInfo:
        binary = (mask_np > 127).astype(np.uint8)
        area_px = int(np.count_nonzero(binary))
        img_area = float(img_w * img_h)
        rel_pct = round((area_px / img_area) * 100.0, 4) if img_area > 0 else 0.0

        if area_px == 0:
            return DefectSegmentationInfo(
                total_area_pixels=0, relative_area_pct=0.0,
                category="None", num_components=0, boundary_perimeter_px=0.0, shapes=[]
            )

        # Trích xuất polygons cho LabelMe JSON
        polys = LabelMeJsonConverter.mask_to_polygons(mask_np)
        shapes = []
        for poly in polys:
            p_area = 0
            if HAS_CV2:
                p_area = int(cv2.contourArea(np.array(poly, dtype=np.float32)))
            shapes.append(DefectPolygonShape(
                label="defect", points=poly, area_pixels=p_area,
                category=cls.classify_defect(p_area)
            ))

        num_components = max(1, len(polys))
        perimeter = BoundaryExtractor.calculate_perimeter(binary)
        cat = cls.classify_defect(area_px)

        return DefectSegmentationInfo(
            total_area_pixels=area_px,
            relative_area_pct=rel_pct,
            category=cat,
            num_components=num_components,
            boundary_perimeter_px=round(perimeter, 2),
            shapes=shapes
        )


# ==============================================================================
# 5. DATASET AUDITOR (KIỂM TOÁN TÍNH TOÀN VẸN & LỌC CHỈ MẪU CÓ DEFECT)
# ==============================================================================

class DatasetAuditor:
    @staticmethod
    def audit_sample(img_path: Path, mask_path: Optional[Path], dataset_name: str,
                     sample_id: str, defect_only: bool = True) -> Optional[SampleRecord]:
        """
        Kiểm toán và trả về SampleRecord.
        Nếu defect_only == True, loại bỏ hoàn toàn các ảnh sạch (không có khuyết tật).
        """
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                channels = len(img.mode) if img.mode in ['RGB', 'RGBA'] else 1
                aspect_ratio = round(w / float(h), 4) if h > 0 else 1.0
        except Exception:
            return None

        if mask_path is None or not mask_path.exists():
            if defect_only:
                return None
            empty_info = DefectSegmentationInfo(0, 0.0, "None", 0, 0.0, [])
            return SampleRecord(
                sample_id=sample_id, dataset_source=dataset_name,
                image_path=img_path, mask_path=None,
                width=w, height=h, channels=channels, aspect_ratio=aspect_ratio,
                has_defect=False, defect_info=empty_info,
                integrity_status="MISSING_MASK"
            )

        try:
            with Image.open(mask_path) as m_img:
                mw, mh = m_img.size
                status = "OK" if (mw == w and mh == h) else f"DIM_MISMATCH (Img:{w}x{h} vs Mask:{mw}x{mh})"
                mask_np = np.array(m_img.convert('L'))
        except Exception:
            return None

        has_defect = (np.count_nonzero(mask_np > 127) > 0)
        if defect_only and not has_defect:
            return None  # LỌC BỎ ẢNH SẠCH

        defect_info = DefectSizeAnalyzer.analyze_mask(mask_np, w, h)
        if defect_only and defect_info.total_area_pixels == 0:
            return None

        return SampleRecord(
            sample_id=sample_id, dataset_source=dataset_name,
            image_path=img_path, mask_path=mask_path,
            width=w, height=h, channels=channels, aspect_ratio=aspect_ratio,
            has_defect=has_defect, defect_info=defect_info,
            integrity_status=status
        )

    @classmethod
    def scan_kolektor(cls, kolektor_dir: Path, defect_only: bool = True) -> List[SampleRecord]:
        """
        Quét KolektorSDD2.
        Đồng nhất tiền tố: `ksdd2_defect_XXXX`
        """
        records = []
        if not kolektor_dir.exists():
            return records

        count = 1
        for split_folder in ["train", "test"]:
            dir_path = kolektor_dir / split_folder
            if not dir_path.exists():
                continue
            for img_file in sorted(dir_path.glob("*.png")):
                if img_file.stem.endswith("_GT"):
                    continue
                mask_file = dir_path / f"{img_file.stem}_GT.png"
                rec = cls.audit_sample(
                    img_path=img_file,
                    mask_path=mask_file if mask_file.exists() else None,
                    dataset_name="KolektorSDD2",
                    sample_id=f"ksdd2_defect_{count:04d}",
                    defect_only=defect_only
                )
                if rec is not None:
                    records.append(rec)
                    count += 1
        return records

    @classmethod
    def locate_magnetic_tile_dir(cls, base_dir: Path) -> Optional[Path]:
        candidates = [
            base_dir / "Magnetic-tile-defect-datasets.-master",
            base_dir / "Magnetic-tile-defect-datasets-master",
            base_dir / "Magnetic-tile-defect-datasets.",
            base_dir / "Magnetic_Tile",
            base_dir / "Magnetic-Tile-Defect"
        ]
        for p in candidates:
            if p.exists() and p.is_dir():
                return p
        for p in base_dir.iterdir():
            if p.is_dir() and "magnetic" in p.name.lower():
                return p
        return None

    @classmethod
    def scan_magnetic_tile(cls, mt_dir: Path, defect_only: bool = True) -> List[SampleRecord]:
        """
        Quét Magnetic Tile (Bỏ qua hoàn toàn thư mục MT_Free chứa ảnh sạch không lỗi).
        Đồng nhất tiền tố: `mt_<defect_type>_defect_XXXX`
        """
        records = []
        if not mt_dir.exists():
            return records

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_files = sorted(list(mt_dir.rglob("*")))
        count = 1

        for item in all_files:
            if not item.is_file() or item.suffix.lower() not in image_extensions:
                continue
            if item.name.lower() in ["dataset.jpg", "dataset.png", "sample.jpg", "readme.jpg"]:
                continue
            if "_gt" in item.name.lower() or "mask" in str(item.parent).lower():
                continue

            # Bỏ qua hoàn toàn thư mục MT_Free (ảnh sạch không lỗi)
            if any(k in item.parts for k in ["MT_Free", "Free", "free", "normal"]):
                continue

            # Xác định loại khuyết tật từ thư mục cha (ví dụ: MT_Blowhole -> blowhole)
            def_type = "defect"
            for part in item.parts:
                if part.startswith("MT_") and part != "MT_Free":
                    def_type = part.replace("MT_", "").lower()
                    break

            # Tìm mask tương ứng (.png cùng tên)
            parent = item.parent
            stem = item.stem
            mask_file = None
            for cand in [parent / f"{stem}.png", parent / f"{stem}_GT.png"]:
                if cand.exists() and cand != item:
                    mask_file = cand
                    break

            if item.suffix.lower() == ".png" and (item.parent / f"{item.stem}.jpg").exists():
                continue

            rec = cls.audit_sample(
                img_path=item,
                mask_path=mask_file,
                dataset_name="MagneticTile",
                sample_id=f"mt_{def_type}_defect_{count:04d}",
                defect_only=defect_only
            )
            if rec is not None:
                records.append(rec)
                count += 1
        return records


# ==============================================================================
# 6. IMAGE HASHING & DEDUPLICATION (256-BIT DHASH INTRA-DATASET)
# ==============================================================================

class ImageHasher:
    @staticmethod
    def compute_dhash(img: Image.Image, hash_size: int = 16) -> np.ndarray:
        resized = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(resized, dtype=np.int16)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return diff.flatten()

    @classmethod
    def deduplicate(cls, samples: List[SampleRecord], threshold: int = 2) -> Tuple[List[SampleRecord], Dict[str, int]]:
        unique_samples: List[SampleRecord] = []
        dup_stats: Dict[str, int] = {}

        for sample in samples:
            if sample.image_hash is None:
                try:
                    with Image.open(sample.image_path) as img:
                        sample.image_hash = cls.compute_dhash(img, hash_size=16)
                except Exception:
                    pass

        sources = sorted(list(set(s.dataset_source for s in samples)))
        for src in sources:
            src_samples = [s for s in samples if s.dataset_source == src]
            src_unique: List[SampleRecord] = []
            src_dups = 0

            hashes = []
            for s in src_samples:
                if s.image_hash is None:
                    src_unique.append(s)
                    continue
                if not hashes:
                    hashes.append(s.image_hash)
                    src_unique.append(s)
                    continue
                u_stack = np.array(hashes)
                dists = np.count_nonzero(u_stack != s.image_hash, axis=1)
                if np.any(dists <= threshold):
                    src_dups += 1
                else:
                    hashes.append(s.image_hash)
                    src_unique.append(s)

            unique_samples.extend(src_unique)
            dup_stats[src] = src_dups

        return unique_samples, dup_stats


# ==============================================================================
# 7. COUPLED SEGMENTATION AUGMENTATION (TĂNG CƯỜNG ĐỒNG BỘ ẢNH + MASK)
# ==============================================================================

class CoupledSegmentationAugmentor:
    """
    Tăng cường dữ liệu đồng bộ (Coupled Augmentation):
    Áp dụng phép biến đổi hình học (Lật ngang, lật dọc, xoay 90 độ)
    đồng thời trên cả Ảnh và Mask nhị phân để bảo toàn tính toàn vẹn 100% của nhãn.
    Chỉ áp dụng trên tập Train.
    """
    @staticmethod
    def augment(image: Image.Image, mask: Image.Image) -> List[Tuple[Image.Image, Image.Image, str]]:
        augmented_pairs = []

        # 1. Lật ngang (Horizontal Flip)
        img_hf = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_hf = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        augmented_pairs.append((img_hf, mask_hf, "hflip"))

        # 2. Lật dọc (Vertical Flip)
        img_vf = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask_vf = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        augmented_pairs.append((img_vf, mask_vf, "vflip"))

        # 3. Xoay 90 độ (Rotate 90)
        img_r90 = image.transpose(Image.Transpose.ROTATE_90)
        mask_r90 = mask.transpose(Image.Transpose.ROTATE_90)
        augmented_pairs.append((img_r90, mask_r90, "rot90"))

        return augmented_pairs


# ==============================================================================
# 8. STRATIFIED SPLITTER & DATASET EXPORTER
# ==============================================================================

class DatasetSplitter:
    @staticmethod
    def split(samples: List[SampleRecord], train_r: float = 0.7, val_r: float = 0.15,
              test_r: float = 0.15, seed: int = 42) -> None:
        random.seed(seed)
        groups: Dict[Tuple[str, str], List[SampleRecord]] = {}
        for s in samples:
            # Phân tầng theo (nguồn, nhóm quy mô khuyết tật)
            key = (s.dataset_source, s.defect_info.category)
            groups.setdefault(key, []).append(s)

        for key, grp in groups.items():
            random.shuffle(grp)
            n_total = len(grp)
            n_train = int(n_total * train_r)
            n_val = int(n_total * val_r)

            for i, s in enumerate(grp):
                if i < n_train:
                    s.split = "train"
                elif i < n_train + n_val:
                    s.split = "val"
                else:
                    s.split = "test"


class LabelMeDatasetExporter:
    """
    Xuất bộ dữ liệu ra cấu trúc phân vùng chuẩn Semantic Segmentation:
    data_segmentation/
    ├── train/
    │   ├── images/       (ksdd2_defect_0001.png, mt_blowhole_defect_0001.png)
    │   ├── masks/        (ksdd2_defect_0001.json - chuẩn LabelMe)
    │   └── masks_png/    (ksdd2_defect_0001.png - mask nhị phân tiện nạp PyTorch DataLoader)
    ├── val/
    │   ├── images/
    │   ├── masks/        (file .json)
    │   └── masks_png/    (file .png)
    └── test/
        ├── images/
        ├── masks/        (file .json)
        └── masks_png/    (file .png)
    """
    @staticmethod
    def export(samples: List[SampleRecord], output_dir: Path, apply_train_aug: bool = True) -> Dict[str, int]:
        if output_dir.exists():
            shutil.rmtree(output_dir)

        counts = {"train": 0, "val": 0, "test": 0}
        for sp in ["train", "val", "test"]:
            (output_dir / sp / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / sp / "masks").mkdir(parents=True, exist_ok=True)
            (output_dir / sp / "masks_png").mkdir(parents=True, exist_ok=True)

        for s in samples:
            sp = s.split
            img_filename = f"{s.sample_id}.png"
            json_filename = f"{s.sample_id}.json"

            img_target = output_dir / sp / "images" / img_filename
            json_target = output_dir / sp / "masks" / json_filename
            mask_png_target = output_dir / sp / "masks_png" / img_filename

            # 1. Lưu ảnh gốc
            with Image.open(s.image_path) as im:
                orig_img = im.convert('RGB')
                orig_img.save(img_target)

            # 2. Tạo và lưu file LabelMe JSON
            poly_points = [shape.points for shape in s.defect_info.shapes]
            json_data = LabelMeJsonConverter.create_labelme_json(
                image_filename=img_filename,
                img_w=s.width, img_h=s.height,
                polygons=poly_points, label="defect"
            )
            with open(json_target, "w", encoding="utf-8") as jf:
                json.dump(json_data, jf, indent=2, ensure_ascii=False)

            # 3. Lưu mask PNG song hành
            if s.mask_path and s.mask_path.exists():
                with Image.open(s.mask_path) as m_im:
                    orig_mask = m_im.convert('L')
                    orig_mask.save(mask_png_target)
            else:
                m_np = LabelMeJsonConverter.json_to_mask(json_data, s.width, s.height)
                orig_mask = Image.fromarray(m_np)
                orig_mask.save(mask_png_target)

            counts[sp] += 1

            # 4. Tăng cường dữ liệu đồng bộ (chỉ trên tập Train và mẫu có defect)
            if apply_train_aug and sp == "train" and s.has_defect:
                aug_pairs = CoupledSegmentationAugmentor.augment(orig_img, orig_mask)
                for aug_img, aug_mask, aug_type in aug_pairs:
                    aug_id = f"{s.sample_id}_{aug_type}"
                    aug_img_name = f"{aug_id}.png"
                    aug_json_name = f"{aug_id}.json"

                    aug_img.save(output_dir / "train" / "images" / aug_img_name)
                    aug_mask.save(output_dir / "train" / "masks_png" / aug_img_name)

                    aug_mask_np = np.array(aug_mask)
                    aug_polys = LabelMeJsonConverter.mask_to_polygons(aug_mask_np)
                    aug_json_data = LabelMeJsonConverter.create_labelme_json(
                        image_filename=aug_img_name,
                        img_w=aug_img.width, img_h=aug_img.height,
                        polygons=aug_polys, label="defect"
                    )
                    with open(output_dir / "train" / "masks" / aug_json_name, "w", encoding="utf-8") as jf:
                        json.dump(aug_json_data, jf, indent=2, ensure_ascii=False)

                    counts["train"] += 1

        return counts


# ==============================================================================
# 9. PYTORCH SEGMENTATION DATASET (ĐỌC TRỰC TIẾP TỪ JSON HOẶC PNG)
# ==============================================================================

if HAS_TORCH:
    class SurfaceDefectSegmentationDataset(Dataset):
        def __init__(self, root_dir: Path, split: str = "train", use_json_mask: bool = True,
                     target_size: Optional[Tuple[int, int]] = None):
            self.split_dir = root_dir / split
            self.images_dir = self.split_dir / "images"
            self.masks_dir = self.split_dir / "masks"
            self.png_dir = self.split_dir / "masks_png"
            self.use_json = use_json_mask
            self.target_size = target_size
            self.image_files = sorted(list(self.images_dir.glob("*.png")))

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_file = self.image_files[idx]
            stem = img_file.stem

            img = Image.open(img_file).convert('RGB')
            w, h = img.size

            if self.use_json:
                json_path = self.masks_dir / f"{stem}.json"
                with open(json_path, "r", encoding="utf-8") as f:
                    jdata = json.load(f)
                mask_np = LabelMeJsonConverter.json_to_mask(jdata, w, h)
                mask = Image.fromarray(mask_np)
            else:
                mask = Image.open(self.png_dir / f"{stem}.png").convert('L')

            if self.target_size:
                img = img.resize(self.target_size, Image.Resampling.BILINEAR)
                mask = mask.resize(self.target_size, Image.Resampling.NEAREST)

            img_np = np.array(img, dtype=np.float32) / 255.0
            mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

            boundary_np = BoundaryExtractor.extract_boundary(mask_np)
            boundary_np = (boundary_np > 0).astype(np.float32)

            return {
                "image": torch.from_numpy(img_np).permute(2, 0, 1),
                "mask": torch.from_numpy(mask_np).unsqueeze(0),
                "boundary": torch.from_numpy(boundary_np).unsqueeze(0),
                "sample_id": stem
            }


# Bí danh tương thích ngược toàn diện
DefectExtractor = DefectSizeAnalyzer
TrainAugmentor = CoupledSegmentationAugmentor
