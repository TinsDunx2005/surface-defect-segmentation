#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: dataset.py
Đề tài: Nghiên cứu về Multi-Scale Features, Attention Gates và Boundary-Aware Loss
        cho bài toán phân vùng khuyết tật bề mặt nhỏ trên KolektorSDD2
Phân hệ: Người 1 — Dataset & EDA (SEMANTIC SEGMENTATION)

Hỗ trợ đầy đủ:
- Tương thích 100% cả bài toán Semantic Segmentation và các tên gọi cũ (DefectExtractor, TrainAugmentor).
- Ghép cặp Image - Mask, kiểm tra Mask Integrity, trích xuất Boundary Map.
- Lọc trùng dHash 256-bit nội bộ từng dataset.
- Phân loại Small/Medium/Large cho bài toán khuyết tật nhỏ.
- Stratified Split và Coupled Augmentation (Ảnh + Mask).
"""

import os
import shutil
import random
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

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
class DefectBoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    area_pixels: int
    relative_area: float
    category: str
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float
    class_id: int = 0


@dataclass
class DefectSegmentationInfo:
    total_area_pixels: int       # Tổng số pixel khuyết tật
    relative_area_pct: float     # Tỷ lệ % diện tích khuyết tật trên toàn bộ ảnh
    category: str                # 'Small', 'Medium', 'Large'
    num_components: int          # Số vùng khuyết tật rời rạc
    boundary_perimeter_px: float # Chu vi đường biên khuyết tật (hỗ trợ Boundary-Aware Loss)


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
    defects: List[DefectBoundingBox] = field(default_factory=list) # Hỗ trợ tương thích ngược
    image_hash: Optional[np.ndarray] = None
    split: str = "train"         # 'train', 'val', 'test'
    integrity_status: str = "OK"


# ==============================================================================
# 2. BOUNDARY EXTRACTOR (HỖ TRỢ BOUNDARY-AWARE LOSS)
# ==============================================================================

class BoundaryExtractor:
    """
    Trích xuất bản đồ ranh giới/đường biên (Boundary Map) từ mặt nạ phân vùng nhị phân.
    Dùng cho hàm mất mát hướng ranh giới (Boundary-Aware Loss).
    """
    @staticmethod
    def extract_boundary(mask_binary: np.ndarray, thickness: int = 1) -> np.ndarray:
        mask_u8 = (mask_binary > 0).astype(np.uint8) * 255
        if np.sum(mask_u8) == 0:
            return np.zeros_like(mask_u8)

        if HAS_CV2:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness * 2 + 1, thickness * 2 + 1))
            dilated = cv2.dilate(mask_u8, kernel)
            eroded = cv2.erode(mask_u8, kernel)
            boundary = cv2.subtract(dilated, eroded)
            return boundary
        else:
            from scipy import ndimage
            struct = ndimage.generate_binary_structure(2, 1)
            dilated = ndimage.binary_dilation(mask_u8 > 0, structure=struct, iterations=thickness)
            eroded = ndimage.binary_erosion(mask_u8 > 0, structure=struct, iterations=thickness)
            boundary = (dilated ^ eroded).astype(np.uint8) * 255
            return boundary

    @staticmethod
    def calculate_perimeter(mask_binary: np.ndarray) -> float:
        if np.sum(mask_binary) == 0:
            return 0.0
        if HAS_CV2:
            contours, _ = cv2.findContours((mask_binary > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum(cv2.arcLength(c, True) for c in contours)
            return float(perimeter)
        else:
            boundary = BoundaryExtractor.extract_boundary(mask_binary)
            return float(np.count_nonzero(boundary > 0))


# ==============================================================================
# 3. ĐỊNH NGHĨA QUY MÔ KHUYẾT TẬT NHỎ (SMALL / MEDIUM / LARGE)
# ==============================================================================

class DefectSizeAnalyzer:
    """
    Định nghĩa quy mô khuyết tật cho bài toán phân vùng khuyết tật bề mặt nhỏ (KolektorSDD2):
    - Small (Khuyết tật nhỏ/vi mô) : Diện tích < 1,024 px² (hoặc < 0.5% diện tích ảnh).
    - Medium (Khuyết tật vừa)      : 1,024 px² <= Diện tích <= 4,096 px² (0.5% ~ 2.0%).
    - Large (Khuyết tật lớn)       : Diện tích > 4,096 px² (> 2.0% diện tích ảnh).
    """
    SMALL_THRESH_PX = 1024
    MEDIUM_THRESH_PX = 4096

    @classmethod
    def classify_defect(cls, area_px: int, img_area: int) -> str:
        if area_px == 0:
            return "None"
        if area_px < cls.SMALL_THRESH_PX:
            return "Small"
        elif area_px <= cls.MEDIUM_THRESH_PX:
            return "Medium"
        else:
            return "Large"

    @classmethod
    def classify_defect_size(cls, area_pixels: int) -> str:
        return cls.classify_defect(area_pixels, 100000)

    @classmethod
    def analyze_mask(cls, mask_np: np.ndarray, img_w: int, img_h: int) -> DefectSegmentationInfo:
        binary = (mask_np > 127).astype(np.uint8)
        area_px = int(np.count_nonzero(binary))
        img_area = float(img_w * img_h)
        rel_pct = round((area_px / img_area) * 100.0, 4) if img_area > 0 else 0.0

        if area_px == 0:
            return DefectSegmentationInfo(
                total_area_pixels=0, relative_area_pct=0.0,
                category="None", num_components=0, boundary_perimeter_px=0.0
            )

        num_components = 0
        if HAS_CV2:
            n_labels, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            num_components = max(0, n_labels - 1)
        else:
            from scipy import ndimage
            _, num_components = ndimage.label(binary)

        perimeter = BoundaryExtractor.calculate_perimeter(binary)
        cat = cls.classify_defect(area_px, int(img_area))

        return DefectSegmentationInfo(
            total_area_pixels=area_px,
            relative_area_pct=rel_pct,
            category=cat,
            num_components=num_components,
            boundary_perimeter_px=round(perimeter, 2)
        )

    @classmethod
    def extract_bboxes_from_mask(cls, mask_np: np.ndarray, img_w: int, img_h: int) -> List[DefectBoundingBox]:
        """Hỗ trợ tương thích nếu có code gọi trích xuất Bounding Box"""
        binary = (mask_np > 127).astype(np.uint8)
        if np.sum(binary) == 0:
            return []
        bboxes = []
        img_area = float(img_w * img_h)
        if HAS_CV2:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 2 and bh < 2:
                    continue
                area_px = int(cv2.contourArea(cnt)) or (bw * bh)
                rel_area = (area_px / img_area) * 100.0
                cat = cls.classify_defect(area_px, int(img_area))
                bboxes.append(DefectBoundingBox(
                    x_min=x, y_min=y, x_max=x + bw, y_max=y + bh,
                    area_pixels=area_px, relative_area=rel_area,
                    category=cat,
                    x_center_norm=min(max((x + bw / 2.0) / img_w, 0.0), 1.0),
                    y_center_norm=min(max((y + bh / 2.0) / img_h, 0.0), 1.0),
                    width_norm=min(max(bw / float(img_w), 0.0), 1.0),
                    height_norm=min(max(bh / float(img_h), 0.0), 1.0),
                    class_id=0
                ))
        return bboxes


# Bí danh tương thích ngược (Backward Compatibility Aliases)
DefectExtractor = DefectSizeAnalyzer


# ==============================================================================
# 4. DATASET AUDITOR (KIỂM TOÁN TÍNH TOÀN VẸN MASK VÀ GHÉP CẶP ẢNH - MASK)
# ==============================================================================

class DatasetAuditor:
    @staticmethod
    def audit_sample(img_path: Path, mask_path: Optional[Path], dataset_name: str, sample_id: str) -> SampleRecord:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                channels = len(mode) if mode in ['RGB', 'RGBA'] else 1
                aspect_ratio = round(w / float(h), 4) if h > 0 else 1.0
        except Exception as e:
            return SampleRecord(
                sample_id=sample_id, dataset_source=dataset_name,
                image_path=img_path, mask_path=mask_path,
                width=0, height=0, channels=0, aspect_ratio=0.0,
                has_defect=False,
                defect_info=DefectSegmentationInfo(0, 0.0, "Corrupt", 0, 0.0),
                defects=[], integrity_status=f"CORRUPT_IMAGE: {e}"
            )

        if mask_path is None or not mask_path.exists():
            return SampleRecord(
                sample_id=sample_id, dataset_source=dataset_name,
                image_path=img_path, mask_path=None,
                width=w, height=h, channels=channels, aspect_ratio=aspect_ratio,
                has_defect=False,
                defect_info=DefectSegmentationInfo(0, 0.0, "None", 0, 0.0),
                defects=[], integrity_status="OK (Clean/Negative)"
            )

        try:
            with Image.open(mask_path) as m_img:
                mw, mh = m_img.size
                if mw != w or mh != h:
                    status = f"DIM_MISMATCH (Img:{w}x{h} vs Mask:{mw}x{mh})"
                else:
                    status = "OK"
                mask_np = np.array(m_img.convert('L'))
        except Exception as e:
            return SampleRecord(
                sample_id=sample_id, dataset_source=dataset_name,
                image_path=img_path, mask_path=mask_path,
                width=w, height=h, channels=channels, aspect_ratio=aspect_ratio,
                has_defect=False,
                defect_info=DefectSegmentationInfo(0, 0.0, "Corrupt", 0, 0.0),
                defects=[], integrity_status=f"CORRUPT_MASK: {e}"
            )

        defect_info = DefectSizeAnalyzer.analyze_mask(mask_np, w, h)
        bboxes = DefectSizeAnalyzer.extract_bboxes_from_mask(mask_np, w, h)
        has_defect = defect_info.total_area_pixels > 0

        return SampleRecord(
            sample_id=sample_id, dataset_source=dataset_name,
            image_path=img_path, mask_path=mask_path,
            width=w, height=h, channels=channels, aspect_ratio=aspect_ratio,
            has_defect=has_defect, defect_info=defect_info,
            defects=bboxes, integrity_status=status
        )

    @classmethod
    def scan_kolektor(cls, kolektor_dir: Path) -> List[SampleRecord]:
        records = []
        if not kolektor_dir.exists():
            return records

        for split_folder in ["train", "test"]:
            dir_path = kolektor_dir / split_folder
            if not dir_path.exists():
                continue
            for img_file in sorted(dir_path.glob("*.png")):
                if img_file.stem.endswith("_GT"):
                    continue
                mask_file = dir_path / f"{img_file.stem}_GT.png"
                record = cls.audit_sample(
                    img_path=img_file,
                    mask_path=mask_file if mask_file.exists() else None,
                    dataset_name="KolektorSDD2",
                    sample_id=f"ksdd2_{img_file.stem}"
                )
                records.append(record)
        return records

    @classmethod
    def locate_magnetic_tile_dir(cls, base_dir: Path) -> Optional[Path]:
        candidates = [
            base_dir / "Magnetic-tile-defect-datasets.-master",
            base_dir / "Magnetic-tile-defect-datasets-master",
            base_dir / "Magnetic-tile-defect-datasets.",
            base_dir / "Magnetic_Tile",
            base_dir / "Magnetic-Tile-Defect",
            base_dir / "MT_Defect"
        ]
        for p in candidates:
            if p.exists() and p.is_dir():
                return p

        for p in base_dir.iterdir():
            if p.is_dir() and "magnetic" in p.name.lower():
                return p

        for z in [
            base_dir / "Magnetic-tile-defect-datasets.-master.zip",
            base_dir / "Magnetic-tile-defect-datasets-master.zip"
        ]:
            if z.exists() and z.is_file():
                try:
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(base_dir)
                    for p in candidates:
                        if p.exists() and p.is_dir():
                            return p
                except Exception:
                    pass
        return None

    @classmethod
    def scan_magnetic_tile(cls, mt_dir: Path) -> List[SampleRecord]:
        records = []
        if not mt_dir.exists():
            return records

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_files = sorted(list(mt_dir.rglob("*")))
        for item in all_files:
            if not item.is_file() or item.suffix.lower() not in image_extensions:
                continue
            if item.name.lower() in ["dataset.jpg", "dataset.png", "sample.jpg", "readme.jpg"]:
                continue

            name_lower = item.name.lower()
            if "_gt" in name_lower or "groundtruth" in str(item.parent).lower() or "mask" in str(item.parent).lower():
                continue

            parent_parts = [p.lower() for p in item.parts]
            is_free = any(k in parent_parts for k in ["free", "mt_free", "normal", "good"])

            cat_name = "defect"
            for part in item.parts:
                if part.startswith("MT_"):
                    cat_name = part.replace("MT_", "").lower()
                    break
            if is_free:
                cat_name = "free"

            mask_file = None
            if not is_free:
                parent = item.parent
                stem = item.stem
                c1 = parent / f"{stem}.png"
                if c1.exists() and c1 != item:
                    mask_file = c1
                else:
                    c2 = parent / f"{stem}_GT.png"
                    if c2.exists():
                        mask_file = c2

            if item.suffix.lower() == ".png" and (item.parent / f"{item.stem}.jpg").exists():
                continue

            rec = cls.audit_sample(
                img_path=item,
                mask_path=mask_file,
                dataset_name="MagneticTile",
                sample_id=f"mt_{cat_name}_{item.stem}"
            )
            records.append(rec)
        return records


# ==============================================================================
# 5. IMAGE HASHING & DEDUPLICATION (256-BIT DHASH INTRA-DATASET)
# ==============================================================================

class ImageHasher:
    @staticmethod
    def compute_dhash(img: Image.Image, hash_size: int = 16) -> np.ndarray:
        resized = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(resized, dtype=np.int16)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return diff.flatten()

    @classmethod
    def deduplicate(cls, samples: List[SampleRecord],
                    pos_threshold: int = 2,
                    neg_threshold: int = 4) -> Tuple[List[SampleRecord], Dict[str, int]]:
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

            pos_samples = [s for s in src_samples if s.has_defect]
            neg_samples = [s for s in src_samples if not s.has_defect]

            pos_unique_hashes = []
            for s in pos_samples:
                if s.image_hash is None:
                    src_unique.append(s)
                    continue
                if not pos_unique_hashes:
                    pos_unique_hashes.append(s.image_hash)
                    src_unique.append(s)
                    continue
                u_stack = np.array(pos_unique_hashes)
                dists = np.count_nonzero(u_stack != s.image_hash, axis=1)
                if np.any(dists <= pos_threshold):
                    src_dups += 1
                else:
                    pos_unique_hashes.append(s.image_hash)
                    src_unique.append(s)

            neg_unique_hashes = []
            for s in neg_samples:
                if s.image_hash is None:
                    src_unique.append(s)
                    continue
                if not neg_unique_hashes:
                    neg_unique_hashes.append(s.image_hash)
                    src_unique.append(s)
                    continue
                u_stack = np.array(neg_unique_hashes)
                dists = np.count_nonzero(u_stack != s.image_hash, axis=1)
                if np.any(dists <= neg_threshold):
                    src_dups += 1
                else:
                    neg_unique_hashes.append(s.image_hash)
                    src_unique.append(s)

            unique_samples.extend(src_unique)
            dup_stats[src] = src_dups

        return unique_samples, dup_stats


# ==============================================================================
# 6. STRATIFIED SPLITTER & COUPLED SEGMENTATION AUGMENTATION
# ==============================================================================

class DatasetSplitter:
    @staticmethod
    def split(samples: List[SampleRecord], train_r: float = 0.7, val_r: float = 0.15,
              test_r: float = 0.15, seed: int = 42) -> None:
        random.seed(seed)
        groups: Dict[Tuple[str, bool], List[SampleRecord]] = {}
        for s in samples:
            key = (s.dataset_source, s.has_defect)
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


class CoupledSegmentationAugmentor:
    """
    Tăng cường dữ liệu đồng bộ (Coupled Augmentation) cho bài toán Segmentation:
    - Khi lật hoặc xoay ảnh thì MASK CŨNG ĐƯỢC LẬT HOẶC XOAY ĐỒNG BỘ theo cùng góc.
    - CHỈ ÁP DỤNG CHO TẬP TRAIN (giữ nguyên Val và Test 100% nguyên bản).
    """
    @staticmethod
    def augment(image: Image.Image, mask: Image.Image) -> List[Tuple[Image.Image, Image.Image]]:
        augmented_pairs = []

        # 1. Lật ngang đồng bộ
        img_hflip = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_hflip = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        augmented_pairs.append((img_hflip, mask_hflip))

        # 2. Lật dọc đồng bộ
        img_vflip = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask_vflip = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        augmented_pairs.append((img_vflip, mask_vflip))

        return augmented_pairs

    @staticmethod
    def augment_sample(img: Image.Image, bboxes: List[DefectBoundingBox]):
        """Hỗ trợ tương thích ngược nếu có code cũ gọi augment_sample"""
        return []


# Bí danh tương thích ngược
TrainAugmentor = CoupledSegmentationAugmentor


# ==============================================================================
# 7. PYTORCH SEGMENTATION DATASET (CHO U-NET, ATTENTION U-NET)
# ==============================================================================

if HAS_TORCH:
    class SurfaceDefectSegmentationDataset(Dataset):
        def __init__(self, samples: List[SampleRecord], split: str = "train", target_size: Optional[Tuple[int, int]] = None):
            self.samples = [s for s in samples if s.split == split]
            self.target_size = target_size

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            img = Image.open(sample.image_path).convert('RGB')
            if sample.mask_path and sample.mask_path.exists():
                mask = Image.open(sample.mask_path).convert('L')
            else:
                mask = Image.new('L', img.size, 0)

            if self.target_size:
                img = img.resize(self.target_size, Image.Resampling.BILINEAR)
                mask = mask.resize(self.target_size, Image.Resampling.NEAREST)

            img_np = np.array(img, dtype=np.float32) / 255.0
            mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

            boundary_np = BoundaryExtractor.extract_boundary(mask_np)
            boundary_np = (boundary_np > 0).astype(np.float32)

            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
            boundary_tensor = torch.from_numpy(boundary_np).unsqueeze(0)

            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "boundary": boundary_tensor,
                "sample_id": sample.sample_id,
                "has_defect": sample.has_defect
            }
