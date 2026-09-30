#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: run_eda_and_pipeline.py
Đề tài: Nghiên cứu về Multi-Scale Features, Attention Gates và Boundary-Aware Loss
        cho bài toán phân vùng khuyết tật bề mặt nhỏ trên KolektorSDD2
Phân hệ: Người 1 — Dataset & EDA (SEMANTIC SEGMENTATION)

Thực thi tự động:
1. Dataset Audit & Mask Integrity Check (Khớp kích thước ảnh - mask, dải pixel nhị phân).
2. Boundary Map Extraction (Trích xuất đường biên phục vụ Boundary-Aware Loss).
3. Image Hashing Deduplication (dHash 256-bit lọc trùng độc lập nội bộ từng tập dữ liệu).
4. Defect Size Analysis & Small/Medium/Large Categorization.
5. Stratified Split: 70% Train - 15% Val - 15% Test.
6. Xuất bảng thống kê dataset_statistics.csv (diện tích pixel, % diện tích, chu vi đường biên, số vùng lỗi).
7. Sinh 7 biểu đồ phân tích EDA lưu vào eda_figures/ (bao gồm hiển thị trực quan Mask và Boundary Map).
8. Xuất cấu trúc thư mục phân vùng ngữ nghĩa data_segmentation/ (train/images, train/masks, v.v.)
   kèm Coupled Augmentation (lật ảnh + mask đồng bộ) cho tập Train.
"""

import sys
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import module dataset phân vùng
from dataset import (
    DatasetAuditor, ImageHasher, DefectSizeAnalyzer, BoundaryExtractor,
    DatasetSplitter, CoupledSegmentationAugmentor, SampleRecord
)

PROJECT_DIR = Path(__file__).resolve().parent
KOLEKTOR_DIR = PROJECT_DIR / "KolektorSDD2"
OUTPUT_SEG_DIR = PROJECT_DIR / "data_segmentation"
FIGURES_DIR = PROJECT_DIR / "eda_figures"
CSV_PATH = PROJECT_DIR / "dataset_statistics.csv"


def generate_eda_figures(df: pd.DataFrame, samples: list, figures_dir: Path):
    """
    Sinh 7 biểu đồ phân tích EDA chuyên biệt cho bài toán Semantic Segmentation
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # -------------------------------------------------------------
    # 1. Phân phối Positive vs Negative
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pos_neg_counts = df['has_defect'].value_counts()
    labels = ['Negative (Clean/Sạch)', 'Positive (Defective/Lỗi)']
    vals = [pos_neg_counts.get(False, 0), pos_neg_counts.get(True, 0)]

    bars = axes[0].bar(labels, vals, color=['#3b82f6', '#ef4444'], width=0.45, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval:,} ({yval/len(df)*100:.1f}%)",
                     ha='center', va='bottom', fontweight='bold')
    axes[0].set_title("Phân bố mẫu Positive vs Negative", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Số lượng ảnh")

    axes[1].pie(vals, labels=labels, autopct='%1.1f%%', startangle=140,
                colors=['#3b82f6', '#ef4444'], explode=(0, 0.08),
                wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
    axes[1].set_title("Tỷ lệ mẫu bề mặt có khuyết tật", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "01_positive_negative_distribution.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Phân bố nguồn dataset
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    source_counts = df['dataset_source'].value_counts()
    bars = plt.bar(source_counts.index, source_counts.values, color='#0284c7', width=0.45, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval:,} ảnh ({yval/len(df)*100:.1f}%)",
                 ha='center', va='bottom', fontweight='bold')
    plt.title("Phân bố dữ liệu theo nguồn (Dataset Sources)", fontsize=12, fontweight='bold')
    plt.ylabel("Số lượng ảnh")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_dataset_sources_breakdown.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Kích thước ảnh và Tỷ lệ khung hình (Aspect Ratio)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scatter = axes[0].scatter(df['width'], df['height'], c=df['aspect_ratio'],
                              cmap='viridis', alpha=0.6, edgecolors='none', s=40)
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Aspect Ratio (W/H)')
    axes[0].set_title("Phân bố kích thước ảnh (Width vs Height)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Chiều rộng (Pixels)")
    axes[0].set_ylabel("Chiều cao (Pixels)")

    axes[1].hist(df['aspect_ratio'], bins=25, color='#8b5cf6', edgecolor='black', alpha=0.8)
    axes[1].set_title("Histogram tỷ lệ khung hình (Aspect Ratio)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Tỷ lệ W / H")
    axes[1].set_ylabel("Tần suất")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_image_dimensions_aspect_ratio.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Phân loại kích thước khuyết tật (Small / Medium / Large)
    # -------------------------------------------------------------
    pos_df = df[df['has_defect']].copy()
    if not pos_df.empty:
        plt.figure(figsize=(9, 5))
        cat_counts = pos_df['defect_size_category'].value_counts()
        order = ['Small', 'Medium', 'Large']
        cat_counts = cat_counts.reindex(order).dropna()
        bar_colors = ['#10b981', '#f59e0b', '#dc2626']
        bars = plt.bar(cat_counts.index, cat_counts.values, color=bar_colors[:len(cat_counts)], width=0.45, edgecolor='black')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10,
                     f"{yval:,} ({yval/len(pos_df)*100:.1f}%)",
                     ha='center', va='bottom', fontweight='bold')
        plt.title("Phân loại quy mô khuyết tật cho bài toán Small Defect Segmentation\n(Small: <1024px², Medium: 1024-4096px², Large: >4096px²)",
                  fontsize=12, fontweight='bold')
        plt.ylabel("Số lượng mẫu khuyết tật")
        plt.tight_layout()
        plt.savefig(figures_dir / "04_defect_size_categories_segmentation.png", dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # 5. Phân phối diện tích khuyết tật (Pixel Area & % Diện tích ảnh)
    # -------------------------------------------------------------
    if not pos_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Histogram diện tích (log-scale)
        axes[0].hist(pos_df['defect_area_pixels'], bins=35, color='#0ea5e9', edgecolor='black', alpha=0.8)
        axes[0].set_yscale('log')
        axes[0].set_title("Phân phối diện tích khuyết tật (Pixels² - Log scale)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Diện tích (Pixels²)")
        axes[0].set_ylabel("Số lượng mẫu (Log scale)")

        # Boxplot tỷ lệ % diện tích khuyết tật trên toàn bộ ảnh
        axes[1].boxplot(pos_df['relative_defect_area_pct'], vert=False, patch_artist=True,
                        boxprops=dict(facecolor='#a7f3d0', color='black'),
                        medianprops=dict(color='red', linewidth=2))
        axes[1].set_title("Tỷ lệ % diện tích khuyết tật trên toàn bộ ảnh", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Tỷ lệ % diện tích (%)")
        plt.tight_layout()
        plt.savefig(figures_dir / "05_defect_area_distribution.png", dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # 6. Trực quan hóa Mẫu phân vùng (Image + Mask + Overlay + Boundary Map)
    # -------------------------------------------------------------
    pos_samples = [s for s in samples if s.has_defect and s.mask_path and s.mask_path.exists()]
    if pos_samples:
        random.seed(42)
        selected = random.sample(pos_samples, min(4, len(pos_samples)))
        fig, axes = plt.subplots(len(selected), 4, figsize=(16, 3.8 * len(selected)))
        if len(selected) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, s in enumerate(selected):
            # Cột 1: Ảnh gốc
            img = Image.open(s.image_path).convert('RGB')
            axes[row_idx, 0].imshow(img)
            axes[row_idx, 0].set_title(f"Ảnh gốc: {s.sample_id}", fontsize=10)
            axes[row_idx, 0].axis('off')

            # Cột 2: Ground Truth Mask
            mask = Image.open(s.mask_path).convert('L')
            mask_np = np.array(mask)
            axes[row_idx, 1].imshow(mask_np, cmap='gray')
            axes[row_idx, 1].set_title(f"Mask ({s.defect_info.total_area_pixels}px - {s.defect_info.category})", fontsize=10)
            axes[row_idx, 1].axis('off')

            # Cột 3: Mask Overlay trên ảnh gốc
            overlay = np.array(img).copy()
            binary_mask = mask_np > 127
            overlay[binary_mask] = [255, 50, 50] # Tô màu đỏ vùng khuyết tật
            axes[row_idx, 2].imshow(overlay)
            axes[row_idx, 2].set_title("Mask Overlay (Vùng tổn thương)", fontsize=10)
            axes[row_idx, 2].axis('off')

            # Cột 4: Boundary Edge Map (phục vụ Boundary-Aware Loss)
            boundary = BoundaryExtractor.extract_boundary(binary_mask)
            axes[row_idx, 3].imshow(boundary, cmap='hot')
            axes[row_idx, 3].set_title(f"Boundary Map (Chu vi: {s.defect_info.boundary_perimeter_px}px)", fontsize=10)
            axes[row_idx, 3].axis('off')

        plt.tight_layout()
        plt.savefig(figures_dir / "06_segmentation_visualizations.png", dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # 7. Phân phối Train / Val / Test Split
    # -------------------------------------------------------------
    split_df = df.groupby(['split', 'has_defect']).size().unstack(fill_value=0)
    split_df.columns = ['Clean (Neg)', 'Defect (Pos)']
    split_df.plot(kind='bar', stacked=True, color=['#3b82f6', '#ef4444'], figsize=(8, 5), edgecolor='black')
    plt.title("Phân bổ mẫu trên 3 tập Train (70%) - Val (15%) - Test (15%)", fontsize=12, fontweight='bold')
    plt.xlabel("Tập dữ liệu")
    plt.ylabel("Số lượng ảnh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "07_train_val_test_split.png", dpi=300)
    plt.close()

    print(f"[OK] Đã xuất 7 biểu đồ phân tích Semantic Segmentation thành công vào: {figures_dir}")


def export_segmentation_dataset(samples: list, output_dir: Path, apply_train_aug: bool = True):
    """
    Xuất tập dữ liệu ra cấu trúc chuẩn cho bài toán Semantic Segmentation:
    data_segmentation/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
    """
    if output_dir.exists():
        print(f"\n---> Đang làm sạch thư mục kết quả phân vùng cũ: {output_dir.name}...")
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        (output_dir / sp / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / sp / "masks").mkdir(parents=True, exist_ok=True)

    print(f"\n---> Đang xuất dữ liệu phân vùng ngữ nghĩa (Semantic Segmentation) ra {output_dir}...")
    counts = {"train": 0, "val": 0, "test": 0}

    for s in samples:
        if not s.image_path.exists():
            continue

        sp = s.split
        img_dest = output_dir / sp / "images" / f"{s.sample_id}.png"
        mask_dest = output_dir / sp / "masks" / f"{s.sample_id}.png"

        try:
            # 1. Lưu ảnh gốc
            with Image.open(s.image_path) as im:
                orig_img = im.convert('RGB')
                orig_img.save(img_dest)

            # 2. Lưu mask tương ứng
            if s.mask_path and s.mask_path.exists():
                with Image.open(s.mask_path) as m_im:
                    orig_mask = m_im.convert('L')
                    orig_mask.save(mask_dest)
            else:
                # Tạo mask rỗng toàn màu đen (0) cho ảnh sạch
                orig_mask = Image.new('L', orig_img.size, 0)
                orig_mask.save(mask_dest)

            counts[sp] += 1

            # 3. Tăng cường dữ liệu đồng bộ (CHỈ ÁP DỤNG CHO TẬP TRAIN VÀ MẪU CÓ LỖI)
            if apply_train_aug and sp == "train" and s.has_defect:
                aug_pairs = CoupledSegmentationAugmentor.augment(orig_img, orig_mask)
                for aug_idx, (aug_img, aug_mask) in enumerate(aug_pairs, 1):
                    aug_id = f"{s.sample_id}_aug{aug_idx}"
                    aug_img_path = output_dir / "train" / "images" / f"{aug_id}.png"
                    aug_mask_path = output_dir / "train" / "masks" / f"{aug_id}.png"
                    aug_img.save(aug_img_path)
                    aug_mask.save(aug_mask_path)
                    counts["train"] += 1

        except Exception as e:
            print(f"[Cảnh báo] Bỏ qua file {s.image_path.name} do lỗi: {e}")

    print(f"  [OK] Hoàn tất xuất Segmentation Dataset: Train: {counts['train']} cặp (đã gồm Augment), Val: {counts['val']} cặp, Test: {counts['test']} cặp.")


def main():
    print("=" * 75)
    print(" PHÂN VÙNG KHUYẾT TẬT BỀ MẶT NHỎ (KOLEKTORSDD2 + MAGNETIC TILE)")
    print(" BÀI TOÁN: SEMANTIC SEGMENTATION & BOUNDARY-AWARE ANALYSIS")
    print("=" * 75)

    # 1. Quét KolektorSDD2
    print("\n[Bước 1/6] Quét dữ liệu KolektorSDD2 (tập chính)...")
    kolektor_samples = DatasetAuditor.scan_kolektor(KOLEKTOR_DIR)
    print(f"  -> Quét được: {len(kolektor_samples):,} ảnh từ KolektorSDD2.")

    # 2. Quét Magnetic Tile
    print("\n[Bước 2/6] Quét dữ liệu Magnetic Tile (tập bổ trợ)...")
    mt_dir = DatasetAuditor.locate_magnetic_tile_dir(PROJECT_DIR)
    mt_samples = []
    if mt_dir:
        print(f"  -> Tìm thấy thư mục: {mt_dir.name}")
        mt_samples = DatasetAuditor.scan_magnetic_tile(mt_dir)
        print(f"  -> Quét được: {len(mt_samples):,} ảnh từ Magnetic Tile.")
    else:
        print(f"  -> [CẢNH BÁO] Không phát hiện thư mục Magnetic Tile, tiến hành với KolektorSDD2.")

    all_samples = kolektor_samples + mt_samples
    print(f"\n---> Tổng số ảnh thu thập ban đầu: {len(all_samples):,} ảnh")

    # 3. Lọc trùng lặp thông minh bằng dHash 256-bit (Intra-dataset)
    print("\n[Bước 3/6] Lọc trùng lặp bằng dHash 256-bit nội bộ từng dataset...")
    unique_samples, dup_stats = ImageHasher.deduplicate(all_samples, pos_threshold=2, neg_threshold=4)
    total_dups = sum(dup_stats.values())
    print("  -> Thống kê lọc trùng:")
    for src, cnt in dup_stats.items():
        print(f"     + Nguồn {src}: loại bỏ {cnt:,} ảnh trùng.")
    print(f"  -> Tổng số ảnh giữ lại: {len(unique_samples):,} ảnh.")

    # 4. Phân chia Stratified Split (70% Train - 15% Val - 15% Test)
    print("\n[Bước 4/6] Phân chia dữ liệu phân tầng 70% Train - 15% Val - 15% Test...")
    DatasetSplitter.split(unique_samples, train_r=0.7, val_r=0.15, test_r=0.15, seed=42)

    # 5. Xuất file dataset_statistics.csv
    print("\n[Bước 5/6] Xuất bảng thống kê phân vùng dataset_statistics.csv...")
    records_data = []
    for s in unique_samples:
        info = s.defect_info
        records_data.append({
            "sample_id": s.sample_id,
            "dataset_source": s.dataset_source,
            "filename": s.image_path.name,
            "width": s.width,
            "height": s.height,
            "channels": s.channels,
            "aspect_ratio": s.aspect_ratio,
            "has_defect": s.has_defect,
            "defect_size_category": info.category,
            "defect_area_pixels": info.total_area_pixels,
            "relative_defect_area_pct": info.relative_area_pct,
            "num_defect_components": info.num_components,
            "boundary_perimeter_px": info.boundary_perimeter_px,
            "split": s.split,
            "integrity_status": s.integrity_status
        })

    df = pd.DataFrame(records_data)
    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"  [OK] Đã lưu bảng thống kê phân vùng ({len(df):,} dòng) tại: {CSV_PATH}")

    # 6. Sinh 7 biểu đồ phân tích EDA chuyên sâu
    print("\n[Bước 6/6] Sinh 7 biểu đồ EDA lưu vào eda_figures/...")
    generate_eda_figures(df, unique_samples, FIGURES_DIR)

    # 7. Xuất cấu trúc thư mục Semantic Segmentation
    export_segmentation_dataset(unique_samples, OUTPUT_SEG_DIR, apply_train_aug=True)

    print("\n" + "=" * 75)
    print("      HOÀN TẤT TIỀN XỬ LÝ CHO BÀI TOÁN SEMANTIC SEGMENTATION!")
    print("=" * 75)
    print(f"1. Thư mục dataset phân vùng : {OUTPUT_SEG_DIR}")
    print(f"   (Bao gồm các cặp train/images & train/masks, val/images & val/masks, ...)")
    print(f"2. Bảng dữ liệu thống kê CSV : {CSV_PATH}")
    print(f"3. Thư mục biểu đồ EDA       : {FIGURES_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
