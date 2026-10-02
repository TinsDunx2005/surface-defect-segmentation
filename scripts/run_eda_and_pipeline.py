#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: run_eda_and_pipeline.py
Đề tài: Nghiên cứu về Multi-Scale Features, Attention Gates và Boundary-Aware Loss
        cho bài toán phân vùng khuyết tật bề mặt nhỏ trên KolektorSDD2
Phân hệ: Người 1 — Dataset & EDA (SEMANTIC SEGMENTATION - DEFECT ONLY)

Quy trình tự động:
1. Dataset Audit toàn diện (quét KolektorSDD2 & Magnetic Tile).
2. Thống kê phân bố Positive vs Negative của dữ liệu thô.
3. Lọc 100% mẫu có khuyết tật (Defect Only) phục vụ bài toán Phân vùng ngữ nghĩa.
4. Lọc trùng lặp thông minh dHash 256-bit nội bộ từng dataset.
5. Phân loại khuyết tật nhỏ (Small < 1,024 px², Medium, Large) & trích xuất đường biên Boundary Map.
6. Phân tầng Stratified Split 70% Train - 15% Val - 15% Test.
7. Xuất bảng thống kê chi tiết dataset_statistics.csv.
8. Sinh trọn bộ 7 biểu đồ phân tích EDA chuẩn khoa học lưu vào eda_figures/.
9. Xuất thư mục dữ liệu phân vùng data_segmentation/ chuẩn Mask LabelMe JSON kèm mask PNG song hành.
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
    DatasetSplitter, CoupledSegmentationAugmentor, LabelMeDatasetExporter,
    LabelMeJsonConverter, SampleRecord, DefectExtractor, TrainAugmentor
)

PROJECT_DIR = Path(__file__).resolve().parent
KOLEKTOR_DIR = PROJECT_DIR / "KolektorSDD2"
OUTPUT_SEG_DIR = PROJECT_DIR / "data_segmentation"
FIGURES_DIR = PROJECT_DIR / "eda_figures"
CSV_PATH = PROJECT_DIR / "dataset_statistics.csv"


def generate_eda_figures(df_defects: pd.DataFrame, total_raw_pos: int, total_raw_neg: int,
                         defect_samples: list, figures_dir: Path):
    """
    Sinh 7 biểu đồ phân tích EDA chuẩn khoa học (300 DPI) cho bài toán Semantic Segmentation
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # -------------------------------------------------------------
    # 1. Phân bố Positive vs Negative ban đầu
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ['Clean / Negative (Không lỗi)', 'Defective / Positive (Có lỗi)']
    vals = [total_raw_neg, total_raw_pos]
    total_raw = total_raw_pos + total_raw_neg

    bars = axes[0].bar(labels, vals, color=['#3b82f6', '#ef4444'], width=0.45, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total_raw * 100.0) if total_raw > 0 else 0
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 20, f"{yval:,} ({pct:.1f}%)",
                     ha='center', va='bottom', fontweight='bold')
    axes[0].set_title("Phân bố mẫu Positive vs Negative trong tập dữ liệu gốc", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Số lượng ảnh")

    axes[1].pie(vals, labels=labels, autopct='%1.1f%%', startangle=140,
                colors=['#3b82f6', '#ef4444'], explode=(0, 0.08),
                wedgeprops={'edgecolor': 'black', 'linewidth': 1.2})
    axes[1].set_title("Tỷ lệ mẫu có lỗi (Segmentation chỉ huấn luyện tập Defective)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "01_positive_negative_distribution.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Phân bố nguồn dataset của tập khuyết tật
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    source_counts = df_defects['dataset_source'].value_counts()
    bars = plt.bar(source_counts.index, source_counts.values, color=['#0284c7', '#0d9488'], width=0.45, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df_defects) * 100.0) if len(df_defects) > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:,} ảnh ({pct:.1f}%)",
                 ha='center', va='bottom', fontweight='bold')
    plt.title("Phân bố mẫu khuyết tật theo nguồn dữ liệu (KolektorSDD2 vs Magnetic Tile)", fontsize=11, fontweight='bold')
    plt.ylabel("Số lượng ảnh khuyết tật")
    plt.tight_layout()
    plt.savefig(figures_dir / "02_dataset_sources_breakdown.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Kích thước ảnh và Tỷ lệ khung hình (Aspect Ratio)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scatter = axes[0].scatter(df_defects['width'], df_defects['height'], c=df_defects['aspect_ratio'],
                              cmap='viridis', alpha=0.6, edgecolors='none', s=45)
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Tỷ lệ Aspect Ratio (Width / Height)', rotation=270, labelpad=15)
    axes[0].set_title("Kích thước ảnh khuyết tật (Width vs Height)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Chiều rộng (Pixels)")
    axes[0].set_ylabel("Chiều cao (Pixels)")

    axes[1].hist(df_defects['aspect_ratio'], bins=25, color='#8b5cf6', edgecolor='black', alpha=0.8)
    axes[1].axvline(df_defects['aspect_ratio'].median(), color='red', linestyle='--', linewidth=1.5,
                    label=f"Trung vị: {df_defects['aspect_ratio'].median():.2f}")
    axes[1].set_title("Phân phối tỷ lệ khung hình (Aspect Ratio)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Tỷ lệ W / H")
    axes[1].set_ylabel("Số lượng mẫu")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "03_image_dimensions_aspect_ratio.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Phân loại quy mô khuyết tật (Small / Medium / Large)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    cat_counts = df_defects['defect_size_category'].value_counts()
    order = ['Small', 'Medium', 'Large']
    cat_counts = cat_counts.reindex(order).dropna()

    colors = {'Small': '#10b981', 'Medium': '#f59e0b', 'Large': '#ef4444'}
    bar_colors = [colors.get(c, '#6b7280') for c in cat_counts.index]
    bars = plt.bar(cat_counts.index, cat_counts.values, color=bar_colors, width=0.45, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df_defects) * 100.0) if len(df_defects) > 0 else 0
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:,} ({pct:.1f}%)",
                 ha='center', va='bottom', fontweight='bold')
    plt.title("Phân loại quy mô khuyết tật bề mặt (Small: <1024px², Medium: 1024-4096px², Large: >4096px²)",
              fontsize=11, fontweight='bold')
    plt.ylabel("Số lượng ảnh khuyết tật")
    plt.tight_layout()
    plt.savefig(figures_dir / "04_defect_size_categories_segmentation.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Phân phối diện tích khuyết tật (Log scale & Relative %)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df_defects['defect_area_pixels'], bins=35, color='#0284c7', edgecolor='black', alpha=0.8)
    axes[0].set_yscale('log')
    axes[0].set_title("Phân phối diện tích khuyết tật (Pixels² - Log scale)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Diện tích khuyết tật (Pixels²)")
    axes[0].set_ylabel("Số lượng mẫu (Thang Log)")

    axes[1].boxplot(df_defects['relative_defect_area_pct'], vert=False, patch_artist=True,
                    boxprops=dict(facecolor='#a7f3d0', color='black'),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_title("Tỷ lệ % diện tích khuyết tật trên toàn bộ ảnh", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Tỷ lệ % diện tích (%)")
    plt.tight_layout()
    plt.savefig(figures_dir / "05_defect_area_distribution.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Trực quan hóa Mẫu phân vùng (Image, Mask, Overlay, Boundary Map)
    # -------------------------------------------------------------
    pos_samples = [s for s in defect_samples if s.mask_path and s.mask_path.exists()]
    n_display = min(4, len(pos_samples))
    if n_display > 0:
        rng = random.Random(42)
        selected = rng.sample(pos_samples, n_display)

        fig, axes = plt.subplots(n_display, 4, figsize=(16, 3.8 * n_display))
        if n_display == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, s in enumerate(selected):
            img = Image.open(s.image_path).convert('RGB')
            axes[row_idx, 0].imshow(img)
            axes[row_idx, 0].set_title(f"Ảnh: {s.sample_id}", fontsize=10, fontweight='bold')
            axes[row_idx, 0].axis('off')

            mask = Image.open(s.mask_path).convert('L')
            mask_np = np.array(mask)
            axes[row_idx, 1].imshow(mask_np, cmap='gray')
            axes[row_idx, 1].set_title(f"Mask ({s.defect_info.total_area_pixels}px - {s.defect_info.category})",
                                       fontsize=10, fontweight='bold')
            axes[row_idx, 1].axis('off')

            overlay = np.array(img).copy()
            binary_mask = mask_np > 127
            overlay[binary_mask] = [255, 40, 40]
            axes[row_idx, 2].imshow(overlay)
            axes[row_idx, 2].set_title("Mask Overlay (Đỏ)", fontsize=10, fontweight='bold')
            axes[row_idx, 2].axis('off')

            boundary = BoundaryExtractor.extract_boundary(binary_mask)
            axes[row_idx, 3].imshow(boundary, cmap='hot')
            axes[row_idx, 3].set_title(f"Boundary Map ({s.defect_info.boundary_perimeter_px}px)",
                                       fontsize=10, fontweight='bold')
            axes[row_idx, 3].axis('off')

        plt.tight_layout()
        plt.savefig(figures_dir / "06_segmentation_visualizations.png", dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # 7. Phân phối trên 3 tập Train / Val / Test
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    split_cat = df_defects.groupby(['split', 'defect_size_category']).size().unstack(fill_value=0)
    for c in ['Small', 'Medium', 'Large']:
        if c not in split_cat.columns:
            split_cat[c] = 0
    split_cat = split_cat[['Small', 'Medium', 'Large']]

    ax = split_cat.plot(kind='bar', stacked=True, color=['#10b981', '#f59e0b', '#ef4444'],
                        figsize=(9, 5), edgecolor='black')
    plt.title("Phân bố mẫu khuyết tật trên 3 tập Train (70%) - Val (15%) - Test (15%)", fontsize=11, fontweight='bold')
    plt.xlabel("Tập dữ liệu")
    plt.ylabel("Số lượng ảnh khuyết tật")
    plt.legend(title="Quy mô lỗi")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "07_train_val_test_split.png", dpi=300)
    plt.close()

    print(f"  [OK] Đã xuất 7 biểu đồ phân tích EDA chất lượng cao vào: {figures_dir}")


def main():
    print("=" * 80)
    print(" BÀI TOÁN PHÂN VÙNG KHUYẾT TẬT BỀ MẶT NHỎ (SEMANTIC SEGMENTATION)")
    print(" ĐỀ TÀI: Multi-Scale Features, Attention Gates & Boundary-Aware Loss")
    print(" PIPELINE: AUDIT -> DEFECT-ONLY FILTER -> LABELME JSON -> TRAIN/VAL/TEST")
    print("=" * 80)

    # -------------------------------------------------------------
    # BƯỚC 1: Quét toàn diện KolektorSDD2 (Cả Positive & Negative để Audit)
    # -------------------------------------------------------------
    print("\n[Bước 1/6] Quét kiểm toán tập dữ liệu gốc KolektorSDD2...")
    all_kolektor = DatasetAuditor.scan_kolektor(KOLEKTOR_DIR, defect_only=False)
    k_pos = sum(1 for s in all_kolektor if s.has_defect)
    k_neg = len(all_kolektor) - k_pos
    print(f"  -> Quét xong KolektorSDD2: Tổng {len(all_kolektor):,} ảnh (Có lỗi: {k_pos:,} | Sạch: {k_neg:,})")

    # -------------------------------------------------------------
    # BƯỚC 2: Quét toàn diện Magnetic Tile
    # -------------------------------------------------------------
    print("\n[Bước 2/6] Quét kiểm toán tập dữ liệu bổ trợ Magnetic Tile...")
    mt_dir = DatasetAuditor.locate_magnetic_tile_dir(PROJECT_DIR)
    all_mt = []
    if mt_dir:
        print(f"  -> Tìm thấy thư mục: {mt_dir.name}")
        all_mt = DatasetAuditor.scan_magnetic_tile(mt_dir, defect_only=False)
        m_pos = sum(1 for s in all_mt if s.has_defect)
        m_neg = len(all_mt) - m_pos
        print(f"  -> Quét xong Magnetic Tile: Tổng {len(all_mt):,} ảnh (Có lỗi: {m_pos:,} | Sạch: {m_neg:,})")
    else:
        m_pos, m_neg = 0, 0
        print("  -> [Cảnh báo] Không tìm thấy thư mục Magnetic Tile.")

    total_raw_pos = k_pos + m_pos
    total_raw_neg = k_neg + m_neg
    total_raw = len(all_kolektor) + len(all_mt)
    print(f"\n---> TỔNG CỘNG THU THẬP BAN ĐẦU: {total_raw:,} ảnh")
    print(f"     + Mẫu có lỗi (Positive) : {total_raw_pos:,} ảnh ({total_raw_pos/total_raw*100:.1f}%)")
    print(f"     + Mẫu sạch (Negative)   : {total_raw_neg:,} ảnh ({total_raw_neg/total_raw*100:.1f}%)")

    # -------------------------------------------------------------
    # BƯỚC 3: Lọc 100% Defect Only & Khử trùng lặp dHash 256-bit
    # -------------------------------------------------------------
    print("\n[Bước 3/6] Lọc 100% mẫu có khuyết tật (Defect Only) & Khử trùng lặp dHash...")
    raw_defect_samples = [s for s in (all_kolektor + all_mt) if s.has_defect]
    print(f"  -> Số mẫu có khuyết tật trước lọc trùng: {len(raw_defect_samples):,} ảnh")

    unique_defect_samples, dup_stats = ImageHasher.deduplicate(raw_defect_samples, threshold=2)
    print("  -> Thống kê lọc trùng lặp nội bộ (Hamming <= 2):")
    for src, cnt in dup_stats.items():
        print(f"     + Nguồn {src}: loại bỏ {cnt:,} ảnh trùng.")
    print(f"  -> Số ảnh khuyết tật sạch duy nhất giữ lại: {len(unique_defect_samples):,} ảnh.")

    # -------------------------------------------------------------
    # BƯỚC 4: Phân chia phân tầng 70% Train - 15% Val - 15% Test
    # -------------------------------------------------------------
    print("\n[Bước 4/6] Phân chia phân tầng Stratified Split (70% Train - 15% Val - 15% Test)...")
    DatasetSplitter.split(unique_defect_samples, train_r=0.7, val_r=0.15, test_r=0.15, seed=42)
    split_counts = pd.Series([s.split for s in unique_defect_samples]).value_counts()
    print(f"  -> Phân bổ: Train: {split_counts.get('train', 0):,} | Val: {split_counts.get('val', 0):,} | Test: {split_counts.get('test', 0):,}")

    # -------------------------------------------------------------
    # BƯỚC 5: Xuất file dataset_statistics.csv
    # -------------------------------------------------------------
    print("\n[Bước 5/6] Xuất bảng thống kê phân vùng dataset_statistics.csv...")
    records_data = []
    for s in unique_defect_samples:
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

    df_defects = pd.DataFrame(records_data)
    df_defects.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"  [OK] Đã lưu bảng thống kê phân vùng ({len(df_defects):,} dòng) tại: {CSV_PATH}")

    # -------------------------------------------------------------
    # BƯỚC 6: Sinh 7 biểu đồ phân tích EDA
    # -------------------------------------------------------------
    print("\n[Bước 6/6] Sinh 7 biểu đồ EDA lưu vào eda_figures/...")
    generate_eda_figures(df_defects, total_raw_pos, total_raw_neg, unique_defect_samples, FIGURES_DIR)

    # -------------------------------------------------------------
    # BƯỚC 7: Xuất cấu trúc thư mục Semantic Segmentation với LabelMe JSON
    # -------------------------------------------------------------
    print(f"\n---> Xuất bộ dữ liệu ra thư mục: {OUTPUT_SEG_DIR}")
    counts = LabelMeDatasetExporter.export(unique_defect_samples, OUTPUT_SEG_DIR, apply_train_aug=True)
    print(f"  [OK] Hoàn tất xuất Segmentation Dataset:")
    print(f"       + Train: {counts['train']} mẫu (đã kèm Coupled Augmentation: HFlip, VFlip, Rot90)")
    print(f"       + Val  : {counts['val']} mẫu (giữ nguyên gốc)")
    print(f"       + Test : {counts['test']} mẫu (giữ nguyên gốc)")
    print(f"       + Mask định dạng: LabelMe JSON (.json) trong thư mục masks/ kèm PNG trong masks_png/")

    print("\n" + "=" * 80)
    print("        HOÀN TẤT PIPELINE TIỀN XỬ LÝ CHO BÀI TOÁN SEGMENTATION!")
    print("=" * 80)
    print(f"1. Thư mục dataset phân vùng : {OUTPUT_SEG_DIR}")
    print(f"2. Bảng thống kê dữ liệu CSV : {CSV_PATH}")
    print(f"3. Thư mục biểu đồ EDA       : {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
