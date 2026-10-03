#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eda_and_pipeline.py — Người 1: Dataset & EDA

Chạy toàn bộ quy trình:
  1. Dataset audit (ghép ảnh–ann, kích thước ảnh, toàn vẹn mask) cho KolektorSDD2 + Magnetic Tile
  2. Thống kê positive / negative, diện tích khuyết tật, Small / Medium / Large
  3. Chia Train / Val / Test phân tầng
  4. Xuất dataset_statistics.csv  +  dataset_audit.json
  5. Sinh các biểu đồ EDA vào eda_figures/
  6. Xuất data_segmentation/{train,val,test}/{img, ann, masks_png} và kiểm tra lại

Ví dụ:
    python run_eda_and_pipeline.py                       # đường dẫn mặc định cạnh file này
    python run_eda_and_pipeline.py --ksdd2-dir D:/data/kolektorsdd2-DatasetNinja \
                                   --mt-dir D:/data/magnetic-tile-surface-defect-DatasetNinja
    python run_eda_and_pipeline.py --no-export           # chỉ audit + EDA, không xuất dataset
    python run_eda_and_pipeline.py --include-negatives --overwrite

Các hàm `plot_*` trả về matplotlib Figure để notebook 01_eda.ipynb dùng lại.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

if __name__ == "__main__":          # chạy script: không cần cửa sổ; notebook giữ backend inline
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataset as D

PROJECT_DIR = Path(__file__).resolve().parent
SRC_COLORS = {"KolektorSDD2": "#2563eb", "MagneticTile": "#0d9488"}
SIZE_COLORS = {"Small": "#10b981", "Medium": "#f59e0b", "Large": "#ef4444"}
POS_COLOR, NEG_COLOR = "#ef4444", "#3b82f6"
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
                     "figure.dpi": 100, "savefig.bbox": "tight"})


def _src_color(s: str) -> str:
    return SRC_COLORS.get(s, "#6b7280")


def _bar_labels(ax, bars, fmt=lambda v: f"{int(v):,}", dy_frac=0.01):
    ymax = max((b.get_height() for b in bars), default=1) or 1
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * dy_frac, fmt(b.get_height()),
                ha="center", va="bottom", fontsize=9, fontweight="bold")


# ==============================================================================
# CÁC HÀM VẼ (mỗi hàm trả về Figure)
# ==============================================================================

def plot_pos_neg(df: pd.DataFrame) -> plt.Figure:
    """Phân bố positive (có lỗi) / negative (sạch) theo từng nguồn và tổng."""
    srcs = sorted(df["dataset_source"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(srcs))
    pos = np.array([int(df[(df.dataset_source == s) & df.is_positive].shape[0]) for s in srcs])
    neg = np.array([int(df[(df.dataset_source == s) & ~df.is_positive].shape[0]) for s in srcs])
    w = 0.38
    b1 = axes[0].bar(x - w / 2, neg, w, color=NEG_COLOR, edgecolor="black", label="Negative (sạch)")
    b2 = axes[0].bar(x + w / 2, pos, w, color=POS_COLOR, edgecolor="black", label="Positive (có lỗi)")
    tot = pos + neg
    for bars, vals in ((b1, neg), (b2, pos)):
        for b, v, t in zip(bars, vals, tot):
            axes[0].text(b.get_x() + b.get_width() / 2, v + max(tot) * 0.01,
                         f"{v:,}\n({100 * v / max(t, 1):.1f}%)", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x, srcs)
    axes[0].set_ylim(0, max(pos.max(), neg.max()) * 1.2)
    axes[0].set_ylabel("Số ảnh")
    axes[0].set_title("Positive vs Negative theo nguồn", fontweight="bold")
    axes[0].legend()
    axes[1].pie([neg.sum(), pos.sum()], labels=["Negative", "Positive"], colors=[NEG_COLOR, POS_COLOR],
                autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "black"})
    axes[1].set_title(f"Toàn bộ ({tot.sum():,} ảnh)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_image_size(df: pd.DataFrame) -> plt.Figure:
    """Kích thước ảnh (W×H) và tỷ lệ khung hình theo nguồn."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for s, g in df.groupby("dataset_source"):
        axes[0].scatter(g.width, g.height, s=14, alpha=0.5, color=_src_color(s), label=f"{s} (n={len(g):,})")
        axes[1].hist(g.aspect_ratio, bins=40, alpha=0.65, color=_src_color(s), label=s, edgecolor="black", linewidth=0.3)
    axes[0].set(xlabel="Chiều rộng (px)", ylabel="Chiều cao (px)")
    axes[0].set_title("Kích thước ảnh (W × H)", fontweight="bold")
    axes[0].legend(markerscale=2)
    axes[1].set(xlabel="W / H", ylabel="Số ảnh")
    axes[1].set_title("Tỷ lệ khung hình", fontweight="bold")
    axes[1].legend()
    pix = (df.width * df.height / 1e3)
    axes[2].boxplot([pix[df.dataset_source == s] for s in sorted(df.dataset_source.unique())],
                    tick_labels=sorted(df.dataset_source.unique()), patch_artist=True)
    axes[2].set_ylabel("Diện tích ảnh (nghìn px)")
    axes[2].set_title("Diện tích ảnh", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_mask_integrity(df: pd.DataFrame, reports: Dict[str, dict]) -> plt.Figure:
    """Toàn vẹn cặp ảnh–ann–mask: số mẫu OK / có lỗi và các lỗi cấp tệp (thiếu ann, ann mồ côi, ảnh hỏng...)."""
    srcs = sorted(df["dataset_source"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ok = [int((df[df.dataset_source == s].integrity_status == "OK").sum()) for s in srcs]
    bad = [int((df[df.dataset_source == s].integrity_status != "OK").sum()) for s in srcs]
    axes[0].bar(srcs, ok, color="#10b981", edgecolor="black", label="OK")
    axes[0].bar(srcs, bad, bottom=ok, color="#ef4444", edgecolor="black", label="Có vấn đề")
    for i, (o, b) in enumerate(zip(ok, bad)):
        axes[0].text(i, o + b, f"{o:,} OK / {b:,} lỗi", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, max(o + b for o, b in zip(ok, bad)) * 1.15)
    axes[0].set_title("Tính toàn vẹn mẫu (ảnh + ann + mask)", fontweight="bold")
    axes[0].legend()

    rows = []
    for s in srcs:
        r, g = reports.get(s, {}), df[df.dataset_source == s]
        rows.append([s, r.get("n_images", "?"), r.get("n_ann_files", "?"), len(r.get("missing_ann", [])),
                     len(r.get("orphan_ann", [])), len(r.get("corrupt_images", [])),
                     len(r.get("bad_ann_json", [])), int(g.integrity_status.str.contains("SIZE_MISMATCH").sum()),
                     int(g.integrity_status.str.contains("BITMAP_OUT_OF_BOUNDS").sum())])
    axes[1].axis("off")
    tb = axes[1].table(cellText=rows, loc="center", cellLoc="center",
                       colLabels=["Nguồn", "Ảnh", "Ann", "Thiếu\nann", "Ann\nmồ côi", "Ảnh\nhỏng",
                                  "Ann\nhỏng", "Lệch\nkích thước", "Bitmap\ntràn biên"])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 2.2)
    axes[1].set_title("Báo cáo kiểm toán cấp tệp", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_defect_area(df: pd.DataFrame) -> plt.Figure:
    """Phân phối diện tích khuyết tật (px², thang log) và % diện tích trên ảnh."""
    pos = df[df.is_positive]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    lo, hi = max(1, pos.defect_area_px.min()), pos.defect_area_px.max()
    bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), 35)
    for s, g in pos.groupby("dataset_source"):
        axes[0].hist(g.defect_area_px, bins=bins, alpha=0.65, color=_src_color(s),
                     edgecolor="black", linewidth=0.3, label=f"{s} (n={len(g):,}, median={g.defect_area_px.median():,.0f})")
    axes[0].set_xscale("log")
    axes[0].axvline(D.SMALL_MAX_PX, color="k", ls="--", lw=1)
    axes[0].axvline(D.MEDIUM_MAX_PX, color="k", ls=":", lw=1)
    axes[0].set(xlabel="Diện tích khuyết tật (px², log)", ylabel="Số ảnh")
    axes[0].set_title("Phân phối diện tích khuyết tật", fontweight="bold")
    axes[0].legend(fontsize=8)
    srcs = sorted(pos.dataset_source.unique())
    axes[1].boxplot([pos[pos.dataset_source == s].relative_defect_area_pct for s in srcs],
                    tick_labels=srcs, patch_artist=True, vert=True)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("% diện tích ảnh bị lỗi (log)")
    axes[1].set_title("Tỷ lệ diện tích lỗi / diện tích ảnh", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_size_categories(df: pd.DataFrame) -> plt.Figure:
    """Số ảnh positive theo Small / Medium / Large, từng nguồn và gộp."""
    pos = df[df.is_positive]
    srcs = sorted(pos.dataset_source.unique()) + ["Tất cả"]
    fig, axes = plt.subplots(1, len(srcs), figsize=(4.6 * len(srcs), 4.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, srcs):
        g = pos if s == "Tất cả" else pos[pos.dataset_source == s]
        vc = g.size_category.value_counts().reindex(D.SIZE_ORDER).fillna(0)
        bars = ax.bar(vc.index, vc.values, color=[SIZE_COLORS[c] for c in vc.index], edgecolor="black")
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vc.max(), 1) * 0.01,
                    f"{int(v):,}\n({100 * v / max(len(g), 1):.1f}%)", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(vc.max(), 1) * 1.25)
        ax.set_title(f"{s} (n={len(g):,})", fontweight="bold")
    fig.suptitle(f"Small < {D.SMALL_MAX_PX:,} px² ≤ Medium ≤ {D.MEDIUM_MAX_PX:,} px² < Large", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def plot_area_ecdf(df: pd.DataFrame) -> plt.Figure:
    """ECDF diện tích khuyết tật: cho thấy ngưỡng Small/Medium/Large cắt phân phối thật ở đâu."""
    pos = df[df.is_positive]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for s, g in pos.groupby("dataset_source"):
        v = np.sort(g.defect_area_px.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=_src_color(s), lw=2, label=s)
        for q in (0.25, 0.5, 0.75):
            ax.plot(np.quantile(v, q), q, "o", color=_src_color(s), ms=4)
    for thr, name in ((D.SMALL_MAX_PX, "Small|Medium"), (D.MEDIUM_MAX_PX, "Medium|Large")):
        ax.axvline(thr, color="k", ls="--", lw=1)
        ax.text(thr, 0.03, f" {name}\n {thr:,}px²", fontsize=8)
    ax.set_xscale("log")
    ax.set(xlabel="Diện tích khuyết tật (px², log)", ylabel="Tỷ lệ tích lũy")
    ax.set_title("ECDF diện tích khuyết tật (chấm = tứ phân vị)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_components(df: pd.DataFrame, comp_df: pd.DataFrame) -> plt.Figure:
    """Phân tích ở mức từng vùng lỗi liên thông: diện tích mỗi vùng và số vùng mỗi ảnh."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    if len(comp_df):
        bins = np.logspace(0, np.log10(max(comp_df.component_area_px.max(), 10)), 35)
        for s, g in comp_df.groupby("dataset_source"):
            axes[0].hist(g.component_area_px, bins=bins, alpha=0.65, color=_src_color(s),
                         edgecolor="black", linewidth=0.3, label=f"{s} (n={len(g):,})")
        axes[0].set_xscale("log")
        axes[0].axvline(D.SMALL_MAX_PX, color="k", ls="--", lw=1)
        axes[0].axvline(D.MEDIUM_MAX_PX, color="k", ls=":", lw=1)
        axes[0].legend(fontsize=8)
        cc = comp_df.groupby(["dataset_source", "size_category"]).size().unstack(fill_value=0)
        cc = cc.reindex(columns=D.SIZE_ORDER, fill_value=0)
        cc.plot(kind="bar", ax=axes[1], color=[SIZE_COLORS[c] for c in cc.columns], edgecolor="black", rot=0)
    axes[0].set(xlabel="Diện tích 1 vùng lỗi (px², log)", ylabel="Số vùng lỗi")
    axes[0].set_title("Diện tích từng vùng lỗi", fontweight="bold")
    axes[1].set_title("Số vùng lỗi theo Small/Medium/Large", fontweight="bold")
    axes[1].set_xlabel("")
    pos = df[df.is_positive]
    nc = pos.num_components.clip(upper=8)
    for i, (s, g) in enumerate(pos.assign(nc=nc).groupby("dataset_source")):
        vc = g.nc.value_counts().sort_index()
        axes[2].bar(vc.index + (i - 0.5) * 0.38, vc.values, 0.38, color=_src_color(s), edgecolor="black", label=s)
    axes[2].set(xlabel="Số vùng lỗi liên thông / ảnh (8 = ≥8)", ylabel="Số ảnh")
    axes[2].set_title("Số vùng lỗi mỗi ảnh", fontweight="bold")
    axes[2].legend()
    fig.tight_layout()
    return fig


def plot_mt_classes(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Magnetic Tile: số ảnh và diện tích lỗi theo loại khuyết tật chính."""
    mt = df[(df.dataset_source == "MagneticTile") & df.is_positive]
    if mt.empty:
        return None
    order = mt.primary_class.value_counts().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    vc = mt.primary_class.value_counts().reindex(order)
    bars = axes[0].bar(vc.index, vc.values, color="#0d9488", edgecolor="black")
    _bar_labels(axes[0], bars)
    axes[0].set(ylabel="Số ảnh")
    axes[0].set_title("Magnetic Tile: số ảnh theo loại lỗi chính", fontweight="bold")
    axes[1].boxplot([mt[mt.primary_class == c].defect_area_px for c in order], tick_labels=order, patch_artist=True)
    axes[1].set_yscale("log")
    axes[1].axhline(D.SMALL_MAX_PX, color="k", ls="--", lw=1)
    axes[1].axhline(D.MEDIUM_MAX_PX, color="k", ls=":", lw=1)
    axes[1].set_ylabel("Diện tích lỗi (px², log)")
    axes[1].set_title("Diện tích lỗi theo loại", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_splits(df: pd.DataFrame) -> plt.Figure:
    """Phân bố Train/Val/Test: theo nguồn, theo positive/negative, theo kích thước."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    sp = [s for s in D.SPLITS if s in set(df.split)]
    t = df.groupby(["split", "dataset_source"]).size().unstack(fill_value=0).reindex(sp)
    t.plot(kind="bar", stacked=True, ax=axes[0], color=[_src_color(c) for c in t.columns], edgecolor="black", rot=0)
    axes[0].set_title("Số ảnh theo split và nguồn", fontweight="bold")
    p = df.groupby(["split", "is_positive"]).size().unstack(fill_value=0).reindex(sp)
    p = p.rename(columns={True: "Positive", False: "Negative"})
    p.plot(kind="bar", ax=axes[1], color={"Positive": POS_COLOR, "Negative": NEG_COLOR}, edgecolor="black", rot=0)
    for c in axes[1].containers:
        axes[1].bar_label(c, fontsize=8)
    axes[1].set_title("Positive / Negative theo split", fontweight="bold")
    z = df[df.is_positive].groupby(["split", "size_category"]).size().unstack(fill_value=0)
    z = z.reindex(index=sp, columns=D.SIZE_ORDER, fill_value=0)
    z.plot(kind="bar", stacked=True, ax=axes[2], color=[SIZE_COLORS[c] for c in z.columns], edgecolor="black", rot=0)
    axes[2].set_title("Small/Medium/Large (positive) theo split", fontweight="bold")
    for a in axes:
        a.set_xlabel("")
        a.margins(y=0.12)
    fig.tight_layout()
    return fig


def _load_overlay(rec: "D.SampleRecord"):
    img = np.array(Image.open(rec.img_path).convert("RGB"))
    with open(rec.ann_path, encoding="utf-8") as f:
        mask, _ = D.ann_to_mask(json.load(f), rec.width, rec.height)
    return img, mask


def plot_samples(records: List["D.SampleRecord"], per_group: int = 1, seed: int = 42) -> Optional[plt.Figure]:
    """Ảnh | mask | overlay | boundary — mỗi (nguồn, Small/Medium/Large) lấy `per_group` mẫu."""
    rng = random.Random(seed)
    picks = []
    for s in sorted({r.source for r in records}):
        for c in D.SIZE_ORDER:
            pool = [r for r in records if r.source == s and r.size_category == c]
            picks += rng.sample(pool, min(per_group, len(pool)))
    if not picks:
        return None
    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.1 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, r in zip(axes, picks):
        img, mask = _load_overlay(r)
        ov = img.copy()
        ov[mask > 0] = (0.4 * ov[mask > 0] + 0.6 * np.array([255, 30, 30])).astype(np.uint8)
        b = D.extract_boundary(mask)
        for a, im, ttl, cm in zip(row, (img, mask, ov, b),
                                  (f"{r.sample_id}", f"Mask ({r.area_px:,}px², {r.size_category})",
                                   f"Overlay [{'/'.join(r.classes)}]", f"Boundary ({r.perimeter_px:.0f}px)"),
                                  (None, "gray", None, "hot")):
            a.imshow(im, cmap=cm)
            a.set_title(ttl, fontsize=9)
            a.axis("off")
    fig.tight_layout()
    return fig


def plot_augmentation(rec: "D.SampleRecord", n: int = 5, seed: int = 0) -> plt.Figure:
    """Minh họa augmentation đồng bộ: hàng trên = ảnh, hàng dưới = mask; cột 0 = bản gốc."""
    img, mask = _load_overlay(rec)
    aug = D.CoupledAugmentor(seed=seed)
    fig, axes = plt.subplots(2, n + 1, figsize=(2.6 * (n + 1), 6.2))
    axes[0, 0].imshow(img)
    axes[1, 0].imshow(mask, cmap="gray")
    axes[0, 0].set_title("Gốc")
    for k in range(1, n + 1):
        a_img, a_mask = aug(img, mask)
        axes[0, k].imshow(a_img)
        axes[1, k].imshow(a_mask, cmap="gray")
        axes[0, k].set_title(f"Aug #{k}")
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(f"Coupled augmentation — {rec.sample_id}", y=1.0)
    fig.tight_layout()
    return fig


# ==============================================================================
# PIPELINE
# ==============================================================================

def audit_all(ksdd2_dir: Path, mt_dir: Path):
    """Quét cả hai dataset. Trả về (records, reports {tên nguồn: report})."""
    records, reports = [], {}
    for spec in (D.make_ksdd2_spec(ksdd2_dir), D.make_mt_spec(mt_dir)):
        if not spec.subsets:
            print(f"  [Cảnh báo] Không tìm thấy dữ liệu {spec.name} tại: {spec.root}")
            continue
        recs, rep = D.scan_dataset(spec)
        records += recs
        reports[spec.name] = rep
        npos = sum(r.is_positive for r in recs)
        print(f"  {spec.name:13s}: {rep['n_images']:,} ảnh / {rep['n_ann_files']:,} ann -> hợp lệ {len(recs):,} "
              f"(positive {npos:,} | negative {len(recs) - npos:,}) | thiếu ann {len(rep['missing_ann'])}, "
              f"ann mồ côi {len(rep['orphan_ann'])}, ảnh hỏng {len(rep['corrupt_images'])}, "
              f"ann hỏng {len(rep['bad_ann_json'])}")
    return records, reports


def save_all_figures(df, records, reports, comp_df, figures_dir: Path, dpi: int = 200) -> List[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    pos_recs = [r for r in records if r.is_positive]
    jobs = [
        ("01_positive_negative_distribution", lambda: plot_pos_neg(df)),
        ("02_image_size_aspect_ratio", lambda: plot_image_size(df)),
        ("03_mask_integrity_audit", lambda: plot_mask_integrity(df, reports)),
        ("04_defect_area_distribution", lambda: plot_defect_area(df)),
        ("05_defect_size_categories", lambda: plot_size_categories(df)),
        ("06_defect_area_ecdf", lambda: plot_area_ecdf(df)),
        ("07_defect_components", lambda: plot_components(df, comp_df)),
        ("08_magnetic_tile_classes", lambda: plot_mt_classes(df)),
        ("09_train_val_test_split", lambda: plot_splits(df)),
        ("10_segmentation_samples", lambda: plot_samples(pos_recs)),
        ("11_augmentation_preview", lambda: plot_augmentation(pos_recs[0]) if pos_recs else None),
    ]
    saved = []
    for name, fn in jobs:
        fig = fn()
        if fig is None:
            continue
        path = figures_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
    return saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dataset audit + EDA + chia split + xuất data_segmentation/")
    ap.add_argument("--ksdd2-dir", type=Path, default=PROJECT_DIR / "kolektorsdd2-DatasetNinja")
    ap.add_argument("--mt-dir", type=Path, default=PROJECT_DIR / "magnetic-tile-surface-defect-DatasetNinja")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "data_segmentation")
    ap.add_argument("--figures-dir", type=Path, default=PROJECT_DIR / "eda_figures")
    ap.add_argument("--csv", type=Path, default=PROJECT_DIR / "dataset_statistics.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-negatives", action="store_true", help="xuất cả ảnh sạch vào data_segmentation/")
    ap.add_argument("--resplit-ksdd2", action="store_true",
                    help="chia lại KSDD2 70/15/15 thay vì giữ tập test chính thức")
    ap.add_argument("--check-duplicates", action="store_true", help="báo cáo ảnh gần trùng (dHash)")
    ap.add_argument("--no-export", action="store_true", help="chỉ audit + EDA, không xuất data_segmentation/")
    ap.add_argument("--overwrite", action="store_true", help="ghi đè train/val/test trong --out-dir")
    a = ap.parse_args(argv)

    print("=" * 78)
    print(" NGƯỜI 1 — DATASET & EDA | KolektorSDD2 + Magnetic Tile")
    print("=" * 78)
    print("\n[1/5] Kiểm toán dataset (ghép ảnh–ann, giải mã mask, đo khuyết tật)...")
    records, reports = audit_all(a.ksdd2_dir, a.mt_dir)
    if not records:
        print("Không đọc được mẫu nào. Kiểm tra lại --ksdd2-dir / --mt-dir.")
        return 1

    print("\n[2/5] Chia Train/Val/Test phân tầng...")
    D.assign_splits(records, seed=a.seed, respect_official=not a.resplit_ksdd2)
    df = D.records_to_dataframe(records, base_dir=PROJECT_DIR)
    comp_df = D.components_dataframe(records)
    print(pd.crosstab([df.dataset_source, df.is_positive], df.split, margins=True).to_string())

    if a.check_duplicates:
        pairs = D.find_near_duplicates(records)
        by_id = {r.sample_id: r for r in records}
        cross = [p for p in pairs if by_id[p[0]].split != by_id[p[1]].split]
        print(f"  Cặp ảnh gần trùng: {len(pairs)} (trong đó khác split — rò rỉ tiềm ẩn: {len(cross)})")

    print("\n[3/5] Ghi dataset_statistics.csv + dataset_audit.json...")
    df["exported"] = df.is_positive | a.include_negatives
    df.to_csv(a.csv, index=False, encoding="utf-8-sig")
    audit_json = a.csv.with_name("dataset_audit.json")
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump({"thresholds_px2": {"small_lt": D.SMALL_MAX_PX, "medium_le": D.MEDIUM_MAX_PX},
                   "reports": reports,
                   "integrity_counts": df.integrity_status.value_counts().to_dict()},
                  f, indent=2, ensure_ascii=False)
    print(f"  -> {a.csv} ({len(df):,} dòng), {audit_json}")

    print("\n[4/5] Sinh biểu đồ EDA...")
    saved = save_all_figures(df, records, reports, comp_df, a.figures_dir)
    print(f"  -> {len(saved)} biểu đồ trong {a.figures_dir}")

    if a.no_export:
        print("\n[5/5] Bỏ qua xuất dataset (--no-export).")
    else:
        print(f"\n[5/5] Xuất dataset ra {a.out_dir} ...")
        counts = D.export_dataset(records, a.out_dir, include_negatives=a.include_negatives, overwrite=a.overwrite)
        print("  Số mẫu:", counts)
        v = D.verify_export(a.out_dir)
        print(f"  Kiểm tra sau xuất: {v['n_checked']:,} ảnh, {len(v['problems'])} vấn đề")
        for p in v["problems"][:10]:
            print("   !", p)
        if v["problems"]:
            return 2
    print("\nHOÀN TẤT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
