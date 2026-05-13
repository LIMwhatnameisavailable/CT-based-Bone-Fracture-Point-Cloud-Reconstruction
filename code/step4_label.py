#!/usr/bin/env python
"""
Step 4 — 断面点标注（碎片间距离法）

在恢复的原始坐标系中，用碎片间最短距离判断断面点，
与 Step 3 boundary_mask 生成逻辑一致，确保独立计算的高度一致性。
"""
import os, json, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import defaultdict

# ============================================================
# 0. 路径与配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAC_DIR = os.path.join(PROJECT_ROOT, 'data', 'fractured')
LABELED_DIR = os.path.join(PROJECT_ROOT, 'data', 'labeled')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results', 'step4_label')
SELECTED_JSON = os.path.join(PROJECT_ROOT, 'results', 'step0_dataset_overview', 'selected_cases.json')
SUMMARY_TXT = os.path.join(FIG_DIR, 'step4_summary.txt')

os.makedirs(LABELED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

K_NEIGHBORS = 64
DISC_PERCENTILE = 80

# ============================================================
# 1. 计算原始空间坐标
# ============================================================
def recover_raw_coords(points, normals, fragment_id, transforms, centroid, scale):
    """
    从位移后归一化坐标恢复位移前原始毫米坐标。

    voronoi.npz 中 points 经过了：归一化（居中+缩放）+ 碎片位移（旋转+平移）。
    利用 transform 矩阵反推回原始空间，在该空间中碎片处于相邻位置，
    断面是碎片间的内部界面，相邻碎片的法向量方向相反。

    前向变换：pts_norm = (pts_raw @ R.T + t - centroid) / scale
    反向变换：pts_raw = (pts_norm * scale + centroid - t) @ R

    注意：normals 在 step3 中从未被旋转（只有 points 被位移+归一化），
    因此 normals 已在 raw mm 空间，直接复制即可。
    """
    n = len(points)
    raw_pts = np.zeros_like(points)
    raw_nrms = normals.copy()  # normals 从未被旋转，已在 raw mm 空间

    for fid in np.unique(fragment_id):
        mask = (fragment_id == fid)
        idx = np.where(mask)[0]

        T = transforms[fid]  # (4, 4) homogeneous: [R | t; 0 | 1]
        R = T[:3, :3]
        t = T[:3, 3]

        # 反向变换：norm → displaced_mm → raw_mm
        pts_mm = points[idx] * float(scale) + centroid  # 归一化 → 毫米（位移后）
        raw_pts[idx] = (pts_mm - t) @ R                 # 毫米（位移后）→ 毫米（位移前）

    return raw_pts, raw_nrms


def normal_discontinuity_raw(points, normals, k=K_NEIGHBORS):
    """
    在原始空间（碎片相邻）计算法向量不连续性。

    关键差异：不使用 |dot|，所以反平行法向量（断面处相邻碎片法向量指向相反）
    给出高不连续性值。在原始空间中，断面点的邻域包含相邻碎片的点，
    其法向量与当前碎片法向量相反 → 不连续性高。
    """
    n = len(points)
    if n <= k + 1:
        return np.zeros(n, dtype=np.float32)

    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(k + 1, n))
    indices = indices[:, 1:]

    neighbor_normals = normals[indices]
    dot_products = np.einsum('ni,nki->nk', normals, neighbor_normals)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    # 不使用 np.abs：反平行法向量 → dot=-1 → arccos(-1)=π → 高不连续性
    angles = np.arccos(dot_products)
    return angles.mean(axis=1).astype(np.float32)


def compute_local_curvature(points, k=K_NEIGHBORS):
    """
    用 PCA 估计局部曲率：最小特征值 / 特征值之和。
    返回 (N,) 每点的曲率值。
    """
    n = len(points)
    if n <= k + 1:
        return np.zeros(n, dtype=np.float32)

    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(k + 1, n))
    indices = indices[:, 1:]

    curvatures = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbors = points[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / k
        eigenvalues = np.linalg.eigvalsh(cov)
        total = eigenvalues.sum()
        curvatures[i] = eigenvalues[0] / total if total > 1e-10 else 0.0
    return curvatures


def label_by_interdistance(raw_pts, frag_ids, tau_mm=4.0):
    """
    在原始坐标系（碎片相邻）中，对每个点计算到其他碎片的
    最短距离，距离 < tau_mm 的点标为断面点（label=1）。
    tau_mm=4.0 与 Step 3 TAU_MM 保持一致，保证可比性。
    复杂度 O(N log N)，使用 cKDTree。
    """
    n = len(raw_pts)
    labels = np.zeros(n, dtype=np.int32)
    unique_ids = np.unique(frag_ids)

    for fid in unique_ids:
        mask_i = (frag_ids == fid)
        pts_other = raw_pts[~mask_i]
        if len(pts_other) == 0:
            continue
        tree = cKDTree(pts_other)
        dists, _ = tree.query(raw_pts[mask_i])
        boundary_local = dists < tau_mm
        global_idx = np.where(mask_i)[0][boundary_local]
        labels[global_idx] = 1

    return labels


# ============================================================
# 2. 一致性评估
# ============================================================
def evaluate_consistency(geometric_label, boundary_mask):
    """计算 geometric_label vs boundary_mask 的一致性指标"""
    tp = np.sum((geometric_label == 1) & (boundary_mask == 1))
    fp = np.sum((geometric_label == 1) & (boundary_mask == 0))
    fn = np.sum((geometric_label == 0) & (boundary_mask == 1))
    tn = np.sum((geometric_label == 0) & (boundary_mask == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(geometric_label) if len(geometric_label) > 0 else 0.0

    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }


# ============================================================
# 3. matplotlib 设置
# ============================================================
def setup_matplotlib():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    })


# ============================================================
# 4. 可视化
# ============================================================
def plot_label_result(case_id, bone_name, bone_group,
                      points, labels, boundary_mask,
                      normal_disc, curvature,
                      metrics, output_path):
    """
    1×2 布局：
    左图：标注结果 XZ 投影（灰=非断面，红=断面）
    右图：混淆矩阵热力图，下方标注 P/R/F1
    """
    setup_matplotlib()

    label_rate = 100.0 * labels.sum() / len(labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -- 左图：标注结果 --
    ax_left = axes[0]
    non_boundary = labels == 0
    boundary = labels == 1

    ax_left.scatter(points[non_boundary, 0], points[non_boundary, 2],
                    c='gray', s=0.5, alpha=0.3, rasterized=True)
    ax_left.scatter(points[boundary, 0], points[boundary, 2],
                    c='red', s=2.0, alpha=0.8, rasterized=True)
    ax_left.set_title(f'Inter-distance Label ({label_rate:.1f}%)', fontsize=14)
    ax_left.set_xlabel('X (normalized)', fontsize=14)
    ax_left.set_ylabel('Z (normalized)', fontsize=14)
    ax_left.tick_params(labelsize=11)
    ax_left.set_aspect('equal')
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)

    # -- 右图：混淆矩阵 --
    ax_right = axes[1]
    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    im = ax_right.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max() + 1)
    ax_right.set_xticks([0, 1])
    ax_right.set_yticks([0, 1])
    ax_right.set_xticklabels(['Neg', 'Pos'], fontsize=11)
    ax_right.set_yticklabels(['Neg', 'Pos'], fontsize=11)
    ax_right.set_xlabel('Predicted (Geometric)', fontsize=11)
    ax_right.set_ylabel('Actual (Boundary Mask)', fontsize=11)

    for i in range(2):
        for j in range(2):
            ax_right.text(j, i, str(cm[i, j]),
                          ha='center', va='center', fontsize=14,
                          color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # 下方标注指标
    info_text = (f'Precision: {metrics["precision"]:.3f}  |  '
                 f'Recall: {metrics["recall"]:.3f}  |  '
                 f'F1: {metrics["f1"]:.3f}  |  '
                 f'Acc: {metrics["accuracy"]:.3f}')
    ax_right.text(0.5, -0.18, info_text,
                  transform=ax_right.transAxes, ha='center', fontsize=11,
                  color='#333333')

    fig.suptitle(f'{bone_group} | {case_id}_{bone_name} | '
                 f'Label Rate: {label_rate:.1f}% | F1: {metrics["f1"]:.3f}',
                 fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 5. 单条目处理
# ============================================================
def process_entry(case_id, bone_name, bone_group):
    """处理单个 case_bone 条目"""
    voronoi_path = os.path.join(FRAC_DIR, f"{case_id}_{bone_name}_voronoi.npz")
    labeled_path = os.path.join(LABELED_DIR, f"{case_id}_{bone_name}_labeled.npz")

    if not os.path.exists(voronoi_path):
        return None, "voronoi npz not found"

    data = np.load(voronoi_path, allow_pickle=True)
    points = data['points']
    normals = data['normals']
    frag_ids = data['fragment_id']
    boundary_mask = data['boundary_mask']
    transforms = data['transform']
    centroid = data['centroid']
    scale_val = data['scale']
    n = len(points)

    # 恢复原始空间坐标（所有碎片回到相邻位置）
    raw_pts, raw_nrms = recover_raw_coords(
        points, normals, frag_ids, transforms, centroid, scale_val)

    # 在原始空间计算几何特征（仅保存供分析，不参与标注）
    nd = normal_discontinuity_raw(raw_pts, raw_nrms)
    curv = compute_local_curvature(raw_pts)

    # 碎片间距离法标注（与 Step 3 boundary_mask 逻辑一致）
    all_labels = label_by_interdistance(raw_pts, frag_ids, tau_mm=4.0)

    # 一致性评估
    metrics = evaluate_consistency(all_labels, boundary_mask)

    # 保存
    np.savez(labeled_path,
             points=points.astype(np.float32),
             labels=all_labels.astype(np.int32),
             fragment_id=frag_ids.astype(np.int32),
             boundary_mask=boundary_mask.astype(bool),
             normal_disc=nd.astype(np.float32),
             curvature=curv.astype(np.float32))

    return metrics, None


# ============================================================
# 6. 主流程
# ============================================================
def main():
    os.makedirs(LABELED_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_matplotlib()

    print("=" * 60)
    print("Step 4 — 断面点几何标注与验证")
    print("=" * 60)

    # 1. 加载 selected_cases
    print("\n[1/5] Loading selected_cases.json...")
    with open(SELECTED_JSON, 'r', encoding='utf-8') as f:
        all_entries = json.load(f)
    print(f"  Loaded {len(all_entries)} entries")

    # 2. 断点续跑
    print("\n[2/5] Checking existing labeled files...")
    existing = set()
    if os.path.isdir(LABELED_DIR):
        for fname in os.listdir(LABELED_DIR):
            if fname.endswith('_labeled.npz'):
                existing.add(fname.replace('_labeled.npz', ''))
    print(f"  Existing: {len(existing)} entries")

    # 3. 逐条目处理
    print(f"\n[3/5] Processing {len(all_entries)} entries...")
    start_t = time.time()
    total_success = 0
    total_skipped = 0
    total_failed = 0
    all_metrics = []
    grouped_metrics = defaultdict(list)
    error_lines = []

    for idx, entry in enumerate(all_entries, 1):
        case_id = entry['case']
        bone_name = entry['bone']
        bone_group = entry['bone_group']
        key = f"{case_id}_{bone_name}"

        if key in existing:
            total_skipped += 1
            continue

        metrics, err = process_entry(case_id, bone_name, bone_group)

        if metrics is not None:
            total_success += 1
            all_metrics.append(metrics)
            grouped_metrics[bone_group].append(metrics['f1'])
            label_rate = 100.0 * (metrics['tp'] + metrics['fp']) / (
                metrics['tp'] + metrics['fp'] + metrics['tn'] + metrics['fn'])
            print(f"  [{idx}/{len(all_entries)}] {bone_name:30s} "
                  f"F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
                  f"R={metrics['recall']:.3f}  label={label_rate:.1f}%")
        else:
            total_failed += 1
            error_lines.append(f"{key}: {err}")
            print(f"  [{idx}/{len(all_entries)}] [FAIL] {bone_name}: {err}")

    elapsed = time.time() - start_t

    # 4. 可视化（每骨骼分组选代表）
    print(f"\n[4/5] Generating representative visualizations...")

    rep_map = {}
    for e in all_entries:
        bg = e['bone_group']
        if bg not in rep_map or e['volume_cm3'] > rep_map[bg]['volume_cm3']:
            rep_map[bg] = e

    vis_count = 0
    for bg, rep in sorted(rep_map.items()):
        case_id = rep['case']
        bone_name = rep['bone']
        labeled_path = os.path.join(LABELED_DIR, f"{case_id}_{bone_name}_labeled.npz")

        if not os.path.exists(labeled_path):
            print(f"    [SKIP] {bg}: labeled data not found")
            continue

        try:
            ld = np.load(labeled_path, allow_pickle=True)
            voronoi_path = os.path.join(FRAC_DIR, f"{case_id}_{bone_name}_voronoi.npz")
            vd = np.load(voronoi_path, allow_pickle=True)

            points = ld['points']
            labels = ld['labels']
            boundary_mask = ld['boundary_mask']
            normal_disc = ld['normal_disc']
            curvature = ld['curvature']

            metrics = evaluate_consistency(labels, boundary_mask)

            out_path = os.path.join(FIG_DIR, f'{bg}_{case_id}_{bone_name}_label.png')
            plot_label_result(case_id, bone_name, bg,
                              points, labels, boundary_mask,
                              normal_disc, curvature, metrics, out_path)
            vis_count += 1
            print(f"    [OK] {bg:25s} -> {os.path.basename(out_path)}")

        except Exception as e:
            import traceback
            print(f"    [FAIL] {bg}: {e}")
            traceback.print_exc()

    # 5. 汇总
    print(f"\n[5/5] Summary...")
    print("\n" + "=" * 60)
    print("Step 4 — Summary")
    print("=" * 60)

    lines = []
    lines.append(f"Total requested:  {len(all_entries)}")
    lines.append(f"Success:          {total_success}")
    lines.append(f"Skipped (resume): {total_skipped}")
    lines.append(f"Failed:           {total_failed}")
    lines.append(f"Total time:       {elapsed:.1f}s")

    if all_metrics:
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        label_rates = []
        for m in all_metrics:
            total = m['tp'] + m['fp'] + m['tn'] + m['fn']
            label_rates.append(100.0 * (m['tp'] + m['fp']) / total if total > 0 else 0)
        avg_label_rate = np.mean(label_rates)
        f1_above_085 = sum(1 for m in all_metrics if m['f1'] >= 0.85)
        f1_above_085_pct = 100.0 * f1_above_085 / len(all_metrics)

        lines.append(f"Avg label rate:   {avg_label_rate:.1f}%")
        lines.append(f"Avg Precision:    {avg_precision:.3f}")
        lines.append(f"Avg Recall:       {avg_recall:.3f}")
        lines.append(f"Avg F1:           {avg_f1:.3f}")
        lines.append(f"F1 >= 0.85:       {f1_above_085}/{len(all_metrics)} ({f1_above_085_pct:.1f}%)")
        lines.append(f"Visualizations:   {vis_count}/{len(rep_map)} groups")

        lines.append("")
        lines.append("F1 by bone group:")
        for bg in sorted(grouped_metrics.keys()):
            f1s = grouped_metrics[bg]
            lines.append(f"  {bg:25s}: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}  (n={len(f1s)})")

    for l in lines:
        print(f"  {l}")
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  [INFO] Summary saved: {SUMMARY_TXT}")


if __name__ == '__main__':
    main()
