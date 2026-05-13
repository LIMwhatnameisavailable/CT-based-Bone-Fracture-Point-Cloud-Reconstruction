#!/usr/bin/env python
"""
Step 2 — 点云采样
对三角网格表面均匀采样，输出归一化点云 .npz。
"""

import os, json, sys, time
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
# 0. 路径与配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHES_DIR = os.path.join(PROJECT_ROOT, 'data', 'meshes')
PC_DIR = os.path.join(PROJECT_ROOT, 'data', 'pointclouds')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results', 'step2_pointcloud')
SELECTED_JSON = os.path.join(PROJECT_ROOT, 'results', 'step0_dataset_overview', 'selected_cases.json')
ERROR_LOG = os.path.join(PC_DIR, 'error_log.txt')
SUMMARY_TXT = os.path.join(FIG_DIR, 'step2_summary.txt')

os.makedirs(PC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_TOTAL = 4096


# ============================================================
# 1. 加载选中列表
# ============================================================
def load_selected(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    grouped = {}
    for e in entries:
        c = e['case']
        if c not in grouped:
            grouped[c] = []
        grouped[c].append(e)
    grouped = dict(sorted(grouped.items()))
    print(f"  Loaded {len(entries)} entries, {len(grouped)} unique cases")
    return grouped, entries


# ============================================================
# 2. 断点续跑
# ============================================================
def get_existing_npzs():
    existing = set()
    if not os.path.isdir(PC_DIR):
        return existing
    for f in os.listdir(PC_DIR):
        if f.endswith('.npz'):
            existing.add(f.replace('.npz', ''))
    return existing


# ============================================================
# 3. 采样与归一化
# ============================================================
def sample_pointcloud(mesh, n_points):
    """表面积加权均匀采样，返回 points_raw 和 face_indices"""
    points_raw, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    return points_raw, face_indices


def compute_normals(mesh, face_indices):
    """从 face_indices 获取法向量并显式归一化"""
    n = mesh.face_normals[face_indices]
    mag = np.linalg.norm(n, axis=1, keepdims=True)
    mag = np.where(mag < 1e-8, 1.0, mag)  # 防止除零
    return n / mag


def normalize_pointcloud(points_raw):
    """中心化 + 单位球缩放，返回 (normalized, centroid, scale)"""
    centroid = points_raw.mean(axis=0)
    centered = points_raw - centroid
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    return normalized, centroid, scale


# ============================================================
# 4. matplotlib 设置
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
# 5. 可视化
# ============================================================
def plot_pointcloud(case_id, bone_name, bone_group, points, output_path):
    """
    点云可视化，1 行 2 列：
    - 左图：前视图 (X-Z) 散点，颜色按 Z 值
    - 右图：XY 平面 hexbin 密度热力图
    """
    setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -- 左图：前视图 (X-Z) --
    ax_left = axes[0]
    sc = ax_left.scatter(points[:, 0], points[:, 2], c=points[:, 2],
                         cmap='viridis', s=1.0, alpha=0.6, rasterized=True)
    ax_left.set_title('Front View (X-Z)', fontsize=14)
    ax_left.set_xlabel('X (normalized)', fontsize=14)
    ax_left.set_ylabel('Z (normalized)', fontsize=14)
    ax_left.tick_params(labelsize=11)
    ax_left.set_aspect('equal')
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    cbar_left = fig.colorbar(sc, ax=ax_left, shrink=0.8)
    cbar_left.set_label('Z (normalized)', fontsize=11)
    cbar_left.ax.tick_params(labelsize=10)

    # -- 右图：XY 平面密度热力图 --
    ax_right = axes[1]
    hb = ax_right.hexbin(points[:, 0], points[:, 1], gridsize=30,
                         cmap='Blues', extent=(-1.1, 1.1, -1.1, 1.1))
    ax_right.set_title('Top View Density (X-Y)', fontsize=14)
    ax_right.set_xlabel('X (normalized)', fontsize=14)
    ax_right.set_ylabel('Y (normalized)', fontsize=14)
    ax_right.tick_params(labelsize=11)
    ax_right.set_aspect('equal')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    cbar_right = fig.colorbar(hb, ax=ax_right, shrink=0.8)
    cbar_right.set_label('Count', fontsize=11)
    cbar_right.ax.tick_params(labelsize=10)

    fig.suptitle(f'{bone_group} | {case_id}_{bone_name} | N={N_TOTAL}',
                 fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 6. 主流程
# ============================================================
def main():
    os.makedirs(PC_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_matplotlib()

    print("=" * 60)
    print("Step 2 — 点云采样")
    print("=" * 60)

    # 1. 读取
    print("\n[1/4] Loading selected_cases.json...")
    grouped_entries, all_entries = load_selected(SELECTED_JSON)

    # 2. 断点续跑
    print("\n[2/4] Checking existing .npz files...")
    existing = get_existing_npzs()
    total_requested = len(all_entries)
    total_skipped = 0
    total_success = 0
    total_failed = 0
    norm_magnitudes = []
    error_lines = []
    if os.path.exists(ERROR_LOG):
        with open(ERROR_LOG, 'r', encoding='utf-8') as f:
            error_lines = f.read().splitlines()

    # 3. 逐条目采样
    print(f"\n[3/4] Sampling {total_requested} point clouds...")
    start_t = time.time()

    for case_idx, (case_id, entries) in enumerate(grouped_entries.items(), 1):
        print(f"\n  Case {case_idx}/{len(grouped_entries)}: {case_id} ({len(entries)} bones)")

        for entry in entries:
            bone = entry['bone']
            bone_group = entry['bone_group']
            npz_name = f"{case_id}_{bone}.npz"
            npz_path = os.path.join(PC_DIR, npz_name)

            if npz_name.replace('.npz', '') in existing:
                total_skipped += 1
                continue

            ply_path = os.path.join(MESHES_DIR, f"{case_id}_{bone}.ply")
            if not os.path.exists(ply_path):
                msg = f"  [SKIP] {bone}: .ply not found"
                print(f"    {msg}")
                error_lines.append(f"{case_id}_{bone}: {msg}")
                total_skipped += 1
                continue

            try:
                mesh = trimesh.load(ply_path)
                if len(mesh.vertices) < 100:
                    msg = f"too few vertices ({len(mesh.vertices)})"
                    print(f"    [SKIP] {bone:30s} -> {msg}")
                    error_lines.append(f"{case_id}_{bone}: {msg}")
                    total_skipped += 1
                    continue

                # 采样
                points_raw, face_indices = sample_pointcloud(mesh, N_TOTAL)
                normals = compute_normals(mesh, face_indices)
                points_norm, centroid, scale = normalize_pointcloud(points_raw)

                # 保存（float32）
                np.savez(npz_path,
                         points=points_norm.astype(np.float32),
                         normals=normals.astype(np.float32),
                         points_raw=points_raw.astype(np.float32),
                         centroid=centroid.astype(np.float32),
                         scale=np.float32(scale))
                total_success += 1

                # 法向量模长统计
                mag = np.linalg.norm(normals, axis=1).mean()
                norm_magnitudes.append(mag)

                print(f"    [OK] {bone:30s} -> {npz_name}  "
                      f"(scale={scale:.1f}, |n|={mag:.4f})")

            except Exception as e:
                print(f"    [FAIL] {bone}: {e}")
                error_lines.append(f"{case_id}_{bone}: {e}")
                total_failed += 1

    elapsed = time.time() - start_t
    print(f"\n  Sampling time: {elapsed:.1f}s")

    # 保存 error log
    if error_lines:
        unique_errors = list(dict.fromkeys(error_lines))
        with open(ERROR_LOG, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_errors))
        print(f"\n  [INFO] Error log saved: {ERROR_LOG} ({len(unique_errors)} entries)")

    # 4. 可视化
    print("\n[4/4] Generating representative visualizations...")

    rep_map = {}
    for e in all_entries:
        bg = e['bone_group']
        if bg not in rep_map or e['volume_cm3'] > rep_map[bg]['volume_cm3']:
            rep_map[bg] = e

    vis_count = 0
    for bg, rep in sorted(rep_map.items()):
        rc = rep['case']
        rb = rep['bone']
        npz_path = os.path.join(PC_DIR, f"{rc}_{rb}.npz")

        if not os.path.exists(npz_path):
            print(f"    [SKIP] {bg}: {rc}_{rb}.npz not found")
            continue

        try:
            data = np.load(npz_path)
            points = data['points']

            out_path = os.path.join(FIG_DIR, f'{bg}_{rc}_{rb}.png')
            plot_pointcloud(rc, rb, bg, points, out_path)
            vis_count += 1
            print(f"    [OK] {bg:25s} -> {os.path.basename(out_path)}")

        except Exception as e:
            print(f"    [FAIL] {bg}: {e}")

    # 5. 统计
    print("\n" + "=" * 60)
    print("Step 2 — Summary")
    print("=" * 60)
    lines = []
    lines.append(f"Total requested:  {total_requested}")
    lines.append(f"Success:          {total_success}")
    lines.append(f"Skipped (resume): {total_skipped}")
    lines.append(f"Failed:           {total_failed}")
    if total_success > 0:
        lines.append(f"Norm magnitude mean: {np.mean(norm_magnitudes):.4f} "
                     f"(should be ≈1.0)")
    lines.append(f"Visualizations:   {vis_count}/{len(rep_map)} groups")
    lines.append(f"Total time:       {elapsed:.1f}s")

    for l in lines:
        print(f"  {l}")
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  [INFO] Summary saved: {SUMMARY_TXT}")


if __name__ == '__main__':
    main()
