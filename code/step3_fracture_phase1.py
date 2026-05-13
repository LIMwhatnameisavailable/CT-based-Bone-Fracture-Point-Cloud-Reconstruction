#!/usr/bin/env python
"""
Step 3 — 模拟骨折破碎（阶段一：随机平面切割）
对完整骨骼网格进行随机平面切割，生成 2~4 块骨折碎片并施加随机位移。
"""

import os, json, sys, time
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ============================================================
# 0. 路径与配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHES_DIR = os.path.join(PROJECT_ROOT, 'data', 'meshes')
PC_DIR = os.path.join(PROJECT_ROOT, 'data', 'pointclouds')
FRAC_DIR = os.path.join(PROJECT_ROOT, 'data', 'fractured')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results', 'step3_fracture')
SELECTED_JSON = os.path.join(PROJECT_ROOT, 'results', 'step0_dataset_overview', 'selected_cases.json')
ERROR_LOG = os.path.join(FRAC_DIR, 'error_log.txt')
SUMMARY_TXT = os.path.join(FIG_DIR, 'step3_phase1_summary.txt')

os.makedirs(FRAC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_TOTAL = 4096
N_MIN_PER_FRAG = 30
FRAG_VERTEX_MIN = 50
ROT_RANGE = 30  # ±30 degrees
TRANS_RANGE = 20.0  # ±20 mm


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
def get_existing_fracs():
    existing = set()
    if not os.path.isdir(FRAC_DIR):
        return existing
    for f in os.listdir(FRAC_DIR):
        if f.endswith('_frac.npz'):
            existing.add(f.replace('_frac.npz', ''))
    return existing


# ============================================================
# 3. 切割工具
# ============================================================
def cut_mesh_by_plane(mesh, plane_origin, plane_normal):
    """用平面切割网格，返回 (above, below) 两个子网格，cap=True 自动封口"""
    above = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal, plane_origin, cap=True)
    below = trimesh.intersections.slice_mesh_plane(
        mesh, -plane_normal, plane_origin, cap=True)
    return above, below


# ============================================================
# 4. 随机平面切割流程
# ============================================================
def generate_cut_plane(mesh):
    """为网格生成一个随机切割平面，切割位置在 20%~80% 投影分位数范围内"""
    plane_normal = np.random.randn(3)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    proj = mesh.vertices @ plane_normal
    p_min, p_max = np.percentile(proj, 20), np.percentile(proj, 80)
    cut_pos = np.random.uniform(p_min, p_max)
    plane_origin = plane_normal * cut_pos

    return plane_origin, plane_normal


def fracture_mesh(mesh):
    """
    对网格执行随机平面切割，每次只切体积最大的碎片，避免指数增长。
    返回 (fragments, degenerates)，退化碎片面积分配给最近的有效碎片。
    """
    n_cuts = np.random.randint(1, 4)  # 1~3 刀，最终 2~4 块
    fragments = [mesh]
    degenerates = []

    for cut_idx in range(n_cuts):
        if len(fragments) == 0:
            break
        # 只切体积最大的那块，而不是所有碎片
        largest_idx = max(range(len(fragments)),
                          key=lambda i: fragments[i].volume
                          if hasattr(fragments[i], 'volume') else 0)
        target = fragments.pop(largest_idx)

        plane_origin, plane_normal = generate_cut_plane(target)
        above, below = cut_mesh_by_plane(target, plane_origin, plane_normal)

        for result in (above, below):
            if result is None or len(result.vertices) == 0:
                continue
            if len(result.vertices) < FRAG_VERTEX_MIN:
                degenerates.append((result, result.centroid))
            else:
                fragments.append(result)

    if len(fragments) == 0:
        fragments = [mesh]  # 切割完全失败时回退到原始网格

    return fragments, degenerates


# ============================================================
# 5. 采样与随机位移
# ============================================================
def sample_fragments(fragments, degenerates):
    """
    对碎片进行面积加权采样，退化碎片的面积分配给最近的有效碎片。
    返回 (points_mm, normals, fragment_ids)。
    """
    if len(fragments) == 0:
        return None, None, None

    # 计算各有效碎片的原始表面积
    areas = np.array([f.area for f in fragments])
    total_area = areas.sum()

    # 退化碎片的面积分配给最近的有效碎片
    if degenerates:
        valid_centroids = np.array([f.centroid for f in fragments])
        valid_tree = cKDTree(valid_centroids)
        for deg_mesh, _ in degenerates:
            if deg_mesh.area > 0:
                nearest_idx = valid_tree.query(deg_mesh.centroid)[1]
                areas[nearest_idx] += deg_mesh.area

    total_area = areas.sum()
    if total_area <= 0:
        return None, None, None

    # 按比例分配采样点数，最少 N_MIN_PER_FRAG
    n_points = np.maximum(
        np.round(areas / total_area * N_TOTAL).astype(int),
        N_MIN_PER_FRAG
    )

    # 调整总数到 N_TOTAL
    diff = n_points.sum() - N_TOTAL
    while diff != 0:
        if diff > 0:
            # 需要减少：从点数最多的碎片减
            idx = np.argmax(n_points)
            if n_points[idx] > N_MIN_PER_FRAG:
                n_points[idx] -= 1
                diff -= 1
            else:
                break
        else:
            # 需要增加：加到点数最多的碎片
            idx = np.argmax(n_points)
            n_points[idx] += 1
            diff += 1

    # 采样
    all_points = []
    all_normals = []
    all_frag_ids = []

    for fid, (frag, n_pts) in enumerate(zip(fragments, n_points)):
        if n_pts <= 0:
            continue
        pts_raw, face_idx = trimesh.sample.sample_surface(frag, n_pts)
        norms = frag.face_normals[face_idx]
        # 显式归一化法向量
        mag = np.linalg.norm(norms, axis=1, keepdims=True)
        mag = np.where(mag < 1e-8, 1.0, mag)
        norms = norms / mag

        all_points.append(pts_raw)
        all_normals.append(norms)
        all_frag_ids.append(np.full(n_pts, fid, dtype=np.int32))

    points_mm = np.concatenate(all_points, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    frag_ids = np.concatenate(all_frag_ids, axis=0)

    return points_mm, normals, frag_ids


def apply_random_transform(points_mm, frag_ids, n_frags):
    """
    对每块碎片施加随机旋转和平移（在毫米坐标系下）。
    返回 (displaced_points, transforms_4x4)。
    transforms_4x4: (n_frags, 4, 4) 齐次变换矩阵
    """
    displaced = np.zeros_like(points_mm)
    transforms_4x4 = np.zeros((n_frags, 4, 4), dtype=np.float32)

    for fid in range(n_frags):
        transforms_4x4[fid, 3, 3] = 1.0  # 齐次矩阵右下角为 1

        mask = frag_ids == fid
        pts = points_mm[mask]
        if len(pts) == 0:
            continue

        # 随机旋转轴
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(-ROT_RANGE, ROT_RANGE)
        angle_rad = np.deg2rad(angle)
        R = trimesh.transformations.rotation_matrix(angle_rad, axis)[:3, :3]

        # 随机平移
        t = np.random.uniform(-TRANS_RANGE, TRANS_RANGE, size=3)

        # 变换：points_displaced = points @ R.T + t
        displaced[mask] = pts @ R.T + t

        transforms_4x4[fid, :3, :3] = R.astype(np.float32)
        transforms_4x4[fid, :3, 3] = t.astype(np.float32)

    return displaced, transforms_4x4


# ============================================================
# 6. matplotlib 设置
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
# 7. 可视化
# ============================================================
def plot_fracture(case_id, bone_name, bone_group, intact_points,
                  frac_points, frag_ids, n_frags, max_t, output_path):
    """
    破碎可视化，1×2 布局：
    - 左图：完整骨骼点云（灰色）
    - 右图：破碎后各碎片按 fragment_id 用 tab10 着色
    """
    setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -- 左图：完整点云（灰色） --
    ax_left = axes[0]
    ax_left.scatter(intact_points[:, 0], intact_points[:, 2],
                    c='gray', s=1.0, alpha=0.5, rasterized=True)
    ax_left.set_title('Intact Bone', fontsize=14)
    ax_left.set_xlabel('X (normalized)', fontsize=14)
    ax_left.set_ylabel('Z (normalized)', fontsize=14)
    ax_left.tick_params(labelsize=11)
    ax_left.set_aspect('equal')
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)

    # -- 右图：破碎后各碎片用不同颜色 --
    ax_right = axes[1]
    cmap = plt.get_cmap('tab10', n_frags)
    for fid in range(n_frags):
        mask = frag_ids == fid
        pts = frac_points[mask]
        color = cmap(fid)
        ax_right.scatter(pts[:, 0], pts[:, 2],
                         c=[color], s=1.0, alpha=0.6, rasterized=True,
                         label=f'Frag {fid}')

    ax_right.set_title('Fractured', fontsize=14)
    ax_right.set_xlabel('X (normalized)', fontsize=14)
    ax_right.set_ylabel('Z (normalized)', fontsize=14)
    ax_right.tick_params(labelsize=11)
    ax_right.set_aspect('equal')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.legend(fontsize=9, markerscale=3, loc='upper right')

    fig.suptitle(f'{bone_group} | {case_id}_{bone_name} | '
                 f'n_frags={n_frags} | max_t={max_t:.1f}mm',
                 fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 8. 主流程
# ============================================================
def main():
    os.makedirs(FRAC_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_matplotlib()

    print("=" * 60)
    print("Step 3 — 模拟骨折破碎（阶段一：随机平面切割）")
    print("=" * 60)

    # 1. 读取
    print("\n[1/4] Loading selected_cases.json...")
    grouped_entries, all_entries = load_selected(SELECTED_JSON)

    # 2. 断点续跑
    print("\n[2/4] Checking existing fractured .npz files...")
    existing = get_existing_fracs()
    total_requested = len(all_entries)
    total_skipped = 0
    total_success = 0
    total_failed = 0
    error_lines = []
    if os.path.exists(ERROR_LOG):
        with open(ERROR_LOG, 'r', encoding='utf-8') as f:
            error_lines = f.read().splitlines()

    # 3. 逐条目破碎
    print(f"\n[3/4] Fracturing {total_requested} entries...")
    start_t = time.time()
    max_displacements = []

    for case_idx, (case_id, entries) in enumerate(grouped_entries.items(), 1):
        print(f"\n  Case {case_idx}/{len(grouped_entries)}: {case_id} ({len(entries)} bones)")

        for entry in entries:
            bone = entry['bone']
            bone_group = entry['bone_group']
            frac_name = f"{case_id}_{bone}_frac.npz"
            frac_path = os.path.join(FRAC_DIR, frac_name)

            if frac_name.replace('_frac.npz', '') in existing:
                total_skipped += 1
                continue

            # 加载 mesh
            ply_path = os.path.join(MESHES_DIR, f"{case_id}_{bone}.ply")
            if not os.path.exists(ply_path):
                msg = f"  [SKIP] {bone}: .ply not found"
                print(f"    {msg}")
                error_lines.append(f"{case_id}_{bone}: {msg}")
                total_skipped += 1
                continue

            # 加载 Step 2 .npz（获取 centroid 和 scale）
            pc_path = os.path.join(PC_DIR, f"{case_id}_{bone}.npz")
            if not os.path.exists(pc_path):
                msg = f"  [SKIP] {bone}: Step 2 .npz not found"
                print(f"    {msg}")
                error_lines.append(f"{case_id}_{bone}: {msg}")
                total_skipped += 1
                continue

            try:
                pc_data = np.load(pc_path)
                centroid = pc_data['centroid']
                scale = pc_data['scale']
                intact_points = pc_data['points']

                mesh = trimesh.load(ply_path)

                # 切割
                fragments, degenerates = fracture_mesh(mesh)

                if len(fragments) == 0:
                    msg = "  [FAIL] {bone}: no valid fragments after cutting"
                    print(f"    {msg}")
                    error_lines.append(f"{case_id}_{bone}: {msg}")
                    total_failed += 1
                    continue

                # 采样
                points_mm, normals, frag_ids = sample_fragments(fragments, degenerates)
                if points_mm is None:
                    msg = f"  [FAIL] {bone}: sampling failed"
                    print(f"    {msg}")
                    error_lines.append(f"{case_id}_{bone}: {msg}")
                    total_failed += 1
                    continue

                n_frags = len(np.unique(frag_ids))

                # 随机位移
                displaced_mm, transforms_4x4 = apply_random_transform(
                    points_mm, frag_ids, n_frags)

                # 归一化（使用 Step 2 的 centroid 和 scale）
                points_norm = (displaced_mm - centroid) / scale

                # 计算最大平移量
                max_t = max(np.linalg.norm(t) for t in transforms_4x4[:, :3, 3])
                max_displacements.append(max_t)

                # 保存 .npz
                np.savez(frac_path,
                         points=points_norm.astype(np.float32),
                         fragment_id=frag_ids.astype(np.int32),
                         normals=normals.astype(np.float32),
                         points_raw=displaced_mm.astype(np.float32),
                         centroid=centroid.astype(np.float32),
                         scale=np.float32(scale),
                         transforms=transforms_4x4.astype(np.float32))

                total_success += 1
                print(f"    [OK] {bone:30s} -> {frac_name}  "
                      f"(frags={n_frags}, max_t={max_t:.1f}mm)")

            except Exception as e:
                msg = f"  [FAIL] {bone}: {e}"
                print(f"    {msg}")
                error_lines.append(f"{case_id}_{bone}: {e}")
                total_failed += 1

    elapsed = time.time() - start_t
    print(f"\n  Fracture time: {elapsed:.1f}s")

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
        frac_path = os.path.join(FRAC_DIR, f"{rc}_{rb}_frac.npz")
        pc_path = os.path.join(PC_DIR, f"{rc}_{rb}.npz")

        if not os.path.exists(frac_path) or not os.path.exists(pc_path):
            print(f"    [SKIP] {bg}: data not found")
            continue

        try:
            frac_data = np.load(frac_path, allow_pickle=True)
            pc_data = np.load(pc_path)

            intact_pts = pc_data['points']
            frac_pts = frac_data['points']
            frag_ids = frac_data['fragment_id']
            transforms_data = frac_data['transforms']
            n_frags = len(np.unique(frag_ids))
            max_t = max(np.linalg.norm(t) for t in transforms_data[:, :3, 3])

            out_path = os.path.join(FIG_DIR, 'phase1', f'{bg}_{rc}_{rb}_phase1.png')
            plot_fracture(rc, rb, bg, intact_pts, frac_pts,
                          frag_ids, n_frags, max_t, out_path)
            vis_count += 1
            print(f"    [OK] {bg:25s} -> {os.path.basename(out_path)}")

        except Exception as e:
            print(f"    [FAIL] {bg}: {e}")

    # 5. 统计
    print("\n" + "=" * 60)
    print("Step 3 — Summary")
    print("=" * 60)
    lines = []
    lines.append(f"Total requested:  {total_requested}")
    lines.append(f"Success:          {total_success}")
    lines.append(f"Skipped (resume): {total_skipped}")
    lines.append(f"Failed:           {total_failed}")
    if max_displacements:
        lines.append(f"Max displacement: {np.mean(max_displacements):.1f}mm (avg)")
    lines.append(f"Visualizations:   {vis_count}/{len(rep_map)} groups")
    lines.append(f"Total time:       {elapsed:.1f}s")

    for l in lines:
        print(f"  {l}")
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  [INFO] Summary saved: {SUMMARY_TXT}")


if __name__ == '__main__':
    main()
