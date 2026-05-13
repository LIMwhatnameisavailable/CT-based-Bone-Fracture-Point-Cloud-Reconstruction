#!/usr/bin/env python
"""
Step 3 — 模拟骨折破碎（阶段二：Voronoi 破碎）
对完整骨骼网格进行 Voronoi 破碎，生成 2~4 块骨折碎片并施加随机位移。
"""
import os, json, sys, time
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly not found, installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly', '-q'])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

# ============================================================
# 0. 路径与配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHES_DIR = os.path.join(PROJECT_ROOT, 'data', 'meshes')
PC_DIR = os.path.join(PROJECT_ROOT, 'data', 'pointclouds')
FRAC_DIR = os.path.join(PROJECT_ROOT, 'data', 'fractured')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results', 'step3_fracture')
PHASE2_FIG_DIR = os.path.join(FIG_DIR, 'phase2')
PHASE2_HTML_DIR = os.path.join(FIG_DIR, 'phase2_interactive')
SELECTED_JSON = os.path.join(PROJECT_ROOT, 'results', 'step0_dataset_overview', 'selected_cases.json')
ERROR_LOG = os.path.join(PROJECT_ROOT, 'code', 'step3_phase2_errors.txt')
SUMMARY_TXT = os.path.join(FIG_DIR, 'step3_phase2_summary.txt')

os.makedirs(FRAC_DIR, exist_ok=True)
os.makedirs(PHASE2_FIG_DIR, exist_ok=True)

N_TOTAL = 4096
N_MIN_PER_FRAG = 30
FRAG_VERTEX_MIN = 50
ROT_RANGE = 30          # ±30 degrees
TRANS_RANGE = 20.0      # ±20 mm
PERTURB_MAX = 0.3       # 断面扰动幅度上限 (mm)
ALPHA = 2.5             # 断面判定阈值系数：tau = ALPHA * d_nn（d_nn 为点云平均近邻间距）
TAU_MM = 4.0            # 物理空间断面判定阈值 (mm)，points_mm 为毫米坐标
MERGE_VOL_RATIO = 0.05  # 小碎片体积合并阈值（占总体积比例）


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
def get_existing_voronoi():
    """扫描 FRAC_DIR 已有 _voronoi.npz，返回 {case_bone} 集合"""
    existing = set()
    if not os.path.isdir(FRAC_DIR):
        return existing
    for f in os.listdir(FRAC_DIR):
        if f.endswith('_voronoi.npz'):
            existing.add(f.replace('_voronoi.npz', ''))
    return existing


# ============================================================
# 3. Voronoi 种子生成
# ============================================================
def generate_voronoi_seeds(mesh, n_seeds, rng):
    """
    用 SVD 求骨骼第一主轴，沿主轴等分位布种子，垂直方向加小幅扰动。
    """
    verts = mesh.vertices
    center = verts.mean(axis=0)
    centered = verts - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis_main, axis_2, axis_3 = Vt[0], Vt[1], Vt[2]

    proj = centered @ axis_main
    positions = np.linspace(np.percentile(proj, 20),
                            np.percentile(proj, 80), n_seeds)

    short_len = np.percentile(np.abs(centered @ axis_2), 90)
    long_len = np.percentile(np.abs(centered @ axis_main), 90)
    flatness = short_len / long_len
    jitter_scale = short_len * (0.10 + 0.20 * (1.0 - flatness))

    seeds = []
    for pos in positions:
        base = center + axis_main * pos
        jitter = (axis_2 * rng.uniform(-1, 1) +
                  axis_3 * rng.uniform(-1, 1)) * jitter_scale
        seeds.append(base + jitter)
    return np.array(seeds)


# ============================================================
# 4. 面片归属分配（按原始网格的面索引操作，避免 vertex re-index 问题）
# ============================================================
def assign_faces_to_seeds(mesh, seeds):
    """
    将每个三角面片分配给最近的种子点。
    返回 (face_indices_list, degenerates)：
    - face_indices_list: 每项为属于该种子的原始面索引数组
    - degenerates: 退化碎片的 (sub_mesh, centroid) 列表
    """
    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    tree = cKDTree(seeds)
    _, assignments = tree.query(face_centroids)

    n_seeds = len(seeds)
    seed_faces = []  # 每项为原始面索引数组
    degenerates = []

    for sid in range(n_seeds):
        face_idx = np.where(assignments == sid)[0]
        if len(face_idx) == 0:
            continue

        cluster_faces = mesh.faces[face_idx]
        sub_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=cluster_faces)

        # 保留最大连通分量
        components = sub_mesh.split(only_watertight=False)
        if components:
            sub_mesh = max(components, key=lambda m: len(m.vertices))

        if len(sub_mesh.vertices) < FRAG_VERTEX_MIN:
            degenerates.append((sub_mesh, sub_mesh.centroid))
        else:
            seed_faces.append(face_idx)

    return seed_faces, degenerates


# ============================================================
# 5. 小碎片合并（在原始面索引层面操作）
# ============================================================
def merge_small_fragments(mesh, seed_faces):
    """
    将体积 < 总体积 5% 的小碎片合并到质心最近的大碎片中。
    直接在原始面索引上操作，避免 vertex re-index 问题。
    返回合并后的 face_indices 列表。
    """
    n_frags = len(seed_faces)
    if n_frags <= 1:
        return seed_faces

    # 计算每个碎片的实际体积（非 watertight 时 fallback 到凸包体积）
    volumes = []
    for faces in seed_faces:
        sub_mesh = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces[faces],
            validate=False)
        vol = sub_mesh.volume
        if vol <= 0:
            try:
                vol = sub_mesh.convex_hull.volume
            except Exception:
                vol = sub_mesh.bounding_box.volume
        volumes.append(vol)
    volumes = np.array(volumes)

    total = volumes.sum()
    big_mask = volumes >= MERGE_VOL_RATIO * total
    small_mask = ~big_mask

    if not small_mask.any() or big_mask.sum() <= 1:
        return seed_faces

    # 计算大碎片的质心用于近邻分配
    big_indices = np.where(big_mask)[0]
    small_indices = np.where(small_mask)[0]

    big_centroids = []
    for i in big_indices:
        verts_of_faces = mesh.vertices[mesh.faces[seed_faces[i]]]
        centroids = verts_of_faces.mean(axis=1)
        big_centroids.append(centroids.mean(axis=0))
    big_centroids = np.array(big_centroids)
    big_tree = cKDTree(big_centroids)

    merged = {i: seed_faces[i].copy() for i in big_indices}
    for i in small_indices:
        # 小碎片质心
        verts_of_faces = mesh.vertices[mesh.faces[seed_faces[i]]]
        centroid = verts_of_faces.mean(axis=1).mean(axis=0)
        nearest_local = big_tree.query(centroid)[1]
        nearest_global = big_indices[nearest_local]
        merged[nearest_global] = np.union1d(merged[nearest_global], seed_faces[i])

    # 按 big_indices 顺序返回
    return [merged[i] for i in big_indices]


# ============================================================
# 5b. 从面索引重建碎片网格
# ============================================================
def build_fragments_from_faces(mesh, face_indices_list):
    """从原始网格的面索引列表重建碎片网格列表"""
    fragments = []
    degenerates = []
    for face_idx in face_indices_list:
        cluster_faces = mesh.faces[face_idx]
        sub_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=cluster_faces)

        components = sub_mesh.split(only_watertight=False)
        if components:
            sub_mesh = max(components, key=lambda m: len(m.vertices))

        if len(sub_mesh.vertices) < FRAG_VERTEX_MIN:
            degenerates.append((sub_mesh, sub_mesh.centroid))
        else:
            fragments.append(sub_mesh)
    return fragments, degenerates


# ============================================================
# 6. 断面凹凸扰动（基于局部采样密度的阈值）
# ============================================================
def estimate_sampling_density(points, k=2):
    """估计点云的平均近邻距离（采样密度尺度）。

    Args:
        points: (N, 3) 点云
        k: 取第 k 个最近邻（k=2 因为 k=1 是自己，距离为 0）

    Returns:
        d_nn: float, 平均近邻间距
    """
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k)
    return float(np.median(dists[:, k-1]))


def perturb_fracture_surface(points_mm, normals, frag_ids,
                              perturb_max=PERTURB_MAX, tau_mm=TAU_MM,
                              rng=None):
    """
    断面点法向扰动。阈值基于物理毫米阈值，直接使用 points_mm 毫米坐标。
    要求 points_mm 处于"未位移"状态。
    复杂度 O(N log N)，向量化实现。
    返回 (perturbed, boundary_mask, tau)。
    """
    if rng is None:
        rng = np.random

    perturbed = points_mm.copy()
    unique_ids = np.unique(frag_ids)
    boundary_mask = np.zeros(len(points_mm), dtype=bool)

    tau = tau_mm   # 直接用物理毫米阈值

    for i in unique_ids:
        mask_i = (frag_ids == i)
        pts_other = points_mm[~mask_i]
        if len(pts_other) == 0:
            continue
        tree = cKDTree(pts_other)
        dists, _ = tree.query(points_mm[mask_i])
        local_boundary = dists < tau    # 用固定 τ，不再 2*median(dists)
        global_idx = np.where(mask_i)[0][local_boundary]
        boundary_mask[global_idx] = True

    # 向量化单侧扰动（沿外法向，避免对穿）
    n_boundary = boundary_mask.sum()
    if n_boundary > 0:
        offsets = rng.uniform(0, perturb_max, size=(n_boundary, 1))
        perturbed[boundary_mask] += normals[boundary_mask] * offsets

    return perturbed, boundary_mask, tau


# ============================================================
# 7. 采样与随机位移
# ============================================================
def sample_fragments(fragments, degenerates):
    """
    对碎片进行面积加权采样，退化碎片的面积分配给最近的有效碎片。
    返回 (points_mm, normals, fragment_ids)。
    """
    if len(fragments) == 0:
        return None, None, None

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

    n_points = np.maximum(
        np.round(areas / total_area * N_TOTAL).astype(int),
        N_MIN_PER_FRAG)

    diff = n_points.sum() - N_TOTAL
    while diff != 0:
        if diff > 0:
            idx = np.argmax(n_points)
            if n_points[idx] > N_MIN_PER_FRAG:
                n_points[idx] -= 1
                diff -= 1
            else:
                break
        else:
            idx = np.argmax(n_points)
            n_points[idx] += 1
            diff += 1

    all_points = []
    all_normals = []
    all_frag_ids = []

    for fid, (frag, n_pts) in enumerate(zip(fragments, n_points)):
        if n_pts <= 0:
            continue
        pts_raw, face_idx = trimesh.sample.sample_surface(frag, n_pts)
        norms = frag.face_normals[face_idx]
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
    使用全局包围盒计算平移范围，防止小碎片飞出视野。
    返回 (displaced_points, transforms_4x4)。
    """
    displaced = np.zeros_like(points_mm)
    transforms_4x4 = np.zeros((n_frags, 4, 4), dtype=np.float32)

    global_bbox_diag = np.linalg.norm(
        points_mm.max(axis=0) - points_mm.min(axis=0))
    global_trans_range = min(global_bbox_diag * 0.15, TRANS_RANGE)

    for fid in range(n_frags):
        transforms_4x4[fid, 3, 3] = 1.0
        mask = frag_ids == fid
        pts = points_mm[mask]
        if len(pts) == 0:
            continue

        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(-ROT_RANGE, ROT_RANGE)
        angle_rad = np.deg2rad(angle)
        R = trimesh.transformations.rotation_matrix(angle_rad, axis)[:3, :3]

        t = np.random.uniform(-global_trans_range, global_trans_range, size=3)

        displaced[mask] = pts @ R.T + t
        transforms_4x4[fid, :3, :3] = R.astype(np.float32)
        transforms_4x4[fid, :3, 3] = t.astype(np.float32)

    return displaced, transforms_4x4


# ============================================================
# 8. matplotlib 设置
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
# 9. 可视化
# ============================================================
def plot_fracture(case_id, bone_name, bone_group, intact_points,
                  frac_points, frag_ids, n_frags, max_t,
                  boundary_pct, output_path):
    """破碎可视化，1×2 布局，标注断面点占比"""
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

    ax_right.set_title('Fractured (Voronoi)', fontsize=14)
    ax_right.set_xlabel('X (normalized)', fontsize=14)
    ax_right.set_ylabel('Z (normalized)', fontsize=14)
    ax_right.tick_params(labelsize=11)
    ax_right.set_aspect('equal')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.legend(fontsize=9, markerscale=3, loc='upper right')

    fig.suptitle(f'{bone_group} | {case_id}_{bone_name} | '
                 f'n_frags={n_frags} | max_t={max_t:.1f}mm | '
                 f'boundary={boundary_pct:.1f}%',
                 fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(case_id, bone_name, bone_group,
                    intact_pts, phase1_pts, phase2_pts,
                    phase1_fid, phase2_fid,
                    n_frags1, n_frags2, output_path):
    """阶段一 vs 阶段二对比图，1×2 布局"""
    setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- 左图：阶段一 --
    ax1 = axes[0]
    cmap1 = plt.get_cmap('tab10', n_frags1)
    for fid in range(n_frags1):
        mask = phase1_fid == fid
        pts = phase1_pts[mask]
        ax1.scatter(pts[:, 0], pts[:, 2],
                    c=[cmap1(fid)], s=1.0, alpha=0.6, rasterized=True)
    ax1.set_title('Phase 1: Plane Cut', fontsize=14)
    ax1.set_xlabel('X (normalized)', fontsize=14)
    ax1.set_ylabel('Z (normalized)', fontsize=14)
    ax1.tick_params(labelsize=11)
    ax1.set_aspect('equal')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # -- 右图：阶段二 --
    ax2 = axes[1]
    cmap2 = plt.get_cmap('tab10', n_frags2)
    for fid in range(n_frags2):
        mask = phase2_fid == fid
        pts = phase2_pts[mask]
        ax2.scatter(pts[:, 0], pts[:, 2],
                    c=[cmap2(fid)], s=1.0, alpha=0.6, rasterized=True)
    ax2.set_title('Phase 2: Voronoi', fontsize=14)
    ax2.set_xlabel('X (normalized)', fontsize=14)
    ax2.set_ylabel('Z (normalized)', fontsize=14)
    ax2.tick_params(labelsize=11)
    ax2.set_aspect('equal')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 统一坐标轴范围
    all_x = np.concatenate([phase1_pts[:, 0], phase2_pts[:, 0]])
    all_z = np.concatenate([phase1_pts[:, 2], phase2_pts[:, 2]])
    x_pad = (all_x.max() - all_x.min()) * 0.05
    z_pad = (all_z.max() - all_z.min()) * 0.05
    xlim = (all_x.min() - x_pad, all_x.max() + x_pad)
    zlim = (all_z.min() - z_pad, all_z.max() + z_pad)
    for ax in [ax1, ax2]:
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)

    fig.suptitle(f'Comparison | {bone_group}_{case_id}_{bone_name} | '
                 f'Frags: {n_frags1} vs {n_frags2}',
                 fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_interactive_html(case_id, bone_name, bone_group, intact_pts,
                               frac_pts, frag_ids, boundary_mask, n_frags,
                               max_t, boundary_pct, output_path):
    """用 plotly 生成交互式 3D HTML 可视化（全量 4096 点，不下采样）"""
    intact_pts_ds = intact_pts
    frac_pts_ds = frac_pts
    frag_ids_ds = frag_ids
    boundary_ds = boundary_mask

    tab10 = plt.get_cmap('tab10')

    trace_intact = go.Scatter3d(
        x=intact_pts_ds[:, 0], y=intact_pts_ds[:, 1], z=intact_pts_ds[:, 2],
        mode='markers',
        marker=dict(size=2.0, color='lightgray', opacity=0.5),
        name='Intact Bone')

    traces_frac = []
    for fid in range(n_frags):
        mask = frag_ids_ds == fid
        pts = frac_pts_ds[mask]
        rgba = tab10(fid / max(10, n_frags))
        hex_color = f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
        traces_frac.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=2.0, color=hex_color, opacity=0.7),
            name=f'Frag {fid}'))

    boundary_pts = frac_pts_ds[boundary_ds]
    if len(boundary_pts) > 0:
        traces_frac.append(go.Scatter3d(
            x=boundary_pts[:, 0], y=boundary_pts[:, 1], z=boundary_pts[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', symbol='x', opacity=1.0),
            name=f'Boundary ({boundary_pct:.1f}%)'))

    title_text = (f'{bone_group} | {case_id}_{bone_name} | '
                  f'n_frags={n_frags} | boundary={boundary_pct:.1f}% | '
                  f'max_t={max_t:.1f}mm')

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Intact Bone', 'Fractured (Voronoi)'))

    fig.add_trace(trace_intact, row=1, col=1)
    for tr in traces_frac:
        fig.add_trace(tr, row=1, col=2)

    axis_dict = dict(showbackground=True, showticklabels=False,
                     showgrid=False, zeroline=False,
                     backgroundcolor='rgba(240,240,240,0.3)')
    fig.update_scenes(dict(xaxis=axis_dict, yaxis=axis_dict, zaxis=axis_dict), row=1, col=1)
    fig.update_scenes(dict(xaxis=axis_dict, yaxis=axis_dict, zaxis=axis_dict), row=1, col=2)

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        width=1400, height=600, showlegend=True,
        legend=dict(font=dict(size=11), itemsizing='constant'),
        margin=dict(l=20, r=20, t=60, b=20))

    fig.write_html(output_path)
    print(f"    [HTML] {bone_group:25s} -> {os.path.basename(output_path)}")


# ============================================================
# 10. Voronoi 破碎主流程
# ============================================================
def voronoi_fracture_mesh(mesh, rng):
    """
    对网格执行 Voronoi 破碎：
    1. 生成种子点
    2. 面片归属分配
    3. 小碎片合并
    4. 采样
    5. 断面扰动
    6. 随机变换
    """
    n_seeds = int(rng.integers(2, 5))  # 2~4 块
    seeds = generate_voronoi_seeds(mesh, n_seeds, rng)

    # 面片归属分配（返回原始面索引）
    seed_faces, degenerates = assign_faces_to_seeds(mesh, seeds)

    if len(seed_faces) == 0:
        return None, None, None, None, None

    # 小碎片合并（在面索引层面操作）
    if len(seed_faces) > 1:
        seed_faces = merge_small_fragments(mesh, seed_faces)

    if len(seed_faces) == 0:
        return None, None, None, None, None

    # 从面索引重建碎片网格
    fragments, degenerates = build_fragments_from_faces(mesh, seed_faces)

    if len(fragments) == 0:
        return None, None, None, None, None

    return fragments, degenerates


# ============================================================
# 11. 主流程
# ============================================================
def main():
    os.makedirs(FRAC_DIR, exist_ok=True)
    os.makedirs(PHASE2_FIG_DIR, exist_ok=True)
    os.makedirs(PHASE2_HTML_DIR, exist_ok=True)
    setup_matplotlib()

    print("=" * 60)
    print("Step 3 — 模拟骨折破碎（阶段二：Voronoi 破碎）")
    print("=" * 60)

    # 1. 读取
    print("\n[1/4] Loading selected_cases.json...")
    grouped_entries, all_entries = load_selected(SELECTED_JSON)

    # 2. 断点续跑
    print("\n[2/4] Checking existing Voronoi .npz files...")
    existing = get_existing_voronoi()
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
    all_frag_counts = []
    max_displacements = []
    boundary_pcts = []

    for case_idx, (case_id, entries) in enumerate(grouped_entries.items(), 1):
        print(f"\n  Case {case_idx}/{len(grouped_entries)}: {case_id} ({len(entries)} bones)")

        for entry in entries:
            bone = entry['bone']
            bone_group = entry['bone_group']
            voronoi_name = f"{case_id}_{bone}_voronoi.npz"
            voronoi_path = os.path.join(FRAC_DIR, voronoi_name)

            if voronoi_name.replace('_voronoi.npz', '') in existing:
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

            # 加载 Step 2 .npz
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

                # 固定随机种子保证可复现
                rng = np.random.default_rng(abs(hash(f"{case_id}_{bone}")) % (2**31))

                # Voronoi 破碎
                fragments, degenerates = voronoi_fracture_mesh(mesh, rng)

                if fragments is None or len(fragments) == 0:
                    msg = f"  [FAIL] {bone}: no valid fragments after Voronoi"
                    print(f"    {msg}")
                    error_lines.append(f"{case_id}_{bone}: {msg}")
                    total_failed += 1
                    continue

                # 采样（在原始位置，未变换前）
                points_mm, normals, frag_ids = sample_fragments(fragments, degenerates)
                if points_mm is None:
                    msg = f"  [FAIL] {bone}: sampling failed"
                    print(f"    {msg}")
                    error_lines.append(f"{case_id}_{bone}: {msg}")
                    total_failed += 1
                    continue

                n_frags = len(np.unique(frag_ids))

                # 断面扰动（在原始位置，变换前）
                points_perturbed, boundary_mask, tau = perturb_fracture_surface(
                    points_mm, normals, frag_ids, perturb_max=PERTURB_MAX,
                    tau_mm=TAU_MM, rng=rng)

                boundary_pct = 100.0 * boundary_mask.sum() / len(points_perturbed)

                # 随机刚体变换
                displaced_mm, transforms_4x4 = apply_random_transform(
                    points_perturbed, frag_ids, n_frags)

                # 归一化（使用 Step 2 的 centroid 和 scale）
                points_norm = (displaced_mm - centroid) / scale

                max_t = max(np.linalg.norm(t) for t in transforms_4x4[:, :3, 3])
                max_displacements.append(max_t)
                all_frag_counts.append(n_frags)
                boundary_pcts.append(boundary_pct)

                # 保存 .npz
                np.savez(voronoi_path,
                         points=points_norm.astype(np.float32),
                         fragment_id=frag_ids.astype(np.int32),
                         transform=transforms_4x4.astype(np.float32),
                         normals=normals.astype(np.float32),
                         boundary_mask=boundary_mask.astype(bool),
                         centroid=centroid.astype(np.float32),
                         scale=np.float32(scale))

                total_success += 1
                print(f"    [OK] {bone:30s} -> {voronoi_name}  "
                      f"(frags={n_frags}, max_t={max_t:.1f}mm, "
                      f"boundary={boundary_pct:.1f}%, "
                      f"tau={tau:.4f}mm)")

            except Exception as e:
                import traceback
                tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                # 只取前 5 行
                tb_short = '\n'.join(tb_str.strip().split('\n')[:5])
                msg = f"  [FAIL] {bone}: {e}"
                print(f"    {msg}")
                error_lines.append(f"{case_id}_{bone} | {type(e).__name__} | {e} | {tb_short}")
                total_failed += 1

    elapsed = time.time() - start_t

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
        voronoi_path = os.path.join(FRAC_DIR, f"{rc}_{rb}_voronoi.npz")
        pc_path = os.path.join(PC_DIR, f"{rc}_{rb}.npz")

        if not os.path.exists(voronoi_path) or not os.path.exists(pc_path):
            print(f"    [SKIP] {bg}: data not found")
            continue

        try:
            v_data = np.load(voronoi_path, allow_pickle=True)
            pc_data = np.load(pc_path)

            intact_pts = pc_data['points']
            frac_pts = v_data['points']
            frag_ids = v_data['fragment_id']
            transforms_data = v_data['transform']
            boundary_mask = v_data['boundary_mask']
            n_frags = len(np.unique(frag_ids))
            max_t = max(np.linalg.norm(t) for t in transforms_data[:, :3, 3])
            boundary_pct = 100.0 * boundary_mask.sum() / len(frac_pts)

            out_path = os.path.join(PHASE2_FIG_DIR,
                                    f'{bg}_{rc}_{rb}_voronoi.png')
            plot_fracture(rc, rb, bg, intact_pts, frac_pts,
                          frag_ids, n_frags, max_t, boundary_pct, out_path)
            vis_count += 1
            print(f"    [OK] {bg:25s} -> {os.path.basename(out_path)}")

        except Exception as e:
            print(f"    [FAIL] {bg}: {e}")

    # 生成对比图（阶段一 vs 阶段二），至少 3 个分组
    print("\n  Generating comparison figures (Phase 1 vs Phase 2)...")
    comparison_groups = ['femur', 'pelvis', 'vertebrae_lumbar']
    comp_count = 0
    for bg in comparison_groups:
        if bg not in rep_map:
            continue
        rep = rep_map[bg]
        rc = rep['case']
        rb = rep['bone']

        frac_path = os.path.join(FRAC_DIR, f"{rc}_{rb}_frac.npz")
        voronoi_path = os.path.join(FRAC_DIR, f"{rc}_{rb}_voronoi.npz")
        pc_path = os.path.join(PC_DIR, f"{rc}_{rb}.npz")

        if not (os.path.exists(frac_path) and os.path.exists(voronoi_path)
                and os.path.exists(pc_path)):
            print(f"    [SKIP] {bg}: comparison data not found")
            continue

        try:
            pc_data = np.load(pc_path)
            f1 = np.load(frac_path, allow_pickle=True)
            f2 = np.load(voronoi_path, allow_pickle=True)

            intact_pts = pc_data['points']
            p1_pts = f1['points']
            p1_fid = f1['fragment_id']
            p2_pts = f2['points']
            p2_fid = f2['fragment_id']
            n1 = len(np.unique(p1_fid))
            n2 = len(np.unique(p2_fid))

            comp_path = os.path.join(PHASE2_FIG_DIR,
                                     f'comparison_{bg}_{rc}_{rb}.png')
            plot_comparison(rc, rb, bg, intact_pts,
                            p1_pts, p2_pts, p1_fid, p2_fid, n1, n2,
                            comp_path)
            comp_count += 1
            print(f"    [OK] {bg:25s} -> comparison saved")

        except Exception as e:
            print(f"    [FAIL] {bg}: comparison failed: {e}")

    # 生成交互式 HTML
    print("\n  Generating interactive 3D HTML visualizations...")
    os.makedirs(PHASE2_HTML_DIR, exist_ok=True)
    html_count = 0
    for bg, rep in sorted(rep_map.items()):
        rc = rep['case']
        rb = rep['bone']
        voronoi_path = os.path.join(FRAC_DIR, f"{rc}_{rb}_voronoi.npz")
        pc_path = os.path.join(PC_DIR, f"{rc}_{rb}.npz")

        if not os.path.exists(voronoi_path) or not os.path.exists(pc_path):
            continue

        try:
            v_data = np.load(voronoi_path, allow_pickle=True)
            pc_data = np.load(pc_path)

            intact_pts = pc_data['points']
            frac_pts = v_data['points']
            frag_ids = v_data['fragment_id']
            transforms_data = v_data['transform']
            boundary_mask = v_data['boundary_mask']
            n_frags = len(np.unique(frag_ids))
            max_t = max(np.linalg.norm(t) for t in transforms_data[:, :3, 3])
            boundary_pct = 100.0 * boundary_mask.sum() / len(frac_pts)

            out_path = os.path.join(PHASE2_HTML_DIR, f'{bg}_{rc}_{rb}_interactive.html')
            generate_interactive_html(rc, rb, bg, intact_pts, frac_pts,
                                      frag_ids, boundary_mask, n_frags,
                                      max_t, boundary_pct, out_path)
            html_count += 1

        except Exception as e:
            print(f"    [FAIL] {bg} HTML: {e}")

    # 5. 统计
    print("\n" + "=" * 60)
    print("Step 3 Phase 2 — Summary")
    print("=" * 60)
    lines = []
    lines.append(f"Total requested:  {total_requested}")
    lines.append(f"Success:          {total_success}")
    lines.append(f"Skipped (resume): {total_skipped}")
    lines.append(f"Failed:           {total_failed}")
    if all_frag_counts:
        lines.append(f"Avg fragments:    {np.mean(all_frag_counts):.2f}")
    if max_displacements:
        lines.append(f"Max displacement: {np.mean(max_displacements):.1f}mm (avg)")
    if boundary_pcts:
        lines.append(f"Boundary points:  {np.mean(boundary_pcts):.1f}% (avg)")
    lines.append(f"Visualizations:   {vis_count}/{len(rep_map)} groups")
    lines.append(f"Comparisons:      {comp_count}/{len(comparison_groups)} groups")
    lines.append(f"Interactive HTML: {html_count}/{len(rep_map)} groups")
    lines.append(f"Total time:       {elapsed:.1f}s")

    for l in lines:
        print(f"  {l}")
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  [INFO] Summary saved: {SUMMARY_TXT}")


if __name__ == '__main__':
    main()
