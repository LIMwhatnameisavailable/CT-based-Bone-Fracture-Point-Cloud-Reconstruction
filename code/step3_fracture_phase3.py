#!/usr/bin/env python
"""
Step 3 — 模拟骨折破碎（阶段三：模态分析物理破碎）
对每骨骼分组各选 2 例（共 24 例），使用有限元模态分析确定裂纹方向，
替代 Phase 2 的随机 Voronoi 种子。其余后处理逻辑（随机位移、boundary_mask
计算、可视化）与 Phase 2 保持一致。

核心算法：
  1. 用 robust-laplacian 构建刚度矩阵 K 和质量矩阵 M
  2. 求解广义特征值问题 K u = λ M u（shift-invert 模式）
  3. 第一非零特征向量 u₁（Fiedler 向量）的零交叉位置 → 裂纹平面
  4. 单平面切割 → 2 块碎片 → 采样 → 断面高斯噪声 → 随机位移 → 归一化
"""
import os, json, sys, time
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
from robust_laplacian import mesh_laplacian

# ============================================================
# 0. 路径与配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHES_DIR = os.path.join(PROJECT_ROOT, 'data', 'meshes')
PC_DIR = os.path.join(PROJECT_ROOT, 'data', 'pointclouds')
FRAC_DIR = os.path.join(PROJECT_ROOT, 'data', 'fractured')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results', 'step3_fracture')
PHASE3_FIG_DIR = os.path.join(FIG_DIR, 'phase3')
SELECTED_JSON = os.path.join(PROJECT_ROOT, 'results', 'step0_dataset_overview', 'selected_cases.json')
SUMMARY_TXT = os.path.join(FIG_DIR, 'step3_phase3_summary.txt')

os.makedirs(PHASE3_FIG_DIR, exist_ok=True)

N_TOTAL = 4096
N_MIN_PER_FRAG = 30
FRAG_VERTEX_MIN = 50
ROT_RANGE = 30
TRANS_RANGE = 20.0
TAU_MM = 2.5             # 物理空间断面判定阈值 (mm)，平面切割断面紧密，阈值可小于 Voronoi
GAUSS_STD = 0.5         # 断面高斯噪声标准差 (mm)，不同于 Phase 2 的 uniform [0, 0.3]
MERGE_VOL_RATIO = 0.05
N_PER_GROUP = 2         # 每骨骼分组选 2 例

# ============================================================
# 1. 加载与样本选择
# ============================================================
def load_selected(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    grouped = {}
    for e in entries:
        bg = e['bone_group']
        if bg not in grouped:
            grouped[bg] = []
        grouped[bg].append(e)
    print(f"  Loaded {len(entries)} entries, {len(grouped)} bone groups")
    return grouped, entries


def select_top_per_group(grouped, n_per_group=N_PER_GROUP):
    """每骨骼分组取体积最大的 n_per_group 例"""
    selected = []
    groups_used = []
    for bg, items in sorted(grouped.items()):
        items_sorted = sorted(items, key=lambda x: x['volume_cm3'], reverse=True)
        chosen = items_sorted[:n_per_group]
        selected.extend(chosen)
        groups_used.append(bg)
        vols = [c['volume_cm3'] for c in chosen]
        cases = [f"{c['case']}_{c['bone']}" for c in chosen]
        print(f"    {bg:25s}: {vols} cm^3  ({', '.join(cases)})")
    print(f"  Total selected: {len(selected)} entries, {len(groups_used)} groups")
    return selected


def get_existing_physical():
    """扫描 FRAC_DIR 已有 _physical.npz，返回 {case_bone} 集合"""
    existing = set()
    if not os.path.isdir(FRAC_DIR):
        return existing
    for f in os.listdir(FRAC_DIR):
        if f.endswith('_physical.npz'):
            existing.add(f.replace('_physical.npz', ''))
    return existing


# ============================================================
# 2. 模态分析 → 裂纹平面
# ============================================================
def modal_analysis(mesh, n_ev=8):
    """
    模态分析：用 robust-laplacian 构建标量 Laplacian L 和质量矩阵 M，
    求解广义特征值问题 L u = λ M u，返回多个特征向量。

    使用标量 Laplacian（而非块对角 3× 扩展），因为：
    - 表面网格的 Laplace-Beltrami 算子本身就能刻画振动模态
    - 不同特征向量的零交叉面给出不同方向的切割
    - 后续通过择优选取得到物理合理的裂纹方向
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    L, M = mesh_laplacian(verts, faces)

    eigenvalues, eigenvectors = eigsh(
        L, k=n_ev, M=M, sigma=1e-6, which='LM',
        tol=1e-8, maxiter=max(10000, 100 * len(verts)))

    return eigenvalues, eigenvectors


def find_zero_crossing_points(verts, faces, scalar_field):
    """
    对标量场检测零交叉边，返回插值后的零交叉点集。
    返回 (midpoints, n_crossings)。
    """
    midpoints = []
    processed_edges = set()

    for face in faces:
        for i in range(3):
            v1, v2 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v1, v2), max(v1, v2))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)

            s1, s2 = scalar_field[v1], scalar_field[v2]
            if s1 * s2 < 0:
                t = s1 / (s1 - s2)
                midpoint = verts[v1] + t * (verts[v2] - verts[v1])
                midpoints.append(midpoint)

    return np.array(midpoints) if midpoints else np.empty((0, 3))


def fit_plane_to_points(points, bone_center):
    """PCA 拟合平面，返回 (plane_origin, plane_normal)"""
    center = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - center, full_matrices=False)
    normal = Vt[2]  # 最小主成分 = 平面法向量
    if np.dot(normal, center - bone_center) < 0:
        normal = -normal
    return center, normal


def score_crack_plane(verts, plane_origin, plane_normal, zc_points):
    """
    评价裂纹平面的质量：
    - balance: 顶点分割均衡程度 (0~1)
    - centeredness: 平面居中程度 (0~1)
    - planarity: 零交叉点云的平面性 — 抑制纵劈（线状零交叉），奖励横断（环状零交叉）

    纵劈时零交叉点沿一条线分布（PCA s₂ << s₁），
    横断时零交叉点形成一个环（PCA s₁ ≈ s₂）。
    """
    center = verts.mean(axis=0)
    proj = (verts - center) @ plane_normal

    n_pos = (proj > 0).sum()
    n_neg = (proj < 0).sum()
    balance = min(n_pos, n_neg) / max(max(n_pos, n_neg), 1)

    proj_range = np.percentile(proj, 95) - np.percentile(proj, 5)
    plane_offset = (plane_origin - center) @ plane_normal
    centeredness = 1.0 - min(abs(plane_offset) / (proj_range * 0.5 + 1e-8), 1.0)

    # 零交叉点云的平面性：s₂/s₁（第二大/最大的奇异值比）
    # 线状=0, 圆环→1, 抑制纵劈奖励横断
    planarity = 0.5
    if len(zc_points) >= 6:
        zc_c = zc_points - zc_points.mean(axis=0)
        _, s, _ = np.linalg.svd(zc_c, full_matrices=False)
        s = s[:3]
        planarity = s[1] / (s[0] + 1e-10)
        planarity = np.clip(planarity, 0.0, 1.0)

    return 0.35 * balance + 0.35 * centeredness + 0.30 * planarity


def extract_crack_plane(mesh, eigenvalues, eigenvectors):
    """
    从多个特征向量中择优提取裂纹平面。

    对每个特征向量：
    1. 用该向量作为标量场，检测零交叉
    2. 零交叉点拟合平面
    3. 评价平面质量（balance + centeredness）
    4. 约束位置在质心 ±20% 范围内
    选择综合评分最高的平面。
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    bone_center = verts.mean(axis=0)

    best_score = -1.0
    best_origin = None
    best_normal = None
    best_ev_idx = -1

    # 跳过第 0 个（零特征值刚体模态），尝试 u₁~u₅
    for ev_idx in range(1, min(6, eigenvectors.shape[1])):
        u = eigenvectors[:, ev_idx]

        # 零交叉检测
        midpoints = find_zero_crossing_points(verts, faces, u)
        if len(midpoints) < 3:
            continue

        # 拟合平面
        origin, normal = fit_plane_to_points(midpoints, bone_center)

        # 约束平面位置在骨骼质心 ±20% 范围内
        proj_all = (verts - bone_center) @ normal
        proj_range = np.percentile(proj_all, 80) - np.percentile(proj_all, 20)
        allowed_max = 0.20 * proj_range

        offset = (origin - bone_center) @ normal
        offset = np.clip(offset, -allowed_max, allowed_max)
        origin = bone_center + normal * offset

        # 评价
        score = score_crack_plane(verts, origin, normal, midpoints)
        if score > best_score:
            best_score = score
            best_origin = origin
            best_normal = normal
            best_ev_idx = ev_idx

    # 回退：用 PCA 主轴切割（基本上就是横向切）
    if best_origin is None:
        _, _, Vt = np.linalg.svd(verts - bone_center, full_matrices=False)
        best_normal = Vt[0]  # 沿主轴方向切割
        proj_all = (verts - bone_center) @ best_normal
        offset = np.clip(0.0,
                         -0.2 * (np.percentile(proj_all, 80) - np.percentile(proj_all, 20)),
                         0.2 * (np.percentile(proj_all, 80) - np.percentile(proj_all, 20)))
        best_origin = bone_center + best_normal * offset
        best_ev_idx = -1

    return best_origin, best_normal, best_ev_idx, best_score


# ============================================================
# 3. 网格切割
# ============================================================
def cut_mesh_by_plane(mesh, plane_origin, plane_normal):
    """用平面切割网格，返回 above 和 below 两部分（断面自动封口）"""
    above = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal, plane_origin, cap=True)
    below = trimesh.intersections.slice_mesh_plane(
        mesh, -plane_normal, plane_origin, cap=True)
    return above, below


def clean_fragment(sub_mesh):
    """保留最大连通分量，检查是否退化"""
    if sub_mesh is None or len(sub_mesh.vertices) == 0:
        return None
    try:
        components = sub_mesh.split(only_watertight=False)
        if components:
            sub_mesh = max(components, key=lambda m: len(m.vertices))
    except Exception:
        pass
    if len(sub_mesh.vertices) < FRAG_VERTEX_MIN:
        return None
    return sub_mesh


# ============================================================
# 4. 采样与随机位移（与 Phase 2 一致）
# ============================================================
def sample_fragments(fragments, degenerates):
    """对碎片进行面积加权采样，返回 (points_mm, normals, frag_ids)"""
    if len(fragments) == 0:
        return None, None, None

    areas = np.array([f.area for f in fragments])
    total_area = areas.sum()

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

    all_points, all_normals, all_frag_ids = [], [], []
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

    return (np.concatenate(all_points, axis=0),
            np.concatenate(all_normals, axis=0),
            np.concatenate(all_frag_ids, axis=0))


def perturb_fracture_surface(points_mm, normals, frag_ids,
                             gauss_std=GAUSS_STD, tau_mm=TAU_MM, rng=None):
    """
    断面点高斯噪声扰动。与 Phase 2 的关键区别：
    - 使用高斯噪声 (std=gauss_std) 替代 uniform [0, perturb_max]
    - 噪声沿法向量双向施加（高斯噪声本身是对称的）
    """
    if rng is None:
        rng = np.random

    perturbed = points_mm.copy()
    unique_ids = np.unique(frag_ids)
    boundary_mask = np.zeros(len(points_mm), dtype=bool)
    tau = tau_mm

    for i in unique_ids:
        mask_i = (frag_ids == i)
        pts_other = points_mm[~mask_i]
        if len(pts_other) == 0:
            continue
        tree = cKDTree(pts_other)
        dists, _ = tree.query(points_mm[mask_i])
        local_boundary = dists < tau
        global_idx = np.where(mask_i)[0][local_boundary]
        boundary_mask[global_idx] = True

    n_boundary = boundary_mask.sum()
    if n_boundary > 0:
        offsets = rng.normal(0, gauss_std, size=(n_boundary, 1))
        perturbed[boundary_mask] += normals[boundary_mask] * offsets

    return perturbed, boundary_mask, tau


def apply_random_transform(points_mm, frag_ids, n_frags):
    """对每块碎片施加随机旋转和平移（全局包围盒自适应范围）"""
    displaced = np.zeros_like(points_mm)
    transforms_4x4 = np.zeros((n_frags, 4, 4), dtype=np.float32)

    global_bbox_diag = np.linalg.norm(
        points_mm.max(axis=0) - points_mm.min(axis=0))
    trans_range = min(global_bbox_diag * 0.15, TRANS_RANGE)

    for fid in range(n_frags):
        transforms_4x4[fid, 3, 3] = 1.0
        mask = frag_ids == fid
        pts = points_mm[mask]
        if len(pts) == 0:
            continue

        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(-ROT_RANGE, ROT_RANGE)
        R = trimesh.transformations.rotation_matrix(
            np.deg2rad(angle), axis)[:3, :3]
        t = np.random.uniform(-trans_range, trans_range, size=3)

        displaced[mask] = pts @ R.T + t
        transforms_4x4[fid, :3, :3] = R.astype(np.float32)
        transforms_4x4[fid, :3, 3] = t.astype(np.float32)

    return displaced, transforms_4x4


# ============================================================
# 5. 裂纹方向角度计算
# ============================================================
def compute_crack_angle(mesh, plane_normal):
    """计算裂纹平面法向量与骨骼主轴（PCA 第一主成分）的夹角"""
    verts = mesh.vertices
    center = verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(verts - center, full_matrices=False)
    main_axis = Vt[0]
    cos_angle = np.abs(np.dot(plane_normal, main_axis))
    cos_angle = np.clip(cos_angle, 0.0, 1.0)
    angle_with_axis = np.degrees(np.arccos(cos_angle))
    # 与横截面的夹角 = 90° - 与主轴夹角
    angle_with_cross = 90.0 - angle_with_axis
    return angle_with_axis, angle_with_cross


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
                  frac_points, frag_ids, n_frags, max_t,
                  boundary_pct, crack_angle, output_path):
    """破碎可视化，1×2 布局，标注裂纹方向角度"""
    setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：完整点云
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

    # 右图：破碎后碎片
    ax_right = axes[1]
    cmap = plt.get_cmap('tab10', n_frags)
    for fid in range(n_frags):
        mask = frag_ids == fid
        pts = frac_points[mask]
        ax_right.scatter(pts[:, 0], pts[:, 2],
                         c=[cmap(fid)], s=1.0, alpha=0.6, rasterized=True,
                         label=f'Frag {fid}')

    ax_right.set_title('Fractured (Modal)', fontsize=14)
    ax_right.set_xlabel('X (normalized)', fontsize=14)
    ax_right.set_ylabel('Z (normalized)', fontsize=14)
    ax_right.tick_params(labelsize=11)
    ax_right.set_aspect('equal')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.legend(fontsize=9, markerscale=3, loc='upper right')

    fig.suptitle(f'{bone_group} | {case_id}_{bone_name} | '
                 f'frags={n_frags} | max_t={max_t:.1f}mm | '
                 f'boundary={boundary_pct:.1f}% | '
                 f'crack_angle={crack_angle:.1f}°',
                 fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(case_id, bone_name, bone_group,
                    intact_pts, voronoi_pts, modal_pts,
                    voronoi_fid, modal_fid,
                    n_frags_v, n_frags_m, output_path):
    """Phase 2 (Voronoi) vs Phase 3 (Modal) 对比图"""
    setup_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Phase 2 Voronoi
    ax1 = axes[0]
    cmap1 = plt.get_cmap('tab10', n_frags_v)
    for fid in range(n_frags_v):
        mask = voronoi_fid == fid
        pts = voronoi_pts[mask]
        ax1.scatter(pts[:, 0], pts[:, 2],
                    c=[cmap1(fid)], s=1.0, alpha=0.6, rasterized=True)
    ax1.set_title('Phase 2: Voronoi', fontsize=14)
    ax1.set_xlabel('X (normalized)', fontsize=14)
    ax1.set_ylabel('Z (normalized)', fontsize=14)
    ax1.tick_params(labelsize=11)
    ax1.set_aspect('equal')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 右图：Phase 3 Modal
    ax2 = axes[1]
    cmap2 = plt.get_cmap('tab10', n_frags_m)
    for fid in range(n_frags_m):
        mask = modal_fid == fid
        pts = modal_pts[mask]
        ax2.scatter(pts[:, 0], pts[:, 2],
                    c=[cmap2(fid)], s=1.0, alpha=0.6, rasterized=True)
    ax2.set_title('Phase 3: Modal Analysis', fontsize=14)
    ax2.set_xlabel('X (normalized)', fontsize=14)
    ax2.set_ylabel('Z (normalized)', fontsize=14)
    ax2.tick_params(labelsize=11)
    ax2.set_aspect('equal')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 统一坐标轴范围
    all_x = np.concatenate([voronoi_pts[:, 0], modal_pts[:, 0]])
    all_z = np.concatenate([voronoi_pts[:, 2], modal_pts[:, 2]])
    x_pad = (all_x.max() - all_x.min()) * 0.05
    z_pad = (all_z.max() - all_z.min()) * 0.05
    xlim = (all_x.min() - x_pad, all_x.max() + x_pad)
    zlim = (all_z.min() - z_pad, all_z.max() + z_pad)
    for ax in [ax1, ax2]:
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)

    fig.suptitle(f'Comparison | {bone_group}_{case_id}_{bone_name} | '
                 f'Voronoi ({n_frags_v} frags) vs Modal ({n_frags_m} frags)',
                 fontsize=16, y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 8. 主流程
# ============================================================
def main():
    os.makedirs(FRAC_DIR, exist_ok=True)
    os.makedirs(PHASE3_FIG_DIR, exist_ok=True)
    setup_matplotlib()

    print("=" * 60)
    print("Step 3 — 模拟骨折破碎（阶段三：模态分析物理破碎）")
    print("=" * 60)

    # 1. 加载并筛选样本
    print("\n[1/5] Loading and selecting samples...")
    grouped_entries, all_entries = load_selected(SELECTED_JSON)
    selected = select_top_per_group(grouped_entries, n_per_group=N_PER_GROUP)
    total_requested = len(selected)

    # 2. 断点续跑
    print("\n[2/5] Checking existing _physical.npz files...")
    existing = get_existing_physical()
    total_skipped = 0
    total_success = 0
    total_failed = 0
    error_lines = []

    # 3. 模态分析破碎
    print(f"\n[3/5] Modal fracture of {total_requested} entries...")
    start_t = time.time()
    all_frag_counts = []
    max_displacements = []
    boundary_pcts = []
    crack_angles = []       # 裂纹方向与主轴夹角
    crack_angles_cross = [] # 裂纹与横截面夹角
    angle_by_group = {}     # 按骨骼分组统计

    for entry in selected:
        case_id = entry['case']
        bone = entry['bone']
        bone_group = entry['bone_group']
        physical_name = f"{case_id}_{bone}_physical.npz"
        physical_path = os.path.join(FRAC_DIR, physical_name)

        if physical_name.replace('_physical.npz', '') in existing:
            total_skipped += 1
            # 加载已有结果用于角度统计
            try:
                data = np.load(physical_path, allow_pickle=True)
                if 'crack_angle' in data:
                    ca = float(data['crack_angle'])
                    crack_angles.append(ca)
                    crack_angles_cross.append(90.0 - ca)
                    if bone_group not in angle_by_group:
                        angle_by_group[bone_group] = []
                    angle_by_group[bone_group].append(ca)
            except Exception:
                pass
            continue

        ply_path = os.path.join(MESHES_DIR, f"{case_id}_{bone}.ply")
        if not os.path.exists(ply_path):
            msg = f"  [SKIP] {bone}: .ply not found"
            print(f"    {msg}")
            error_lines.append(f"{case_id}_{bone}: {msg}")
            total_skipped += 1
            continue

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

            # 模态分析（标量 Laplacian，多特征向量）
            eigenvalues, eigenvectors = modal_analysis(mesh, n_ev=8)

            # 择优提取裂纹平面
            plane_origin, plane_normal, best_ev, plane_score = \
                extract_crack_plane(mesh, eigenvalues, eigenvectors)

            # 计算裂纹方向与主轴夹角
            angle_axis, angle_cross = compute_crack_angle(mesh, plane_normal)

            # 单平面切割 → 2 块碎片
            above, below = cut_mesh_by_plane(mesh, plane_origin, plane_normal)

            fragments = []
            degenerates = []
            for sub in [above, below]:
                cleaned = clean_fragment(sub)
                if cleaned is None:
                    degenerates.append((sub if sub else trimesh.Trimesh(), np.zeros(3)))
                else:
                    fragments.append(cleaned)

            if len(fragments) < 2:
                msg = (f"  [FAIL] {bone}: only {len(fragments)} valid "
                       f"fragment(s) after cut")
                print(f"    {msg}")
                error_lines.append(f"{case_id}_{bone}: {msg}")
                total_failed += 1
                continue

            n_frags = 2

            # 采样
            points_mm, normals, frag_ids = sample_fragments(fragments, degenerates)
            if points_mm is None:
                msg = f"  [FAIL] {bone}: sampling failed"
                print(f"    {msg}")
                error_lines.append(f"{case_id}_{bone}: {msg}")
                total_failed += 1
                continue

            rng = np.random.default_rng(
                abs(hash(f"phase3_{case_id}_{bone}")) % (2**31))

            # 断面高斯噪声扰动（在原始位置，变换前）
            points_perturbed, boundary_mask, tau = perturb_fracture_surface(
                points_mm, normals, frag_ids, gauss_std=GAUSS_STD,
                tau_mm=TAU_MM, rng=rng)

            boundary_pct = 100.0 * boundary_mask.sum() / len(points_perturbed)

            # 随机刚体变换
            displaced_mm, transforms_4x4 = apply_random_transform(
                points_perturbed, frag_ids, n_frags)

            # 归一化
            points_norm = (displaced_mm - centroid) / scale

            max_t = max(np.linalg.norm(t) for t in transforms_4x4[:, :3, 3])
            max_displacements.append(max_t)
            all_frag_counts.append(n_frags)
            boundary_pcts.append(boundary_pct)
            crack_angles.append(angle_axis)
            crack_angles_cross.append(angle_cross)
            if bone_group not in angle_by_group:
                angle_by_group[bone_group] = []
            angle_by_group[bone_group].append(angle_axis)

            # 保存 .npz（字段与 _voronoi.npz 一致，额外保存 crack_angle）
            np.savez(physical_path,
                     points=points_norm.astype(np.float32),
                     fragment_id=frag_ids.astype(np.int32),
                     transform=transforms_4x4.astype(np.float32),
                     normals=normals.astype(np.float32),
                     boundary_mask=boundary_mask.astype(bool),
                     centroid=centroid.astype(np.float32),
                     scale=np.float32(scale),
                     crack_angle=np.float32(angle_axis))

            total_success += 1
            print(f"    [OK] {bone:30s} -> {physical_name}  "
                  f"(ev={best_ev}, score={plane_score:.2f}, "
                  f"max_t={max_t:.1f}mm, "
                  f"boundary={boundary_pct:.1f}%, "
                  f"angle={angle_axis:.1f}°)")

        except Exception as e:
            import traceback
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            tb_short = '\n'.join(tb_str.strip().split('\n')[:5])
            msg = f"  [FAIL] {bone}: {e}"
            print(f"    {msg}")
            error_lines.append(
                f"{case_id}_{bone} | {type(e).__name__} | {e} | {tb_short}")
            total_failed += 1

    elapsed = time.time() - start_t

    # 保存 error log
    if error_lines:
        unique_errors = list(dict.fromkeys(error_lines))
        err_path = os.path.join(PROJECT_ROOT, 'code', 'step3_phase3_errors.txt')
        with open(err_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_errors))
        print(f"\n  [INFO] Error log saved: {err_path} ({len(unique_errors)} entries)")

    # 4. 可视化
    print("\n[4/5] Generating visualizations...")

    vis_count = 0
    comp_count = 0
    for entry in selected:
        case_id = entry['case']
        bone = entry['bone']
        bone_group = entry['bone_group']
        physical_path = os.path.join(FRAC_DIR, f"{case_id}_{bone}_physical.npz")
        pc_path = os.path.join(PC_DIR, f"{case_id}_{bone}.npz")

        if not os.path.exists(physical_path) or not os.path.exists(pc_path):
            continue

        try:
            p_data = np.load(physical_path, allow_pickle=True)
            pc_data = np.load(pc_path)

            intact_pts = pc_data['points']
            frac_pts = p_data['points']
            frag_ids = p_data['fragment_id']
            transforms_data = p_data['transform']
            boundary_mask = p_data['boundary_mask']
            n_frags = len(np.unique(frag_ids))
            max_t = max(np.linalg.norm(t) for t in transforms_data[:, :3, 3])
            boundary_pct = 100.0 * boundary_mask.sum() / len(frac_pts)
            crack_angle = float(p_data.get('crack_angle', 0))

            out_path = os.path.join(PHASE3_FIG_DIR,
                                    f'{bone_group}_{case_id}_{bone}_physical.png')
            plot_fracture(case_id, bone, bone_group, intact_pts, frac_pts,
                          frag_ids, n_frags, max_t, boundary_pct,
                          crack_angle, out_path)
            vis_count += 1
        except Exception as e:
            print(f"    [FAIL] {bone_group} vis: {e}")

    # Phase 2 vs Phase 3 对比图
    print("\n  Generating comparison figures (Voronoi vs Modal)...")
    for entry in selected:
        case_id = entry['case']
        bone = entry['bone']
        bone_group = entry['bone_group']

        voronoi_path = os.path.join(FRAC_DIR, f"{case_id}_{bone}_voronoi.npz")
        physical_path = os.path.join(FRAC_DIR, f"{case_id}_{bone}_physical.npz")
        pc_path = os.path.join(PC_DIR, f"{case_id}_{bone}.npz")

        if not (os.path.exists(voronoi_path) and os.path.exists(physical_path)
                and os.path.exists(pc_path)):
            continue

        try:
            pc_data = np.load(pc_path)
            v_data = np.load(voronoi_path, allow_pickle=True)
            p_data = np.load(physical_path, allow_pickle=True)

            intact_pts = pc_data['points']
            v_pts = v_data['points']
            v_fid = v_data['fragment_id']
            p_pts = p_data['points']
            p_fid = p_data['fragment_id']
            n_v = len(np.unique(v_fid))
            n_p = len(np.unique(p_fid))

            comp_path = os.path.join(PHASE3_FIG_DIR,
                                     f'comparison_{bone_group}_{case_id}_{bone}.png')
            plot_comparison(case_id, bone, bone_group, intact_pts,
                            v_pts, p_pts, v_fid, p_fid, n_v, n_p, comp_path)
            comp_count += 1
        except Exception as e:
            print(f"    [FAIL] {bone_group} comparison: {e}")

    # 5. 统计报告
    print("\n" + "=" * 60)
    print("Step 3 Phase 3 — Summary")
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
    if crack_angles:
        lines.append(f"Crack vs axis:    {np.mean(crack_angles):.1f}° (avg)")
        lines.append(f"Crack vs cross:   {np.mean(crack_angles_cross):.1f}° (avg)")

        # 按骨骼分组报告
        lines.append("\nCrack angles by bone group (angle vs main axis):")
        for bg in sorted(angle_by_group.keys()):
            angles = angle_by_group[bg]
            lines.append(f"  {bg:25s}: {np.mean(angles):.1f}° ± {np.std(angles):.1f}°")

        # 长骨检查：femur, humerus 应 < 45°
        long_bones = ['femur', 'humerus']
        for lb in long_bones:
            if lb in angle_by_group:
                mean_a = np.mean(angle_by_group[lb])
                status = "PASS" if mean_a < 45 else "FAIL"
                lines.append(f"\n  {lb} mean angle: {mean_a:.1f}° - {status} (< 45°)")

    lines.append(f"\nVisualizations:   {vis_count}/{len(selected)} entries")
    lines.append(f"Comparisons:      {comp_count}/{len(selected)} entries")
    lines.append(f"Total time:       {elapsed:.1f}s")

    for l in lines:
        print(f"  {l}")

    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  [INFO] Summary saved: {SUMMARY_TXT}")
    print("  [INFO] Phase 3 figures saved to:", PHASE3_FIG_DIR)


if __name__ == '__main__':
    main()
