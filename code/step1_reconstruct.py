#!/usr/bin/env python
"""
Step 1 — 骨骼三维重建
从 CT 原图 + mask 重建三角网格，输出 .ply 文件。
"""

import os, json, sys, time
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 0. 路径与配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
MESHES_DIR = os.path.join(PROJECT_ROOT, 'data', 'meshes')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results', 'step1_reconstruction')
SELECTED_JSON = os.path.join(PROJECT_ROOT, 'results', 'step0_dataset_overview', 'selected_cases.json')
ERROR_LOG = os.path.join(MESHES_DIR, 'error_log.txt')

os.makedirs(MESHES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PAD = 2  # bounding box padding (voxels)

# ============================================================
# 1. 加载选中列表
# ============================================================
def load_selected(json_path):
    """读取 selected_cases.json，按 case 分组"""
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # 按 case 分组
    grouped = {}
    for e in entries:
        c = e['case']
        if c not in grouped:
            grouped[c] = []
        grouped[c].append(e)

    # 按 case 排序
    grouped = dict(sorted(grouped.items()))
    print(f"  Loaded {len(entries)} entries, {len(grouped)} unique cases")
    return grouped, entries


# ============================================================
# 2. 断点续跑：检查已有 .ply
# ============================================================
def get_existing_plys():
    """扫描 meshes 目录已有 .ply，返回 {case_bone} 集合"""
    existing = set()
    if not os.path.isdir(MESHES_DIR):
        return existing
    for f in os.listdir(MESHES_DIR):
        if f.endswith('.ply'):
            existing.add(f.replace('.ply', ''))
    return existing


# ============================================================
# 3. 重建单个骨骼
# ============================================================
def reconstruct_bone(case_id, bone_name, ct_img, spacing, mask_spacing):
    """
    对单个骨骼执行 marching cubes 重建。
    返回 (vertices_mm, faces, mesh) 或抛出异常。
    """
    # 加载 mask
    seg_dir = os.path.join(RAW_DIR, case_id, 'segmentations')
    mask_path = os.path.join(seg_dir, f'{bone_name}.nii.gz')
    mask_img = nib.load(mask_path)
    mask_data = np.asanyarray(mask_img.dataobj)

    # spacing 一致性检查
    if not np.allclose(spacing, mask_spacing, atol=0.01):
        print(f"  [WARN] {case_id}/{bone_name}: CT spacing {spacing} != mask spacing {mask_spacing}")

    # 找出 mask 非零区域
    coords = np.argwhere(mask_data > 0)
    if len(coords) == 0:
        raise ValueError("Empty mask (zero non-voxel count)")

    min_coords = coords.min(axis=0)  # (z, y, x)
    max_coords = coords.max(axis=0)

    # Bounding box 裁剪 + padding
    crop_min = np.maximum(min_coords - PAD, 0)
    crop_max = np.minimum(max_coords + PAD + 1, np.array(mask_data.shape))
    cropped = mask_data[crop_min[0]:crop_max[0],
                        crop_min[1]:crop_max[1],
                        crop_min[2]:crop_max[2]]

    # Marching Cubes
    verts, faces, _, _ = marching_cubes(cropped, level=0.5)

    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("Marching cubes produced empty mesh")

    # 坐标恢复：cropped 坐标 → 原始体素坐标 → 毫米
    verts_full_voxel = verts + crop_min  # crop_min = min_coords - pad
    verts_mm = verts_full_voxel * spacing

    # 构建 trimesh
    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces)

    # 保留最大连通分量
    split_meshes = mesh.split(only_watertight=False)
    if len(split_meshes) == 0:
        raise ValueError("No connected components found")
    mesh = max(split_meshes, key=lambda m: len(m.vertices))

    if len(mesh.vertices) == 0:
        raise ValueError("Largest component has zero vertices")

    # 网格修复：补洞、修复法向、移除退化三角面
    mesh = repair_mesh(mesh)

    return mesh


# ============================================================
# 4. 网格修复
# ============================================================
def repair_mesh(mesh):
    """修复网格：补洞、修复法向、移除退化面"""
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    mask = mesh.area_faces > 1e-10
    mesh.update_faces(mask)
    return mesh


# ============================================================
# 5. 可视化
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


def plot_reconstruction(case_id, bone_name, bone_group, ct_data, ct_affine,
                         mask_data, mesh, output_path):
    """
    生成重建可视化，左图 = CT 切片 + mask 轮廓，右图 = 点云投影三视图。
    """
    setup_matplotlib()

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2])

    # -- 左图：CT 轴向中间切片 + mask 轮廓 --
    ax_ct = fig.add_subplot(gs[0, 0])

    # 找出 mask 非零区域的中间轴向切片
    coords = np.argwhere(mask_data > 0)
    z_min, z_max = coords[:, 0].min(), coords[:, 0].max()
    z_mid = (z_min + z_max) // 2

    # CT 切片（轴向）
    ct_slice = np.asanyarray(ct_data.dataobj)[z_mid, :, :]
    mask_slice = mask_data[z_mid, :, :]

    ax_ct.imshow(ct_slice, cmap='gray', aspect='auto')
    ax_ct.contour(mask_slice, levels=[0.5], colors='#E64B35', linewidths=1.2)
    ax_ct.set_title(f'{bone_group}\n{case_id}_{bone_name}\nSlice {z_mid}', fontsize=16)
    ax_ct.set_xlabel('X (pixels)', fontsize=14)
    ax_ct.set_ylabel('Y (pixels)', fontsize=14)
    ax_ct.tick_params(labelsize=11)
    ax_ct.spines['top'].set_visible(False)
    ax_ct.spines['right'].set_visible(False)

    # -- 右图：三视图（2D 散点投影） --
    gs_right = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1],
                                                 wspace=0.35)
    verts = mesh.vertices
    bbox = mesh.bounding_box.extents  # (L, W, H) in mm
    bbox_label = f'{bbox[0]:.0f} x {bbox[1]:.0f} x {bbox[2]:.0f} mm'

    # 动态散点大小和透明度
    n_verts = len(verts)
    pt_size = float(np.clip(5000.0 / n_verts, 0.3, 3.0))
    pt_alpha = float(np.clip(0.8 - n_verts / 150000, 0.3, 0.8))

    # Front view: X-Z, Side view: Y-Z, Top view: X-Y
    projections = [
        ('Front (X-Z)', 0, 2, 0, 1),   # title, h_idx, v_idx, xlabel_idx, ylabel_idx
        ('Side (Y-Z)', 1, 2, 1, 0),
        ('Top (X-Y)', 0, 1, 0, 1),
    ]
    axis_labels = ['X (mm)', 'Y (mm)', 'Z (mm)']

    ax_front = None
    for idx, (title, h_idx, v_idx, xl_idx, yl_idx) in enumerate(projections):
        ax = fig.add_subplot(gs_right[0, idx])

        ax.scatter(verts[:, h_idx], verts[:, v_idx],
                   s=pt_size, c='steelblue', alpha=pt_alpha, rasterized=True)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(axis_labels[xl_idx], fontsize=14)
        ax.set_ylabel(axis_labels[yl_idx], fontsize=14)
        ax.tick_params(labelsize=11)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if idx == 0:
            ax_front = ax

    # BBox 标注放在右列区域底部
    ax_front.text(0.5, -0.35, f'BBox: {bbox_label}',
                  transform=ax_front.transAxes, ha='center',
                  fontsize=11, color='#555555')

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 5. 主流程
# ============================================================
def main():
    os.makedirs(MESHES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_matplotlib()

    print("=" * 60)
    print("Step 1 — 骨骼三维重建")
    print("=" * 60)

    # 1. 读取选中列表
    print("\n[1/4] Loading selected_cases.json...")
    grouped_entries, all_entries = load_selected(SELECTED_JSON)

    # 2. 断点续跑检查
    print("\n[2/4] Checking existing .ply files...")
    existing = get_existing_plys()
    total_requested = len(all_entries)
    total_skipped = 0
    total_success = 0
    total_failed = 0
    vertex_counts = []
    face_counts = []
    error_lines = []
    error_log_path = os.path.join(MESHES_DIR, 'error_log.txt')

    # Load existing error log to preserve history
    if os.path.exists(error_log_path):
        with open(error_log_path, 'r', encoding='utf-8') as f:
            error_lines = f.read().splitlines()

    # 3. 逐 case 处理
    print(f"\n[3/4] Processing {total_requested} entries ({len(grouped_entries)} cases)...")
    start_t = time.time()

    for case_idx, (case_id, entries) in enumerate(grouped_entries.items(), 1):
        print(f"\n  Case {case_idx}/{len(grouped_entries)}: {case_id} ({len(entries)} bones)")

        # 加载 CT（该 case 下所有骨骼复用）
        ct_path = os.path.join(RAW_DIR, case_id, 'ct.nii.gz')
        try:
            ct_img = nib.load(ct_path)
            ct_spacing = ct_img.header.get_zooms()[:3]
        except Exception as e:
            msg = f"  [ERROR] Failed to load CT for {case_id}: {e}"
            print(msg)
            for bone_entry in entries:
                error_lines.append(f"{case_id}_{bone_entry['bone']}: {msg}")
                total_failed += 1
            continue

        for entry in entries:
            bone = entry['bone']
            bone_group = entry['bone_group']
            ply_name = f"{case_id}_{bone}.ply"
            ply_path = os.path.join(MESHES_DIR, ply_name)

            # 断点续跑：跳过已有
            if ply_name.replace('.ply', '') in existing:
                total_skipped += 1
                continue

            # 重建
            try:
                # 加载 mask（需要单独获取 mask 的 spacing）
                seg_dir = os.path.join(RAW_DIR, case_id, 'segmentations')
                mask_path = os.path.join(seg_dir, f'{bone}.nii.gz')
                mask_img = nib.load(mask_path)
                mask_spacing = mask_img.header.get_zooms()[:3]
                mask_data = np.asanyarray(mask_img.dataobj)

                mesh = reconstruct_bone(case_id, bone, ct_img, ct_spacing, mask_spacing)

                # 保存 .ply
                mesh.export(ply_path)
                vertex_counts.append(len(mesh.vertices))
                face_counts.append(len(mesh.faces))
                total_success += 1

                print(f"    [OK] {bone:30s} -> {ply_name}  "
                      f"(V={len(mesh.vertices)}, F={len(mesh.faces)})")

            except Exception as e:
                msg = f"  [FAIL] {bone}: {e}"
                print(f"    {msg}")
                error_lines.append(f"{case_id}_{bone}: {e}")
                total_failed += 1
                continue

        # 释放 CT 内存
        del ct_img

    elapsed = time.time() - start_t
    print(f"\n  Processing time: {elapsed:.1f}s")

    # 保存 error log
    if error_lines:
        # 重新写（去重）
        unique_errors = list(dict.fromkeys(error_lines))
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_errors))
        print(f"\n  [INFO] Error log saved: {error_log_path} ({len(unique_errors)} entries)")

    # 4. 可视化：每组选最大体积的代表
    print("\n[4/4] Generating representative visualizations...")
    fig_start = time.time()

    # 按 bone_group 分组，每组选 volume_cm3 最大的
    rep_map = {}
    for e in all_entries:
        bg = e['bone_group']
        if bg not in rep_map or e['volume_cm3'] > rep_map[bg]['volume_cm3']:
            rep_map[bg] = e

    vis_count = 0
    for bg, rep in sorted(rep_map.items()):
        rc = rep['case']
        rb = rep['bone']
        ply_name = f"{rc}_{rb}.ply"
        ply_path = os.path.join(MESHES_DIR, ply_name)

        # 如果该 ply 不存在（之前失败了），跳过
        if not os.path.exists(ply_path):
            print(f"    [SKIP] {bg}: {ply_name} not found (was not generated)")
            continue

        try:
            # 加载 CT 和 mask 用于左图
            ct_path = os.path.join(RAW_DIR, rc, 'ct.nii.gz')
            ct_data = nib.load(ct_path)
            mask_path = os.path.join(RAW_DIR, rc, 'segmentations', f'{rb}.nii.gz')
            mask_data = np.asanyarray(nib.load(mask_path).dataobj)

            # 加载网格
            mesh = trimesh.load(ply_path)

            # 输出图片
            out_path = os.path.join(FIG_DIR, f'{bg}_{rc}_{rb}.png')
            plot_reconstruction(rc, rb, bg, ct_data, None, mask_data, mesh, out_path)
            vis_count += 1
            print(f"    [OK] {bg:25s} -> {os.path.basename(out_path)}")

        except Exception as e:
            print(f"    [FAIL] {bg}: {e}")

    print(f"  Visualization time: {time.time() - fig_start:.1f}s")

    # 5. 统计摘要
    print("\n" + "=" * 60)
    print("Step 1 — Summary")
    print("=" * 60)
    print(f"  Total requested:  {total_requested}")
    print(f"  Success:          {total_success}")
    print(f"  Skipped (resume): {total_skipped}")
    print(f"  Failed:           {total_failed}")
    if total_success > 0:
        print(f"  Avg vertices:     {np.mean(vertex_counts):.0f}")
        print(f"  Avg faces:        {np.mean(face_counts):.0f}")
        print(f"  Min vertices:     {np.min(vertex_counts)}")
        print(f"  Max vertices:     {np.max(vertex_counts)}")
    print(f"  Visualizations:   {vis_count}/{len(rep_map)} groups")
    print(f"  Total time:       {elapsed:.1f}s")


if __name__ == '__main__':
    main()