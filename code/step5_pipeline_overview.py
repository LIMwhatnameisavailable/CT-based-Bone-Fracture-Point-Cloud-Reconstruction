#!/usr/bin/env python
"""
Generate comprehensive pipeline overview (5 rows x 2 columns).
Step 1: CT slice + mask contour (L) | 3-view mesh projection (R)
Step 2: Front view Z-color scatter (L) | XY density hexbin (R)
Step 3a: Intact bone (L) | Plane cut colored by fragment (R)
Step 3b: Voronoi fracture (L) | Spectral Modal fracture (R)
Step 4: Fracture surface red/gray (L) | Confusion matrix heatmap (R)
"""
import os, glob
import numpy as np
import nibabel as nib
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAC_DIR = os.path.join(PROJECT_ROOT, 'data', 'fractured')
PC_DIR = os.path.join(PROJECT_ROOT, 'data', 'pointclouds')
LABELED_DIR = os.path.join(PROJECT_ROOT, 'data', 'labeled')
MESH_DIR = os.path.join(PROJECT_ROOT, 'data', 'meshes')
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')

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


def select_best_case(bone_group='femur'):
    """
    From the given bone_group, pick the case with highest boundary_mask fraction
    that also has a physical file. If none in the group has physical, fall back
    to the best boundary% voronoi in the group.
    Returns (case_id, bone_name, bone_group).
    """
    # Get all voronoi files for this bone group with boundary%
    pattern = os.path.join(FRAC_DIR, f'*{bone_group}*_voronoi.npz')
    voronoi_files = glob.glob(pattern)
    if not voronoi_files:
        raise FileNotFoundError(f"No {bone_group} voronoi files found.")

    candidates = []
    for f in voronoi_files:
        d = np.load(f, allow_pickle=True)
        pct = d['boundary_mask'].sum() / len(d['boundary_mask'])
        d.close()
        basename = os.path.basename(f)
        prefix = basename.replace('_voronoi.npz', '')
        parts = prefix.split('_', 1)
        candidates.append((pct, parts[0], parts[1]))
    candidates.sort(key=lambda x: x[0], reverse=True)

    # First try: best boundary% case that also has physical file
    for pct, cid, bname in candidates:
        phys_path = os.path.join(FRAC_DIR, f"{cid}_{bname}_physical.npz")
        if os.path.exists(phys_path):
            print(f"  Selected: {cid}_{bname} (boundary={pct*100:.1f}%, has physical)")
            return cid, bname, bone_group

    # Fallback: no case in this group has both — use best boundary% only
    pct, cid, bname = candidates[0]
    print(f"  [WARN] No {bone_group} physical file found; using {cid}_{bname} only")
    return cid, bname, bone_group


def load_ct_slice_and_mask(case_id, bone_name):
    """
    Load CT volume and mask, return middle axial slice (intensity image + mask contour).
    Returns (ct_slice, mask_slice) as 2D arrays.
    """
    ct_path = os.path.join(RAW_DIR, case_id, 'ct.nii.gz')
    mask_path = os.path.join(RAW_DIR, case_id, 'segmentations', f'{bone_name}.nii.gz')
    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        print(f"  [WARN] CT/mask not found for {case_id}_{bone_name}, using synthetic data")
        return None, None

    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # Find the axial slice with the most mask pixels
    best_slice = 0
    best_count = 0
    for i in range(mask_data.shape[2]):
        cnt = mask_data[:, :, i].sum()
        if cnt > best_count:
            best_count = cnt
            best_slice = i

    ct_slice = np.rot90(ct_data[:, :, best_slice])
    mask_slice = np.rot90(mask_data[:, :, best_slice])
    return ct_slice, mask_slice


def load_mesh_vertices(case_id, bone_name):
    """Load mesh ply and return vertices array."""
    mesh_path = os.path.join(MESH_DIR, f"{case_id}_{bone_name}.ply")
    if not os.path.exists(mesh_path):
        print(f"  [WARN] Mesh not found: {mesh_path}")
        return None
    mesh = trimesh.load(mesh_path)
    return mesh.vertices


def plot_step1_left(ax, ct_slice, mask_slice):
    """CT slice + mask contour."""
    if ct_slice is None:
        ax.text(0.5, 0.5, 'CT data N/A', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        return
    ax.imshow(ct_slice, cmap='gray', aspect='equal')
    ax.contour(mask_slice, levels=[0.5], colors='#E64B35', linewidths=1.2)
    ax.set_title('Step 1: CT Slice + Mask Contour', fontsize=16)
    ax.set_xlabel('X (pixel)', fontsize=14)
    ax.set_ylabel('Y (pixel)', fontsize=14)
    ax.tick_params(labelsize=11)


def plot_step1_right(fig, subplot_spec, vertices):
    """
    3-view projection (Front/Side/Top) via inner GridSpec 1x3.
    Takes a SubplotSpec (not Axes) so GridSpecFromSubplotSpec works.
    """
    if vertices is None:
        ax = fig.add_subplot(subplot_spec)
        ax.text(0.5, 0.5, 'Mesh data N/A', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
        return

    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    inner_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=subplot_spec,
                                        wspace=0.25, hspace=0)
    views = [
        ('Front View', 0, 2, 'X (mm)', 'Z (mm)'),
        ('Side View', 1, 2, 'Y (mm)', 'Z (mm)'),
        ('Top View', 0, 1, 'X (mm)', 'Y (mm)'),
    ]
    for idx, (title, xi, yi, xl, yl) in enumerate(views):
        inner_ax = fig.add_subplot(inner_gs[0, idx])
        inner_ax.scatter(vertices[:, xi], vertices[:, yi],
                         s=0.3, c='steelblue', alpha=0.4, rasterized=True)
        inner_ax.set_title(title, fontsize=14)
        inner_ax.set_xlabel(xl, fontsize=14)
        inner_ax.set_ylabel(yl, fontsize=14)
        inner_ax.tick_params(labelsize=11)
        inner_ax.set_aspect('equal')
        inner_ax.spines['top'].set_visible(False)
        inner_ax.spines['right'].set_visible(False)
        # BBox annotation
        inner_ax.text(0.5, -0.18,
                      f'BBox: {bbox[xi]:.0f} × {bbox[yi]:.0f} mm',
                      transform=inner_ax.transAxes, ha='center',
                      fontsize=11, color='#555555')


def plot_step2_left(ax, pc_data):
    """Front view (X-Z) scatter, color by Z with viridis."""
    pts = pc_data['points']
    ax.scatter(pts[:, 0], pts[:, 2], c=pts[:, 2],
               s=0.8, cmap='viridis', alpha=0.6, rasterized=True)
    ax.set_title('Step 2: Point Cloud (Z-color)', fontsize=16)
    ax.set_xlabel('X (normalized)', fontsize=14)
    ax.set_ylabel('Z (normalized)', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.set_aspect('equal')


def plot_step2_right(ax, pc_data):
    """XY density hexbin."""
    pts = pc_data['points']
    ax.hexbin(pts[:, 0], pts[:, 1], gridsize=30, cmap='Blues', alpha=0.7)
    ax.set_title('Step 2: XY Density', fontsize=16)
    ax.set_xlabel('X (normalized)', fontsize=14)
    ax.set_ylabel('Y (normalized)', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.set_aspect('equal')


def plot_step3_left(ax, pc_data):
    """Intact bone point cloud (gray)."""
    pts = pc_data['points']
    ax.scatter(pts[:, 0], pts[:, 2], c='gray', s=0.5, alpha=0.4, rasterized=True)
    ax.set_title('Step 3a: Intact Bone', fontsize=16)
    ax.set_xlabel('X (normalized)', fontsize=14)
    ax.set_ylabel('Z (normalized)', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.set_aspect('equal')


def plot_step3_right(ax, frac_data):
    """Plane cut result colored by fragment_id (tab10)."""
    pts = frac_data['points']
    fid = frac_data['fragment_id']
    n_frags = len(np.unique(fid))
    cmap = plt.get_cmap('tab10')

    for f in np.unique(fid):
        mask = fid == f
        color = cmap(f / 10.0)
        ax.scatter(pts[mask, 0], pts[mask, 2],
                   c=[color], s=1.0, alpha=0.7, rasterized=True)
    ax.set_title(f'Step 3a: Plane Cut ({n_frags} frags)', fontsize=16)
    ax.set_xlabel('X (normalized)', fontsize=14)
    ax.set_ylabel('Z (normalized)', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.set_aspect('equal')


def plot_step3b_left(ax, voronoi_data):
    """Voronoi fracture (tab10)."""
    pts = voronoi_data['points']
    fid = voronoi_data['fragment_id']
    n_frags = len(np.unique(fid))
    cmap = plt.get_cmap('tab10')

    for f in np.unique(fid):
        mask = fid == f
        color = cmap(f / 10.0)
        ax.scatter(pts[mask, 0], pts[mask, 2],
                   c=[color], s=1.0, alpha=0.7, rasterized=True)
    ax.set_title(f'Step 3b: Voronoi ({n_frags} frags)', fontsize=16)
    ax.set_xlabel('X (normalized)', fontsize=14)
    ax.set_ylabel('Z (normalized)', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.set_aspect('equal')


def plot_step3b_right(ax, phys_data):
    """Spectral Modal fracture (tab10)."""
    pts = phys_data['points']
    fid = phys_data['fragment_id']
    n_frags = len(np.unique(fid))
    cmap = plt.get_cmap('tab10')

    for f in np.unique(fid):
        mask = fid == f
        color = cmap(f / 10.0)
        ax.scatter(pts[mask, 0], pts[mask, 2],
                   c=[color], s=1.0, alpha=0.7, rasterized=True)
    ax.set_title(f'Step 3c: Spectral Modal ({n_frags} frags)', fontsize=16)
    ax.set_xlabel('X (normalized)', fontsize=14)
    ax.set_ylabel('Z (normalized)', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.set_aspect('equal')


def plot_step4_left(ax, points, labels, boundary_mask):
    """Fracture surface scatter: gray=non-fracture, red=fracture."""
    non_b = labels == 0
    is_b = labels == 1
    ax.scatter(points[non_b, 0], points[non_b, 2],
               c='gray', s=0.5, alpha=0.3, rasterized=True)
    ax.scatter(points[is_b, 0], points[is_b, 2],
               c='red', s=2.0, alpha=0.8, rasterized=True)
    label_rate = 100.0 * labels.sum() / len(labels)
    ax.set_title(f'Step 4: Fracture Surface ({label_rate:.1f}%)', fontsize=16)
    ax.set_xlabel('X (normalized)', fontsize=14)
    ax.set_ylabel('Z (normalized)', fontsize=14)
    ax.tick_params(labelsize=11)
    ax.set_aspect('equal')


def plot_step4_right(ax, labels, boundary_mask):
    """Confusion matrix with corrected labels."""
    tp = np.sum((labels == 1) & (boundary_mask == 1))
    fp = np.sum((labels == 1) & (boundary_mask == 0))
    fn = np.sum((labels == 0) & (boundary_mask == 1))
    tn = np.sum((labels == 0) & (boundary_mask == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max() + 1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Neg', 'Pos'], fontsize=11)
    ax.set_yticklabels(['Neg', 'Pos'], fontsize=11)
    ax.set_xlabel('Predicted (Inter-distance)', fontsize=14)
    ax.set_ylabel('Actual (Boundary Mask)', fontsize=14)
    ax.tick_params(labelsize=11)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=14,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    info = (f'P:{precision:.3f}  R:{recall:.3f}  '
            f'F1:{f1:.3f}  Acc:{accuracy:.3f}')
    ax.text(0.5, -0.22, info, transform=ax.transAxes,
            ha='center', fontsize=11, color='#333333')


def main(bone_group='femur'):
    print("=" * 60)
    print(f"Step 5 — Pipeline Overview (5x2 Grid) — {bone_group}")
    print("=" * 60)

    output_path = os.path.join(OUTPUT_DIR, f'pipeline_overview_{bone_group}.png')

    # 1. Select best case (voronoi and physical guaranteed same case)
    print(f"\n[1/4] Selecting best {bone_group} case...")
    case_id, bone_name, bone_group = select_best_case(bone_group)
    print(f"  Selected: {case_id}_{bone_name}")

    # 2. Load all required data
    print("\n[2/4] Loading data...")

    ct_slice, mask_slice = load_ct_slice_and_mask(case_id, bone_name)
    vertices = load_mesh_vertices(case_id, bone_name)

    pc_path = os.path.join(PC_DIR, f"{case_id}_{bone_name}.npz")
    pc_data = np.load(pc_path, allow_pickle=True)

    frac_path = os.path.join(FRAC_DIR, f"{case_id}_{bone_name}_frac.npz")
    frac_data = np.load(frac_path, allow_pickle=True)

    voronoi_path = os.path.join(FRAC_DIR, f"{case_id}_{bone_name}_voronoi.npz")
    voronoi_data = np.load(voronoi_path, allow_pickle=True)

    physical_path = os.path.join(FRAC_DIR, f"{case_id}_{bone_name}_physical.npz")
    phys_data = None
    if os.path.exists(physical_path):
        phys_data = np.load(physical_path, allow_pickle=True)
        print(f"  Physical: {case_id}_{bone_name}_physical.npz")
    else:
        print(f"  [WARN] Physical file not found for {case_id}_{bone_name}")

    labeled_path = os.path.join(LABELED_DIR, f"{case_id}_{bone_name}_labeled.npz")
    if os.path.exists(labeled_path):
        labeled_data = np.load(labeled_path, allow_pickle=True)
    else:
        print(f"  [WARN] Labeled data not found, using voronoi boundary_mask")
        labeled_data = voronoi_data

    # 3. Build figure
    print("\n[3/4] Building 5x2 figure...")
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.2,
                  left=0.06, right=0.98, top=0.95, bottom=0.04)

    # === Row 0: Step 1 ===
    ax0_l = fig.add_subplot(gs[0, 0])
    plot_step1_left(ax0_l, ct_slice, mask_slice)
    plot_step1_right(fig, gs[0, 1], vertices)

    # === Row 1: Step 2 ===
    ax1_l = fig.add_subplot(gs[1, 0])
    plot_step2_left(ax1_l, pc_data)
    ax1_r = fig.add_subplot(gs[1, 1])
    plot_step2_right(ax1_r, pc_data)

    # === Row 2: Step 3a — Plane Cut (unified axes) ===
    ax2_l = fig.add_subplot(gs[2, 0])
    plot_step3_left(ax2_l, pc_data)
    ax2_r = fig.add_subplot(gs[2, 1])
    plot_step3_right(ax2_r, frac_data)

    # Unify Step 3a axis ranges
    pc_pts = pc_data['points']
    frac_pts = frac_data['points']
    all_x = np.concatenate([pc_pts[:, 0], frac_pts[:, 0]])
    all_z = np.concatenate([pc_pts[:, 2], frac_pts[:, 2]])
    x_pad = (all_x.max() - all_x.min()) * 0.05
    z_pad = (all_z.max() - all_z.min()) * 0.05
    for ax in [ax2_l, ax2_r]:
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_z.min() - z_pad, all_z.max() + z_pad)

    # === Row 3: Step 3b — Voronoi vs Modal (unified axes) ===
    ax3_l = fig.add_subplot(gs[3, 0])
    plot_step3b_left(ax3_l, voronoi_data)
    ax3_r = fig.add_subplot(gs[3, 1])
    if phys_data is not None:
        plot_step3b_right(ax3_r, phys_data)
    else:
        ax3_r.text(0.5, 0.5, 'Spectral Modal data N/A',
                   transform=ax3_r.transAxes, ha='center', va='center',
                   fontsize=14, color='gray')

    # Unify Step 3b axis ranges (skip if no physical)
    v_pts = voronoi_data['points']
    if phys_data is not None:
        p_pts = phys_data['points']
        all_x = np.concatenate([v_pts[:, 0], p_pts[:, 0]])
        all_z = np.concatenate([v_pts[:, 2], p_pts[:, 2]])
    else:
        all_x = v_pts[:, 0]
        all_z = v_pts[:, 2]
    x_pad = (all_x.max() - all_x.min()) * 0.05
    z_pad = (all_z.max() - all_z.min()) * 0.05
    for ax in [ax3_l, ax3_r]:
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_z.min() - z_pad, all_z.max() + z_pad)

    # === Row 4: Step 4 ===
    points_labeled = labeled_data['points']
    if 'labels' in labeled_data:
        labels_labeled = labeled_data['labels']
    else:
        labels_labeled = labeled_data['boundary_mask'].astype(np.int32)

    boundary_mask_labeled = labeled_data['boundary_mask']

    ax4_l = fig.add_subplot(gs[4, 0])
    plot_step4_left(ax4_l, points_labeled, labels_labeled, boundary_mask_labeled)
    ax4_r = fig.add_subplot(gs[4, 1])
    plot_step4_right(ax4_r, labels_labeled, boundary_mask_labeled)

    # === Dynamic row labels from subplot positions ===
    fig.canvas.draw()
    row_axes = [ax0_l, ax1_l, ax2_l, ax3_l, ax4_l]
    row_labels = ['Step 1', 'Step 2', 'Step 3a', 'Step 3b&c', 'Step 4']
    for ax, label in zip(row_axes, row_labels):
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.005, y_center, label, fontsize=14, fontweight='bold',
                 va='center', rotation='vertical')

    # === Overall title ===
    fig.suptitle(f'Complete Fracture Point Cloud Pipeline\n'
                 f'{bone_group} | {case_id}_{bone_name}',
                 fontsize=18, y=1.01, fontweight='bold')

    plt.tight_layout(pad=3.0, h_pad=4.0)

    # 4. Save
    print(f"\n[4/4] Saving to {output_path} ...")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Pipeline overview saved: {output_path}")
    return output_path


if __name__ == '__main__':
    main('femur')
    main('clavicula')
