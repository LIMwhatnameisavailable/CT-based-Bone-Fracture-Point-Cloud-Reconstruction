#!/usr/bin/env python
"""
Step 0 — 数据集筛选与概览
扫描所有 case 的 segmentation mask，结合 meta.csv 信息，
按骨骼分组配额筛选，输出 selected_cases.json 及可视化。
"""

import os, json, sys, time
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# 0. 路径与配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
META_PATH = os.path.join(RAW_DIR, 'meta.csv')
OUTPUT_JSON = os.path.join(PROJECT_ROOT, 'results', 'step0_dataset_overview', 'selected_cases.json')
OUTPUT_TXT = os.path.join(PROJECT_ROOT, 'results', 'step0_dataset_overview', 'step0_summary.txt')
FIG_DIR = os.path.join(PROJECT_ROOT, 'results')

STEP0_FIG_DIR = os.path.join(FIG_DIR, 'step0_dataset_overview')

MAX_PER_GROUP = 10

# 骨骼分组定义
BONE_GROUPS = {
    'femur':              ['femur_left', 'femur_right'],
    'humerus':            ['humerus_left', 'humerus_right'],
    'vertebrae_lumbar':   [f'vertebrae_L{i}' for i in range(1, 6)],
    'vertebrae_thoracic': [f'vertebrae_T{i}' for i in range(1, 13)],
    'vertebrae_cervical': [f'vertebrae_C{i}' for i in range(1, 8)],
    'rib':                [f'rib_{side}_{i}' for side in ['left', 'right'] for i in range(1, 13)],
    'pelvis':             ['hip_left', 'hip_right', 'sacrum'],
    'skull':              ['skull'],
    'clavicula':          ['clavicula_left', 'clavicula_right'],
    'scapula':            ['scapula_left', 'scapula_right'],
    'sternum':            ['sternum'],
}

# 构建骨骼名→分组的反向映射
BONE_TO_GROUP = {}
for group, bones in BONE_GROUPS.items():
    for b in bones:
        BONE_TO_GROUP[b] = group


def get_bone_group(bone_name):
    """返回骨骼所属分组名，不在预定义分组中则返回 'other'"""
    return BONE_TO_GROUP.get(bone_name, 'other')


# ============================================================
# 1. 读取 meta.csv
# ============================================================
def load_meta(meta_path):
    """读取 meta.csv，返回 case_id → info 的字典"""
    df = pd.read_csv(meta_path, sep=';')
    # 确保必要列存在
    required = ['image_id', 'age', 'gender', 'pathology', 'study_type']
    for col in required:
        if col not in df.columns:
            print(f"[ERROR] meta.csv missing column: {col}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)

    meta = {}
    for _, row in df.iterrows():
        case_id = str(row['image_id']).strip()
        meta[case_id] = {
            'age': row['age'],
            'gender': str(row['gender']).strip(),
            'pathology': str(row['pathology']).strip(),
            'study_type': str(row['study_type']).strip(),
        }
    return meta


# ============================================================
# 2. 扫描 mask 文件，计算体素信息
# ============================================================
def scan_masks(case_dir, case_id, meta_info):
    """扫描单个 case 的 segmentations/ 目录，返回骨骼信息列表"""
    seg_dir = os.path.join(case_dir, 'segmentations')
    if not os.path.isdir(seg_dir):
        return []

    results = []
    for fname in sorted(os.listdir(seg_dir)):
        if not fname.endswith('.nii.gz'):
            continue
        bone_name = fname.replace('.nii.gz', '')

        mask_path = os.path.join(seg_dir, fname)
        try:
            img = nib.load(mask_path)
            data = img.get_fdata()
            spacing = img.header.get_zooms()[:3]
        except Exception as e:
            print(f"  [WARN] Failed to read {mask_path}: {e}")
            continue

        voxel_count = int(np.sum(data > 0))
        if voxel_count == 0:
            continue

        # 体积 (cm^3) = 体素数 × spacing_x × spacing_y × spacing_z / 1000
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        volume_cm3 = voxel_count * voxel_volume_mm3 / 1000.0

        bone_group = get_bone_group(bone_name)

        results.append({
            'case': case_id,
            'bone': bone_name,
            'bone_group': bone_group,
            'voxel_count': voxel_count,
            'volume_cm3': round(volume_cm3, 2),
            **meta_info,
        })

    return results


# 并行扫描包装函数（需在模块顶层，以便 ProcessPoolExecutor pickle）
def _scan_case_wrapper(args):
    """供 ProcessPoolExecutor 调用的顶层包装函数"""
    case_id, raw_dir, meta_info = args
    case_dir = os.path.join(raw_dir, case_id)
    return scan_masks(case_dir, case_id, meta_info)


# ============================================================
# 3. 分组配额筛选
# ============================================================
def pathology_priority(pathology):
    """病理优先级：no_pathology > trauma > tumor > other"""
    p = pathology.lower().replace(' ', '_')
    if p in ('no_pathology', 'no pathology'):
        return 0
    elif p == 'trauma':
        return 1
    elif p == 'tumor':
        return 2
    else:
        return 3


def simplify_pathology(pathology):
    """将病理归类为 no_pathology / trauma / tumor / other"""
    p = pathology.lower().replace(' ', '_')
    if p in ('no_pathology', 'no pathology'):
        return 'no_pathology'
    elif p == 'trauma':
        return 'trauma'
    elif p == 'tumor':
        return 'tumor'
    else:
        return 'other'


def select_cases(all_bones):
    """
    按分组配额筛选：
    1. 每个分组最多 MAX_PER_GROUP 例
    2. 同一个 case 的同一分组只计一次
    3. 配额分配：no_pathology 最多占 70%（10例中最多7例）；
       剩余配额按 trauma > tumor > other 优先级填充。
       各病理类别内部按体素数降序排列。
    """
    PATH_ORDER = ['no_pathology', 'trauma', 'tumor', 'other']

    # 按分组聚合
    grouped = {}
    for entry in all_bones:
        g = entry['bone_group']
        if g not in grouped:
            grouped[g] = []
        grouped[g].append(entry)

    selected = []
    selected_case_groups = set()  # 记录 (case, group) 去重

    for group_name, entries in sorted(grouped.items()):
        # 按病理类别分桶，去重
        buckets = {p: [] for p in PATH_ORDER}
        for e in entries:
            p_simple = simplify_pathology(e['pathology'])
            key = (e['case'], group_name)
            if key not in selected_case_groups:
                buckets[p_simple].append(e)

        # 各桶内按体素数降序排序
        for p in PATH_ORDER:
            buckets[p].sort(key=lambda x: -x['voxel_count'])

        # 配额分配
        max_no_pathology = int(MAX_PER_GROUP * 0.7)  # 70% = 7
        assigned = []

        # 先取 no_pathology（最多7例）
        no_path_count = min(len(buckets['no_pathology']), max_no_pathology)
        assigned.extend(buckets['no_pathology'][:no_path_count])

        remaining = MAX_PER_GROUP - len(assigned)
        # 再按 trauma > tumor > other 填充
        for p in ['trauma', 'tumor', 'other']:
            if remaining <= 0:
                break
            take = min(len(buckets[p]), remaining)
            assigned.extend(buckets[p][:take])
            remaining -= take

        # 加入最终列表
        for entry in assigned:
            key = (entry['case'], group_name)
            selected_case_groups.add(key)
            selected.append(entry)

    # 按 case + bone 排序输出
    selected.sort(key=lambda e: (e['case'], e['bone']))

    return selected


# ============================================================
# 4. 可视化
# ============================================================
def setup_matplotlib():
    """设置 matplotlib 全局风格"""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_overview(selected, output_path):
    """第一张图：各分组选中数量（水平柱状图）+ 年龄分布直方图"""
    df = pd.DataFrame(selected)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左图：各分组选中数量水平柱状图
    group_counts = df['bone_group'].value_counts().sort_values()
    ax1 = axes[0]
    bars = ax1.barh(range(len(group_counts)), group_counts.values, color='steelblue', height=0.5)
    ax1.set_yticks(range(len(group_counts)))
    ax1.set_yticklabels(group_counts.index)
    ax1.set_xlabel('Selected Cases', fontsize=14)
    ax1.set_ylabel('Bone Group', fontsize=14)
    ax1.set_title('Cases per Bone Group', fontsize=16)
    ax1.tick_params(axis='both', labelsize=11)
    # 在柱状条右侧标注数字
    for i, (_, v) in enumerate(group_counts.items()):
        ax1.text(v + 0.3, i, str(v), va='center', fontsize=11)
    ax1.set_xlim(0, group_counts.max() + 3)

    # 右图：年龄分布直方图
    ax2 = axes[1]
    ages = df['age'].dropna()
    if len(ages) > 0:
        ax2.hist(ages, bins=10, color='steelblue', edgecolor='white')
    ax2.set_xlabel('Age', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.set_title('Age Distribution', fontsize=16)
    ax2.tick_params(axis='both', labelsize=11)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved overview figure: {output_path}")


def plot_pathology_dist(selected, output_path):
    """第二张图：pathology 饼图 + 骨骼体积箱线图"""
    df = pd.DataFrame(selected)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：pathology 分布饼图（归为四大类）
    ax1 = axes[0]
    df['pathology_simple'] = df['pathology'].apply(simplify_pathology)
    path_counts = df['pathology_simple'].value_counts()
    colors_path = {'no_pathology': '#2ecc71', 'trauma': '#e74c3c', 'tumor': '#f39c12', 'other': '#95a5a6'}
    path_colors = [colors_path.get(p, '#95a5a6') for p in path_counts.index]
    wedges, texts, autotexts = ax1.pie(
        path_counts.values, labels=path_counts.index,
        autopct='%1.1f%%', colors=path_colors,
        textprops={'fontsize': 11}
    )
    for t in texts:
        t.set_fontsize(11)
    for t in autotexts:
        t.set_fontsize(11)
    ax1.set_title('Pathology Distribution', fontsize=16)

    # 如果只有一种 pathology 类别，添加注释说明
    if len(path_counts) == 1:
        only_category = path_counts.index[0]
        if only_category == 'no_pathology':
            note = "All selected cases are no_pathology;\ntrauma/tumor cases exist in dataset\nbut were deprioritized"
        else:
            note = f"All selected cases are {only_category}"
        ax1.annotate(
            note, xy=(0.5, -0.15), xycoords='axes fraction',
            ha='center', va='top', fontsize=11, color='gray',
            linespacing=1.5
        )

    # 右图：按骨骼分组的体积箱线图
    ax2 = axes[1]
    group_order = df.groupby('bone_group')['volume_cm3'].median().sort_values(ascending=False).index
    data_groups = [df[df['bone_group'] == g]['volume_cm3'].values for g in group_order]
    bp = ax2.boxplot(data_groups, labels=group_order, patch_artist=True)

    # 给箱线图上色
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)

    ax2.set_xlabel('Bone Group', fontsize=14)
    ax2.set_ylabel('Volume (cm$^3$)', fontsize=14)
    ax2.set_title('Bone Volume Distribution', fontsize=16)
    ax2.tick_params(axis='both', labelsize=11)
    ax2.set_xticklabels(group_order, rotation=45, ha='right', fontsize=11)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved pathology distribution figure: {output_path}")


# ============================================================
# 5. JSON 序列化辅助（模块顶层，可被 pickle）
# ============================================================
def convert_native(obj):
    """递归将 numpy 类型转换为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: convert_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# ============================================================
# 6. 保存统计摘要
# ============================================================
def save_summary(selected, output_path):
    """保存统计摘要到 txt 文件"""
    df = pd.DataFrame(selected)

    lines = []
    lines.append("=" * 60)
    lines.append("Step 0 — Dataset Screening Summary")
    lines.append("=" * 60)
    lines.append(f"Total selected entries: {len(selected)}")
    lines.append(f"Total unique cases:     {df['case'].nunique()}")
    lines.append("")

    # 各分组数量
    lines.append("--- Per Group Count ---")
    group_counts = df['bone_group'].value_counts()
    for g, cnt in group_counts.items():
        lines.append(f"  {g:25s}: {cnt}")
    lines.append("")

    # 年龄
    ages = df['age'].dropna()
    if len(ages) > 0:
        lines.append(f"Age range:  {int(ages.min())} ~ {int(ages.max())}")
        lines.append(f"Mean age:   {ages.mean():.1f}")
        lines.append(f"Median age: {ages.median():.1f}")
    lines.append("")

    # Pathology 分布
    lines.append("--- Pathology Distribution ---")
    df_s = df.copy()
    df_s['pathology_simple'] = df_s['pathology'].apply(simplify_pathology)
    path_counts = df_s['pathology_simple'].value_counts()
    for p, cnt in path_counts.items():
        lines.append(f"  {p:25s}: {cnt} ({cnt/len(selected)*100:.1f}%)")
    lines.append("")

    # 体积统计
    lines.append("--- Volume Statistics (cm^3) ---")
    vol = df['volume_cm3']
    lines.append(f"  Total volume: {vol.sum():.1f}")
    lines.append(f"  Mean volume:  {vol.mean():.1f}")
    lines.append(f"  Median volume:{vol.median():.1f}")
    lines.append(f"  Min volume:   {vol.min():.1f}")
    lines.append(f"  Max volume:   {vol.max():.1f}")
    lines.append("")
    lines.append("=" * 60)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [OK] Saved summary: {output_path}")


# ============================================================
# 6. 断点续跑检查
# ============================================================
def check_existing_output():
    """检查已有输出文件，询问是否重新生成"""
    if os.path.exists(OUTPUT_JSON):
        print(f"[INFO] {OUTPUT_JSON} already exists.")
        resp = input("  Regenerate? (y/N): ").strip().lower()
        if resp != 'y':
            print("[INFO] Skipping Step 0. Loading existing selection...")
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(STEP0_FIG_DIR, exist_ok=True)
    setup_matplotlib()

    # 断点续跑检查
    existing = check_existing_output()
    if existing is not None:
        # 如果已有且用户选择不重新生成，只输出可视化
        print(f"[INFO] Loaded {len(existing)} entries from existing JSON.")
        # 仍生成可视化
        plot_overview(existing, os.path.join(STEP0_FIG_DIR, 'step0_dataset_overview.png'))
        plot_pathology_dist(existing, os.path.join(STEP0_FIG_DIR, 'step0_pathology_dist.png'))
        save_summary(existing, OUTPUT_TXT)
        print("[DONE] Step 0 complete (visualization regenerated).")
        return

    print("=" * 60)
    print("Step 0 — Dataset Screening")
    print("=" * 60)

    # 1. 读取 meta
    print("\n[1/5] Loading meta.csv...")
    meta = load_meta(META_PATH)
    print(f"  Loaded {len(meta)} case metadata entries.")

    # 2. 并行扫描所有 case
    print("\n[2/5] Scanning segmentation masks (parallel)...")
    all_bones = []
    case_dirs = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d)) and d.startswith('s')
    ])
    print(f"  Found {len(case_dirs)} case directories.")

    # 确定并行数
    num_workers = os.cpu_count() or 4
    print(f"  Using {num_workers} worker processes.")

    # 准备参数
    default_meta = {
        'age': np.nan, 'gender': 'unknown',
        'pathology': 'unknown', 'study_type': 'unknown'
    }
    task_args = []
    for case_id in case_dirs:
        meta_info = meta.get(case_id, default_meta.copy())
        task_args.append((case_id, RAW_DIR, meta_info))

    # 并行执行
    start_t = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_scan_case_wrapper, arg): arg[0] for arg in task_args}
        done = 0
        for future in as_completed(futures):
            case_id = futures[future]
            done += 1
            try:
                bones = future.result()
                all_bones.extend(bones)
            except Exception as e:
                print(f"  [ERROR] Case {case_id} failed: {e}")
            if done % 20 == 0 or done == len(case_dirs):
                elapsed = time.time() - start_t
                print(f"  Progress: {done}/{len(case_dirs)} cases done ({len(all_bones)} bones found, {elapsed:.0f}s elapsed)")

    elapsed = time.time() - start_t
    print(f"  Scanning complete: {len(all_bones)} bone entries in {elapsed:.1f}s")

    # 3. 分组筛选
    print("\n[3/5] Selecting cases by group quota...")
    selected = select_cases(all_bones)
    print(f"  Selected {len(selected)} entries (unique cases: {len(set(e['case'] for e in selected))})")

    # 打印各分组数量
    df_sel = pd.DataFrame(selected)
    group_counts = df_sel['bone_group'].value_counts()
    for g in sorted(group_counts.index):
        print(f"    {g:25s}: {group_counts[g]}")

    # 4. 保存 JSON
    print("\n[4/5] Saving selected_cases.json...")

    # 移除 id 字段（中间计算用）
    output_entries = []
    for e in selected:
        entry = {k: v for k, v in e.items() if k != 'id'}
        output_entries.append(convert_native(entry))
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_entries, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(output_entries)} entries to {OUTPUT_JSON}")

    # 5. 可视化 + 统计摘要
    print("\n[5/5] Generating figures and summary...")
    plot_overview(selected, os.path.join(STEP0_FIG_DIR, 'step0_dataset_overview.png'))
    plot_pathology_dist(selected, os.path.join(STEP0_FIG_DIR, 'step0_pathology_dist.png'))
    save_summary(selected, OUTPUT_TXT)

    print("\n" + "=" * 60)
    print("Step 0 complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
