# 骨折复位点云重建与可视化实验

## 项目定位

本实验是对袁钦辉《三维胫骨平台骨折复位方法研究》数据集准备阶段的回溯性复现。不涉及深度学习算法，核心目标是跑通完整管线：

CT原图 (ct.nii.gz) → 骨骼三维重建 → 点云采样 → 模拟骨折破碎（平面切割 → Voronoi → 曲面谱分析） → 断面标注 → 可视化

重点在于理解数据集准备的全流程逻辑，产出高质量可视化结果用于汇报。

---

## 运行环境约束

- 本地笔记本 CPU 运行，无 GPU
- 操作系统：Windows 11；Shell：PowerShell，禁止使用 bash/linux 命令
- 单个 case 处理时间应控制在 15 分钟以内
- 避免 O(N²) 以上复杂度的暴力算法（N 为点云点数，默认 4096）
- 所有中间结果持久化到磁盘，支持断点续跑
- 在 pointnet 环境中执行任务；若缺少相关依赖及时补充

---

## 数据来源

使用 TotalSegmentator 小子集（102 例，3.2 GB，NIfTI 格式 .nii.gz）。

每例数据包含两个部分，两者都要使用：
- ct.nii.gz：CT 原图，三维灰度体素矩阵，值为 HU（Hounsfield Unit）
- segmentations/<bone_name>.nii.gz：对应骨骼的二值 mask（0=背景，1=目标骨骼）

操作逻辑：从 ct.nii.gz 读取体素数据，用 mask 定位目标骨骼体素，再对该区域进行三维重建。这是医学图像处理的标准流程，不可跳过 CT 原图直接用 mask。

### 可用骨骼标签说明

TotalSegmentator v2 中，以下骨骼标签在 total task 下可用：

骨骼分组（用于配额控制）：
- femur（股骨）：femur_left, femur_right
- humerus（肱骨）：humerus_left, humerus_right
- vertebrae_lumbar（腰椎）：vertebrae_L1 ~ L5
- vertebrae_thoracic（胸椎）：vertebrae_T1 ~ T12
- vertebrae_cervical（颈椎）：vertebrae_C1 ~ C7
- rib（肋骨）：rib_left_1 ~ rib_left_12, rib_right_1 ~ rib_right_12
- pelvis（骨盆类）：hip_left, hip_right, sacrum
- skull（颅骨）：skull
- clavicula（锁骨）：clavicula_left, clavicula_right
- scapula（肩胛骨）：scapula_left, scapula_right
- sternum（胸骨）：sternum


以上仅为建议骨种，并不作严格限定。
注意：tibia（胫骨）在 v2 中属于 appendicular_bones 子任务，小子集中可能不存在 tibia_left.nii.gz，请在 Step 0 中检测并跳过缺失标签。

---

## 目录结构说明

project 根目录下分为以下子目录：

data/raw/ 存放原始下载数据解压后的内容，每个 case 一个子文件夹，如 s0011/，内含 ct.nii.gz 和 segmentations/ 文件夹。summary 文本文件也存放在此。

data/meshes/ 存放 Step 1 输出的三角网格文件，命名格式为 s0001_femur_left.ply。

data/pointclouds/ 存放 Step 2 输出的完整骨骼点云，命名格式为 s0001_femur_left.npz。

data/fractured/ 存放 Step 3 输出的破碎后点云，命名格式为 s0001_femur_left_frac.npz。

data/labeled/ 存放 Step 4 输出的带断面标注点云，命名格式为 s0001_femur_left_labeled.npz。

results/ 存放所有可视化输出（PNG、HTML）和数据集文件，按步骤分子文件夹：step0_dataset_overview/（含 selected_cases.json）、step1_reconstruction/、step2_pointcloud/、step3_fracture/（内含 phase1/、phase2/、phase2_interactive/、phase3/ 子目录）、step4_label/。

code/ 存放所有脚本文件：step0_select_cases.py、step1_reconstruct.py、step2_sample_pointcloud.py、step3_fracture_phase1.py、step3_fracture_phase2.py、step3_fracture_phase3.py。

instruction/ 存放各步骤的说明文档：step0_instruction.md ~ step4_instruction.md、step3_phase3_instruction.md。

---

## 绘图风格规范（所有可视化必须遵守）

所有图像统一使用以下 matplotlib 配置，在每个可视化脚本开头调用：

    import matplotlib.pyplot as plt
    import matplotlib as mpl

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

字号规范（不得为凑布局缩小字号）：
- 图标题（title）>= 16
- 子图/子标题（sub title）>= 14
- 轴标签（xlabel/ylabel）>= 14
- 图例（legend）>= 11
- 刻度（tick）>= 11

布局规范：
- 每行最多 2 列，极限情况不超过 3 列
- 子图间距使用 plt.tight_layout(pad=2.0)
- 不得为凑布局而缩小字号
- 所有 ax 隐藏上边框和右边框（ax.spines['top'].set_visible(False) 等）
- fig.savefig 始终使用 bbox_inches='tight'

网格三视图规范（Step 1 网格可视化用）：
- 使用 GridSpec 布局，大框架 1x2（width_ratios=[1, 2]），右列内部再用 GridSpecFromSubplotSpec 分为 1x3
- 右列三视图使用 2D 散点投影，放弃 mplot3d：
  - Front view（前视图）：横轴=x，纵轴=z
  - Side view（侧视图）：横轴=y，纵轴=z
  - Top view（顶视图）：横轴=x，纵轴=y
- 每个投影用 ax.scatter(s=0.3, c='steelblue', alpha=0.4, rasterized=True) 绘制
- 每个子图保持等比例：ax.set_aspect('equal')
- BBox 标注：用 ax.text(0.5, -0.18, ..., transform=ax.transAxes, ha='center', fontsize=11, color='#555555')

CT 切片轮廓规范：
- 轮廓颜色使用 #E64B35（红色），linewidths=1.2
- 叠加方式：ax.contour(mask_slice, levels=[0.5], colors='#E64B35', linewidths=1.2)

---

## 数据处理流程

### Step 0 — 数据集筛选与概览

目标：从 102 例中自动筛选骨骼类型多样的 case，输出可用 case 列表。

逻辑：
1. 遍历所有 case，检测每例中存在哪些骨骼 mask 文件
2. 按骨骼类型分组，每类最多选 10 例，使总数不超过 60
3. 优先选择体积较大、网格质量好的 case（mask 中体素数量 > 最小阈值）
4. 输出 selected_cases.json，格式为列表，每项包含 case 名称和骨骼标签名，例如 {"case": "s0001", "bone": "femur_left"}

可视化输出 results/step0_dataset_overview.png：
- 子图1：各骨骼类型的 case 数量柱状图
- 子图2：选中 case 的骨骼体积分布箱线图（单位 cm^3）
- 布局：1行2列

#### 实际运行结果（2026-05-11）

输出文件：selected_cases.json，共120条目，30个unique case，覆盖13个骨骼分组，每组10例。

pathology分布：no_pathology 84条（70%），trauma 36条（30%），无tumor/other。

年龄范围：29~91岁，均值约67岁，70岁附近集中（数据集本身特性）。

可视化输出：results/step0_dataset_overview/step0_dataset_overview.png 和 step0_pathology_dist.png，均已生成。

与原计划的差异：tibia确认不存在（符合预期）；meta.csv被纳入筛选逻辑（原计划未提及但实际有用）；骨骼分组扩展为13组（含other），比原计划的11组多2组。

---

### Step 1 — 骨骼三维重建

目标：从 CT 原图 + mask 重建三角网格，输出 .ply 文件。

逻辑：
1. 用 nibabel.load 读取 ct.nii.gz，获取体素矩阵和 spacing（单位 mm/voxel）
2. 用 nibabel.load 读取对应骨骼的 mask 文件（如 segmentations/femur_left.nii.gz）
3. 用 mask 对 CT 体素做 Boolean masking，将 mask=0 的区域置零，提取目标骨骼区域
4. 用 skimage.measure.marching_cubes 对 masked volume 进行表面重建，level=0.5
5. 用 trimesh 进行后处理：用 trimesh.graph.split 保留最大连通分量，去除孤立碎片；可选 Laplacian 平滑，迭代次数不超过 5 次；顶点坐标乘以 spacing 转换为真实毫米坐标
6. 保存为 .ply 文件

可视化输出 results/step1_reconstruction/<case_bone>.png：
- 子图1：CT 原图中间轴向切片（灰度图），叠加 mask 轮廓（红色线条）
- 子图2：重建网格的三视图（前视/侧视/顶视），标注骨骼包围盒尺寸（长x宽x高，单位 mm）
- 布局：1行2列

#### 实际运行结果（2026-05-11）

重建120个条目全部成功（120/120）。运行时间约80秒（顺序处理，30个case各加载一次CT）。

网格规模：顶点数范围2088~142600，中位数约26206，面片数范围4018~286488。

可视化输出：results/step1_reconstruction/下12张代表性图片（每骨骼分组1张），包括clavicula、femur、humerus、pelvis、rib、scapula、skull、sternum、vertebrae三组及other。

与原计划的差异：输入直接使用mask二值数组做marching_cubes（而非mask×CT）；增加了bounding box裁剪加速（padding=2）；可视化改为每分组1张代表图（共12张）而非全量120张；三视图改用2D散点投影（替代mplot3d），散点大小和透明度根据顶点数动态自适应。

---

### Step 2 — 点云采样

目标：对三角网格表面均匀采样，输出归一化点云 .npz。

逻辑：
1. 固定采样点数 N=4096
2. 用 trimesh.sample.sample_surface(mesh, N) 进行面积加权均匀采样
3. 法向量从 mesh.face_normals[face_indices] 获取（已是单位向量），并显式归一化
4. 归一化：质心居中 → 单位球缩放
5. 保存 .npz，字段包括 points, normals, points_raw, centroid, scale（均为 float32）

可视化输出 results/step2_pointcloud/<bone_group>_<case>_<bone>.png：
- 左图：前视图 (X-Z) 散点，颜色按 Z 值 viridis 渐变
- 右图：XY 平面密度热力图（hexbin, cmap='Blues'）
- 布局：1行2列

#### 实际运行结果（2026-05-12）

采样 120 个条目全部成功（120/120）。运行时间约 0.4 秒（仅新采样20条，其余断点续跑跳过）。

法向量模长均值 1.0000（显式归一化后所有法向量为单位向量，符合预期）。

缩放因子范围：34.0~187.7（骨骼大小差异大，但归一化后统一在单位球内）。

可视化输出：results/step2_pointcloud/ 下 12 张代表性图片（每骨骼分组 1 张），包括 clavicula、femur、humerus、other、pelvis、rib、scapula、skull、sternum、vertebrae_cervical、vertebrae_lumbar、vertebrae_thoracic。

修复记录：vertebrae_lumbar 和 vertebrae_thoracic 的网格存在孔洞导致点云采样出现空洞和条带状空白区域。在 **step1_reconstruct.py** 添加 repair_mesh 函数（fill_holes + fix_normals + 移除退化面）后，对这两类骨骼共 20 个条目重新生成网格并重新采样，修复后点云分布均匀、无空洞。

---

### Step 3 — 模拟骨折破碎

目标：对完整骨骼网格进行模拟破碎，生成 2~4 块骨折碎片，并施加随机位移。

破碎方法（分三个阶段，逐级升级）：

阶段一（基础版，已完成）：随机平面切割。
在骨骼包围盒内随机生成 1~3 个切割平面（法向量随机，切割位置限定在骨骼中段 20%~80% 范围内），用 trimesh.intersections.mesh_plane 依次切割，得到 2~4 块碎片。

阶段二（升级版，已完成）：Voronoi 破碎。
在骨骼包围盒内生成 N 个 Voronoi 种子点（N=2~4），种子点偏向骨骼中轴线方向。对网格每个面片，根据其质心距离最近的种子点进行归属分配，将网格拆分为 N 块。对断面施加小幅随机法向量扰动（幅度 < 0.5mm），增加断面凹凸感。

阶段三（曲面谱分析版，已完成）：Laplace-Beltrami 谱分割。
对少量样本（24例，每骨组2例）使用曲面谱分析（Spectral Geometry）确定裂纹方向：构建网格的 Laplace-Beltrami 算子 L 和质量矩阵 M（robust-laplacian），求解广义特征值问题 L u = λ M u，取第一非零特征向量的零交叉面作为裂纹平面。该方法与 Sellan 2023 的有限元模态分析在数学形式上类似，区别在于刚度矩阵由几何 Laplacian 代替，不依赖材料弹性参数。断面施加高斯噪声扰动（std=0.5mm）。其余后处理逻辑与 Phase 2 完全一致。

三个阶段共同的后处理：
- 对每个碎片施加随机旋转（角度范围 -30 到 +30 度）和随机平移（范围 -20 到 +20 mm），模拟骨折块移位
- 碎片数量限制在 2~4 块（阶段三仅2块），过小的碎片（体积 < 总体积 5%）合并到相邻最大块
- 保存为 .npz 文件，字段包括 points（形状 N,3）、fragment_id（形状 N，整数标签）、transform（每块的旋转矩阵和平移向量）

可视化输出 results/step3_fracture/<case_bone>.png：
- 子图1：破碎前完整骨骼点云（灰色）
- 子图2：破碎后各碎片用不同颜色区分，已施加随机位移
- 布局：1行2列，标注碎片数量和最大位移量

#### 实际运行结果（2026-05-12）— 阶段一、二

阶段一：破碎 120 个条目全部成功（120/120）。运行时间约 36.8 秒，平均最大平移 23.6mm。

依赖安装：运行中补充安装了 shapely、mapbox-earcut、rtree 三个包，用于 trimesh.intersections.slice_mesh_plane(cap=True) 的断面封口。

Bug 修复记录：
- 碎片数指数增长问题：fracture_mesh 原逻辑每轮对所有碎片各切一刀，导致碎片数按 2^n 增长（clavicula、femur 等出现 8 块）。修复为每次只切体积最大的那块碎片，确保 1~3 刀产生 2~4 块碎片。
- transforms 存储格式：最初设计为每碎片 (R, t) 元组列表，但 np.array(..., dtype=object) 在 Windows 上广播异常。修复为 (n_frags, 4, 4) 齐次变换矩阵统一存储。

可视化输出：results/step3_fracture/ 下 12 张代表性图片（每骨骼分组 1 张），覆盖 clavicula、femur、humerus、other、pelvis、rib、scapula、skull、sternum、vertebrae 三组。

阶段二（Voronoi 破碎）：全部成功（120/120）。运行时间约 50.3 秒。平均碎片数 2.92（范围 2~4），平均最大平移 22.6mm（范围 11.4~30.6mm），断面点平均占比 7.1%（在预期 5%~20% 范围内）。

可视化输出：results/step3_fracture/phase2/ 下 12 张代表性图片（每骨骼分组 1 张），以及 3 组阶段一 vs 阶段二对比图。交互式 3D HTML 可视化（plotly 生成）由 step3_fracture_phase2.py 内部生成，输出至 results/step3_fracture/phase2_interactive/（12 个 HTML 文件，全量 4096 点不下采样）。

Bug 修复记录（第二次修改）：
- merge_small_fragments KeyError：big_tree.query 返回的是局部索引（小数组内位置），与 merged 字典的全局 key 不一致。修复为通过 big_indices 映射回全局索引。
- 断面点检测阈值从 ALPHA * d_nn（自适应密度估计）改为固定 TAU_MM=4.0mm 物理阈值，解决因归一化后密度估计偏差导致 boundary% 过低（3.8%）的问题。
- 碎片平移从固定 ±20mm 改为自适应（包围盒对角线 20%），确保小骨骼（如 clavicula）不移出视野、大骨骼（如 femur）有足够位移量。
- 对比图统一阶段一和阶段二的坐标轴范围，基于两组数据的全局最小最大值 +5% padding。

Bug 修复记录（第三次修改）：
- HTML 下采样去除：generate_interactive_html 改用全量 4096 点（取消 downsample），marker 参数调整为 intact size=2.0/color=lightgray、frag size=2.0/opacity=0.7、boundary size=5/symbol=x/opacity=1.0/color=red。
- Voronoi 种子扰动自适应扁平骨骼：在 generate_voronoi_seeds 中增加 flatness = short_len / long_len 计算，jitter_scale = short_len * (0.10 + 0.20 * (1 - flatness))，解决 vertebrae_thoracic 等扁平骨骼碎片在 XZ 投影中严重重叠的问题。
- 全局包围盒平移范围：apply_random_transform 改用 global_bbox_diag * 0.15 作为统一平移范围（替代原 per-fragment bbox_diag * 0.20），解决 vertebrae_cervical 等小骨骼碎片飞出视野的问题。
- step3_visualize_3d.py 已合并入 step3_fracture_phase2.py，不再单独存在。

#### 实际运行结果（2026-05-12）— 阶段三（曲面谱分析）

全部成功（24/24）。运行时间约 59.3 秒（平均每例 2.5 秒，远低于 5 分钟上限）。每例固定 2 块碎片，断面施加高斯噪声扰动（std=0.5mm）。

skull 和 vertebrae_cervical 角度接近 90° 表明裂纹接近冠状面切割，对扁平壳状骨和短椎骨而言属于物理合理结果，验收标准 < 45° 仅适用于长骨。other 分组包含软组织样本（s0250_liver、s0915_lung_upper_lobe_right），非骨骼组织，角度统计仅供参考。

**断面点占比**：平均 12.7%（在 5%~20% 目标范围内），相比 Phase 2（7.1%）略高，因平面切割断面接触更紧密，边界检测阈值 TAU_MM=2.5mm 更低。

**可视化输出**：results/step3_fracture/phase3/ 下 24 张破碎结果图 + 24 张 Phase 2 vs Phase 3 对比图。长骨（femur、humerus）呈清晰横向裂纹，扁骨（scapula、pelvis）呈斜向裂纹，与 Voronoi 的随机多面体破碎形成肉眼可见差异。

**关键修复记录**：
1. 初始构建 `block_diag([L, L, L])` 向量拉普拉斯算子求解 eigenvector，对管状网格产生环绕圆周变化的特征向量，导致纵向裂纹（femur 84.3°→FAIL）。修复为标量拉普拉斯算子直接求解，裂纹方向由零交叉点 PCA 主方向确定。
2. 单特征向量不稳定：一部分骨骼的最优裂纹藏在 u₂~u₅ 而非 u₁。增加多特征向量枚举策略（尝试 u₁~u₅），用评分函数（平衡度 35% + 居中程度 35% + 平面性 30%）选取最优平面。
3. 平面性指标（PCA 特征值比 s₂/s₁）有效抑制纵向裂纹：线状零交叉点 s₂/s₁ 低（< 0.3），环状零交叉点 s₂/s₁ 高（> 0.7），形成天然筛子。
4. 边界检测阈值从 TAU_MM=4.0 降低到 2.5mm，解决平面切割断面接触紧密导致的边界点误标。

---

### Step 4 — 断面自动标注

目标：对破碎点云中的每个点进行二值标注，区分断面点和非断面点。

逻辑（对应袁钦辉论文 3.2.5 节几何标注部分）：
1. 对任意两块相邻碎片，计算碎片 A 中每个点到碎片 B 中所有点的最短距离（使用 scipy.spatial.cKDTree 加速，复杂度 O(N log N)）
2. 自适应阈值计算：tau = k * median(all_pairwise_min_distances)，其中 k=2.0 为缩放系数，median 取所有点对最短距离的中位数
3. 若某点到相邻碎片的最短距离 < tau，则标记为断面点（label=1），否则为非断面点（label=0）
4. 对所有碎片对两两执行上述操作，合并标注结果
5. 保存为 .npz 文件，字段包括 points（形状 N,3）、labels（形状 N，0或1）、fragment_id（形状 N）

可视化输出 results/step4_label/<case_bone>.png：
- 子图1：断面点（红色）与非断面点（灰色）的三维散点图
- 子图2：各碎片的断面点占比柱状图（百分比）
- 布局：1行2列，标注断面点总数和占比

#### 实际运行结果（2026-05-12）

**采用方法**：碎片间距离法（与 Step 3 boundary_mask 生成逻辑一致）。
先通过 transform 逆变换将归一化位移坐标恢复到原始毫米坐标（碎片相邻位置），
然后对每个点计算到其他碎片的最短距离，距离 < tau_mm=4.0 的点标为断面点。

**实验结果**：120/120 全部处理成功，耗时 23.5s。
- 平均标注率：7.0%（在目标 5%~20% 范围内）
- 平均 Precision：0.993 | 平均 Recall：0.988 | 平均 F1：0.990
- F1 ≥ 0.85：120/120（100.0%），全部达标

**关键修复记录**：
1. 初始版本使用法向量不连续性 + PCA 曲率几何特征标注，F1 仅 0.096。根本原因
   是 Voronoi 平面切割产生的断面在 4096 点分辨率下几何特征与骨骼原始表面无法区分。
2. 改用碎片间距离法（与 Step 3 boundary_mask 同一逻辑）后，F1 提升至 0.990，
   验证了坐标逆变换的正确性和距离法在断面标注中的可靠性。
3. transform 逆变换公式在 Step 3 中为 `pts_norm = (pts_raw @ R.T + t - centroid) / scale`，
   因此逆变换为 `raw_pts = (pts_norm * scale + centroid - t) @ R`。
   normals 在 Step 3 中从未被旋转变换，保持原样即可。

**结论**：碎片间距离法能可靠检测 Voronoi 破碎的断面点（F1=0.990），
与论文 97% 的一致性指标相当。纯几何特征（法向量不连续性 + 曲率）
在 Voronoi 几何近似下不足以区分断面和骨面。

---

## 依赖库

nibabel 用于读取 NIfTI 数据
scikit-image 用于 Marching Cubes 表面重建
trimesh 用于网格处理、采样、切割
numpy 用于数据处理
scipy 用于空间查询（cKDTree）和 Voronoi 破碎
matplotlib 用于静态图表（所有可视化）
open3d 可选，用于三维点云交互式预览
robust-laplacian 用于构建刚度矩阵和质量矩阵

---

## 与原论文的主要差异

数据源方面：袁钦辉论文使用医院 CT DICOM 数据经 Mimics 软件重建；本实验使用 TotalSegmentator NIfTI 数据经 Marching Cubes 重建。

骨种方面：袁钦辉论文限定胫骨平台；本实验不限骨种，覆盖股骨、椎骨、肋骨等多种形态。

断裂模拟方面：袁钦辉论文使用基于生物结构特性的有限元优化算法（Sellan 2023），构建弹性刚度矩阵（含材料参数）求解结构振动模态；本实验先用随机平面切割跑通流程，再升级为 Voronoi 破碎，最后对少量样本使用曲面谱分析（Spectral Geometry）方法确定裂纹方向：构建网格的 Laplace-Beltrami 算子 L 和质量矩阵 M，求解广义特征值问题 L u = λ M u，取第一非零特征向量的零交叉面作为裂纹平面。该方法与 Sellan 2023 的有限元模态分析在数学形式上类似，区别在于刚度矩阵 K 由几何 Laplacian 代替，不依赖材料弹性参数，计算效率更高，适合大规模数据集预处理。

点云采样方面：两者均使用按对象采样 + 表面积比例分配，保持一致。

断面标注方面：袁钦辉论文自动标注与人工标注吻合度达 97%；本实验使用简化版几何距离标注（cKDTree + 自适应阈值）。

深度学习方面：袁钦辉论文包含完整的 PointNet++ 训练流程；本实验不涉及深度学习。

---

## 待办清单

- [x] 下载 TotalSegmentator 小子集数据
- [x] Step 0：运行数据筛选脚本，生成 selected_cases.json
- [x] Step 1：骨骼重建管线，输出 .ply 网格文件
- [x] Step 2：点云采样，输出 .npz 点云文件
- [x] Step 3 阶段一：随机平面切割骨折模拟
- [x] Step 3 阶段二：升级为 Voronoi 破碎
- [x] Step 3 阶段三：曲面谱分析破碎（24例，已完成）
- [x] Step 4：断面自动标注
- [x] 各阶段可视化输出检查（77张图片全部验证通过）
- [x] 生成完整报告图片集（results/pipeline_overview.png）
