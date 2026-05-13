# Step 3 编写破碎模拟脚本，注意以下要求：

## 阶段一 — 随机平面切割

编写 code/step3_fracture_phase1.py。

核心流程：data/meshes/<case_bone>.ply → 随机平面切割 → 碎片点云采样 → 随机位移 → data/fractured/<case_bone>_frac.npz

1. 输入与断点续跑：读取 selected_cases.json 组装 (case, bone) 列表；扫描 data/fractured/ 已有 .npz，跳过已完成条目。

2. 随机平面切割：使用 trimesh.intersections.slice_mesh_plane，cap=True 自动封口断面。随机决定 n_cuts ∈ {1, 2, 3}，最终碎片数 2~4 块。切割平面法向量随机生成，切割位置在投影值 20%~80% 分位数范围内随机采样。对当前碎片列表中每块网格执行切割。顶点数 < 50 的退化碎片丢弃，其采样点按质心距离分配给最近的有效碎片。

3. 碎片后处理：对每块有效碎片按表面积比例分配采样点，单块最少 30 点，总点数 N = 4096。随机位移在毫米坐标系下施加（旋转 ±30°，平移 ±20mm）。归一化使用 Step 2 .npz 的 centroid 和 scale，不得重新计算。

4. 输出 .npz 字段：points (N,3)、fragment_id (N,)、normals (N,3)、points_raw (N,3)、centroid (3,)、scale (scalar)、transforms（每碎片 R 和 t）。

5. 可视化：每骨骼分组选 1 张代表图，1x2 布局（左：完整点云灰色，右：破碎后 tab10 着色），输出至 results/step3_fracture/phase1/。

---

# 以下为第一次修改要求（Phase 1 碎片数 Bug）

clavicula、femur 等骨骼产生了 8 块碎片，超出 2~4 块上限。根本原因是每轮对所有碎片各切一刀，碎片数按 2^n 指数增长。修复为每次只切体积最大的那块碎片。修复后删除所有 _frac.npz 重新生成。

---

# 以下为第二次修改要求：实现 Voronoi 破碎（Phase 2）

新建独立脚本 code/step3_fracture_phase2.py。输出命名：点云 data/fractured/<case_bone>_voronoi.npz，可视化 results/step3_fracture/phase2/。

1. Voronoi 种子生成：用 SVD 求网格顶点主成分，沿第一主轴等分位布种子，垂直方向加小幅扰动（jitter_scale = short_len * 0.10）。n_seeds ∈ [2, 4]，使用 np.random.default_rng(seed) 保证可复现。

2. 面片归属分配：用 cKDTree 将每个面片分配给最近种子。为每个种子构造子网格，保留最大连通分量；顶点数 < 50 的退化碎片直接丢弃。

3. 小碎片合并：体积 < 总体积 5% 的碎片合并到质心最近的最大碎片 faces 中，最终碎片数 2~4 块。

4. 断面凹凸扰动：调用顺序为切割 → 采样 → 扰动（原位、未位移前）→ 刚体变换 → 归一化。扰动沿法向量单侧施加（uniform(0, 0.3mm)），返回 boundary_mask。

5. 后处理与输出：每碎片旋转 ±30°、平移 ±20mm，存储为 (n_frags, 4, 4) 齐次矩阵。归一化复用 Step 2 centroid/scale。输出字段较阶段一新增 boundary_mask。

6. 可视化：1x2 布局（左：完整点云灰色，右：碎片 tab10 着色），标注碎片数/最大位移/断面点占比。每分组 1 张共 12 张。

7. 性能约束：单 case < 15 分钟，禁止 O(N²)，必须用 cKDTree，支持断点续跑。

# 以下为第三次修改要求（Phase 2 断面点占比修复）

当前 perturb_fracture_surface 中 tau = 2.0 * median(dists) 在未位移场景失效，几乎所有点被误判为断面点。修复为基于全局采样密度的阈值 tau = alpha * d_nn，新增 estimate_sampling_density 函数。验收标准：断面点平均占比 5%~25%，不同骨骼有合理差异。

# 以下为第四次修改要求（Phase 2 第二轮修复）

1. 修复 merge_small_fragments KeyError：局部索引映射回全局索引。

2. Boundary 占比偏低（1.1%~1.8%）：改用物理阈值 TAU_MM = 4.0mm。

3. 碎片坐标范围失控：平移范围改为自适应（包围盒对角线 20%，上限 20mm）。

4. 阶段一/二对比图坐标轴不统一：统一设置 xlim/zlim。

5. 新增交互式 HTML 可视化：新建 step3_visualize_3d.py，用 plotly 生成，输出至 phase2_interactive/。

# 以下为第五次修改要求（Phase 2 第三轮修复）

1. HTML 可视化恢复全量 4096 点（取消下采样），调整 marker 参数。

2. 扁平骨骼（椎体等）碎片重叠：增大垂直扰动幅度，自适应 jitter_scale = short_len * (0.10 + 0.20 * (1 - flatness))。

3. 小骨骼碎片飞出视野：用整体点云包围盒计算 trans_range（global_bbox_diag * 0.15），取代碎片自身包围盒。
