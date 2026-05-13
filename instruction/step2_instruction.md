# Step 2 编写 step2_sample_pointcloud.py，要求如下：

1. 输入输出路径：
   - 输入：results/step0_dataset_overview/selected_cases.json 和 data/meshes/<case>_<bone>.ply
   - 输出：data/pointclouds/<case>_<bone>.npz
   - 可视化：results/step2_pointcloud/<bone_group>_<case>_<bone>.png

2. 采样逻辑：
   - 固定采样点数 N = 4096
   - 使用 trimesh.sample.sample_surface(mesh, N)，返回 (points_raw, face_indices)
   - 法向量：mesh.face_normals[face_indices]，已是单位向量，无需额外归一化
   - 归一化：centroid = points_raw.mean(axis=0)；centered = points_raw - centroid；scale = np.max(np.linalg.norm(centered, axis=1))；points = centered / scale
   - 跳过顶点数 < 100 的网格，记录到 error log

3. npz 保存字段（必须全部保存，后续步骤依赖）：
   - points：归一化坐标，形状 (4096, 3)，float32
   - normals：法向量，形状 (4096, 3)，float32
   - points_raw：归一化前 mm 坐标，形状 (4096, 3)，float32
   - centroid：质心，形状 (3,)，float32
   - scale：缩放因子（最大半径），标量 float32

4. 断点续跑：扫描 data/pointclouds/ 已有 .npz，跳过已完成条目。

5. 可视化（每骨骼分组选 volume_cm3 最大的代表，共约 13 张）：布局 figsize=(12, 5)，1 行 2 列，遵循 CLAUDE.md 绘图规范。

   左图（前视图散点）：
   - 横轴 = points[:, 0] (X)，纵轴 = points[:, 2] (Z)，颜色按 points[:, 2] 的值用 viridis 渐变
   - ax.scatter(points[:, 0], points[:, 2], c=points[:, 2], cmap='viridis', s=1.0, alpha=0.6, rasterized=True)
   - colorbar 用 fig.colorbar(sc, ax=ax_left, shrink=0.8, label='Z (normalized)')，label fontsize = 11
   - 标题 'Front View (X-Z)'，fontsize = 14
   - xlabel = 'X (normalized)'，ylabel = 'Z (normalized)'，fontsize = 14
   - ax.set_aspect('equal')，隐藏上右边框

   右图（XY 平面密度热力图）：
   - ax.hexbin(points[:, 0], points[:, 1], gridsize=30, cmap='Blues', extent=(-1.1, 1.1, -1.1, 1.1))
   - colorbar 用 fig.colorbar(hb, ax=ax_right, shrink=0.8, label='Count')，label fontsize = 11
   - 标题 'Top View Density (X-Y)'，fontsize = 14
   - xlabel = 'X (normalized)'，ylabel = 'Y (normalized)'，fontsize = 14
   - ax.set_aspect('equal')，隐藏上右边框

   整张图标题（fig.suptitle）：'{bone_group} | {case}_{bone} | N=4096'，fontsize = 16，y=1.02
   plt.tight_layout(pad=2.0)，保存 dpi = 300，bbox_inches='tight'

6. 脚本末尾打印统计摘要：成功/失败/跳过数量，采样点数均值（应全为 4096），法向量模长均值（应接近 1.0）；计入 step2_summary.txt。

7. 读取 results/step0_dataset_overview/selected_cases.json，按 case 字段分组，同一 case 的 .ply 文件顺序处理（不需要复用 CT，Step 2 不涉及 CT）。

---

# 以下为第一次修改要求

存在的问题：

1. 法向量未归一化存储。trimesh.face_normals 通常已归一化，但经过 Laplacian 平滑后的网格不保证法向量模长恰好为 1.0。建议显式归一化。

# 以下为第二次修改要求

修复 Step 1 网格质量问题并重新采样受影响的 case。vertebrae_lumbar 和 vertebrae_thoracic 这两类骨骼的前视图存在明显的点云空洞和条带状空白区域。经分析，根本原因是 Step 1 输出的网格存在孔洞或退化三角面，导致 trimesh.sample_surface 按面积加权采样时跳过了这些区域。注意：不需要增加采样点数，4096 已经足够，问题在网格质量。

请按以下步骤修复：

1. 在 step1 的网格生成逻辑中（或 step2 加载 .ply 之后、采样之前），加入如下网格修复函数，并在每个 mesh 加载后调用它：

   ```python
   def repair_mesh(mesh):
       components = mesh.split(only_watertight=False)
       if len(components) > 1:
           mesh = max(components, key=lambda m: len(m.faces))
       trimesh.repair.fill_holes(mesh)
       trimesh.repair.fix_normals(mesh)
       mask = mesh.area_faces > 1e-10
       mesh.update_faces(mask)
       return mesh
   ```

2. 仅对以下两类骨骼重新生成网格并重新采样，其余骨骼已有合格结果，不要重跑：
   - vertebrae_lumbar（所有 case）
   - vertebrae_thoracic（所有 case）
   重跑时请先删除这两类骨骼对应的 .ply 和 .npz 文件，让断点续跑机制自动识别需要重新处理的条目。

3. 重新生成这两类骨骼的可视化图，确认前视图中不再出现大面积空洞或条带状空白。对比修复前后的图，在输出日志中注明修复效果。

skull 的结果正常，不需要处理。
