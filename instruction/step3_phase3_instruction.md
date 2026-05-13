# Step 3 Phase 3 编写模态分析物理破碎脚本，注意以下要求：

当前 Phase 2（Voronoi 破碎）使用几何切割，裂纹方向随机，与真实骨折物理规律不符。Phase 3 对少量样本（每分组 2 例，共 24 例）使用模态分析确定裂纹方向，与 Phase 2 形成对比。

1. 核心算法：从 .ply 网格用 robust-laplacian 构建刚度矩阵 K 和质量矩阵 M，求解广义特征值问题 K u = λ M u（scipy.sparse.linalg.eigsh，shift-invert 模式，sigma=1e-6），取前 6 个最小非零特征向量。第一特征向量 u1 对应最易断裂的振动模态。

2. 提取裂纹平面：将 u1 reshape 为 (n_vertices, 3)，投影到主方向得标量场，零交叉位置为裂纹区域。裂纹平面法向量 = u1 主方向，位置限定在质心 ±20% 范围内。切割后断面施加高斯噪声扰动（std=0.5mm）。

3. 后续随机位移、boundary_mask、npz 保存逻辑与 Phase 2 完全一致。输出 data/fractured/<case_bone>_physical.npz。

4. 输出两类图至 results/step3_fracture/phase3/：每个样本破碎结果图 + Voronoi vs Modal 对比图。

5. 样本选择：每骨骼分组取体积最大的 2 例，共 24 例。每个样本固定 2 块碎片。脚本为新建 code/step3_fracture_phase3.py。

6. 验收标准：24/24 处理成功；长骨裂纹与主轴夹角 < 45°；对比图中 Modal 与 Voronoi 形态差异肉眼可见；单 case < 5 分钟。

---

# 以下为第一次修改要求

1. 对比图坐标轴 bug：plot_comparison 函数中 all_z 错误使用了 [:, 1]（Y 轴），应改为 [:, 2]（Z 轴）。修复后重新生成 24 个对比图，不需要重新跑模态分析。

2. CLAUDE.md 数字更正：skull 角度应为 85.2° ± 4.3°，vertebrae_cervical 应为 85.8° ± 0.3°。补充说明 skull 和 vertebrae_cervical 接近 90° 为物理合理结果（冠状面切割），验收标准 < 45° 仅适用于长骨。other 分组包含软组织样本，角度仅供参考。
