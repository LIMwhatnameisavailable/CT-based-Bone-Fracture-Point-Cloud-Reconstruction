# Step 4 编写断面点标注脚本，注意以下要求：

目标：对破碎点云二值标注（断面点 = 1，非断面点 = 0），与 Step 3 boundary_mask 对比验证一致性。对应袁钦辉论文 §3.2.5。

输入：data/fractured/<case_bone>_voronoi.npz
输出：data/labeled/<case_bone>_labeled.npz
可视化：results/step4_label/

1. 核心概念：Step 3 的 boundary_mask 是碎片位移前在毫米坐标系下用固定阈值 tau=4.0mm 计算的，可作为 ground truth。Step 4 的 geometric_label 用纯几何特征（法向量不连续性 + 曲率）标注。所有距离计算必须在 points_raw（位移前毫米坐标）上进行，不可用位移后的 points。

2. 特征 1 — 法向量不连续性：对每个点找 k=16 个最近邻，计算邻域内法向量平均角度偏差。断面区域法向量差异大。用 cKDTree 实现，复杂度 O(N log N)。

3. 特征 2 — 局部曲率：用 PCA 估计最小特征值 / 特征值之和。断面区域曲率突变。N=4096 时约 0.5s。

4. 标注决策：双特征融合，法向量不连续性 OR 曲率超过各自百分位数阈值（默认 75%）的点标为断面点，取并集。

5. 输出字段：points (N,3)、labels (N,)、fragment_id (N,)、boundary_mask (N, bool)、normal_disc (N,)、curvature (N,)。

6. 可视化：每分组 1 张代表图，1x2 布局。左图 XZ 投影（灰 = 非断面，红 = 断面），右图混淆矩阵热力图（行 = boundary_mask，列 = geometric_label），下方标注 P/R/F1。

7. 验收标准：成功率 120/120，平均标注率 15%~35%，平均 F1 ≥ 0.5，F1 ≥ 0.6 占比 ≥ 50%，单 case < 2 分钟。

---

# 以下为第一次修改要求

当前使用法向量不连续性 + 曲率双特征方法，平均 F1 = 0.096，原因是 Voronoi 平面切割的断面与原始骨面在几何特征上无法区分。修复为碎片间距离法（与 Step 3 boundary_mask 同一逻辑）：

1. 删除 label_fracture_surface 函数，替换为 label_by_interdistance(raw_pts, frag_ids, tau_mm=4.0)：在原始坐标系中计算每个点到其他碎片的最短距离，距离 < tau_mm 标为断面点。

2. process_entry 中标注调用从 label_fracture_surface(nd, curv) 改为 label_by_interdistance(raw_pts, frag_ids, tau_mm=4.0)。nd 和 curv 仍计算保存供分析，不参与标注。

3. 验收标准更新：平均 F1 ≥ 0.85，平均 label rate 5%~20%，断面点集中在碎片边缘（视觉可验证）。
