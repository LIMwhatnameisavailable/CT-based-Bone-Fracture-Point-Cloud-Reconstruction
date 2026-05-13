# Step 1 编写 step1_reconstruct.py，注意以下要求：

1. Marching Cubes 的输入必须是 mask 本身（二值 0/1 数组），不是 mask 乘以 CT。CT 数据仅用于可视化左图（CT 切片灰度显示 + mask 轮廓叠加），不参与网格重建计算。marching_cubes(mask_data, level=0.5) 直接对 mask 运行。

2. Bounding box 裁剪：对 mask 非零区域做 bounding box 裁剪，各方向 padding = 2 个体素，在裁剪后的小体积上运行 marching_cubes。顶点坐标恢复公式：verts_mm = (verts_in_crop + min_coords - pad) * spacing，其中 min_coords 是裁剪前 mask 非零区域的最小坐标（体素单位）。

3. 不使用进程并行，改为顺序处理。按 case 分组，同一个 case 的 ct.nii.gz 只加载一次，该 case 的所有骨骼复用同一个 CT 数组。处理完一个 case 的所有骨骼后再加载下一个 case 的 CT。

4. Spacing 一致性检查：用 np.allclose(ct_spacing, mask_spacing, atol=0.01) 检查，不一致时打印 WARNING 但继续执行，不要用 assert。

5. 断点续跑：脚本启动时扫描 data/meshes/ 已有的 .ply 文件，跳过已完成的条目。

6. 错误处理：marching_cubes 失败（ValueError/空网格）、最大连通分量为空，均记录到 data/meshes/error_log.txt，不中断整体流程。

7. 可视化：不对全部 120 条目生成图片，改为每个骨骼分组各取 1 张代表性图片（共最多 13 张），选取该分组中 volume_cm3 最大的条目作为代表。输出路径：results/step1_reconstruction/<bone_group>_<case>_<bone>.png。每张图 1 行 2 列：
   - 左图：CT 轴向中间切片（灰度，cmap='gray'），叠加 mask 轮廓（plt.contour，红色，levels=[0.5]）；标注切片编号和骨骼名称
   - 右图：重建网格的三视图（前视/侧视/顶视）用 matplotlib 的 mpl_toolkits.mplot3d 绘制，标注包围盒尺寸（长 x 宽 x 高，单位 mm，从 mesh.bounding_box.extents 获取）
   遵循 CLAUDE.md 绘图规范（Times New Roman，标题 >= 16，轴标签 >= 14，dpi = 300）。

8. 脚本末尾打印统计摘要：成功/失败/跳过数量，以及成功网格的顶点数/面片数均值。

9. 读取 results/step0_dataset_overview/selected_cases.json，按 case 字段分组后处理。输出 .ply 文件命名格式：data/meshes/<case>_<bone>.ply

---

# 以下为第一次修改要求

彻底重写 plot_reconstruction 函数，要求如下：

1. 布局：整张图 figsize=(16, 5)，使用 GridSpec 分为 1 行 2 列：
   - 左列（width_ratio = 1）：CT 切片 + mask 轮廓，占整图左半部分
   - 右列（width_ratio = 2）：三视图区域，在右列内部再用 GridSpecFromSubplotSpec 分为 1 行 3 列
   注意：右侧三个小子图在视觉上属于右列，整体布局仍然是 "1 行 2 列" 的大框架。

2. 字体与风格，在函数开头调用 setup_matplotlib()，并额外设置：
   - 所有 ax 隐藏上边框和右边框
   - 所有字号：标题 >= 16，轴标签 >= 14，刻度 >= 11

3. 左图（CT 切片）：
   - 用 ax.imshow(ct_slice, cmap='gray', aspect='auto') 显示 CT 切片
   - 用 ax.contour(mask_slice, levels=[0.5], colors='#E64B35', linewidths=1.2) 叠加 mask 轮廓（红色，线宽 1.2）
   - 标题三行：第一行骨骼分组名，第二行 case_bone，第三行 Slice N，fontsize = 16
   - xlabel = 'X (pixels)'，ylabel = 'Y (pixels)'，fontsize = 14
   - 刻度 fontsize = 11

4. 右侧三视图，放弃 mpl_toolkits.mplot3d，改用 2D 投影：取 mesh.vertices（形状 N x 3，列为 x, y, z mm 坐标），分别做以下三个投影：
   - Front view（前视图）：横轴 = x，纵轴 = z，标题 = 'Front (X-Z)'
   - Side view（侧视图）：横轴 = y，纵轴 = z，标题 = 'Side (Y-Z)'
   - Top view（顶视图）：横轴 = x，纵轴 = y，标题 = 'Top (X-Y)'
   每个投影用 ax.scatter(h_coords, v_coords, s=0.3, c='steelblue', alpha=0.4, rasterized=True) 绘制点云投影。每个子图：隐藏上右边框；xlabel 和 ylabel 分别标注对应轴名称（如 'X (mm)'），fontsize = 14；标题 fontsize = 14；刻度 fontsize = 11；保持等比例：ax.set_aspect('equal')

5. BBox 标注：在右侧三视图区域的第一个子图（Front）下方用 ax.text 标注：'BBox: L x W x H mm'（从 mesh.bounding_box.extents 获取），用 ax.text(0.5, -0.18, ..., transform=ax.transAxes, ha='center', fontsize=11, color='#555555')

6. 修复 main() 中的变量名遮蔽 bug：在 CT 加载失败的 except 块里，循环变量应为 for bone_entry in entries，不能用 e（会覆盖 except Exception as e 的变量）。

7. 图片保存：plt.savefig(output_path, dpi=300, bbox_inches='tight')，关闭后释放内存 plt.close(fig)。

# 以下为第二次修改要求

对 step1_reconstruct.py 的 plot_reconstruction 函数做以下两处修改，然后重新生成所有可视化图片：

1. BBox 标注位置修复：将 BBox 标注从 Front 子图的 ax.text 改为放在整个右列区域的底部。在三个子图循环结束后，在 Front 子图（保存为 ax_front 变量）下方用更大的偏移：ax_front.text(0.5, -0.25, f'BBox: {bbox_label}', transform=ax_front.transAxes, ha='center', fontsize=11, color='#555555')。同时将 gs_right 的 wspace 从 0.3 改为 0.35，给底部标注留出更多空间。

2. 散点大小自适应：在三视图循环内，scatter 调用之前，根据顶点数动态计算点大小：n_verts = len(verts)；pt_size = float(np.clip(5000.0 / n_verts, 0.3, 3.0))；pt_alpha = float(np.clip(0.8 - n_verts / 150000, 0.3, 0.8))。用 s=pt_size 和 alpha=pt_alpha 替换原来的固定值。

重新运行可视化部分（可以只跑第 4 步，跳过重建，因为 .ply 已存在）。
