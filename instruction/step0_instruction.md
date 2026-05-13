# Step 0 编写 step0_select_cases.py，注意以下要求：

1. 骨种不限定，扫描每个 case 下 segmentations/ 目录中所有实际存在的 .nii.gz 文件，不做白名单过滤。

2. 必须读取 meta.csv（分隔符为分号），将其信息与 case 对应，用于筛选和输出。

3. 骨骼按以下分组进行配额控制（每组最多 10 例），同一个 case 的同一骨骼分组只计一次：
   - femur：femur_left, femur_right
   - humerus：humerus_left, humerus_right
   - vertebrae_lumbar：vertebrae_L1 到 vertebrae_L5
   - vertebrae_thoracic：vertebrae_T1 到 vertebrae_T12
   - vertebrae_cervical：vertebrae_C1 到 vertebrae_C7
   - rib：rib_left_1 到 rib_left_12，rib_right_1 到 rib_right_12
   - pelvis：hip_left, hip_right, sacrum
   - skull：skull
   - clavicula：clavicula_left, clavicula_right
   - scapula：scapula_left, scapula_right
   - sternum：sternum
   - 其他骨骼（不在上述分组中的）：归入 other 组，最多 10 例

4. 每个分组内优先选：no_pathology 的 case 优先，其次 trauma，其次其他；同等条件下按 mask 体素数量降序排列（体素数 = mask 非零体素数量，用 nibabel 读取计算，不要加载 ct.nii.gz，只读 mask）。

5. selected_cases.json（输出至 results/step0_dataset_overview/）每项包含字段：case, bone（具体骨骼名如 femur_left）, bone_group, age, gender, pathology, study_type, voxel_count, volume_cm3（体素数乘以 spacing 三个分量之积再除以 1000）。

6. 可视化输出两张图，均遵循 CLAUDE.md 绘图规范（Times New Roman，标题 >= 16，轴标签 >= 14，图例 >= 11，刻度 >= 11，每行最多 2 列，隐藏上右边框，无图例边框，dpi = 300）：

   第一张图 results/step0_dataset_overview.png，1 行 2 列：
   - 左图：各骨骼分组的选中 case 数量水平柱状图，横轴为数量，纵轴为骨骼分组名称
   - 右图：选中 case 的年龄分布直方图，bins = 10

   第二张图 results/step0_pathology_dist.png，1 行 2 列：
   - 左图：选中 case 的 pathology 分布饼图（no_pathology / trauma / tumor / other）
   - 右图：选中 case 的骨骼体积分布箱线图，按骨骼分组分组，横轴为分组名，纵轴为体积 cm3，x 轴标签旋转 45 度

7. 脚本末尾保存统计摘要至独立 txt 文件：总选中条目数、各分组数量、平均年龄、pathology 分布。

8. 支持断点续跑：如果 results/step0_dataset_overview/selected_cases.json 已存在，询问是否重新生成。

---

# 以下为第一次修改要求

针对 step0_select_cases.py 的以下问题进行修复：

1. 修复 pathology 多样性问题：在 select_cases 函数中，每个骨骼分组内的配额分配改为：优先选 no_pathology，但最多占该组配额的 70%（即 MAX_PER_GROUP = 10 时最多 7 例）；剩余配额优先填入 trauma，再填 tumor，最后填 other。如果某组 no_pathology 不足 7 例，则用其实际数量，剩余配额继续按优先级填充。实现方式：对每个分组，先按 pathology 类别分桶，再按体素数降序在各桶内取样，最后合并。

2. 删除 plot_overview 函数中的无效代码：ax2 = axes[0] if False else axes[1]。只保留后面的 ax2 = axes[1]。

3. 删除 main 函数内部的 import os as _os，直接使用顶部已导入的 os.cpu_count()。

4. 将 convert_native 函数从 main() 内部移到模块顶层（与其他函数并列）。

5. 可视化修复：
   - step0_dataset_overview.png：将图像整体高度从 5 增加到 6，增大左图的行间距（barh 的 height 从 0.6 改为 0.5，这样条间空隙更大）
   - step0_pathology_dist.png：饼图如果只有一种类别，在标题下方添加文字注释说明 "All selected cases are no_pathology; trauma/tumor cases exist in dataset but were deprioritized"，字号 11，颜色灰色

重新运行后，预期 pathology 饼图应出现至少 2 种颜色。

# 以下为第二次修改要求

完成以下两项收尾工作：

1. 更新 CLAUDE.md 中 Step 0 的待办状态，将 [ ] Step 0 改为 [x] Step 0。

2. 在 CLAUDE.md 的 Step 0 章节末尾添加 "实际运行结果" 小节。
