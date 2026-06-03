# 可视化约束 · 胫骨平台骨折术前规划项目

> 版本：v1.0 | 创建人：LIM | 创建日期：2026-06-03 | 适用范围：所有项目绘图输出（代码图表 + AI重绘 + 流程示意图）
> 本文档同时服务于三类使用者：1.编写可视化代码的组员；2.调用 AI Agent 自动生成图表的脚本；3.使用 Midjourney / DALL·E / Stable Diffusion 等工具重绘示意图的组员。
> **所有图表在提交前必须对照本文档逐项自查。**

---

## 目录

1. [字体系统](#1-字体系统)
2. [配色系统](#2-配色系统)
3. [图幅与布局](#3-图幅与布局)
4. [坐标轴与刻度](#4-坐标轴与刻度)
5. [线条与标记](#5-线条与标记)
6. [文字层级与标注](#6-文字层级与标注)
7. [图例与色条](#7-图例与色条)
8. [三维点云与骨骼渲染专项规范](#8-三维点云与骨骼渲染专项规范)
9. [流程图与架构图专项规范](#9-流程图与架构图专项规范)
10. [AI重绘通用Prompt基准](#10-ai重绘通用prompt基准)
11. [输出格式与文件命名](#11-输出格式与文件命名)
12. [rcParams 全局模板](#12-rcparams-全局模板)
13. [图表改进任务清单（ABCDE分类）](#13-图表改进任务清单abcde分类)
14. [禁止事项](#14-禁止事项)

---

## 1. 字体系统

### 1.1 字体优先级

所有图表字体统一遵循以下优先级链，按序回退：

```
Times New Roman → Georgia → DejaVu Serif → serif（系统默认衬线）
```

数学公式字体：`stix`（与 Times New Roman 视觉匹配，符合 LaTeX 出版标准）

**AI重绘场景补充说明：** 在使用 Figma / Illustrator / 在线绘图工具重绘流程图和示意图时，文字字体统一使用 **Inter**（无衬线，现代感强，适合技术架构图）或 **Source Han Sans（思源黑体）**（中文标注场景）。代码生成的数据图表仍使用 Times New Roman 体系。两类图表字体体系不混用。

### 1.2 字体使用规则

| 使用场景 | 字体风格 | 说明 |
|---|---|---|
| 图题（Figure Title） | Times New Roman，正体，Bold | 最高层级，最大字号 |
| 坐标轴标签（Axis Label） | Times New Roman，斜体（变量）/ 正体（单位） | 物理量/变量名斜体，单位正体 |
| 刻度标签（Tick Label） | Times New Roman，正体 | 数字一律正体 |
| 图例文字（Legend） | Times New Roman，正体 | 组别名称正体，变量名斜体 |
| 标注文字（Annotation） | Times New Roman，正体 | 说明性文字正体 |
| 数学符号/变量 | Times New Roman，*斜体* | 所有数学变量、物理量符号 |
| 单位 | Times New Roman，正体，括号包裹 | 例：`°`、`mm`、`MAE` |
| 子图标签（a, b, c） | Times New Roman，Bold | 左上角，格式 **(a)** |
| 流程图节点文字 | Inter / 思源黑体，正体 | 中文标注场景用思源黑体 |
| 架构图模块标题 | Inter，Bold | 区块标题加粗 |

### 1.3 字号层级

| 层级 | 场景 | 字号（pt） |
|---|---|---|
| L1 | 图题 / 大标题 | 14 |
| L2 | 坐标轴标签 / 流程图模块标题 | 12 |
| L3 | 图例 / 子图标题 / 流程图节点文字 | 11 |
| L4 | 刻度标签 | 10 |
| L5 | 标注文字 / 说明 / 箭头标注 | 9 |
| L6 | 角标 / 次要注释 / 数学下标 | 8 |

---

## 2. 配色系统

### 2.1 项目核心语义配色

以下配色在本项目所有图表中保持语义一致，**不得随意更换颜色与语义的对应关系**：

```python
COLORS = {
    # ── 骨折块标识（点云/网格渲染核心配色）──────────────────
    "piece_main":     "#4A90D9",   # 主骨折块 / piece_0（天蓝，清晰主体）
    "piece_alt":      "#E64B35",   # 次骨折块 / piece_1（朱红，强对比）
    "piece_third":    "#2A9D8F",   # 第三骨折块 / piece_2（青绿）
    "piece_fourth":   "#E9C46A",   # 第四骨折块（琥珀黄）
    "piece_fifth":    "#6A4C93",   # 第五骨折块（深紫）
    "bone_body":      "#7EC8A4",   # 骨干主体点云（浅绿，与骨折块区分）

    # ── 断面标识 ──────────────────────────────────────────
    "fracture_face":  "#E64B35",   # 断裂面点云高亮（朱红）
    "fracture_gt":    "#E64B35",   # Ground Truth 断面（朱红）
    "fracture_pred":  "#FF6B35",   # Prediction 断面（橙红，与GT区分）

    # ── 算法模块配色（流程图/架构图）────────────────────────
    "module_extract": "#D4E8F7",   # 特征提取模块背景（浅天蓝）
    "module_cls":     "#FFF3CD",   # 分类模块背景（浅琥珀）
    "module_match":   "#D6EAF8",   # 匹配模块背景（中天蓝）
    "module_reg":     "#D5F5E3",   # 配准模块背景（浅绿）
    "module_border":  "#2C7BB6",   # 模块边框（深蓝）
    "module_arrow":   "#444444",   # 流程箭头（深灰）

    # ── 损失函数/创新模块强调色 ──────────────────────────────
    "rig_loss":       "#E64B35",   # Rig Loss 标注（朱红，强调）
    "cfr_module":     "#2C7BB6",   # CFR 模块标注（深蓝，强调）

    # ── 骨骼三维渲染 ──────────────────────────────────────
    "bone_render":    "#D4C5A9",   # 骨骼三维渲染基础色（米棕，CT重建感）
    "bone_highlight": "#F0C060",   # 骨骼高亮区域（金黄，术前规划标注）
    "implant":        "#B0B0B0",   # 钢板/植入物（中灰，金属感）
    "screw_path":     "#E64B35",   # 螺钉路径规划线（朱红）
    "guide_plate":    "#A8D8EA",   # 钻孔导板（浅蓝，半透明感）

    # ── 通用背景与文字 ────────────────────────────────────
    "bg_panel":       "#FAFAFA",   # 子图背景（近白）
    "bg_dark":        "#1E2A3A",   # 深色UI背景（交互界面截图参考色）
    "text_primary":   "#333333",   # 主要文字（深灰，非纯黑）
    "text_secondary": "#666666",   # 次要文字（中灰）
    "axis_line":      "#444444",   # 坐标轴线（深灰）
    "grid_line":      "#E0E0E0",   # 网格线（极浅灰）

    # ── 状态/结果标注 ─────────────────────────────────────
    "success":        "#2A9D8F",   # 复位成功 / 达标（青绿）
    "warning":        "#E9C46A",   # 临界值 / 警告（琥珀黄）
    "error":          "#E64B35",   # 超出阈值 / 失败（朱红）
    "neutral":        "#D3D3D3",   # 无统计意义 / 背景（浅灰）
}
```

### 2.2 骨折块多碎片配色扩展顺序

当骨折碎片超过 3 块时，按以下顺序依次取色：

```
#4A90D9 → #E64B35 → #2A9D8F → #E9C46A → #6A4C93 → #FF6B35 → #8B5E3C
```

**注意：** 绿色系（`#7EC8A4`）专用于骨干主体，不参与碎片编号配色序列，避免与 piece_third 混淆。

### 2.3 连续型色图（Colormap）规范

| 使用场景 | 推荐色图 | 说明 |
|---|---|---|
| 点云深度着色 | `viridis` | 色盲友好，感知均匀 |
| 匹配亲和度矩阵热图 | `Blues` | 单调递增，浅→深对应低→高匹配概率 |
| 误差分布图（有符号） | `RdBu_r` | 负值蓝，正值红，零点白 |
| 复位误差热图 | `YlOrRd` | 黄→橙→红，直觉对应误差由小到大 |
| CT 图像显示 | `gray` | 标准灰阶，不得使用彩色 |
| 骨折分型示意（线稿填充） | 手动指定 `#E64B35` | 见 §2.1 fracture_face |

**禁止使用**：`jet`、`rainbow`、`hsv`（感知不均匀，不适合学术出版）

---

## 3. 图幅与布局

### 3.1 标准图幅尺寸

| 图表类型 | 推荐尺寸（英寸） | 说明 |
|---|---|---|
| 单图（单列） | `(6, 5)` | 期刊单栏标准 |
| 单图（双列） | `(10, 6)` | 期刊双栏或宽幅展示 |
| 1×2 子图（对比图） | `(12, 5)` | 复位前后对比、GT vs Pred |
| 1×4 子图（消融对比） | `(18, 5)` | 四组消融实验并排 |
| 2×2 子图 | `(10, 8)` | 标准四格布局 |
| 2×3 子图 | `(14, 8)` | 分割+复位联合展示 |
| 流程图 / 架构图 | `(14, 8)` | 算法框架、模块结构 |
| 骨折分型示意图 | `(14, 5)` | 四分型并排 |
| 数据统计折线图 | `(8, 5)` | 单条或多条折线 |

### 3.2 间距规范

```python
plt.subplots_adjust(
    left=0.10,
    right=0.95,
    top=0.92,
    bottom=0.12,
    wspace=0.30,
    hspace=0.35,
)
# 推荐替代方案：
fig, axes = plt.subplots(..., layout='constrained')
```

### 3.3 DPI 规范

| 输出用途 | DPI |
|---|---|
| 屏幕预览 / 草稿 | 100 |
| 正式输出 PNG | 300 |
| 出版级 PNG | *600* |
| SVG | 矢量，DPI 参数设为 300（元数据用） |

---

## 4. 坐标轴与刻度

### 4.1 坐标轴线条

```python
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1.0)
    ax.spines[spine].set_color('#444444')
```

### 4.2 刻度规范

```python
ax.tick_params(
    axis='both', which='major',
    direction='out', length=4, width=1.0,
    labelsize=10, colors='#333333', pad=4,
)
ax.tick_params(
    axis='both', which='minor',
    direction='out', length=2, width=0.8,
)
```

### 4.3 网格线

```python
# 默认关闭，折线图/柱状图按需启用
ax.grid(True, which='major', linestyle='--', linewidth=0.5,
        color='#E0E0E0', alpha=0.7, zorder=0)
ax.set_axisbelow(True)
```

### 4.4 坐标轴标签格式

```python
# 误差指标轴标签示例
ax.set_ylabel(r'$\it{Rotation\ MAE}$ (°)', fontsize=12)
ax.set_ylabel(r'$\it{Translation\ MAE}$ (mm)', fontsize=12)
ax.set_xlabel(r'Model Configuration', fontsize=12)

# 临床阈值参考线
ax.axhline(y=5.0, color='#E9C46A', linestyle='--',
           linewidth=1.2, label='Clinical threshold (5°)')
ax.axhline(y=2.0, color='#E9C46A', linestyle='--',
           linewidth=1.2, label='Clinical threshold (2 mm)')
```

---

## 5. 线条与标记

### 5.1 线条规范

| 用途 | 线宽 | 线型 | 颜色 |
|---|---|---|---|
| 主数据线 | 2.0 | `-` | 语义配色 |
| 次要数据线 | 1.5 | `-` | 语义配色 |
| 临床阈值参考线 | 1.2 | `--` | `#E9C46A` |
| 误差范围边界 | 0.8 | `-` | 同数据线色，alpha=0.3填充 |
| 标注引线 | 0.8 | `-` | `#666666` |
| 流程图连接箭头 | 1.5 | `-` | `#444444` |
| 模块边框 | 1.5 | `-` | `#2C7BB6` |

### 5.2 散点标记规范

| 用途 | 标记形状 | 大小 | 颜色 |
|---|---|---|---|
| 主数据点 | `o`（圆） | 50 | `#4A90D9` |
| 对比方法点 | `s`（方） | 50 | `#E64B35` |
| 本文算法点 | `*`（星） | 80 | `#E64B35`，加粗边框 |
| 均值/最优标记 | `D`（菱形） | 80 | 突出显示 |
| 散点透明度 | — | — | `alpha=0.75` |
| 散点边框 | — | — | `edgecolors='white'`, `linewidths=0.5` |

---

## 6. 文字层级与标注

### 6.1 图题

```python
ax.set_title('Figure Title Here',
             fontsize=14, fontweight='bold',
             color='#333333', pad=10)

fig.suptitle('Overall Figure Title',
             fontsize=14, fontweight='bold',
             color='#333333', y=1.01)
```

### 6.2 子图标签

```python
ax.text(-0.10, 1.05, '(a)',
        transform=ax.transAxes,
        fontsize=12, fontweight='bold',
        va='top', ha='left', color='#333333')
```

### 6.3 临床阈值标注

```python
# 在误差对比图中标注临床标准线
ax.axhline(y=2.0, color='#E9C46A', linestyle='--', linewidth=1.2, zorder=3)
ax.text(x_max * 0.98, 2.15, 'Clinical threshold (2 mm)',
        ha='right', va='bottom', fontsize=9,
        color='#E9C46A', fontstyle='italic')
```

### 6.4 数据标注

```python
# 柱状图顶部数值
ax.text(x, y + 0.05, f'{value:.2f}°',
        ha='center', va='bottom',
        fontsize=9, color='#333333')

# 最优结果高亮标注
ax.annotate('Ours (Best)',
            xy=(x_ours, y_ours),
            xytext=(x_ours + 0.3, y_ours + 0.5),
            fontsize=9, color='#E64B35',
            arrowprops=dict(arrowstyle='->', color='#E64B35', lw=1.0))
```

---

## 7. 图例与色条

### 7.1 图例规范

```python
legend = ax.legend(
    loc='best',
    frameon=True,
    framealpha=0.9,
    edgecolor='#CCCCCC',
    fancybox=False,
    fontsize=11,
    handlelength=2.0,
    handletextpad=0.5,
    borderpad=0.6,
    labelspacing=0.4,
)
```

### 7.2 色条规范

```python
cbar = plt.colorbar(im, ax=ax,
                    fraction=0.046, pad=0.04, shrink=0.85)
cbar.ax.tick_params(labelsize=10, direction='out', length=3)
cbar.set_label(r'$\it{Affinity\ Score}$',
               fontsize=11, labelpad=8)
cbar.outline.set_linewidth(0.8)
```

---

## 8. 三维点云与骨骼渲染专项规范

本节为本项目特有规范，适用于所有点云可视化代码（Open3D / PyVista / matplotlib 3D）。

### 8.1 点云渲染规范

```python
# 推荐使用 Open3D 或 PyVista，禁止使用 matplotlib ax.scatter 裸输出
# Open3D 示例
import open3d as o3d

def render_fracture_pointcloud(pieces: list, colors: list = None):
    """
    标准骨折点云渲染函数
    pieces: list of np.ndarray, shape (N, 3)
    colors: list of hex color strings, 按 COLORS 配色序列取色
    """
    default_colors = [
        [0.290, 0.565, 0.851],   # #4A90D9 piece_0
        [0.902, 0.294, 0.208],   # #E64B35 piece_1
        [0.165, 0.616, 0.561],   # #2A9D8F piece_2
        [0.914, 0.769, 0.412],   # #E9C46A piece_3
    ]
    pcds = []
    for i, pts in enumerate(pieces):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        c = default_colors[i % len(default_colors)]
        pcd.paint_uniform_color(c)
        pcds.append(pcd)
    return pcds
```

### 8.2 点云可视化输出规范

- **背景色**：统一使用白色背景 `[1.0, 1.0, 1.0]`，禁止黑色或灰色背景
- **点大小**：统一 `point_size=3.0`，密集点云可降至 `2.0`
- **视角**：统一使用前视图（front view）为默认展示角度，消融对比图各子图视角必须完全一致
- **坐标轴**：可视化结果图中**隐藏绝对坐标轴数值**，仅在需要说明尺度时添加比例尺标注
- **点云密度**：每个碎片采样点数不少于 **2000 点**，展示图不少于 **5000 点**

### 8.3 骨骼三维网格渲染规范

- 骨骼网格基础色：`#D4C5A9`（米棕，CT重建感），材质使用 Phong 光照模型
- 骨折块区分：在网格渲染模式下，不同碎片使用 §2.2 配色序列，透明度 `alpha=0.85`
- 植入物（钢板）：`#B0B0B0`（中灰），金属质感，透明度 `alpha=0.7`
- 螺钉路径：红色圆柱体 `#E64B35`，直径 1.5mm 等比例渲染
- 钻孔导板：`#A8D8EA`（浅蓝），透明度 `alpha=0.5`，突出与骨骼的区分

### 8.4 复位前后对比图规范

```python
# 标准复位对比图模板（matplotlib 3D 版本）
def plot_reduction_comparison(pre_pieces, post_pieces,
                               piece_colors=None, figsize=(14, 6)):
    """
    复位前后标准对比图
    左图：Pre-Reduction，右图：Post-Reduction
    """
    fig = plt.figure(figsize=figsize)
    titles = ['Pre-Reduction', 'Post-Reduction']
    data_list = [pre_pieces, post_pieces]

    default_colors = ['#4A90D9', '#E64B35', '#2A9D8F', '#E9C46A']
    if piece_colors is None:
        piece_colors = default_colors

    for idx, (title, pieces) in enumerate(zip(titles, data_list)):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        for i, pts in enumerate(pieces):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=piece_colors[i % len(piece_colors)],
                       s=1.5, alpha=0.6,
                       edgecolors='none')
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color='#333333', pad=8)
        # 隐藏绝对坐标轴数值，保留轴线
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_facecolor('white')
        # 统一视角
        ax.view_init(elev=20, azim=45)
        # 子图标签
        ax.text2D(-0.05, 1.02, f'({chr(97+idx)})',
                  transform=ax.transAxes,
                  fontsize=12, fontweight='bold', color='#333333')

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig
```

---

## 9. 流程图与架构图专项规范

本节适用于算法框架图、模块结构图等非数据类图表，主要通过 Figma / draw.io / Python 手动绘制。

### 9.1 模块色块规范

| 模块类型 | 背景色 | 边框色 | 说明 |
|---|---|---|---|
| 特征提取模块 | `#D4E8F7` | `#2C7BB6` | 浅天蓝背景 |
| 分类/分割模块 | `#FFF3CD` | `#D4A017` | 浅琥珀背景 |
| 匹配模块（含CFR） | `#D6EAF8` | `#2C7BB6` | 中天蓝背景 |
| 配准模块 | `#D5F5E3` | `#1A7A4A` | 浅绿背景 |
| 创新模块强调框 | `#FDECEA` | `#E64B35` | 浅红背景，虚线边框 |
| 输入/输出节点 | `#F5F5F5` | `#888888` | 浅灰，圆角矩形 |

### 9.2 箭头与连线规范

- 主流程箭头：实线，线宽 1.5pt，颜色 `#444444`，实心箭头
- 数据流箭头：实线，线宽 1.2pt，颜色 `#2C7BB6`，开放箭头
- 损失函数反向传播箭头：虚线，线宽 1.0pt，颜色 `#E64B35`
- 模块间跳跃连接（skip link）：虚线，线宽 1.0pt，颜色 `#888888`
- 所有箭头圆角半径统一：`border-radius: 4px`

### 9.3 节点形状规范

| 节点类型 | 形状 | 说明 |
|---|---|---|
| 处理模块 | 圆角矩形（radius=6px） | 主要计算步骤 |
| 判断/选择 | 菱形 | 条件分支 |
| 输入/输出 | 平行四边形 | 数据输入输出 |
| 开始/结束 | 椭圆 / 圆角胶囊 | 流程起止 |
| 损失函数 | 六边形 | 特殊标识损失项 |
| 注意力机制 | 圆形 | 区分于普通模块 |

### 9.4 子图内嵌骨骼渲染规范

在架构图中嵌入骨骼三维渲染图时：
- 渲染图必须有 `1px` 浅灰色（`#CCCCCC`）圆角边框
- 渲染图背景必须为白色，不得使用透明背景
- 嵌入尺寸不小于 `80×80px`（印刷版），确保骨折块颜色可辨

---

## 10. AI重绘通用Prompt基准

本节提供给使用 Midjourney / DALL·E / Stable Diffusion / GPT-4o 等工具重绘示意图的组员。

### 10.1 全局风格基准Prompt（所有AI重绘图必须包含）

```
Style: Clean academic illustration for biomedical engineering paper.
White background (#FAFAFA). No gradients on background.
Color palette: steel blue (#4A90D9), vermillion red (#E64B35),
teal green (#2A9D8F), amber (#E9C46A), deep purple (#6A4C93).
Font: Inter or sans-serif for labels. All text in English or Chinese,
no decorative fonts.
Line weight: 1.5pt for borders, 1.0pt for connectors.
Rounded rectangle modules with 6px radius.
No drop shadows. No 3D perspective on flat diagrams.
High contrast, publication-ready, IEEE/Nature style.
Resolution: 300 DPI equivalent.
```

### 10.2 骨骼示意图专用Prompt补充

```
3D bone rendering style: photorealistic CT reconstruction appearance.
Bone color: warm beige (#D4C5A9) with subtle Phong shading.
Fracture fragments: differentiated by color
  (fragment 0: #4A90D9 blue, fragment 1: #E64B35 red,
   fragment 2: #2A9D8F teal).
White background. No cartoon style. No watercolor.
Medical illustration quality. Anatomically accurate tibial plateau.
```

### 10.3 流程图/架构图专用Prompt补充

```
Technical architecture diagram. Flat design, no 3D effects.
Module boxes: light blue (#D4E8F7) background with blue (#2C7BB6) border.
Innovation modules: light red (#FDECEA) background with red (#E64B35)
dashed border.
Arrows: dark gray (#444444), solid, with filled arrowheads.
Loss function nodes: hexagon shape, red (#E64B35).
Layout: left-to-right data flow. Clean whitespace between modules.
```

---

## 11. 输出格式与文件命名

### 11.1 输出格式

**所有正式图表必须同时保存 PNG 和 SVG 两种格式：**

```python
def save_figure(fig, output_dir, filename_stem, dpi_png=300):
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename_stem}.png"
    svg_path = output_dir / f"{filename_stem}.svg"

    fig.savefig(png_path, dpi=dpi_png, bbox_inches='tight',
                facecolor='white', transparent=False)
    fig.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', transparent=False)

    print(f"   PNG saved: {png_path}")
    print(f"   SVG saved: {svg_path}")
    return png_path, svg_path
```

### 11.2 文件命名规范

```
{fig_id}_{内容描述}_{版本或数据集}.{ext}

示例：
  fig1_fracture_trend_2017_2023.png
  fig1_fracture_trend_2017_2023.svg
  fig2_icp_pipeline_redesign.png
  fig3_pointnet_architecture_combined.png
  fig4_algorithm_framework_full.png
  fig5_cfr_module_structure.png
  fig6_ablation_3d_comparison.png
  fig7_ctpelvic_reduction_comparison.png
  fig8_ui_screenshot.png
```

**命名规则：**
- 全小写，下划线分隔，不含空格
- 前缀 `fig` + 数字标识主图
- 内容描述不超过 4 个词
- 重绘版本加后缀 `_v2`、`_redesign`

---

## 12. rcParams 全局模板

```python
import matplotlib.pyplot as plt

def apply_global_style():
    """
    全局 matplotlib 样式配置
    在每个绘图脚本开头调用一次
    """
    plt.rcParams.update({
        # ── 字体 ──────────────────────────────────────────
        'font.family':           'serif',
        'font.serif':            ['Times New Roman', 'Georgia',
                                  'DejaVu Serif', 'serif'],
        'mathtext.fontset':      'stix',
        'axes.unicode_minus':    False,

        # ── 字号 ──────────────────────────────────────────
        'font.size':             10,
        'axes.titlesize':        14,
        'axes.labelsize':        12,
        'xtick.labelsize':       10,
        'ytick.labelsize':       10,
        'legend.fontsize':       11,
        'figure.titlesize':      14,

        # ── 坐标轴 ────────────────────────────────────────
        'axes.linewidth':        1.0,
        'axes.edgecolor':        '#444444',
        'axes.facecolor':        'white',
        'axes.labelcolor':       '#333333',
        'axes.spines.top':       False,
        'axes.spines.right':     False,

        # ── 刻度 ──────────────────────────────────────────
        'xtick.direction':       'out',
        'ytick.direction':       'out',
        'xtick.major.width':     1.0,
        'ytick.major.width':     1.0,
        'xtick.minor.width':     0.8,
        'ytick.minor.width':     0.8,
        'xtick.major.size':      4.0,
        'ytick.major.size':      4.0,
        'xtick.minor.size':      2.0,
        'ytick.minor.size':      2.0,
        'xtick.color':           '#333333',
        'ytick.color':           '#333333',

        # ── 线条 ──────────────────────────────────────────
        'lines.linewidth':       2.0,
        'lines.markersize':      6.0,
        'patch.linewidth':       1.0,

        # ── 图例 ──────────────────────────────────────────
        'legend.frameon':        True,
        'legend.framealpha':     0.9,
        'legend.edgecolor':      '#CCCCCC',
        'legend.fancybox':       False,
        'legend.handlelength':   2.0,
        'legend.borderpad':      0.6,
        'legend.labelspacing':   0.4,

        # ── 图幅 ──────────────────────────────────────────
        'figure.facecolor':      'white',
        'figure.dpi':            100,
        'savefig.dpi':           300,
        'savefig.bbox':          'tight',
        'savefig.facecolor':     'white',
        'savefig.transparent':   False,

        # ── 网格 ──────────────────────────────────────────
        'axes.grid':             False,
        'grid.color':            '#E0E0E0',
        'grid.linestyle':        '--',
        'grid.linewidth':        0.5,
        'grid.alpha':            0.7,

        # ── 颜色循环（按项目配色序列）────────────────────────
        'axes.prop_cycle': plt.cycler(color=[
            '#4A90D9', '#E64B35', '#2A9D8F',
            '#E9C46A', '#6A4C93', '#FF6B35',
        ]),
    })

# 调用方式（每个脚本开头）：
apply_global_style()
```

---

## 13. 图表改进任务清单（ABCDE分类）

> 本节为图表改进工作的执行指南。
> **A** = 彻底重构/重新跑代码，图形本身无法挽救
> **B** = AI/PS/Figma辅助重绘或风格统一修改，内容正确但视觉需提升
> **C** = 合并为大图，单独存在信息量不足或重复
> **D** = 直接删除，对全文无实质增益
> **E** = 无需改动或仅需小修，质量已达标

---

### 🅰️ A类 — 彻底重构

#### 图1.1.1 · 骨折诊疗人次折线图

**问题：**
- Excel默认导出风格，白底单色，无任何设计感
- 作为报告**第一张图**，严重影响读者第一印象
- **存在明显笔误**：图注写"2017-2022年"，但图中数据实际显示至2023年

**重构方案：**
```python
# 使用 matplotlib 精细绘制，调用 apply_global_style()
# 折线颜色：#4A90D9（piece_main 天蓝）
# 加入浅蓝色面积填充（alpha=0.15）增加视觉层次
# 2020年回落点加注释标注（"COVID-19 impact"或留空）
# 数据标注字体改为 L5（9pt），仅标注首尾和极值点
# 修正图注：改为"2017-2023年"
# 加入临床需求增长趋势线（线性拟合虚线，#E9C46A）
```

---

#### 图1.2.2 · 传统ICP配准流程图

**问题：**
- 纯黑白Word流程图风格，无配色，全英文
- 与报告整体科技感完全不符，是报告中视觉质量最低的图
- 流程内容本身正确，但呈现方式需彻底重做

**重构方案：**
- 使用 Figma 或 draw.io 按 §9 流程图规范重绘
- 配色：输入节点用 `#F5F5F5`，处理步骤用 `#D4E8F7`，判断节点用 `#FFF3CD`，结束节点用 `#D5F5E3`
- 标注改为中文，箭头统一为 `#444444` 实心箭头
- 在"Calculate coordinate transformations"节点旁加注"对初始位姿敏感"的缺陷标注（红色小标签），呼应正文对ICP局限性的讨论

---

#### 图3.2.3 · CTPelvic1K复位前后三维点云对比

**问题：**
- matplotlib `ax.scatter()` 裸输出，白底默认字体，无任何美化
- 三维坐标轴显示绝对数值（1500-1650），对读者毫无意义且视觉杂乱
- 点云密度偏低，立体感差

**重构方案：**
```python
# 改用 Open3D 或 PyVista 渲染，调用 §8.4 plot_reduction_comparison()
# 隐藏绝对坐标轴数值，统一视角 elev=20, azim=45
# piece_0: #4A90D9，piece_1: #E64B35
# 点大小: point_size=2.0，密度提升至每片 ≥ 5000 点
# 背景：纯白
# 在图下方加入简洁的误差数值标注（Rot MAE / Trans MAE）
```

---

### 🅱️ B类 — AI/PS辅助重绘

#### 图1.2.1 · 镜像模板骨折复位流程

**问题：**
- 四图之间缺乏步骤箭头，流程顺序依赖图注文字，视觉引导性不足
- 图像本身质量尚可，无需重新渲染骨骼

**修改方案：**
- 在 PS / Figma 中，在四图之间加入统一风格的步骤箭头（`#444444`，线宽1.5pt）
- 每图左上角加入步骤编号圆形标签（`①②③④`，背景 `#4A90D9`，白色数字）
- 整体加一个浅灰色（`#F5F5F5`）圆角背景框统一视觉边界

---

#### 图2.2.1 · PointNet++网络结构图

**问题：**
- 左侧SA/FP子模块图与右侧流程图风格略有差异（虚线框 vs 实线矩形）
- 整体配色（粉/蓝/绿）与本项目配色体系不完全一致

**修改方案：**
- 使用 §10.3 架构图Prompt重绘左侧子模块图，使其与右侧流程图风格统一
- Sample节点改为 `#4A90D9`，Grouping改为 `#2A9D8F`，PointNet改为 `#D4E8F7`
- 右侧流程图 Set Abstraction 框改为 `#D4E8F7` 背景，Feature Propagation 改为 `#D5F5E3` 背景

---

#### 图2.3.2 · 凹凸描述符匹配示意图

**问题：**
- 两个"骨折块"用圆形区域表示过于抽象
- 弯曲连线密度高，视觉略显凌乱
- 色块（深紫/浅蓝）与本项目配色体系不一致

**修改方案：**
- 使用 §10.1 + §10.2 Prompt重绘，将圆形区域替换为骨折断面的简化轮廓（梯形或不规则多边形，模拟断面形状）
- 凸描述符改为 `#4A90D9`（天蓝），凹描述符改为 `#E64B35`（朱红），与骨折块配色语义一致
- 连线改为有方向感的渐变弧线，减少连线数量至3-4条典型示例，去除视觉冗余

---

#### 图2.3.4 · 刚体约束示意图

**问题：**
- 卡通骨头插画风格（暖棕色渐变、米黄背景）与报告整体科技感严重不符
- 内容正确且必要，仅需风格统一

**修改方案：**
- 使用 §10.2 骨骼Prompt重绘，将卡通骨头替换为与报告一致的灰白色三维骨骼渲染风格（`#D4C5A9`）
- 背景改为纯白（`#FAFAFA`）
- 保留虚线矩形框和 i/j/k/l 标注，字体改为 Inter，颜色 `#333333`
- 双向距离箭头改为 `#4A90D9`（天蓝），与骨折块主色一致

---

#### 图3.2.1 · 断面分割可视化

**问题：**
- 点云密度偏低，背景纯白显得空旷
- 两张图之间缺乏视觉框架（无边框区分）
- 指标数值（Acc/F1/IoU）显示方式过于朴素

**修改方案：**
```python
# 重新跑可视化代码
# 提高点云采样密度至 ≥ 5000 点/碎片
# 为左右两图加入浅灰色（#F5F5F5）背景框，圆角 4px，1px #CCCCCC 边框
# 断面点云颜色：#E64B35（朱红），骨骼主体：#D3D3D3（浅灰）
# 指标数值改为彩色高亮：Acc用#2A9D8F，F1用#4A90D9，IoU用#E9C46A
# 字体统一为 Times New Roman，L3（11pt）
```

---

### 🅲️ C类 — 建议合并

#### 图2.2.1 + 图2.2.2 → 合并为"PointNet++完整架构图"

**理由：**
两张图均在阐述 PointNet++ 的结构，图2.2.1讲模块细节，图2.2.2讲整体层次架构，单独看都不完整，合并后形成从"局部模块→整体架构"的完整认知链，减少读者翻页跳跃，节省报告篇幅约半页。

**合并方式：**
- 左侧：SA模块 + FP模块细节示意（图2.2.1内容，按B类方案重绘风格统一版）
- 右侧：层次化架构全图（图2.2.2内容，中文化标注版）
- 中间用竖向虚线（`#CCCCCC`，0.8pt）分隔
- 整体标题："PointNet++网络结构与层次化架构"
- 子图标签：(a) 模块结构，(b) 层次化架构
- 图幅：`(16, 7)`

---

#### 图3.2.1 + 图3.2.2 → 合并为"分割与复位实验结果综合展示"

**理由：**
图3.2.1展示断面分割结果，图3.2.2展示消融复位对比，两者都是第3章实验可视化，点云渲染风格相同，存在天然的因果关系（分割质量→复位质量）。合并为一张大图可以更紧凑地呈现完整实验链条，同时节省约半页篇幅。

**合并方式：**
- 上行（1×2）：分割结果，Ground Truth | Prediction，子图标签 (a)(b)
- 下行（1×4）：消融复位对比，GT | +CFR | +Rig Loss | 完整模型，子图标签 (c)(d)(e)(f)
- 整体标题："断面分割与三维复位消融实验可视化"
- 图幅：`(18, 10)`，使用 `layout='constrained'`

---

### 🅳️ D类 — 建议删除

#### 图2.2.2 · PointNet++层次化网络架构（原图）

**理由：**
- 直接引用自 PointNet++ 原论文的英文图，与报告中文语境不符
- 其传达的内容（层次化特征提取）在正文中已有详细文字描述，在图2.1.1整体框架图中也有所体现
- **采纳C类合并方案后**，此图的内容将以中文化重绘版本融入合并大图，原图可直接删除
- 若不采纳合并方案，则至少需要对此图进行完整中文化处理后方可保留，不得以现有英文原图形式出现在报告中

---

### 🅴️ E类 — 无需改动或仅需小修

#### 图2.1.1 · 骨折算法复位整体简化框架 ⭐ 8.5/10
四色分区专业，骨骼渲染与流程框图有机融合，信息密度高而不杂乱。**无需改动。**

#### 图2.2.3 · 胫骨平台不同数据类型对比 ⭐ 8.5/10
三种数据类型渲染质量均高，配色各具特色，排版工整。**无需改动。**

#### 图2.3.1 · 胫骨平台骨折分型插画 ⭐ 7.5/10
医学插画风格统一，蓝色骨骼+橙红色骨折线配色专业。**小修建议：** 将橙红色骨折线颜色从原图色统一为 `#E64B35`，与报告 `fracture_face` 语义色保持一致。

#### 图2.3.3 · CFR模块结构示意图 ⭐ 7/10
双层架构清晰，粗/精匹配分支视觉区分合理。**小修建议：** 将"粗匹配"背景框颜色改为 `#D4E8F7`，"精匹配微调"文字颜色改为 `#E64B35`，与 §9.1 模块色块规范对齐。

#### 图1.2.3 · Schatzker Ⅵ型临床病例 ⭐ 7.5/10
金黄渲染与X光片混排，临床说服力强。**无需改动**（引用自临床文献，保持原样）。

#### 图3.2.2 · 消融实验三维复位对比 ⭐ 7.5/10
三色点云对比有力，四图并排说服力强。**小修建议：** 若采纳C类合并方案，此图将作为合并大图的下行，届时按 §8 规范统一点云渲染风格即可。

#### 图4.1.1 · 项目算法总体框架 ⭐ 9/10
全报告最精美的技术图，达到顶级学术会议论文水准。**无需改动。**

#### 图4.1.2 · 软件交互界面截图 ⭐ 9/10
深色UI设计震撼，商业软件质感。**无需改动**（截图类图表，保持原样）。

#### 图4.2.1 · 公共数据集可视化示例 ⭐ 7/10
左侧骨骼渲染质量高，右侧Breaking Bad对比图色彩丰富。**小修建议：** 在左右两部分之间加入竖向分隔线（`#CCCCCC`，0.8pt），并为左侧三图加入统一的浅灰背景框，减少左右风格割裂感。

#### 图4.2.2 · 虚拟手术规划流程 ⭐ 9/10
五步三维渲染质量极高，视觉冲击力最强。**无需改动**（引用自展望性文献图，保持原样）。

---

## 14. 禁止事项

### 14.1 字体禁止项
- 使用 `sans-serif` / Arial / Helvetica 作为数据图表主字体
- 在同一图表中混用超过 2 种字体家族
- 数学变量使用正体（`Rot MAE`、`p` 作为变量时必须斜体）
- 单位使用斜体（`mm`、`°` 作为单位时必须正体）

### 14.2 配色禁止项
- 使用 `jet`、`rainbow`、`hsv` 色图
- CT 图像使用非灰阶色图显示原始 HU 数据
- 随意更改语义配色的颜色-含义对应关系（如将 `#4A90D9` 用于实验组误差标注）
- 使用纯黑 `#000000` 作为文字或轴线颜色（使用 `#333333` 或 `#444444`）
- 骨折块配色与骨干主体色（`#7EC8A4`）混用

### 14.3 点云可视化禁止项
- 使用 matplotlib `ax.scatter()` 直接输出三维点云（必须使用 Open3D 或 PyVista）
- 点云图保留绝对坐标轴数值（隐藏数值，仅保留轴线或比例尺）
- 不同子图使用不同视角（消融对比图必须统一视角参数）
- 点云背景使用黑色或深色（统一使用白色背景）

### 14.4 流程图禁止项
- 使用 Word 自带流程图工具绘制（必须使用 Figma / draw.io / matplotlib patches）
- 全英文标注（中文报告中流程图节点文字必须为中文）
- 直接引用外文文献原图而不做中文化处理
- 卡通/插画风格示意图（必须使用学术技术插图风格）

### 14.5 输出禁止项
- 仅保存 PNG 而不保存 SVG
- PNG 输出 DPI 低于 300
- 保存时未设置 `bbox_inches='tight'`
- 文件名包含空格、中文字符或特殊符号

---

*文档结束 | Visualization Style Guide v1.0 · Tibial Plateau Fracture Project | 2026-06-03*
