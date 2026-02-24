# 去倾斜算法评估程序

## 概述

本评估程序用于测试三种去倾斜算法在 DISE 2021 数据集上的性能和速度对比。

## 文件说明

### 核心模块

- `deskew_angle_only.py`: My 方法的角度估计模块，基于 Coarse-to-Fine 两阶段搜索策略

### 评估程序

1. `my_method_eval.py`: My 方法（Coarse-to-Fine 两阶段搜索）的精度评估程序
2. `jdeskew_lib_eval.py`: jdeskew lib 方法（官方库版）的精度评估程序  
3. `jdeskew_exp_eval.py`: jdeskew exp 方法（论文复现版）的精度评估程序

### 速度测评

- `speed_benchmark_comprehensive.py`: 三种方法的速度对比测评程序

### 文档

- `README.md`: 本文件
- `USAGE.md`: 使用说明
- `JDESKEW_EXP_PERF_ANALYSIS.md`: jdeskew exp 单进程性能分析（py-spy + line_profiler）

## 三种去倾斜方法说明

在基于频谱分析的图像去倾斜任务中，本项目对比了三种不同的实现路径。虽然它们的核心思想都是利用傅里叶变换（FFT）的频谱能量分布来检测文本行的倾斜角度，但在具体的搜索策略和鲁棒性处理上存在显著差异。

### 1. jdeskew exp 方法（论文复现版）

这是对 jdeskew 原始论文（及 `reproduce.ipynb`）的忠实复现。它的核心特点是**自适应性（Adaptive）**。

**核心机制：**
- **双角度计算**：同时计算两个候选角度
  - **`a_init`**：基于全频谱的径向投影计算出的角度，稳定性好，但可能受文档背景或低频噪声干扰
  - **`a_correct`**：屏蔽掉频谱中心低频分量（由参数 W 控制遮挡宽度）后计算出的角度，对文本纹理的高频成分更敏感，潜在精度更高

- **动态决策机制**：通过比较两个角度的差异 `dist = |a_init - a_correct|` 与阈值 `D`（如 0.55）：
  - 若 `dist <= D`，认为中心遮挡后的结果可信，采用高精度的 **`a_correct`**
  - 若 `dist > D`，认为高频噪声过大或纹理不清晰，回退采用更稳健的 **`a_init`**

**特点：** 这是一种"双保险"策略，在精度和稳定性之间做了动态平衡，但依赖于对 `W` 和 `D` 两个超参数的调优。

### 2. jdeskew lib 方法（官方库版）

这是作者发布的 Python 库（`pip install jdeskew`）中的实现。相较于 `exp` 方法，它进行了大幅简化，旨在提供一个开箱即用的通用版本。

**核心机制：**
- **核心简化**：移除了中心遮挡参数 `W` 和决策阈值 `D`，直接对整个频谱图进行径向投影积分，仅返回单一角度
- **与 exp 的关系**：从代码逻辑看，它等同于只保留了 `exp` 方法中的 **`a_init`** 计算分支，完全去除了自适应校正逻辑

**特点：** 计算更直接，速度稍快，且无需调参。但在复杂背景或文本纹理较弱的场景下，由于缺乏 `a_correct` 的高频增强和 `D` 的异常回退机制，精度可能不如 `exp` 方法极致。

### 3. My 方法（Coarse-to-Fine 两阶段搜索）

个人基于 `jdeskew exp` 进行的改造。不同于原论文的"阈值切换"逻辑，采用了更确定性的 **Coarse-to-Fine（粗细两阶段）搜索策略**。

**核心机制：**
- **两阶段搜索**：
  1. **粗略估计（Coarse）**：首先在较大范围内（如 -90° 到 90°），以较大步长（如 1.0°）快速定位能量最强的粗略角度
  2. **精细估计（Fine）**：在粗估角度的邻域内（如 ±2°），以极小步长（如 0.1°）进行密集插值搜索，锁定最终精确角度

- **思路演变**：
  - 不再纠结于 `a_init` 和 `a_correct` 的选择困境（不再需要阈值 `D` 进行回退判断）
  - 通过"粗定位+精搜索"的机制，实质上强制执行了类似 `a_correct` 的高精度路径，同时避免了单次细粒度搜索带来的巨大计算开销
  - 代码中虽然保留了小幅度的中心遮挡（半径为 2），但主要依靠搜索策略的提升来保证精度

**特点：** 这种方法更加"自信"且确定性强，通过增加计算密度（两轮搜索）换取更高的角度分辨率。准确率比 lib 好但不如 exp，速度比较快。

## 评估指标

参考 jdeskew 论文（ICIP 2022），使用以下指标：

1. **AED** (Average Error in Degrees): 平均误差（度）
2. **TOP80**: 前80%样本的平均误差（度）
3. **CE** (Correct Estimation): 误差 ≤ 0.1度的样本比例
4. **WORST**: 最大误差（度）

## 数据集

本项目评估以下数据集：

- **DISE 2021 (15°)**: 倾斜角度范围 -15° 到 +15°
  - 路径: `C:\data\datasets\dise2021_15\test`
  - 图像数: 1,491 张测试图像
  
- **DISE 2021 (45°)**: 倾斜角度范围 -44.9° 到 +44.9°
  - 路径: `C:\data\datasets\dise2021_45\test`
  - 图像数: 2,800 张测试图像

数据集中的图像文件名格式为 `xxx[angle].ext`，其中 `[angle]` 是真实的倾斜角度。

## 使用方法

### 精度评估

运行三种方法的精度评估程序：

```bash
# 评估 My 方法
python my_method_eval.py

# 评估 jdeskew lib 方法
python jdeskew_lib_eval.py

# 评估 jdeskew exp 方法
python jdeskew_exp_eval.py
```

每个评估程序会：
1. 处理 DISE 2021 (15°) 和 DISE 2021 (45°) 两个数据集
2. 计算 AED、TOP80、CE、WORST 四项指标
3. 生成 JSON 格式的指标文件和详细结果文件
4. 在终端打印评估结果汇总

### 速度测评

运行速度对比测评：

```bash
python speed_benchmark_comprehensive.py
```

速度测评程序会：
1. 在 DISE 2021 (15°) 和 (45°) 数据集上各测试 50 张图片
2. 对比三种方法的处理速度
3. 对 jdeskew exp 方法测试多种参数配置（V1024/V1500/V2048/V3072/V4096）
4. 输出各方法的平均耗时和性能比较

### 单独测试某张图像

使用 My 方法测试单张图像：

```bash
python deskew_angle_only.py <image_path>
```

例如：
```bash
python deskew_angle_only.py test_image[5.5].png
```

## 输出文件

### 精度评估输出

每个评估程序会生成以下文件：

1. `<method>_metrics_<dataset>_YYYYMMDD_HHMMSS.json`: 评估指标的 JSON 格式
2. `<method>_details_<dataset>_YYYYMMDD_HHMMSS.txt`: 每张图像的详细结果

其中 `<method>` 为：
- `my_method`: My 方法
- `jdeskew_lib`: jdeskew lib 方法
- `jdeskew_exp`: jdeskew exp 方法

`<dataset>` 为：
- `dise15`: DISE 2021 (15°)
- `dise45`: DISE 2021 (45°)

### 速度测评输出

速度测评程序在终端打印结果，包括：
1. 每个数据集上各方法的平均耗时
2. 各方法的总平均耗时汇总
3. 性能比较（相对于最快算法的倍数）

## 性能参考

根据 jdeskew 论文（ICIP 2022），在 DISE 2021 数据集上的性能对比：

### DISE 2021 (15°)

| 方法             | AED  | TOP80 | CE   | WORST |
|------------------|------|-------|------|-------|
| Our (3072)       | 0.07 | 0.04  | 0.86 | 1.13  |
| LRDE-EPITA-a     | 0.14 | 0.06  | 0.66 | 10.61 |
| CMC-MSU          | 0.27 | 0.11  | 0.43 | 23.2  |

### DISE 2021 (45°)

| 方法             | AED  | TOP80 | CE   | WORST |
|------------------|------|-------|------|-------|
| Our (3072)       | 0.05 | 0.02  | 0.89 | 1.06  |
| LRDE-EPITA-a     | 0.10 | 0.04  | 0.73 | 7.02  |
| CMC-MSU          | 0.20 | 0.07  | 0.56 | 17.46 |

## 注意事项

1. 评估程序使用多进程并行处理，默认使用 CPU 核心数的 70%
2. 如果数据集路径不存在，请修改各评估程序中的 `dataset_15` 和 `dataset_45` 变量
3. 确保图像文件名包含正确的角度标注格式 `[angle]`
4. jdeskew lib 方法需要安装 jdeskew 库：`pip install jdeskew`

## 依赖项

```txt
numpy
opencv-python
tqdm
jdeskew  # 仅 jdeskew lib 方法需要
```

安装依赖：
```bash
pip install numpy opencv-python tqdm jdeskew
```

## My 方法 API 说明

`deskew_angle_only.py` 提供了简单的 API 用于角度估计：

```python
from deskew_angle_only import get_skew_angle

# 基本用法
angle = get_skew_angle(image, max_skew_angle=45.0)

# 高级参数（对应 jdeskew 论文参数）
angle = get_skew_angle(
    image,
    max_skew_angle=45.0,      # 对应论文中的 amax
    coarse_angle_step=1.0,    # 粗略估计的步长
    fine_angle_range=2.0,     # 精细估计的范围
    fine_angle_step=0.1,      # 精细估计的步长
)
```

### 参数说明

- `max_skew_angle`: 最大倾斜角度（度），默认 45.0
- `coarse_angle_step`: 粗略估计的角度步长（度），默认 1.0
- `fine_angle_range`: 精细估计在粗略角度附近的搜索范围（度），默认 2.0
- `fine_angle_step`: 精细估计的角度步长（度），默认 0.1

### 返回值

返回估计的倾斜角度（度）：
- 正值表示图像顺时针倾斜
- 负值表示图像逆时针倾斜

## 参考文献

L. Pham, H. Hoang, X.T. Mai, T. A. Tran, "Adaptive Radial Projection on Fourier Magnitude Spectrum for Document Image Skew Estimation", ICIP, 2022.

GitHub: [phamquiluan/jdeskew](https://github.com/phamquiluan/jdeskew)
