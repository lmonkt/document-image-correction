# 文档图像去倾斜与旋转校正工具

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于**传统图像处理算法**和**深度学习**的文档图像矫正工具，可自动修正扫描文档中的倾斜和方向错误问题，适用于 OCR、版面分析等下游任务的预处理。

> 详细技术文档请参阅：[技术分享-文档图像去倾斜与旋转校正方法.md](技术分享-文档图像去倾斜与旋转校正方法.md)

---

## 📌 项目特点

### 🎯 核心功能
- **去倾斜（Deskew）**：修正 ±45° 范围内的任意角度倾斜
- **旋转校正（Rotation Correction）**：识别并修正 0°/90°/180°/270° 的方向错误
- **单次旋转优化**：合并两种角度，避免二次插值导致的质量损失

### 🔬 技术亮点
1. **去倾斜**：基于 **FFT 频谱分析 + 自适应径向投影**
   - 无需训练模型，纯图像处理实现
   - 精度高（平均误差 < 0.1°）
   
2. **方向分类**：轻量级 **ONNX 深度学习模型**
   - 4分类（0°/90°/180°/270°）
   - 支持置信度阈值过滤

3. **工程优化**
   - 单次仿射变换，避免多次旋转的质量损失
   - 可选的文本区域裁剪，去除旋转后的白边
   - 完整的命令行接口和 Python API

---

## 🚀 快速开始

### 1. 环境要求
- Python 3.7+
- 操作系统：Linux / macOS / Windows

### 2. 安装依赖
```bash
# 克隆项目
git clone <repository_url>
cd doc_img_corr

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 运行示例
```bash
# 处理单张图像（自动保存到 corrected/ 目录）
python deskew_and_rotation_correction.py image.png

# 指定输出路径
python deskew_and_rotation_correction.py input.jpg -o output.jpg

# 指定模型路径
python deskew_and_rotation_correction.py input.jpg -m model/inference.onnx
```

---

## 📂 项目结构

```
doc_img_corr/
├── deskew_and_rotation_correction.py  # 主程序：完整的矫正流程
├── image_orientation.py               # 方向分类模块（ONNX推理）
├── image_rotation.py                  # 图像旋转工具
├── text_crop_optimized.py             # 文本区域裁剪（去白边）
├── visualize_crop_pipeline.py         # 裁剪流程可视化
├── generate_test_data.py              # 测试数据生成
├── pdf_page_to_image.py               # PDF转图像工具
├── model/
│   └── inference.onnx                 # 方向分类模型（必需）
├── for_eval_deskew/                   # 算法评估脚本
│   ├── jdeskew_exp_eval.py            # 论文复现版评估
│   ├── jdeskew_lib_eval.py            # 官方库版评估
│   ├── my_method_eval.py              # 自定义方法评估
│   └── speed_benchmark_comprehensive.py # 性能基准测试
├── requirements.txt                   # 依赖列表
└── 技术分享-文档图像去倾斜与旋转校正方法.md  # 详细技术文档
```

---

## 💡 使用场景

### 适用场景
- ✅ 扫描文档的倾斜校正（如歪斜的扫描件）
- ✅ 方向错误的文档矫正（如横向文档被误存为竖向）
- ✅ OCR 预处理：提升文字识别准确率
- ✅ 版面分析预处理：确保模型输入标准化
- ✅ 文档图像数据清洗

### 不适用场景
- ❌ 自然场景图片（如照片、风景）
- ❌ 透视畸变严重的文档（需要透视变换）
- ❌ 极低分辨率或严重模糊的图像

---

## 🛠️ 核心模块说明

### 1. deskew_and_rotation_correction.py
**主程序**，整合了完整的矫正流程：
```python
# Python API 调用示例
from deskew_and_rotation_correction import process_image

corrected_img = process_image(
    image_path='input.jpg',
    model_path='model/inference.onnx',
    output_path='output.jpg'
)
```

**流程说明**：
1. 读取图像
2. FFT 频谱分析估计倾斜角度
3. ONNX 模型预测方向角度
4. 合并角度并单次旋转
5. 保存结果

---

### 2. image_orientation.py
**方向分类模块**，基于 ONNX 推理：
```python
from image_orientation import ImageOrientationClassifier

classifier = ImageOrientationClassifier('model/inference.onnx')
angle_index, confidence = classifier.predict(image)
# angle_index: 0=0°, 1=90°, 2=180°, 3=270°
```

**技术细节**：
- 预处理：Resize(短边256) → CenterCrop(224) → Normalize
- 模型：基于 PaddleClas 训练的轻量级分类器
- 输出：包含置信度，可设阈值过滤误判

---

### 3. text_crop_optimized.py
**文本区域裁剪**，去除旋转后的白边：
```python
from text_crop_optimized import TextRegionCropEnlargeStep

cropper = TextRegionCropEnlargeStep(
    enlarge_target_size=(1200, 1600),
    expand_pixels=30
)
cropped_img = cropper.execute(rotated_img)
```

**功能**：
- 自动检测文本区域
- 去除大量空白边缘
- 智能缩放到目标尺寸

**推荐搭配**：去倾斜/旋转后使用，效果更佳

---

### 4. 其他辅助工具

| 文件 | 功能 | 用途 |
|------|------|------|
| `image_rotation.py` | 高保真图像旋转 | 生成测试数据 |
| `generate_test_data.py` | 测试数据生成 | 创建倾斜+旋转混合数据 |
| `pdf_page_to_image.py` | PDF转图像 | 批量提取PDF页面 |
| `visualize_crop_pipeline.py` | 裁剪流程可视化 | 调试裁剪算法参数 |

---

## 🔬 算法原理

### 去倾斜算法（基于 jdeskew）
1. **FFT变换**：将图像从空间域转到频域
2. **自适应阈值二值化**：增强文本纹理
3. **径向投影**：在不同角度上积分频谱能量
4. **中心遮挡优化**：屏蔽低频噪声，提升精度
5. **自适应决策**：根据 `a_init` 和 `a_correct` 的差异选择最优角度

**参数配置**：
- `amax=45`：搜索范围 ±45°
- `V=2048`：预处理高度（越大越精确但越慢）
- `W=304`：中心遮挡宽度
- `D=0.55`：决策阈值

**参考论文**：
> Pham, Q. L., et al. "Adaptive Radial Projection on Fourier Magnitude Spectrum for Document Image Skew Estimation." ICIP 2022.

---

### 方向分类模型（PP-LCNet_x1_0_doc_ori）
- **模型名**：PP-LCNet_x1_0_doc_ori（https://www.paddleocr.ai/latest/version3.x/module_usage/doc_img_orientation_classification.html#_3）
- **来源**：https://github.com/RapidAI/RapidOrientation

---

## 📖 详细技术文档

本项目提供了详尽的技术分享文档，内容包括：

1. **数据场景分析**：三种典型倾斜/旋转场景
2. **去倾斜算法详解**：三种实现方法对比（jdeskew-exp / jdeskew-lib / 自定义方法）
3. **评估指标与实验**：多个公开数据集的性能对比
4. **方向分类详解**：模型推理、置信度使用
5. **完整方案设计**：如何组合去倾斜和旋转校正
6. **图像预处理优化**：文本裁剪的原理和效果

👉 **[点击查看完整技术文档](技术分享-文档图像去倾斜与旋转校正方法.md)**

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [jdeskew](https://github.com/phamquiluan/jdeskew)：去倾斜算法的参考实现
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)：方向分类模型训练框架
- [RapidOrientation](https://github.com/RapidAI/RapidOrientation)
- OpenCV：图像处理基础库
