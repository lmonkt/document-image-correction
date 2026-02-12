#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档图像去倾斜+旋转校正工具（主程序）

对应博客文档：四、完整去倾斜+旋转校正方案

功能说明：
  本脚本是项目的核心应用程序，整合了文档图像矫正的完整pipeline：
  1. 去倾斜（Deskew）：基于 FFT 频谱分析 + 自适应径向投影算法
     - 可处理 ±45° 范围内的任意角度倾斜
     - 参数配置：amax=45, V=2048, W=304, D=0.55
     - 算法原理：对应博客 二、文档图像去倾斜 章节
  
  2. 方向校正（Rotation Correction）：基于轻量级深度学习分类模型
     - 使用 ONNX 模型预测图像方向（0°/90°/180°/270°）
     - 模型路径：model/inference.onnx
     - 算法原理：对应博客 三、文档图像方向分类与矫正 章节
  
  3. 单次旋转优化：合并去倾斜和方向校正的角度，避免二次插值带来的质量损失

使用场景：
  - 处理扫描文档中的倾斜问题（如歪斜的扫描件）
  - 处理方向错误的文档（如倒置或横向竖排的文档）
  - 作为 OCR、版面分析等下游任务的预处理步骤

命令行用法：
  python deskew_and_rotation_correction.py <input_image> [--output <output_path>] [--conf_threshold <float>]
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import argparse


# ========= 去倾斜模块 =========

def ensure_gray(image: np.ndarray) -> np.ndarray:
    """确保图像为灰度图"""
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return image


def ensure_optimal_square(image: np.ndarray) -> np.ndarray:
    """扩展图像到最优DFT尺寸"""
    nw = nh = cv2.getOptimalDFTSize(max(image.shape[:2]))
    output_image = cv2.copyMakeBorder(
        src=image,
        top=0,
        bottom=nh - image.shape[0],
        left=0,
        right=nw - image.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )
    return output_image


def get_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """获取FFT幅度谱"""
    gray = ensure_gray(image)
    opt_gray = ensure_optimal_square(gray)

    # 自适应阈值处理
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    # FFT变换
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)

    # 获取幅度
    magnitude = np.abs(shifted_dft)
    return magnitude


def _get_angle_adaptive(m: np.ndarray, amax: float, W: int) -> tuple:
    """径向投影 + 中心遮挡"""
    assert m.shape[0] == m.shape[1]
    r = c = m.shape[0] // 2

    num = 20
    tr = np.linspace(-1 * amax, amax, int(amax * num * 2)) / 180 * np.pi

    # 预分配
    li_init = np.zeros_like(tr)
    li_correct = np.zeros_like(tr)

    for i, t in enumerate(tr):
        x = np.arange(0, r)
        y = c + np.int32(x * np.cos(t))
        x_proj = c + np.int32(-1 * x * np.sin(t))
        valid = (y >= 0) & (y < m.shape[0]) & (x_proj >= 0) & (x_proj < m.shape[1])
        vals = m[y[valid], x_proj[valid]]
        if vals.size == 0:
            continue
        li_init[i] = np.sum(vals)
        li_correct[i] = np.sum(vals[W:]) if W > 0 else li_init[i]

    a_init = tr[np.argmax(li_init)] / np.pi * 180
    a_correct = tr[np.argmax(li_correct)] / np.pi * 180

    dist = abs(a_init - a_correct)
    return -1 * a_init, -1 * a_correct, dist


def estimate_deskew_angle(image: np.ndarray) -> float:
    """
    估计去倾斜角度（固定参数：amax=45, V=2048, W=304, D=0.55）
    """
    amax = 45
    V = 2048
    W = 304
    D = 0.55

    # 缩放图像
    ratio = V / image.shape[0]
    resized_image = cv2.resize(image, None, fx=ratio, fy=ratio)

    # FFT分析
    magnitude = get_fft_magnitude(resized_image)
    a_init, a_correct, dist = _get_angle_adaptive(magnitude, amax=amax, W=W)

    # 根据距离选择角度
    if dist <= D:
        return a_correct
    return a_init


# ========= 旋转校正模块 =========

class Preprocess:
    """图像预处理流程"""
    
    @staticmethod
    def resize_image(img: np.ndarray, resize_short: int = 256):
        """缩放：短边缩放到指定大小"""
        img_h, img_w = img.shape[:2]
        percent = float(resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        return cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def crop_image(img: np.ndarray, size: int = 224):
        """中心裁剪"""
        img_h, img_w = img.shape[:2]
        if img_h < size or img_w < size:
            raise ValueError(
                f"图像尺寸 ({img_h}, {img_w}) 小于裁剪尺寸 ({size}, {size})"
            )
        w_start = (img_w - size) // 2
        h_start = (img_h - size) // 2
        return img[h_start:h_start + size, w_start:w_start + size, :]
    
    @staticmethod
    def normalize_image(img: np.ndarray):
        """归一化：(img/255 - mean) / std"""
        scale = np.float32(1.0 / 255.0)
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype("float32")
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype("float32")
        
        img = np.array(img).astype(np.float32)
        img = (img * scale - mean) / std
        return img.astype(np.float32)
    
    @staticmethod
    def to_chw_image(img: np.ndarray):
        """通道转换：HWC -> CHW"""
        img = np.array(img)
        return img.transpose((2, 0, 1))
    
    def __call__(self, img: np.ndarray):
        """完整的预处理流程"""
        img = self.resize_image(img, resize_short=256)
        img = self.crop_image(img, size=224)
        img = self.normalize_image(img)
        img = self.to_chw_image(img)
        img = img[None, ...]
        return img.astype(np.float32)


def estimate_rotation_angle(image: np.ndarray, model_path: str) -> int:
    """
    使用ONNX模型估计旋转角度（0°, 90°, 180°, 270°）
    """
    preprocessor = Preprocess()
    input_data = preprocessor(image)
    
    # ONNX推理
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    output = session.run([output_name], {input_name: input_data})
    
    # 获取预测结果
    probs = output[0][0]
    pred_idx = np.argmax(probs)
    
    # 标签映射
    angles = [0, 90, 180, 270]
    return angles[pred_idx]


# ========= 单次旋转模块 =========

def rotate_image_once(image: np.ndarray, angle: float) -> np.ndarray:
    """
    单次高质量旋转（合并去倾斜和旋转校正角度）
    
    Args:
        image: 输入图像
        angle: 旋转角度（正值逆时针，负值顺时针）
    
    Returns:
        旋转后的图像
    """
    height, width = image.shape[:2]
    
    # 计算旋转矩阵
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后需要的画布大小
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # 调整旋转矩阵的平移参数
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # 高质量旋转
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    return rotated_image


# ========= 主处理流程 =========

def process_image(image_path: str, model_path: str = "model/inference.onnx", output_path: str = None):
    """
    完整的图像处理流程：去倾斜 + 旋转校正
    
    Args:
        image_path: 输入图像路径
        model_path: 旋转校正模型路径
        output_path: 输出图像路径（可选）
    """
    print(f"\n{'='*60}")
    print(f"处理图像: {image_path}")
    print(f"{'='*60}")
    
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return None
    
    print(f"✓ 原始尺寸: {image.shape[1]} x {image.shape[0]} 像素")
    
    # 2. 估计去倾斜角度
    deskew_angle = estimate_deskew_angle(image)
    print(f"✓ 去倾斜角度: {deskew_angle:.2f}°")
    
    # 3. 估计旋转校正角度
    rotation_angle = estimate_rotation_angle(image, model_path)
    print(f"✓ 旋转校正角度: {rotation_angle}°")
    
    # 4. 合并角度（单次旋转）
    # 旋转校正是90度倍数，去倾斜是小角度，需要合并
    # 注意：旋转校正预测的是"当前图像需要逆时针旋转多少度才能正常"
    # 所以最终角度 = 去倾斜角度 + 旋转校正角度
    correction_for_deskew = -deskew_angle  # 去倾斜需要的校正（逆时针）
    total_angle = correction_for_deskew + rotation_angle  # 加上旋转校正
    print(f"✓ 合并角度: {total_angle:.2f}° (逆时针旋转)")
    
    # 5. 单次旋转
    corrected_image = rotate_image_once(image, total_angle)
    print(f"✓ 旋转后尺寸: {corrected_image.shape[1]} x {corrected_image.shape[0]} 像素")
    
    # 6. 保存结果
    if output_path is None:
        input_path = Path(image_path)
        output_dir = input_path.parent / "corrected"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_corrected{input_path.suffix}"
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_file), corrected_image)
    print(f"✓ 已保存到: {output_file}")
    print(f"{'='*60}\n")
    
    return corrected_image


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='文档图像去倾斜+旋转校正工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 处理单张图像
  python deskew_and_rotation_correction.py input.jpg
  
  # 指定模型路径和输出路径
  python deskew_and_rotation_correction.py input.jpg -m 1/inference.onnx -o output.jpg
        """
    )
    
    parser.add_argument('image', help='输入图像路径')
    parser.add_argument('-m', '--model', default='model/inference.onnx', help='ONNX模型路径')
    parser.add_argument('-o', '--output', help='输出图像路径（可选）')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.model).exists():
        print(f"错误：模型文件不存在: {args.model}")
        return
    
    # 处理图像
    process_image(args.image, args.model, args.output)


if __name__ == '__main__':
    main()
