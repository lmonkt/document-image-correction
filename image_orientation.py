#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档图像方向分类模块

对应博客文档：三、文档图像方向分类与矫正

功能说明：
  使用轻量级深度学习模型（ONNX格式）对文档图像进行方向分类：
  - 分类结果：0°, 90°, 180°, 270° 四个方向
  - 输出概率分布和置信度，便于阈值过滤误判
  - 预处理流程：Resize(短边256) -> CenterCrop(224) -> Normalize -> CHW格式转换

核心类：
  - Preprocess: 完整的图像预处理pipeline（对应PaddleClas的标准流程）
  - ImageOrientationClassifier: 封装ONNX推理引擎，提供方向预测接口

模型来源：
  基于 PaddleClas 训练的文档方向分类模型
  模型文件：model/inference.onnx

使用场景：
  - 单独对方向错误的文档进行矫正（如横向文档被误存为竖向）
  - 与去倾斜算法结合，处理任意角度的旋转问题
  - 可设置置信度阈值，对低置信度结果不进行旋转，避免误判

调用示例：
  classifier = ImageOrientationClassifier(model_path='model/inference.onnx')
  angle_index, confidence = classifier.predict(image)
  # angle_index: 0=0°, 1=90°, 2=180°, 3=270°
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class Preprocess:
    """图像预处理流程"""
    def __init__(self):
        pass
    
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
        # 1. 缩放短边到256
        img = self.resize_image(img, resize_short=256)
        
        # 2. 中心裁剪为224x224
        img = self.crop_image(img, size=224)
        
        # 3. 归一化
        img = self.normalize_image(img)
        
        # 4. 转换为CHW格式
        img = self.to_chw_image(img)
        
        # 5. 增加batch维度：(3, 224, 224) -> (1, 3, 224, 224)
        img = img[None, ...]
        
        return img.astype(np.float32)


class Postprocess:
    """后处理：Top-1 + 标签映射（不使用 Softmax）"""
    def __init__(self):
        # 方向标签映射
        self.label_list = ['0°', '90°', '180°', '270°']
    
    def __call__(self, output: np.ndarray):
        """
        处理模型输出
        Args:
            output: 模型输出，形状 (1, 4)
        Returns:
            angle: 预测的角度
            confidence: 对应的置信度
        """
        # 获取batch第一个样本的输出（不进行 Softmax）
        probs = output[0]
        
        # Top-1：获取最大概率对应的索引
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        
        # 标签映射
        angle = self.label_list[pred_idx]
        
        return angle, confidence


def predict(image_path: str, model_path: str = "rapid_orientation.onnx"):
    """
    推理函数
    Args:
        image_path: 输入图像路径
        model_path: ONNX模型路径
    """
    # 1. 加载图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    
    # 2. 预处理
    preprocessor = Preprocess()
    input_data = preprocessor(img)
    
    # 3. ONNX推理
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    output = session.run([output_name], {input_name: input_data})
    
    # 4. 后处理
    postprocessor = Postprocess()
    angle, confidence = postprocessor(output[0])
    
    # 5. 输出结果
    print(f"预测方向：{angle}")
    print(f"置信度：{confidence:.4f}")
    
    return angle, confidence


if __name__ == "__main__":
    # 使用示例：预测并根据结果校正保存图像（最小必要实现）
    image_path = "data/rot/3.png"
    model_path = "1/inference.onnx"

    angle, confidence = predict(image_path, model_path=model_path)

    # 读取原图并根据预测结果进行校正并保存
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    angle_num = int(str(angle).replace("°", ""))

    if angle_num == 90:
        corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle_num == 180:
        corrected_img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle_num == 270:
        corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        corrected_img = img

    output_path = "data/rot/2_corrected.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, corrected_img)
    print(f"已保存纠正后的图像到: {output_path}")
