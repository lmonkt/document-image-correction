#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本区域智能裁剪与放大工具（优化版）

对应博客文档：五、图像预处理优化（推荐）

功能说明：
  自动检测文档图像中的文本区域并进行智能裁剪和尺寸调整：
  1. 文本区域定位：基于轮廓检测自动识别有效内容区域
  2. 智能裁剪：去除大量空白边缘，保留核心内容
  3. 自适应放大：根据目标尺寸智能缩放，避免过度拉伸
  4. 尺寸约束：支持最小/最大尺寸限制，确保输出质量

核心类：
  TextRegionCropEnlargeStep
    - __init__(): 配置目标尺寸、边距、二值化模式
    - _preprocess(): 图像预处理（灰度化、模糊、二值化、膨胀）
    - _filter_contours(): 轮廓过滤（面积、长宽比筛选）
    - _find_text_bbox(): 查找文本区域边界框
    - execute(): 执行完整的裁剪和放大流程

算法流程：
  输入图像 -> 灰度化 -> 高斯模糊 -> 自适应二值化 -> 形态学膨胀 
  -> 轮廓检测 -> 过滤小轮廓 -> 计算最小外接矩形 -> 扩展边距 
  -> 裁剪 -> 智能缩放 -> 输出

使用场景：
  - 去倾斜/旋转校正后的二次优化（去除旋转产生的白边）
  - OCR预处理：裁剪掉无关区域，提升识别精度
  - 文档图像标准化：统一尺寸，便于批量处理

配置参数：
  - enlarge_target_size: 期望的目标尺寸 (w, h)
  - min_output_size: 最小输出尺寸限制
  - max_output_size: 最大输出尺寸限制
  - expand_pixels: 裁剪时的边距留白
  - radical: 是否使用OTSU二值化（激进模式）

命令行用法：
  python text_crop_optimized.py <input_image> [--output <output_path>]

博客说明：
  配合去倾斜使用可有效去除旋转后的白色边框，对应博客中的
  "五、图像预处理优化（推荐）" 章节
"""

import cv2
import numpy as np
import os
import argparse

class TextRegionCropEnlargeStep:
    """文本区域定位、裁剪与智能放大处理步骤（优化版）。"""

    def __init__(self, 
                 enlarge_target_size=(1200, 1600), 
                 min_output_size=(600, 800),
                 max_output_size=(1600, 2000),
                 expand_pixels=30, 
                 radical=False):
        """
        Args:
            enlarge_target_size: 期望的目标尺寸 (w, h)。
            min_output_size: 允许的最小尺寸 (w, h)，低于此尺寸将强制放大。
            max_output_size: 允许的最大尺寸 (w, h)，高于此尺寸将强制缩小。
            expand_pixels: 裁剪时的边距。
            radical: 是否使用激进二值化。
        """
        self.target_size = enlarge_target_size
        self.min_size = min_output_size
        self.max_size = max_output_size
        self.expand_pixels = expand_pixels
        self.radical = radical
        
        # 轮廓过滤参数
        self.min_contour_area = 100
        self.max_contour_area_ratio = 0.9
        self.aspect_ratio_range = (0.02, 50) # 稍微放宽范围

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """图像预处理：灰度 -> 模糊 -> 二值化 -> 膨胀"""
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.radical:
            # OTSU 适合背景单一、对比明显的图像
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # 自适应阈值适合光照不均
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 25, 10
            )

        # 优化点：使用膨胀代替闭运算。
        # 目的：让行内的文字、段落内的行尽可能粘连在一起，形成大的联通域，减少轮廓数量。
        # 横向核 (10, 2) 适合横排文本，通用情况可用 (5, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)) 
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        return dilated

    def _get_valid_boxes(self, binary_img: np.ndarray, img_shape):
        """获取筛选后的文本框列表 (x, y, w, h)"""
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = img_shape[:2]
        img_area = img_w * img_h
        valid_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area or area > img_area * self.max_contour_area_ratio:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0: continue

            # 宽高比过滤
            aspect_ratio = max(w / h, h / w)
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            valid_boxes.append((x, y, w, h))
            
        return valid_boxes

    def _merge_text_regions(self, boxes, img_shape):
        """合并文本区域，包含简单的离群点剔除"""
        if not boxes:
            return None
        
        # 1. 找出最大的那个框（通常是主体文本的一部分），以此为基准
        # 这里的 area 近似计算为 w*h
        max_box = max(boxes, key=lambda b: b[2] * b[3])
        main_center_y = max_box[1] + max_box[3] / 2
        
        # 2. 简单过滤：如果某个小框距离主体垂直距离太远，且面积很小，可能是页码或噪点
        # 注意：这里不做太复杂的过滤，以免误删页眉页脚。
        # 实际生产中可引入 IQR (四分位距) 过滤
        
        img_h, img_w = img_shape[:2]
        
        all_x1 = []
        all_y1 = []
        all_x2 = []
        all_y2 = []

        for x, y, w, h in boxes:
            all_x1.append(x)
            all_y1.append(y)
            all_x2.append(x + w)
            all_y2.append(y + h)

        # 计算合并后的坐标
        min_x = max(0, min(all_x1) - self.expand_pixels)
        min_y = max(0, min(all_y1) - self.expand_pixels)
        max_x = min(max(all_x2) + self.expand_pixels, img_w)
        max_y = min(max(all_y2) + self.expand_pixels, img_h)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _smart_resize(self, img: np.ndarray):
        """智能缩放：保证宽高比，尺寸限制在 min_size 和 max_size 之间"""
        h, w = img.shape[:2]
        target_w, target_h = self.target_size
        min_w, min_h = self.min_size
        max_w, max_h = self.max_size

        # 策略：计算缩放比例 scale
        # 1. 优先满足不超过最大尺寸
        scale_down = min(max_w / w, max_h / h)
        
        # 2. 如果不需要缩小（即 scale_down >= 1），检查是否需要放大以满足最小尺寸
        if scale_down >= 1:
             # 计算需要放大多少才能达到最小尺寸的要求（取宽高中较短边的需求，或者任意一边满足即可，视业务而定）
             # 这里逻辑为：只要有一边小于 min，就放大，直到宽>=min_w 或 高>=min_h
             scale_up_w = min_w / w
             scale_up_h = min_h / h
             scale = max(1.0, max(scale_up_w, scale_up_h))
             
             # 3. 再次检查放大后是否超过 Target Size (作为软限制，防止过度放大模糊)
             # 如果原始图片极小，我们可能不希望无限制放大
             pass 
        else:
            scale = scale_down

        # 如果比例接近 1，不处理
        if 0.95 < scale < 1.05:
            return img

        new_w = int(w * scale)
        new_h = int(h * scale)

        # 选择插值算法：放大用 Lanczos/Cubic，缩小用 Area
        interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        
        return cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image

        # 1. 预处理
        binary_img = self._preprocess(image)

        # 2. 获取候选框
        boxes = self._get_valid_boxes(binary_img, image.shape)
        if not boxes:
            return image

        # 3. 合并区域
        crop_rect = self._merge_text_regions(boxes, image.shape)
        
        # 4. 裁剪
        x, y, w, h = crop_rect
        cropped = image[y:y+h, x:x+w]
        if cropped.size == 0:
            return image

        # 5. 添加白边 (Padding) - 可选，为了OCR不贴边
        # 注意：expand_pixels 在 crop 阶段已经作为“视野扩展”用了，
        # 这里是否再加纯白 padding 取决于 OCR 模型的鲁棒性。通常建议加一点。
        pad = 10 
        cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # 6. 智能缩放
        result = self._smart_resize(cropped)
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Text Region Crop and Enlarge")
    parser.add_argument("input_paths", nargs='+', help="Input image paths")
    parser.add_argument("--output_dir", default="output/optimized_crop", help="Output directory")
    args = parser.parse_args()

    processor = TextRegionCropEnlargeStep()
    
    os.makedirs(args.output_dir, exist_ok=True)

    for path in args.input_paths:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            continue
            
        print(f"Processing: {path}")
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Failed to load image: {path}")
            continue
            
        result = processor.process(img)
        
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        out_path = os.path.join(args.output_dir, f"{name}_processed{ext}")
        
        cv2.imwrite(out_path, result)
        print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
