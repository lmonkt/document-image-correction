#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本裁剪流程可视化工具

对应博客文档：五、图像预处理优化（算法调试辅助）

功能说明：
  将 text_crop_optimized.py 的处理流程可视化，逐步展示每个图像处理步骤：
  STEP 1: 灰度化 (Grayscale)
  STEP 2: 高斯模糊 (Gaussian Blur) - 去除噪点
  STEP 3: 自适应二值化 (Adaptive Threshold) - 分离前景和背景
  STEP 4: 形态学膨胀 (Dilation) - 连接文字笔画
  STEP 5: 轮廓检测与可视化 (Contour Detection)
  STEP 6: 候选区域绘制 (Bounding Box Visualization)
  STEP 7: 最终裁剪结果 (Final Cropped Result)

使用场景：
  - 调试文本裁剪算法参数（如二值化阈值、膨胀核大小）
  - 理解每一步图像处理的效果
  - 为新场景调优裁剪策略
  - 教学演示图像处理流程

输出：
  在指定目录下保存每个步骤的中间结果图像：
  01_gray.jpg, 02_blurred.jpg, 03_binary.jpg, 04_dilated.jpg,
  05_contours_vis.jpg, 06_candidates_vis.jpg, 07_final_crop.jpg

命令行用法：
  python visualize_crop_pipeline.py <input_image> <output_dir>

与主流程的关系：
  本脚本是 text_crop_optimized.py 的可视化版本，用于理解算法原理
  实际应用中请使用 text_crop_optimized.py
"""

import cv2
import numpy as np
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def visualize_pipeline(image_path, output_dir):
    ensure_dir(output_dir)
    print(f"处理图片: {image_path}")
    print(f"结果将保存在: {output_dir}")

    # 读取原图
    img = cv2.imread(image_path)
    if img is None:
        print("错误: 无法读取图片")
        return

    # 1. 灰度化
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    cv2.imwrite(os.path.join(output_dir, "01_gray.jpg"), gray)
    print("STEP 1: 灰度化完成")

    # 2. 高斯模糊
    # 作用：去除噪点，平滑图像，由 (5,5) 改为 (9,9) 可视化效果更明显
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "02_blurred.jpg"), blurred)
    print("STEP 2: 高斯模糊完成")

    # 3. 二值化 (使用自适应阈值)
    # 作用：将图像变为只有黑白两色。cv2.THRESH_BINARY_INV 使文字变白，背景变黑，方便找轮廓
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 25, 10
    )
    cv2.imwrite(os.path.join(output_dir, "03_binary.jpg"), binary)
    print("STEP 3: 二值化完成")

    # 4. 膨胀
    # 作用：让离散的文字笔画粘连成一个整体块
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)) 
    dilated = cv2.dilate(binary, kernel, iterations=2)
    cv2.imwrite(os.path.join(output_dir, "04_dilated.jpg"), dilated)
    print("STEP 4: 膨胀完成")

    # 5. 可视化候选框
    # 这一步是为了找出所有可能的文字区域
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 复制一份原图用于画框
    debug_img_candidates = img.copy()
    
    img_h, img_w = img.shape[:2]
    img_area = img_w * img_h
    valid_boxes = []

    # 过滤参数
    min_contour_area = 100
    max_contour_area_ratio = 0.9
    aspect_ratio_range = (0.02, 50)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 记录所有轮廓（黄色，细线），用于对比
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(debug_img_candidates, (x, y), (x+w, y+h), (0, 255, 255), 1)

        # 过滤
        if area < min_contour_area or area > img_area * max_contour_area_ratio:
            continue
        
        if w == 0 or h == 0: continue

        aspect_ratio = max(w / h, h / w)
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue
        
        valid_boxes.append((x, y, w, h))
        # 绘制有效轮廓（绿色，粗线）
        cv2.rectangle(debug_img_candidates, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_dir, "05_candidates.jpg"), debug_img_candidates)
    print(f"STEP 5: 候选框提取完成 (找到 {len(valid_boxes)} 个有效区域)")

    # 6. 合并区域
    if not valid_boxes:
        print("未找到有效文本区域")
        return

    all_x1 = [x for x, y, w, h in valid_boxes]
    all_y1 = [y for x, y, w, h in valid_boxes]
    all_x2 = [x + w for x, y, w, h in valid_boxes]
    all_y2 = [y + h for x, y, w, h in valid_boxes]

    expand_pixels = 30
    min_x = max(0, min(all_x1) - expand_pixels)
    min_y = max(0, min(all_y1) - expand_pixels)
    max_x = min(max(all_x2) + expand_pixels, img_w)
    max_y = min(max(all_y2) + expand_pixels, img_h)

    # 绘制最终合并框（红色）
    debug_img_merged = img.copy()
    cv2.rectangle(debug_img_merged, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
    cv2.imwrite(os.path.join(output_dir, "06_merged_region.jpg"), debug_img_merged)
    print("STEP 6: 区域合并完成")

    # 7. 最终裁剪并添加白边
    crop_w = max_x - min_x
    crop_h = max_y - min_y
    cropped = img[min_y:max_y, min_x:max_x]
    
    pad = 10
    final_result = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.imwrite(os.path.join(output_dir, "07_final_result.jpg"), final_result)
    print("STEP 7: 最终裁剪完成")

if __name__ == "__main__":
    input_image = "quqinxie/data/skew_images/corrected/page_46_45_37.9_corrected.png"
    output_directory = "output/debug_steps"
    
    if os.path.exists(input_image):
        visualize_pipeline(input_image, output_directory)
    else:
        print(f"文件不存在: {input_image}")
