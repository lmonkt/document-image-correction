#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jdeskew lib 方法（官方库版）评估程序

方法说明：
这是作者发布的 Python 库（pip install jdeskew）中的实现。相较于 exp 方法，
它进行了大幅简化，旨在提供一个开箱即用的通用版本。

核心机制：
1. 核心简化：移除了中心遮挡参数 W 和决策阈值 D，直接对整个频谱图进行径向投影积分，
            仅返回单一角度

2. 与 exp 的关系：从代码逻辑看，它等同于只保留了 exp 方法中的 a_init 计算分支，
                完全去除了自适应校正逻辑

3. 特点：计算更直接，速度稍快，且无需调参。但在复杂背景或文本纹理较弱的场景下，
        由于缺乏 a_correct 的高频增强和 D 的异常回退机制，精度可能不如 exp 方法极致

评估数据集：
- DISE 2021 (15°)
- DISE 2021 (45°)

评估指标：
- AED (Average Error in Degrees): 平均误差（度）
- TOP80: 前80%样本的平均误差（度）
- CE (Correct Estimation): 误差 ≤ 0.1度的样本比例
- WORST: 最大误差（度）

运行示例:
    python jdeskew_lib_eval.py
"""

import os
import glob
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime

# 导入 jdeskew 库
from jdeskew.estimator import get_angle


def parse_ground_truth_from_filename(filename: str) -> float:
    """
    从文件名中解析真实角度值（DISE 数据集格式）
    
    文件名格式: xxx[angle].png 或 xxx[angle].jpg
    例如: image[5.5].png 表示真实角度为 5.5 度
    
    参数:
        filename: 文件名（含扩展名）
        
    返回:
        float: 真实角度值
    """
    try:
        # 找到方括号中的内容
        start_idx = filename.find('[')
        end_idx = filename.find(']')
        if start_idx != -1 and end_idx != -1:
            angle_str = filename[start_idx + 1:end_idx]
            return float(angle_str)
        else:
            raise ValueError(f"无法从文件名中解析角度: {filename}")
    except Exception as e:
        print(f"警告: 解析文件名 {filename} 时出错: {e}")
        return 0.0


def evaluate_single_image(image_path: str) -> Tuple[str, float, float, float]:
    """
    评估单张图像
    
    参数:
        image_path: 图像路径
        
    返回:
        Tuple: (图像路径, 真实角度, 预测角度, 误差)
    """
    try:
        # 从文件名获取真实角度
        filename = os.path.basename(image_path)
        ground_truth = parse_ground_truth_from_filename(filename)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            return image_path, ground_truth, 0.0, abs(ground_truth)
        
        # 使用 jdeskew 库预测角度，angle_max 设置为 45
        # 注意：jdeskew 库返回的角度需要取反才是图像的倾斜角度
        predicted_angle = -get_angle(image, angle_max=45.0)
        
        # 计算误差（取绝对值），按参考实现四舍五入到 2 位小数
        error = round(abs(ground_truth - predicted_angle), 2)
        
        return image_path, ground_truth, predicted_angle, error
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return image_path, 0.0, 0.0, 0.0


def _worker_eval(image_path):
    """多进程工作函数"""
    return evaluate_single_image(image_path)


def compute_metrics(errors: List[float]) -> Dict:
    """
    计算评估指标
    
    参数:
        errors: 误差列表
        
    返回:
        Dict: 包含所有指标的字典
    """
    if not errors:
        return {}
    
    aed = sum(errors) / len(errors)
    sorted_errors = sorted(errors)
    top80_count = max(1, int(len(sorted_errors) * 0.8))
    top80 = sum(sorted_errors[:top80_count]) / top80_count
    ce = sum(1 for e in errors if e <= 0.1) / len(errors)
    worst = max(errors)
    
    return {
        "AED": round(aed, 2),
        "TOP80": round(top80, 2),
        "CE": round(ce, 2),
        "WORST": round(worst, 2),
        "total_images": len(errors),
    }


class JDeskewLibEvaluator:
    """jdeskew 库评估器"""
    
    def __init__(self, dataset_dirs: List[str], num_workers: Optional[int] = None):
        """
        初始化评估器
        
        参数:
            dataset_dirs: 数据集目录列表
            num_workers: 并行进程数，None表示使用CPU核心数的70%
        """
        self.dataset_dirs = dataset_dirs
        cpu_count = os.cpu_count() or 4
        self.num_workers = num_workers or max(1, int(cpu_count * 0.7))
        self.results = []
        
    def collect_images(self) -> List[str]:
        """收集所有图像路径"""
        all_images = []
        for dataset_dir in self.dataset_dirs:
            if not os.path.exists(dataset_dir):
                print(f"警告: 数据集目录不存在 {dataset_dir}")
                continue
                
            # 支持多种图像格式
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
                pattern = os.path.join(dataset_dir, ext)
                images = glob.glob(pattern)
                all_images.extend(images)
        
        return all_images
    
    def evaluate(self) -> Dict:
        """
        执行评估
        
        返回:
            Dict: 包含所有评估指标的字典
        """
        # 收集图像
        image_paths = self.collect_images()
        if not image_paths:
            print("错误: 未找到任何图像文件")
            return {}
        
        # 使用多进程处理
        with Pool(self.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_worker_eval, image_paths),
                    total=len(image_paths),
                    desc="处理进度",
                )
            )
        
        self.results = results
        errors = [err for _, _, _, err in results]
        
        return compute_metrics(errors)
    
    def save_results(self, metrics: Dict, dataset_name: str):
        """
        保存评估结果
        
        参数:
            metrics: 评估指标字典
            dataset_name: 数据集名称（用于文件名）
        """
        if not metrics:
            return
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.dirname(os.path.abspath(__file__))
        
        metrics_path = os.path.join(out_dir, f"jdeskew_lib_metrics_{dataset_name}_{ts}.json")
        details_path = os.path.join(out_dir, f"jdeskew_lib_details_{dataset_name}_{ts}.txt")
        
        # 保存指标为 JSON
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # 保存详细信息
        with open(details_path, "w", encoding="utf-8") as f:
            f.write("图像路径\t真实角度\t预测角度\t误差\n")
            for ipath, gt, pd_angle, err in sorted(self.results, key=lambda x: x[3], reverse=True):
                f.write(f"{ipath}\t{gt:.2f}\t{pd_angle:.2f}\t{err:.2f}\n")
        
        print(f"指标已保存: {metrics_path}")
        print(f"详情已保存: {details_path}")


def main():
    print("="*60)
    print("jdeskew lib 方法（官方库版）评估")
    print("="*60)
    print("使用 jdeskew.estimator.get_angle() 函数")
    print("angle_max 参数设置为 45.0")
    print("="*60)
    
    # 数据集配置（仅保留 DISE 2021 数据集）
    datasets = [
        {
            "name": "dise2021_15",
            "dirs": [r"C:\data\datasets\dise2021_15\test"],
            "description": "DISE 2021 (15°)",
        },
        {
            "name": "dise2021_45",
            "dirs": [r"C:\data\datasets\dise2021_45\test"],
            "description": "DISE 2021 (45°)",
        },
    ]
    
    # 显示配置信息
    print("\n配置的数据集路径:")
    print("-"*60)
    for ds in datasets:
        print(f"\n{ds['description']}:")
        for d in ds['dirs']:
            exists = "✓" if os.path.exists(d) else "✗"
            print(f"  {exists} {d}")
    print("-"*60)
    
    # 存储所有结果用于对比
    all_results = {}
    
    # 逐个评估数据集
    for ds in datasets:
        dataset_name = ds['name']
        dirs = ds['dirs']
        description = ds['description']
        
        # 检查目录是否存在
        if not any(os.path.exists(d) for d in dirs):
            print(f"\n⚠ 警告: 数据集 {description} 的目录不存在，跳过")
            continue
        
        print(f"\n{'#'*60}")
        print(f"# 评估数据集: {description}")
        print(f"{'#'*60}")
        
        evaluator = JDeskewLibEvaluator(dirs)
        metrics = evaluator.evaluate()
        
        if metrics:
            print(f"\n总图像数:   {metrics['total_images']}")
            print(f"AED:        {metrics['AED']:.2f}°")
            print(f"TOP80:      {metrics['TOP80']:.2f}°")
            print(f"CE (≤0.1°): {metrics['CE']:.2%}")
            print(f"WORST:      {metrics['WORST']:.2f}°")
            evaluator.save_results(metrics, dataset_name=dataset_name)
            all_results[description] = metrics
    
    # 汇总对比
    if len(all_results) > 0:
        print("\n" + "="*60)
        print("评估结果汇总 (jdeskew lib 方法)")
        print("="*60)
        print(f"{'数据集':<20} {'AED':>6} {'TOP80':>8} {'CE':>8} {'WORST':>8}")
        print("-"*60)
        for description, metrics in all_results.items():
            print(f"{description:<20} {metrics['AED']:>6.2f} {metrics['TOP80']:>8.2f} "
                  f"{metrics['CE']:>8.2%} {metrics['WORST']:>8.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
