#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My 方法（Coarse-to-Fine 两阶段搜索）评估程序

方法说明：
个人基于 jdeskew exp 进行的改造。不同于原论文的"阈值切换"逻辑，
采用了更确定性的 Coarse-to-Fine（粗细两阶段）搜索策略。

核心机制：
1. 两阶段搜索：
   - 粗略估计（Coarse）：首先在较大范围内（如 -90° 到 90°），以较大步长（如 1.0°）
                        快速定位能量最强的粗略角度
   - 精细估计（Fine）：在粗估角度的邻域内（如 ±2°），以极小步长（如 0.1°）
                       进行密集插值搜索，锁定最终精确角度

2. 思路演变：
   - 不再纠结于 a_init 和 a_correct 的选择困境（不再需要阈值 D 进行回退判断）
   - 通过"粗定位+精搜索"的机制，实质上强制执行了类似 a_correct 的高精度路径，
     同时避免了单次细粒度搜索带来的巨大计算开销
   - 代码中虽然保留了小幅度的中心遮挡（半径为 2），但主要依靠搜索策略的提升来保证精度

3. 特点：这种方法更加"自信"且确定性强，通过增加计算密度（两轮搜索）换取更高的角度分辨率。
        准确率比 lib 好但不如 exp，速度比较快

评估数据集：
- DISE 2021 (15°)
- DISE 2021 (45°)

评估指标：
- AED (Average Error in Degrees): 平均误差（度）
- TOP80: 前80%样本的平均误差（度）
- CE (Correct Estimation): 误差 ≤ 0.1度的样本比例
- WORST: 最大误差（度）

运行示例:
    python my_method_eval.py
"""

import os
import glob
import cv2
import numpy as np
from multiprocessing import Pool, Manager
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime

# 导入我们的角度估计器
from deskew_angle_only import get_skew_angle


def parse_ground_truth_from_filename(filename: str) -> float:
    """
    从文件名中解析真实角度值
    
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


def evaluate_single_image(image_path: str, max_skew_angle: float = 45.0) -> Tuple[float, float, float]:
    """
    评估单张图像
    
    参数:
        image_path: 图像路径
        max_skew_angle: 最大倾斜角度
        
    返回:
        Tuple[float, float, float]: (真实角度, 预测角度, 误差)
    """
    try:
        # 从文件名获取真实角度
        filename = os.path.basename(image_path)
        ground_truth = parse_ground_truth_from_filename(filename)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            return ground_truth, 0.0, abs(ground_truth)
        
        # 预测角度
        predicted_angle = get_skew_angle(image, max_skew_angle=max_skew_angle)
        
        # 计算误差（取绝对值），并按参考实现四舍五入到 2 位小数用于 CE 统计
        error = round(abs(ground_truth - predicted_angle), 2)
        
        return ground_truth, predicted_angle, error
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return 0.0, 0.0, 0.0


def worker_function(args):
    """多进程工作函数"""
    image_path, max_skew_angle, error_list = args
    gt, pred, error = evaluate_single_image(image_path, max_skew_angle)
    error_list.append(error)
    return (image_path, gt, pred, error)


class MyMethodEvaluator:
    """My 方法评估器"""
    
    def __init__(self, dataset_dirs: List[str], max_skew_angle: float = 45.0, num_workers: Optional[int] = None):
        """
        初始化评估器
        
        参数:
            dataset_dirs: 数据集目录列表
            max_skew_angle: 最大倾斜角度（15 或 45）
            num_workers: 并行进程数，None表示使用CPU核心数的70%
        """
        self.dataset_dirs = dataset_dirs
        self.max_skew_angle = max_skew_angle
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 4  # 默认值
        self.num_workers = num_workers or int(cpu_count * 0.7)
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
        manager = Manager()
        error_list = manager.list()
        
        # 准备参数
        args_list = [(path, self.max_skew_angle, error_list) for path in image_paths]
        
        # 并行处理
        with Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_function, args_list),
                total=len(image_paths),
                desc="处理进度"
            ))
        
        self.results = results
        
        # 计算指标
        errors = list(error_list)
        
        if not errors:
            print("错误: 没有有效的评估结果")
            return {}
        
        # 计算各项指标
        aed = sum(errors) / len(errors)  # 平均误差
        
        # TOP80: 前80%的平均误差
        sorted_errors = sorted(errors)
        top80_count = int(len(sorted_errors) * 0.8)
        top80_errors = sorted_errors[:top80_count]
        top80 = sum(top80_errors) / len(top80_errors) if top80_errors else 0.0
        
        # CE: 误差≤0.1度的比例
        ce = sum(1 for e in errors if e <= 0.1) / len(errors)
        
        # WORST: 最大误差
        worst = max(errors)
        
        metrics = {
            'AED': round(aed, 2),
            'TOP80': round(top80, 2),
            'CE': round(ce, 2),
            'WORST': round(worst, 2),
            'total_images': len(errors),
            'max_skew_angle': self.max_skew_angle
        }
        
        return metrics
    
    def save_results(self, metrics: Dict, tag: str):
        """
        保存详细结果到文件
        
        参数:
            metrics: 评估指标
            tag: 文件名标签（如 'dise15'）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 保存指标
        metrics_file = os.path.join(output_dir, f"my_method_metrics_{tag}_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"评估指标已保存到: {metrics_file}")
        
        # 保存详细结果
        details_file = os.path.join(output_dir, f"my_method_details_{tag}_{timestamp}.txt")
        with open(details_file, 'w', encoding='utf-8') as f:
            f.write("图像路径\t真实角度\t预测角度\t误差\n")
            for image_path, gt, pred, error in sorted(self.results, key=lambda x: x[3], reverse=True):
                f.write(f"{image_path}\t{gt:.2f}\t{pred:.2f}\t{error:.2f}\n")
        print(f"详细结果已保存到: {details_file}")


def main():
    """主函数"""
    print("="*60)
    print("My 方法（Coarse-to-Fine 两阶段搜索）评估")
    print("="*60)
    
    # 数据集路径（仅保留 DISE 2021 数据集）
    dataset_15_dirs = [r"C:\data\datasets\dise2021_15\test"]
    dataset_45_dirs = [r"C:\data\datasets\dise2021_45\test"]
    
    # 评估 DISE 2021 (15度)
    print("\n# 评估 DISE 2021 (15°)")
    evaluator_15 = MyMethodEvaluator(dataset_15_dirs, max_skew_angle=15.0)
    metrics_15 = evaluator_15.evaluate()
    if metrics_15:
        print(f"\n总图像数:   {metrics_15['total_images']}")
        print(f"AED:        {metrics_15['AED']:.2f}°")
        print(f"TOP80:      {metrics_15['TOP80']:.2f}°")
        print(f"CE (≤0.1°): {metrics_15['CE']:.2%}")
        print(f"WORST:      {metrics_15['WORST']:.2f}°")
        evaluator_15.save_results(metrics_15, tag='dise15')
    
    # 评估 DISE 2021 (45度)
    print("\n# 评估 DISE 2021 (45°)")
    evaluator_45 = MyMethodEvaluator(dataset_45_dirs, max_skew_angle=45.0)
    metrics_45 = evaluator_45.evaluate()
    if metrics_45:
        print(f"\n总图像数:   {metrics_45['total_images']}")
        print(f"AED:        {metrics_45['AED']:.2f}°")
        print(f"TOP80:      {metrics_45['TOP80']:.2f}°")
        print(f"CE (≤0.1°): {metrics_45['CE']:.2%}")
        print(f"WORST:      {metrics_45['WORST']:.2f}°")
        evaluator_45.save_results(metrics_45, tag='dise45')
    
    # 汇总打印
    if metrics_15 and metrics_45:
        print("\n" + "="*60)
        print("评估结果汇总 (My 方法)")
        print("="*60)
        print(f"{'数据集':<20} {'AED':>6} {'TOP80':>8} {'CE':>8} {'WORST':>8}")
        print("-"*60)
        print(f"{'DISE 2021 (15°)':<20} {metrics_15['AED']:>6.2f} {metrics_15['TOP80']:>8.2f} {metrics_15['CE']:>8.2%} {metrics_15['WORST']:>8.2f}")
        print(f"{'DISE 2021 (45°)':<20} {metrics_45['AED']:>6.2f} {metrics_45['TOP80']:>8.2f} {metrics_45['CE']:>8.2%} {metrics_45['WORST']:>8.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
