#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合速度评估代码

在 DISE 2021 数据集上对比三种倾斜角度估计算法的速度：
1. My 方法 (deskew_angle_only.py) - Coarse-to-Fine 两阶段搜索
2. jdeskew lib 方法 (jdeskew.estimator.get_angle) - 官方库版
3. jdeskew exp 方法 (jdeskew_exp_eval.py) - 论文复现版（多种参数配置）

每个数据集使用前50张有效图片进行测试

方法说明：
- My 方法：采用粗细两阶段搜索策略，准确率比 lib 好但不如 exp，速度较快
- jdeskew lib 方法：开箱即用的通用版本，计算直接，速度稍快
- jdeskew exp 方法：论文复现版，使用自适应双角度计算机制，精度最高
"""

import os
import cv2
import time
import numpy as np
from typing import List, Tuple, Dict
import glob

# 导入三种算法
import jdeskew_exp_eval
import deskew_angle_only
from jdeskew.estimator import get_angle as jdeskew_get_angle


# 数据集配置（仅保留 DISE 2021 数据集）
DATASETS = [
    {
        "name": "DISE 2021 (15°)",
        "path": r"C:\data\datasets\dise2021_15\test",
        "pattern": "*.png"
    },
    {
        "name": "DISE 2021 (45°)",
        "path": r"C:\data\datasets\dise2021_45\test",
        "pattern": "*.png"
    },
]

# jdeskew exp 方法的参数配置（5种）
JDESKEW_EXP_CONFIGS = [
    {"name": "V1024_W274", "amax": 45, "V": 1024, "W": 274, "D": 0.7},
    {"name": "V1500_W328", "amax": 45, "V": 1500, "W": 328, "D": 0.55},
    {"name": "V2048_W304", "amax": 45, "V": 2048, "W": 304, "D": 0.55},
    {"name": "V3072_W328", "amax": 45, "V": 3072, "W": 328, "D": 0.55},
    {"name": "V4096_W250", "amax": 45, "V": 4096, "W": 250, "D": 0.5},
]

NUM_IMAGES_PER_DATASET = 50
WARMUP_RUNS = 2
MAX_ANGLE = 45.0


def load_images_from_dataset(dataset: Dict) -> List[np.ndarray]:
    """
    从数据集加载前N张有效图片
    
    参数:
        dataset: 数据集配置字典
        
    返回:
        List[np.ndarray]: 图片列表
    """
    path = dataset["path"]
    pattern = dataset["pattern"]
    
    if not os.path.exists(path):
        print(f"警告: 数据集路径不存在 {path}")
        return []
    
    # 获取文件列表
    file_pattern = os.path.join(path, pattern)
    files = glob.glob(file_pattern)
    files.sort()
    files = files[:NUM_IMAGES_PER_DATASET]
    
    print(f"  正在加载 {len(files)} 张图片...")
    images = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            images.append(img)
    
    print(f"  成功加载 {len(images)} 张图片")
    return images


def benchmark_algorithm(images: List[np.ndarray], algo_name: str, algo_func, warmup: bool = True) -> float:
    """
    对单个算法进行速度测试
    
    参数:
        images: 图片列表
        algo_name: 算法名称
        algo_func: 算法函数
        warmup: 是否进行热身
        
    返回:
        float: 平均耗时（秒）
    """
    if not images:
        return 0.0
    
    # 热身
    if warmup:
        for img in images[:WARMUP_RUNS]:
            _ = algo_func(img)
    
    # 正式测试
    start_time = time.perf_counter()
    for img in images:
        _ = algo_func(img)
    total_time = time.perf_counter() - start_time
    avg_time = total_time / len(images)
    
    return avg_time


def run_comprehensive_benchmark():
    """运行综合速度评估"""
    print("="*80)
    print("综合速度评估 - 三种算法在 DISE 2021 数据集上的性能对比")
    print("="*80)
    print(f"每个数据集使用前 {NUM_IMAGES_PER_DATASET} 张有效图片")
    print(f"热身运行: {WARMUP_RUNS} 次")
    print("\n三种方法：")
    print("  1. My 方法 (deskew_angle_only) - Coarse-to-Fine 两阶段搜索")
    print("  2. jdeskew lib 方法 - 官方库版")
    print("  3. jdeskew exp 方法 - 论文复现版（多种参数配置）")
    print("="*80)
    
    # 存储所有结果
    results = {}
    
    # 对每个数据集进行测试
    for dataset in DATASETS:
        dataset_name = dataset["name"]
        print(f"\n{'#'*80}")
        print(f"# 数据集: {dataset_name}")
        print(f"# 路径: {dataset['path']}")
        print(f"{'#'*80}")
        
        # 加载图片
        images = load_images_from_dataset(dataset)
        if not images:
            print(f"跳过数据集 {dataset_name}（无有效图片）")
            continue
        
        avg_height = np.mean([img.shape[0] for img in images])
        avg_width = np.mean([img.shape[1] for img in images])
        print(f"  平均图片尺寸: {avg_width:.0f} x {avg_height:.0f}")
        
        dataset_results = {}
        
        # 1. 测试 My 方法 (deskew_angle_only.py)
        print("\n  测试算法 1: My 方法 (deskew_angle_only)")
        algo_func = lambda img: deskew_angle_only.get_skew_angle(img, max_skew_angle=MAX_ANGLE)
        avg_time = benchmark_algorithm(images, "my_method", algo_func)
        dataset_results["my_method"] = avg_time
        print(f"    平均耗时: {avg_time:.4f} 秒")
        
        # 2. 测试 jdeskew lib 方法
        print("\n  测试算法 2: jdeskew lib 方法 (jdeskew.estimator.get_angle)")
        algo_func = lambda img: -jdeskew_get_angle(img, angle_max=MAX_ANGLE)
        avg_time = benchmark_algorithm(images, "jdeskew_lib", algo_func)
        dataset_results["jdeskew_lib"] = avg_time
        print(f"    平均耗时: {avg_time:.4f} 秒")
        
        # 3. 测试 jdeskew exp 方法 (多种配置)
        print("\n  测试算法 3: jdeskew exp 方法 (论文复现版，多种配置)")
        for config in JDESKEW_EXP_CONFIGS:
            config_name = config["name"]
            amax = config["amax"]
            V = config["V"]
            W = config["W"]
            D = config["D"]
            
            algo_func = lambda img: jdeskew_exp_eval.get_angle(img, amax=amax, V=V, W=W, D=D)
            avg_time = benchmark_algorithm(images, f"jdeskew_exp_{config_name}", algo_func, warmup=False)
            dataset_results[f"jdeskew_exp_{config_name}"] = avg_time
            print(f"    配置 {config_name} (V={V}, W={W}, D={D}): {avg_time:.4f} 秒")
        
        results[dataset_name] = dataset_results
    
    # 输出汇总结果
    print("\n" + "="*80)
    print("汇总结果 - 各算法在各数据集上的平均耗时（秒）")
    print("="*80)
    
    # 准备表头
    if results:
        all_algo_names = list(next(iter(results.values())).keys())
        
        # 打印表头
        print(f"\n{'数据集':<25}", end="")
        for algo_name in all_algo_names:
            print(f" | {algo_name:<20}", end="")
        print()
        print("-" * (25 + len(all_algo_names) * 23))
        
        # 打印每个数据集的结果
        for dataset_name, dataset_results in results.items():
            print(f"{dataset_name:<25}", end="")
            for algo_name in all_algo_names:
                avg_time = dataset_results.get(algo_name, 0.0)
                print(f" | {avg_time:>20.4f}", end="")
            print()
        
        print("="*80)
        
        # 计算每个算法的总平均耗时
        print("\n各算法的总平均耗时（秒）：")
        print("-"*80)
        for algo_name in all_algo_names:
            total_avg = np.mean([results[ds][algo_name] for ds in results.keys()])
            print(f"  {algo_name:<35}: {total_avg:.4f}")
        print("="*80)
        
        # 找出最快的算法
        avg_times = {algo_name: np.mean([results[ds][algo_name] for ds in results.keys()]) 
                     for algo_name in all_algo_names}
        fastest_algo = min(avg_times, key=avg_times.get)
        fastest_time = avg_times[fastest_algo]
        
        print(f"\n最快算法: {fastest_algo} (平均 {fastest_time:.4f} 秒)")
        print("\n性能比较（相对于最快算法）：")
        print("-"*80)
        for algo_name, avg_time in sorted(avg_times.items(), key=lambda x: x[1]):
            if avg_time > 0 and fastest_time > 0:
                ratio = avg_time / fastest_time
                slower_pct = (ratio - 1) * 100
                print(f"  {algo_name:<35}: {ratio:.2f}x", end="")
                if slower_pct > 0.1:
                    print(f" (慢 {slower_pct:.1f}%)")
                else:
                    print(" (最快)")
            else:
                print(f"  {algo_name:<35}: N/A")
        print("="*80)


if __name__ == "__main__":
    run_comprehensive_benchmark()
