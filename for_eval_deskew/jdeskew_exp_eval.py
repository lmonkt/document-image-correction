#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jdeskew exp 方法（论文复现版）评估程序

方法说明：
这是对 jdeskew 原始论文（及 reproduce.ipynb）的忠实复现。它的核心特点是自适应性（Adaptive）。

核心机制：
1. 双角度计算：该方法同时计算两个候选角度
   - a_init：基于全频谱的径向投影计算出的角度，稳定性好，但可能受文档背景或低频噪声干扰
   - a_correct：屏蔽掉频谱中心低频分量（由参数 W 控制遮挡宽度）后计算出的角度，
               对文本纹理的高频成分更敏感，潜在精度更高

2. 动态决策机制：通过比较两个角度的差异 dist = |a_init - a_correct| 与阈值 D（如 0.55）：
   - 若 dist <= D，认为中心遮挡后的结果可信，采用高精度的 a_correct
   - 若 dist > D，认为高频噪声过大或纹理不清晰，回退采用更稳健的 a_init

3. 特点：这是一种"双保险"策略，在精度和稳定性之间做了动态平衡，
         但依赖于对 W 和 D 两个超参数的调优

评估数据集：
- DISE 2021 (15°)
- DISE 2021 (45°)

评估指标：
- AED (Average Error in Degrees): 平均误差（度）
- TOP80: 前80%样本的平均误差（度）
- CE (Correct Estimation): 误差 ≤ 0.1度的样本比例
- WORST: 最大误差（度）

运行示例:
    python jdeskew_exp_eval.py
"""

import os
import glob
import json
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ========= 核心算法 (忠实还原 reproduce.ipynb) =========

def ensure_gray(image: np.ndarray) -> np.ndarray:
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return image


def ensure_optimal_square(image: np.ndarray) -> np.ndarray:
    assert image is not None, image
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
    gray = ensure_gray(image)
    opt_gray = ensure_optimal_square(gray)

    # thresh (同 notebook: ~opt_gray + 自适应阈值)
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    # perform fft
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)

    # get magnitude
    magnitude = np.abs(shifted_dft)
    return magnitude


def _get_angle_adaptive(
    m: np.ndarray, amax: Optional[float] = None, num: Optional[int] = None, W: Optional[int] = None
) -> Tuple[float, float, float]:
    """径向投影 + 中心遮挡 (与 reproduce.ipynb 一致)."""
    assert m.shape[0] == m.shape[1]
    r = c = m.shape[0] // 2

    if W is None:
        W = m.shape[0] // 10

    if amax is None:
        amax = 15

    if num is None:
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


def get_angle(
    image: np.ndarray,
    amax: Optional[float] = None,
    V: Optional[int] = None,
    W: Optional[int] = None,
    D: Optional[float] = None,
    train_D: bool = False,
) -> float:
    """获取倾斜角度 (忠实 reproduce.ipynb 逻辑)."""
    assert isinstance(image, np.ndarray), image

    if amax is None:
        amax = 45
    if V is None:
        V = 1024
    if W is None:
        W = 0
    if D is None:
        D = 0.45

    ratio = V / image.shape[0]
    image = cv2.resize(image, None, fx=ratio, fy=ratio)

    magnitude = get_fft_magnitude(image)
    a_init, a_correct, dist = _get_angle_adaptive(magnitude, amax=amax, W=W)

    if train_D:
        return a_init, a_correct, dist

    if dist <= D:
        return a_correct
    return a_init


# ========= 评估工具 =========

def parse_ground_truth_from_filename(filename: str) -> float:
    start_idx = filename.find("[")
    end_idx = filename.find("]")
    if start_idx != -1 and end_idx != -1:
        try:
            return float(filename[start_idx + 1 : end_idx])
        except Exception:
            pass
    raise ValueError(f"无法从文件名解析角度: {filename}")


def evaluate_single_image(
    image_path: str, amax: float, V: int, W: int, D: float
) -> Tuple[str, float, float, float]:
    filename = os.path.basename(image_path)
    try:
        gt = parse_ground_truth_from_filename(filename)
    except Exception:
        gt = 0.0

    image = cv2.imread(image_path)
    if image is None:
        return image_path, gt, 0.0, abs(gt)

    pd_angle = get_angle(image, amax=amax, V=V, W=W, D=D)
    error = round(abs(gt - pd_angle), 2)  # 与 notebook 一致，先 round 再统计 CE
    return image_path, gt, pd_angle, error


def _worker_eval(args):
    """顶层 worker，避免 Windows 下本地函数无法 pickle 的问题。"""
    path, amax, V, W, D = args
    return evaluate_single_image(path, amax, V, W, D)


def compute_metrics(errors: List[float]) -> Dict[str, float]:
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


class JDeskewEvaluator:
    def __init__(
        self,
        dataset_dirs: List[str],
        *,
        amax: float,
        V: int,
        W: int,
        D: float,
        num_workers: Optional[int] = None,
    ):
        self.dataset_dirs = dataset_dirs
        self.amax = amax
        self.V = V
        self.W = W
        self.D = D
        cpu_count = os.cpu_count() or 4
        self.num_workers = num_workers or max(1, int(cpu_count * 0.98))
        self.results: List[Tuple[str, float, float, float]] = []

    def collect_images(self) -> List[str]:
        all_imgs: List[str] = []
        for d in self.dataset_dirs:
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
                all_imgs.extend(glob.glob(os.path.join(d, ext)))
        return all_imgs

    def evaluate(self) -> Dict[str, float]:
        image_paths = self.collect_images()
        if not image_paths:
            print("未找到图像")
            return {}

        args = [(p, self.amax, self.V, self.W, self.D) for p in image_paths]

        with Pool(self.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_worker_eval, args),
                    total=len(image_paths),
                    desc="处理进度",
                )
            )

        self.results = [(ipath, gt, pd_angle, err) for ipath, gt, pd_angle, err in results]
        errors = [err for _, _, _, err in results]

        return compute_metrics(errors)

    def save_results(self, metrics: Dict[str, float], tag: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.dirname(os.path.abspath(__file__))

        metrics_path = os.path.join(out_dir, f"jdeskew_exp_metrics_{tag}_{ts}.json")
        details_path = os.path.join(out_dir, f"jdeskew_exp_details_{tag}_{ts}.txt")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        with open(details_path, "w", encoding="utf-8") as f:
            f.write("图像路径\t真实角度\t预测角度\t误差\n")
            for ipath, gt, pd_angle, err in sorted(
                self.results, key=lambda x: x[3], reverse=True
            ):
                f.write(f"{ipath}\t{gt:.2f}\t{pd_angle:.2f}\t{err:.2f}\n")

        print(f"指标已保存: {metrics_path}")
        print(f"详情已保存: {details_path}")


# ========= 主程序 =========

def main():
    # 数据集路径（仅保留 DISE 2021 数据集）
    dataset_15 = [r"C:\data\datasets\dise2021_15\test"]
    dataset_45 = [r"C:\data\datasets\dise2021_45\test"]

    # reproduce.ipynb 中性能最优的一组参数
    cfg_15 = {"amax": 15.0, "V": 3072, "W": 328, "D": 0.55}
    cfg_45 = {"amax": 45.0, "V": 2048, "W": 304, "D": 0.55}

    print("="*60)
    print("jdeskew exp 方法（论文复现版）评估")
    print("="*60)
    
    print("\n# 评估 DISE 2021 (15°)")
    eval15 = JDeskewEvaluator(dataset_15, **cfg_15)
    metrics15 = eval15.evaluate()
    if metrics15:
        print(metrics15)
        eval15.save_results(metrics15, tag="dise15")

    print("\n# 评估 DISE 2021 (45°)")
    eval45 = JDeskewEvaluator(dataset_45, **cfg_45)
    metrics45 = eval45.evaluate()
    if metrics45:
        print(metrics45)
        eval45.save_results(metrics45, tag="dise45")

    # 汇总打印
    if metrics15 and metrics45:
        print("\n" + "="*60)
        print("评估结果汇总 (jdeskew exp 方法)")
        print("="*60)
        print(f"{'数据集':<18} {'AED':>6} {'TOP80':>8} {'CE':>8} {'WORST':>8}")
        print("-"*60)
        print(f"{'DISE 2021 (15°)':<18} {metrics15['AED']:>6.2f} {metrics15['TOP80']:>8.2f} {metrics15['CE']:>8.2f} {metrics15['WORST']:>8.2f}")
        print(f"{'DISE 2021 (45°)':<18} {metrics45['AED']:>6.2f} {metrics45['TOP80']:>8.2f} {metrics45['CE']:>8.2f} {metrics45['WORST']:>8.2f}")
        print("="*60)


if __name__ == "__main__":
    main()
