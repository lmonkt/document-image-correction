#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jdeskew exp 单进程性能分析脚本

目标：
1. 单进程串行执行（便于定位真实热点）
2. 输出端到端与分阶段耗时统计（read / resize / FFT / radial projection / decision）
3. 兼容 py-spy 火焰图与 line_profiler（kernprof）

示例：
  python for_eval_deskew/jdeskew_exp_single_process_profile.py \
    --input assets \
    --max-images 20 \
    --warmup 3 \
    --repeat 5 \
    --save-json

line_profiler：
  kernprof -l -v for_eval_deskew/jdeskew_exp_single_process_profile.py \
    --input assets --max-images 5 --warmup 1 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):  # type: ignore
        return func


@profile
def ensure_gray(image: np.ndarray) -> np.ndarray:
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return image


@profile
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


@profile
def get_fft_magnitude_timed(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    gray = ensure_gray(image)
    timings["gray_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    opt_gray = ensure_optimal_square(gray)
    timings["optimal_square_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )
    timings["adaptive_threshold_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)
    magnitude = np.abs(shifted_dft)
    timings["fft_and_magnitude_s"] = time.perf_counter() - t0

    timings["fft_total_s"] = (
        timings["gray_s"]
        + timings["optimal_square_s"]
        + timings["adaptive_threshold_s"]
        + timings["fft_and_magnitude_s"]
    )
    return magnitude, timings


@profile
def get_angle_adaptive_timed(
    m: np.ndarray,
    amax: Optional[float] = None,
    num: Optional[int] = None,
    W: Optional[int] = None,
) -> Tuple[float, float, float, Dict[str, float]]:
    timings: Dict[str, float] = {}

    assert m.shape[0] == m.shape[1]
    r = c = m.shape[0] // 2

    if W is None:
        W = m.shape[0] // 10
    if amax is None:
        amax = 15
    if num is None:
        num = 20

    t0 = time.perf_counter()
    tr = np.linspace(-1 * amax, amax, int(amax * num * 2)) / 180 * np.pi

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
    timings["radial_projection_s"] = time.perf_counter() - t0

    return -1 * a_init, -1 * a_correct, dist, timings


@profile
def get_angle_timed(
    image: np.ndarray,
    amax: Optional[float] = None,
    V: Optional[int] = None,
    W: Optional[int] = None,
    D: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    assert isinstance(image, np.ndarray), image

    if amax is None:
        amax = 45
    if V is None:
        V = 1024
    if W is None:
        W = 0
    if D is None:
        D = 0.45

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    ratio = V / image.shape[0]
    image = cv2.resize(image, None, fx=ratio, fy=ratio)
    timings["resize_s"] = time.perf_counter() - t0

    magnitude, fft_timings = get_fft_magnitude_timed(image)
    timings.update(fft_timings)

    a_init, a_correct, dist, proj_timings = get_angle_adaptive_timed(
        magnitude, amax=amax, W=W
    )
    timings.update(proj_timings)

    t0 = time.perf_counter()
    angle = a_correct if dist <= D else a_init
    timings["decision_s"] = time.perf_counter() - t0

    timings["algo_total_s"] = (
        timings["resize_s"]
        + timings["fft_total_s"]
        + timings["radial_projection_s"]
        + timings["decision_s"]
    )
    return angle, timings


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    paths: List[Path] = []
    for ext in exts:
        paths.extend(sorted(input_path.glob(ext)))
    return paths


def aggregate_stage_timings(stage_records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not stage_records:
        return {}

    keys = sorted({k for rec in stage_records for k in rec.keys()})
    result: Dict[str, float] = {}
    for key in keys:
        values = [rec[key] for rec in stage_records if key in rec]
        result[f"{key}_mean_ms"] = float(np.mean(values) * 1000)
        result[f"{key}_p95_ms"] = float(np.percentile(values, 95) * 1000)
    return result


def print_summary(records: Sequence[Dict[str, float]], stage_stats: Dict[str, float]) -> None:
    if not records:
        print("无有效记录")
        return

    total_ms = [r["total_s"] * 1000 for r in records]
    algo_ms = [r["algo_total_s"] * 1000 for r in records]
    io_ms = [r["read_s"] * 1000 for r in records]

    print("=" * 72)
    print("jdeskew exp 单进程性能汇总")
    print("=" * 72)
    print(f"样本数                : {len(records)}")
    print(f"端到端平均耗时 (ms)    : {np.mean(total_ms):.3f}")
    print(f"端到端P95耗时 (ms)     : {np.percentile(total_ms, 95):.3f}")
    print(f"算法平均耗时 (ms)      : {np.mean(algo_ms):.3f}")
    print(f"读图平均耗时 (ms)      : {np.mean(io_ms):.3f}")

    print("\n分阶段平均耗时 / P95 (ms):")
    ordered = [
        "resize_s",
        "gray_s",
        "optimal_square_s",
        "adaptive_threshold_s",
        "fft_and_magnitude_s",
        "fft_total_s",
        "radial_projection_s",
        "decision_s",
        "algo_total_s",
    ]
    for key in ordered:
        mk = f"{key}_mean_ms"
        pk = f"{key}_p95_ms"
        if mk in stage_stats and pk in stage_stats:
            print(f"  - {key:<24}: {stage_stats[mk]:>10.3f} / {stage_stats[pk]:>10.3f}")
    print("=" * 72)


@profile
def run_profile(
    image_paths: Sequence[Path],
    amax: float,
    V: int,
    W: int,
    D: float,
    warmup: int,
    repeat: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    records: List[Dict[str, float]] = []
    stage_records: List[Dict[str, float]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        for _ in range(max(0, warmup)):
            get_angle_timed(image, amax=amax, V=V, W=W, D=D)

        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            image2 = cv2.imread(str(image_path))
            read_s = time.perf_counter() - t0
            if image2 is None:
                continue

            t1 = time.perf_counter()
            pred_angle, timings = get_angle_timed(image2, amax=amax, V=V, W=W, D=D)
            algo_wall_s = time.perf_counter() - t1

            total_s = read_s + algo_wall_s
            rec = {
                "read_s": read_s,
                "algo_total_s": timings["algo_total_s"],
                "algo_wall_s": algo_wall_s,
                "total_s": total_s,
                "pred_angle": pred_angle,
            }
            rec.update(timings)
            records.append(rec)
            stage_records.append(timings)

    return records, stage_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="jdeskew exp 单进程性能分析")
    parser.add_argument("--input", type=str, required=True, help="输入图片或目录")
    parser.add_argument("--max-images", type=int, default=20, help="最多分析图片数量")
    parser.add_argument("--amax", type=float, default=45.0)
    parser.add_argument("--V", type=int, default=2048)
    parser.add_argument("--W", type=int, default=304)
    parser.add_argument("--D", type=float, default=0.55)
    parser.add_argument("--warmup", type=int, default=2, help="每图预热次数")
    parser.add_argument("--repeat", type=int, default=3, help="每图重复次数")
    parser.add_argument("--save-json", action="store_true", help="保存统计结果")
    return parser.parse_args()


def save_json_report(
    output_dir: Path,
    args: argparse.Namespace,
    used_images: Sequence[Path],
    records: Sequence[Dict[str, float]],
    stage_stats: Dict[str, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"jdeskew_exp_single_profile_{ts}.json"

    payload = {
        "timestamp": ts,
        "config": {
            "amax": args.amax,
            "V": args.V,
            "W": args.W,
            "D": args.D,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "input": str(args.input),
        "used_images": [str(p) for p in used_images],
        "num_records": len(records),
        "stage_stats": stage_stats,
        "records": records,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return report_path


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    image_paths = collect_images(input_path)
    if not image_paths:
        raise RuntimeError(f"未在路径中找到图像: {input_path}")

    used_images = image_paths[: max(1, args.max_images)]
    print(f"将分析 {len(used_images)} 张图像（单进程）")

    records, stage_records = run_profile(
        image_paths=used_images,
        amax=args.amax,
        V=args.V,
        W=args.W,
        D=args.D,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    stage_stats = aggregate_stage_timings(stage_records)
    print_summary(records, stage_stats)

    if args.save_json:
        out_dir = Path(__file__).resolve().parent / "profile_reports"
        report = save_json_report(out_dir, args, used_images, records, stage_stats)
        print(f"JSON 报告已保存: {report}")


if __name__ == "__main__":
    main()
