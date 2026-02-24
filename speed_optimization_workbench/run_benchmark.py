#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from jdeskew_variants import (
    AngleConfig,
    get_angle_with_cupy,
    get_angle_with_variant,
    is_cupy_available,
)


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    paths: List[Path] = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
        paths.extend(sorted(input_path.glob(ext)))
    return paths


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def benchmark_variant(
    images: List[np.ndarray],
    cfg: AngleConfig,
    variant: str,
    warmup: int,
    repeat: int,
) -> Dict[str, float]:
    records_ms: List[float] = []
    fft_ms: List[float] = []
    proj_ms: List[float] = []

    for img in images:
        for _ in range(max(0, warmup)):
            if variant == "cupy":
                get_angle_with_cupy(img, cfg)
            else:
                get_angle_with_variant(img, cfg, variant)

        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            if variant == "cupy":
                _, timings = get_angle_with_cupy(img, cfg)
            else:
                _, timings = get_angle_with_variant(img, cfg, variant)
            cost_ms = (time.perf_counter() - t0) * 1000.0
            records_ms.append(cost_ms)
            fft_ms.append(timings["fft_total_s"] * 1000.0)
            proj_ms.append(timings["radial_projection_s"] * 1000.0)

    return {
        "variant": variant,
        "samples": len(records_ms),
        "avg_ms": float(np.mean(records_ms)),
        "p95_ms": percentile(records_ms, 95),
        "fft_avg_ms": float(np.mean(fft_ms)),
        "proj_avg_ms": float(np.mean(proj_ms)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="jdeskew 速度优化基准脚本")
    parser.add_argument("--input", type=str, default="assets", help="输入图片目录")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--amax", type=float, default=45.0)
    parser.add_argument("--V", type=int, default=2048)
    parser.add_argument("--W", type=int, default=304)
    parser.add_argument("--D", type=float, default=0.55)
    parser.add_argument(
        "--stage",
        type=str,
        choices=["cpu", "cupy"],
        default="cpu",
        help="cpu: original/cpu_numpy_vec/cpu_numba_jit; cupy: 在cpu基础上再加cupy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    image_paths = collect_images(input_path)
    if not image_paths:
        raise RuntimeError(f"未找到图像: {input_path}")

    selected_paths = image_paths[: max(1, args.max_images)]
    images: List[np.ndarray] = []
    for p in selected_paths:
        im = cv2.imread(str(p))
        if im is not None:
            images.append(im)

    if not images:
        raise RuntimeError("图像读取失败")

    cfg = AngleConfig(amax=args.amax, V=args.V, W=args.W, D=args.D)

    variants = ["original", "cpu_numpy_vec", "cpu_numba_jit"]
    if args.stage == "cupy":
        if is_cupy_available():
            variants.append("cupy")
        else:
            print("[WARN] CuPy 未安装，跳过 cupy 变体")

    all_results: List[Dict[str, float]] = []
    for v in variants:
        print(f"[RUN] variant={v}")
        res = benchmark_variant(images, cfg, v, args.warmup, args.repeat)
        all_results.append(res)

    base = next(r for r in all_results if r["variant"] == "original")
    for r in all_results:
        r["speedup_vs_original"] = base["avg_ms"] / r["avg_ms"] if r["avg_ms"] > 0 else 0.0

    print("=" * 88)
    print(f"Benchmark stage={args.stage}, images={len(images)}, repeat={args.repeat}")
    print("=" * 88)
    print(f"{'variant':<16} {'avg(ms)':>10} {'p95(ms)':>10} {'fft(ms)':>10} {'proj(ms)':>10} {'speedup':>10}")
    for r in all_results:
        print(
            f"{r['variant']:<16} {r['avg_ms']:>10.3f} {r['p95_ms']:>10.3f} {r['fft_avg_ms']:>10.3f} {r['proj_avg_ms']:>10.3f} {r['speedup_vs_original']:>10.3f}"
        )
    print("=" * 88)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_{args.stage}_{ts}.json"

    payload = {
        "timestamp": ts,
        "stage": args.stage,
        "config": {
            "amax": args.amax,
            "V": args.V,
            "W": args.W,
            "D": args.D,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "images": len(images),
            "input": str(input_path),
        },
        "results": all_results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
