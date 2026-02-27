#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch_npu  # noqa: F401

from method_numpy_reference import get_angle_with_timings as get_angle_numpy
from method_torch_npu import get_angle_with_timings as get_angle_npu


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


def benchmark_numpy(
    images: List[np.ndarray],
    warmup: int,
    repeat: int,
    amax: float,
    V: int,
    W: int,
    D: float,
    num: int,
) -> Dict[str, float]:
    records_ms: List[float] = []
    fft_ms: List[float] = []
    proj_ms: List[float] = []

    for img in images:
        for _ in range(max(0, warmup)):
            get_angle_numpy(img, amax=amax, V=V, W=W, D=D, num=num)

        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            _, timings = get_angle_numpy(img, amax=amax, V=V, W=W, D=D, num=num)
            cost_ms = (time.perf_counter() - t0) * 1000.0
            records_ms.append(cost_ms)
            fft_ms.append(timings["fft_total_s"] * 1000.0)
            proj_ms.append(timings["radial_projection_s"] * 1000.0)

    return {
        "variant": "numpy_ref",
        "samples": len(records_ms),
        "avg_ms": float(np.mean(records_ms)),
        "p95_ms": percentile(records_ms, 95),
        "fft_avg_ms": float(np.mean(fft_ms)),
        "proj_avg_ms": float(np.mean(proj_ms)),
    }


def benchmark_torch_npu(
    images: List[np.ndarray],
    warmup: int,
    repeat: int,
    amax: float,
    V: int,
    W: int,
    D: float,
    num: int,
    device: str,
) -> Dict[str, float]:
    records_ms: List[float] = []
    fft_ms: List[float] = []
    proj_ms: List[float] = []

    for img in images:
        for _ in range(max(0, warmup)):
            get_angle_npu(img, amax=amax, V=V, W=W, D=D, num=num, device=device)

        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            _, timings = get_angle_npu(img, amax=amax, V=V, W=W, D=D, num=num, device=device)
            cost_ms = (time.perf_counter() - t0) * 1000.0
            records_ms.append(cost_ms)
            fft_ms.append(timings["fft_total_s"] * 1000.0)
            proj_ms.append(timings["radial_projection_s"] * 1000.0)

    return {
        "variant": f"torch_npu({device})",
        "samples": len(records_ms),
        "avg_ms": float(np.mean(records_ms)),
        "p95_ms": percentile(records_ms, 95),
        "fft_avg_ms": float(np.mean(fft_ms)),
        "proj_avg_ms": float(np.mean(proj_ms)),
    }


def run_accuracy_eval(
    images: List[np.ndarray],
    amax: float,
    V: int,
    W: int,
    D: float,
    num: int,
    device: str,
) -> Dict[str, float]:
    diffs: List[float] = []
    pred_numpy: List[float] = []
    pred_npu: List[float] = []

    for img in images:
        a_np, _ = get_angle_numpy(img, amax=amax, V=V, W=W, D=D, num=num)
        a_npu, _ = get_angle_npu(img, amax=amax, V=V, W=W, D=D, num=num, device=device)
        pred_numpy.append(float(a_np))
        pred_npu.append(float(a_npu))
        diffs.append(abs(float(a_np) - float(a_npu)))

    return {
        "count": len(diffs),
        "mae": float(np.mean(diffs)) if diffs else 0.0,
        "max_abs_diff": float(np.max(diffs)) if diffs else 0.0,
        "numpy_mean_angle": float(np.mean(pred_numpy)) if pred_numpy else 0.0,
        "npu_mean_angle": float(np.mean(pred_npu)) if pred_npu else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch NPU 适配基准脚本")
    parser.add_argument("--input", type=str, default="assets", help="输入图片目录")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--amax", type=float, default=45.0)
    parser.add_argument("--V", type=int, default=2048)
    parser.add_argument("--W", type=int, default=304)
    parser.add_argument("--D", type=float, default=0.55)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--device", type=str, default="npu:0", help="npu:0 / cpu")
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

    if args.device.startswith("npu") and (not torch.npu.is_available()):
        raise RuntimeError("torch.npu 不可用，请检查 torch_npu 与运行环境")

    numpy_res = benchmark_numpy(
        images=images,
        warmup=args.warmup,
        repeat=args.repeat,
        amax=args.amax,
        V=args.V,
        W=args.W,
        D=args.D,
        num=args.num,
    )
    npu_res = benchmark_torch_npu(
        images=images,
        warmup=args.warmup,
        repeat=args.repeat,
        amax=args.amax,
        V=args.V,
        W=args.W,
        D=args.D,
        num=args.num,
        device=args.device,
    )

    numpy_res["speedup_vs_numpy_ref"] = 1.0
    npu_res["speedup_vs_numpy_ref"] = numpy_res["avg_ms"] / npu_res["avg_ms"] if npu_res["avg_ms"] > 0 else 0.0

    acc = run_accuracy_eval(
        images=images,
        amax=args.amax,
        V=args.V,
        W=args.W,
        D=args.D,
        num=args.num,
        device=args.device,
    )

    print("=" * 96)
    print(f"Benchmark images={len(images)}, repeat={args.repeat}, warmup={args.warmup}, device={args.device}")
    print("=" * 96)
    print(f"{'variant':<24} {'avg(ms)':>10} {'p95(ms)':>10} {'fft(ms)':>10} {'proj(ms)':>10} {'speedup':>10}")
    for r in [numpy_res, npu_res]:
        print(
            f"{r['variant']:<24} {r['avg_ms']:>10.3f} {r['p95_ms']:>10.3f} {r['fft_avg_ms']:>10.3f} {r['proj_avg_ms']:>10.3f} {r['speedup_vs_numpy_ref']:>10.3f}"
        )
    print("-" * 96)
    print(f"accuracy: mae={acc['mae']:.6f}, max_abs_diff={acc['max_abs_diff']:.6f}, count={acc['count']}")
    print("=" * 96)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_npu_{ts}.json"

    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "npu_available": torch.npu.is_available(),
        "npu_device_count": torch.npu.device_count() if torch.npu.is_available() else 0,
        "npu_device_name": torch.npu.get_device_name(0) if torch.npu.is_available() else "",
    }

    payload = {
        "timestamp": ts,
        "config": {
            "amax": args.amax,
            "V": args.V,
            "W": args.W,
            "D": args.D,
            "num": args.num,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "images": len(images),
            "input": str(input_path),
            "device": args.device,
        },
        "env": env,
        "results": [numpy_res, npu_res],
        "accuracy": acc,
        "samples": [str(p) for p in selected_paths],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
