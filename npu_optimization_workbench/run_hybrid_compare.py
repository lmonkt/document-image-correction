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
import torch
import torch_npu  # noqa: F401

from method_hybrid_torch_asnumpy import get_angle_with_timings as get_angle_hybrid
from method_numpy_reference import get_angle_with_timings as get_angle_numpy
from method_torch_npu import get_angle_with_timings as get_angle_torch_npu


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


def run_variant(
    variant: str,
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

    fn = {
        "numpy_ref": get_angle_numpy,
        "torch_npu": get_angle_torch_npu,
        "hybrid_torch_fft_asnumpy": get_angle_hybrid,
    }[variant]

    for img in images:
        for _ in range(max(0, warmup)):
            if variant == "numpy_ref":
                fn(img, amax=amax, V=V, W=W, D=D, num=num)
            else:
                fn(img, amax=amax, V=V, W=W, D=D, num=num, device=device)

        for _ in range(max(1, repeat)):
            t0 = time.perf_counter()
            if variant == "numpy_ref":
                _, timings = fn(img, amax=amax, V=V, W=W, D=D, num=num)
            else:
                _, timings = fn(img, amax=amax, V=V, W=W, D=D, num=num, device=device)
            records_ms.append((time.perf_counter() - t0) * 1000.0)
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


def eval_accuracy(
    images: List[np.ndarray],
    amax: float,
    V: int,
    W: int,
    D: float,
    num: int,
    device: str,
) -> Dict[str, float]:
    diffs_torch: List[float] = []
    diffs_hybrid: List[float] = []

    for img in images:
        ref, _ = get_angle_numpy(img, amax=amax, V=V, W=W, D=D, num=num)
        t, _ = get_angle_torch_npu(img, amax=amax, V=V, W=W, D=D, num=num, device=device)
        h, _ = get_angle_hybrid(img, amax=amax, V=V, W=W, D=D, num=num, device=device)
        diffs_torch.append(abs(float(t) - float(ref)))
        diffs_hybrid.append(abs(float(h) - float(ref)))

    return {
        "torch_npu_mae": float(np.mean(diffs_torch)) if diffs_torch else 0.0,
        "torch_npu_max_abs": float(np.max(diffs_torch)) if diffs_torch else 0.0,
        "hybrid_mae": float(np.mean(diffs_hybrid)) if diffs_hybrid else 0.0,
        "hybrid_max_abs": float(np.max(diffs_hybrid)) if diffs_hybrid else 0.0,
        "count": len(diffs_torch),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="混合方案对比：numpy / torch_npu / hybrid")
    parser.add_argument("--input", type=str, default="assets")
    parser.add_argument("--max-images", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--amax", type=float, default=45.0)
    parser.add_argument("--V", type=int, default=2048)
    parser.add_argument("--W", type=int, default=304)
    parser.add_argument("--D", type=float, default=0.55)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--device", type=str, default="npu:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("npu") and (not torch.npu.is_available()):
        raise RuntimeError("torch.npu 不可用")

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

    variants = ["numpy_ref", "torch_npu", "hybrid_torch_fft_asnumpy"]
    all_results: List[Dict[str, float]] = []
    for v in variants:
        all_results.append(
            run_variant(
                variant=v,
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
        )

    base = next(r for r in all_results if r["variant"] == "numpy_ref")
    for r in all_results:
        r["speedup_vs_numpy_ref"] = base["avg_ms"] / r["avg_ms"] if r["avg_ms"] > 0 else 0.0

    acc = eval_accuracy(
        images=images,
        amax=args.amax,
        V=args.V,
        W=args.W,
        D=args.D,
        num=args.num,
        device=args.device,
    )

    print("=" * 108)
    print(f"Hybrid compare images={len(images)}, repeat={args.repeat}, warmup={args.warmup}, device={args.device}")
    print("=" * 108)
    print(f"{'variant':<30} {'avg(ms)':>10} {'p95(ms)':>10} {'fft(ms)':>10} {'proj(ms)':>10} {'speedup':>10}")
    for r in all_results:
        print(
            f"{r['variant']:<30} {r['avg_ms']:>10.3f} {r['p95_ms']:>10.3f} {r['fft_avg_ms']:>10.3f} {r['proj_avg_ms']:>10.3f} {r['speedup_vs_numpy_ref']:>10.3f}"
        )
    print("-" * 108)
    print(
        f"accuracy vs numpy_ref: torch_npu mae={acc['torch_npu_mae']:.6e}, hybrid mae={acc['hybrid_mae']:.6e}, count={acc['count']}"
    )
    print("=" * 108)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"hybrid_compare_{ts}.json"

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
        "results": all_results,
        "accuracy": acc,
        "samples": [str(p) for p in selected_paths],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
