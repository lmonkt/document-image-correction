#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import asnumpy as ap
import numpy as np


@dataclass
class BenchConfig:
    rows: int
    cols: int
    warmup: int
    repeat: int
    seed: int


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def bench_numpy_mul_sum(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.multiply(a, b)))


def bench_asnumpy_mul_sum(a: np.ndarray, b: np.ndarray) -> float:
    a_npu = ap.ndarray.from_numpy(a)
    b_npu = ap.ndarray.from_numpy(b)
    c_npu = ap.multiply(a_npu, b_npu)
    return float(ap.sum(c_npu))


def bench_numpy_trig_reduce(x: np.ndarray) -> float:
    return float(np.sum(np.sin(x) + np.cos(x)))


def bench_asnumpy_trig_reduce(x: np.ndarray) -> float:
    x_npu = ap.ndarray.from_numpy(x)
    y_npu = ap.sin(x_npu)
    z_npu = ap.cos(x_npu)
    s_npu = ap.add(y_npu, z_npu)
    return float(ap.sum(s_npu))


def run_case(
    name: str,
    cpu_fn: Callable[..., float],
    npu_fn: Callable[..., float],
    cpu_args: Tuple,
    npu_args: Tuple,
    warmup: int,
    repeat: int,
) -> Dict[str, float]:
    cpu_times: List[float] = []
    npu_times: List[float] = []

    for _ in range(max(0, warmup)):
        cpu_fn(*cpu_args)
        npu_fn(*npu_args)

    cpu_last = 0.0
    npu_last = 0.0
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        cpu_last = cpu_fn(*cpu_args)
        cpu_times.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        npu_last = npu_fn(*npu_args)
        npu_times.append((time.perf_counter() - t1) * 1000.0)

    abs_diff = abs(cpu_last - npu_last)
    cpu_avg = float(np.mean(cpu_times))
    npu_avg = float(np.mean(npu_times))

    return {
        "case": name,
        "cpu_avg_ms": cpu_avg,
        "cpu_p95_ms": percentile(cpu_times, 95),
        "npu_avg_ms": npu_avg,
        "npu_p95_ms": percentile(npu_times, 95),
        "speedup": cpu_avg / npu_avg if npu_avg > 0 else 0.0,
        "last_abs_diff": abs_diff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AsNumpy 加速基准（NumPy vs AsNumpy）")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260225)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchConfig(
        rows=args.rows,
        cols=args.cols,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
    )

    np.random.seed(cfg.seed)
    a = np.random.normal(0.0, 1.0, size=(cfg.rows, cfg.cols)).astype(np.float32)
    b = np.random.normal(0.0, 1.0, size=(cfg.rows, cfg.cols)).astype(np.float32)
    x = np.random.uniform(-np.pi, np.pi, size=(cfg.rows, cfg.cols)).astype(np.float32)

    results = [
        run_case(
            name="multiply+sum",
            cpu_fn=bench_numpy_mul_sum,
            npu_fn=bench_asnumpy_mul_sum,
            cpu_args=(a, b),
            npu_args=(a, b),
            warmup=cfg.warmup,
            repeat=cfg.repeat,
        ),
        run_case(
            name="sin+cos+add+sum",
            cpu_fn=bench_numpy_trig_reduce,
            npu_fn=bench_asnumpy_trig_reduce,
            cpu_args=(x,),
            npu_args=(x,),
            warmup=cfg.warmup,
            repeat=cfg.repeat,
        ),
    ]

    print("=" * 96)
    print(
        f"AsNumpy benchmark rows={cfg.rows}, cols={cfg.cols}, warmup={cfg.warmup}, repeat={cfg.repeat}"
    )
    print("=" * 96)
    print(f"{'case':<24} {'numpy(ms)':>12} {'asnumpy(ms)':>12} {'speedup':>10} {'abs_diff':>14}")
    for r in results:
        print(
            f"{r['case']:<24} {r['cpu_avg_ms']:>12.3f} {r['npu_avg_ms']:>12.3f} {r['speedup']:>10.3f} {r['last_abs_diff']:>14.6e}"
        )
    print("=" * 96)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"asnumpy_benchmark_{ts}.json"

    payload = {
        "timestamp": ts,
        "config": {
            "rows": cfg.rows,
            "cols": cfg.cols,
            "warmup": cfg.warmup,
            "repeat": cfg.repeat,
            "seed": cfg.seed,
        },
        "results": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
