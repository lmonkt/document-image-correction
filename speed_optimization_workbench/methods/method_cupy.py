#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Tuple

import cv2
import cupy as cp
import numpy as np


def ensure_gray(image: np.ndarray) -> np.ndarray:
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return image


def ensure_optimal_square(image: np.ndarray) -> np.ndarray:
    nw = nh = cv2.getOptimalDFTSize(max(image.shape[:2]))
    return cv2.copyMakeBorder(
        src=image,
        top=0,
        bottom=nh - image.shape[0],
        left=0,
        right=nw - image.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )


def get_fft_magnitude_gpu(image: np.ndarray) -> cp.ndarray:
    gray = ensure_gray(image)
    opt_gray = ensure_optimal_square(gray)
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    g = cp.asarray(opt_gray, dtype=cp.float32)
    dft = cp.fft.fft2(g)
    shifted = cp.fft.fftshift(dft)
    return cp.abs(shifted)


def _get_angle_adaptive_gpu(m: cp.ndarray, amax: float, num: int, W: int) -> Tuple[float, float, float]:
    r = c = m.shape[0] // 2
    tr = cp.linspace(-amax, amax, int(amax * num * 2), dtype=cp.float32) / 180.0 * cp.pi

    x = cp.arange(r, dtype=cp.float32)
    cos_t = cp.cos(tr)[:, None]
    sin_t = cp.sin(tr)[:, None]

    y = c + (x[None, :] * cos_t).astype(cp.int32)
    x_proj = c + (-x[None, :] * sin_t).astype(cp.int32)

    valid = (y >= 0) & (y < m.shape[0]) & (x_proj >= 0) & (x_proj < m.shape[1])
    gathered = cp.zeros_like(y, dtype=cp.float32)
    gathered[valid] = m[y[valid], x_proj[valid]]

    li_init = cp.sum(gathered, axis=1)
    li_correct = cp.sum(gathered[:, W:], axis=1) if (W > 0 and W < gathered.shape[1]) else li_init

    a_init = float((tr[cp.argmax(li_init)] / cp.pi * 180.0).get())
    a_correct = float((tr[cp.argmax(li_correct)] / cp.pi * 180.0).get())
    dist = abs(a_init - a_correct)
    return -a_init, -a_correct, dist


def get_angle(
    image: np.ndarray,
    amax: float = 45.0,
    V: int = 2048,
    W: int = 304,
    D: float = 0.55,
    num: int = 20,
) -> float:
    ratio = V / image.shape[0]
    image = cv2.resize(image, None, fx=ratio, fy=ratio)

    magnitude = get_fft_magnitude_gpu(image)
    a_init, a_correct, dist = _get_angle_adaptive_gpu(magnitude, amax=amax, num=num, W=W)
    cp.cuda.Stream.null.synchronize()
    return a_correct if dist <= D else a_init


def get_angle_from_path(
    image_path: str,
    amax: float = 45.0,
    V: int = 2048,
    W: int = 304,
    D: float = 0.55,
    num: int = 20,
) -> float:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return get_angle(image, amax=amax, V=V, W=W, D=D, num=num)


def main() -> None:
    parser = argparse.ArgumentParser(description="jdeskew CuPy GPU 方法（独立版）")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--amax", type=float, default=45.0)
    parser.add_argument("--V", type=int, default=2048)
    parser.add_argument("--W", type=int, default=304)
    parser.add_argument("--D", type=float, default=0.55)
    parser.add_argument("--num", type=int, default=20)
    args = parser.parse_args()

    angle = get_angle_from_path(
        args.image,
        amax=args.amax,
        V=args.V,
        W=args.W,
        D=args.D,
        num=args.num,
    )
    print(f"pred_angle={angle:.4f}")


if __name__ == "__main__":
    main()
