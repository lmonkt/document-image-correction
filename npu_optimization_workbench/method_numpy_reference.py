#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import cv2
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


def get_fft_magnitude_numpy(image: np.ndarray) -> np.ndarray:
    gray = ensure_gray(image)
    opt_gray = ensure_optimal_square(gray)
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)
    return np.abs(shifted_dft)


def _get_angle_adaptive_numpy(m: np.ndarray, amax: float, num: int, W: int) -> Tuple[float, float, float]:
    r = c = m.shape[0] // 2
    tr = np.linspace(-amax, amax, max(2, int(amax * num * 2)), dtype=np.float64) / 180.0 * np.pi

    x = np.arange(r, dtype=np.float64)
    cos_t = np.cos(tr)[:, None]
    sin_t = np.sin(tr)[:, None]

    y = c + np.int32(x[None, :] * cos_t)
    x_proj = c + np.int32(-x[None, :] * sin_t)

    valid = (y >= 0) & (y < m.shape[0]) & (x_proj >= 0) & (x_proj < m.shape[1])
    gathered = np.zeros_like(y, dtype=np.float64)
    gathered[valid] = m[y[valid], x_proj[valid]]

    li_init = np.sum(gathered, axis=1)
    li_correct = np.sum(gathered[:, W:], axis=1) if (W > 0 and W < gathered.shape[1]) else li_init

    a_init = tr[np.argmax(li_init)] / np.pi * 180.0
    a_correct = tr[np.argmax(li_correct)] / np.pi * 180.0
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
    magnitude = get_fft_magnitude_numpy(image)
    a_init, a_correct, dist = _get_angle_adaptive_numpy(magnitude, amax=amax, num=num, W=W)
    return a_correct if dist <= D else a_init


def get_angle_with_timings(
    image: np.ndarray,
    amax: float = 45.0,
    V: int = 2048,
    W: int = 304,
    D: float = 0.55,
    num: int = 20,
) -> Tuple[float, Dict[str, float]]:
    timings: Dict[str, float] = {}

    t0 = cv2.getTickCount()
    ratio = V / image.shape[0]
    resized = cv2.resize(image, None, fx=ratio, fy=ratio)
    t1 = cv2.getTickCount()
    timings["resize_s"] = (t1 - t0) / cv2.getTickFrequency()

    t2 = cv2.getTickCount()
    magnitude = get_fft_magnitude_numpy(resized)
    t3 = cv2.getTickCount()
    timings["fft_total_s"] = (t3 - t2) / cv2.getTickFrequency()

    t4 = cv2.getTickCount()
    a_init, a_correct, dist = _get_angle_adaptive_numpy(magnitude, amax=amax, num=num, W=W)
    t5 = cv2.getTickCount()
    timings["radial_projection_s"] = (t5 - t4) / cv2.getTickFrequency()

    t6 = cv2.getTickCount()
    angle = a_correct if dist <= D else a_init
    t7 = cv2.getTickCount()
    timings["decision_s"] = (t7 - t6) / cv2.getTickFrequency()

    timings["algo_total_s"] = timings["resize_s"] + timings["fft_total_s"] + timings["radial_projection_s"] + timings["decision_s"]
    return angle, timings


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
    parser = argparse.ArgumentParser(description="NumPy 参考实现")
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
