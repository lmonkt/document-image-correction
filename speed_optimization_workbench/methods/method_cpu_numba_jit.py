#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Tuple

import cv2
import numpy as np
from numba import njit


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


def get_fft_magnitude(image: np.ndarray) -> np.ndarray:
    gray = ensure_gray(image)
    opt_gray = ensure_optimal_square(gray)
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)
    return np.abs(shifted_dft)


@njit(cache=True, fastmath=True)
def _radial_projection_numba_core(m: np.ndarray, tr: np.ndarray, W: int) -> Tuple[np.ndarray, np.ndarray]:
    r = m.shape[0] // 2
    c = r
    li_init = np.zeros(tr.shape[0], dtype=np.float64)
    li_correct = np.zeros(tr.shape[0], dtype=np.float64)

    for i in range(tr.shape[0]):
        t = tr[i]
        s0 = 0.0
        s1 = 0.0
        for j in range(r):
            y = c + int(j * np.cos(t))
            x_proj = c + int(-j * np.sin(t))
            if y < 0 or y >= m.shape[0] or x_proj < 0 or x_proj >= m.shape[1]:
                continue
            v = m[y, x_proj]
            s0 += v
            if j >= W:
                s1 += v
        li_init[i] = s0
        li_correct[i] = s1 if W > 0 else s0
    return li_init, li_correct


def _get_angle_adaptive(m: np.ndarray, amax: float, num: int, W: int) -> Tuple[float, float, float]:
    tr = np.linspace(-amax, amax, int(amax * num * 2), dtype=np.float64) / 180.0 * np.pi
    li_init, li_correct = _radial_projection_numba_core(m.astype(np.float64), tr, int(W))

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
    magnitude = get_fft_magnitude(image)
    a_init, a_correct, dist = _get_angle_adaptive(magnitude, amax=amax, num=num, W=W)
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
    parser = argparse.ArgumentParser(description="jdeskew CPU Numba JIT 方法（独立版）")
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
