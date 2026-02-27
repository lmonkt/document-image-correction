#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch_npu  # noqa: F401


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


def get_fft_magnitude_npu(image: np.ndarray, device: torch.device) -> torch.Tensor:
    gray = ensure_gray(image)
    opt_gray = ensure_optimal_square(gray)
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    g = torch.from_numpy(opt_gray).to(device=device, dtype=torch.float32)
    dft = torch.fft.fft2(g)
    shifted = torch.fft.fftshift(dft)
    real = shifted.real
    imag = shifted.imag
    return torch.sqrt(real * real + imag * imag)


def _get_angle_adaptive_npu(
    m: torch.Tensor,
    amax: float,
    num: int,
    W: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    r = c = m.shape[0] // 2
    tr = torch.linspace(-amax, amax, max(2, int(amax * num * 2)), dtype=torch.float32, device=device)
    tr = tr / 180.0 * torch.pi

    x = torch.arange(r, dtype=torch.float32, device=device)
    cos_t = torch.cos(tr).unsqueeze(1)
    sin_t = torch.sin(tr).unsqueeze(1)

    y = c + (x.unsqueeze(0) * cos_t).to(torch.int64)
    x_proj = c + (-x.unsqueeze(0) * sin_t).to(torch.int64)

    valid = (y >= 0) & (y < m.shape[0]) & (x_proj >= 0) & (x_proj < m.shape[1])

    gathered = torch.zeros_like(y, dtype=torch.float32, device=device)
    if torch.any(valid):
        gathered[valid] = m[y[valid], x_proj[valid]].to(torch.float32)

    li_init = torch.sum(gathered, dim=1)
    li_correct = torch.sum(gathered[:, W:], dim=1) if (W > 0 and W < gathered.shape[1]) else li_init

    idx_init = torch.argmax(li_init)
    idx_correct = torch.argmax(li_correct)

    a_init = float((tr[idx_init] / torch.pi * 180.0).item())
    a_correct = float((tr[idx_correct] / torch.pi * 180.0).item())
    dist = abs(a_init - a_correct)
    return -a_init, -a_correct, dist


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.npu.is_available():
            return torch.device("npu:0")
        return torch.device("cpu")
    return torch.device(device)


def get_angle(
    image: np.ndarray,
    amax: float = 45.0,
    V: int = 2048,
    W: int = 304,
    D: float = 0.55,
    num: int = 20,
    device: str = "auto",
) -> float:
    dev = resolve_device(device)
    ratio = V / image.shape[0]
    image = cv2.resize(image, None, fx=ratio, fy=ratio)

    magnitude = get_fft_magnitude_npu(image, dev)
    a_init, a_correct, dist = _get_angle_adaptive_npu(magnitude, amax=amax, num=num, W=W, device=dev)

    if dev.type == "npu":
        torch.npu.synchronize(dev)

    return a_correct if dist <= D else a_init


def get_angle_with_timings(
    image: np.ndarray,
    amax: float = 45.0,
    V: int = 2048,
    W: int = 304,
    D: float = 0.55,
    num: int = 20,
    device: str = "auto",
) -> Tuple[float, Dict[str, float]]:
    timings: Dict[str, float] = {}
    dev = resolve_device(device)

    t0 = cv2.getTickCount()
    ratio = V / image.shape[0]
    resized = cv2.resize(image, None, fx=ratio, fy=ratio)
    t1 = cv2.getTickCount()
    timings["resize_s"] = (t1 - t0) / cv2.getTickFrequency()

    t2 = cv2.getTickCount()
    magnitude = get_fft_magnitude_npu(resized, dev)
    if dev.type == "npu":
        torch.npu.synchronize(dev)
    t3 = cv2.getTickCount()
    timings["fft_total_s"] = (t3 - t2) / cv2.getTickFrequency()

    t4 = cv2.getTickCount()
    a_init, a_correct, dist = _get_angle_adaptive_npu(magnitude, amax=amax, num=num, W=W, device=dev)
    if dev.type == "npu":
        torch.npu.synchronize(dev)
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
    device: str = "auto",
) -> float:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    return get_angle(image, amax=amax, V=V, W=W, D=D, num=num, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch NPU 方法（独立版）")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--amax", type=float, default=45.0)
    parser.add_argument("--V", type=int, default=2048)
    parser.add_argument("--W", type=int, default=304)
    parser.add_argument("--D", type=float, default=0.55)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", help="auto / npu:0 / cpu")
    args = parser.parse_args()

    angle = get_angle_from_path(
        args.image,
        amax=args.amax,
        V=args.V,
        W=args.W,
        D=args.D,
        num=args.num,
        device=args.device,
    )
    print(f"pred_angle={angle:.4f}")


if __name__ == "__main__":
    main()
