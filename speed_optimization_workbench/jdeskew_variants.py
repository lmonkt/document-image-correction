#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


@dataclass
class AngleConfig:
    amax: float = 45.0
    V: int = 2048
    W: int = 304
    D: float = 0.55
    num: int = 20


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


def _angles_radians(amax: float, num: int) -> np.ndarray:
    return np.linspace(-1 * amax, amax, int(amax * num * 2), dtype=np.float64) / 180.0 * np.pi


def _radial_projection_original(
    m: np.ndarray,
    amax: float,
    num: int,
    W: int,
) -> Tuple[float, float, float]:
    r = c = m.shape[0] // 2
    tr = _angles_radians(amax, num)

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

    a_init = tr[np.argmax(li_init)] / np.pi * 180.0
    a_correct = tr[np.argmax(li_correct)] / np.pi * 180.0
    dist = abs(a_init - a_correct)
    return -1.0 * a_init, -1.0 * a_correct, dist


def _radial_projection_numpy_vectorized(
    m: np.ndarray,
    amax: float,
    num: int,
    W: int,
) -> Tuple[float, float, float]:
    r = c = m.shape[0] // 2
    tr = _angles_radians(amax, num)

    x = np.arange(r, dtype=np.float64)
    cos_t = np.cos(tr)[:, None]
    sin_t = np.sin(tr)[:, None]

    y = c + np.int32(x[None, :] * cos_t)
    x_proj = c + np.int32(-x[None, :] * sin_t)

    valid = (y >= 0) & (y < m.shape[0]) & (x_proj >= 0) & (x_proj < m.shape[1])

    gathered = np.zeros_like(y, dtype=np.float64)
    gathered[valid] = m[y[valid], x_proj[valid]]

    li_init = gathered.sum(axis=1)
    if W > 0 and W < gathered.shape[1]:
        li_correct = gathered[:, W:].sum(axis=1)
    else:
        li_correct = li_init

    a_init = tr[np.argmax(li_init)] / np.pi * 180.0
    a_correct = tr[np.argmax(li_correct)] / np.pi * 180.0
    dist = abs(a_init - a_correct)
    return -1.0 * a_init, -1.0 * a_correct, dist


if njit is not None:
    @njit(cache=True, fastmath=True)
    def _radial_projection_numba_core(
        m: np.ndarray,
        tr: np.ndarray,
        W: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
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


    def _radial_projection_numba(
        m: np.ndarray,
        amax: float,
        num: int,
        W: int,
    ) -> Tuple[float, float, float]:
        tr = _angles_radians(amax, num)
        li_init, li_correct = _radial_projection_numba_core(m.astype(np.float64), tr, int(W))
        a_init = tr[np.argmax(li_init)] / np.pi * 180.0
        a_correct = tr[np.argmax(li_correct)] / np.pi * 180.0
        dist = abs(a_init - a_correct)
        return -1.0 * a_init, -1.0 * a_correct, dist
else:
    def _radial_projection_numba(
        m: np.ndarray,
        amax: float,
        num: int,
        W: int,
    ) -> Tuple[float, float, float]:
        return _radial_projection_numpy_vectorized(m, amax, num, W)


def get_angle_with_variant(
    image: np.ndarray,
    cfg: AngleConfig,
    variant: str,
) -> Tuple[float, Dict[str, float]]:
    timings: Dict[str, float] = {}

    t0 = cv2.getTickCount()
    ratio = cfg.V / image.shape[0]
    resized = cv2.resize(image, None, fx=ratio, fy=ratio)
    t1 = cv2.getTickCount()
    timings["resize_s"] = (t1 - t0) / cv2.getTickFrequency()

    t2 = cv2.getTickCount()
    magnitude = get_fft_magnitude_numpy(resized)
    t3 = cv2.getTickCount()
    timings["fft_total_s"] = (t3 - t2) / cv2.getTickFrequency()

    t4 = cv2.getTickCount()
    if variant == "original":
        a_init, a_correct, dist = _radial_projection_original(magnitude, cfg.amax, cfg.num, cfg.W)
    elif variant == "cpu_numpy_vec":
        a_init, a_correct, dist = _radial_projection_numpy_vectorized(magnitude, cfg.amax, cfg.num, cfg.W)
    elif variant == "cpu_numba_jit":
        a_init, a_correct, dist = _radial_projection_numba(magnitude, cfg.amax, cfg.num, cfg.W)
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    t5 = cv2.getTickCount()
    timings["radial_projection_s"] = (t5 - t4) / cv2.getTickFrequency()

    t6 = cv2.getTickCount()
    angle = a_correct if dist <= cfg.D else a_init
    t7 = cv2.getTickCount()
    timings["decision_s"] = (t7 - t6) / cv2.getTickFrequency()

    timings["algo_total_s"] = timings["resize_s"] + timings["fft_total_s"] + timings["radial_projection_s"] + timings["decision_s"]
    return angle, timings


def is_cupy_available() -> bool:
    try:
        import cupy as _  # noqa: F401
        return True
    except Exception:
        return False


def get_angle_with_cupy(
    image: np.ndarray,
    cfg: AngleConfig,
) -> Tuple[float, Dict[str, float]]:
    import cupy as cp

    timings: Dict[str, float] = {}

    t0 = cv2.getTickCount()
    ratio = cfg.V / image.shape[0]
    resized = cv2.resize(image, None, fx=ratio, fy=ratio)
    t1 = cv2.getTickCount()
    timings["resize_s"] = (t1 - t0) / cv2.getTickFrequency()

    t2 = cv2.getTickCount()
    gray = ensure_gray(resized)
    opt_gray = ensure_optimal_square(gray)
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    g = cp.asarray(opt_gray, dtype=cp.float32)
    dft = cp.fft.fft2(g)
    shifted = cp.fft.fftshift(dft)
    mag = cp.abs(shifted)
    cp.cuda.Stream.null.synchronize()
    t3 = cv2.getTickCount()
    timings["fft_total_s"] = (t3 - t2) / cv2.getTickFrequency()

    t4 = cv2.getTickCount()
    r = mag.shape[0] // 2
    c = r
    tr = cp.linspace(-cfg.amax, cfg.amax, int(cfg.amax * cfg.num * 2), dtype=cp.float32) / 180.0 * cp.pi
    x = cp.arange(r, dtype=cp.float32)
    cos_t = cp.cos(tr)[:, None]
    sin_t = cp.sin(tr)[:, None]

    y = c + (x[None, :] * cos_t).astype(cp.int32)
    x_proj = c + (-x[None, :] * sin_t).astype(cp.int32)

    valid = (y >= 0) & (y < mag.shape[0]) & (x_proj >= 0) & (x_proj < mag.shape[1])

    gathered = cp.zeros_like(y, dtype=cp.float32)
    gathered[valid] = mag[y[valid], x_proj[valid]]

    li_init = cp.sum(gathered, axis=1)
    if cfg.W > 0 and cfg.W < gathered.shape[1]:
        li_correct = cp.sum(gathered[:, cfg.W:], axis=1)
    else:
        li_correct = li_init

    a_init = float((tr[cp.argmax(li_init)] / cp.pi * 180.0).get())
    a_correct = float((tr[cp.argmax(li_correct)] / cp.pi * 180.0).get())
    dist = abs(a_init - a_correct)
    cp.cuda.Stream.null.synchronize()
    t5 = cv2.getTickCount()
    timings["radial_projection_s"] = (t5 - t4) / cv2.getTickFrequency()

    t6 = cv2.getTickCount()
    angle = -a_correct if dist <= cfg.D else -a_init
    t7 = cv2.getTickCount()
    timings["decision_s"] = (t7 - t6) / cv2.getTickFrequency()

    timings["algo_total_s"] = timings["resize_s"] + timings["fft_total_s"] + timings["radial_projection_s"] + timings["decision_s"]
    return angle, timings
