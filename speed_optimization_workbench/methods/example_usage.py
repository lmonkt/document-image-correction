#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import cv2

from .method_original import get_angle as get_angle_original
from .method_cpu_numpy_vec import get_angle as get_angle_cpu_numpy_vec
from .method_cpu_numba_jit import get_angle as get_angle_cpu_numba_jit
from .method_cupy import get_angle as get_angle_cupy


def main() -> None:
    parser = argparse.ArgumentParser(description="四种方法对同一图像进行角度预测")
    parser.add_argument("--image", type=str, default="assets/image-20260122211453-r856wkw.png")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {args.image}")

    print(f"input={args.image}")
    print(f"original      : {get_angle_original(image):.4f}")
    print(f"cpu_numpy_vec : {get_angle_cpu_numpy_vec(image):.4f}")
    print(f"cpu_numba_jit : {get_angle_cpu_numba_jit(image):.4f}")
    print(f"cupy          : {get_angle_cupy(image):.4f}")


if __name__ == "__main__":
    main()
