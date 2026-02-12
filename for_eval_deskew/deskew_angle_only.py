#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像倾斜角度估计模块（仅返回角度）

从 image_pipeline.py 中的 DeskewImg 类改编，专门用于评估
"""

import os
import sys
import numpy as np
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)


class DeskewAngleEstimator:
    """基于傅里叶频谱的粗细两阶段校正（仅返回角度）。

    该类专门用于评估，只返回倾斜角度，不进行图像旋转。
    """

    def __init__(
        self,
        coarse_angle_step: float = 1.0,
        fine_angle_range: float = 2.0,
        fine_angle_step: float = 0.1,
        max_skew_angle: float = 45.0,
    ):
        self.coarse_angle_step = coarse_angle_step
        self.fine_angle_range = fine_angle_range
        self.fine_angle_step = fine_angle_step
        self.max_skew_angle = max_skew_angle

    def estimate_angle(self, image: np.ndarray) -> float:
        """
        估计图像的倾斜角度
        
        参数:
            image: 输入图像（numpy数组）
            
        返回:
            float: 估计的倾斜角度（度），正值表示顺时针倾斜，负值表示逆时针倾斜
        """
        if not isinstance(image, np.ndarray):
            return 0.0
            
        gray = self._to_gray(image)
        magnitude = self._calculate_fft_magnitude(gray)
        
        # 粗略估计
        coarse_angles = np.arange(-90.0, 90.0, self.coarse_angle_step)
        coarse_fft_angle = self._find_best_angle(magnitude, coarse_angles)
        coarse_image_angle = self._map_fft_angle_to_image_angle(coarse_fft_angle)
        
        # 精细估计
        fine_angles = np.arange(
            coarse_image_angle - self.fine_angle_range,
            coarse_image_angle + self.fine_angle_range,
            self.fine_angle_step,
        )
        fine_image_angle = self._find_best_angle(
            magnitude, fine_angles, is_image_angle=True
        )
        
        # 限制角度范围
        final_angle = np.clip(
            fine_image_angle, -self.max_skew_angle, self.max_skew_angle
        )
        
        return float(final_angle)

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _calculate_fft_magnitude(self, gray_image: np.ndarray) -> np.ndarray:
        h, w = gray_image.shape
        optimal_size = cv2.getOptimalDFTSize(max(h, w))
        padded = cv2.copyMakeBorder(
            gray_image,
            0,
            optimal_size - h,
            0,
            optimal_size - w,
            cv2.BORDER_CONSTANT,
            value=(255,),
        )
        binary = cv2.adaptiveThreshold(
            np.bitwise_not(padded),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            -10,
        )
        dft = cv2.dft(
            np.asarray(binary, dtype=np.float32), flags=cv2.DFT_COMPLEX_OUTPUT
        )
        dft_shifted = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
        cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        cv2.circle(magnitude, (cx, cy), radius=2, color=(0, 0, 0), thickness=-1)
        return magnitude

    def _find_best_angle(
        self, magnitude: np.ndarray, angles: np.ndarray, is_image_angle: bool = False
    ) -> float:
        center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        max_radius = min(center_y, center_x)
        best_angle = 0.0
        max_energy = -np.inf
        for angle_deg in angles:
            fft_angle_deg = angle_deg + 90 if is_image_angle else angle_deg
            angle_rad = np.deg2rad(fft_angle_deg)
            radii = np.arange(max_radius)
            x_coords = np.clip(
                np.round(center_x + radii * np.cos(angle_rad)).astype(int),
                0,
                magnitude.shape[1] - 1,
            )
            y_coords = np.clip(
                np.round(center_y - radii * np.sin(angle_rad)).astype(int),
                0,
                magnitude.shape[0] - 1,
            )
            energy = np.sum(magnitude[y_coords, x_coords])
            if energy > max_energy:
                max_energy = energy
                best_angle = angle_deg
        return best_angle

    def _map_fft_angle_to_image_angle(self, fft_angle: float) -> float:
        image_angle = fft_angle - 90
        if image_angle < -90:
            image_angle += 180
        elif image_angle > 90:
            image_angle -= 180
        if abs(image_angle) > self.max_skew_angle:
            alt_angle1 = image_angle - 90
            alt_angle2 = image_angle + 90
            if abs(alt_angle1) < abs(image_angle):
                image_angle = alt_angle1
            elif abs(alt_angle2) < abs(image_angle):
                image_angle = alt_angle2
        return image_angle


def get_skew_angle(image: np.ndarray, max_skew_angle: float = 45.0) -> float:
    """
    便捷函数：估计图像的倾斜角度
    
    参数:
        image: 输入图像（numpy数组）
        max_skew_angle: 最大倾斜角度范围（度），默认45度
        
    返回:
        float: 估计的倾斜角度（度）
    """
    estimator = DeskewAngleEstimator(max_skew_angle=max_skew_angle)
    return estimator.estimate_angle(image)


if __name__ == "__main__":
    # 简单测试
    import sys
    if len(sys.argv) > 1:
        test_image = cv2.imread(sys.argv[1])
        if test_image is not None:
            angle = get_skew_angle(test_image)
            print(f"Estimated skew angle: {angle:.2f} degrees")
        else:
            print(f"Failed to load image: {sys.argv[1]}")
    else:
        print("Usage: python deskew_angle_only.py <image_path>")
