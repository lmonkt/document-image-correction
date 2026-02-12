#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高保真图片旋转工具

对应博客文档：辅助工具模块（数据预处理）

功能说明：
  基于 OpenCV 实现的图像旋转工具，用于生成测试数据或单独旋转图像：
  - 支持任意角度旋转
  - 自动计算扩展画布，确保旋转后内容不被裁剪
  - 可调节输出质量（1-100）
  - 白色填充背景，适配文档场景

算法特点：
  - 正值角度：逆时针旋转
  - 负值角度：顺时针旋转
  - 使用 cv2.warpAffine 进行仿射变换
  - 画布尺寸自适应，避免内容丢失

使用场景：
  - 生成倾斜测试数据（配合 generate_test_data.py）
  - 手动旋转图像验证算法效果
  - 批量图像预处理

命令行用法：
  python image_rotation.py <input_image> <angle> [--output <output_path>] [--quality <1-100>]

示例：
  python image_rotation.py input.jpg 25.5 --output rotated.jpg
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys


def rotate_image(image_path, angle, output_path=None, quality=95):
    """
    旋转图片并保护完整性
    
    Args:
        image_path (str): 输入图片路径
        angle (float): 旋转角度（度数），正值为逆时针，负值为顺时针
        output_path (str): 输出图片路径，默认保存到仓库的 `quqinxie/data/skew_images`，文件名与输入一致
        quality (int): 输出图片质量 (1-100)，默认95
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    
    # 检查输入文件是否存在
    input_file = Path(image_path)
    if not input_file.exists():
        print(f"错误：图片文件不存在: {image_path}")
        return False
    
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误：无法读取图片文件: {image_path}")
        return False
    
    print(f"✓ 已读取图片: {image_path}")
    print(f"  原始尺寸: {image.shape[1]} x {image.shape[0]} 像素")
    
    # 获取图片尺寸
    height, width = image.shape[:2]
    
    # 计算旋转矩阵
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后需要的画布大小（保证图片完整）
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    # 调整旋转矩阵的平移参数，使旋转后的图片居中
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # 进行旋转（使用高质量插值）
    # cv2.INTER_CUBIC: 三次插值，质量好但速度较慢
    # cv2.INTER_LANCZOS4: Lanczos插值，最高质量
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LANCZOS4,  # 高质量插值
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # 白色边界
    )
    
    print(f"✓ 旋转角度: {angle}°")
    print(f"  旋转后尺寸: {new_width} x {new_height} 像素")
    
    # 确定输出路径
    if output_path is None:
        # 默认放到仓库下的 quqinxie/data/skew_images，文件名与输入一致
        default_dir = Path(__file__).resolve().parent / 'quqinxie' / 'data' / 'skew_images'
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = default_dir / input_file.name

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        print(f"⚠️ 警告：输出文件已存在，将被覆盖: {output_file}")
    
    # 保存图片（根据格式选择参数）
    if output_file.suffix.lower() in ['.jpg', '.jpeg']:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif output_file.suffix.lower() == '.png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        params = []
    
    success = cv2.imwrite(str(output_file), rotated_image, params)
    
    if success:
        print(f"✓ 旋转完成！")
        print(f"  输出文件: {output_file}")
        return True
    else:
        print(f"错误：无法保存图片到 {output_file}")
        return False


def rotate_image_interactive():
    """交互模式：输入图片路径和旋转角度"""
    print("=" * 50)
    print("  高保真图片旋转工具")
    print("=" * 50)
    
    # 获取图片路径
    while True:
        image_path = input("\n请输入图片路径: ").strip()
        if Path(image_path).exists():
            break
        print("错误：文件不存在，请重新输入")
    
    # 获取旋转角度
    while True:
        try:
            angle = float(input("请输入旋转角度（度数，正值逆时针，负值顺时针）: "))
            break
        except ValueError:
            print("错误：请输入有效的数字")
    
    # 获取输出路径（可选）
    output_path = input("请输入输出路径（按回车使用默认）: ").strip()
    if not output_path:
        output_path = None
    
    # 获取质量参数（可选）
    while True:
        try:
            quality_input = input("请输入图片质量 1-100（按回车使用95）: ").strip()
            quality = int(quality_input) if quality_input else 95
            if 1 <= quality <= 100:
                break
            print("错误：质量值应在 1-100 之间")
        except ValueError:
            print("错误：请输入有效的数字")
    
    # 执行旋转
    rotate_image(image_path, angle, output_path, quality)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='高保真图片旋转工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 旋转45度
  python image_rotation.py input.jpg -a 45
  
  # 旋转45度并指定输出路径
  python image_rotation.py input.jpg -a 45 -o output.jpg
  
  # 旋转45度并指定质量为80
  python image_rotation.py input.jpg -a 45 -q 80
  
  # 交互模式
  python image_rotation.py

  # 默认输出目录（未指定时）
  # 保存到仓库内的 quqinxie/data/skew_images，文件名与输入一致
        """
    )
    
    parser.add_argument('image', nargs='?', help='输入图片路径')
    parser.add_argument('-a', '--angle', type=float, default=0, help='旋转角度（度数），默认0')
    parser.add_argument('-o', '--output', help='输出图片路径（可选）')
    parser.add_argument('-q', '--quality', type=int, default=95, help='输出质量 1-100，默认95')
    
    args = parser.parse_args()
    
    # 如果没有提供图片路径，进入交互模式
    if not args.image:
        rotate_image_interactive()
    else:
        rotate_image(args.image, args.angle, args.output, args.quality)


if __name__ == '__main__':
    main()
