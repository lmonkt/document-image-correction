#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据生成脚本

对应博客文档：一、数据展示（第三种数据场景）

功能说明：
  自动生成用于测试的倾斜+旋转混合数据，模拟真实场景中的复杂情况：
  - 输入：quqinxie/data/normal_image 下的标准文档图像（0°）
  - 输出：quqinxie/data/skew_images 带有随机旋转角度的测试图像
  
生成策略：
  - 基准角度：45°, 90°, 180°, 270°
  - 每个基准角度添加 ±2°~±10° 的随机偏移
  - 确保不是 45° 的精确倍数（避免边界情况）
  - 对每张输入图像生成 4 个不同角度的变体

使用场景：
  - 生成综合测试集，验证 去倾斜+方向矫正 组合方案的效果
  - 模拟真实扫描场景中的复杂旋转情况
  - 对应博客中的 四、完整去倾斜+旋转校正方案 验证数据

依赖：
  调用 image_rotation.py 执行实际的旋转操作

运行方式：
  python generate_test_data.py
"""

import sys
from pathlib import Path
import subprocess
import random


def generate_test_angles():
    """
    生成测试角度（非45度倍数）
    - 45°往上：45 + 偏移
    - 90°往上：90 + 偏移
    - 180°往上：180 + 偏移
    - 270°往上：270 + 偏移
    """
    # 偏移范围：避免刚好是45度倍数
    offset_min = 2
    offset_max = 10
    
    angles = []
    base_angles = [45, 90, 180, 270]
    
    for base in base_angles:
        # 随机选择正负偏移
        offset = random.uniform(offset_min, offset_max)
        if random.choice([True, False]):
            offset = -offset
        angle = base + offset
        angles.append((base, angle))
    
    return angles


def rotate_images():
    """
    使用 image_rotation.py 旋转图片
    """
    # 路径设置
    script_dir = Path(__file__).resolve().parent
    rotation_script = script_dir / "image_rotation.py"
    input_dir = script_dir / "quqinxie" / "data" / "normal_image"
    output_dir = script_dir / "quqinxie" / "data" / "skew_images"
    
    # 检查脚本和输入目录
    if not rotation_script.exists():
        print(f"错误：找不到 image_rotation.py")
        return False
    
    if not input_dir.exists():
        print(f"错误：输入目录不存在: {input_dir}")
        return False
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有输入图片
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    if not image_files:
        print(f"错误：输入目录中没有图片文件: {input_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"测试数据生成")
    print(f"{'='*60}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(image_files)} 张图片")
    print(f"{'='*60}\n")
    
    # 生成测试角度
    test_angles = generate_test_angles()
    
    # 对每张图片应用所有测试角度
    total_count = 0
    success_count = 0
    
    for img_file in image_files:
        for base_angle, actual_angle in test_angles:
            # 构建输出文件名：原文件名_base角度_actual角度.扩展名
            # 例如：page_1_45_47.3.png
            output_filename = f"{img_file.stem}_{int(base_angle)}_{actual_angle:.1f}{img_file.suffix}"
            output_path = output_dir / output_filename
            
            # 调用 image_rotation.py
            cmd = [
                sys.executable,
                str(rotation_script),
                str(img_file),
                "-a", str(actual_angle),
                "-o", str(output_path),
                "-q", "95"
            ]
            
            print(f"处理: {img_file.name} -> {base_angle}° (实际 {actual_angle:.1f}°)")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                success_count += 1
                print(f"  ✓ 成功: {output_filename}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ 失败: {e}")
                if e.stderr:
                    print(f"    错误信息: {e.stderr}")
            
            total_count += 1
    
    print(f"\n{'='*60}")
    print(f"生成完成")
    print(f"{'='*60}")
    print(f"总计: {total_count} 个任务")
    print(f"成功: {success_count}")
    print(f"失败: {total_count - success_count}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    return success_count == total_count


def main():
    """主函数"""
    random.seed(42)  # 固定随机种子，确保可重复性
    
    success = rotate_images()
    
    if success:
        print("✓ 测试数据生成成功！")
        print("\n下一步：")
        print("  使用 deskew_and_rotation_correction.py 处理这些图片")
        print("  示例：python deskew_and_rotation_correction.py quqinxie/data/skew_images/page_1_45_47.3.png")
    else:
        print("✗ 测试数据生成失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
