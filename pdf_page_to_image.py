#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 页面提取与转换工具

对应博客文档：辅助工具模块（数据预处理）

功能说明：
  将 PDF 文档的指定页面提取为图像或单页PDF，支持批量处理：
  1. 提取为高清图像（支持多种格式：jpg, png等）
  2. 提取为单页 PDF 文件
  3. 合并指定页面为新的 PDF 文档

核心功能：
  - get_pdf_total_pages(): 获取PDF总页数
  - is_valid_page_range(): 验证页码范围有效性
  - extract_pages_to_images_and_pdfs(): 批量提取页面为图像和PDF
  - extract_all_pages(): 提取PDF的所有页面

使用场景：
  - 从大型PDF文档中提取需要处理的页面
  - 将PDF转为图像后进行去倾斜/旋转校正
  - 文档图像预处理的数据准备阶段

配置选项：
  CONFIG['save_image']: 是否保存为图像
  CONFIG['save_pdf']: 是否保存为单页PDF
  CONFIG['save_merged_pdf']: 是否保存合并PDF
  CONFIG['temp_dir']: 临时文件目录

命令行用法：
  python pdf_page_to_image.py
  # 交互式输入PDF路径和页码范围

依赖库：
  - pdf2image: PDF转图像
  - PyPDF2: PDF读取和写入
"""

import os
import sys
from pathlib import Path
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
import logging

# 禁用冗余日志
logging.getLogger('pdf2image').setLevel(logging.ERROR)

# 全局配置
CONFIG = {
    'save_image': True,
    'save_pdf': True,
    'save_merged_pdf': True,
    'temp_dir': 'temp'
}

def ensure_temp_dir():
    temp_dir = Path(CONFIG['temp_dir'])
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def get_pdf_total_pages(pdf_path):
    """获取PDF总页数"""
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"无法读取 PDF 文件: {e}")
        return 0

def is_valid_page_range(pdf_path, start_page, end_page):
    """检查页码范围是否有效"""
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        if start_page < 1 or end_page > total_pages or start_page > end_page:
            return False, total_pages
        return True, total_pages
    except Exception as e:
        print(f"无法读取 PDF 文件: {e}")
        return False, 0

def extract_pages_to_images_and_pdfs(pdf_path, start_page, end_page, is_all=False):
    """提取多页到图片、单页PDF和合并PDF"""
    temp_dir = ensure_temp_dir()
    success_count = 0
    total_pages = end_page - start_page + 1
    
    if is_all:
        print(f"开始提取所有页面，共 {total_pages} 页...")
    else:
        print(f"开始提取第 {start_page} 到 {end_page} 页，共 {total_pages} 页...")
    
    # === 1. 保存为图片 ===
    if CONFIG['save_image']:
        try:
            images = convert_from_path(
                pdf_path,
                first_page=start_page,
                last_page=end_page,
                dpi=96
                # 如果你之前加了 poppler_path，这里也要加上，例如：
                # poppler_path=r"C:\poppler\bin"
            )
            
            for i, image in enumerate(images):
                page_num = start_page + i
                img_path = temp_dir / f"page_{page_num}.png"
                image.save(img_path, "PNG")
                print(f"✅ 图片已保存: {img_path.name}")
                success_count += 1
                
        except Exception as e:
            print(f"❌ 图片转换失败: {e}")
    
    # === 2. 保存为单页 PDF ===
    if CONFIG['save_pdf']:
        try:
            reader = PdfReader(pdf_path)
            
            for page_num in range(start_page, end_page + 1):
                try:
                    writer = PdfWriter()
                    writer.add_page(reader.pages[page_num - 1])  # PyPDF2 是 0-based

                    pdf_path_out = temp_dir / f"page_{page_num}.pdf"
                    with open(pdf_path_out, "wb") as f:
                        writer.write(f)
                    print(f"✅ 单页PDF已保存: {pdf_path_out.name}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ 第 {page_num} 页 PDF 提取失败: {e}")
                    
        except Exception as e:
            print(f"❌ PDF 读取失败: {e}")
    
    # === 3. 保存为合并的PDF文件（在all模式下禁用）===
    if CONFIG['save_merged_pdf'] and not is_all:
        try:
            reader = PdfReader(pdf_path)
            writer = PdfWriter()
            
            # 添加指定范围内的所有页面到同一个PDF
            for page_num in range(start_page, end_page + 1):
                writer.add_page(reader.pages[page_num - 1])
            
            # 生成合并PDF的文件名
            if start_page == end_page:
                pdf_filename = f"page_{start_page}.pdf"  # 单页时使用简单名称
            else:
                pdf_filename = f"pages_{start_page}_to_{end_page}.pdf"
            
            merged_pdf_path = temp_dir / pdf_filename
            
            with open(merged_pdf_path, "wb") as f:
                writer.write(f)
            
            print(f"📚 合并PDF已保存: {merged_pdf_path.name}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 合并PDF保存失败: {e}")
    
    print(f"🎉 提取完成！成功处理 {success_count} 个文件")

def show_config():
    """显示当前配置"""
    image_status = "✅" if CONFIG['save_image'] else "❌"
    pdf_status = "✅" if CONFIG['save_pdf'] else "❌"
    merged_pdf_status = "✅" if CONFIG['save_merged_pdf'] else "❌"
    
    print("\n当前配置:")
    print(f"  保存图片: {image_status}")
    print(f"  保存单页PDF: {pdf_status}")
    print(f"  保存合并PDF: {merged_pdf_status} (在'all'模式下自动禁用)")
    print(f"  输出目录: {CONFIG['temp_dir']}")

def show_help():
    """显示帮助信息"""
    print("\n可用命令:")
    print("  <PDF文件路径> <页码>              - 提取单页")
    print("  <PDF文件路径> <起始页码> <终止页码> - 提取多页")
    print("  <PDF文件路径> all                 - 提取所有页面为单页文件")
    print("  config                            - 显示当前配置")
    print("  set image <on/off>               - 开启/关闭保存图片")
    print("  set pdf <on/off>                 - 开启/关闭保存单页PDF")
    print("  set merged_pdf <on/off>          - 开启/关闭保存合并PDF")
    print("  set dir <目录名>                  - 设置输出目录")
    print("  help                              - 显示此帮助")
    print("  quit / exit                      - 退出程序")

def parse_page_range(page_str, pdf_path=None):
    """解析页码范围，支持'all'关键字"""
    # 处理'all'关键字
    if page_str.lower() == 'all':
        if pdf_path:
            total_pages = get_pdf_total_pages(pdf_path)
            if total_pages > 0:
                return 1, total_pages
            else:
                return None, None
        else:
            return None, None
    
    # 处理原有格式
    if '-' in page_str:
        try:
            start, end = map(int, page_str.split('-'))
            return start, end
        except ValueError:
            return None, None
    else:
        try:
            page = int(page_str)
            return page, page
        except ValueError:
            return None, None

def main():
    print("PDF 页面转图片 + PDF 提取工具（增强版）")
    print("支持单页和多页提取，可配置输出选项")
    
    show_config()
    show_help()

    while True:
        try:
            user_input = input("\n>>> ").strip()
            if not user_input:
                continue

            parts = user_input.split()
            command = parts[0].lower()

            # 处理特殊命令
            if command in ('quit', 'exit', 'q'):
                print("再见！")
                break
                
            elif command == 'help':
                show_help()
                continue
                
            elif command == 'config':
                show_config()
                continue
                
            elif command == 'set' and len(parts) >= 3:
                setting_type = parts[1].lower()
                value = parts[2].lower()
                
                if setting_type == 'image':
                    CONFIG['save_image'] = (value == 'on')
                    print(f"{'✅' if CONFIG['save_image'] else '❌'} 保存图片: {'开启' if CONFIG['save_image'] else '关闭'}")
                    
                elif setting_type == 'pdf':
                    CONFIG['save_pdf'] = (value == 'on')
                    print(f"{'✅' if CONFIG['save_pdf'] else '❌'} 保存单页PDF: {'开启' if CONFIG['save_pdf'] else '关闭'}")
                
                elif setting_type == 'merged_pdf':
                    CONFIG['save_merged_pdf'] = (value == 'on')
                    print(f"{'✅' if CONFIG['save_merged_pdf'] else '❌'} 保存合并PDF: {'开启' if CONFIG['save_merged_pdf'] else '关闭'}")
                    
                elif setting_type == 'dir' and len(parts) >= 3:
                    new_dir = parts[2]
                    CONFIG['temp_dir'] = new_dir
                    print(f"📁 输出目录设置为: {new_dir}")
                    ensure_temp_dir()  # 立即创建新目录
                    
                else:
                    print("❌ 无效的设置命令")
                continue

            # 处理文件提取命令
            is_all_command = False
            if len(parts) == 2:
                # 格式: <PDF路径> <页码或页码范围或all>
                pdf_path, page_range_str = parts
                
                # 检查是否为'all'
                if page_range_str.lower() == 'all':
                    total_pages = get_pdf_total_pages(pdf_path)
                    if total_pages == 0:
                        continue
                    start_page, end_page = 1, total_pages
                    is_all_command = True
                    print(f"📄 检测到PDF共有 {total_pages} 页，将提取所有页面为单页文件")
                else:
                    start_page, end_page = parse_page_range(page_range_str)
                
            elif len(parts) == 3:
                # 格式: <PDF路径> <起始页码> <终止页码>
                pdf_path, start_str, end_str = parts
                try:
                    start_page, end_page = int(start_str), int(end_str)
                except ValueError:
                    print("❌ 页码必须是整数")
                    continue
            else:
                print("❌ 请按格式输入: <PDF路径> <页码/all> 或 <PDF路径> <起始页> <终止页>")
                continue

            # 验证文件
            if not os.path.isfile(pdf_path):
                print(f"❌ 文件不存在: {pdf_path}")
                continue

            # 验证页码
            if start_page is None or end_page is None:
                print("❌ 页码格式错误，请使用: 5 或 1-10 或 all")
                continue

            # 验证页码范围（除了'all'的情况）
            if not is_all_command:
                is_valid, total_pages = is_valid_page_range(pdf_path, start_page, end_page)
                if not is_valid:
                    print(f"❌ 页码范围 {start_page}-{end_page} 无效，PDF 共 {total_pages} 页")
                    continue

            # 检查是否有输出选项被启用
            if not CONFIG['save_image'] and not CONFIG['save_pdf']:
                # 在all模式下，合并PDF被禁用，所以只需要检查图片和单页PDF
                print("❌ 请至少开启一个输出选项（图片或单页PDF）")
                print("使用: set image on 或 set pdf on")
                continue

            # 执行提取
            extract_pages_to_images_and_pdfs(pdf_path, start_page, end_page, is_all_command)

        except KeyboardInterrupt:
            print("\n\n程序被中断，再见！")
            break
        except Exception as e:
            print(f"⚠️ 未知错误: {e}")

if __name__ == "__main__":
    main()