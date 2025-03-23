#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通义千问OCR客户端测试主程序

该程序用于测试通义千问OCR客户端的各种功能，包括：
1. 单张网络图片处理
2. 单张本地图片处理
3. 统一接口处理
4. 多图像处理
5. 视频帧序列处理
6. 手机端大模型评测对话识别
7. 视频文件处理

运行方法：
1. 直接运行: python -m img_recongition_module.ocr_main
2. 指定测试类型: python -m img_recongition_module.ocr_main --test_type url
3. 指定本地图片: python -m img_recongition_module.ocr_main --local_image /path/to/image.jpg
4. 指定本地视频: python -m img_recongition_module.ocr_main --video_file /path/to/video.mp4 --test_type video_file
5. 指定API密钥: python -m img_recongition_module.ocr_main --api_key your_api_key
6. 测试特定场景: python -m img_recongition_module.ocr_main --test_type chat_single
7. 快速处理视频: python -m img_recongition_module.ocr_main --quick_process /path/to/video.mp4
8. 自定义视频帧设置: python -m img_recongition_module.ocr_main --quick_process /path/to/video.mp4 --frame_interval 3 --max_frames 15
9. 保存结果为JSON: python -m img_recongition_module.ocr_main --quick_process /path/to/video.mp4 --save_json output.json

此外，我们还提供了一个更简单的独立命令行工具用于直接处理视频:
python -m img_recongition_module.ocr_video /path/to/video.mp4 [--output output.json] [--frame_interval 2] [--max_frames 20] [--api_key your_api_key]

注意：请不要使用pytest运行此文件，因为它不是一个测试文件。
"""

import os
import sys
import json
import logging
import argparse
import pytest  # 导入pytest以使用标记
from img_recongition_module.ocr.qwen_ocr import QwenOCR

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 使用pytest.mark.skip标记，使pytest忽略这些函数
@pytest.mark.skip(reason="这不是一个测试函数，而是一个演示函数")
def run_url_image_test(ocr):
    """测试处理单张网络图片"""
    url_image = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
    print("=== 测试网络图片 ===")
    print(f"图片URL: {url_image}")
    result = ocr.process_url_image(url_image)
    print(f"识别结果: {result}")
    print()
    return url_image


@pytest.mark.skip(reason="这不是一个测试函数，而是一个演示函数")
def run_local_image_test(ocr, image_path=None):
    """测试处理单张本地图片"""
    if image_path is None:
        image_path = r"D:\Projects\IDE_Projects\Pycharm\model-access-system\img_recongition_module\data\input\test_01.jpg"
    
    print("=== 测试本地图片 ===")
    print(f"图片路径: {image_path}")
    result = ocr.process_local_image(image_path)  # 自动检测格式
    print(f"识别结果: {result}")
    print()
    
    return image_path  # 返回路径供后续测试使用


@pytest.mark.skip(reason="这不是一个测试函数，而是一个演示函数")
def run_unified_interface_test(ocr, url_image, local_image):
    """测试统一接口"""
    print("=== 测试统一接口 ===")
    
    print("处理网络图片...")
    result = ocr.process_image(url_image)
    print(f"识别结果: {result}")
    print()
    
    print("处理本地图片...")
    result = ocr.process_image(local_image)
    print(f"识别结果: {result}")
    print()


@pytest.mark.skip(reason="这不是一个测试函数，而是一个演示函数")
def run_multiple_images_test(ocr, local_image):
    """测试多图像处理"""
    print("=== 测试多图像处理 ===")
    
    # 多个网络图片
    url_images = [
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg",
        "https://dashscope.oss-cn-beijing.aliyuncs.com/images/tiger.png"
    ]
    print("处理多个网络图片...")
    result = ocr.process_multiple_images(url_images)
    print(f"识别结果: {result}")
    print()
    
    # 混合网络图片和本地图片
    mixed_images = [
        "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg",
        local_image
    ]
    print("处理混合图片...")
    result = ocr.process_multiple_images(mixed_images)
    print(f"识别结果: {result}")
    print()



@pytest.mark.skip(reason="这不是一个测试函数，而是一个演示函数")
def run_chat_single_test(ocr, image_path=None):
    """测试单张手机对话截图识别"""
    if image_path is None:
        # 使用默认图片，这个应该替换为实际的手机截图路径
        image_path = r"D:\Projects\IDE_Projects\Pycharm\model-access-system\img_recongition_module\data\input\test_01.jpg"
    
    print("=== 测试单张手机对话截图识别 ===")
    print(f"图片路径: {image_path}")
    
    # 使用单张对话提示词
    result = ocr.process_image(image_path, prompt_type='single')
    
    # 解析结果
    try:
        chat_data = ocr.parse_ocr_result(result)
        print("\n提取的对话内容:")
        print("用户消息:")
        for i, msg in enumerate(chat_data.get("user_messages", [])):
            print(f"  {i+1}. {msg}")
        
        print("\nAI助手消息:")
        for i, msg in enumerate(chat_data.get("assistant_messages", [])):
            print(f"  {i+1}. {msg}")
    except Exception as e:
        print(f"解析结果失败: {str(e)}")
        print(f"原始结果: {result}")
    
    print()
    return image_path


@pytest.mark.skip(reason="这不是一个测试函数，而是一个演示函数")
def run_chat_multiple_test(ocr):
    """测试多张手机对话截图识别"""
    print("=== 测试多张手机对话截图识别 ===")
    
    # 这些应该替换为实际的手机对话截图路径
    image_paths = [
        r"D:\Projects\IDE_Projects\Pycharm\model-access-system\img_recongition_module\data\input\test_02.1.jpg",
        r"D:\Projects\IDE_Projects\Pycharm\model-access-system\img_recongition_module\data\input\test_02.2.jpg"
    ]
    
    print(f"图片路径: {image_paths}")
    
    # 使用多张对话提示词
    result = ocr.process_multiple_images(image_paths, prompt_type='multiple')
    
    # 解析结果
    try:
        chat_data = ocr.parse_ocr_result(result)
        print("\n提取的完整对话内容:")
        print("用户消息:")
        for i, msg in enumerate(chat_data.get("user_messages", [])):
            print(f"  {i+1}. {msg}")
        
        print("\nAI助手消息:")
        for i, msg in enumerate(chat_data.get("assistant_messages", [])):
            print(f"  {i+1}. {msg}")
    except Exception as e:
        print(f"解析结果失败: {str(e)}")
        print(f"原始结果: {result}")
    
    print()


@pytest.mark.skip(reason="这不是一个测试函数，而是一个演示函数")
def run_chat_video_test(ocr, video_path=None):
    """测试手机对话视频识别"""
    if video_path is None:
        # 使用默认视频
        video_path = r".\data\input\test_03.mp4"
    
    print("=== 测试手机对话视频识别 ===")
    print(f"视频路径: {video_path}")
    
    # 处理视频文件，使用默认参数
    result = ocr.process_video_file(video_path, frame_interval=2, max_frames=80, prompt_type='video')
    
    # 解析结果
    try:
        chat_data = ocr.parse_ocr_result(result)
        print("\n提取的完整视频对话内容:")
        print("用户消息:")
        for i, msg in enumerate(chat_data.get("user_messages", [])):
            print(f"  {i+1}. {msg}")
        
        print("\nAI助手消息:")
        for i, msg in enumerate(chat_data.get("assistant_messages", [])):
            print(f"  {i+1}. {msg}")
            
        # 打印用户操作行为
        if "user_actions" in chat_data:
            print("\n用户操作行为:")
            print(chat_data["user_actions"])
    except Exception as e:
        print(f"解析结果失败: {str(e)}")
        print(f"原始结果: {result}")
    
    print()
    return result


@pytest.mark.skip(reason="这不是一个测试函数，而是一个演示函数")
def run_video_file_test(ocr, video_path=None):
    """测试视频文件处理"""
    if video_path is None:
        # 使用默认视频
        video_path = r"D:\Projects\IDE_Projects\Pycharm\model-access-system\img_recongition_module\data\input\test_03.mp4"
    
    print("=== 测试视频文件处理 ===")
    print(f"视频路径: {video_path}")
    
    # 处理视频文件，使用默认参数
    result = ocr.process_video_file(video_path, frame_interval=2, max_frames=20)
    
    # 解析结果
    try:
        chat_data = ocr.parse_ocr_result(result)
        print("\n提取的完整视频对话内容:")
        print("用户消息:")
        for i, msg in enumerate(chat_data.get("user_messages", [])):
            print(f"  {i+1}. {msg}")
        
        print("\nAI助手消息:")
        for i, msg in enumerate(chat_data.get("assistant_messages", [])):
            print(f"  {i+1}. {msg}")
            
        # 打印用户操作行为
        if "user_actions" in chat_data:
            print("\n用户操作行为:")
            print(chat_data["user_actions"])
    except Exception as e:
        print(f"解析结果失败: {str(e)}")
        print(f"原始结果: {result}")
    
    print()
    return result


@pytest.mark.skip(reason="这不是一个测试函数，而是主函数")
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="通义千问OCR客户端测试程序")
    parser.add_argument("--api_key", help="API密钥，如果不提供则从环境变量或配置文件中获取")
    parser.add_argument("--local_image", help="本地图片路径，用于测试本地图片处理")
    parser.add_argument("--video_file", help="本地视频文件路径，用于测试视频文件处理")
    parser.add_argument("--frame_interval", type=int, default=2, help="视频帧间隔，每隔多少帧提取一次，默认为2")
    parser.add_argument("--max_frames", type=int, default=20, help="最大提取帧数，默认为20")
    parser.add_argument("--test_type", 
                        choices=["all", "url", "local", "unified", "multiple", "video", 
                                 "chat_single", "chat_multiple", "chat_video", "video_file"], 
                        default="all", help="测试类型，默认为all")
    parser.add_argument("--quick_process", help="快速处理视频文件路径（简洁接口）")
    parser.add_argument("--save_json", help="将处理结果保存为JSON文件的路径")
    
    args = parser.parse_args()
    
    try:
        # 创建OCR客户端
        ocr = QwenOCR(api_key=args.api_key)
        
        # 快速处理模式
        if args.quick_process:
            print(f"=== 快速处理视频文件 ===")
            print(f"视频路径: {args.quick_process}")
            print(f"帧间隔: {args.frame_interval}, 最大帧数: {args.max_frames}")
            
            # 处理视频文件
            result = ocr.process_video_file(
                args.quick_process, 
                frame_interval=args.frame_interval, 
                max_frames=args.max_frames
            )
            
            # 解析结果
            chat_data = ocr.parse_ocr_result(result)
            
            print("\n提取的完整视频对话内容:")
            print("用户消息:")
            for i, msg in enumerate(chat_data.get("user_messages", [])):
                print(f"  {i+1}. {msg}")
            
            print("\nAI助手消息:")
            for i, msg in enumerate(chat_data.get("assistant_messages", [])):
                print(f"  {i+1}. {msg}")
                
            # 打印用户操作行为
            if "user_actions" in chat_data:
                print("\n用户操作行为:")
                print(chat_data["user_actions"])
                
            # 保存结果到JSON文件
            if args.save_json:
                with open(args.save_json, 'w', encoding='utf-8') as f:
                    json.dump(chat_data, f, ensure_ascii=False, indent=2)
                print(f"\n结果已保存到: {args.save_json}")
                
            return
        
        # 常规测试模式
        # # 测试URL图片
        # url_image = None
        # if args.test_type in ["all", "url"]:
        #     url_image = run_url_image_test(ocr)
        # else:
        #     url_image = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
        #
        # # 测试本地图片
        # local_image = None
        # if args.test_type in ["all", "local"]:
        #     local_image = run_local_image_test(ocr, args.local_image)
        # elif args.local_image:
        #     local_image = args.local_image
        # else:
        #     local_image = r".\data\input\test_01.jpg"
        #
        # # 测试统一接口
        # if args.test_type in ["all", "unified"]:
        #     run_unified_interface_test(ocr, url_image, local_image)
        #
        # # 测试多图像处理
        # if args.test_type in ["all", "multiple"]:
        #     run_multiple_images_test(ocr, local_image)
        #
        # # 测试单张手机对话截图识别
        # if args.test_type in ["all", "chat_single"]:
        #     run_chat_single_test(ocr, args.local_image)
        #
        # # 测试多张手机对话截图识别
        # if args.test_type in ["all", "chat_multiple"]:
        #     run_chat_multiple_test(ocr)
        
        # 测试手机对话视频识别————用于实际视频识别，下面的 run_video_file_test 功能不完善
        if args.test_type in ["all", "chat_video"]:
            run_chat_video_test(ocr, args.video_file)
            
        # # 测试视频文件处理
        # if args.test_type in ["all", "video_file"]:
        #     run_video_file_test(ocr, args.video_file)
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        print(f"测试失败: {str(e)}")


# 添加便捷函数用于直接从命令行调用视频处理
def process_video(video_path=None, frame_interval=2, max_frames=80, api_key=None, save_json=None):
    """
    便捷函数，用于直接处理视频文件
    
    Args:
        video_path: 视频文件路径，如果为None则使用默认视频
        frame_interval: 帧间隔，每隔多少帧提取一次，默认为2
        max_frames: 最大提取帧数，默认为20
        api_key: API密钥，如果为None则从settings中获取
        save_json: 将结果保存为JSON文件的路径，如果为None则不保存
        
    Returns:
        Dict: 解析后的字典，包含user_messages、assistant_messages和user_actions
    """
    if video_path is None:
        video_path = r".\data\input\test_03.mp4"
        
    try:
        # 创建OCR客户端
        ocr = QwenOCR(api_key=api_key)
        
        # 处理视频文件
        result = ocr.process_video_file(
            video_path, 
            frame_interval=frame_interval, 
            max_frames=max_frames
        )
        
        # 解析结果
        chat_data = ocr.parse_ocr_result(result)
        
        # 保存结果到JSON文件
        if save_json:
            save_dir = os.path.dirname(save_json)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            with open(save_json, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
                
        return chat_data
    except Exception as e:
        logger.error(f"视频处理失败: {str(e)}")
        raise


# 为简化直接视频处理添加一个专门的命令行接口
def ocr_video_cli():
    """简化版独立命令行接口，专门用于视频OCR处理"""
    parser = argparse.ArgumentParser(description="通义千问视频OCR处理工具")
    parser.add_argument("video_path", help="要处理的视频文件路径")
    parser.add_argument("--output", "-o", help="将结果保存为JSON文件的路径")
    parser.add_argument("--frame_interval", "-i", type=int, default=2, help="视频帧间隔，每隔多少帧提取一次，默认为2")
    parser.add_argument("--max_frames", "-m", type=int, default=20, help="最大提取帧数，默认为20")
    parser.add_argument("--api_key", "-k", help="API密钥，如果不提供则从环境变量或配置文件中获取")
    parser.add_argument("--quiet", "-q", action="store_true", help="安静模式，不输出处理过程信息，只输出结果")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        if not args.quiet:
            print(f"=== 处理视频文件 ===")
            print(f"视频路径: {args.video_path}")
            print(f"帧间隔: {args.frame_interval}, 最大帧数: {args.max_frames}")
            if args.output:
                print(f"输出JSON: {args.output}")
            print("开始处理...\n")
        
        # 处理视频
        chat_data = process_video(
            args.video_path,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            api_key=args.api_key,
            save_json=args.output
        )
        
        # 打印结果
        print("提取的对话内容:")
        print("用户消息:")
        for i, msg in enumerate(chat_data.get("user_messages", [])):
            print(f"  {i+1}. {msg}")
        
        print("\nAI助手消息:")
        for i, msg in enumerate(chat_data.get("assistant_messages", [])):
            print(f"  {i+1}. {msg}")
            
        # 打印用户操作行为
        if "user_actions" in chat_data:
            print("\n用户操作行为:")
            print(chat_data["user_actions"])
            
        if args.output and not args.quiet:
            print(f"\n结果已保存到: {args.output}")
            
        return 0
    except Exception as e:
        print(f"处理失败: {str(e)}", file=sys.stderr)
        return 1


# 确保只有在直接运行此文件时才执行main函数
if __name__ == "__main__":
    # 如果是以简化视频OCR模式运行
    if len(sys.argv) > 1 and os.path.basename(sys.argv[0]) == 'ocr_video.py':
        sys.exit(ocr_video_cli())
    else:
        main() 