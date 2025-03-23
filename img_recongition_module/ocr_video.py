#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通义千问视频OCR处理工具

这是一个简化版的命令行工具，专门用于处理视频文件并提取其中的对话内容和用户操作行为。
该工具使用通义千问大语言模型的视觉能力，将视频转换为帧序列后进行OCR和内容理解。

运行方法：
python -m img_recongition_module.ocr_video /path/to/video.mp4
python -m img_recongition_module.ocr_video /path/to/video.mp4 --output output.json
python -m img_recongition_module.ocr_video /path/to/video.mp4 --frame_interval 3 --max_frames 15 --api_key your_api_key
"""

import sys
from img_recongition_module.ocr_main import ocr_video_cli

if __name__ == "__main__":
    sys.exit(ocr_video_cli()) 