#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通义千问OCR客户端测试运行脚本

这个脚本是一个简单的包装器，用于运行OCR测试。
它避免了PyCharm将测试函数误认为pytest测试的问题。
"""

import sys
import os
from img_recongition_module.ocr_main import main

if __name__ == "__main__":
    # 将命令行参数传递给main函数
    main()
    print("\n测试完成！") 