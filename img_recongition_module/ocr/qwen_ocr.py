from openai import OpenAI
import os
import base64
import logging
import imghdr
from pathlib import Path
from typing import List, Union
from img_recongition_module.config import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QwenOCR:
    """
    通义千问OCR客户端类
    
    该类提供了使用通义千问模型进行图像识别的功能，支持网络图片、本地图片和视频理解。
    """
    
    def __init__(self, api_key=None):
        """
        初始化通义千问OCR客户端
        
        Args:
            api_key: API密钥，如果为None则从settings中获取
        """
        self.api_key = api_key or settings.DASHSCOPE_API_KEY
        if not self.api_key:
            raise ValueError("API密钥不能为空，请在settings中配置DASHSCOPE_API_KEY或直接传入api_key参数")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen-vl-max-2025-01-25"
        logger.info(f"通义千问OCR客户端初始化完成，使用模型: {self.model}")
    
    def encode_image(self, image_path):
        """
        将本地图片转换为base64编码
        
        Args:
            image_path: 图片路径
            
        Returns:
            str: base64编码后的图片
            
        Raises:
            FileNotFoundError: 如果图片文件不存在
            IOError: 如果读取图片文件失败
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            logger.error(f"图片文件不存在: {image_path}")
            raise
        except IOError as e:
            logger.error(f"读取图片文件失败: {image_path}, 错误: {str(e)}")
            raise
    
    def process_url_image(self, image_url, prompt="图中描绘的是什么景象?"):
        """
        处理单张网络图片
        
        Args:
            image_url: 图片URL
            prompt: 提示文本
            
        Returns:
            str: 模型返回的结果
            
        Raises:
            Exception: 如果API调用失败
        """
        try:
            logger.info(f"处理网络图片: {image_url}")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"处理网络图片失败: {image_url}, 错误: {str(e)}")
            raise
    
    def process_local_image(self, image_path, prompt="图中描绘的是什么景象?", image_format=None):
        """
        处理单张本地图片
        
        Args:
            image_path: 图片路径
            prompt: 提示文本
            image_format: 图片格式，支持png、jpeg、webp，如果为None则自动检测
            
        Returns:
            str: 模型返回的结果
            
        Raises:
            FileNotFoundError: 如果图片文件不存在
            IOError: 如果读取图片文件失败
            Exception: 如果API调用失败
        """
        try:
            # 如果未指定格式，则自动检测
            if image_format is None:
                detected_format = imghdr.what(image_path)
                if detected_format == 'jpeg':
                    image_format = 'jpeg'
                elif detected_format == 'png':
                    image_format = 'png'
                elif detected_format == 'webp':
                    image_format = 'webp'
                else:
                    # 默认使用jpeg
                    image_format = 'jpeg'
                    
            logger.info(f"处理本地图片: {image_path}, 格式: {image_format}")
            base64_image = self.encode_image(image_path)
            
            # 根据图片格式构建URL
            image_url = f"data:image/{image_format};base64,{base64_image}"
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"处理本地图片失败: {image_path}, 错误: {str(e)}")
            raise
            
    def process_image(self, image, prompt="图中描绘的是什么景象?"):
        """
        统一处理单张图片的方法，自动判断是网络图片还是本地图片
        
        Args:
            image: 图片路径或URL
            prompt: 提示文本
            
        Returns:
            str: 模型返回的结果
        """
        # 判断是否为URL
        if image.startswith(('http://', 'https://')):
            return self.process_url_image(image, prompt)
        else:
            # 本地图片，自动检测格式
            return self.process_local_image(image, prompt)
            
    def process_multiple_images(self, images: List[str], prompt="这些图描绘了什么内容？"):
        """
        处理多张图片，支持混合网络图片和本地图片
        
        Args:
            images: 图片路径或URL列表
            prompt: 提示文本
            
        Returns:
            str: 模型返回的结果
            
        Raises:
            Exception: 如果API调用失败
        """
        try:
            logger.info(f"处理多张图片，数量: {len(images)}")
            
            # 构建用户内容
            user_content = []
            
            # 添加每张图片
            for image in images:
                if image.startswith(('http://', 'https://')):
                    # 网络图片
                    logger.info(f"添加网络图片: {image}")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": image}
                    })
                else:
                    # 本地图片，自动检测格式
                    image_format = None
                    detected_format = imghdr.what(image)
                    if detected_format == 'jpeg':
                        image_format = 'jpeg'
                    elif detected_format == 'png':
                        image_format = 'png'
                    elif detected_format == 'webp':
                        image_format = 'webp'
                    else:
                        # 默认使用jpeg
                        image_format = 'jpeg'
                        
                    logger.info(f"添加本地图片: {image}, 格式: {image_format}")
                    base64_image = self.encode_image(image)
                    image_url = f"data:image/{image_format};base64,{base64_image}"
                    
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
            
            # 添加提示文本
            user_content.append({"type": "text", "text": prompt})
            
            # 调用API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"处理多张图片失败，错误: {str(e)}")
            raise
            
    def process_video_frames(self, frames: List[str], prompt="描述这个视频的具体过程"):
        """
        处理视频帧序列，支持网络图片和本地图片作为帧
        
        Args:
            frames: 视频帧图片路径或URL列表
            prompt: 提示文本
            
        Returns:
            str: 模型返回的结果
            
        Raises:
            Exception: 如果API调用失败
        """
        try:
            logger.info(f"处理视频帧序列，帧数量: {len(frames)}")
            
            # 处理每一帧
            processed_frames = []
            for frame in frames:
                if frame.startswith(('http://', 'https://')):
                    # 网络图片
                    logger.info(f"添加网络视频帧: {frame}")
                    processed_frames.append(frame)
                else:
                    # 本地图片，自动检测格式
                    image_format = None
                    detected_format = imghdr.what(frame)
                    if detected_format == 'jpeg':
                        image_format = 'jpeg'
                    elif detected_format == 'png':
                        image_format = 'png'
                    elif detected_format == 'webp':
                        image_format = 'webp'
                    else:
                        # 默认使用jpeg
                        image_format = 'jpeg'
                        
                    logger.info(f"添加本地视频帧: {frame}, 格式: {image_format}")
                    base64_image = self.encode_image(frame)
                    image_url = f"data:image/{image_format};base64,{base64_image}"
                    processed_frames.append(image_url)
            
            # 调用API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": processed_frames
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"处理视频帧序列失败，错误: {str(e)}")
            raise