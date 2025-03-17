from openai import OpenAI
import os
import base64
import logging
import imghdr
import json
from pathlib import Path
from typing import List, Union, Dict, Any
from img_recongition_module.config import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QwenOCR:
    """
    通义千问OCR客户端类
    
    该类提供了使用通义千问模型进行图像识别的功能，支持网络图片、本地图片和视频理解。
    专门优化用于识别手机端大模型评测中的对话截图和录像。
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
        
    def generate_prompt(self, prompt_type='single', custom_prompt=None):
        """
        生成针对手机端大模型评测场景的提示词
        
        Args:
            prompt_type: 提示词类型，可选值: 'single', 'multiple', 'video', 'custom'
            custom_prompt: 自定义提示词，当prompt_type为'custom'时使用
            
        Returns:
            str: 生成的提示词
        """
        if custom_prompt:
            return custom_prompt
            
        prompts = {
            'single': """
请识别这张手机截图中的对话内容。这是手机端大模型对话的截图，请提取用户和AI助手之间的对话文本。

对话特征说明:
- 右侧的信息为用户发送的消息（通常是深色气泡）
- 左侧的回复为AI大模型的回答（通常是浅色气泡）
- AI大模型回答后通常会在底部自动生成几个引导用户继续对话的小问题
- 请忽略这些位于大模型回复主体下方的引导问题，不要将它们包含在assistant_messages中
- 只提取真正的对话内容

按照以下格式输出:
1. 清晰区分用户和AI助手的发言，标明是谁在说话
2. 保持原始文本的格式，包括段落和标点
3. 如果有代码块、列表等特殊格式，请保留
4. 忽略界面中的其他元素（如按钮、时间戳等）
5. 忽略AI助手回复下方的自动生成引导问题
6. 如果某些文字无法辨认，请用[无法识别]标注
7. 返回JSON格式，包含"user_messages"和"assistant_messages"两个数组字段

示例输出:
{
  "user_messages": ["你好，请问你是谁？", "你能做什么？"],
  "assistant_messages": ["我是AI助手Claude，很高兴为您服务！", "我可以回答问题、提供信息、创作内容等。请告诉我您需要什么帮助。"]
}
            """,
            
            'multiple': """
请识别这些手机截图中的完整对话内容。这些是同一个对话的连续截图，表示一个完整的对话记录。请按顺序提取用户和AI助手之间的所有对话文本。

对话特征说明:
- 右侧的信息为用户发送的消息（通常是深色气泡）
- 左侧的回复为AI大模型的回答（通常是浅色气泡）
- AI大模型回答后通常会在底部自动生成几个引导用户继续对话的小问题
- 请忽略这些位于大模型回复主体下方的引导问题，不要将它们包含在assistant_messages中
- 只提取真正的对话内容

请注意:
1. 这些图片是连续的，代表了一个连贯的对话流程
2. 请合并图片间的内容，确保没有重复或遗漏
3. 清晰区分用户和AI助手的发言，标明是谁在说话
4. 保持原始文本的格式，包括段落和标点
5. 如果有代码块、列表等特殊格式，请保留
6. 忽略界面中的其他元素（如按钮、时间戳等）
7. 忽略AI助手回复下方的自动生成引导问题
8. 如果某些文字无法辨认，请用[无法识别]标注
9. 返回JSON格式，包含"user_messages"和"assistant_messages"两个数组字段，按对话顺序排列

示例输出:
{
  "user_messages": ["你好，请问你是谁？", "你能做什么？", "请给我写一段Python代码计算斐波那契数列"],
  "assistant_messages": ["我是AI助手Claude，很高兴为您服务！", "我可以回答问题、提供信息、创作内容等。请告诉我您需要什么帮助。", "以下是计算斐波那契数列的Python代码：\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    else:\n        fib = [0, 1]\n        for i in range(2, n):\n            fib.append(fib[i-1] + fib[i-2])\n        return fib\n\nprint(fibonacci(10))\n```"]
}
            """,
            
            'video': """
请分析这个视频（由多个连续帧组成）中的手机对话内容。这是在手机上与AI大模型对话的录屏，请提取视频中用户和AI助手之间的完整对话文本。

对话特征说明:
- 右侧的信息为用户发送的消息（通常是深色气泡）
- 左侧的回复为AI大模型的回答（通常是浅色气泡）
- AI大模型回答后通常会在底部自动生成几个引导用户继续对话的小问题
- 请忽略这些位于大模型回复主体下方的引导问题，不要将它们包含在assistant_messages中
- 只提取真正的对话内容

请注意:
1. 这些视频帧是连续的，代表了一个连贯的对话流程
2. 请合并帧间的内容，确保没有重复或遗漏
3. 清晰区分用户和AI助手的发言，标明是谁在说话
4. 保持原始文本的格式，包括段落和标点
5. 如果有代码块、列表等特殊格式，请保留
6. 忽略界面中的其他元素（如按钮、时间戳等）
7. 忽略AI助手回复下方的自动生成引导问题
8. 如果某些文字无法辨认，请用[无法识别]标注
9. 返回JSON格式，包含"user_messages"和"assistant_messages"两个数组字段，按对话顺序排列

示例输出:
{
  "user_messages": ["你好，请问你是谁？", "你能做什么？", "请给我写一段Python代码计算斐波那契数列"],
  "assistant_messages": ["我是AI助手Claude，很高兴为您服务！", "我可以回答问题、提供信息、创作内容等。请告诉我您需要什么帮助。", "以下是计算斐波那契数列的Python代码：\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    else:\n        fib = [0, 1]\n        for i in range(2, n):\n            fib.append(fib[i-1] + fib[i-2])\n        return fib\n\nprint(fibonacci(10))\n```"]
}
            """
        }
        
        return prompts.get(prompt_type, prompts['single'])
    
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
    
    def process_url_image(self, image_url, prompt_type='single', custom_prompt=None):
        """
        处理单张网络图片
        
        Args:
            image_url: 图片URL
            prompt_type: 提示词类型，可选值: 'single', 'multiple', 'video', 'custom'
            custom_prompt: 自定义提示词，当prompt_type为'custom'时使用
            
        Returns:
            str: 模型返回的结果
            
        Raises:
            Exception: 如果API调用失败
        """
        try:
            prompt = self.generate_prompt(prompt_type, custom_prompt)
            logger.info(f"处理网络图片: {image_url}")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant specialized in extracting text from mobile chat screenshots and videos. Pay close attention to the structure of mobile chat interfaces: user messages typically appear on the right side (often in lighter bubbles), while AI assistant responses appear on the left side (often in darker bubbles). IMPORTANT: The AI assistant often suggests follow-up questions at the bottom of its responses - these should NOT be included as part of the assistant's message. Only extract the actual conversation content, ignoring any suggested follow-up questions."}],
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
                response_format={"type": "json_object"}
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"处理网络图片失败: {image_url}, 错误: {str(e)}")
            raise
    
    def process_local_image(self, image_path, prompt_type='single', custom_prompt=None, image_format=None):
        """
        处理单张本地图片
        
        Args:
            image_path: 图片路径
            prompt_type: 提示词类型，可选值: 'single', 'multiple', 'video', 'custom'
            custom_prompt: 自定义提示词，当prompt_type为'custom'时使用
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
            
            prompt = self.generate_prompt(prompt_type, custom_prompt)        
            logger.info(f"处理本地图片: {image_path}, 格式: {image_format}")
            base64_image = self.encode_image(image_path)
            
            # 根据图片格式构建URL
            image_url = f"data:image/{image_format};base64,{base64_image}"
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant specialized in extracting text from mobile chat screenshots and videos. Pay close attention to the structure of mobile chat interfaces: user messages typically appear on the right side (often in lighter bubbles), while AI assistant responses appear on the left side (often in darker bubbles). IMPORTANT: The AI assistant often suggests follow-up questions at the bottom of its responses - these should NOT be included as part of the assistant's message. Only extract the actual conversation content, ignoring any suggested follow-up questions."}]
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
                response_format={"type": "json_object"}
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"处理本地图片失败: {image_path}, 错误: {str(e)}")
            raise
            
    def process_image(self, image, prompt_type='single', custom_prompt=None):
        """
        统一处理单张图片的方法，自动判断是网络图片还是本地图片
        
        Args:
            image: 图片路径或URL
            prompt_type: 提示词类型，可选值: 'single', 'multiple', 'video', 'custom'
            custom_prompt: 自定义提示词，当prompt_type为'custom'时使用
            
        Returns:
            str: 模型返回的结果
        """
        # 判断是否为URL
        if image.startswith(('http://', 'https://')):
            return self.process_url_image(image, prompt_type, custom_prompt)
        else:
            # 本地图片，自动检测格式
            return self.process_local_image(image, prompt_type, custom_prompt)
            
    def process_multiple_images(self, images: List[str], prompt_type='multiple', custom_prompt=None):
        """
        处理多张图片，支持混合网络图片和本地图片
        
        Args:
            images: 图片路径或URL列表
            prompt_type: 提示词类型，可选值: 'single', 'multiple', 'video', 'custom'
            custom_prompt: 自定义提示词，当prompt_type为'custom'时使用
            
        Returns:
            str: 模型返回的结果
            
        Raises:
            Exception: 如果API调用失败
        """
        try:
            prompt = self.generate_prompt(prompt_type, custom_prompt)
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
                        "content": [{"type": "text", "text": "You are a helpful assistant specialized in extracting text from mobile chat screenshots and videos. Pay close attention to the structure of mobile chat interfaces: user messages typically appear on the right side (often in lighter bubbles), while AI assistant responses appear on the left side (often in darker bubbles). IMPORTANT: The AI assistant often suggests follow-up questions at the bottom of its responses - these should NOT be included as part of the assistant's message. Only extract the actual conversation content, ignoring any suggested follow-up questions."}]
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"处理多张图片失败，错误: {str(e)}")
            raise
            
    def process_video_frames(self, frames: List[str], prompt_type='video', custom_prompt=None):
        """
        处理视频帧序列，支持网络图片和本地图片作为帧
        
        Args:
            frames: 视频帧图片路径或URL列表
            prompt_type: 提示词类型，可选值: 'single', 'multiple', 'video', 'custom'
            custom_prompt: 自定义提示词，当prompt_type为'custom'时使用
            
        Returns:
            str: 模型返回的结果
            
        Raises:
            Exception: 如果API调用失败
        """
        try:
            prompt = self.generate_prompt(prompt_type, custom_prompt)
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
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant specialized in extracting text from mobile chat screenshots and videos. Pay close attention to the structure of mobile chat interfaces: user messages typically appear on the right side (often in lighter bubbles), while AI assistant responses appear on the left side (often in darker bubbles). IMPORTANT: The AI assistant often suggests follow-up questions at the bottom of its responses - these should NOT be included as part of the assistant's message. Only extract the actual conversation content, ignoring any suggested follow-up questions."}]
                    },
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
                response_format={"type": "json_object"}
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"处理视频帧序列失败，错误: {str(e)}")
            raise
            
    def parse_ocr_result(self, ocr_result: str) -> Dict[str, Any]:
        """
        解析OCR结果为Python字典
        
        Args:
            ocr_result: OCR返回的JSON字符串
            
        Returns:
            Dict: 解析后的字典，包含user_messages和assistant_messages
        """
        try:
            return json.loads(ocr_result)
        except json.JSONDecodeError as e:
            logger.error(f"解析OCR结果失败: {str(e)}")
            # 如果JSON解析失败，尝试提取有效的JSON部分
            import re
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, ocr_result)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            
            # 如果仍然失败，返回一个基本结构
            return {
                "user_messages": [],
                "assistant_messages": [],
                "raw_text": ocr_result
            }