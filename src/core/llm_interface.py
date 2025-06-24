"""
LLM接口模块
提供与大语言模型交互的功能
"""
import logging
import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Union, AsyncIterator, Any, Optional
from dotenv import load_dotenv
from src.config import Config

class LLMInterface:
    """LLM接口类，用于与大型语言模型进行交互"""
    
    def __init__(self):
        """初始化LLM接口"""
        # 加载配置
        load_dotenv()
        config = Config()
        
        # 从配置获取API信息
        self.api_key = config.llm_api_key
        self.api_base = config.llm_api_base
        self.model = config.llm_model
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4000"))
        
        # 初始化aiohttp会话
        self.session = None
        
        logging.info(f"LLM接口初始化完成，使用模型: {self.model}, API基础URL: {self.api_base}")
        
    async def get_session(self) -> aiohttp.ClientSession:
        """获取或创建aiohttp会话"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
        
    async def close(self):
        """关闭aiohttp会话"""
        if self.session and not self.session.closed:
            await self.session.close()
            
    async def call_llm(self, prompt: str, system_message: str = None, model: str = None) -> str:
        """
        调用LLM生成回复
        
        Args:
            prompt: 用户提示词
            system_message: 系统消息
            model: 指定使用的模型，如果为None则使用默认模型
            
        Returns:
            生成的回复文本
        """
        try:
            # 使用指定的模型或默认模型
            use_model = model or self.model
            
            # 构建消息列表
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # 构建API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": use_model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            # 获取会话
            session = await self.get_session()
            
            # 调用API
            async with session.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API返回错误: {response.status}, {error_text}")
                
                # 解析响应
                result = await response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    reply = result["choices"][0]["message"]["content"]
                    return reply
                else:
                    raise Exception(f"API响应格式错误: {result}")
            
        except Exception as e:
            logging.error(f"调用LLM时出错: {e}")
            return f"调用LLM时出错: {str(e)}"
            
    async def call_llm_stream(self, prompt: str, system_message: str = None, model: str = None) -> AsyncIterator[str]:
        """
        流式调用LLM生成回复
        
        Args:
            prompt: 用户提示词
            system_message: 系统消息
            model: 指定使用的模型，如果为None则使用默认模型
            
        Returns:
            生成的回复文本流
        """
        try:
            # 使用指定的模型或默认模型
            use_model = model or self.model
            
            # 构建消息列表
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # 构建API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": use_model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True
            }
            
            # 获取会话
            session = await self.get_session()
            
            # 调用API进行流式生成
            async with session.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API返回错误: {response.status}, {error_text}")
                
                # 处理流式响应
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        line = line[6:]  # 移除 'data: ' 前缀
                        if line == '[DONE]':
                            break
                        try:
                            chunk = json.loads(line)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            logging.warning(f"无法解析流式响应行: {line}")
                            continue
                        
        except Exception as e:
            logging.error(f"流式调用LLM时出错: {e}")
            yield f"流式调用LLM时出错: {str(e)}"
            
    async def translate_text(self, text: str, target_language: str = "en") -> str:
        """
        使用LLM翻译文本
        
        Args:
            text: 要翻译的文本
            target_language: 目标语言，默认为英语(en)
            
        Returns:
            翻译后的文本
        """
        try:
            language_map = {
                "en": "英语(English)",
                "zh": "中文(Chinese)",
                "ja": "日语(Japanese)",
                "ko": "韩语(Korean)",
                "fr": "法语(French)",
                "de": "德语(German)",
                "es": "西班牙语(Spanish)",
                "ru": "俄语(Russian)"
            }
            
            target_lang_name = language_map.get(target_language, target_language)
            
            prompt = f"""
            请将以下文本准确翻译成{target_lang_name}，保留原文的所有专业术语和学术含义：
            
            {text}
            
            只需返回翻译结果，不要包含任何解释或其他内容。
            """
            
            # 调用LLM进行翻译
            translated_text = await self.call_llm(prompt)
            return translated_text.strip()
            
        except Exception as e:
            logging.error(f"翻译文本时出错: {e}")
            return text  # 如果翻译失败，返回原文
            
    async def parallel_process(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        并行处理多个LLM任务
        
        Args:
            tasks: 任务列表，每个任务是一个字典，包含:
                  - prompt: 提示词
                  - system_message: 系统消息(可选)
                  - model: 使用的模型(可选)
                  - task_id: 任务ID(可选)
        
        Returns:
            处理结果列表，每个结果是一个字典，包含:
                  - task_id: 任务ID(如果提供)
                  - result: 生成的文本
                  - error: 错误信息(如果有)
        """
        async def process_task(task):
            task_id = task.get("task_id")
            prompt = task.get("prompt")
            system_message = task.get("system_message")
            model = task.get("model")
            
            result = {"task_id": task_id}
            
            try:
                if not prompt:
                    raise ValueError("任务缺少prompt字段")
                
                response = await self.call_llm(prompt, system_message, model)
                result["result"] = response
            except Exception as e:
                logging.error(f"处理任务 {task_id} 时出错: {e}")
                result["error"] = str(e)
                
            return result
            
        # 创建所有任务的协程
        coroutines = [process_task(task) for task in tasks]
        
        # 并行执行所有任务
        results = await asyncio.gather(*coroutines)
        
        return results

    async def yield_with_title(self, title: str, llm_prompt: str, system_msg: str = "你是一位资深写作者。") -> AsyncIterator[str]:
        """通用小工具：先把标题丢给前端，再流式产出正文"""
        # 先把第一行标题 yield 出去，避免前端一片空白
        yield title + "\n\n"
        async for tok in self.call_llm_stream(llm_prompt, system_msg):
            yield tok 