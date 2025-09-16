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

# 进程级session（避免未关闭连接）
_GLOBAL_SESSION: Optional[aiohttp.ClientSession] = None

_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=60)
_MAX_RETRIES = 2

class LLMInterface:
    """LLM接口类，用于与大型语言模型进行交互"""
    
    def __init__(self, model_override: Optional[str] = None):
        """初始化LLM接口"""
        load_dotenv()
        config = Config()
        
        self.api_key = config.llm_api_key
        self.api_base = config.llm_api_base
        self.model = model_override or config.llm_model
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        raw_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.max_tokens = max(1, min(raw_max_tokens, 8192))
        logging.info(f"LLM接口初始化完成，使用模型: {self.model}, API基础URL: {self.api_base}")
    
    async def get_session(self) -> aiohttp.ClientSession:
        global _GLOBAL_SESSION
        if _GLOBAL_SESSION is None or _GLOBAL_SESSION.closed:
            _GLOBAL_SESSION = aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT)
        return _GLOBAL_SESSION
        
    async def close(self):
        global _GLOBAL_SESSION
        if _GLOBAL_SESSION and not _GLOBAL_SESSION.closed:
            await _GLOBAL_SESSION.close()
            _GLOBAL_SESSION = None
            
    async def _post_json(self, path: str, payload: dict) -> aiohttp.ClientResponse:
        session = await self.get_session()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        return await session.post(f"{self.api_base}{path}", headers=headers, json=payload)

    async def call_llm(self, prompt: str, system_message: str = None, model: str = None) -> str:
        use_model = model or self.model
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        last_err = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with (await self._post_json("/v1/chat/completions", payload)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API返回错误: {response.status}, {error_text}")
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    raise Exception(f"API响应格式错误: {result}")
            except Exception as e:
                last_err = e
                logging.warning(f"LLM调用失败(第{attempt+1}次): {e}")
                await asyncio.sleep(min(1 + attempt, 3))
        logging.error(f"调用LLM时出错: {last_err}")
        return f"调用LLM时出错: {str(last_err)}"
            
    async def call_llm_stream(self, prompt: str, system_message: str = None, model: str = None) -> AsyncIterator[str]:
        use_model = model or self.model
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": use_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        try:
            async with (await self._post_json("/v1/chat/completions", payload)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API返回错误: {response.status}, {error_text}")
                async for line in response.content:
                    line = line.decode('utf-8', errors='ignore').strip()
                    if not line.startswith('data: '):
                        continue
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
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
            prompt = f"请将以下文本准确翻译成{target_lang_name}，保留原文的所有专业术语和学术含义：\n\n{text}\n\n只需返回翻译结果，不要包含任何解释或其他内容。"
            translated_text = await self.call_llm(prompt)
            return translated_text.strip()
        except Exception as e:
            logging.error(f"翻译文本时出错: {e}")
            return text
            
    async def parallel_process(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        return await asyncio.gather(*[process_task(task) for task in tasks])

    async def yield_with_title(self, title: str, llm_prompt: str, system_msg: str = "你是一位资深写作者。") -> AsyncIterator[str]:
        yield title + "\n\n"
        async for tok in self.call_llm_stream(llm_prompt, system_msg):
            yield tok 