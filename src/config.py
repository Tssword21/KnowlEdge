"""
KnowlEdge项目配置模块
包含全局配置、常量和环境变量管理
"""
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志（默认INFO，避免过多DEBUG噪音）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 常量配置
CONFIG = {
    "API_KEYS": {
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        "qwen": os.getenv("QWEN_API_KEY"),
        "serper": os.getenv("SERPER_API_KEY"),
        "baidu_translate": os.getenv("BAIDU_API_KEY"),
    },
    "MODELS": {
        "BERT": "bert-base-multilingual-cased",
        "LLM": "deepseek-reasoner",
    },
    "DATA_DIR": "./user_data",
    "DB_PATH": "./user_data/user_profiles.db"
}

# 确保数据目录存在
os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)

# 惰性加载BERT，避免启动时间过长
_tokenizer = None
_bert_model = None

def get_bert():
    global _tokenizer, _bert_model
    if _tokenizer is None or _bert_model is None:
        from transformers import BertTokenizer, BertModel
        import torch  # noqa: F401
        _tokenizer = BertTokenizer.from_pretrained(CONFIG["MODELS"]["BERT"])
        _bert_model = BertModel.from_pretrained(CONFIG["MODELS"]["BERT"])
    return _tokenizer, _bert_model

class Config:
    def __init__(self, config_path=None):
        load_dotenv(dotenv_path=config_path)
        self.llm_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.llm_api_base = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.baidu_translate_key = os.getenv("BAIDU_API_KEY", "")
        self.data_dir = os.getenv("DATA_DIR", "./user_data")
        self.user_db_path = os.getenv("DB_PATH", os.path.join(self.data_dir, "user_profiles.db"))
        self.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
        self.search_timeout = int(os.getenv("SEARCH_TIMEOUT", "30"))
        os.makedirs(self.data_dir, exist_ok=True)
    def get_model_config(self):
        return {
            "llm_model": self.llm_model,
            "llm_api_base": self.llm_api_base
        }
    def get_api_keys(self):
        return {
            "llm_api_key": self.llm_api_key,
            "serper_api_key": self.serper_api_key,
            "google_api_key": self.google_api_key,
            "baidu_translate_key": self.baidu_translate_key
        } 