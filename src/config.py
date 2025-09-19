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

# 惰性加载BERT，避免启动时间过长
_tokenizer = None
_bert_model = None

def get_bert():
    """获取BERT模型和分词器（惰性加载）"""
    global _tokenizer, _bert_model
    if _tokenizer is None or _bert_model is None:
        from transformers import BertTokenizer, BertModel
        import torch  # noqa: F401
        model_name = "bert-base-multilingual-cased"
        _tokenizer = BertTokenizer.from_pretrained(model_name)
        _bert_model = BertModel.from_pretrained(model_name)
        logging.info(f"BERT模型加载完成: {model_name}")
    return _tokenizer, _bert_model

class Config:
    """统一的配置管理类"""
    
    def __init__(self, config_path=None):
        """初始化配置"""
        load_dotenv(dotenv_path=config_path)
        
        # LLM配置
        self.llm_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.llm_api_base = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        
        # 搜索API配置
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.baidu_translate_key = os.getenv("BAIDU_API_KEY", "")
        
        # 数据存储配置
        self.data_dir = os.getenv("DATA_DIR", "./user_data")
        self.user_db_path = os.getenv("DB_PATH", os.path.join(self.data_dir, "user_profiles.db"))
        
        # 搜索配置
        self.max_search_results = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
        self.search_timeout = int(os.getenv("SEARCH_TIMEOUT", "30"))
        
        # 模型配置
        self.bert_model_name = os.getenv("BERT_MODEL", "bert-base-multilingual-cased")
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        logging.info(f"配置加载完成 - 数据目录: {self.data_dir}")
    
    def get_model_config(self):
        """获取模型配置"""
        return {
            "llm_model": self.llm_model,
            "llm_api_base": self.llm_api_base,
            "bert_model": self.bert_model_name
        }
    
    def get_api_keys(self):
        """获取API密钥配置"""
        return {
            "llm_api_key": self.llm_api_key,
            "serper_api_key": self.serper_api_key,
            "google_api_key": self.google_api_key,
            "baidu_translate_key": self.baidu_translate_key
        } 
    
    def validate_config(self):
        """验证配置完整性"""
        missing_keys = []
        
        if not self.llm_api_key.strip():
            missing_keys.append("DEEPSEEK_API_KEY")
        
        return {
            "valid": len(missing_keys) == 0,
            "missing_keys": missing_keys,
            "warnings": {
                "serper_api_key": not self.serper_api_key.strip(),
                "google_api_key": not self.google_api_key.strip()
            }
        }

# 全局配置实例（向后兼容）
_global_config = None

def get_config():
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config 