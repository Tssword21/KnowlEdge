"""
KnowlEdge项目工具模块
包含通用工具函数和类
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
import hashlib
import uuid
import asyncio
from typing import Dict, List, Optional, Any

from src.config import Config

# 初始化配置
config = Config()

def setup_logging():
    """设置日志配置（控制台 + 按日轮转文件）"""
    log_dir = os.path.join(Config().data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "knowledge.log")

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 控制台处理器
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # 文件轮转处理器
    if not any(isinstance(h, TimedRotatingFileHandler) for h in root.handlers):
        fh = TimedRotatingFileHandler(log_file, when='midnight', backupCount=7, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    for noisy in [
        "multipart",
        "multipart.multipart",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access"
    ]:
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING if 'multipart' in noisy else logging.INFO)
        except Exception:
            pass
    logging.info("日志系统初始化完成（控制台+文件轮转）")


def get_user_data_path():
    """获取用户数据目录路径"""
    cfg = Config()
    return cfg.data_dir


def verify_database():
    """验证数据库结构是否正确，必要时自动初始化缺失表"""
    try:
        try:
            from src.db_utils import get_db_connection, initialize_database
        except ImportError:
            from db_utils import get_db_connection, initialize_database
        initialize_database()
        conn = get_db_connection()
        if conn is None:
            logging.error("无法连接到数据库")
            return False
        cursor = conn.cursor()
        required_tables = [
            'users',
            'user_profiles',
            'search_history',
            'user_interests',
            'user_skills',
            'user_education',
            'user_work_experience',
            'user_interactions',
            'interest_categories'
        ]
        existing = set(name for (name,) in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
        missing = [t for t in required_tables if t not in existing]
        if missing:
            logging.warning(f"数据库仍缺少表: {', '.join(missing)}")
            return False
        return True
    except Exception as e:
        logging.error(f"验证数据库时出错: {str(e)}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass 

def generate_user_id(username: str, email: str = None, extra_data: str = None) -> str:
    """
    生成唯一的用户ID
    
    Args:
        username: 用户名
        email: 邮箱（可选）
        extra_data: 额外数据（可选）
    
    Returns:
        32位哈希用户ID
    """
    # 构建唯一字符串
    unique_parts = [username]
    if email:
        unique_parts.append(email)
    if extra_data:
        unique_parts.append(extra_data)
    
    # 添加随机元素确保唯一性
    unique_parts.append(uuid.uuid4().hex[:8])
    
    unique_string = "-".join(unique_parts)
    return hashlib.md5(unique_string.encode()).hexdigest()

def generate_temp_user_id(temp_name: str) -> str:
    """
    为临时/游客用户生成用户ID
    
    Args:
        temp_name: 临时用户名
    
    Returns:
        32位哈希用户ID
    """
    import time
    temp_string = f"temp-{temp_name}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    return hashlib.md5(temp_string.encode()).hexdigest() 

def check_system_health() -> Dict[str, Any]:
    """
    检查系统整体健康状况
    
    Returns:
        包含系统状态信息的字典
    """
    health_status = {
        "overall": "healthy",
        "issues": [],
        "warnings": [],
        "components": {}
    }
    
    try:
        # 检查配置
        config = Config()
        validation = config.validate_config()
        health_status["components"]["config"] = {
            "status": "healthy" if validation["valid"] else "error",
            "details": validation
        }
        
        if not validation["valid"]:
            health_status["issues"].extend([f"缺少配置: {key}" for key in validation["missing_keys"]])
            health_status["overall"] = "error"
        
        # 检查警告
        if validation["warnings"]["serper_api_key"]:
            health_status["warnings"].append("SERPER_API_KEY未配置，Google搜索功能不可用")
        
        # 检查数据库
        db_ok = verify_database()
        health_status["components"]["database"] = {
            "status": "healthy" if db_ok else "error",
            "path": config.user_db_path
        }
        
        if not db_ok:
            health_status["issues"].append("数据库验证失败")
            health_status["overall"] = "error"
        
        # 检查数据目录
        data_dir_exists = os.path.exists(config.data_dir)
        health_status["components"]["data_directory"] = {
            "status": "healthy" if data_dir_exists else "error",
            "path": config.data_dir
        }
        
        if not data_dir_exists:
            health_status["issues"].append(f"数据目录不存在: {config.data_dir}")
            health_status["overall"] = "error"
        
        # 检查兴趣分类文件
        interest_file = os.path.join(config.data_dir, "interest_categories.json")
        interest_file_exists = os.path.exists(interest_file)
        health_status["components"]["interest_categories"] = {
            "status": "healthy" if interest_file_exists else "warning",
            "path": interest_file
        }
        
        if not interest_file_exists:
            health_status["warnings"].append("兴趣分类文件不存在")
        
        # 如果有问题但不是错误，设置为警告状态
        if health_status["overall"] == "healthy" and (health_status["warnings"] or health_status["issues"]):
            health_status["overall"] = "warning" if not health_status["issues"] else "error"
            
    except Exception as e:
        health_status["overall"] = "error"
        health_status["issues"].append(f"系统检查异常: {str(e)}")
        logging.error(f"系统健康检查失败: {e}", exc_info=True)
    
    return health_status

async def safe_llm_call(llm_interface, prompt: str, system_message: str = None, max_retries: int = 3) -> str:
    """
    安全的LLM调用，带有重试机制
    
    Args:
        llm_interface: LLM接口实例
        prompt: 用户提示词
        system_message: 系统消息
        max_retries: 最大重试次数
    
    Returns:
        LLM响应结果
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = await llm_interface.call_llm(prompt, system_message)
            if result and result.strip():
                return result.strip()
            else:
                last_error = "LLM返回空结果"
        except Exception as e:
            last_error = str(e)
            logging.warning(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2 ** attempt, 10))  # 指数退避
    
    error_msg = f"LLM调用失败，已重试{max_retries}次，最后错误: {last_error}"
    logging.error(error_msg)
    raise Exception(error_msg)

def format_error_response(error_type: str, message: str, details: str = None) -> Dict[str, Any]:
    """
    格式化错误响应
    
    Args:
        error_type: 错误类型代码
        message: 错误消息
        details: 详细信息（可选）
    
    Returns:
        格式化的错误响应字典
    """
    response = {
        "error": message,
        "code": error_type,
        "timestamp": get_current_timestamp()
    }
    
    if details:
        response["details"] = details
    
    return response

def get_current_timestamp() -> str:
    """获取当前时间戳字符串"""
    from datetime import datetime
    return datetime.now().isoformat()

def validate_search_params(query: str, platform: str, num_results: int) -> Optional[str]:
    """
    验证搜索参数
    
    Args:
        query: 搜索查询
        platform: 搜索平台
        num_results: 结果数量
    
    Returns:
        如果验证失败返回错误消息，否则返回None
    """
    if not query or not query.strip():
        return "搜索查询不能为空"
    
    if len(query.strip()) < 2:
        return "搜索查询至少需要2个字符"
    
    if len(query) > 500:
        return "搜索查询过长，请限制在500字符以内"
    
    valid_platforms = ["arXiv论文", "谷歌学术", "混合搜索", "综合资讯", "全平台"]
    if platform not in valid_platforms:
        return f"不支持的搜索平台: {platform}"
    
    if not isinstance(num_results, int) or num_results < 1 or num_results > 50:
        return "结果数量必须在1-50之间"
    
    return None 