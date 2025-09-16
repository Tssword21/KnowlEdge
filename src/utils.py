"""
KnowlEdge项目工具模块
包含通用工具函数和类
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler

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