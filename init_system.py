# init_system.py
import os
import logging
import json

# 导入数据库工具模块
from db_utils import initialize_database, get_db_connection

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 系统配置
CONFIG = {
    "DATA_DIR": "./user_data",
    "DB_PATH": "./user_data/user_profiles.db",
    "LOGS_DIR": "./user_data/logs",
    "CACHE_DIR": "./user_data/cache",
    "INTEREST_CATEGORIES_FILE": "./user_data/interest_categories.json"
}

def create_directory_structure():
    """创建必要的目录结构"""
    directories = [
        CONFIG["DATA_DIR"],
        CONFIG["LOGS_DIR"],
        CONFIG["CACHE_DIR"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"创建目录: {directory}")
    
    return True

def create_interest_categories():
    """创建默认的兴趣分类体系"""
    interest_categories = {
        "技术": ["人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉", 
               "大数据", "云计算", "区块链", "物联网", "网络安全", "数据库"],
        "科学": ["物理学", "化学", "生物学", "天文学", "数学", "医学", "地质学", "环境科学"],
        "商业": ["管理", "市场营销", "金融", "创业", "投资", "电子商务", "人力资源"],
        "艺术": ["绘画", "音乐", "电影", "文学", "设计", "摄影", "建筑"],
        "教育": ["教学方法", "学习理论", "教育技术", "高等教育", "职业教育"],
        "健康": ["营养", "健身", "心理健康", "医疗技术", "公共卫生"]
    }
    
    with open(CONFIG["INTEREST_CATEGORIES_FILE"], 'w', encoding='utf-8') as f:
        json.dump(interest_categories, f, ensure_ascii=False, indent=4)
    
    logger.info(f"兴趣分类体系已创建: {CONFIG['INTEREST_CATEGORIES_FILE']}")
    return True

def verify_system():
    """验证系统初始化是否成功"""
    checks = {
        "目录结构": os.path.exists(CONFIG["DATA_DIR"]) and 
                 os.path.exists(CONFIG["LOGS_DIR"]) and 
                 os.path.exists(CONFIG["CACHE_DIR"]),
        "数据库": os.path.exists(CONFIG["DB_PATH"]),
        "兴趣分类": os.path.exists(CONFIG["INTEREST_CATEGORIES_FILE"])
    }
    
    all_passed = all(checks.values())
    
    if all_passed:
        logger.info("系统验证通过，所有组件已正确初始化")
    else:
        failed = [k for k, v in checks.items() if not v]
        logger.error(f"系统验证失败，以下组件未正确初始化: {', '.join(failed)}")
    
    return all_passed

def main():
    """执行初始化流程"""
    logger.info("开始系统初始化...")
    
    create_directory_structure()
    initialize_database()
    create_interest_categories()
    
    if verify_system():
        logger.info("系统初始化完成，可以开始使用了")
        return True
    else:
        logger.error("系统初始化失败，请检查错误并重试")
        return False

if __name__ == "__main__":
    main()