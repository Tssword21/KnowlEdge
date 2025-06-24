#!/usr/bin/env python
"""
初始化KnowlEdge系统
- 创建必要的目录结构
- 初始化数据库
- 配置环境变量
"""
import os
import sys
import json
import logging
import sqlite3
from pathlib import Path

# 添加项目根目录到Python路径
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
sys.path.append(os.path.join(parent_path, "src"))

from src.utils import setup_logging, get_user_data_path, verify_database
from src.config import Config

# 设置日志
setup_logging()

def create_directories():
    """创建所需的目录结构"""
    logging.info("创建必要的目录结构...")
    
    # 获取用户数据路径
    user_data_path = get_user_data_path()
    
    # 确保目录存在
    os.makedirs(user_data_path, exist_ok=True)
    os.makedirs(os.path.join(parent_path, "templates"), exist_ok=True)
    os.makedirs(os.path.join(parent_path, "static"), exist_ok=True)
    
    logging.info(f"目录结构创建完成: {user_data_path}")
    return True

def initialize_database():
    """初始化SQLite数据库"""
    logging.info("初始化数据库...")
    
    # 获取数据库路径
    config = Config()
    db_path = config.user_db_path
    
    if verify_database():
        logging.info("数据库已经存在并且结构有效，跳过初始化。")
        return True
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 连接到数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            occupation TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建用户画像表（向后兼容）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            profile_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建搜索历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query TEXT,
            platform TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # 创建兴趣分类表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interest_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            subcategories TEXT
        )
        ''')
        
        # 创建用户兴趣表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_interests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            topic TEXT,
            category TEXT,
            weight REAL DEFAULT 5.0,
            reason TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # 创建用户技能表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            skill TEXT,
            level TEXT,
            category TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # 创建用户教育背景表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_education (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            institution TEXT,
            major TEXT,
            degree TEXT,
            time_period TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # 创建用户工作经历表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_work_experience (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            company TEXT,
            position TEXT,
            time_period TEXT,
            description TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # 创建用户交互记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            content_id TEXT,
            action_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # 提交更改并关闭连接
        conn.commit()
        conn.close()
        
        logging.info(f"数据库初始化成功: {db_path}")
        return True
    
    except Exception as e:
        logging.error(f"数据库初始化失败: {str(e)}")
        return False

def copy_template_env():
    """复制.env模板文件"""
    logging.info("准备环境变量配置...")
    
    template_env_path = os.path.join(parent_path, "src", "template.env")
    env_path = os.path.join(parent_path, ".env")
    
    if os.path.exists(env_path):
        logging.info(f".env文件已存在: {env_path}")
        return True
    
    if not os.path.exists(template_env_path):
        logging.warning("template.env文件不存在，无法创建.env文件")
        return False
    
    try:
        # 复制模板文件
        with open(template_env_path, "r", encoding="utf-8") as template:
            template_content = template.read()
            
        with open(env_path, "w", encoding="utf-8") as env_file:
            env_file.write(template_content)
            
        logging.info(f".env文件创建成功: {env_path}")
        print(f"\n请编辑 {env_path} 文件，填入您的API密钥和其他配置信息。\n")
        return True
    
    except Exception as e:
        logging.error(f"创建.env文件失败: {str(e)}")
        return False

def create_interest_categories():
    """创建兴趣类别数据"""
    logging.info("创建兴趣类别数据...")
    
    # 获取数据库路径
    config = Config()
    db_path = config.user_db_path
    
    # 兴趣类别数据
    interest_data_path = os.path.join(get_user_data_path(), "interest_categories.json")
    
    # 检查是否已存在
    if os.path.exists(interest_data_path):
        logging.info(f"兴趣类别数据已存在: {interest_data_path}")
        return True
    
    # 默认兴趣类别
    default_categories = {
        "计算机科学": ["人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉", 
                  "数据库", "网络安全", "分布式系统", "云计算", "软件工程"],
        "生物医学": ["分子生物学", "遗传学", "蛋白质组学", "生物信息学", "药理学", 
                 "流行病学", "免疫学", "神经科学", "微生物学", "医学成像"],
        "物理学": ["量子力学", "粒子物理学", "天体物理学", "凝聚态物理", "光学", 
               "热力学", "电磁学", "相对论", "核物理学", "声学"],
        "化学": ["有机化学", "无机化学", "物理化学", "分析化学", "生物化学", 
              "材料化学", "高分子化学", "催化", "电化学", "计算化学"],
        "工程学": ["机械工程", "电气工程", "土木工程", "化学工程", "航空航天工程", 
               "生物医学工程", "环境工程", "材料工程", "系统工程", "工业工程"],
        "数学": ["代数", "几何", "分析", "统计学", "概率论", "离散数学", 
              "数值分析", "拓扑学", "运筹学", "密码学"],
        "经济与商业": ["宏观经济学", "微观经济学", "金融", "市场营销", "管理", 
                 "会计", "国际贸易", "经济政策", "创业", "商业分析"],
        "社会科学": ["心理学", "社会学", "人类学", "政治学", "教育学", 
               "传播学", "语言学", "地理学", "法学", "历史学"],
        "艺术与人文": ["文学", "哲学", "历史", "艺术史", "音乐", 
                "电影研究", "戏剧", "语言", "宗教研究", "文化研究"],
        "环境与地球科学": ["生态学", "气候学", "地质学", "海洋学", "大气科学", 
                   "环境科学", "地球物理学", "水文学", "地理信息系统", "可持续发展"]
    }
    
    try:
        # 保存到JSON文件
        with open(interest_data_path, "w", encoding="utf-8") as f:
            json.dump(default_categories, f, ensure_ascii=False, indent=2)
            
        # 同时更新数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 清空现有数据
        cursor.execute("DELETE FROM interest_categories")
        
        # 插入新数据
        for category, subcategories in default_categories.items():
            cursor.execute(
                "INSERT INTO interest_categories (category, subcategories) VALUES (?, ?)",
                (category, json.dumps(subcategories, ensure_ascii=False))
            )
            
        conn.commit()
        conn.close()
        
        logging.info("兴趣类别数据创建成功")
        return True
    
    except Exception as e:
        logging.error(f"创建兴趣类别数据失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("\n===== KnowlEdge系统初始化 =====\n")
    
    # 创建目录结构
    if not create_directories():
        print("❌ 创建目录结构失败，请检查日志并重试。")
        return
    print("✅ 目录结构创建成功")
    
    # 初始化数据库
    if not initialize_database():
        print("❌ 数据库初始化失败，请检查日志并重试。")
        return
    print("✅ 数据库初始化成功")
    
    # 创建.env文件
    if not copy_template_env():
        print("⚠️ .env文件创建失败，您需要手动配置环境变量。")
    else:
        print("✅ .env模板文件创建成功")
    
    # 创建兴趣类别数据
    if not create_interest_categories():
        print("⚠️ 兴趣类别数据创建失败，系统将使用默认类别。")
    else:
        print("✅ 兴趣类别数据创建成功")
    
    print("\n✨ KnowlEdge系统初始化完成！✨")
    print("\n运行说明:")
    print("1. 确保您已填写正确的API密钥和其他配置信息")
    print("2. 启动应用: cd src && python -m uvicorn app:app --reload --port 5001")
    print("3. 在浏览器中访问: http://localhost:5001\n")

if __name__ == "__main__":
    main()