# db_utils.py
import os
import sqlite3
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    "DATA_DIR": "./user_data",
    "DB_PATH": "./user_data/user_profiles.db"
}

def get_db_connection():
    """获取数据库连接，如果数据库不存在则创建并初始化表结构"""
    db_path = DB_CONFIG["DB_PATH"]
    
    # 确保数据目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        # 检查数据库文件是否存在
        db_exists = os.path.exists(db_path)
        
        # 创建连接
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # 如果数据库文件不存在，初始化表结构
        if not db_exists:
            logger.info(f"数据库文件不存在，正在创建: {db_path}")
            
            # 创建表
            conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                occupation TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.execute('''
            CREATE TABLE IF NOT EXISTS user_interests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic TEXT,
                category TEXT,
                weight REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            conn.execute('''
            CREATE TABLE IF NOT EXISTS user_searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                query TEXT,
                platform TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            conn.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                content_id TEXT,
                action_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            conn.execute('''
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
            
            conn.commit()
            logger.info("数据库表结构已创建")
        
        return conn
    except sqlite3.Error as e:
        logger.error(f"数据库连接错误: {e}")
        return None

def verify_database():
    """验证数据库是否正确创建和可写入"""
    logger.info("验证数据库...")
    
    conn = get_db_connection()
    if conn is None:
        logger.error("无法连接到数据库，请检查路径和权限")
        return False
    
    try:
        # 尝试写入测试数据
        import time
        test_id = f"test_{int(time.time())}"
        conn.execute(
            "INSERT INTO users (id, name, occupation, email) VALUES (?, ?, ?, ?)",
            (test_id, "测试用户", "测试职业", "test@example.com")
        )
        conn.commit()
        
        # 验证是否写入成功
        user = conn.execute("SELECT * FROM users WHERE id = ?", (test_id,)).fetchone()
        if user:
            logger.info("数据库验证成功：可以正常写入和读取数据")
            
            # 清理测试数据
            conn.execute("DELETE FROM users WHERE id = ?", (test_id,))
            conn.commit()
            
            # 显示数据库信息
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            logger.info(f"数据库包含以下表: {', '.join([t['name'] for t in tables])}")
            
            for table in [t['name'] for t in tables]:
                count = conn.execute(f"SELECT COUNT(*) as count FROM {table}").fetchone()['count']
                logger.info(f"表 {table}: {count} 条记录")
            
            return True
        else:
            logger.error("数据库验证失败：无法读取写入的测试数据")
            return False
    except sqlite3.Error as e:
        logger.error(f"数据库验证失败: {e}")
        return False
    finally:
        conn.close()

def initialize_database():
    """初始化数据库，创建必要的表结构"""
    conn = get_db_connection()
    if conn:
        conn.close()
        logger.info("数据库初始化完成")
        return True
    else:
        logger.error("数据库初始化失败")
        return False