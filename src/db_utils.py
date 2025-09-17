# db_utils.py
import os
import sqlite3
import logging
import time
from src.config import Config

# 初始化配置
config = Config()

logger = logging.getLogger(__name__)
if not logger.handlers:
    # 遵循全局日志配置，仅微调本模块级别
    logger.setLevel(logging.INFO)

def get_db_connection():
    """获取数据库连接，如果数据库不存在则创建并初始化表结构"""
    db_path = config.user_db_path
    
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
            
            # 创建用户表
            conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                occupation TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 认证表
            conn.execute('''
            CREATE TABLE IF NOT EXISTS user_auth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                username TEXT UNIQUE,
                email TEXT,
                password_hash TEXT,
                is_admin INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # 创建用户画像表（向后兼容）
            conn.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                profile_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 创建搜索历史表
            conn.execute('''
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
            conn.execute('''
            CREATE TABLE IF NOT EXISTS interest_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                subcategories TEXT
            )
            ''')
            
            # 创建用户兴趣表
            conn.execute('''
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
            
            # 创建用户教育背景表
            conn.execute('''
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
            conn.execute('''
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
            
            conn.commit()
            logger.info("数据库表结构已创建")
        else:
            # 检查必要的表是否存在，如果不存在则创建
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [table['name'] for table in tables]
            
            if 'users' not in table_names:
                logger.warning("数据库缺少users表")
                conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    occupation TEXT,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
            
            if 'user_auth' not in table_names:
                logger.warning("数据库缺少user_auth表")
                conn.execute('''
                CREATE TABLE IF NOT EXISTS user_auth (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    username TEXT UNIQUE,
                    email TEXT,
                    password_hash TEXT,
                    is_admin INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                ''')
            # 确保 user_auth 存在 is_admin 列
            try:
                cols = [r['name'] for r in conn.execute("PRAGMA table_info(user_auth)").fetchall()]
                if 'is_admin' not in cols:
                    conn.execute("ALTER TABLE user_auth ADD COLUMN is_admin INTEGER DEFAULT 0")
                    logger.info("已为 user_auth 表增加 is_admin 列")
            except Exception as e:
                logger.debug(f"检查/添加 is_admin 列失败: {e}")
                
            # 补建 user_profiles 表
            if 'user_profiles' not in table_names:
                logger.warning("数据库缺少user_profiles表")
                conn.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    username TEXT,
                    profile_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
            if 'search_history' not in table_names:
                logger.warning("数据库缺少search_history表")
                conn.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    platform TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
                ''')
                
            # 补建 interest_categories 表
            if 'interest_categories' not in table_names:
                logger.warning("数据库缺少interest_categories表")
                conn.execute('''
                CREATE TABLE IF NOT EXISTS interest_categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    subcategories TEXT
                )
                ''')
                
            if 'user_interests' not in table_names:
                logger.warning("数据库缺少user_interests表")
                conn.execute('''
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
                
            if 'user_skills' not in table_names:
                logger.warning("数据库缺少user_skills表")
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
                
            if 'user_education' not in table_names:
                logger.warning("数据库缺少user_education表")
                conn.execute('''
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
                
            if 'user_work_experience' not in table_names:
                logger.warning("数据库缺少user_work_experience表")
                conn.execute('''
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
                
            if 'user_interactions' not in table_names:
                logger.warning("数据库缺少user_interactions表")
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
                
            conn.commit()
        
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