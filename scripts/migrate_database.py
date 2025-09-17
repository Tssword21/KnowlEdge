#!/usr/bin/env python
"""
数据库迁移脚本
用于修复和升级数据库schema
"""
import os
import sys
import sqlite3
import logging

# 添加项目根目录到Python路径
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
sys.path.append(os.path.join(parent_path, "src"))

from src.utils import setup_logging, get_user_data_path
from src.config import Config

# 设置日志
setup_logging()

def check_column_exists(cursor, table_name, column_name):
    """检查表中是否存在指定列"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return column_name in columns
    except Exception as e:
        logging.error(f"检查列 {table_name}.{column_name} 失败: {e}")
        return False

def add_column_if_missing(cursor, table_name, column_name, column_def):
    """如果列不存在则添加"""
    if not check_column_exists(cursor, table_name, column_name):
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")
            logging.info(f"✅ 已添加列: {table_name}.{column_name}")
            return True
        except Exception as e:
            logging.error(f"❌ 添加列失败 {table_name}.{column_name}: {e}")
            return False
    else:
        logging.info(f"✓ 列已存在: {table_name}.{column_name}")
        return True

def migrate_users_table(cursor):
    """迁移users表"""
    logging.info("检查和迁移 users 表...")
    
    # 添加缺失的列
    migrations = [
        ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ]
    
    for column_name, column_def in migrations:
        add_column_if_missing(cursor, "users", column_name, column_def)

def migrate_user_auth_table(cursor):
    """迁移user_auth表"""
    logging.info("检查和迁移 user_auth 表...")
    
    # 确保user_auth表存在
    cursor.execute('''
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
    
    # 添加缺失的列
    add_column_if_missing(cursor, "user_auth", "is_admin", "INTEGER DEFAULT 0")

def migrate_all_tables(cursor):
    """迁移所有表"""
    logging.info("开始数据库迁移...")
    
    # 确保所有必要的表都存在
    tables_to_create = [
        ("""user_profiles""", '''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            profile_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''),
        ("""search_history""", '''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query TEXT,
            platform TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        '''),
        ("""interest_categories""", '''
        CREATE TABLE IF NOT EXISTS interest_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            subcategories TEXT
        )
        '''),
        ("""user_interests""", '''
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
        '''),
        ("""user_skills""", '''
        CREATE TABLE IF NOT EXISTS user_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            skill TEXT,
            level TEXT,
            category TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        '''),
        ("""user_education""", '''
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
        '''),
        ("""user_work_experience""", '''
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
        '''),
        ("""user_interactions""", '''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            content_id TEXT,
            action_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        '''),
    ]
    
    for table_name, create_sql in tables_to_create:
        try:
            cursor.execute(create_sql)
            logging.info(f"✅ 表 {table_name} 已确保存在")
        except Exception as e:
            logging.error(f"❌ 创建表 {table_name} 失败: {e}")
    
    # 特定表的迁移
    migrate_users_table(cursor)
    migrate_user_auth_table(cursor)

def main():
    """主函数"""
    print("\n===== KnowlEdge数据库迁移 =====\n")
    
    # 获取数据库路径
    config = Config()
    db_path = config.user_db_path
    
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        print("请先运行 init_system.py 初始化数据库")
        return
    
    try:
        # 备份数据库
        backup_path = f"{db_path}.backup"
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ 数据库已备份到: {backup_path}")
        
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 执行迁移
        migrate_all_tables(cursor)
        
        # 提交更改
        conn.commit()
        
        # 验证迁移结果
        print("\n📊 迁移后的表结构:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"  {table_name}: {len(columns)} 列")
            for col in columns:
                print(f"    - {col[1]} ({col[2]})")
        
        conn.close()
        
        print(f"\n🎉 数据库迁移完成!")
        print(f"现在可以正常使用管理员后台功能了")
        
    except Exception as e:
        logging.error(f"数据库迁移失败: {e}")
        print(f"\n❌ 迁移失败: {e}")
        print("如有问题，可以从备份恢复数据库")

if __name__ == "__main__":
    main() 