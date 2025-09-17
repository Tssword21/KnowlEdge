#!/usr/bin/env python
"""
快速修复管理员后台数据库问题
只处理必要的列添加，确保管理员后台正常工作
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

from src.config import Config

def fix_users_table(cursor):
    """修复users表"""
    try:
        # 检查updated_at列是否存在
        cursor.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'updated_at' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            print("✅ 已添加 users.updated_at 列")
        else:
            print("✓ users.updated_at 列已存在")
            
    except Exception as e:
        print(f"❌ 修复users表失败: {e}")

def fix_user_interests_table(cursor):
    """修复user_interests表"""
    try:
        # 检查reason列是否存在
        cursor.execute("PRAGMA table_info(user_interests)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'reason' not in columns:
            cursor.execute("ALTER TABLE user_interests ADD COLUMN reason TEXT")
            print("✅ 已添加 user_interests.reason 列")
        else:
            print("✓ user_interests.reason 列已存在")
            
    except Exception as e:
        print(f"❌ 修复user_interests表失败: {e}")

def fix_user_auth_table(cursor):
    """修复user_auth表"""
    try:
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
        
        # 检查is_admin列是否存在
        cursor.execute("PRAGMA table_info(user_auth)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'is_admin' not in columns:
            cursor.execute("ALTER TABLE user_auth ADD COLUMN is_admin INTEGER DEFAULT 0")
            print("✅ 已添加 user_auth.is_admin 列")
        else:
            print("✓ user_auth.is_admin 列已存在")
            
    except Exception as e:
        print(f"❌ 修复user_auth表失败: {e}")

def main():
    """主函数"""
    print("🔧 快速修复管理员后台数据库问题")
    print("=" * 40)
    
    config = Config()
    db_path = config.user_db_path
    
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        print("请先运行: python scripts/init_system.py")
        return
    
    try:
        # 备份数据库
        backup_path = f"{db_path}.backup_admin_fix"
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ 数据库已备份到: {backup_path}")
        
        # 修复数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\n🔍 检查和修复表结构...")
        fix_users_table(cursor)
        fix_user_interests_table(cursor) 
        fix_user_auth_table(cursor)
        
        # 提交更改
        conn.commit()
        conn.close()
        
        print(f"\n🎉 数据库修复完成!")
        print("现在管理员后台应该可以正常工作了")
        print("\n📝 接下来的步骤:")
        print("1. 启动应用: cd src && python -m uvicorn app:app --reload --port 5001")
        print("2. 访问管理后台: http://localhost:5001/admin")
        
    except Exception as e:
        print(f"\n❌ 修复失败: {e}")
        print("如有问题，可以从备份恢复数据库")

if __name__ == "__main__":
    main() 