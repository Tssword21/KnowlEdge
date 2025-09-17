#!/usr/bin/env python
"""
创建管理员账户脚本
用于快速创建管理员账户进行测试
"""
import os
import sys
import bcrypt

# 添加项目根目录到Python路径
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
sys.path.append(os.path.join(parent_path, "src"))

from src.db_utils import get_db_connection

def create_admin_user(username="admin", password="admin123", email="admin@example.com"):
    """创建管理员用户"""
    conn = get_db_connection()
    if not conn:
        print("❌ 无法连接到数据库")
        return False
    
    try:
        # 检查用户是否已存在
        existing = conn.execute("SELECT username FROM user_auth WHERE username=?", (username,)).fetchone()
        if existing:
            print(f"⚠️ 用户 {username} 已存在")
            # 更新为管理员
            conn.execute("UPDATE user_auth SET is_admin=1 WHERE username=?", (username,))
            conn.commit()
            print(f"✅ 已将 {username} 设置为管理员")
            return True
        
        # 创建基础用户记录
        user_id = username.lower()
        conn.execute(
            "INSERT OR IGNORE INTO users (id, name, occupation, email) VALUES (?, ?, ?, ?)",
            (user_id, username, "系统管理员", email)
        )
        
        # 创建认证记录
        pwd_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn.execute(
            "INSERT INTO user_auth (user_id, username, email, password_hash, is_admin) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, email, pwd_hash, 1)
        )
        
        conn.commit()
        print(f"✅ 管理员账户创建成功")
        print(f"   用户名: {username}")
        print(f"   密码: {password}")
        print(f"   邮箱: {email}")
        return True
        
    except Exception as e:
        print(f"❌ 创建管理员账户失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    print("\n===== 创建管理员账户 =====\n")
    
    username = input("输入管理员用户名 (默认: admin): ").strip() or "admin"
    password = input("输入密码 (默认: admin123): ").strip() or "admin123"
    email = input("输入邮箱 (默认: admin@example.com): ").strip() or "admin@example.com"
    
    if create_admin_user(username, password, email):
        print(f"\n🎉 管理员账户设置完成!")
        print(f"现在可以使用以下信息登录管理后台:")
        print(f"  1. 启动应用: cd src && python -m uvicorn app:app --reload --port 5001")
        print(f"  2. 访问: http://localhost:5001/auth")
        print(f"  3. 使用 {username}/{password} 登录")
        print(f"  4. 登录后访问: http://localhost:5001/admin")
    else:
        print("\n❌ 创建失败，请检查数据库配置")

if __name__ == "__main__":
    main() 