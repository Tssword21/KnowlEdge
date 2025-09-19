#!/usr/bin/env python
"""
数据库查看工具
提供命令行界面查看和导出数据库中的用户数据
"""
import sqlite3
import os
import sys
import time

# 添加项目根目录到Python路径
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
sys.path.append(os.path.join(parent_path, "src"))

from src.config import Config

def check_database():
    """检查数据库是否存在"""
    config = Config()
    db_path = config.user_db_path
    
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return False
    return True

def view_all_users():
    """查看所有用户"""
    config = Config()
    db_path = config.user_db_path
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    users = conn.execute("SELECT * FROM users").fetchall()
    print(f"共有 {len(users)} 个用户:")
    
    for user in users:
        print(f"\n用户ID: {user['id']}")
        print(f"姓名: {user['name']}")
        print(f"职业: {user['occupation']}")
        print(f"邮箱: {user['email']}")
        print(f"创建时间: {user['created_at']}")
        
        # 查看用户技能
        skills = conn.execute(
            "SELECT skill, level, category FROM user_skills WHERE user_id = ?",
            (user['id'],)
        ).fetchall()
        
        print(f"\n技能 ({len(skills)}):")
        for skill in skills:
            print(f"  - {skill['skill']} ({skill['category']}): {skill['level']}")
        
        # 查看用户兴趣
        interests = conn.execute(
            "SELECT topic, category, weight FROM user_interests WHERE user_id = ?",
            (user['id'],)
        ).fetchall()
        
        print(f"\n兴趣 ({len(interests)}):")
        for interest in interests:
            print(f"  - {interest['topic']} ({interest['category']}): {interest['weight']}")
        
        print("-" * 50)
    
    conn.close()

def view_user_by_id(user_id):
    """查看特定用户的详细信息"""
    config = Config()
    db_path = config.user_db_path
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    
    if not user:
        print(f"未找到ID为 {user_id} 的用户")
        return
    
    print(f"\n用户详情:")
    print(f"ID: {user['id']}")
    print(f"姓名: {user['name']}")
    print(f"职业: {user['occupation']}")
    print(f"邮箱: {user['email']}")
    print(f"创建时间: {user['created_at']}")
    
    # 查看用户技能
    skills = conn.execute(
        "SELECT skill, level, category FROM user_skills WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    
    print(f"\n技能 ({len(skills)}):")
    for skill in skills:
        print(f"  - {skill['skill']} ({skill['category']}): {skill['level']}")
    
    # 查看用户兴趣
    interests = conn.execute(
        "SELECT topic, category, weight FROM user_interests WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    
    print(f"\n兴趣 ({len(interests)}):")
    for interest in interests:
        print(f"  - {interest['topic']} ({interest['category']}): {interest['weight']}")
    
    # 查看搜索历史
    searches = conn.execute(
        "SELECT query, platform, timestamp FROM search_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5",
        (user_id,)
    ).fetchall()
    
    print(f"\n最近搜索 ({len(searches)}):")
    for search in searches:
        print(f"  - {search['query']} ({search['platform']}): {search['timestamp']}")
    
    conn.close()

def view_database_stats():
    """查看数据库统计信息"""
    config = Config()
    db_path = config.user_db_path
    
    conn = sqlite3.connect(db_path)
    
    tables = ["users", "user_interests", "search_history", "user_interactions", "user_skills"]
    
    print("数据库统计信息:")
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count} 条记录")
        except sqlite3.OperationalError:
            print(f"  {table}: 表不存在")
    
    # 数据库文件大小
    file_size = os.path.getsize(db_path) / 1024  # KB
    print(f"\n数据库文件大小: {file_size:.2f} KB")
    
    conn.close()

def export_user_data(user_id, output_file):
    """导出特定用户的数据到文件"""
    config = Config()
    db_path = config.user_db_path
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    
    if not user:
        print(f"未找到ID为 {user_id} 的用户")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"用户数据导出 - {user['name']}\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"基本信息:\n")
        f.write(f"ID: {user['id']}\n")
        f.write(f"姓名: {user['name']}\n")
        f.write(f"职业: {user['occupation']}\n")
        f.write(f"邮箱: {user['email']}\n")
        f.write(f"创建时间: {user['created_at']}\n\n")
        
        # 导出技能
        skills = conn.execute(
            "SELECT skill, level, category FROM user_skills WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        
        f.write(f"技能 ({len(skills)}):\n")
        for skill in skills:
            f.write(f"  - {skill['skill']} ({skill['category']}): {skill['level']}\n")
        f.write("\n")
        
        # 导出兴趣
        interests = conn.execute(
            "SELECT topic, category, weight FROM user_interests WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        
        f.write(f"兴趣 ({len(interests)}):\n")
        for interest in interests:
            f.write(f"  - {interest['topic']} ({interest['category']}): {interest['weight']}\n")
        f.write("\n")
        
        # 导出搜索历史
        searches = conn.execute(
            "SELECT query, platform, timestamp FROM search_history WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,)
        ).fetchall()
        
        f.write(f"搜索历史 ({len(searches)}):\n")
        for search in searches:
            f.write(f"  - {search['query']} ({search['platform']}): {search['timestamp']}\n")
    
    print(f"用户数据已导出到: {output_file}")
    conn.close()

def main():
    """主菜单"""
    if not check_database():
        return
    
    while True:
        print("\n数据库查看工具")
        print("=" * 30)
        print("1. 查看所有用户")
        print("2. 查看特定用户")
        print("3. 查看数据库统计")
        print("4. 导出用户数据")
        print("5. 退出")
        
        choice = input("\n请选择操作 (1-5): ").strip()
        
        if choice == "1":
            view_all_users()
        elif choice == "2":
            user_id = input("请输入用户ID: ").strip()
            if user_id:
                view_user_by_id(user_id)
        elif choice == "3":
            view_database_stats()
        elif choice == "4":
            user_id = input("请输入用户ID: ").strip()
            output_file = input("请输入输出文件名: ").strip()
            if user_id and output_file:
                export_user_data(user_id, output_file)
        elif choice == "5":
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()