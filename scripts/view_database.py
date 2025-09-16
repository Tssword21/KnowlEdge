# view_database.py
import sqlite3
import os
import sys
import time

# 配置
DB_PATH = "./user_data/user_profiles.db"

def check_database():
    """检查数据库是否存在"""
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库文件不存在: {DB_PATH}")
        return False
    return True

def view_all_users():
    """查看所有用户"""
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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
    
    # 查看搜索历史（修正表名）
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    tables = ["users", "user_skills", "user_interests", "search_history", "user_interactions"]
    
    print("\n数据库统计信息:")
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) as count FROM {table}").fetchone()["count"]
        print(f"表 {table}: {count} 条记录")
    
    # 查看数据库文件大小
    size_bytes = os.path.getsize(DB_PATH)
    size_kb = size_bytes / 1024
    print(f"\n数据库文件大小: {size_kb:.2f} KB")
    
    conn.close()

def export_user_data(user_id, output_file=None):
    """导出用户数据到文件"""
    if not output_file:
        output_file = f"user_{user_id}_{int(time.time())}.txt"
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    
    if not user:
        print(f"未找到ID为 {user_id} 的用户")
        return
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"用户ID: {user['id']}\n")
        f.write(f"姓名: {user['name']}\n")
        f.write(f"职业: {user['occupation']}\n")
        f.write(f"邮箱: {user['email']}\n")
        f.write(f"创建时间: {user['created_at']}\n\n")
        
        # 导出用户技能
        skills = conn.execute(
            "SELECT skill, level, category FROM user_skills WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        
        f.write(f"技能 ({len(skills)}):\n")
        for skill in skills:
            f.write(f"  - {skill['skill']} ({skill['category']}): {skill['level']}\n")
        
        # 导出用户兴趣
        interests = conn.execute(
            "SELECT topic, category, weight FROM user_interests WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        
        f.write(f"\n兴趣 ({len(interests)}):\n")
        for interest in interests:
            f.write(f"  - {interest['topic']} ({interest['category']}): {interest['weight']}\n")
        
        # 导出搜索历史（修正表名）
        searches = conn.execute(
            "SELECT query, platform, timestamp FROM search_history WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,)
        ).fetchall()
        
        f.write(f"\n搜索历史 ({len(searches)}):\n")
        for search in searches:
            f.write(f"  - {search['query']} ({search['platform']}): {search['timestamp']}\n")
    
    print(f"用户数据已导出到文件: {output_file}")
    conn.close()

def main():
    """主函数"""
    if not check_database():
        return
    
    print("\n===== 数据库查看工具 =====")
    print("1. 查看所有用户")
    print("2. 查看特定用户")
    print("3. 查看数据库统计信息")
    print("4. 导出用户数据")
    print("0. 退出")
    
    choice = input("\n请选择操作: ").strip()
    
    if choice == "1":
        view_all_users()
    elif choice == "2":
        user_id = input("请输入用户ID: ").strip()
        view_user_by_id(user_id)
    elif choice == "3":
        view_database_stats()
    elif choice == "4":
        user_id = input("请输入要导出的用户ID: ").strip()
        output_file = input("请输入输出文件名(留空使用默认): ").strip()
        export_user_data(user_id, output_file if output_file else None)
    elif choice == "0":
        print("退出程序")
        return
    else:
        print("无效的选择")
    
    # 询问是否继续
    if input("\n是否继续? (y/n): ").strip().lower() == 'y':
        main()

if __name__ == "__main__":
    main()