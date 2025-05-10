# verify_profile.py
import os
import sqlite3
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 系统配置
CONFIG = {
    "DATA_DIR": "./user_data",
    "DB_PATH": "./user_data/user_profiles.db",
}

def get_db_connection():
    """获取数据库连接"""
    if not os.path.exists(CONFIG["DB_PATH"]):
        print(f"错误: 数据库文件不存在: {CONFIG['DB_PATH']}")
        return None
    
    conn = sqlite3.connect(CONFIG["DB_PATH"])
    conn.row_factory = sqlite3.Row
    return conn

def check_database_tables():
    """检查数据库表是否存在并包含数据"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        # 检查表是否存在
        tables = ["users", "user_interests", "user_searches", "user_interactions", "user_skills"]
        all_tables_exist = True
        
        for table in tables:
            cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if not cursor.fetchone():
                print(f"表 '{table}' 不存在")
                all_tables_exist = False
        
        if not all_tables_exist:
            return False
        
        # 检查用户数据
        user_count = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()["count"]
        print(f"数据库中有 {user_count} 个用户")
        
        if user_count > 0:
            # 获取所有用户
            users = conn.execute("SELECT id, name, occupation, email FROM users").fetchall()
            for user in users:
                print(f"\n用户ID: {user['id']}")
                print(f"姓名: {user['name']}")
                print(f"职业: {user['occupation']}")
                print(f"邮箱: {user['email']}")
                
                # 检查用户兴趣
                interests = conn.execute(
                    "SELECT topic, category, weight FROM user_interests WHERE user_id = ? ORDER BY weight DESC LIMIT 5", 
                    (user['id'],)
                ).fetchall()
                
                print(f"\n用户兴趣 (前5项):")
                if interests:
                    for interest in interests:
                        print(f"  - {interest['topic']} ({interest['category']}): {interest['weight']:.2f}")
                else:
                    print("  未找到兴趣数据")
                
                # 检查用户技能
                skills = conn.execute(
                    "SELECT skill, level, category FROM user_skills WHERE user_id = ? ORDER BY level DESC LIMIT 5", 
                    (user['id'],)
                ).fetchall()
                
                print(f"\n用户技能 (前5项):")
                if skills:
                    for skill in skills:
                        print(f"  - {skill['skill']} ({skill['category']}): {skill['level']}")
                else:
                    print("  未找到技能数据")
                
                # 检查搜索历史
                searches = conn.execute(
                    "SELECT query, platform, timestamp FROM user_searches WHERE user_id = ? ORDER BY timestamp DESC LIMIT 3", 
                    (user['id'],)
                ).fetchall()
                
                print(f"\n最近搜索 (前3项):")
                if searches:
                    for search in searches:
                        print(f"  - {search['query']} ({search['platform']}): {search['timestamp']}")
                else:
                    print("  未找到搜索历史")
                
                print("-" * 50)
            
            return True
        else:
            print("数据库中没有用户数据")
            return False
        
    except Exception as e:
        print(f"检查数据库时出错: {e}")
        return False
    finally:
        conn.close()

def check_interest_categories():
    """检查兴趣分类文件是否存在"""
    categories_file = os.path.join(CONFIG["DATA_DIR"], "interest_categories.json")
    
    if os.path.exists(categories_file):
        try:
            with open(categories_file, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            
            print(f"\n兴趣分类系统:")
            for category, topics in categories.items():
                print(f"  - {category}: {len(topics)} 个主题")
            
            return True
        except Exception as e:
            print(f"读取兴趣分类文件时出错: {e}")
            return False
    else:
        print(f"兴趣分类文件不存在: {categories_file}")
        return False

def main():
    """验证用户画像系统"""
    print("\n===== 用户画像系统验证 =====")
    
    # 检查数据目录
    if not os.path.exists(CONFIG["DATA_DIR"]):
        print(f"错误: 数据目录不存在: {CONFIG['DATA_DIR']}")
        print("请先运行初始化脚本 init_system.py")
        return False
    
    # 检查数据库
    print("\n检查数据库...")
    db_result = check_database_tables()
    
    # 检查兴趣分类
    print("\n检查兴趣分类系统...")
    categories_result = check_interest_categories()
    
    # 总结
    print("\n===== 验证结果 =====")
    if db_result and categories_result:
        print("用户画像系统验证通过！系统已正确设置并包含用户数据。")
        return True
    else:
        print("用户画像系统验证失败。请检查上述错误并修复问题。")
        return False

if __name__ == "__main__":
    main()
