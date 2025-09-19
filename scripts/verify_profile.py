#!/usr/bin/env python
"""
用户画像验证脚本
验证用户画像系统的核心组件是否配置正确并包含预期数据
"""
import os
import sys
import sqlite3
import json
import logging

# 添加项目根目录到Python路径
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
sys.path.append(os.path.join(parent_path, "src"))

from src.config import Config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """获取数据库连接"""
    config = Config()
    db_path = config.user_db_path
    
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def check_database_tables():
    """检查数据库表是否存在并显示用户数据"""
    print("🔍 检查数据库表结构...")
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # 检查所有必要的表
        required_tables = [
            'users', 'user_profiles', 'search_history', 'user_interests', 
            'user_skills', 'user_education', 'user_work_experience', 
            'user_interactions', 'interest_categories', 'user_auth'
        ]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        print(f"现有表: {', '.join(existing_tables)}")
        
        missing_tables = [table for table in required_tables if table not in existing_tables]
        if missing_tables:
            print(f"⚠️ 缺少表: {', '.join(missing_tables)}")
        else:
            print("✅ 所有必要的表都存在")
        
        # 检查用户数据
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"\n📊 用户总数: {user_count}")
        
        if user_count > 0:
            print("\n👥 用户信息:")
            cursor.execute("SELECT id, name, occupation, email, created_at FROM users LIMIT 10")
            users = cursor.fetchall()
            
            for user in users:
                print(f"\n用户 ID: {user['id']}")
                print(f"  姓名: {user['name']}")
                print(f"  职业: {user['occupation']}")
                print(f"  邮箱: {user['email']}")
                print(f"  创建时间: {user['created_at']}")
                
                # 显示用户兴趣
                cursor.execute(
                    "SELECT topic, category, weight FROM user_interests WHERE user_id=? ORDER BY weight DESC LIMIT 5",
                    (user['id'],)
                )
                interests = cursor.fetchall()
                if interests:
                    print("  前5项兴趣:")
                    for interest in interests:
                        print(f"    - {interest['topic']} ({interest['category']}) - 权重: {interest['weight']}")
                
                # 显示用户技能
                cursor.execute(
                    "SELECT skill, level, category FROM user_skills WHERE user_id=? LIMIT 5",
                    (user['id'],)
                )
                skills = cursor.fetchall()
                if skills:
                    print("  前5项技能:")
                    for skill in skills:
                        print(f"    - {skill['skill']} ({skill['level']}) - 类别: {skill['category']}")
                
                # 显示最近搜索历史（修复表名）
                cursor.execute(
                    "SELECT query, platform, timestamp FROM search_history WHERE user_id=? ORDER BY timestamp DESC LIMIT 3",
                    (user['id'],)
                )
                searches = cursor.fetchall()
                if searches:
                    print("  最近3次搜索:")
                    for search in searches:
                        print(f"    - {search['query']} ({search['platform']}) - {search['timestamp']}")
        
        return True
        
    except Exception as e:
        print(f"检查数据库时出错: {e}")
        return False
    finally:
        conn.close()

def check_interest_categories():
    """检查兴趣分类文件"""
    print("\n🏷️ 检查兴趣分类文件...")
    config = Config()
    categories_file = os.path.join(config.data_dir, "interest_categories.json")
    
    if not os.path.exists(categories_file):
        print(f"⚠️ 兴趣分类文件不存在: {categories_file}")
        return False
    
    try:
        with open(categories_file, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        
        print("✅ 兴趣分类文件存在")
        print("分类概览:")
        for category, subcategories in categories.items():
            print(f"  {category}: {len(subcategories)}个子主题")
        
        return True
        
    except Exception as e:
        print(f"读取兴趣分类文件时出错: {e}")
        return False

def check_system_health():
    """检查系统整体健康状况"""
    print("\n🏥 系统健康检查...")
    config = Config()
    
    # 检查数据目录
    if not os.path.exists(config.data_dir):
        print(f"❌ 数据目录不存在: {config.data_dir}")
        return False
    
    print(f"✅ 数据目录存在: {config.data_dir}")
    
    # 检查配置有效性
    validation = config.validate_config()
    if validation["valid"]:
        print("✅ 基本配置有效")
    else:
        print(f"⚠️ 配置问题: 缺少 {', '.join(validation['missing_keys'])}")
    
    if validation["warnings"]["serper_api_key"]:
        print("⚠️ SERPER_API_KEY 未设置，Google搜索功能将不可用")
    
    return True

def main():
    """主函数"""
    print("🔍 KnowlEdge 系统验证")
    print("=" * 50)
    
    # 检查数据库
    db_ok = check_database_tables()
    
    # 检查兴趣分类
    categories_ok = check_interest_categories()
    
    # 系统健康检查
    system_ok = check_system_health()
    
    print("\n" + "=" * 50)
    if db_ok and categories_ok and system_ok:
        print("✅ 系统验证通过！")
    else:
        print("⚠️ 系统存在问题，请检查上述输出")

if __name__ == "__main__":
    main()
