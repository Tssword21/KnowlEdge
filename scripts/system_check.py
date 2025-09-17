#!/usr/bin/env python
"""
系统诊断脚本
检查KnowlEdge系统的状态和配置
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
from src.utils import get_user_data_path

def check_database():
    """检查数据库状态"""
    print("🔍 检查数据库状态...")
    config = Config()
    db_path = config.user_db_path
    
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'users', 'user_auth', 'user_profiles', 'search_history',
            'user_interests', 'user_skills', 'user_education', 
            'user_work_experience', 'user_interactions', 'interest_categories'
        ]
        
        missing_tables = [t for t in expected_tables if t not in tables]
        if missing_tables:
            print(f"⚠️ 缺少表: {', '.join(missing_tables)}")
        else:
            print("✅ 所有必要的表都存在")
        
        # 检查users表结构
        cursor.execute("PRAGMA table_info(users)")
        user_columns = [row[1] for row in cursor.fetchall()]
        print(f"📋 users表列: {', '.join(user_columns)}")
        
        if 'updated_at' not in user_columns:
            print("⚠️ users表缺少updated_at列，建议运行数据库迁移")
        
        # 统计数据
        for table in expected_tables:
            if table in tables:
                try:
                    count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    print(f"📊 {table}: {count} 条记录")
                except Exception as e:
                    print(f"❌ 无法查询表 {table}: {e}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据库检查失败: {e}")
        return False

def check_config():
    """检查配置"""
    print("\n🔍 检查系统配置...")
    
    config = Config()
    
    # 检查API密钥
    if config.llm_api_key:
        print("✅ DEEPSEEK_API_KEY 已配置")
    else:
        print("❌ DEEPSEEK_API_KEY 未配置")
    
    if config.serper_api_key:
        print("✅ SERPER_API_KEY 已配置")
    else:
        print("⚠️ SERPER_API_KEY 未配置（Google搜索功能将不可用）")
    
    # 检查数据目录
    data_dir = get_user_data_path()
    if os.path.exists(data_dir):
        print(f"✅ 数据目录存在: {data_dir}")
        
        # 检查兴趣分类文件
        interest_file = os.path.join(data_dir, "interest_categories.json")
        if os.path.exists(interest_file):
            print("✅ 兴趣分类文件存在")
        else:
            print("⚠️ 兴趣分类文件不存在")
            
        # 检查日志目录
        log_dir = os.path.join(data_dir, "logs")
        if os.path.exists(log_dir):
            print("✅ 日志目录存在")
        else:
            print("⚠️ 日志目录不存在")
    else:
        print(f"❌ 数据目录不存在: {data_dir}")

def check_dependencies():
    """检查依赖"""
    print("\n🔍 检查Python依赖...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'aiohttp', 'requests', 
        'bcrypt', 'python-dotenv', 'jinja2', 'arxiv'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 未安装")

def suggest_fixes():
    """提供修复建议"""
    print("\n💡 修复建议:")
    print("1. 如果数据库有问题，运行: python scripts/migrate_database.py")
    print("2. 如果缺少配置，运行: python scripts/init_system.py") 
    print("3. 如果需要创建管理员，运行: python scripts/create_admin.py")
    print("4. 如果依赖缺失，运行: pip install -r src/requirements.txt")

def main():
    """主函数"""
    print("🔧 KnowlEdge系统诊断工具")
    print("=" * 50)
    
    all_ok = True
    
    # 检查各个组件
    if not check_database():
        all_ok = False
    
    check_config()
    check_dependencies()
    
    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 系统状态良好!")
    else:
        print("⚠️ 发现一些问题，请查看上方详情")
        suggest_fixes()

if __name__ == "__main__":
    main() 