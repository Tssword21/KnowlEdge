#!/usr/bin/env python
"""
数据库清理脚本
清理超过指定天数的旧数据
"""
import sqlite3
import os
import sys
import datetime

# 添加项目根目录到Python路径
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
sys.path.append(os.path.join(parent_path, "src"))

from src.config import Config

def clean_old_data(days=90):
    """清理超过指定天数的旧数据"""
    config = Config()
    db_path = config.user_db_path
    
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return False
    
    # 计算截止日期
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_path)
    
    try:
        # 清理旧的搜索记录
        result = conn.execute(
            "DELETE FROM search_history WHERE timestamp < ?",
            (cutoff_date,)
        )
        searches_deleted = result.rowcount
        
        # 清理旧的交互记录
        result = conn.execute(
            "DELETE FROM user_interactions WHERE timestamp < ?",
            (cutoff_date,)
        )
        interactions_deleted = result.rowcount
        
        # 提交更改
        conn.commit()
        
        print(f"数据清理完成:")
        print(f"- 删除了 {searches_deleted} 条旧搜索记录")
        print(f"- 删除了 {interactions_deleted} 条旧交互记录")
        
        return True
    except sqlite3.Error as e:
        print(f"数据清理失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    days = input("请输入要清理的天数(默认90天，即删除超过90天及之前的数据): ").strip()
    days = int(days) if days.isdigit() else 90
    clean_old_data(days)