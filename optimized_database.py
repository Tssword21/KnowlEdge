# optimize_database.py
import sqlite3
import os

DB_PATH = "./user_data/user_profiles.db"

def optimize_database():
    """优化数据库"""
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库文件不存在: {DB_PATH}")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 执行VACUUM操作，重新组织数据库文件，减少碎片
        conn.execute("VACUUM")
        
        # 执行ANALYZE操作，更新统计信息
        conn.execute("ANALYZE")
        
        # 执行PRAGMA操作，检查完整性
        integrity_check = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity_check == "ok":
            print("数据库完整性检查通过")
        else:
            print(f"数据库完整性检查失败: {integrity_check}")
        
        print("数据库优化完成")
        return True
    except sqlite3.Error as e:
        print(f"数据库优化失败: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    optimize_database()