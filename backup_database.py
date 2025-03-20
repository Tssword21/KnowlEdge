# backup_database.py
import os
import shutil
import datetime

DB_PATH = "./user_data/user_profiles.db"
BACKUP_DIR = "./user_data/backups"

def backup_database():
    """备份数据库文件"""
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库文件不存在: {DB_PATH}")
        return False
    
    # 创建备份目录
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # 生成备份文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"user_profiles_{timestamp}.db")
    
    # 复制数据库文件
    shutil.copy2(DB_PATH, backup_file)
    
    # 验证备份
    if os.path.exists(backup_file):
        print(f"数据库已成功备份到: {backup_file}")
        return True
    else:
        print("备份失败")
        return False

if __name__ == "__main__":
    backup_database()