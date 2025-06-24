"""
KnowlEdge项目工具模块
包含通用工具函数和类
"""
import os
import json
import logging
import sqlite3
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config import tokenizer, model, Config

# 初始化配置
config = Config()

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    logging.info("日志系统初始化完成")

def get_user_data_path():
    """获取用户数据目录路径"""
    config = Config()
    return config.data_dir

def verify_database():
    """验证数据库结构是否正确"""
    config = Config()
    db_path = config.user_db_path
    
    if not os.path.exists(db_path):
        logging.warning(f"数据库文件不存在: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查用户画像表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profiles'")
        if not cursor.fetchone():
            logging.warning("数据库缺少user_profiles表")
            conn.close()
            return False
            
        # 检查搜索历史表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_history'")
        if not cursor.fetchone():
            logging.warning("数据库缺少search_history表")
            conn.close()
            return False
        
        conn.close()
        return True
    except Exception as e:
        logging.error(f"验证数据库时出错: {str(e)}")
        return False

class NumpyEncoder(json.JSONEncoder):
    """自定义的 JSON 编码器，用于处理 NumPy 数据类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_bert_embeddings(text: str) -> torch.Tensor:
    """获取文本的BERT嵌入表示"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def compute_similarity(text1: str, text2: str) -> float:
    """计算两个文本之间的余弦相似度"""
    embedding1 = get_bert_embeddings(text1)
    embedding2 = get_bert_embeddings(text2)
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]

def collect_user_input() -> dict:
    """收集用户输入信息"""
    print("\n=== 欢迎使用 KnowlEdge 个性化知识引擎 ===")
    print("请输入以下信息，帮助我们为您提供更精准的内容推荐")
    
    # 收集用户信息
    user_name = input("\n您的姓名: ")
    occupation = input("您的职业: ")
    
    # 收集搜索偏好
    print("\n时间范围偏好（最近几天的内容）:")
    day_options = [("最近一周", 7), ("最近一个月", 30), ("最近三个月", 90), ("最近半年", 180)]
    for i, (label, _) in enumerate(day_options, 1):
        print(f"{i}. {label}")
    day_choice = int(input("请选择 (1-4): "))
    day = day_options[day_choice-1][1]
    
    # 收集平台偏好
    print("\n内容平台偏好:")
    platform_options = ["学术期刊", "科技博客", "新闻媒体", "社交平台", "视频网站"]
    for i, platform in enumerate(platform_options, 1):
        print(f"{i}. {platform}")
    platform_choice = int(input("请选择 (1-5): "))
    platform = platform_options[platform_choice-1]
    
    # 收集内容偏好
    content_type = input("\n您感兴趣的主题/领域: ")
    
    # 可选联系方式
    email = input("\n电子邮箱 (用于接收报告，可选): ")
    
    return {
        "user_name": user_name,
        "occupation": occupation,
        "day": day,
        "platform": platform,
        "content_type": content_type,
        "email": email
    } 