import json
from db_utils import get_db_connection, verify_database
import datetime
import logging
import requests
from openai import OpenAI
from xml.etree import ElementTree
from deep_translator import GoogleTranslator, BaiduTranslator
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os
import sqlite3
import hashlib
from typing import Dict, Any, List, Tuple, Optional, AsyncIterator 
import PyPDF2
import docx
import pandas as pd
import pytesseract
from PIL import Image
import asyncio
import re
from dotenv import load_dotenv

load_dotenv()

# 配置日志
# logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("requests").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 常量配置
CONFIG = {
    "API_KEYS": {
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        "qwen": os.getenv("QWEN_API_KEY"),
        "serper": os.getenv("SERPER_API_KEY"),
        "baidu_translate": os.getenv("BAIDU_API_KEY"),
    },
    "MODELS": {
        "BERT": "bert-base-multilingual-cased",
        "LLM": "deepseek-chat",
    },
    "DATA_DIR": "./user_data",
    "DB_PATH": "./user_data/user_profiles.db"
}

# 确保数据目录存在
os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)

# 初始化BERT模型
tokenizer = BertTokenizer.from_pretrained(CONFIG["MODELS"]["BERT"])
model = BertModel.from_pretrained(CONFIG["MODELS"]["BERT"])

# 创建一个自定义的 NumpyEncoder 类来处理 NumPy 数据类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_bert_embeddings(text: str) -> torch.Tensor:
    """ 获取文本的BERT嵌入表示 """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def compute_similarity(text1: str, text2: str) -> float:
    """ 计算两个文本之间的余弦相似度 """
    embedding1 = get_bert_embeddings(text1)
    embedding2 = get_bert_embeddings(text2)
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]

class UserProfileManager:
    """用户画像管理器，负责创建、更新和存储用户画像"""

    def __init__(self, client=None):
        """初始化用户画像管理器"""
        self.client = client or OpenAI(
            api_key=CONFIG["API_KEYS"]["deepseek"],
            base_url="https://api.deepseek.com"
        )
        self.interest_categories = self._load_interest_categories()
        logging.info("用户画像管理器初始化完成")

    def _load_interest_categories(self):
        """加载预定义的兴趣分类体系"""
        categories_file = os.path.join(CONFIG["DATA_DIR"], "interest_categories.json")

        if os.path.exists(categories_file):
            try:
                with open(categories_file, 'r', encoding='utf-8') as f:
                    self.interest_categories = json.load(f)
                logging.info(f"已加载兴趣分类体系，共 {len(self.interest_categories)} 个类别")
            except Exception as e:
                logging.error(f"加载兴趣分类体系出错: {e}")
                self._create_default_categories()
        else:
            logging.warning(f"兴趣分类文件不存在: {categories_file}，将创建默认分类")
            self._create_default_categories()

    def _create_default_categories(self):
        """创建默认的兴趣分类体系"""
        self.interest_categories = {
            "技术": ["人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉", "大语言模型",
                     "大数据", "云计算", "区块链", "物联网", "网络安全", "数据库"],
            "科学": ["物理学", "化学", "生物学", "天文学", "数学", "医学", "地质学", "环境科学"],
            "商业": ["管理", "市场营销", "金融", "创业", "投资", "电子商务", "人力资源"],
            "艺术": ["绘画", "音乐", "电影", "文学", "设计", "摄影", "建筑"],
            "教育": ["教学方法", "学习理论", "教育技术", "高等教育", "职业教育"],
            "健康": ["营养", "健身", "心理健康", "医疗技术", "公共卫生"]
        }

        # 保存到文件
        categories_file = os.path.join(CONFIG["DATA_DIR"], "interest_categories.json")
        try:
            with open(categories_file, 'w', encoding='utf-8') as f:
                json.dump(self.interest_categories, f, ensure_ascii=False, indent=4)
            logging.info(f"已创建默认兴趣分类体系并保存到: {categories_file}")
        except Exception as e:
            logging.error(f"保存默认兴趣分类体系出错: {e}")

    def _generate_user_id(self, user_info: Dict) -> str:
        """根据用户信息生成唯一ID"""
        key_string = f"{user_info.get('name', '')}-{user_info.get('email', '')}-{user_info.get('occupation', '')}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def create_user(self, user_info: Dict) -> str:
        """
        创建新用户并存储基本信息

        Args:
            user_info: 用户基本信息，包含姓名、职业、邮箱等

        Returns:
            用户ID
        """
        user_id = self._generate_user_id(user_info)

        conn = get_db_connection()
        try:
            # 检查用户是否已存在
            existing_user = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()

            if not existing_user:
                conn.execute(
                    "INSERT INTO users (id, name, occupation, email) VALUES (?, ?, ?, ?)",
                    (user_id, user_info.get("name", ""), user_info.get("occupation", ""), user_info.get("email", ""))
                )
                conn.commit()
                logging.info(f"创建新用户: {user_id}")
            else:
                logging.info(f"用户已存在: {user_id}")

        finally:
            conn.close()

        return user_id

    def extract_skills_from_resume(self, user_id: str, resume_text: str, max_skills: int = 8) -> List[Dict]:
        """
        从简历文本中提取技能信息

        Args:
            user_id: 用户ID
            resume_text: 简历文本内容
            max_skills: 最多提取的技能数量

        Returns:
            技能列表，每个技能包含名称、级别和分类
        """
        print(f"\n开始从简历中提取最重要的{max_skills}项技能...")

        prompt = f"""
        请从以下简历文本中提取最重要的{max_skills}项技能，并为每个技能提供以下信息：
        1. 技能名称
        2. 熟练程度（初级/中级/高级/专家）
        3. 技能类别（技术技能、软技能、语言技能、管理技能等）

        请按照技能的重要性和熟练程度排序，最重要和最熟练的技能排在前面。

        请严格按照以下JSON格式返回，不要添加任何其他格式标记如```json或```：
        [
            {{"skill": "技能名称", "level": "熟练程度", "category": "技能类别"}},
            ...
        ]

        简历文本：
        {resume_text}
    """

        try:
            print("正在分析简历中的技能...")
            response = self.client.chat.completions.create(
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system",
                     "content": "你是一个专业的简历分析助手，擅长提取简历中的技能信息并进行分类和评估。请只返回JSON格式的结果，不要添加任何其他标记。"},
                    {"role": "user", "content": prompt}
                ]
            )

            skills_text = response.choices[0].message.content

            # 清理可能的格式标记
            skills_text = skills_text.strip()
            if skills_text.startswith("```json"):
                skills_text = skills_text[7:]
            if skills_text.startswith("```"):
                skills_text = skills_text[3:]
            if skills_text.endswith("```"):
                skills_text = skills_text[:-3]
            skills_text = skills_text.strip()

            logging.debug(f"清理后的技能JSON文本: {skills_text}")

            # 尝试解析JSON
            try:
                skills = json.loads(skills_text)
                print(f"成功提取 {len(skills)} 项技能")

                # 确保skills是列表类型
                if not isinstance(skills, list):
                    logging.error(f"解析的技能不是列表类型: {type(skills)}")
                    skills = []
                    print("解析的技能格式不正确，将使用空列表")

                # 保存到数据库
                if skills:  # 只有当skills非空时才尝试保存
                    conn = get_db_connection()
                    try:
                        for i, skill in enumerate(skills):
                            if isinstance(skill, dict):  # 确保每个技能是字典类型
                                conn.execute(
                                    "INSERT INTO user_skills (user_id, skill, level, category) VALUES (?, ?, ?, ?)",
                                    (user_id, skill.get("skill", ""), skill.get("level", ""), skill.get("category", ""))
                                )
                                # 显示进度
                                print(f"保存技能 {i + 1}/{len(skills)}: {skill.get('skill', '')}")
                        conn.commit()
                        print("所有技能已保存到数据库")
                    except Exception as e:
                        logging.error(f"保存技能到数据库时出错: {e}")
                        print(f"保存技能时出错: {e}")
                    finally:
                        conn.close()

                return skills if skills else []

            except json.JSONDecodeError as e:
                logging.error(f"无法解析技能JSON: {skills_text}, 错误: {e}")
                print(f"解析技能信息失败: {e}")
                # 尝试手动解析简单格式
                if "[" in skills_text and "]" in skills_text:
                    try:
                        # 尝试修复常见的JSON格式问题
                        fixed_text = skills_text.replace("'", "\"")
                        skills = json.loads(fixed_text)
                        print(f"修复后成功解析，提取了 {len(skills)} 项技能")
                        return skills
                    except:
                        pass
                return []

        except Exception as e:
            logging.error(f"提取技能时出错: {e}")
            print(f"提取技能时出错: {e}")
            return []

    def extract_interests_from_resume(self, user_id: str, resume_text: str, max_interests: int = 8) -> List[Dict]:
        """
        从简历中提取用户兴趣并分类

        Args:
            user_id: 用户ID
            resume_text: 简历文本
            max_interests: 最多提取的兴趣数量

        Returns:
            兴趣列表，每个兴趣包含主题、分类和初始权重
        """
        print(f"\n开始从简历中提取最重要的{max_interests}项兴趣...")

        # 构建兴趣分类提示
        categories_text = "\n".join([f"{cat}: {', '.join(topics)}" for cat, topics in self.interest_categories.items()])

        prompt = f"""
        请分析以下简历文本，提取用户最重要的{max_interests}项兴趣领域和专业方向。
        将提取的兴趣根据以下分类系统进行归类：

        {categories_text}

        如果发现的兴趣不在上述分类中，请归入最相近的类别。
        对于每个识别的兴趣，根据在简历中的明显程度，给出一个0到1之间的权重。
        请按照兴趣的重要性排序，最重要的兴趣排在前面。

        请严格按照以下JSON格式返回，不要添加任何其他格式标记如```json或```：
        [
            {{"topic": "兴趣主题", "category": "所属类别", "weight": 权重值}},
            ...
        ]

        简历文本：
        {resume_text}
        """

        try:
            print("正在分析简历中的兴趣...")
            response = self.client.chat.completions.create(
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system",
                     "content": "你是一个专业的兴趣分析助手，擅长从文本中提取人们的兴趣爱好并进行分类。请只返回JSON格式的结果，不要添加任何其他标记。"},
                    {"role": "user", "content": prompt}
                ]
            )

            interests_text = response.choices[0].message.content

            # 记录原始响应以便调试
            logging.debug(f"原始兴趣响应: {interests_text}")

            # 清理可能的格式标记
            interests_text = interests_text.strip()
            if interests_text.startswith("```json"):
                interests_text = interests_text[7:]
            if interests_text.startswith("```"):
                interests_text = interests_text[3:]
            if interests_text.endswith("```"):
                interests_text = interests_text[:-3]
            interests_text = interests_text.strip()

            logging.debug(f"清理后的兴趣JSON文本: {interests_text}")

            # 尝试解析JSON
            try:
                interests = json.loads(interests_text)
                print(f"成功提取 {len(interests)} 项兴趣")

                # 确保interests是列表类型
                if not isinstance(interests, list):
                    logging.error(f"解析的兴趣不是列表类型: {type(interests)}")
                    interests = []
                    print("解析的兴趣格式不正确，将使用空列表")

                # 保存到数据库
                if interests:  # 只有当interests非空时才尝试保存
                    conn = get_db_connection()
                    try:
                        for i, interest in enumerate(interests):
                            if isinstance(interest, dict):  # 确保每个兴趣是字典类型
                                # 确保所有必要的键都存在
                                topic = interest.get("topic", "未知兴趣")
                                category = interest.get("category", "未分类")
                                weight = interest.get("weight", 0.5)

                                # 确保weight是浮点数
                                try:
                                    weight = float(weight)
                                except (ValueError, TypeError):
                                    weight = 0.5

                                conn.execute(
                                    "INSERT INTO user_interests (user_id, topic, category, weight) VALUES (?, ?, ?, ?)",
                                    (user_id, topic, category, weight)
                                )
                                # 显示进度
                                print(f"保存兴趣 {i + 1}/{len(interests)}: {topic} (权重: {weight:.2f})")
                        conn.commit()
                        print("所有兴趣已保存到数据库")
                    except Exception as e:
                        logging.error(f"保存兴趣到数据库时出错: {e}")
                        print(f"保存兴趣时出错: {e}")
                        conn.rollback()  # 回滚事务
                    finally:
                        conn.close()

                return interests if interests else []

            except json.JSONDecodeError as e:
                logging.error(f"无法解析兴趣JSON: {interests_text}, 错误: {e}")
                print(f"解析兴趣信息失败: {e}")

                # 尝试手动解析简单格式
                if "[" in interests_text and "]" in interests_text:
                    try:
                        # 尝试修复常见的JSON格式问题
                        fixed_text = interests_text.replace("'", "\"").replace("None", "null")
                        interests = json.loads(fixed_text)
                        print(f"修复后成功解析，提取了 {len(interests)} 项兴趣")
                        return interests
                    except Exception as parse_e:
                        logging.error(f"尝试修复JSON后仍然失败: {parse_e}")

                # 如果无法解析，创建一些基本兴趣
                print("无法解析兴趣，将创建基本兴趣")
                basic_interests = []

                # 从简历文本中提取一些关键词作为基本兴趣
                keywords = ["人工智能", "机器学习", "数据分析", "编程", "算法"]
                for i, keyword in enumerate(keywords):
                    if keyword.lower() in resume_text.lower():
                        interest = {
                            "topic": keyword,
                            "category": "技术",
                            "weight": 0.7
                        }
                        basic_interests.append(interest)

                        # 保存到数据库
                        try:
                            conn = get_db_connection()
                            if conn:
                                conn.execute(
                                    "INSERT INTO user_interests (user_id, topic, category, weight) VALUES (?, ?, ?, ?)",
                                    (user_id, keyword, "技术", 0.7)
                                )
                                conn.commit()
                                print(f"保存基本兴趣: {keyword}")
                                conn.close()
                        except Exception as db_e:
                            logging.error(f"保存基本兴趣到数据库时出错: {db_e}")

                return basic_interests

        except Exception as e:
            logging.error(f"提取兴趣时出错: {e}")
            print(f"提取兴趣时出错: {e}")
            return []

    def record_search(self, user_id: str, query: str, platform: str):
        """
        记录用户搜索行为

        Args:
            user_id: 用户ID
            query: 搜索查询
            platform: 搜索平台
        """
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO user_searches (user_id, query, platform) VALUES (?, ?, ?)",
                (user_id, query, platform)
            )
            conn.commit()
            logging.info(f"记录用户搜索: {user_id}, 查询: {query}")
        finally:
            conn.close()

    def record_interaction(self, user_id: str, content_id: str, action_type: str):
        """
        记录用户与内容的交互

        Args:
            user_id: 用户ID
            content_id: 内容ID（如文章URL）
            action_type: 交互类型（如"点击"、"收藏"、"分享"）
        """
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO user_interactions (user_id, content_id, action_type) VALUES (?, ?, ?)",
                (user_id, content_id, action_type)
            )
            conn.commit()
            logging.info(f"记录用户交互: {user_id}, 内容: {content_id}, 行为: {action_type}")
        finally:
            conn.close()

    def update_interest_weights(self, user_id: str, topic: str, adjustment: float):
        """
        更新用户兴趣权重

        Args:
            user_id: 用户ID
            topic: 兴趣主题
            adjustment: 权重调整值，正数表示增加，负数表示减少
        """
        conn = get_db_connection()
        try:
            # 查找现有兴趣
            interest = conn.execute(
                "SELECT id, weight FROM user_interests WHERE user_id = ? AND topic = ? ORDER BY timestamp DESC LIMIT 1",
                (user_id, topic)
            ).fetchone()

            if interest:
                # 更新权重，确保在0-1范围内
                new_weight = max(0, min(1, interest["weight"] + adjustment))

                conn.execute(
                    "UPDATE user_interests SET weight = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                    (new_weight, interest["id"])
                )
                conn.commit()
                logging.info(f"更新用户兴趣权重: {user_id}, 主题: {topic}, 新权重: {new_weight}")
            else:
                # 如果不存在，创建新的兴趣项
                weight = max(0, min(1, 0.5 + adjustment))  # 默认权重0.5加上调整值

                # 尝试确定类别
                category = "未分类"
                for cat, topics in self.interest_categories.items():
                    if any(compute_similarity(topic, t) > 0.7 for t in topics):
                        category = cat
                        break

                conn.execute(
                    "INSERT INTO user_interests (user_id, topic, category, weight) VALUES (?, ?, ?, ?)",
                    (user_id, topic, category, weight)
                )
                conn.commit()
                logging.info(f"创建新用户兴趣: {user_id}, 主题: {topic}, 权重: {weight}")
        finally:
            conn.close()

    def apply_time_decay(self, user_id: str, decay_factor: float = 0.9, days_threshold: int = 30):
        """
        应用时间衰减模型，降低旧兴趣的权重

        Args:
            user_id: 用户ID
            decay_factor: 衰减因子(0-1)
            days_threshold: 多少天前的兴趣开始衰减
        """
        threshold_date = datetime.datetime.now() - datetime.timedelta(days=days_threshold)
        threshold_str = threshold_date.strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db_connection()
        try:
            old_interests = conn.execute(
                "SELECT id, topic, category, weight, timestamp FROM user_interests WHERE user_id = ? AND timestamp < ?",
                (user_id, threshold_str)
            ).fetchall()

            for interest in old_interests:
                # 计算时间差（天数）
                interest_date = datetime.datetime.strptime(interest["timestamp"], "%Y-%m-%d %H:%M:%S")
                days_diff = (datetime.datetime.now() - interest_date).days

                # 计算衰减倍数（随时间增加而增加衰减）
                decay_multiplier = days_diff // days_threshold

                # 计算新权重
                new_weight = interest["weight"] * (decay_factor ** decay_multiplier)

                # 更新权重
                conn.execute(
                    "UPDATE user_interests SET weight = ? WHERE id = ?",
                    (new_weight, interest["id"])
                )

            conn.commit()
            logging.info(f"应用时间衰减模型: {user_id}, 处理 {len(old_interests)} 条旧兴趣")
        finally:
            conn.close()

    def get_top_interests(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        获取用户的顶级兴趣

        Args:
            user_id: 用户ID
            limit: 返回的兴趣数量

        Returns:
            兴趣列表，按权重排序
        """
        conn = get_db_connection()
        try:
            # 对于每个主题，只取最新的一条记录
            interests = conn.execute("""
                SELECT i1.topic, i1.category, i1.weight, i1.timestamp
                FROM user_interests i1
                INNER JOIN (
                    SELECT topic, MAX(timestamp) as max_time
                    FROM user_interests
                    WHERE user_id = ?
                    GROUP BY topic
                ) i2 ON i1.topic = i2.topic AND i1.timestamp = i2.max_time
                WHERE user_id = ?
                ORDER BY i1.weight DESC
                LIMIT ?
            """, (user_id, user_id, limit)).fetchall()

            return [dict(i) for i in interests]
        finally:
            conn.close()

    def analyze_search_patterns(self, user_id: str, days: int = 30) -> Dict:
        """
        分析用户搜索模式

        Args:
            user_id: 用户ID
            days: 分析的天数范围

        Returns:
            分析结果，包含常用平台、热门查询等
        """
        threshold_date = datetime.datetime.now() - datetime.timedelta(days=days)
        threshold_str = threshold_date.strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db_connection()
        try:
            # 获取搜索记录
            searches = conn.execute(
                "SELECT query, platform, timestamp FROM user_searches WHERE user_id = ? AND timestamp > ?",
                (user_id, threshold_str)
            ).fetchall()

            if not searches:
                return {"status": "无搜索记录"}

            # 统计平台使用情况
            platforms = {}
            for search in searches:
                platform = search["platform"]
                platforms[platform] = platforms.get(platform, 0) + 1

            # 提取查询内容用于语义分析
            queries = [search["query"] for search in searches]

            # 使用LLM分析查询主题
            prompt = f"""
            请分析以下搜索查询列表，识别主要的搜索主题和模式。
            将分析结果以JSON格式返回，包含以下字段：
            1. dominant_topics: 主导主题列表，按重要性排序
            2. search_patterns: 搜索模式描述

            搜索查询列表：
            {queries}
            """

            try:
                response = self.client.chat.completions.create(
                    model=CONFIG["MODELS"]["LLM"],
                    messages=[
                        {"role": "system", "content": "你是一个专业的搜索行为分析助手。"},
                        {"role": "user", "content": prompt}
                    ]
                )

                analysis_text = response.choices[0].message.content

                try:
                    analysis = json.loads(analysis_text)
                except json.JSONDecodeError:
                    analysis = {"dominant_topics": [], "search_patterns": "无法解析分析结果"}

            except Exception as e:
                logging.error(f"分析搜索模式时出错: {e}")
                analysis = {"dominant_topics": [], "search_patterns": f"分析出错: {str(e)}"}

            # 整合结果
            return {
                "platform_stats": platforms,
                "search_count": len(searches),
                "analysis": analysis,
                "timeframe": f"过去{days}天"
            }

        finally:
            conn.close()

    def generate_recommendations(self, user_id: str, count: int = 5) -> List[str]:
        """
        基于用户画像生成内容推荐

        Args:
            user_id: 用户ID
            count: 推荐数量

        Returns:
            推荐的主题列表
        """
        # 获取用户顶级兴趣
        top_interests = self.get_top_interests(user_id, limit=5)

        if not top_interests:
            return ["未找到用户兴趣数据"]

        # 提取兴趣主题
        interest_topics = [i["topic"] for i in top_interests]

        # 使用LLM生成推荐
        prompt = f"""
        基于以下用户兴趣主题，推荐{count}个具体的、精细的研究或学习主题，这些主题应该是前沿的、有深度的，并与用户的兴趣紧密相关。

        用户兴趣主题：{", ".join(interest_topics)}

        请列出具体的推荐主题，每个主题应包含足够的细节和专业性，以便能够直接用于学术研究或专业学习。
        以JSON数组格式返回，每个元素包含'topic'和'reason'字段。
        """

        try:
            response = self.client.chat.completions.create(
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system",
                     "content": "你是一个专业的学习内容推荐助手，擅长为用户提供高质量、前沿的学习主题推荐。"},
                    {"role": "user", "content": prompt}
                ]
            )

            recs_text = response.choices[0].message.content

            try:
                recommendations = json.loads(recs_text)
                return recommendations
            except json.JSONDecodeError:
                logging.error(f"无法解析推荐JSON: {recs_text}")
                return [{"topic": "解析推荐失败", "reason": "请稍后再试"}]

        except Exception as e:
            logging.error(f"生成推荐时出错: {e}")
            return [{"topic": "生成推荐出错", "reason": str(e)}]

    def get_user_profile_summary(self, user_id: str) -> Dict:
        """
        获取用户画像摘要

        Args:
            user_id: 用户ID

        Returns:
            用户画像摘要信息
        """
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return {"error": "无法连接到数据库"}

            # 获取基本信息
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

            if not user:
                return {"error": "用户不存在"}

            # 获取顶级兴趣
            interests = []
            try:
                interests_query = conn.execute("""
                    SELECT i1.topic, i1.category, i1.weight, i1.timestamp 
                    FROM user_interests i1
                    INNER JOIN (
                        SELECT topic, MAX(timestamp) as max_time
                        FROM user_interests
                        WHERE user_id = ?
                        GROUP BY topic
                    ) i2 ON i1.topic = i2.topic AND i1.timestamp = i2.max_time
                    WHERE user_id = ?
                    ORDER BY i1.weight DESC
                    LIMIT 10
                """, (user_id, user_id))

                interests = [dict(i) for i in interests_query.fetchall()]
            except Exception as e:
                logging.error(f"获取用户兴趣时出错: {e}")
                interests = []

            # 获取技能
            skills = []
            try:
                skills_query = conn.execute(
                    "SELECT skill, level, category FROM user_skills WHERE user_id = ? ORDER BY level DESC",
                    (user_id,)
                )
                skills = [dict(s) for s in skills_query.fetchall()]
            except Exception as e:
                logging.error(f"获取用户技能时出错: {e}")
                skills = []

            # 获取搜索统计
            search_count = 0
            try:
                search_count_query = conn.execute(
                    "SELECT COUNT(*) as count FROM user_searches WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
                if search_count_query:
                    search_count = search_count_query["count"]
            except Exception as e:
                logging.error(f"获取搜索统计时出错: {e}")

            # 获取交互统计
            interaction_count = 0
            try:
                interaction_count_query = conn.execute(
                    "SELECT COUNT(*) as count FROM user_interactions WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
                if interaction_count_query:
                    interaction_count = interaction_count_query["count"]
            except Exception as e:
                logging.error(f"获取交互统计时出错: {e}")

            # 获取最近5次搜索
            recent_searches = []
            try:
                searches_query = conn.execute(
                    "SELECT query, platform, timestamp FROM user_searches WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5",
                    (user_id,)
                )
                recent_searches = [dict(s) for s in searches_query.fetchall()]
            except Exception as e:
                logging.error(f"获取最近搜索时出错: {e}")

            # 整合数据
            profile = {
                "basic_info": dict(user),
                "top_interests": interests,
                "skills": skills,
                "activity": {
                    "search_count": search_count,
                    "interaction_count": interaction_count,
                    "recent_searches": recent_searches
                }
            }

            return profile

        except Exception as e:
            logging.error(f"获取用户画像摘要时出错: {e}")
            return {"error": f"获取用户画像摘要时出错: {str(e)}"}
        finally:
            if conn:
                conn.close()


class KnowledgeFlow:
    def __init__(self):
        self.context = {}
        self.client = OpenAI(api_key=CONFIG["API_KEYS"]["deepseek"], base_url="https://api.deepseek.com")
        self.serper_api_key = CONFIG["API_KEYS"]["serper"]
        self.user_profile_manager = UserProfileManager(client=self.client)

    async def _call_llm_stream(self, prompt: str, system_message: str = "你是一个知识渊博且表达能力优秀的AI助手。", stream: bool = True) -> AsyncIterator[str]:
        """
        流式调用：每拿到一个 token 就 yield 给上层，
        用法示例：
            async for tok in self._call_llm_stream(prompt):
                yield tok        # 直接转给前端
        """
        logging.debug(f"调用LLM(流式)。系统消息: '{system_message[:100]}...', 用户Prompt: '{prompt[:150]}...'")
        try:
            # OpenAI 的同步接口放在线程池里跑，返回一个迭代器
            response_iter = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                stream=True
            )

            # Stream对象是同步迭代器，不是异步迭代器，所以不能使用async for
            # 使用普通for循环迭代，每次yield一个token
            for chunk in response_iter:
                if (chunk.choices
                        and chunk.choices[0].delta
                        and (tok := chunk.choices[0].delta.content)):
                    yield tok
                    # 添加一个小的延迟，避免过快输出
                    await asyncio.sleep(0.01)

        except Exception as e:
            logging.error(f"调用LLM(流式)时发生错误: {e}", exc_info=True)
            yield f"\n\n[错误]: {str(e)}"
            
    async def _call_llm(self, prompt: str, system_message: str = "你是一个知识渊博且表达能力优秀的AI助手。") -> str:
        """
        非流式调用LLM：等待完整响应后一次性返回
        """
        logging.debug(f"调用LLM(非流式)。系统消息: '{system_message[:100]}...', 用户Prompt: '{prompt[:150]}...'")
        try:
            # 使用线程池执行同步调用
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                stream=False
            )
            
            # 返回完整内容
            result = response.choices[0].message.content
            return result
            
        except Exception as e:
            logging.error(f"调用LLM(非流式)时发生错误: {e}", exc_info=True)
            return f"生成内容时发生错误: {str(e)}"

    def _prepare_llm_context_from_search_results(self, search_results: Dict, max_items_per_source: int = 3, max_chars: int = 8000) -> str:
        """
        从搜索结果中准备文本上下文给LLM。
        """
        context_parts = []
        total_chars = 0
        for source, data in search_results.items():
            if data and isinstance(data, dict) and 'results' in data:
                items_processed = 0
                for item in data['results']:
                    if items_processed >= max_items_per_source:
                        break
                    title = item.get('title', '')
                    snippet = item.get('snippet', item.get('abstract', '')) 
                    if title and snippet:
                        item_text = f"来源: {source}\n标题: {title}\n摘要: {snippet}\n---\n"
                        if total_chars + len(item_text) > max_chars:
                            remaining_chars = max_chars - total_chars
                            if remaining_chars > 50: 
                                context_parts.append(item_text[:remaining_chars] + "...")
                            break 
                        context_parts.append(item_text)
                        total_chars += len(item_text)
                        items_processed += 1
            if total_chars >= max_chars:
                break
        
        context_str = "\n".join(context_parts)
        if not context_str:
            return "未从搜索结果中提取到足够的上下文信息。"
        return context_str

    def start_node(self, user_input: Dict[str, Any]) -> Dict:
        """ 收集用户初始信息 """
        logging.info("开始收集用户输入信息...")
        required_fields = ['occupation', 'day', 'platform']
        for field in required_fields:
            if field not in user_input:
                raise ValueError(f"缺少必要字段: {field}")

        logging.debug(f"用户输入信息: {user_input}")
        self.context.update(user_input)

        # 创建或获取用户ID
        if 'user_id' not in self.context and 'email' in user_input:
            user_info = {
                'name': user_input.get('user_name', '未知用户'),
                'occupation': user_input.get('occupation', ''),
                'email': user_input.get('email', '')
            }
            self.context['user_id'] = self.user_profile_manager.create_user(user_info)
        
        # 存储文献数量偏好
        self.context['num_papers'] = user_input.get('num_papers', 10)
        logging.info(f"用户请求文献数量: {self.context['num_papers']}")

        # platform_type = self.context.get('platform')
        self.context['update_cycle'] = self.calculate_update_cycle(user_input['day'])
        return self.context

    def calculate_update_cycle(self, days: int) -> Dict:
        """ 计算时间范围 """
        logging.info("计算时间范围...")
        end_date = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(days=days)
        logging.debug(f"时间范围计算结果: 起始日期={start_date}, 结束日期={end_date}")
        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }

    def get_user_profile(self, cv_text: str, task: str) -> Dict:
        """使用大语言模型API初步分析用户画像，并根据不同任务执行不同的prompt"""
        logging.info(f"执行任务：{task}，分析简历内容...")
        print(f"\n===== 开始执行用户画像分析任务：{task} =====")

        # 初始化返回值，确保始终返回一个字典
        profile_analysis = {"skills": [], "interests": []}

        # 如果有用户ID，先调用用户画像管理器处理简历
        if 'user_id' in self.context:
            user_id = self.context['user_id']
            print(f"用户ID: {user_id}")

            # 提取技能
            print("\n第1步：提取用户技能")
            skills = self.user_profile_manager.extract_skills_from_resume(user_id, cv_text)
            logging.info(f"从简历中提取了 {len(skills)} 项技能")
            print(f"技能提取完成，共 {len(skills)} 项")

            # 提取兴趣
            print("\n第2步：提取用户兴趣")
            interests = self.user_profile_manager.extract_interests_from_resume(user_id, cv_text)
            logging.info(f"从简历中提取了 {len(interests)} 项兴趣")
            print(f"兴趣提取完成，共 {len(interests)} 项")

            # 合并技能和兴趣信息到分析结果
            profile_analysis = {
                "skills": skills,
                "interests": interests
            }

            # 定义不同任务的prompt模板
            print("\n第3步：生成综合分析报告")
            prompt_templates = {
                "analyze_resume": f"分析以下简历内容，提供全面的职业画像分析：{cv_text}",
                "user_interest": f"根据以下简历内容，识别用户的职业兴趣和专注领域：{cv_text}",
                "skill_assessment": f"根据以下简历内容，评估用户的技能并提出改进建议：{cv_text}",
                "career_development": f"根据以下简历内容，分析用户的职业发展路径并提供建议：{cv_text}"
            }

            # 确保指定的任务在模板中存在
            if task not in prompt_templates:
                logging.warning(f"未知任务: {task}，将使用默认分析")
                task = "analyze_resume"

            prompt = prompt_templates[task]
            logging.debug(f"任务的prompt: {prompt}")

            try:
                # 调用大语言模型API获取任务的处理结果
                print("正在生成综合分析报告...")
                response = self.client.chat.completions.create(
                    model=CONFIG["MODELS"]["LLM"],
                    messages=[
                        {"role": "system", "content": "你是一个帮助助手，负责根据用户提供的信息分析并生成建议。"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )

                # 从大语言模型响应中提取分析结果
                llm_analysis = response.choices[0].message.content
                logging.debug(f"大语言模型返回的分析结果: {llm_analysis}")

                # 合并LLM分析结果到profile_analysis
                profile_analysis["llm_analysis"] = llm_analysis
            except Exception as e:
                logging.error(f"生成综合分析报告时出错: {e}")
                print(f"生成综合分析报告时出错: {e}")
                profile_analysis["llm_analysis"] = "无法生成分析报告"

            # 更新上下文，保存用户画像分析结果
            self.context.update({"profile_analysis": profile_analysis})
            print("\n用户画像分析完成！")
            return {"profile_analysis": profile_analysis}
        else:
            # 没有用户ID，回退到原来的方法
            print("未找到用户ID，将使用简化版用户画像分析")
            # 定义不同任务的prompt模板
            prompt_templates = {
                "analyze_resume": f"分析以下简历内容：{cv_text}",
                "user_interest": f"根据以下简历内容，识别用户的职业兴趣和专注领域：{cv_text}",
                "skill_assessment": f"根据以下简历内容，评估用户的技能并提出改进建议：{cv_text}",
            }

            # 确保指定的任务在模板中存在
            if task not in prompt_templates:
                tem_task = task
                task = "analyze_resume"
                raise ValueError(f"未知任务: {tem_task}. 请定义一个有效的任务。")

            prompt = prompt_templates[task]
            logging.debug(f"任务的prompt: {prompt}")

            try:
                # 调用大语言模型API获取任务的处理结果
                print("正在生成简化版分析报告...")
                response = self.client.chat.completions.create(
                    model=CONFIG["MODELS"]["LLM"],
                    messages=[
                        {"role": "system", "content": "你是一个帮助助手，负责根据用户提供的信息分析并生成建议。"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False
                )

                # 从大语言模型响应中提取分析结果
                profile_data = response.choices[0].message.content
                logging.debug(f"大语言模型返回的分析结果: {profile_data}")

                # 更新上下文，保存用户画像分析结果
                profile_analysis = {"text_analysis": profile_data}
                self.context.update({"profile_analysis": profile_analysis})
            except Exception as e:
                logging.error(f"生成简化版分析报告时出错: {e}")
                print(f"生成简化版分析报告时出错: {e}")
                profile_analysis = {"text_analysis": "无法生成分析报告"}
                self.context.update({"profile_analysis": profile_analysis})

            print("\n简化版用户画像分析完成！")
            return {"profile_analysis": profile_analysis}

    def build_user_profile(self, user_input: Dict, cv_text: str) -> Dict:
        """
        构建用户画像，确保用户ID存在并分析用户简历

        Args:
            user_input: 用户输入的基本信息
            cv_text: 用户简历文本

        Returns:
            用户画像信息
        """
        try:
            if cv_text.strip():
                print("已收到简历，开始分析...")

                # 确保用户有ID - 如果没有，创建一个新用户
                if 'user_id' not in self.context and 'email' in user_input:
                    print("检测到新用户，正在创建用户档案...")
                    user_info = {
                        'name': user_input.get('name', ''),
                        'occupation': user_input.get('occupation', ''),
                        'email': user_input.get('email', '')
                    }
                    self.context['user_id'] = self.user_profile_manager.create_user(user_info)
                    print(f"已创建新用户，ID: {self.context['user_id']}")

                # 如果仍然没有用户ID（可能是因为没有提供邮箱），创建一个临时ID
                if 'user_id' not in self.context:
                    import hashlib
                    import time
                    temp_id = hashlib.md5(f"{time.time()}-{user_input.get('occupation', '')}-temp".encode()).hexdigest()
                    user_info = {
                        'name': user_input.get('name', '临时用户'),
                        'occupation': user_input.get('occupation', ''),
                        'email': f"temp_{temp_id[:8]}@example.com"  # 创建临时邮箱
                    }
                    self.context['user_id'] = self.user_profile_manager.create_user(user_info)
                    print(f"已创建临时用户，ID: {self.context['user_id']}")
                    print("注意：由于未提供邮箱，此用户为临时用户，数据可能不会长期保存")

                # 现在可以确保有用户ID了，继续进行用户画像分析
                try:
                    task = "analyze_resume"  # 或根据需要选择其他任务
                    user_profile = self.get_user_profile(cv_text, task)
                    # 更新用户画像信息
                    if user_profile:
                        self.context.update(user_profile)
                except Exception as e:
                    logging.error(f"分析用户画像时出错: {e}")
                    print(f"分析用户画像时出错: {e}")
                    print("将继续使用基本用户信息")
                    user_profile = {"basic_profile": True}

                # 显示用户画像摘要
                try:
                    self.display_profile_summary()
                except Exception as e:
                    logging.error(f"显示用户画像摘要时出错: {e}")
                    print(f"显示用户画像摘要时出错: {e}")

                return user_profile or {"basic_profile": True}
            else:
                print("未提供简历，但仍将创建基本用户档案")

                # 即使没有简历，也要确保用户有ID
                if 'user_id' not in self.context and 'email' in user_input:
                    print("创建基本用户档案...")
                    user_info = {
                        'name': user_input.get('name', '未知用户'),
                        'occupation': user_input.get('occupation', ''),
                        'email': user_input.get('email', '')
                    }
                    self.context['user_id'] = self.user_profile_manager.create_user(user_info)
                    print(f"已创建新用户，ID: {self.context['user_id']}")

                    # 从用户输入中提取基本兴趣
                    if 'content_type' in user_input and user_input['content_type']:
                        print(f"基于您提供的关注领域'{user_input['content_type']}'添加初始兴趣")
                        try:
                            self.user_profile_manager.update_interest_weights(
                                self.context['user_id'],
                                user_input['content_type'],
                                0.8  # 较高的初始权重
                            )
                        except Exception as e:
                            logging.error(f"添加初始兴趣时出错: {e}")
                            print(f"添加初始兴趣时出错: {e}")

                return {"basic_profile": True}
        except Exception as e:
            logging.error(f"构建用户画像时出错: {e}")
            print(f"构建用户画像时出错: {e}")
            print("将继续使用基本用户信息")
            return {"basic_profile": True}

    def display_profile_summary(self):
        """显示用户画像摘要"""
        if 'user_id' in self.context:
            try:
                profile_summary = self.user_profile_manager.get_user_profile_summary(self.context['user_id'])
                if not profile_summary or 'error' in profile_summary:
                    print("\n--- 用户画像摘要 ---")
                    print(f"用户ID: {self.context['user_id']}")
                    print("无法获取完整的用户画像信息")
                    print("-----------------\n")
                    return

                print("\n--- 用户画像摘要 ---")
                print(f"用户ID: {self.context['user_id']}")
                print(f"用户名: {self.context['user_name']}")
                print(f"职业: {profile_summary.get('basic_info', {}).get('occupation', '未知')}")

                # 安全地获取技能数量
                skills = profile_summary.get('skills', [])
                print(f"技能数量: {len(skills)}")

                # 安全地获取顶级兴趣
                interests = profile_summary.get('top_interests', [])
                print("顶级兴趣:")
                if interests:
                    for interest in interests[:3]:
                        print(f"  - {interest.get('topic', '未知')} (权重: {interest.get('weight', 0):.2f})")
                else:
                    print("  暂无兴趣数据")
                print("-----------------\n")
            except Exception as e:
                logging.error(f"显示用户画像摘要时出错: {e}")
                print(f"\n显示用户画像摘要时出错: {e}")
                print("将继续执行后续步骤")

    def _update_interests_from_query(self, query: str, weight_adjustment: float = 0.05):
        """
        从用户查询中提取可能的兴趣点并更新用户画像

        Args:
            query: 用户查询
            weight_adjustment: 权重调整幅度
        """
        if 'user_id' not in self.context:
            return

        user_id = self.context['user_id']

        # 使用LLM提取查询中的兴趣点
        prompt = f"""
        请从以下搜索查询中提取最多3个主要的兴趣领域或关键主题。
        只返回提取的主题列表，格式为JSON数组，例如：["人工智能", "机器学习", "自然语言处理"]

        搜索查询: {query}
        """

        try:
            response = self.client.chat.completions.create(
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system", "content": "你是一个专业的主题提取助手，擅长从文本中识别核心主题。"},
                    {"role": "user", "content": prompt}
                ]
            )

            topics_text = response.choices[0].message.content

            try:
                topics = json.loads(topics_text)

                # 更新每个主题的权重
                for topic in topics:
                    self.user_profile_manager.update_interest_weights(user_id, topic, weight_adjustment)
                    logging.info(f"从查询中更新用户兴趣: {topic}, 调整: +{weight_adjustment}")

            except json.JSONDecodeError:
                logging.error(f"无法解析主题JSON: {topics_text}")

        except Exception as e:
            logging.error(f"提取查询主题时出错: {e}")

    async def translate_query(self, query):
        """使用多引擎翻译并比较结果"""
        translations = {}

        try:
            translations['谷歌'] = GoogleTranslator(source='auto', target='en').translate(query)
        except Exception as e:
            logging.error(f"谷歌翻译失败: {e}")

        # try:
        #     translations['百度'] = BaiduTranslator(appid='YOUR_ID', appkey='YOUR_KEY').translate(query, dst='en')
        # except Exception as e:
        #     logging.error(f"百度翻译失败: {e}")

        # 其他翻译API
        # try:
        #     translations['DeepL'] = DeepLTranslator(source='ZH', target='EN').translate(query)
        # except Exception as e:
        #     logging.error(f"DeepL翻译失败: {e}")

        if not translations:
            logging.error("所有翻译引擎均失败")
            return query

        translations_text = "\n".join([f"{engine}翻译：{result}" for engine, result in translations.items()])

        try:
            validation = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": "你是专业翻译验证助手。回答只用翻译后的内容本身！以下是不同引擎翻译的结果，请你思考它们从中文到英文翻译的准确性，并提供最准确的翻译。最终结果只用翻译后的内容本身。"},
                    {"role": "user", "content": f"原文：{query}\n{translations_text}"}
                ]
            )

            translated_query = validation.choices[0].message.content
            translations['大模型'] = translated_query

            # 如果有用户ID，记录这次翻译
            if 'user_id' in self.context:
                # 更新用户兴趣 - 从查询中提取可能的兴趣点
                self._update_interests_from_query(query)

            best_translation = None
            best_similarity = -1

            for engine, translated_text in translations.items():
                similarity = compute_similarity(query, translated_text)
                logging.info(f"{engine}翻译为：{translated_text}，相似度: {similarity}")

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_translation = translated_text

            logging.info(f"最终翻译结果:{translated_query} ，相似度是：{best_similarity}")
            return translated_query  # 目前匹配效果不佳，暂用大模型结果

        except Exception as e:
            logging.error(f"大语言模型验证失败: {e}")
            # 如果验证失败，返回第一个可用的翻译结果
            return next(iter(translations.values()))

    async def build_search_query(self) -> Dict:
        """根据用户需求构建搜索查询，并将查询词转换为英文"""
        logging.info("构建搜索查询并进行翻译...")

        query_base = self.context.get('content_type', " ".join(self.context.get('content_focus', [])))

        # 翻译查询
        translated_query_base = await self.translate_query(query_base)
        if translated_query_base == query_base:
            logging.warning("翻译内容与原查询一致，可能未成功翻译。")

        # 构建基于时间范围的搜索查询
        time_range = f"after:{self.context['update_cycle']['start_date']} before:{self.context['update_cycle']['end_date']}"
        logging.debug(f"构建的搜索查询：query_google={translated_query_base} {time_range} site:google.com")

        # 如果有用户ID，记录这次搜索
        if 'user_id' in self.context:
            self.user_profile_manager.record_search(
                self.context['user_id'],
                query_base,
                self.context.get('platform', '')
            )

        return {
            "query_google": f"{translated_query_base} {time_range} site:google.com",
            "query_arxiv": f'\"{translated_query_base}\" AND submittedDate:[{self.context["update_cycle"]["start_date"].replace("-", "")} TO {self.context["update_cycle"]["end_date"].replace("-", "")}]',
            "query_google_arxiv": f"{translated_query_base} {time_range} arXiv site:arxiv.org"
        }

    def execute_search(self, queries: Dict) -> Dict:
        """执行实际的搜索操作"""
        logging.info("执行搜索操作...")
        results = {}

        platform_type = self.context.get('platform', '新闻类')
        num_papers = self.context.get('num_papers', 10)  # Default to 10 if not set, range 5-20 controlled by frontend/FastAPI
        # Ensure num_papers is within a sane range even if somehow bypassed, though FastAPI should enforce it.
        num_papers = max(5, min(20, num_papers))

        if platform_type == "学术期刊":
            logging.info(f"用户选择学术期刊类，执行ArXiv及Google_ArXiv搜索，目标文献数: {num_papers} per source type")
            if queries.get('query_google_arxiv'):
                logging.debug(f"执行Google_ArXiv搜索: {queries['query_google_arxiv']}")
                results['google_arxiv'] = self.arxiv_search(queries['query_google_arxiv'], max_results=num_papers)
            if queries.get('query_arxiv'): # This one is the direct ArXiv API search
                logging.debug(f"执行ArXiv搜索: {queries['query_arxiv']}")
                results['arxiv'] = self.arxiv_search(queries['query_arxiv'], max_results=num_papers)

        elif platform_type == "新闻类":
            logging.info(f"用户选择新闻类，执行Google搜索，目标文献数: {num_papers}")
            if queries.get('query_google'):
                logging.debug(f"执行Google搜索: {queries['query_google']}")
                results['google'] = self.google_search(queries['query_google'], max_results=num_papers)

        else:  # 综合类或其他类型
            # For "综合类", distribute the num_papers, ensuring each source gets a reasonable amount.
            # Let's aim for roughly num_papers / 2 for the two primary distinct sources (Google general, ArXiv direct)
            # and google_arxiv can also aim for that.
            # If num_papers = 5, num_papers_per_source_综合 = max(2, 5//2) = 2. (Google 2, Arxiv 2, Google_Arxiv 2) -> total up to 6
            # If num_papers = 10, num_papers_per_source_综合 = max(2, 10//2) = 5. (Google 5, Arxiv 5, Google_Arxiv 5) -> total up to 15
            # If num_papers = 20, num_papers_per_source_综合 = max(2, 20//2) = 10. (Google 10, Arxiv 10, Google_Arxiv 10) -> total up to 30
            # This strategy tries to give each source a good number of results based on user's preference.
            num_papers_per_source_综合 = max(2, num_papers // 2) # Ensure at least 2 from each if num_papers is small like 5
            
            logging.info(f"用户选择综合类，执行Google和ArXiv搜索，每类目标文献数: {num_papers_per_source_综合} (源自用户总目标 {num_papers})")
            if queries.get('query_google'):
                logging.debug(f"执行Google搜索: {queries['query_google']}")
                results['google'] = self.google_search(queries['query_google'], max_results=num_papers_per_source_综合)
            if queries.get('query_arxiv'):
                logging.debug(f"执行ArXiv搜索: {queries['query_arxiv']}")
                results['arxiv'] = self.arxiv_search(queries['query_arxiv'], max_results=num_papers_per_source_综合)
            if queries.get('query_google_arxiv'): # This is also an ArXiv search via Google
                logging.debug(f"执行Google_ArXiv搜索: {queries['query_google_arxiv']}")
                # For google_arxiv, it's parsing Google results which are links to ArXiv.
                # The method self.arxiv_search is for direct ArXiv API. self.google_arxiv_search is more appropriate or parse_google_results
                # Re-checking original code: google_arxiv_search calls parse_google_results.
                # So, the max_results for google_arxiv_search should control how many Google results are fetched for ArXiv links.
                results['google_arxiv'] = self.google_arxiv_search(queries['query_google_arxiv'], max_results=num_papers_per_source_综合)

        logging.debug(f"搜索结果: {results}")
        return results

    def google_search(self, query: str, max_results) -> Dict:
        """执行Google搜索，并限制结果数量"""
        logging.info(f"执行Google搜索，限制结果数量为{max_results}...")
        api_url = f"https://google.serper.dev/search?q={query}&num={max_results}"
        headers = {'X-API-KEY': self.serper_api_key}
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            search_results = response.json()
            logging.debug(f"Google搜索返回结果数量: {len(search_results.get('organic', []))}")
            logging.debug(f"Google搜索结果: {search_results}")
            # 限制结果数量
            if 'organic' in search_results:
                search_results['organic'] = search_results['organic'][:max_results]
            return self.parse_google_results(search_results, query)
        else:
            logging.error(f"Google搜索失败，状态码: {response.status_code}")
            return {"error": "Google搜索失败"}

    def parse_google_results(self, data: Dict, query: str) -> Dict:
        """解析Google搜索结果"""
        results = []
        for item in data.get('organic', []):
            title = item.get('title', '')
            snippet = item.get('snippet', '')[:1400]  # 限制摘要长度
            link = item.get('link', '')
            date = item.get('date', '')
            position = item.get('position', '')

            # 计算标题和摘要与查询的相似度
            title_similarity = compute_similarity(query, title)
            snippet_similarity = compute_similarity(query, snippet)
            overall_similarity = 0.3 * title_similarity + 0.7 * snippet_similarity

            results.append({
                'title': title,
                'snippet': snippet,
                'link': link,
                'date': date,
                'position': position,
                'similarity': overall_similarity,
            })

        # 按相似度排序
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)

        return {'results': results}

    def arxiv_search(self, query: str, max_results) -> Dict:
        """执行ArXiv搜索，并限制结果数量"""
        logging.info(f"执行ArXiv搜索，限制结果数量为{max_results}...")
        api_url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}'
        response = requests.get(api_url)

        if response.status_code == 200:
            # logging.debug(f"ArXiv API response (first 500 chars): {response.text[:500]}...")
            arxiv_results = self.parse_arxiv_response(response.text, query)
            logging.debug(f"ArXiv搜索返回结果数量: {len(arxiv_results.get('results', []))}")
            return arxiv_results
        else:
            logging.error(f"ArXiv搜索失败，状态码: {response.status_code}")
            return {"error": "ArXiv搜索失败"}

    def parse_arxiv_response(self, xml_data: str, query: str) -> Dict:
        """解析ArXiv的响应数据"""
        tree = ElementTree.fromstring(xml_data)
        results = []
        for entry in tree.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text

            link = ""
            for link_element in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link_element.get("rel") == "alternate":
                    link = link_element.get("href")
                    break

            if not link:
                link_element = entry.find("{http://www.w3.org/2005/Atom}link")
                if link_element is not None:
                    link = link_element.get("href", "")

            # 计算标题和摘要与查询的相似度
            title_similarity = compute_similarity(query, title)
            summary_similarity = compute_similarity(query, summary)
            overall_similarity = 0.7 * title_similarity + 0.3 * summary_similarity

            # 提取发布日期
            # date = entry.find("{http://www.w3.org/2005/Atom}published").text if entry.find("{http://www.w3.org/2005/Atom}published") is not None else ""

            results.append({
                'title': title,
                'snippet': summary[:1400],
                'link': link,
                # 'date': date,
                'similarity': overall_similarity
            })

        # 按照相似度进行排序
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        return {"results": results}

    def google_arxiv_search(self, query: str, max_results) -> Dict:
        """执行Google搜索，用于搜索ArXiv文献，并限制结果数量"""
        logging.info(f"执行Google搜索用于查找ArXiv文献，限制结果数量为{max_results}...")
        api_url = f"https://google.serper.dev/search?q={query}&num={max_results}"
        headers = {'X-API-KEY': self.serper_api_key}
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            search_results = response.json()
            logging.debug(f"Google_ArXiv搜索返回结果数量: {len(search_results.get('organic', []))}")
            if 'organic' in search_results:
                search_results['organic'] = search_results['organic'][:max_results]
            # logging.debug(f"Google ArXiv搜索结果: {search_results}")
            return self.parse_google_results(search_results, query)
        else:
            logging.error(f"Google ArXiv搜索失败，状态码: {response.status_code}")
            return {"error": "Google ArXiv搜索失败"}

    def integrate_with_large_model(self, search_results: Dict) -> str:
        """调用大语言模型进行整合"""
        logging.info("调用大语言模型进行整合搜索结果...")

        # 将搜索结果转换为大语言模型所需的格式
        search_results_str = json.dumps(search_results, ensure_ascii=False, cls=NumpyEncoder)

        try:
            # 调用大语言模型的API进行结果整合
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                # model="qwen-plus",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个内容整理助手，负责根据用户提供的信息整合搜索结果并生成报告。使用中文。不要md格式。"
                            "将报告中的内容，以：\"来源：...（Google/arXiv/Google_arXiv等平台） \n 标题：... \n 摘要：... \n 原文网址：...（只使用搜索结果中提供的真实链接）\n BERT嵌入的余弦相似度:... \"的形式呈现出来。"
                            "如果搜索结果中没有提供原文网址，则写'原文网址：未提供'。不要编造或猜测网址。"
                            "报告使用用户交谈时的语言，如果原文不是，则准确的转化为用户使用的语言。目前的用户使用的是中文，将结果也转化为中文！"
                            "如果无法完成就直接翻译用snippet的内容回答将\"摘要：\"改为\"片段：\"。"
                            "并且回答严格按照规范来，就算无法完成任务也不要说别的不符合规范的话。"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"请整合以下搜索结果并生成最终报告：{search_results_str}"
                    }
                ],
                stream=False
            )
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                integration_result = response.choices[0].message.content
                logging.debug(f"大语言模型整合结果: {integration_result}")
            else:
                raise ValueError("API响应没有有效的choices字段")

        except Exception as e:
            logging.error(f"调用大语言模型整合时发生错误: {e}")
            integration_result = "由于API错误，无法生成整合结果。"

        return integration_result

    def _extract_interest_from_content(self, title: str, snippet: str, weight_adjustment: float = 0.03):
        """从内容中提取兴趣点并更新用户模型"""
        if 'user_id' not in self.context:
            return

        user_id = self.context['user_id']
        combined_text = f"{title}\n{snippet}"

        # 提取主题
        prompt = f"""
        请从以下文本中提取最多3个核心学术或专业主题。
        只返回主题列表，格式为JSON数组，例如：["强化学习", "计算机视觉", "神经网络"]

        文本:
        {combined_text}
        """

        try:
            response = self.client.chat.completions.create(
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system", "content": "你是一个专业的主题提取助手，擅长从文本中识别核心学术主题。"},
                    {"role": "user", "content": prompt}
                ]
            )

            topics_text = response.choices[0].message.content

            try:
                topics = json.loads(topics_text)

                # 更新每个主题的权重
                for topic in topics:
                    self.user_profile_manager.update_interest_weights(user_id, topic, weight_adjustment)
                    logging.info(f"从内容中提取用户兴趣: {topic}, 调整: +{weight_adjustment}")

            except json.JSONDecodeError:
                logging.error(f"无法解析主题JSON: {topics_text}")

        except Exception as e:
            logging.error(f"提取内容主题时出错: {e}")

    def generate_report(self, search_results: Dict) -> str:
        """生成最终的搜索报告"""
        logging.info("生成最终的搜索报告...")
        report = []
        platform_type = self.context.get('platform')

        # 先构建Google、ArXiv等来源的报告内容
        for source, data in search_results.items():
            if 'error' in data:
                report.append(f"来源：{source}\n错误：{data['error']}")
            else:
                for item in data.get('results', [])[:3]:
                    if isinstance(item, dict):
                        title = item.get('title', '')
                        snippet = item.get('snippet', '')
                        link = item.get('link', '')
                        report.append(f"来源：{source}\n标题：{title}\n摘要：{snippet}...\n链接：{link}")

                    # 如果有用户ID，记录这个内容为潜在兴趣点
                    # if 'user_id' in self.context and link:
                    #     # 从标题中提取可能的兴趣点
                    #     self._extract_interest_from_content(title, snippet, weight_adjustment=0.03)

                    #     # 记录内容交互
                    #     self.user_profile_manager.record_interaction(
                    #         self.context['user_id'],
                    #         link,
                    #         "search_result"
                    #     )

        # 再调用大语言模型整合结果
        final_report = self.integrate_with_large_model(search_results)

        # 最终替换为大语言模型整合后的内容
        report.clear()

        # 如果有用户ID，记录这个内容为潜在兴趣点
        # if 'user_id' in self.context and link:
        #     # 从标题中提取可能的兴趣点
        #     self._extract_interest_from_content(title, snippet, weight_adjustment=0.03)

        #     # 记录内容交互
        #     self.user_profile_manager.record_interaction(
        #         self.context['user_id'],
        #         link,
        #         "search_result"
        #     )

        # 如果有用户ID，添加个性化推荐
        # if 'user_id' in self.context:
        #     # 应用时间衰减模型
        #     self.user_profile_manager.apply_time_decay(self.context['user_id'])

        #     # 获取推荐内容
        #     try:
        #         recommendations = self.user_profile_manager.generate_recommendations(self.context['user_id'], count=3)

        #         # 添加推荐内容到报告
        #         if recommendations and len(recommendations) > 0:
        #             rec_text = "\n\n--- 基于您的兴趣，我们还为您推荐以下主题 ---\n\n"
        #             for rec in recommendations:
        #                 rec_text += f"主题：{rec.get('topic', '')}\n"
        #                 rec_text += f"原因：{rec.get('reason', '')}\n\n"

        #             report.append(rec_text)
        #     except Exception as e:
        #         logging.error(f"生成推荐时出错: {e}")

        report.append(f"\n\n--- 根据您选择的【{platform_type}】平台，近几日的行业内最新进展已整理好，请查收！ ---\n")
        report.append(final_report)

        logging.debug(f"生成的报告内容: {report}")
        return "\n\n".join(report)

    async def _yield_with_title(self, title: str, llm_prompt: str,
                                system_msg: str = "你是一位资深写作者。") -> AsyncIterator[str]:
        """通用小工具：先把标题丢给前端，再流式产出正文"""
        # 先把第一行标题 yield 出去，避免前端一片空白
        yield title + "\n\n"
        async for tok in self._call_llm_stream(llm_prompt, system_msg):
            yield tok

    async def generate_literature_review(self, search_results: Dict, original_query: str) -> str:
        logging.info(f"开始为查询 '{original_query}' 生成文献综述...")
        llm_context_str = self._prepare_llm_context_from_search_results(search_results, max_items_per_source=8)

        references_list = []
        for source, data in search_results.items():
            if data and isinstance(data, dict) and 'results' in data and data['results']:
                for item in data['results']:
                    if isinstance(item, dict):
                        title = item.get('title', '未知标题')
                        link = item.get('link', '#')
                        if link and link != '#':
                            references_list.append({"title": title, "link": link, "source": source})
        
        unique_references = []
        seen_links = set()
        for ref in references_list:
            if ref["link"] not in seen_links:
                unique_references.append(ref)
                seen_links.add(ref["link"])

        system_message_lit_review = "你是一名资深的学术研究员和文献综述撰写专家。请严格按照用户要求的结构和格式生成内容。确保所有输出均为结构良好、干净的Markdown格式，段落间使用双换行符分隔，列表使用标准的Markdown语法。不要在段落中随意插入不必要的换行。用户在Prompt中章节标题后用括号 () 或 （） 包含的文字是对您生成该章节内容的引导和提示，这些括号及其内部的文字绝对不能出现在最终的输出中。您只需要生成这些括号提示之外的、针对该章节的实际内容。所有章节标题和结构由用户在Prompt中指定，请严格遵循。"
        
        prompt = f"""
        请基于主题 "{original_query}" 和以下提供的相关学术文献信息，撰写一份全面且结构清晰的文献综述报告。
        报告应包含以下Markdown结构和内容。请直接在每个章节标题下撰写对应的内容，不要重复标题，也不要输出括号中的提示文字。
        # 文献综述报告: {original_query}
        ## 1. 引言
        (请在此处撰写引言：简要介绍 "{original_query}" 主题的背景和重要性，以及本综述的目的和范围。)
        ## 2. 主要研究方向和核心概念
        (请在此处撰写主要研究方向和核心概念：识别并系统总结文献中涉及的核心研究方向、关键理论模型和重要概念。)
        ## 3. 关键文献回顾与贡献
        (请在此处撰写关键文献回顾与贡献：挑选几篇（3-5篇）具有里程碑意义或代表性的文献进行重点回顾，阐述其主要研究方法、核心发现和学术贡献。在提及具体文献时，可以使用其标题，例如"在《文献标题》中，作者指出..."。)
        ## 4. 研究方法的演进与比较
        (请在此处撰写研究方法的演进与比较：如果文献信息支持，讨论该领域常用的研究方法，它们是如何随时间演变的，并对比不同方法的优缺点。)
        ## 5. 现有研究的局限性与未来研究空白
        (请在此处撰写现有研究的局限性与未来研究空白：基于现有文献，批判性地指出当前研究中存在的不足、争议点或尚未解决的关键问题，从而识别潜在的未来研究空白和方向。)
        ## 6. 结论与展望
        (请在此处撰写结论与展望：对整个综述进行总结，并对 "{original_query}" 领域的未来发展趋势进行展望。)

        ---\n*重要提示：*\n*- 请勿在综述正文中包含URL或直接的链接地址。参考文献列表将由外部程序在报告末尾单独提供。*\n*- 避免简单罗列文献摘要的堆砌，重点在于进行深入的分析、归纳、综合和批判性评价。*\n*- 报告全文请使用中文撰写。如果原始文献材料是其他语言，请确保核心信息的准确翻译和自然融入。*\n

        ---文献信息参考---\n{llm_context_str}\n---文献信息参考结束---\n

        请严格按照以上Markdown结构和要求生成完整的文献综述内容，直接从第一个章节的内容开始写，不要重复最顶层的报告标题，每段话之间至多一句换行，标题和下面内容之间不换行，别部分也不要多余的换行。
        """
        review_text = await self._call_llm(prompt, system_message_lit_review)

        # Prepend the main title that Python controls
        final_report = f"# 文献综述报告: {original_query}\n\n{review_text}"

        if unique_references:
            references_section = "\n\n---\n## 参考文献\n\n"
            for i, ref in enumerate(unique_references):
                references_section += f"{i+1}. **{ref['title']}** (来源: {ref['source']})\n   - 链接: [{ref['link']}]({ref['link']})\n"
            final_report += references_section
        else:
            final_report += "\n\n---\n未找到可引用的文献链接。"
            
        return final_report

    async def generate_industry_research_report(self, search_results: Dict, user_input: Dict, original_query: str) -> str:
        logging.info(f"开始生成行业调研报告: {original_query}")
        context_str = self._prepare_llm_context_from_search_results(search_results)
        
        system_message_industry_analyst = "你是一位经验丰富的行业分析师和商业顾问。请严格按照用户在Prompt中指定的Markdown结构和内容要求进行撰写。确保所有输出均为结构良好、干净的Markdown格式，段落间使用双换行符分隔，列表和表格使用标准的Markdown语法。不要在段落中随意插入不必要的换行。用户在Prompt中章节标题后用括号 () 或 （） 包含的文字是对您生成该章节内容的引导和提示，这些括号及其内部的文字绝对不能出现在最终的输出中。您只需要生成这些括号提示之外的、针对该章节的实际内容。所有章节标题和结构由用户在Prompt中指定，请严格遵循。"

        prompt = f"""
        请针对主题 "{original_query}"，并结合以下提供的背景信息，撰写一份结构清晰、内容详实的行业调研报告。
        报告应严格遵循以下Markdown结构，并在每个章节标题之下直接撰写对应的内容。不要重复章节标题本身，也不要输出括号中的提示性文字。
        # 行业调研报告: {original_query}
        ## 1. 执行摘要
        (请在此处撰写200-300字的执行摘要：概括核心现状、技术特点、关键应用、成熟度判断、机遇与挑战。面向企业决策者或项目负责人。)
        ## 2. 技术概览与核心组件分析
        (请在此处撰写技术概览：分析 "{original_query}" 的核心技术组件、方法论或主要技术分支。描述其原理、特征和作用。确保专业准确，语言清晰。)
        ## 3. 技术对比分析
        (请在此处进行技术对比分析：如果领域内存在多种主流技术路径值得对比，选择2-3个关键点，对比优势、劣势、适用场景、技术差异。尽量使用Markdown表格或列表。如果技术路径单一或不适合对比，请说明原因。)
        ## 4. 技术成熟度评估
        (请在此处评估技术成熟度：评估 "{original_query}" 相关核心技术/分支的当前技术成熟度，如概念验证、研发攻坚、早期市场、快速成长、成熟应用、衰退期。给出判断理由和阶段特征。)
        ## 5. 应用前景与市场机遇
        (请在此处分析应用前景与市场机遇：分析 "{original_query}" 技术未来3-5年的主要应用前景和潜在市场机遇。从行业、场景角度展开，指出驱动因素。)
        ## 6. 近期趋势与潜在风险点
        (请在此处总结近期趋势与潜在风险点：总结 "{original_query}" 领域的最新发展趋势、研究进展或行业动态。识别技术瓶颈、市场风险、伦理或合规挑战。)
        ## 7. 初步战略建议
        (请在此处提供初步战略建议：针对希望在 "{original_query}" 领域进行技术评估、项目投入或战略布局的机构，提供3-5条操作性初步战略建议，聚焦于把握机遇、规避风险、关键切入点。)
        ---\n*免责声明：本报告基于公开信息和AI模型分析生成，仅供参考，不构成任何具体的投资或决策建议。*\n

        ---文献信息参考---\n{context_str}\n---文献信息参考结束---

        请严格按照以上Markdown结构和要求生成完整的报告内容，直接从第一个章节的内容开始写，不要重复最顶层的报告标题，每段话之间至多一句换行，标题和下面内容之间不换行，别部分也不要多余的换行。
        """
        
        full_report_content = await self._call_llm(prompt, system_message_industry_analyst)
        # Prepend the main title that Python controls
        final_report = f"# 行业调研报告: {original_query}\n\n{full_report_content}"
        return final_report

    async def generate_popular_science_report(self, search_results: Dict, user_input: Dict, original_query: str) -> str:
        logging.info(f"开始生成知识科普报告: {original_query}")
        context_str = self._prepare_llm_context_from_search_results(search_results, max_items_per_source=2, max_chars=6000)

        system_msg_science_writer = "你是一位优秀的科普作家和沟通专家。请严格按照用户在Prompt中指定的Markdown结构和内容要求进行撰写。确保所有输出均为结构良好、干净的Markdown格式，段落间使用双换行符分隔，列表和图表（如Mermaid）使用标准Markdown语法。不要在段落中随意插入不必要的换行。用户在Prompt中章节标题后用括号 () 或 （） 包含的文字是对您生成该章节内容的引导和提示，这些括号及其内部的文字绝对不能出现在最终的输出中。您只需要生成这些括号提示之外的、针对该章节的实际内容。所有章节标题和结构由用户在Prompt中指定，请严格遵循。"

        prompt = f"""
        请针对科普主题 "{original_query}"，并结合以下提供的背景信息，为希望拓展知识边界的成年非专业人士撰写一篇生动有趣、易于理解的知识科普文章。
        文章应严格遵循以下Markdown结构，并在每个章节标题之下直接撰写对应的内容。不要重复章节标题本身，也不要输出括号中的提示性文字。
        # 知识科普: 解密 {original_query}
        ## 1. 什么是{original_query} ？
        (请为以上标题下的内容，提供针对成年非专业人士的清晰、简洁且易于理解的核心概念解释。必须包含一个生动且贴切的生活化类比，帮助快速抓住要点。确保解释既准确又不失趣味性。避免不必要的专业术语；如果必须使用，请立即给出通俗解释。)
        ## 2. 为什么我们应该关心 {original_query}？
        (请为以上标题下的内容，阐述对于成年人而言，了解 "{original_query}" 的实际意义或潜在价值。它可能如何影响工作、生活或对世界的认知？请列举2-3点，并用简洁明了的语言阐述。)
        ## 3. 核心概念三连击 (由浅入深)
        (请为以上标题下的内容，采用"三层渐进式解读"的方式阐述其核心概念。
        请按以下结构提供三层解读，可以使用Markdown加粗文本强调分层，但不要自行添加更低级别的Markdown标题如###等：
        *   **第一层：核心概要 (一句话点睛):** [此处填写您的解读内容]
        *   **第二层：工作原理浅析 (形象化解析):** [此处填写您的解读内容]
        *   **第三层：关键特性与延伸思考 (启发性细节):** [此处填写您的解读内容]
        每一层解读都应力求清晰、准确且易于理解。)
        ## 4. 如何开始学习 {original_query}？
        (请为以上标题下的内容，提供一份简洁实用的入门学习指南。指南应包含：
        1.  推荐的学习资源类型（例如，高质量的在线课程、权威科普文章/白皮书、专家访谈或讲座视频、互动式学习网站等）。
        2.  初学者应首先掌握的核心概念或基础知识。
        3.  一个建议的学习步骤或顺序。如果合适，您可以使用Mermaid的graph LR代码块（包裹在 \\\`\\\`\\\`mermaid ... \\\`\\\`\\\` 中）来可视化学习路线图。
        请以清晰的列表或分点形式呈现。)
        ## 5. 关于 {original_query} 的快问快答
        (请为以上标题下的内容，模拟一次面向成年人的入门级Q&A环节。设计2-3个典型且富有启发性的问题。对每个问题给出清晰、准确且通俗易懂的回答，格式如下，不要添加额外标题：
        **Q1: [问题1]**
        A1: [回答1]
        **Q2: [问题2]**
        A2: [回答2]
        )
        ## 6. 结语：探索 {original_query} 的更多乐趣
        (请为以上标题下的内容，撰写一个精炼（约50-80字）、能够激发成年读者持续学习兴趣和探索欲望的结语。结语应积极正面，并强调持续学习的重要性。)
        ---\n*本内容由KnowlEdge AI生成，旨在知识普及，力求通俗易懂。专业细节请参考学术文献。*\n
        ---文献信息参考---\n{context_str}\n---文献信息参考结束---\n
        请严格按照以上Markdown结构和要求生成完整的科普文章内容，直接从第一个章节的内容开始写，不要重复最顶层的报告标题，每段话之间至多一句换行，标题和下面内容之间不换行，别部分也不要多余的换行。
        """
        
        full_report_content = await self._call_llm(prompt, system_msg_science_writer)
        # Prepend the main title that Python controls
        final_report = f"# 知识科普: 解密 {original_query}\n\n{full_report_content}"
        return final_report

    async def generate_report_stream(self, search_results: Dict) -> AsyncIterator[str]:
        """流式生成最终的搜索报告"""
        logging.info("流式生成最终的搜索报告...")
        platform_type = self.context.get('platform')
        
        # 调用大语言模型整合结果
        llm_context_str = self._prepare_llm_context_from_search_results(search_results)
        
        prompt = f"""
        请根据以下搜索结果，生成一份结构清晰、信息丰富的综合报告。
        报告中应该包含对重要信息的提取、归纳总结，并按照逻辑关系组织内容。
        
        ---搜索结果---
        {llm_context_str}
        ---搜索结果结束---
        
        请确保报告：
        1. 有明确的结构和标题
        2. 逻辑清晰，内容连贯
        3. 提炼关键信息，避免冗余
        4. 使用Markdown格式进行排版
        
        根据用户选择的【{platform_type}】平台特点，请特别关注该领域的最新进展和重要信息。
        """
        
        title = f"# 【{platform_type}】综合报告\n\n"
        yield title
        
        async for chunk in self._call_llm_stream(prompt):
            yield chunk
            
        yield f"\n\n---\n\n根据您选择的【{platform_type}】平台，近几日的行业内最新进展已整理好，请查收！"

    async def generate_literature_review_stream(self, search_results: Dict, original_query: str) -> AsyncIterator[str]:
        """流式生成文献综述"""
        logging.info(f"开始流式生成文献综述: {original_query}")
        llm_context_str = self._prepare_llm_context_from_search_results(search_results, max_items_per_source=8)

        references_list = []
        for source, data in search_results.items():
            if data and isinstance(data, dict) and 'results' in data and data['results']:
                for item in data['results']:
                    if isinstance(item, dict):
                        title = item.get('title', '未知标题')
                        link = item.get('link', '#')
                        if link and link != '#':
                            references_list.append({"title": title, "link": link, "source": source})
        
        unique_references = []
        seen_links = set()
        for ref in references_list:
            if ref["link"] not in seen_links:
                unique_references.append(ref)
                seen_links.add(ref["link"])

        system_message_lit_review = "你是一名资深的学术研究员和文献综述撰写专家。请严格按照用户要求的结构和格式生成内容。确保所有输出均为结构良好、干净的Markdown格式，段落间使用双换行符分隔，列表使用标准的Markdown语法。不要在段落中随意插入不必要的换行。用户在Prompt中章节标题后用括号 () 或 （） 包含的文字是对您生成该章节内容的引导和提示，这些括号及其内部的文字绝对不能出现在最终的输出中。您只需要生成这些括号提示之外的、针对该章节的实际内容。所有章节标题和结构由用户在Prompt中指定，请严格遵循。"
        
        prompt = f"""
        请基于主题 "{original_query}" 和以下提供的相关学术文献信息，撰写一份全面且结构清晰的文献综述报告。
        报告应包含以下Markdown结构和内容。请直接在每个章节标题下撰写对应的内容，不要重复标题，也不要输出括号中的提示文字。
        # 文献综述报告: {original_query}
        ## 1. 引言
        (请在此处撰写引言：简要介绍 "{original_query}" 主题的背景和重要性，以及本综述的目的和范围。)
        ## 2. 主要研究方向和核心概念
        (请在此处撰写主要研究方向和核心概念：识别并系统总结文献中涉及的核心研究方向、关键理论模型和重要概念。)
        ## 3. 关键文献回顾与贡献
        (请在此处撰写关键文献回顾与贡献：挑选几篇（3-5篇）具有里程碑意义或代表性的文献进行重点回顾，阐述其主要研究方法、核心发现和学术贡献。在提及具体文献时，可以使用其标题，例如"在《文献标题》中，作者指出..."。)
        ## 4. 研究方法的演进与比较
        (请在此处撰写研究方法的演进与比较：如果文献信息支持，讨论该领域常用的研究方法，它们是如何随时间演变的，并对比不同方法的优缺点。)
        ## 5. 现有研究的局限性与未来研究空白
        (请在此处撰写现有研究的局限性与未来研究空白：基于现有文献，批判性地指出当前研究中存在的不足、争议点或尚未解决的关键问题，从而识别潜在的未来研究空白和方向。)
        ## 6. 结论与展望
        (请在此处撰写结论与展望：对整个综述进行总结，并对 "{original_query}" 领域的未来发展趋势进行展望。)

        ---\n*重要提示：*\n*- 请勿在综述正文中包含URL或直接的链接地址。参考文献列表将由外部程序在报告末尾单独提供。*\n*- 避免简单罗列文献摘要的堆砌，重点在于进行深入的分析、归纳、综合和批判性评价。*\n*- 报告全文请使用中文撰写。如果原始文献材料是其他语言，请确保核心信息的准确翻译和自然融入。*\n

        ---文献信息参考---\n{llm_context_str}\n---文献信息参考结束---\n

        请严格按照以上Markdown结构和要求生成完整的文献综述内容，直接从第一个章节的内容开始写，不要重复最顶层的报告标题，每段话之间至多一句换行，标题和下面内容之间不换行，别部分也不要多余的换行。
        """
        
        # 先输出标题
        title = f"# 文献综述报告: {original_query}\n\n"
        yield title
        
        # 流式输出主体内容
        async for chunk in self._call_llm_stream(prompt, system_message_lit_review):
            yield chunk
            
        # 输出参考文献部分
        if unique_references:
            references_section = "\n\n---\n## 参考文献\n\n"
            for i, ref in enumerate(unique_references):
                references_section += f"{i+1}. **{ref['title']}** (来源: {ref['source']})\n   - 链接: [{ref['link']}]({ref['link']})\n"
            yield references_section
        else:
            yield "\n\n---\n未找到可引用的文献链接。"

    async def generate_industry_research_report_stream(self, search_results: Dict, user_input: Dict, original_query: str) -> AsyncIterator[str]:
        """流式生成行业调研报告"""
        logging.info(f"开始流式生成行业调研报告: {original_query}")
        context_str = self._prepare_llm_context_from_search_results(search_results)
        
        system_message_industry_analyst = "你是一位经验丰富的行业分析师和商业顾问。请严格按照用户在Prompt中指定的Markdown结构和内容要求进行撰写。确保所有输出均为结构良好、干净的Markdown格式，段落间使用双换行符分隔，列表和表格使用标准的Markdown语法。不要在段落中随意插入不必要的换行。用户在Prompt中章节标题后用括号 () 或 （） 包含的文字是对您生成该章节内容的引导和提示，这些括号及其内部的文字绝对不能出现在最终的输出中。您只需要生成这些括号提示之外的、针对该章节的实际内容。所有章节标题和结构由用户在Prompt中指定，请严格遵循。"

        prompt = f"""
        请针对主题 "{original_query}"，并结合以下提供的背景信息，撰写一份结构清晰、内容详实的行业调研报告。
        报告应严格遵循以下Markdown结构，并在每个章节标题之下直接撰写对应的内容。不要重复章节标题本身，也不要输出括号中的提示性文字。
        # 行业调研报告: {original_query}
        ## 1. 执行摘要
        (请在此处撰写200-300字的执行摘要：概括核心现状、技术特点、关键应用、成熟度判断、机遇与挑战。面向企业决策者或项目负责人。)
        ## 2. 技术概览与核心组件分析
        (请在此处撰写技术概览：分析 "{original_query}" 的核心技术组件、方法论或主要技术分支。描述其原理、特征和作用。确保专业准确，语言清晰。)
        ## 3. 技术对比分析
        (请在此处进行技术对比分析：如果领域内存在多种主流技术路径值得对比，选择2-3个关键点，对比优势、劣势、适用场景、技术差异。尽量使用Markdown表格或列表。如果技术路径单一或不适合对比，请说明原因。)
        ## 4. 技术成熟度评估
        (请在此处评估技术成熟度：评估 "{original_query}" 相关核心技术/分支的当前技术成熟度，如概念验证、研发攻坚、早期市场、快速成长、成熟应用、衰退期。给出判断理由和阶段特征。)
        ## 5. 应用前景与市场机遇
        (请在此处分析应用前景与市场机遇：分析 "{original_query}" 技术未来3-5年的主要应用前景和潜在市场机遇。从行业、场景角度展开，指出驱动因素。)
        ## 6. 近期趋势与潜在风险点
        (请在此处总结近期趋势与潜在风险点：总结 "{original_query}" 领域的最新发展趋势、研究进展或行业动态。识别技术瓶颈、市场风险、伦理或合规挑战。)
        ## 7. 初步战略建议
        (请在此处提供初步战略建议：针对希望在 "{original_query}" 领域进行技术评估、项目投入或战略布局的机构，提供3-5条操作性初步战略建议，聚焦于把握机遇、规避风险、关键切入点。)
        ---\n*免责声明：本报告基于公开信息和AI模型分析生成，仅供参考，不构成任何具体的投资或决策建议。*\n

        ---文献信息参考---\n{context_str}\n---文献信息参考结束---

        请严格按照以上Markdown结构和要求生成完整的报告内容，直接从第一个章节的内容开始写，不要重复最顶层的报告标题，每段话之间至多一句换行，标题和下面内容之间不换行，别部分也不要多余的换行。
        """
        
        # 先输出标题
        title = f"# 行业调研报告: {original_query}\n\n"
        yield title
        
        # 流式输出内容
        async for chunk in self._call_llm_stream(prompt, system_message_industry_analyst):
            yield chunk

    async def generate_popular_science_report_stream(self, search_results: Dict, user_input: Dict, original_query: str) -> AsyncIterator[str]:
        """流式生成知识科普报告"""
        logging.info(f"开始流式生成知识科普报告: {original_query}")
        context_str = self._prepare_llm_context_from_search_results(search_results, max_items_per_source=2, max_chars=6000)

        system_msg_science_writer = "你是一位优秀的科普作家和沟通专家。请严格按照用户在Prompt中指定的Markdown结构和内容要求进行撰写。确保所有输出均为结构良好、干净的Markdown格式，段落间使用双换行符分隔，列表和图表（如Mermaid）使用标准Markdown语法。不要在段落中随意插入不必要的换行。用户在Prompt中章节标题后用括号 () 或 （） 包含的文字是对您生成该章节内容的引导和提示，这些括号及其内部的文字绝对不能出现在最终的输出中。您只需要生成这些括号提示之外的、针对该章节的实际内容。所有章节标题和结构由用户在Prompt中指定，请严格遵循。"

        prompt = f"""
        请针对科普主题 "{original_query}"，并结合以下提供的背景信息，为希望拓展知识边界的成年非专业人士撰写一篇生动有趣、易于理解的知识科普文章。
        文章应严格遵循以下Markdown结构，并在每个章节标题之下直接撰写对应的内容。不要重复章节标题本身，也不要输出括号中的提示性文字。
        # 知识科普: 解密 {original_query}
        ## 1. 什么是{original_query} ？
        (请为以上标题下的内容，提供针对成年非专业人士的清晰、简洁且易于理解的核心概念解释。必须包含一个生动且贴切的生活化类比，帮助快速抓住要点。确保解释既准确又不失趣味性。避免不必要的专业术语；如果必须使用，请立即给出通俗解释。)
        ## 2. 为什么我们应该关心 {original_query}？
        (请为以上标题下的内容，阐述对于成年人而言，了解 "{original_query}" 的实际意义或潜在价值。它可能如何影响工作、生活或对世界的认知？请列举2-3点，并用简洁明了的语言阐述。)
        ## 3. 核心概念三连击 (由浅入深)
        (请为以上标题下的内容，采用"三层渐进式解读"的方式阐述其核心概念。
        请按以下结构提供三层解读，可以使用Markdown加粗文本强调分层，但不要自行添加更低级别的Markdown标题如###等：
        *   **第一层：核心概要 (一句话点睛):** [此处填写您的解读内容]
        *   **第二层：工作原理浅析 (形象化解析):** [此处填写您的解读内容]
        *   **第三层：关键特性与延伸思考 (启发性细节):** [此处填写您的解读内容]
        每一层解读都应力求清晰、准确且易于理解。)
        ## 4. 如何开始学习 {original_query}？
        (请为以上标题下的内容，提供一份简洁实用的入门学习指南。指南应包含：
        1.  推荐的学习资源类型（例如，高质量的在线课程、权威科普文章/白皮书、专家访谈或讲座视频、互动式学习网站等）。
        2.  初学者应首先掌握的核心概念或基础知识。
        3.  一个建议的学习步骤或顺序。如果合适，您可以使用Mermaid的graph LR代码块（包裹在 \\\`\\\`\\\`mermaid ... \\\`\\\`\\\` 中）来可视化学习路线图。
        请以清晰的列表或分点形式呈现。)
        ## 5. 关于 {original_query} 的快问快答
        (请为以上标题下的内容，模拟一次面向成年人的入门级Q&A环节。设计2-3个典型且富有启发性的问题。对每个问题给出清晰、准确且通俗易懂的回答，格式如下，不要添加额外标题：
        **Q1: [问题1]**
        A1: [回答1]
        **Q2: [问题2]**
        A2: [回答2]
        )
        ## 6. 结语：探索 {original_query} 的更多乐趣
        (请为以上标题下的内容，撰写一个精炼（约50-80字）、能够激发成年读者持续学习兴趣和探索欲望的结语。结语应积极正面，并强调持续学习的重要性。)
        ---\n*本内容由KnowlEdge AI生成，旨在知识普及，力求通俗易懂。专业细节请参考学术文献。*\n
        ---文献信息参考---\n{context_str}\n---文献信息参考结束---\n
        请严格按照以上Markdown结构和要求生成完整的科普文章内容，直接从第一个章节的内容开始写，不要重复最顶层的报告标题，每段话之间至多一句换行，标题和下面内容之间不换行，别部分也不要多余的换行。
        """
        
        # 先输出标题
        title = f"# 知识科普: 解密 {original_query}\n\n"
        yield title
        
        # 流式输出内容
        async for chunk in self._call_llm_stream(prompt, system_msg_science_writer):
            yield chunk

    def send_email(self, report: str):
        """发送邮件（示例）"""
        logging.info("准备发送邮件...")
        if 'email' in self.context and report:
            print(f"已发送邮件到 {self.context['email']}:\n{report}")
        else:
            print("未找到邮件地址或报告为空，未发送邮件。")

    def process_user_feedback(self, content_id: str, feedback_type: str, feedback_text: str = "") -> str:
        """
        处理用户对内容的反馈

        Args:
            content_id: 内容ID（URL或其他标识）
            feedback_type: 反馈类型（"like"、"dislike"、"save"、"share"等）
            feedback_text: 反馈文本（可选）

        Returns:
            处理结果描述
        """
        if 'user_id' not in self.context:
            return "无用户信息，无法处理反馈"

        user_id = self.context['user_id']

        # 记录交互
        self.user_profile_manager.record_interaction(user_id, content_id, feedback_type)

        # 根据反馈类型调整兴趣权重
        if feedback_type in ("like", "save", "share"):
            # 提取内容关联主题并增加权重
            try:
                # 获取内容摘要（实际应用中可能需要从数据库或通过URL获取）
                content_summary = feedback_text if feedback_text else "用户喜欢的内容"

                # 提取主题
                prompt = f"""
                请从以下用户反馈中提取最多2个可能的兴趣主题。
                只返回主题列表，格式为JSON数组。

                用户反馈类型: {feedback_type}
                反馈内容: {content_summary}
                """

                response = self.client.chat.completions.create(
                    model=CONFIG["MODELS"]["LLM"],
                    messages=[
                        {"role": "system", "content": "你是一个专业的兴趣分析助手，擅长分析用户反馈。"},
                        {"role": "user", "content": prompt}
                    ]
                )

                topics_text = response.choices[0].message.content

                try:
                    topics = json.loads(topics_text)

                    # 增加各主题权重
                    for topic in topics:
                        self.user_profile_manager.update_interest_weights(user_id, topic, 0.1)
                        logging.info(f"基于正面反馈增加兴趣权重: {topic} +0.1")

                    return f"已记录您对'{content_id}'的{feedback_type}反馈，并更新了您的兴趣模型"

                except json.JSONDecodeError:
                    logging.error(f"无法解析主题JSON: {topics_text}")
                    return "已记录您的反馈，但分析主题时出错"

            except Exception as e:
                logging.error(f"处理反馈时出错: {e}")
                return f"处理反馈时出错: {str(e)}"

        elif feedback_type == "dislike":
            # 提取内容关联主题并减少权重
            try:
                content_summary = feedback_text if feedback_text else "用户不喜欢的内容"

                # 提取主题
                prompt = f"""
                请从以下用户不喜欢的内容中提取最多2个可能的兴趣主题。
                只返回主题列表，格式为JSON数组。

                不喜欢的内容: {content_summary}
                """

                response = self.client.chat.completions.create(
                    model=CONFIG["MODELS"]["LLM"],
                    messages=[
                        {"role": "system", "content": "你是一个专业的兴趣分析助手，擅长分析用户反馈。"},
                        {"role": "user", "content": prompt}
                    ]
                )

                topics_text = response.choices[0].message.content

                try:
                    topics = json.loads(topics_text)

                    # 减少各主题权重
                    for topic in topics:
                        self.user_profile_manager.update_interest_weights(user_id, topic, -0.1)
                        logging.info(f"基于负面反馈减少兴趣权重: {topic} -0.1")

                    return f"已记录您对'{content_id}'的不喜欢反馈，并更新了您的兴趣模型"

                except json.JSONDecodeError:
                    logging.error(f"无法解析主题JSON: {topics_text}")
                    return "已记录您的反馈，但分析主题时出错"

            except Exception as e:
                logging.error(f"处理反馈时出错: {e}")
                return f"处理反馈时出错: {str(e)}"

        return f"已记录您对'{content_id}'的{feedback_type}反馈"


class ResumeReader:
    """用于读取多种格式简历文件的类"""

    def __init__(self):
        self.supported_formats = {
            '.txt': self.read_txt,
            '.pdf': self.read_pdf,
            '.docx': self.read_docx,
            '.doc': self.read_doc,
            '.xlsx': self.read_excel,
            '.xls': self.read_excel,
            '.jpg': self.read_image,
            '.jpeg': self.read_image,
            '.png': self.read_image,
        }
        logging.info("初始化ResumeReader，支持的格式：%s", list(self.supported_formats.keys()))

    def read_resume(self, file_path=None):
        """读取简历文件或请求用户输入"""
        if not file_path:
            choice = input(
                "请选择输入方式：1.直接输入文本 2.上传文件 (如有需要进行用户画像构建请输入数字，不需要可回车（或确认）跳过): ")

            if choice == "":
                return ""
            elif choice == "1":
                return input("请输入您的简历文本：")
            elif choice == "2":
                file_path = input("请输入简历文件的完整路径：").strip().strip('"')
            else:
                logging.warning("无效的选择，默认使用文本输入方式")
                return input("请输入您的简历文本：")

        while not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            choice = input(f"文件不存在: {file_path}\n请输入有效的文件路径，或输入 'q' 退出：").strip()

            if choice.lower() == 'q':
                logging.info("用户选择退出")
                return ""
            else:
                file_path = choice.strip()

        file_path = file_path.strip('\"')
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in self.supported_formats:
            logging.warning(f"不支持的文件格式: {file_ext}，支持的格式有: {list(self.supported_formats.keys())}")
            return self.ask_for_input()

        # 调用相应的文件读取方法
        try:
            text = self.supported_formats[file_ext](file_path)
            logging.info(f"成功读取{file_ext}格式简历文件")
            return text
        except Exception as e:
            logging.error(f"读取文件时出错: {e}")
            return self.ask_for_input()

    def ask_for_input(self):
        """帮助用户提供错误反馈并重新输入"""
        return input("请输入您的简历文本：")

    def read_txt(self, file_path):
        """读取txt文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()

    def read_pdf(self, file_path):
        """读取PDF文件"""
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
        return text

    def read_docx(self, file_path):
        """读取Word docx文件"""
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def read_doc(self, file_path):
        """读取旧版 Word doc文件 (需要转换)"""
        logging.warning("直接读取.doc文件需要额外依赖，建议转换为.docx或.pdf格式")
        return f"无法直接读取.doc文件: {file_path}，请转换为.docx或.pdf格式后重试。"

    def read_excel(self, file_path):
        """读取Excel文件"""
        df = pd.read_excel(file_path)
        return df.to_string(index=False)

    def read_image(self, file_path):
        """使用OCR读取图片中的文本"""
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            return text
        except Exception as e:
            logging.error(f"OCR处理图片时出错: {e}")
            return f"OCR处理图片时出错: {e}"


def collect_user_input() -> Dict:
    """收集真实用户输入"""
    print("\n===== 欢迎使用KnowlEdge系统 =====")
    print("请提供以下信息，以便我们为您提供个性化的行业知识更新")

    # user_name = input("请输入您的用户名: ").strip()
    # occupation = input("请输入您的职业: ").strip() or "算法工程师"
    # days = int(input("请输入获取知识更新周期（天数，默认10）: ").strip() or "10")
    # platform = input("请输入消息来源平台（学术期刊/新闻类/综合类，默认学术期刊）: ").strip() or "学术期刊"
    # content_type = input("请输入关注领域（如：大语言模型，默认大语言模型）: ").strip() or "大语言模型"
    # email = input("请输入您的邮箱（用于接收报告和识别用户）: ").strip() or "example@example.com"

    print("\n您的信息已收集完毕，系统将基于这些信息为您提供个性化服务")

    user_name = "Tssword3"
    occupation = "算法工程师"
    days = 7
    platform = "学术期刊"
    content_type = "自然语言处理"
    email = "114514@qq.com"

    return {
        "user_name": user_name,
        "occupation": occupation,
        "day": days,
        "platform": platform,
        "content_type": content_type,
        "email": email
    }


async def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print("\n===== KnowlEdge系统启动 =====")

    # 验证数据库
    if not verify_database():
        print("数据库验证失败，系统可能无法正常工作")
        choice = input("是否继续运行? (y/n): ").strip().lower()
        if choice != 'y':
            print("系统退出")
            return

    # 检查系统初始化状态
    if not os.path.exists(CONFIG["DATA_DIR"]) or not os.path.exists(CONFIG["DB_PATH"]):
        print("系统尚未初始化，正在进行初始化...")
        # 导入并运行初始化脚本
        try:
            import scripts.init_system as init_system
            init_result = init_system.main()
            if not init_result:
                print("系统初始化失败，请检查日志并解决问题后重试")
                return
        except ImportError:
            print("找不到初始化脚本，请确保init_system.py文件存在")
            return

    workflow = KnowledgeFlow()
    print("KnowledgeFlow引擎已初始化")

    # 步骤 1：收集用户输入
    print("\n步骤 1/6: 收集用户信息")
    user_input = collect_user_input()
    workflow.start_node(user_input)
    print("用户信息已收集并处理")

    # 步骤 2：分析用户画像（可选）
    print("\n步骤 2/6: 用户画像分析")
    resume_reader = ResumeReader()
    print("请提供您的简历以进行更精确的用户画像分析（可选）")
    cv_text = resume_reader.read_resume()

    # 构建用户画像
    workflow.build_user_profile(user_input, cv_text)

    # 步骤 3：构建搜索参数
    print("\n步骤 3/6: 构建搜索参数")
    print("正在根据您的需求和兴趣构建搜索参数...")
    queries = await workflow.build_search_query()
    print(f"搜索参数构建完成，将在以下平台搜索: {', '.join(queries.keys())}")

    # 步骤 4：执行Google搜索、ArXiv搜索、和Google搜索ArXiv文献
    print("\n步骤 4/6: 执行搜索")
    print(f"正在搜索与{user_input['content_type']}相关的最新信息...")
    search_results = workflow.execute_search(queries)
    result_count = sum(len(data.get('results', [])) for source, data in search_results.items() if 'results' in data)
    print(f"搜索完成，共找到 {result_count} 条相关信息")

    # 步骤 5：生成报告
    print("\n步骤 5/6: 生成报告")
    print("正在整合搜索结果并生成报告...")
    report = workflow.generate_report(search_results)
    print("报告生成完成")

    # 步骤 6：发送邮件
    print("\n步骤 6/6: 发送报告")
    workflow.send_email(report)

    print("流程执行完成")

import asyncio

if __name__ == "__main__":
    asyncio.run(main())
