"""
用户画像管理模块
负责用户画像的创建、更新和管理
"""
import os
import json
import hashlib
import logging
import datetime
from typing import Dict, List
import sqlite3
from openai import OpenAI
from src.db_utils import get_db_connection
from src.config import CONFIG

class UserProfileManager:
    """用户画像管理器，负责创建、更新和存储用户画像"""

    def __init__(self, client=None):
        """初始化用户画像管理器"""
        # 检查API密钥
        api_key = CONFIG["API_KEYS"]["deepseek"]
        if not api_key or api_key.strip() == "":
            logging.warning("LLM API密钥未设置，用户画像分析功能将受限。请在.env文件中设置DEEPSEEK_API_KEY")
            self.is_mock_mode = True
            self.client = client
        else:
            self.is_mock_mode = False
            self.client = client or OpenAI(
                api_key=api_key,
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

    def get_user_profile(self, user_id: str) -> Dict:
        """
        获取用户的完整画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户画像字典
        """
        return self.get_user_profile_summary(user_id)

    def create_user_profile(self, user_id: str, username: str) -> Dict:
        """
        创建新的用户画像
        
        Args:
            user_id: 用户ID
            username: 用户名称
            
        Returns:
            新创建的用户画像
        """
        conn = get_db_connection()
        try:
            # 检查用户是否已存在
            existing_user = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            
            if not existing_user:
                # 创建新用户
                conn.execute(
                    "INSERT INTO users (id, name) VALUES (?, ?)",
                    (user_id, username)
                )
                conn.commit()
                logging.info(f"创建新用户画像: {user_id}, 用户名: {username}")
            else:
                logging.info(f"用户 {user_id} 已存在")
                
            # 返回新创建的用户画像
            return self.get_user_profile(user_id)
            
        except Exception as e:
            logging.error(f"创建用户画像时出错: {e}")
            return {"error": str(e), "user_id": user_id}
        finally:
            conn.close()

    def update_user_profile(self, user_id: str, education=None, work_experience=None, skills=None) -> Dict:
        """
        更新用户画像的基本信息
        
        Args:
            user_id: 用户ID
            education: 教育背景
            work_experience: 工作经历
            skills: 技能列表
            
        Returns:
            更新后的用户画像
        """
        conn = get_db_connection()
        try:
            # 检查用户是否存在
            user = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if not user:
                logging.error(f"尝试更新不存在的用户: {user_id}")
                return {"error": f"用户 {user_id} 不存在"}
                
            # 更新教育背景
            if education:
                # 先清除旧的教育背景
                conn.execute("DELETE FROM user_education WHERE user_id = ?", (user_id,))
                
                # 添加新的教育背景
                for edu in education:
                    conn.execute(
                        """INSERT INTO user_education 
                           (user_id, institution, major, degree, time_period)
                           VALUES (?, ?, ?, ?, ?)""",
                        (user_id, edu.get("institution", ""), edu.get("major", ""), 
                         edu.get("degree", ""), edu.get("time", ""))
                    )
                    
            # 更新工作经历
            if work_experience:
                # 先清除旧的工作经历
                conn.execute("DELETE FROM user_work_experience WHERE user_id = ?", (user_id,))
                
                # 添加新的工作经历
                for work in work_experience:
                    conn.execute(
                        """INSERT INTO user_work_experience 
                           (user_id, company, position, time_period, description)
                           VALUES (?, ?, ?, ?, ?)""",
                        (user_id, work.get("company", ""), work.get("position", ""), 
                         work.get("time", ""), work.get("description", ""))
                    )
                    
            # 更新技能
            if skills:
                # 先清除旧的技能
                conn.execute("DELETE FROM user_skills WHERE user_id = ?", (user_id,))
                
                # 添加新的技能
                for skill in skills:
                    if isinstance(skill, str):
                        # 如果技能是字符串，使用默认值
                        conn.execute(
                            "INSERT INTO user_skills (user_id, skill) VALUES (?, ?)",
                            (user_id, skill)
                        )
                    elif isinstance(skill, dict):
                        # 如果技能是字典，提取相关字段
                        conn.execute(
                            """INSERT INTO user_skills 
                               (user_id, skill, level, category)
                               VALUES (?, ?, ?, ?)""",
                            (user_id, skill.get("skill", ""), skill.get("level", ""), 
                             skill.get("category", ""))
                        )
                        
            conn.commit()
            logging.info(f"用户画像更新成功: {user_id}")
            
            # 返回更新后的用户画像
            return self.get_user_profile(user_id)
            
        except Exception as e:
            logging.error(f"更新用户画像时出错: {e}")
            return {"error": str(e), "user_id": user_id}
        finally:
            conn.close()

    def update_user_interests(self, user_id: str, interests: List[str]) -> Dict:
        """
        更新用户的兴趣标签
        
        Args:
            user_id: 用户ID
            interests: 兴趣标签列表
            
        Returns:
            更新后的用户画像
        """
        logging.info(f"更新用户 {user_id} 的兴趣标签: {interests}")
        
        conn = get_db_connection()
        try:
            # 检查用户是否存在
            user = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if not user:
                logging.error(f"尝试更新不存在的用户: {user_id}")
                return {"error": f"用户 {user_id} 不存在"}
                
            # 添加新的兴趣
            for interest in interests:
                # 尝试确定类别
                category = "未分类"
                for cat, topics in self.interest_categories.items():
                    if any(t.lower() in interest.lower() for t in topics):
                        category = cat
                        break
                        
                # 检查是否已存在该兴趣
                existing = conn.execute(
                    "SELECT topic FROM user_interests WHERE user_id = ? AND topic = ?",
                    (user_id, interest)
                ).fetchone()
                
                if not existing:
                    # 添加新兴趣
                    conn.execute(
                        """INSERT INTO user_interests 
                           (user_id, topic, category, weight, last_updated)
                           VALUES (?, ?, ?, ?, datetime('now'))""",
                        (user_id, interest, category, 5.0)  # 默认权重为5
                    )
                else:
                    # 增加现有兴趣的权重
                    conn.execute(
                        """UPDATE user_interests 
                           SET weight = weight + 1, last_updated = datetime('now')
                           WHERE user_id = ? AND topic = ?""",
                        (user_id, interest)
                    )
                    
            conn.commit()
            logging.info(f"用户兴趣更新成功: {user_id}")
            
            # 返回更新后的用户画像
            return self.get_user_profile(user_id)
            
        except Exception as e:
            logging.error(f"更新用户兴趣时出错: {e}")
            return {"error": str(e), "user_id": user_id}
        finally:
            conn.close()

    def add_search_history(self, user_id: str, query: str, platforms: List[str]):
        """
        添加用户搜索历史
        
        Args:
            user_id: 用户ID
            query: 搜索查询
            platforms: 搜索平台列表
        """
        platforms_str = ",".join(platforms)
        self.record_search(user_id, query, platforms_str)

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

        # 如果处于模拟模式，返回模拟数据
        if hasattr(self, 'is_mock_mode') and self.is_mock_mode:
            logging.warning("系统处于模拟模式，将返回模拟技能数据")
            mock_skills = [
                {"skill": "Python编程", "level": "高级", "category": "技术技能"},
                {"skill": "数据分析", "level": "中级", "category": "技术技能"},
                {"skill": "机器学习", "level": "初级", "category": "技术技能"}
            ]
            
            # 保存到数据库
            conn = get_db_connection()
            try:
                for skill in mock_skills:
                    conn.execute(
                        "INSERT INTO user_skills (user_id, skill, level, category) VALUES (?, ?, ?, ?)",
                        (user_id, skill["skill"], skill["level"], skill["category"])
                    )
                conn.commit()
                print("模拟技能数据已保存到数据库")
            finally:
                conn.close()
                
            return mock_skills

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
                    finally:
                        conn.close()
                return skills

            except json.JSONDecodeError as e:
                logging.error(f"解析技能JSON时出错: {e}")
                print(f"解析技能JSON时出错: {e}")
                return []

        except Exception as e:
            logging.error(f"提取技能时出错: {e}")
            print(f"提取技能时出错: {e}")
            return []

    def extract_interests_from_resume(self, user_id: str, resume_text: str, max_interests: int = 8) -> List[Dict]:
        """
        从简历文本中提取兴趣信息

        Args:
            user_id: 用户ID
            resume_text: 简历文本内容
            max_interests: 最多提取的兴趣数量

        Returns:
            兴趣列表，每个兴趣包含名称、分类和置信度
        """
        print(f"\n开始从简历中提取最重要的{max_interests}项兴趣...")

        # 如果处于模拟模式，返回模拟数据
        if hasattr(self, 'is_mock_mode') and self.is_mock_mode:
            logging.warning("系统处于模拟模式，将返回模拟兴趣数据")
            mock_interests = [
                {"topic": "人工智能", "category": "技术", "confidence": 0.95, "reason": "简历中多次提到AI相关项目"},
                {"topic": "数据科学", "category": "技术", "confidence": 0.90, "reason": "有数据分析经验"},
                {"topic": "机器学习", "category": "技术", "confidence": 0.85, "reason": "提到了机器学习算法"}
            ]
            
            # 保存到数据库
            conn = get_db_connection()
            try:
                for interest in mock_interests:
                    # 计算初始权重（基于置信度）
                    weight = float(interest.get("confidence", 0.5)) * 10
                    
                    # 保存到user_interests表
                    conn.execute(
                        "INSERT INTO user_interests (user_id, topic, category, weight, reason) VALUES (?, ?, ?, ?, ?)",
                        (user_id, interest.get("topic", ""), interest.get("category", ""), 
                            weight, interest.get("reason", ""))
                    )
                conn.commit()
                print("模拟兴趣数据已保存到数据库")
            finally:
                conn.close()
                
            return mock_interests

        # 获取所有可能的兴趣分类和子类别
        all_categories = []
        all_subcategories = []
        for category, subcategories in self.interest_categories.items():
            all_categories.append(category)
            all_subcategories.extend(subcategories)

        # 构建提示词
        prompt = f"""
        请从以下简历文本中提取出最多{max_interests}个用户可能感兴趣的专业领域或主题，并根据简历内容分析其兴趣强度。

        可以考虑的兴趣大类包括：{', '.join(all_categories)}
        可以考虑的具体领域包括：{', '.join(all_subcategories)}
        
        但也可以提出简历中明确提到但不在上述列表中的其他兴趣领域。

        请按以下JSON格式返回，不要添加额外的markdown标记：
        [
            {{"topic": "兴趣主题", "category": "所属大类", "confidence": 0.95, "reason": "简短的推断理由"}},
            ...
        ]

        兴趣按照推断的置信度从高到低排序，confidence值应在0到1之间。请确保返回的是有效的JSON格式。

        简历文本：
        {resume_text}
        """

        try:
            print("正在分析简历中的兴趣...")
            response = self.client.chat.completions.create(
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system",
                     "content": "你是一个专业的用户兴趣分析师，擅长从个人资料中推断用户的专业兴趣和研究领域。请只返回JSON格式的结果。"},
                    {"role": "user", "content": prompt}
                ]
            )

            interests_text = response.choices[0].message.content

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
                                # 计算初始权重（基于置信度）
                                weight = float(interest.get("confidence", 0.5)) * 10
                                
                                # 保存到user_interests表
                                conn.execute(
                                    "INSERT INTO user_interests (user_id, topic, category, weight, reason) VALUES (?, ?, ?, ?, ?)",
                                    (user_id, interest.get("topic", ""), interest.get("category", ""), 
                                     weight, interest.get("reason", ""))
                                )
                                # 显示进度
                                print(f"保存兴趣 {i + 1}/{len(interests)}: {interest.get('topic', '')}")
                        conn.commit()
                        print("所有兴趣已保存到数据库")
                    finally:
                        conn.close()
                
                return interests

            except json.JSONDecodeError as e:
                logging.error(f"解析兴趣JSON时出错: {e}")
                print(f"解析兴趣JSON时出错: {e}")
                return []

        except Exception as e:
            logging.error(f"提取兴趣时出错: {e}")
            print(f"提取兴趣时出错: {e}")
            return []

    def record_search(self, user_id: str, query: str, platform: str):
        """记录用户搜索历史"""
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO search_history (user_id, query, platform, timestamp) VALUES (?, ?, ?, datetime('now'))",
                (user_id, query, platform)
            )
            conn.commit()
            logging.info(f"记录用户 {user_id} 的搜索: {query}")
        finally:
            conn.close()

    def record_interaction(self, user_id: str, content_id: str, action_type: str):
        """记录用户与内容的交互"""
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO user_interactions (user_id, content_id, action_type, timestamp) VALUES (?, ?, ?, datetime('now'))",
                (user_id, content_id, action_type)
            )
            conn.commit()
            logging.info(f"记录用户 {user_id} 与内容 {content_id} 的交互: {action_type}")
        finally:
            conn.close()

    def update_interest_weights(self, user_id: str, topic: str, adjustment: float):
        """更新用户兴趣权重"""
        conn = get_db_connection()
        try:
            # 检查该兴趣是否已存在
            existing = conn.execute(
                "SELECT topic, weight FROM user_interests WHERE user_id = ? AND topic = ?",
                (user_id, topic)
            ).fetchone()

            if existing:
                # 更新权重
                new_weight = existing[1] + adjustment
                conn.execute(
                    "UPDATE user_interests SET weight = ?, last_updated = datetime('now') WHERE user_id = ? AND topic = ?",
                    (new_weight, user_id, topic)
                )
                logging.info(f"更新用户 {user_id} 的兴趣 '{topic}' 权重: {existing[1]} -> {new_weight}")
            else:
                # 创建新兴趣
                # 尝试确定类别
                category = "未分类"
                for cat, topics in self.interest_categories.items():
                    if any(t.lower() in topic.lower() for t in topics):
                        category = cat
                        break

                conn.execute(
                    "INSERT INTO user_interests (user_id, topic, category, weight, last_updated) VALUES (?, ?, ?, ?, datetime('now'))",
                    (user_id, topic, category, adjustment)
                )
                logging.info(f"为用户 {user_id} 创建新兴趣 '{topic}', 权重: {adjustment}")

            conn.commit()
        finally:
            conn.close()

    def apply_time_decay(self, user_id: str, decay_factor: float = 0.9, days_threshold: int = 30):
        """应用时间衰减模型，降低长期未更新的兴趣权重"""
        conn = get_db_connection()
        try:
            # 获取所有兴趣
            interests = conn.execute(
                """
                SELECT topic, weight, last_updated
                FROM user_interests
                WHERE user_id = ?
                """,
                (user_id,)
            ).fetchall()

            current_time = datetime.datetime.now()

            for topic, weight, last_updated_str in interests:
                if last_updated_str:
                    # 解析上次更新时间
                    last_updated = datetime.datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
                    
                    # 计算天数差
                    days_diff = (current_time - last_updated).days
                    
                    # 如果超过阈值，应用衰减
                    if days_diff > days_threshold:
                        decay_times = (days_diff - days_threshold) // 30  # 每过30天衰减一次
                        if decay_times > 0:
                            new_weight = weight * (decay_factor ** decay_times)
                            
                            # 更新权重
                            conn.execute(
                                "UPDATE user_interests SET weight = ? WHERE user_id = ? AND topic = ?",
                                (new_weight, user_id, topic)
                            )
                            logging.info(f"应用时间衰减: 用户 {user_id} 的兴趣 '{topic}' 权重: {weight} -> {new_weight}")

            conn.commit()
        finally:
            conn.close()

    def get_top_interests(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户的顶级兴趣"""
        conn = get_db_connection()
        try:
            # 查询排序后的兴趣
            interests = conn.execute(
                """
                SELECT topic, category, weight, reason 
                FROM user_interests
                WHERE user_id = ?
                ORDER BY weight DESC
                LIMIT ?
                """,
                (user_id, limit)
            ).fetchall()

            # 格式化结果
            result = [
                {
                    "topic": topic,
                    "category": category,
                    "weight": weight,
                    "reason": reason
                }
                for topic, category, weight, reason in interests
            ]

            return result
        finally:
            conn.close()

    def analyze_search_patterns(self, user_id: str, days: int = 30) -> Dict:
        """分析用户的搜索模式"""
        conn = get_db_connection()
        try:
            # 获取最近的搜索历史
            searches = conn.execute(
                """
                SELECT query, platform, timestamp
                FROM search_history
                WHERE user_id = ? AND timestamp >= datetime('now', '-? days')
                ORDER BY timestamp DESC
                """,
                (user_id, days)
            ).fetchall()

            # 提取关键词频率
            keywords = {}
            platforms = {}
            search_times = []

            for query, platform, timestamp in searches:
                # 记录平台使用频率
                platforms[platform] = platforms.get(platform, 0) + 1

                # 使用简单的空格分割作为示例
                # 实际应用中可能需要更复杂的文本处理
                for word in query.lower().split():
                    if len(word) > 2:  # 忽略非常短的词
                        keywords[word] = keywords.get(word, 0) + 1

                # 解析时间用于时间模式分析
                search_time = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                search_times.append(search_time)

            # 按频率排序关键词
            sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
            top_keywords = sorted_keywords[:10] if len(sorted_keywords) > 10 else sorted_keywords

            # 平台使用分析
            sorted_platforms = sorted(platforms.items(), key=lambda x: x[1], reverse=True)

            # 简单的时间模式分析
            time_patterns = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}
            weekday_patterns = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # 0=Monday, 6=Sunday

            for t in search_times:
                hour = t.hour
                if 5 <= hour < 12:
                    time_patterns["morning"] += 1
                elif 12 <= hour < 17:
                    time_patterns["afternoon"] += 1
                elif 17 <= hour < 22:
                    time_patterns["evening"] += 1
                else:
                    time_patterns["night"] += 1

                weekday_patterns[t.weekday()] += 1

            # 返回分析结果
            return {
                "search_count": len(searches),
                "top_keywords": [{"keyword": k, "count": c} for k, c in top_keywords],
                "platform_usage": [{"platform": p, "count": c} for p, c in sorted_platforms],
                "time_patterns": [{"period": p, "count": c} for p, c in time_patterns.items()],
                "weekday_patterns": [{"day": d, "count": c} for d, c in weekday_patterns.items()]
            }

        finally:
            conn.close()

    def generate_recommendations(self, user_id: str, count: int = 5) -> List[Dict]:
        """基于用户画像生成推荐内容"""
        # 获取用户的顶级兴趣
        top_interests = self.get_top_interests(user_id, limit=3)
        
        if not top_interests:
            logging.warning(f"用户 {user_id} 没有足够的兴趣数据生成推荐")
            return []
            
        # 分析搜索模式
        search_patterns = self.analyze_search_patterns(user_id)
        
        # 准备推荐生成的提示词
        interests_str = ", ".join([interest["topic"] for interest in top_interests])
        
        prompt = f"""
        根据用户的兴趣领域：{interests_str}，生成{count}条个性化的主题推荐。

        每条推荐应该包含：
        1. 推荐的主题
        2. 为什么推荐（基于用户兴趣的理由）
        3. 可能的价值或受益

        推荐应该具有多样性，并覆盖不同的知识面。请确保推荐是有具体内容的，而不是笼统的大类。

        请以JSON格式返回：
        [
            {{"topic": "推荐主题1", "reason": "推荐理由", "value": "可能的价值"}},
            ...
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=CONFIG["MODELS"]["LLM"],
                messages=[
                    {"role": "system", 
                     "content": "你是一个专业的内容推荐系统，负责根据用户兴趣生成有价值的主题推荐。"},
                    {"role": "user", "content": prompt}
                ]
            )
            
            recommendations_text = response.choices[0].message.content
            
            # 清理JSON
            recommendations_text = recommendations_text.strip()
            if recommendations_text.startswith("```json"):
                recommendations_text = recommendations_text[7:]
            if recommendations_text.startswith("```"):
                recommendations_text = recommendations_text[3:]
            if recommendations_text.endswith("```"):
                recommendations_text = recommendations_text[:-3]
            recommendations_text = recommendations_text.strip()
            
            try:
                recommendations = json.loads(recommendations_text)
                logging.info(f"为用户 {user_id} 生成了 {len(recommendations)} 条推荐")
                return recommendations
            except json.JSONDecodeError as e:
                logging.error(f"解析推荐JSON时出错: {e}")
                return []
                
        except Exception as e:
            logging.error(f"生成推荐时出错: {e}")
            return []

    def get_user_profile_summary(self, user_id: str) -> Dict:
        """获取用户画像摘要"""
        conn = get_db_connection()
        try:
            # 获取基本用户信息
            user_info = conn.execute(
                "SELECT name, occupation, email FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()
            
            if not user_info:
                return {"error": f"未找到用户 {user_id}"}
                
            name, occupation, email = user_info
            
            # 获取技能信息
            skills = conn.execute(
                """
                SELECT skill, level, category
                FROM user_skills
                WHERE user_id = ?
                ORDER BY level DESC
                """,
                (user_id,)
            ).fetchall()
            
            # 获取兴趣信息
            interests = conn.execute(
                """
                SELECT topic, category, weight
                FROM user_interests
                WHERE user_id = ?
                ORDER BY weight DESC
                LIMIT 10
                """,
                (user_id,)
            ).fetchall()
            
            # 获取最近搜索
            recent_searches = conn.execute(
                """
                SELECT query, platform, timestamp
                FROM search_history
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 5
                """,
                (user_id,)
            ).fetchall()
            
            # 构建摘要
            return {
                "user_id": user_id,
                "basic_info": {
                    "name": name,
                    "occupation": occupation,
                    "email": email
                },
                "skills": [
                    {"skill": skill, "level": level, "category": category}
                    for skill, level, category in skills
                ],
                "interests": [
                    {"topic": topic, "category": category, "weight": weight}
                    for topic, category, weight in interests
                ],
                "recent_searches": [
                    {"query": query, "platform": platform, "timestamp": timestamp}
                    for query, platform, timestamp in recent_searches
                ]
            }
            
        finally:
            conn.close() 