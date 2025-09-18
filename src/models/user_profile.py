"""
用户画像管理模块
负责用户画像的创建、更新和管理
"""
import os
import json
import hashlib
import logging
import datetime
from typing import Dict, List, Optional, Tuple
import sqlite3
from src.db_utils import get_db_connection
from src.config import Config
from src.core.llm_interface import LLMInterface
from dataclasses import dataclass
from datetime import datetime
import re

@dataclass
class InterestItem:
    """兴趣项数据结构"""
    topic: str
    category: str
    weight: float
    interest_level: int  # 0-10级
    search_count: int    # 搜索次数
    reason: Optional[str] = None
    last_updated: Optional[str] = None

@dataclass  
class SkillItem:
    """技能项数据结构"""
    skill: str
    level: str
    category: str
    skill_level: int      # 0-10级数值
    skill_category: str   # 技能分类

class EnhancedUserProfileExtractor:
    """增强的用户画像提取器"""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
        
        # 兴趣关键词映射和权重
        self.interest_keywords = {
            '人工智能': {'keywords': ['ai', 'artificial intelligence', '人工智能', '机器学习', 'ml', 'deep learning', '深度学习', 'neural network', '神经网络'], 'category': 'AI技术', 'base_level': 3},
            '自然语言处理': {'keywords': ['nlp', 'natural language', '自然语言', 'language model', '语言模型', 'chatgpt', 'bert', 'transformer'], 'category': 'AI技术', 'base_level': 4},
            '计算机视觉': {'keywords': ['cv', 'computer vision', '计算机视觉', 'image', '图像', 'vision', 'opencv', '图像处理'], 'category': 'AI技术', 'base_level': 4},
            '数据科学': {'keywords': ['data science', '数据科学', 'analytics', '数据分析', 'statistics', '统计', 'pandas', 'numpy', '数据挖掘'], 'category': '数据分析', 'base_level': 3},
            '大数据': {'keywords': ['big data', '大数据', 'hadoop', 'spark', 'hive', 'kafka', '数据工程'], 'category': '数据分析', 'base_level': 3},
            '区块链': {'keywords': ['blockchain', '区块链', 'crypto', '加密货币', 'bitcoin', 'ethereum', 'web3'], 'category': '新兴技术', 'base_level': 2},
            '云计算': {'keywords': ['cloud', '云计算', 'aws', 'azure', 'docker', 'kubernetes', '容器', 'devops'], 'category': '基础设施', 'base_level': 3},
            '网络安全': {'keywords': ['security', '安全', 'cyber', '网络安全', 'encryption', '密码学', '渗透测试'], 'category': '安全技术', 'base_level': 4},
            '前端开发': {'keywords': ['frontend', '前端', 'react', 'vue', 'javascript', 'css', 'html', 'typescript', 'angular'], 'category': '软件开发', 'base_level': 2},
            '后端开发': {'keywords': ['backend', '后端', 'api', 'server', 'database', 'spring', 'django', 'flask', 'node.js'], 'category': '软件开发', 'base_level': 3},
            '移动开发': {'keywords': ['mobile', '移动', 'android', 'ios', 'flutter', 'react native', 'swift', 'kotlin'], 'category': '软件开发', 'base_level': 3},
            '游戏开发': {'keywords': ['game', '游戏开发', 'unity', 'unreal', 'cocos', '游戏引擎'], 'category': '软件开发', 'base_level': 3},
            '数据库技术': {'keywords': ['database', '数据库', 'mysql', 'postgresql', 'mongodb', 'redis', 'sql'], 'category': '基础设施', 'base_level': 2},
            '软件架构': {'keywords': ['architecture', '架构', 'microservices', '微服务', '系统设计', 'design pattern'], 'category': '软件开发', 'base_level': 4},
            '产品管理': {'keywords': ['product', '产品', 'pm', '产品经理', '需求分析', 'prototype'], 'category': '管理', 'base_level': 2},
            '项目管理': {'keywords': ['project', '项目管理', 'scrum', 'agile', '敏捷', 'kanban'], 'category': '管理', 'base_level': 2},
            '用户体验': {'keywords': ['ux', 'ui', '用户体验', '交互设计', 'figma', 'sketch', 'prototype'], 'category': '设计', 'base_level': 2},
            '金融科技': {'keywords': ['fintech', '金融科技', '支付', 'payment', '风控', '量化'], 'category': '行业应用', 'base_level': 3},
            '物联网': {'keywords': ['iot', '物联网', '传感器', '嵌入式', 'raspberry pi', 'arduino'], 'category': '新兴技术', 'base_level': 3},
            '运维工程': {'keywords': ['devops', '运维', 'ops', 'monitoring', '监控', 'deployment', '部署'], 'category': '基础设施', 'base_level': 3},
        }
        
        # 技能等级评估关键词
        self.skill_level_indicators = {
            # 等级描述词
            'expert': 9, 'expertise': 9, 'specialist': 9,    # 专家级
            'senior': 7, 'lead': 7, 'principal': 8,    # 高级/领导级
            'advanced': 6, 'proficient': 6,  # 进阶/熟练
            'intermediate': 4, 'experienced': 5,  # 中级/有经验
            'beginner': 2, 'junior': 2,  # 初级
            'basic': 1, 'fundamental': 1,     # 基础
            # 中文描述
            '专家': 9, '资深': 8, '高级': 7, '进阶': 6, '熟练': 5, '中级': 4, '初级': 2, '基础': 1,
            # 年限指标
            '5年以上': 7, '3-5年': 6, '2-3年': 4, '1-2年': 3, '1年': 2,
            '5+年': 7, '3+年': 5, '2+年': 4, '1+年': 3,
            # 项目经验指标
            '项目经验': 4, '商业项目': 5, '大型项目': 6, '核心开发': 6,
            '架构设计': 7, '技术负责': 7, '团队领导': 8,
            # 认证和教育指标
            '认证': 3, '证书': 3, '培训': 2, '学习': 2,
            '毕业设计': 3, '研究': 4, '论文': 5,
        }
        
        # 技能分类 - 扩展更多分类
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'go', 'rust', 'php', 'c#', 'swift', 'kotlin', 'scala'],
            'frontend': ['html', 'css', 'javascript', 'typescript', 'react', 'vue', 'angular', 'sass', 'less'],
            'backend': ['node.js', 'django', 'flask', 'spring', 'express', 'fastapi', 'gin', 'rails'],
            'mobile': ['android', 'ios', 'flutter', 'react native', 'xamarin', 'ionic'],
            'database': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible'],
            'ai_ml': ['tensorflow', 'pytorch', 'scikit-learn', 'opencv', 'keras', 'pandas', 'numpy'],
            'tools': ['git', 'linux', 'docker', 'jenkins', 'nginx', 'vim', 'vscode'],
            'design': ['photoshop', 'illustrator', 'figma', 'sketch', 'ui', 'ux'],
            'management': ['agile', 'scrum', 'kanban', 'jira', 'confluence', '项目管理'],
        }
        
        # 技能熟练度推断关键词
        self.proficiency_keywords = {
            # 高熟练度指标 (7-9分)
            'architect': 8, 'design': 6, 'optimize': 6, 'implement': 5,
            'develop': 4, 'build': 4, 'create': 4, 'maintain': 5,
            '架构': 8, '设计': 6, '优化': 6, '实现': 5, '开发': 4, '构建': 4, '维护': 5,
            # 协作和管理指标
            'collaborate': 4, 'mentor': 6, 'teach': 6, 'review': 5,
            '协作': 4, '指导': 6, '教学': 6, '评审': 5, '带队': 7,
            # 问题解决指标
            'troubleshoot': 5, 'debug': 4, 'solve': 4, 'fix': 3,
            '排错': 5, '调试': 4, '解决': 4, '修复': 3,
        }
        
        # 搜索行为技能关键词映射
        self.search_skill_keywords = {
            'Python': {'keywords': ['python', 'django', 'flask', 'fastapi'], 'category': 'programming', 'base_level': 2},
            'JavaScript': {'keywords': ['javascript', 'js', 'node.js', 'nodejs'], 'category': 'programming', 'base_level': 2},
            'React': {'keywords': ['react', 'reactjs', 'jsx'], 'category': 'framework', 'base_level': 2},
            'Vue': {'keywords': ['vue', 'vuejs'], 'category': 'framework', 'base_level': 2},
            'Docker': {'keywords': ['docker', 'container', '容器'], 'category': 'cloud', 'base_level': 3},
            'Kubernetes': {'keywords': ['kubernetes', 'k8s'], 'category': 'cloud', 'base_level': 4},
            'TensorFlow': {'keywords': ['tensorflow', 'tf'], 'category': 'ai_ml', 'base_level': 3},
            'PyTorch': {'keywords': ['pytorch', 'torch'], 'category': 'ai_ml', 'base_level': 3},
            'MySQL': {'keywords': ['mysql', 'sql'], 'category': 'database', 'base_level': 2},
            'MongoDB': {'keywords': ['mongodb', 'mongo', 'nosql'], 'category': 'database', 'base_level': 3},
        }

    def extract_interests_from_search(self, search_content: str, user_id: str) -> List[InterestItem]:
        """基于搜索内容提取兴趣并更新等级"""
        interests = []
        search_lower = search_content.lower()
        
        for topic, config in self.interest_keywords.items():
            # 检查是否匹配关键词
            match_count = sum(1 for keyword in config['keywords'] if keyword.lower() in search_lower)
            
            if match_count > 0:
                # 基础兴趣等级 + 匹配程度加成
                interest_level = min(10, config['base_level'] + match_count)
                
                interest = InterestItem(
                    topic=topic,
                    category=config['category'],
                    weight=0.5 + (match_count * 0.2),  # 基础权重 + 匹配加成
                    interest_level=interest_level,
                    search_count=1,
                    reason=f"基于搜索内容'{search_content[:50]}...'分析得出",
                    last_updated=datetime.now().isoformat()
                )
                interests.append(interest)
        
        # 更新数据库中的兴趣（默认为搜索来源）
        self._update_user_interests(user_id, interests, "search")
        return interests

    def extract_interests_from_resume(self, resume_content: str, user_id: str) -> List[InterestItem]:
        """基于简历内容提取兴趣（静态，不累加）"""
        interests = []
        resume_lower = resume_content.lower()
        
        for topic, config in self.interest_keywords.items():
            # 检查是否匹配关键词
            match_count = sum(1 for keyword in config['keywords'] if keyword.lower() in resume_lower)
            
            if match_count > 0:
                # 简历兴趣等级基于匹配程度，但相对保守
                interest_level = min(8, config['base_level'] + match_count)  # 最高8级
                
                interest = InterestItem(
                    topic=topic,
                    category=config['category'],
                    weight=0.8 + (match_count * 0.1),  # 简历权重稍高但不会过高
                    interest_level=interest_level,
                    search_count=0,  # 简历兴趣不计搜索次数
                    reason=f"基于简历内容分析得出，匹配度: {match_count}",
                    last_updated=datetime.now().isoformat()
                )
                interests.append(interest)
        
        # 更新数据库中的兴趣（标记为简历来源）
        self._update_user_interests(user_id, interests, "resume")
        return interests

    def extract_skills_from_search(self, search_content: str, user_id: str) -> List[SkillItem]:
        """基于搜索内容提取技能"""
        skills = []
        search_lower = search_content.lower()
        
        for skill_name, config in self.search_skill_keywords.items():
            # 检查是否匹配关键词
            match_count = sum(1 for keyword in config['keywords'] if keyword.lower() in search_lower)
            
            if match_count > 0:
                # 基于搜索内容推断的技能等级（保守）
                skill_level = min(6, config['base_level'] + match_count)  # 搜索推断最高6级
                
                skill = SkillItem(
                    skill=skill_name,
                    level="搜索推断",
                    category="搜索发现",
                    skill_level=skill_level,
                    skill_category=config['category']
                )
                skills.append(skill)
        
        # 更新数据库中的技能（标记为搜索来源）
        self._update_user_skills(user_id, skills, "search")
        return skills

    def _update_user_skills(self, user_id: str, new_skills: List[SkillItem], source_type: str = "resume"):
        """更新用户技能到数据库，支持区分来源"""
        try:
            from src.db_utils import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            
            for skill in new_skills:
                if source_type == "search":
                    # 搜索来源：检查是否已存在，不存在才插入
                    existing = cursor.execute(
                        "SELECT id FROM user_skills WHERE user_id=? AND skill=? AND source_type='search'",
                        (user_id, skill.skill)
                    ).fetchone()
                    
                    if not existing:
                        cursor.execute("""
                            INSERT INTO user_skills 
                            (user_id, skill, level, category, skill_level, skill_category, source_type)
                            VALUES (?, ?, ?, ?, ?, ?, 'search')
                        """, (user_id, skill.skill, skill.level, skill.category, 
                              skill.skill_level, skill.skill_category))
                
                elif source_type == "resume":
                    # 简历来源：检查是否已存在，不存在才插入
                    existing = cursor.execute(
                        "SELECT id FROM user_skills WHERE user_id=? AND skill=? AND source_type='resume'",
                        (user_id, skill.skill)
                    ).fetchone()
                    
                    if not existing:
                        cursor.execute("""
                            INSERT INTO user_skills 
                            (user_id, skill, level, category, skill_level, skill_category, source_type)
                            VALUES (?, ?, ?, ?, ?, ?, 'resume')
                        """, (user_id, skill.skill, skill.level, skill.category, 
                              skill.skill_level, skill.skill_category))
            
            conn.commit()
            conn.close()
            logging.info(f"已更新用户 {user_id} 的{source_type}技能: {[s.skill for s in new_skills]}")
            
        except Exception as e:
            logging.error(f"更新用户技能失败: {e}")

    def _update_user_interests(self, user_id: str, new_interests: List[InterestItem], source_type: str = "search"):
        """更新用户兴趣到数据库，支持区分来源和累加搜索次数"""
        try:
            from src.db_utils import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            
            for interest in new_interests:
                if source_type == "search":
                    # 搜索来源：只更新搜索类型的兴趣，支持累加
                    existing = cursor.execute(
                        "SELECT interest_level, search_count FROM user_interests WHERE user_id=? AND topic=? AND source_type='search'",
                        (user_id, interest.topic)
                    ).fetchone()
                    
                    if existing:
                        # 累加搜索次数，提升兴趣等级
                        new_search_count = existing[1] + 1 if existing[1] else 1
                        new_level = min(10, (existing[0] or 1) + 1)  # 每次搜索提升1级，最高10级
                        
                        cursor.execute("""
                            UPDATE user_interests 
                            SET interest_level=?, search_count=?, weight=?, timestamp=datetime('now')
                            WHERE user_id=? AND topic=? AND source_type='search'
                        """, (new_level, new_search_count, interest.weight, user_id, interest.topic))
                    else:
                        # 插入新的搜索兴趣
                        cursor.execute("""
                            INSERT INTO user_interests 
                            (user_id, topic, category, weight, interest_level, search_count, reason, source_type, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                        """, (user_id, interest.topic, interest.category, interest.weight, 
                              interest.interest_level, interest.search_count, interest.reason, 'search'))
                
                elif source_type == "resume":
                    # 简历来源：检查是否已存在，不存在才插入，不累加
                    existing = cursor.execute(
                        "SELECT id FROM user_interests WHERE user_id=? AND topic=? AND source_type='resume'",
                        (user_id, interest.topic)
                    ).fetchone()
                    
                    if not existing:
                        # 插入新的简历兴趣（搜索次数设为0）
                        cursor.execute("""
                            INSERT INTO user_interests 
                            (user_id, topic, category, weight, interest_level, search_count, reason, source_type, timestamp)
                            VALUES (?, ?, ?, ?, ?, 0, ?, 'resume', datetime('now'))
                        """, (user_id, interest.topic, interest.category, interest.weight, 
                              interest.interest_level, interest.reason))
            
            conn.commit()
            conn.close()
            logging.info(f"已更新用户 {user_id} 的{source_type}兴趣: {[i.topic for i in new_interests]}")
            
        except Exception as e:
            logging.error(f"更新用户兴趣失败: {e}")

    def assess_skill_level(self, skill_text: str, context_text: str = "") -> Tuple[int, str]:
        """
        智能评估技能等级 (1-10) 和类别 - 重构版本，真正的差异化评分
        
        Args:
            skill_text: 技能名称和描述
            context_text: 上下文信息（工作描述、项目经验等）
        
        Returns:
            (技能等级, 技能类别)
        """
        skill_lower = skill_text.lower()
        context_lower = context_text.lower()
        combined_text = f"{skill_lower} {context_lower}"
        
        # 基础等级评估 - 大幅降低基础分
        base_level = 1  # 从1分开始，真正保守
        
        # 1. 只有明确的技能等级关键词才能提升基础分
        explicit_level_found = False
        for indicator, score in self.skill_level_indicators.items():
            if indicator in combined_text:
                base_level = max(base_level, min(4, score - 3))  # 大幅降低基础评分
                explicit_level_found = True
                break
        
        # 如果没有明确等级指标，保持1分
        if not explicit_level_found:
            base_level = 1
        
        # 2. 技能复杂度评估 - 只给少量加分
        complexity_bonus = 0
        high_complexity_skills = [
            'kubernetes', 'tensorflow', 'pytorch', 'scala', 'rust', 'golang',
            'microservices', 'distributed', 'machine learning', 'deep learning',
            'blockchain', 'devops', 'architecture', '架构', '机器学习', '深度学习'
        ]
        
        if any(skill in skill_lower for skill in high_complexity_skills):
            complexity_bonus = 1  # 高复杂度技能只加1分
        
        # 3. 工作年限评估 - 非常保守
        year_bonus = 0
        import re
        year_patterns = [r'(\d+)\+?\s*年', r'(\d+)\+?\s*years?', r'over\s+(\d+)\s+years?']
        
        for pattern in year_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                try:
                    years = int(matches[0])
                    if years >= 8:
                        year_bonus = 2  # 8年以上才加2分
                    elif years >= 5:
                        year_bonus = 1  # 5年以上加1分
                    # 5年以下不加分
                    break
                except:
                    continue
        
        # 4. 高级职责评估 - 非常严格
        senior_bonus = 0
        senior_keywords = ['architect', 'lead', 'principal', '架构师', '技术负责人', '首席']
        if any(keyword in combined_text for keyword in senior_keywords):
            senior_bonus = 1  # 高级职责只加1分
        
        # 5. 计算最终等级
        final_level = base_level + complexity_bonus + year_bonus + senior_bonus
        
        # 严格限制最高分数
        final_level = max(1, min(6, final_level))  # 最高只能6分
        
        # 6. 根据技能名称的常见程度进一步调整
        common_skills = ['html', 'css', 'javascript', 'python', 'java', 'sql', 'git']
        if any(skill in skill_lower for skill in common_skills):
            final_level = max(1, min(4, final_level))  # 常见技能最高4分
        
        # 7. 判断技能类别
        category = self._determine_skill_category(skill_lower)
        
        logging.info(f"技能评估详情: {skill_text[:20]}... -> 基础:{base_level}, 复杂度:+{complexity_bonus}, 年限:+{year_bonus}, 职责:+{senior_bonus} = 最终:{final_level}/10")
        
        return final_level, category
    
    def _determine_skill_category(self, skill_lower: str) -> str:
        """确定技能类别"""
        category = 'general'
        max_matches = 0
        
        for cat, skills in self.skill_categories.items():
            matches = sum(1 for skill in skills if skill in skill_lower)
            if matches > max_matches:
                max_matches = matches
                category = cat
        
        # 特殊分类
        if category == 'general':
            if any(keyword in skill_lower for keyword in ['管理', 'management', 'lead', 'scrum', 'agile']):
                category = 'management'
            elif any(keyword in skill_lower for keyword in ['设计', 'design', 'ui', 'ux']):
                category = 'design'
        
        return category

    def enhance_extracted_skills(self, skills: List[Dict], context: str = "") -> List[SkillItem]:
        """
        增强技能提取，添加等级和类别评估
        
        Args:
            skills: 技能列表字典
            context: 上下文信息（简历全文、工作描述等）
        """
        enhanced_skills = []
        
        for skill_dict in skills:
            skill_name = skill_dict.get('skill', '')
            original_level = skill_dict.get('level', '')
            
            # 构建评估文本：技能名 + 等级描述 + 上下文
            assessment_text = f"{skill_name} {original_level}"
            
            # 评估数值等级和类别，传入上下文
            skill_level, skill_category = self.assess_skill_level(assessment_text, context)
            
            # 如果LLM也给了skill_level，进行综合评估（极度保守）
            if 'skill_level' in skill_dict and isinstance(skill_dict['skill_level'], (int, float)):
                llm_level = min(6, max(1, skill_dict['skill_level']))  # LLM评分也限制在1-6
                
                # 极度保守的综合评估：我们的评估为主
                # 权重：我们的评估占90%，LLM占10%
                combined_level = (skill_level * 0.9 + llm_level * 0.1)
                
                # 向下取整，进一步保守
                combined_level = int(combined_level)
                
                # 最终分数：绝对不超过5分
                skill_level = max(1, min(5, combined_level))
                
                logging.info(f"技能综合评估: {skill_name} - 算法:{skill_level}, LLM:{llm_level}, 最终:{skill_level}")
            else:
                # 没有LLM评分时，直接使用我们的评估，但限制更严格
                skill_level = max(1, min(4, skill_level))  # 无LLM时最高4分
                logging.info(f"技能算法评估: {skill_name} - 最终评分:{skill_level}")
            
            enhanced_skill = SkillItem(
                skill=skill_name,
                level=original_level,
                category=skill_dict.get('category', ''),
                skill_level=skill_level,
                skill_category=skill_category
            )
            enhanced_skills.append(enhanced_skill)
        
        return enhanced_skills

    async def extract_from_resume_with_enhancement(self, resume_content: str, user_id: str) -> Dict:
        """增强的简历解析，包含兴趣和技能等级评估 - 并行处理"""
        logging.info(f"开始增强简历分析，用户ID: {user_id}")
        
        # 并行执行技能提取和兴趣提取
        import asyncio
        
        # 定义并行任务
        async def extract_skills_task():
            """技能提取任务"""
            try:
                system_msg = "你是专业的技能分析师。请从简历中提取技能信息并评估等级。"
                prompt = f"""
请分析以下简历内容，提取技能信息并评估等级（1-10分，保守评分）：

{resume_content}

请返回JSON格式，包含技能列表：
{{
  "skills": [
    {{"skill": "技能名称", "level": "等级描述", "skill_level": 数值1-10, "category": "技能类别"}}
  ]
}}
"""
                result = await self.llm.call_llm(prompt, system_message=system_msg)
                # 清理JSON
                result = result.strip()
                if result.startswith("```json"):
                    result = result[7:]
                if result.startswith("```"):
                    result = result[3:]
                if result.endswith("```"):
                    result = result[:-3]
                result = result.strip()
                
                import json
                parsed = json.loads(result)
                return parsed.get('skills', [])
            except Exception as e:
                logging.error(f"技能提取失败: {e}")
                return []
        
        async def extract_interests_task():
            """兴趣提取任务"""
            try:
                # 使用现有的兴趣提取方法（但不保存到数据库）
                interests_data = []
                resume_lower = resume_content.lower()
                
                logging.info(f"开始从简历中提取兴趣，简历长度: {len(resume_content)} 字符")
                logging.info(f"简历内容前300字符: {resume_content[:300]}...")
                logging.info(f"兴趣关键词库有 {len(self.interest_keywords)} 个兴趣领域")
                
                for topic, config in self.interest_keywords.items():
                    # 检查是否匹配关键词
                    matches = [keyword for keyword in config['keywords'] if keyword.lower() in resume_lower]
                    match_count = len(matches)
                    
                    logging.debug(f"检查兴趣领域 '{topic}': 匹配 {match_count} 个关键词: {matches}")
                    
                    if match_count > 0:
                        # 简历兴趣等级基于匹配程度，但相对保守
                        interest_level = min(8, config['base_level'] + match_count)  # 最高8级
                        
                        interest = InterestItem(
                            topic=topic,
                            category=config['category'],
                            weight=0.8 + (match_count * 0.1),  # 简历权重稍高但不会过高
                            interest_level=interest_level,
                            search_count=0,  # 简历兴趣不计搜索次数
                            reason=f"基于简历内容分析得出，匹配度: {match_count}，匹配关键词: {matches}",
                            last_updated=datetime.now().isoformat()
                        )
                        interests_data.append(interest)
                        logging.info(f"✅ 提取到兴趣: {topic} (等级:{interest_level}, 匹配:{matches})")
                
                logging.info(f"兴趣提取完成，共提取到 {len(interests_data)} 项兴趣")
                return interests_data
            except Exception as e:
                logging.error(f"兴趣提取失败: {e}", exc_info=True)
                return []
        
        # 并行执行两个任务
        try:
            skills_data, interests_data = await asyncio.gather(
                extract_skills_task(),
                extract_interests_task(),
                return_exceptions=True
            )
            
            # 处理可能的异常
            if isinstance(skills_data, Exception):
                logging.error(f"技能提取任务失败: {skills_data}")
                skills_data = []
            if isinstance(interests_data, Exception):
                logging.error(f"兴趣提取任务失败: {interests_data}")
                interests_data = []
                
        except Exception as e:
            logging.error(f"并行任务执行失败: {e}")
            skills_data = []
            interests_data = []
        
        # 增强技能评估
        enhanced_skills = []
        if skills_data:
            enhanced_skills = self.enhance_extracted_skills(skills_data, resume_content)
        
        # 保存兴趣到数据库（标记为简历来源）
        if interests_data:
            try:
                self._update_user_interests(user_id, interests_data, "resume")
                logging.info(f"✅ 已成功保存 {len(interests_data)} 项简历兴趣到数据库")
                for interest in interests_data:
                    logging.info(f"  - {interest.topic} (等级: {interest.interest_level}, 类别: {interest.category})")
            except Exception as e:
                logging.error(f"❌ 保存简历兴趣到数据库失败: {e}", exc_info=True)
        else:
            logging.warning("⚠️ 未从简历中提取到任何兴趣")
        
        # 构建返回数据
        profile_data = {
            'enhanced_skills': [
                {
                    'skill': s.skill,
                    'level': s.level,
                    'category': s.category,
                    'skill_level': s.skill_level,
                    'skill_category': s.skill_category
                }
                for s in enhanced_skills
            ],
            'extracted_interests': [
                {
                    'topic': i.topic,
                    'category': i.category,
                    'interest_level': i.interest_level,
                    'reason': i.reason,
                    'source_type': 'resume'
                }
                for i in interests_data
            ]
        }
        
        logging.info(f"并行简历分析完成 - 技能: {len(enhanced_skills)}, 兴趣: {len(interests_data)}")
        return profile_data

    async def extract_user_profile(self, resume_content: str) -> Dict:
        """基础的用户画像提取方法"""
        try:
            # 使用LLM分析简历内容
            system_msg = "你是一个专业的简历分析助手。请分析简历内容并提取关键信息。"
            prompt = f"""
请分析以下简历内容，提取并返回JSON格式的信息：

{resume_content}

请提取：
1. 基本信息：姓名、职业、联系方式
2. 技能列表：技能名称、熟练程度、所属类别
3. 工作经历：公司、职位、时间、描述
4. 教育背景：学校、专业、学位、时间

返回JSON格式，确保数据结构清晰。
"""
            
            response = await self.llm.call_llm(prompt, system_message=system_msg)
            
            # 解析LLM返回的JSON
            import json
            try:
                profile_data = json.loads(response)
                return profile_data
            except json.JSONDecodeError:
                # 如果LLM返回的不是有效JSON，尝试提取
                logging.warning("LLM返回的不是有效JSON，使用默认结构")
                return {
                    "name": "未知",
                    "occupation": "未知",
                    "skills": [],
                    "education": [],
                    "work_experience": []
                }
                
        except Exception as e:
            logging.error(f"简历分析失败: {e}")
            return {}

class UserProfileManager:
    """用户画像管理器，负责创建、更新和存储用户画像"""

    def __init__(self, client=None):
        """初始化用户画像管理器"""
        # 统一配置与LLM接口
        self.config = Config()
        if not self.config.llm_api_key or self.config.llm_api_key.strip() == "":
            logging.warning("LLM API密钥未设置，用户画像分析功能将受限。请在.env文件中设置DEEPSEEK_API_KEY")
            self.is_mock_mode = True
            self.llm = None
        else:
            self.is_mock_mode = False
            self.llm = LLMInterface()
        
        self.interest_categories = self._load_interest_categories()
        logging.info("用户画像管理器初始化完成")

    def _call_llm_sync(self, prompt: str, system_message: str = None) -> str:
        """在同步环境中调用异步LLM接口。"""
        import asyncio
        if not self.llm:
            raise RuntimeError("LLM 未初始化")
        coro = self.llm.call_llm(prompt, system_message=system_message)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中，使用线程安全方式等待结果
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                return fut.result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            # 无事件循环的环境
            return asyncio.run(coro)

    def _load_interest_categories(self):
        """加载预定义的兴趣分类体系"""
        categories_file = os.path.join(self.config.data_dir, "interest_categories.json")

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
        categories_file = os.path.join(self.config.data_dir, "interest_categories.json")
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
                            "INSERT INTO user_skills (user_id, skill, level, category, skill_level, skill_category, source_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (user_id, skill, "未知", "通用", 1, "general", "manual")
                        )
                    elif isinstance(skill, dict):
                        # 如果技能是字典，提取相关字段（包括新的等级字段）
                        # 检查表结构，动态插入字段
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA table_info(user_skills)")
                        columns = [row[1] for row in cursor.fetchall()]
                        
                        if "skill_level" in columns and "skill_category" in columns and "source_type" in columns:
                            # 新表结构，包含增强字段
                            conn.execute(
                                """INSERT INTO user_skills 
                                   (user_id, skill, level, category, skill_level, skill_category, source_type)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (user_id, skill.get("skill", ""), skill.get("level", ""), 
                                 skill.get("category", ""), skill.get("skill_level", 1), 
                                 skill.get("skill_category", "general"), skill.get("source_type", "resume"))
                            )
                        else:
                            # 旧表结构
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

    def update_user_interests_from_resume(self, user_id: str, interests: List[str]) -> Dict:
        """
        从简历更新用户兴趣标签，避免叠加，只在不存在时添加
        
        Args:
            user_id: 用户ID
            interests: 兴趣标签列表
            
        Returns:
            更新后的用户画像
        """
        logging.info(f"从简历更新用户 {user_id} 的兴趣标签: {interests}")
        
        conn = get_db_connection()
        try:
            # 检查用户是否存在
            user = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if not user:
                logging.error(f"尝试更新不存在的用户: {user_id}")
                return {"error": f"用户 {user_id} 不存在"}
                
            # 清除该用户所有简历来源的兴趣，避免重复
            conn.execute(
                "DELETE FROM user_interests WHERE user_id = ? AND source_type = 'resume'",
                (user_id,)
            )
            
            # 添加新的兴趣（仅简历来源）
            for interest in interests:
                # 尝试确定类别
                category = "未分类"
                for cat, topics in self.interest_categories.items():
                    if any(t.lower() in interest.lower() for t in topics):
                        category = cat
                        break
                        
                # 插入新兴趣，标记为简历来源
                conn.execute(
                    """INSERT INTO user_interests 
                       (user_id, topic, category, weight, source_type, interest_level, search_count, timestamp)
                       VALUES (?, ?, ?, ?, 'resume', ?, 0, datetime('now'))""",
                    (user_id, interest, category, 3.0, 3)  # 简历兴趣默认权重3.0，等级3
                )
                    
            conn.commit()
            logging.info(f"从简历更新用户兴趣成功: {user_id}")
            
            # 返回更新后的用户画像
            return self.get_user_profile(user_id)
            
        except Exception as e:
            logging.error(f"从简历更新用户兴趣时出错: {e}")
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
            system_msg = "你是一个专业的简历分析助手，擅长提取简历中的技能信息并进行分类和评估。请只返回JSON格式的结果，不要添加任何其他标记。"
            skills_text = self._call_llm_sync(prompt, system_message=system_msg)

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
            system_msg = "你是一个专业的用户兴趣分析师，擅长从个人资料中推断用户的专业兴趣和研究领域。请只返回JSON格式的结果。"
            interests_text = self._call_llm_sync(prompt, system_message=system_msg)

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

            current_time = datetime.now()

            for topic, weight, last_updated_str in interests:
                if last_updated_str:
                    # 解析上次更新时间
                    last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
                    
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
                search_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
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
            system_msg = "你是一个专业的内容推荐系统，负责根据用户兴趣生成有价值的主题推荐。请仅返回JSON。"
            recommendations_text = self._call_llm_sync(prompt, system_message=system_msg)
            
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