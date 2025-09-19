"""
知识流模块
KnowlEdge核心类，整合用户画像、搜索功能和报告生成功能
"""
import logging
import asyncio
import json
import os
import sys
import re
import requests
from typing import Dict, List, Optional, Any, Tuple, AsyncIterator, Union

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 修复导入路径问题
try:
    # 当在项目根目录运行时
    from src.config import Config
    from src.utils import setup_logging
    from src.models.user_profile import UserProfileManager, EnhancedUserProfileExtractor
    from src.models.resume_reader import ResumeReader
    from src.core.search import (
        SearchEngine, 
        GoogleScholarSearch,
        ArxivSearch,
        PatentSearch,
        WebSearch,
        NewsSearch,
        SearchManager
    )
    from src.core.generators import ReportGenerator
    from src.core.llm_interface import LLMInterface
    from src.core.reference_formatter import ReferenceFormatter
except ImportError:
    # 当在src目录下运行时
    from config import Config
    from utils import setup_logging
    from models.user_profile import UserProfileManager, EnhancedUserProfileExtractor
    from models.resume_reader import ResumeReader
    from core.search import (
        SearchEngine, 
        GoogleScholarSearch,
        ArxivSearch,
        PatentSearch,
        WebSearch,
        NewsSearch,
        SearchManager
    )
    from core.generators import ReportGenerator
    from core.llm_interface import LLMInterface
    from core.reference_formatter import ReferenceFormatter

# 设置日志
setup_logging()

class KnowledgeFlow:
    """知识流程管理类"""
    
    def __init__(self, config_path: Optional[str] = None, llm_model: Optional[str] = None):
        """初始化知识流程实例"""
        # 加载配置
        self.config = Config(config_path)
        
        # 初始化各组件
        self.llm = LLMInterface(model_override=llm_model)
        
        self.profile_manager = UserProfileManager()
        self.enhanced_profile_extractor = EnhancedUserProfileExtractor(self.llm)
        self.resume_reader = ResumeReader()
        self.search_engine = SearchEngine(self.config)
        self.report_generator = ReportGenerator(self.llm)
        self.search_manager = SearchManager()
        self.reference_formatter = ReferenceFormatter()
        
        # 可用搜索平台映射
        self.search_platforms = {
            "google_scholar": GoogleScholarSearch,
            "arxiv": ArxivSearch,
            "patent": PatentSearch,
            "web": WebSearch,
            "news": NewsSearch
        }
        
        # 系统状态
        self.is_initialized = True
        logging.info("KnowledgeFlow初始化完成")
        
    async def get_or_create_user_profile(self, user_id: str, username: str = None) -> Dict:
        """获取或创建用户画像"""
        profile = self.profile_manager.get_user_profile(user_id)
        if not profile:
            logging.info(f"为用户 {user_id} 创建新画像")
            self.profile_manager.create_user_profile(user_id, username or f"用户_{user_id}")
            profile = self.profile_manager.get_user_profile(user_id)
        return profile
        
    async def update_user_interests(self, user_id: str, interests: List[str]) -> Dict:
        """更新用户兴趣标签"""
        return self.profile_manager.update_user_interests(user_id, interests)
        
    async def analyze_resume(self, user_id: str, file_path: str) -> Dict:
        """分析简历文件更新用户画像（使用增强的提取器）"""
        logging.info(f"开始为用户 {user_id} 分析简历")
        
        # 读取简历文本
        resume_text = self.resume_reader.read_resume(file_path)
        if not resume_text:
            return {"success": False, "message": "无法读取简历文件"}
            
        # 使用增强的简历分析
        try:
            enhanced_profile = await self.enhanced_profile_extractor.extract_from_resume_with_enhancement(resume_text, user_id)
            if enhanced_profile:
                logging.info(f"增强简历分析完成，提取到技能数量: {len(enhanced_profile.get('enhanced_skills', []))}")
                
                # 保存增强的技能信息到数据库
                if 'enhanced_skills' in enhanced_profile:
                    self.profile_manager.update_user_profile(
                        user_id,
                        skills=enhanced_profile['enhanced_skills']
                    )
                    logging.info(f"已保存 {len(enhanced_profile['enhanced_skills'])} 项增强技能")
                
                # 注意：兴趣已经在extract_from_resume_with_enhancement中通过extract_interests_from_resume保存
                # 不需要重复处理，避免叠加问题
                
                return {"success": True, "message": "简历分析完成", "profile": enhanced_profile}
        except Exception as e:
            logging.error(f"增强简历分析失败: {e}")
            
        # 回退到原始LLM分析
        system_msg = (
            "你是专业的简历分析师。请分析简历内容，提取技能信息时需要评估技能等级。"
        )
        prompt = f"""
请分析下列简历文本，提取信息并评估技能等级（1-10分，保守评分）：

简历文本：
{resume_text}

请返回JSON格式，包含：
1. 教育背景
2. 工作经历  
3. 技能列表（包含技能名称、等级描述、数值等级1-10）
4. 研究兴趣

JSON格式：
{{
  "education": [
    {{"institution": "", "major": "", "degree": "", "time": ""}}
  ],
  "work_experience": [
    {{"company": "", "position": "", "time": "", "description": ""}}
  ],
  "skills": [
    {{"skill": "技能名称", "level": "等级描述", "skill_level": 数值, "category": "类别"}}
  ],
  "research_interests": [],
  "keywords": []
}}
"""
 
        try:
            def _strip_fences(s: str) -> str:
                t = s.strip()
                if t.startswith("```json"):
                    t = t[7:]
                if t.startswith("```"):
                    t = t[3:]
                if t.endswith("```"):
                    t = t[:-3]
                return t.strip()

            def _extract_json_fragment(s: str) -> Optional[str]:
                # 粗略提取第一个花括号或方括号包裹的JSON片段
                import re as _re
                for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
                    m = _re.search(pattern, s)
                    if m:
                        return m.group(0)
                return None

            result = await self.llm.call_llm(prompt, system_message=system_msg)
            cleaned = _strip_fences(result)
            try:
                parsed_result = json.loads(cleaned)
            except Exception:
                frag = _extract_json_fragment(cleaned)
                if not frag:
                    # 进行一次重试，请求仅返回JSON
                    retry_prompt = (
                        "仅返回JSON（无任何解释与前后缀），字段为 education, work_experience, skills, research_interests, keywords。\n"
                        "若缺失信息请给空数组/空字符串。\n\n"
                        f"简历文本：\n{resume_text}"
                    )
                    retry = await self.llm.call_llm(retry_prompt, system_message=system_msg)
                    cleaned_retry = _strip_fences(retry)
                    frag = _extract_json_fragment(cleaned_retry) or cleaned_retry
                parsed_result = json.loads(frag)
             
            # 提取兴趣标签 - 避免叠加，使用新的逻辑
            interests = []
            if "research_interests" in parsed_result:
                interests.extend(parsed_result["research_interests"])
            if "keywords" in parsed_result:
                interests.extend(parsed_result["keywords"])
             
            # 更新用户画像
            self.profile_manager.update_user_profile(
                user_id,
                education=parsed_result.get("education", []),
                work_experience=parsed_result.get("work_experience", []),
                skills=parsed_result.get("skills", [])
            )
             
            # 更新兴趣标签 - 使用新的不叠加逻辑
            if interests:
                # 去重并清洗
                uniq_interests = []
                seen = set()
                for it in interests:
                    if not isinstance(it, str):
                        continue
                    key = it.strip()
                    if key and key not in seen:
                        uniq_interests.append(key)
                        seen.add(key)
                if uniq_interests:
                    # 使用新的不叠加更新逻辑
                    self.profile_manager.update_user_interests_from_resume(user_id, uniq_interests)
                 
            return {
                "success": True, 
                "profile": self.profile_manager.get_user_profile(user_id),
                "message": "简历分析完成并更新用户画像"
            }
             
        except Exception as e:
            logging.error(f"简历分析失败: {str(e)}")
            return {"success": False, "message": f"简历分析失败: {str(e)}"}
            
    async def search_multiple_platforms(
        self, 
        query: str, 
        platforms: List[str], 
        user_id: Optional[str], 
        max_results: Union[int, Dict[str, int]],
        sort_by: str = "relevance",
        time_range: Optional[Dict] = None,
        categories: Optional[List] = None
    ) -> Dict:
        """在多个平台上执行搜索（并发）"""
        results: Dict[str, Any] = {}
        platform_engines = {
            "web": WebSearch(self.config),
            "arxiv": ArxivSearch(self.config),
            "google_scholar": GoogleScholarSearch(self.config),
            "patent": PatentSearch(self.config),
            "news": NewsSearch(self.config)
        }

        async def run_one(p: str):
            if p not in platform_engines:
                return p, {"error": f"不支持的平台: {p}", "results": []}
            engine = platform_engines[p]
            try:
                platform_max = max_results[p] if isinstance(max_results, dict) else max_results
                if p == "arxiv":
                    data = await engine.search(query, platform_max, sort_by=sort_by, time_range=time_range, categories=categories)
                else:
                    data = await engine.search(query, platform_max)
                logging.info(f"{p} 搜索完成，找到 {data.get('result_count', len(data.get('results', [])))} 条结果")
                return p, data
            except Exception as e:
                logging.error(f"{p} 搜索失败: {e}")
                return p, {"error": str(e), "query": query, "results": []}

        coros = [run_one(p) for p in platforms]
        done = await asyncio.gather(*coros)
        for p, data in done:
            results[p] = data

        if user_id:
            self.profile_manager.add_search_history(user_id, query, platforms)
        return results
        
    async def generate_report(
        self, 
        search_results: Dict, 
        user_input: Dict,
        report_type: str = "standard"
    ) -> str:
        """根据搜索结果生成报告"""
        logging.info(f"生成报告类型: {report_type}")
        
        original_query = user_input.get("query", "")
        
        if report_type == "literature_review":
            report = await self.report_generator.generate_literature_review(
                search_results, original_query
            )
        elif report_type == "corporate_research":
            report = await self.report_generator.generate_corporate_research_report(
                search_results, user_input, original_query
            )
        elif report_type == "popular_science":
            report = await self.report_generator.generate_popular_science_report(
                search_results, user_input, original_query
            )
        else:  # 默认标准报告
            report = await self.report_generator.generate_report(search_results)
            
        return report
        
    async def generate_report_stream(
        self, 
        search_results: Dict, 
        user_input: Dict,
        report_type: str = "standard"
    ) -> AsyncIterator[str]:
        """流式生成报告"""
        logging.info(f"流式生成报告类型: {report_type}")
        
        original_query = user_input.get("query", "")
        
        if report_type == "literature_review":
            # 使用增强版文献综述生成
            async for chunk in self.report_generator.generate_enhanced_literature_review_stream(
                search_results, original_query
            ):
                yield chunk
        elif report_type == "corporate_research":
            async for chunk in self.report_generator.generate_corporate_research_report_stream(
                search_results, user_input, original_query
            ):
                yield chunk
        elif report_type == "popular_science":
            async for chunk in self.report_generator.generate_popular_science_report_stream(
                search_results, user_input, original_query
            ):
                yield chunk
        else:  # 默认标准报告
            async for chunk in self.report_generator.generate_enhanced_standard_report_stream(search_results, user_input):
                yield chunk
    
    async def generate_enhanced_literature_review(self, query: str, platform: str = "arxiv", 
                                         num_results: int = 8) -> str:
        """
        使用多模型并行处理生成增强版文献综述
        
        Args:
            query: 用户查询
            platform: 搜索平台
            num_results: 结果数量
            
        Returns:
            生成的增强版文献综述
        """
        logging.info(f"使用多模型生成增强版文献综述: '{query}', 平台: {platform}, 结果数量: {num_results}")
        
        # 执行搜索，获取更多结果用于文献综述
        search_results = await self.search_manager.search(query, platform, num_results)
        
        # 使用增强版文献综述生成器
        report_chunks = []
        async for chunk in self.report_generator.generate_enhanced_literature_review_stream(search_results, query):
            report_chunks.append(chunk)
        
        return "".join(report_chunks)
        
    async def generate_enhanced_literature_review_stream(self, query: str, platform: str = "arxiv", 
                                               num_results: int = 8) -> AsyncIterator[str]:
        """
        流式生成增强版文献综述
        
        Args:
            query: 用户查询
            platform: 搜索平台 
            num_results: 结果数量
            
        Yields:
            流式的文献综述内容
        """
        logging.info(f"流式生成增强版文献综述: '{query}', 平台: {platform}, 结果数量: {num_results}")
        
        # 执行搜索
        search_results = await self.search_manager.search(query, platform, num_results)
        
        # 统计检索结果
        found_count = 0
        for source, data in search_results.items():
            if isinstance(data, dict) and 'results' in data:
                found_count += len(data['results'])
        
        yield f"🔍 **搜索完成**：在 {platform} 平台检索到 {found_count} 篇相关文献\n\n"
        yield "📖 **开始生成增强版文献综述**：确保所有文献都被分析并保持内容一致性...\n\n"
        
        # 使用增强版生成器
        async for chunk in self.report_generator.generate_enhanced_literature_review_stream(search_results, query):
            yield chunk
        
    async def personalized_search(
        self, 
        user_id: str, 
        query: str, 
        platforms: List[str],
        max_results: Union[int, Dict[str, int]],
        sort_by: str = "relevance",
        time_range: Optional[Dict] = None,
        categories: Optional[List] = None
    ) -> Dict:
        """个性化搜索：优化查询并执行搜索"""
        logging.info(f"记录用户 {user_id} 的搜索: {query}")
        
        # 基于搜索内容分析和更新用户兴趣和技能
        try:
            if user_id and query:
                # 提取兴趣
                interests = self.enhanced_profile_extractor.extract_interests_from_search(query, user_id)
                if interests:
                    logging.info(f"为用户 {user_id} 提取到 {len(interests)} 个搜索兴趣: {[i.topic for i in interests]}")
                
                # 提取技能
                skills = self.enhanced_profile_extractor.extract_skills_from_search(query, user_id)
                if skills:
                    logging.info(f"为用户 {user_id} 提取到 {len(skills)} 个搜索技能: {[s.skill for s in skills]}")
        except Exception as e:
            logging.error(f"用户画像分析失败: {e}")
        
        profile = self.profile_manager.get_user_profile(user_id)
        
        # 使用LLM优化查询
        enhanced_query = await self._optimize_query(query, profile)
        
        # 这里统一用 enhanced_query 执行和记录
        logging.info(f"在平台 {', '.join(platforms)} 上搜索(已优化): '{enhanced_query}'")
        search_results = await self.search_multiple_platforms(
            enhanced_query, platforms, user_id, max_results, sort_by, time_range, categories
        )
        
        return {
            "results": search_results,
            "original_query": query,
            "enhanced_query": enhanced_query,
            "sort_by": sort_by,
            "time_range": time_range,
            "categories": categories,
            "platform_type": ", ".join(platforms)
        }
        
    async def _optimize_query(self, query: str, profile: dict = None) -> str:
        """使用LLM优化查询，使其更适合学术搜索"""
        # 检测是否包含中文字符
        if re.search(r'[\u4e00-\u9fff]', query):
            # 如果查询是对话式请求，转换为适合学术搜索的形式
            if any(phrase in query.lower() for phrase in [
                "帮我", "请", "我想", "能否", "可以", "希望", "麻烦", "告诉我", "跟踪"
            ]):
                prompt = f"""
                请将以下对话式查询翻译为简洁的英文学术搜索关键词，保留核心主题和概念，去除对话性质的词语。
                查询: {query}
                只返回转换后的英文关键词。
                """
                try:
                    enhanced_query = await self.llm.call_llm(prompt)
                    enhanced_query = enhanced_query.strip('" \n\t')
                    if re.search(r'[\u4e00-\u9fff]', enhanced_query):
                        enhanced_query = await self._traditional_translate(query)
                    return enhanced_query
                except Exception:
                    return await self._traditional_translate(query)
            else:
                return await self._traditional_translate(query)
        return query
        
    async def _traditional_translate(self, text: str, source_lang="zh-CN", target_lang="en") -> str:
        """使用传统翻译API翻译文本"""
        try:
            translate_api_url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": source_lang,
                "tl": target_lang,
                "dt": "t",
                "q": text
            }
            response = await asyncio.to_thread(
                requests.get, 
                translate_api_url, 
                params=params, 
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                if result and isinstance(result, list) and len(result) > 0:
                    translations = []
                    for sentence in result[0]:
                        if sentence and isinstance(sentence, list) and len(sentence) > 0:
                            translations.append(sentence[0])
                    translated = " ".join(translations)
                    logging.info(f"传统API翻译: '{text}' -> '{translated}'")
                    return translated
            return self._simple_translate(text)
        except Exception as e:
            logging.error(f"传统翻译API调用失败: {e}")
            return self._simple_translate(text)
        
    def _simple_translate(self, query: str) -> str:
        """简单的中文关键词映射"""
        translations = {
            "深度学习": "deep learning",
            "机器学习": "machine learning",
            "人工智能": "artificial intelligence",
            "神经网络": "neural network",
            "卷积神经网络": "convolutional neural network CNN",
            "自然语言处理": "natural language processing NLP",
            "计算机视觉": "computer vision",
            "强化学习": "reinforcement learning",
            "生成对抗网络": "generative adversarial network GAN",
            "迁移学习": "transfer learning",
            "注意力机制": "attention mechanism",
            "图神经网络": "graph neural network",
            "知识图谱": "knowledge graph",
            "语义分割": "semantic segmentation",
            "目标检测": "object detection",
            "图像分类": "image classification",
            "语音识别": "speech recognition",
            "推荐系统": "recommendation system",
            "情感分析": "sentiment analysis",
            "聚类": "clustering",
            "分类": "classification",
            "回归": "regression",
            "优化": "optimization",
            "医学影像": "medical imaging",
            "自动驾驶": "autonomous driving",
            "大语言模型": "large language model LLM",
            "变换器": "transformer",
            "预训练": "pre-training",
            "微调": "fine-tuning",
            "嵌入": "embedding",
            "向量化": "vectorization",
            "多模态": "multimodal",
            "数据增强": "data augmentation",
            "过拟合": "overfitting",
            "正则化": "regularization"
        }
        translated_query = query
        for zh, en in translations.items():
            if zh in query:
                translated_query = translated_query.replace(zh, en)
        if translated_query == query:
            english_terms = re.findall(r'[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)*', query)
            if english_terms:
                return " ".join(english_terms)
            else:
                return "recent research papers"
        logging.info(f"简单映射翻译: '{query}' -> '{translated_query}'")
        return translated_query

    async def generate_literature_review_stream(
        self, 
        search_results: Dict, 
        original_query: str
    ) -> AsyncIterator[str]:
        """流式生成文献综述"""
        logging.info(f"流式生成文献综述: {original_query}")
        async for chunk in self.report_generator.generate_literature_review_stream(
            search_results, original_query
        ):
            yield chunk

    async def generate_corporate_research_report_stream(
        self, 
        search_results: Dict,
        user_input: Dict,
        original_query: str
    ) -> AsyncIterator[str]:
        """流式生成企业调研报告"""
        logging.info(f"流式生成企业调研报告: {original_query}")
        async for chunk in self.report_generator.generate_corporate_research_report_stream(
            search_results, user_input, original_query
        ):
            yield chunk

    async def generate_popular_science_report_stream(
        self, 
        search_results: Dict,
        user_input: Dict,
        original_query: str
    ) -> AsyncIterator[str]:
        """流式生成科普知识报告"""
        logging.info(f"流式生成科普知识报告: {original_query}")
        async for chunk in self.report_generator.generate_popular_science_report_stream(
            search_results, user_input, original_query
        ):
            yield chunk

    async def process_query(self, query: str, platform: str = "arxiv", 
                     num_results: int = 5, report_type: str = "standard") -> str:
        """
        处理用户查询，执行搜索并生成报告
        """
        logging.info(f"处理查询: '{query}', 平台: {platform}, 结果数量: {num_results}, 报告类型: {report_type}")
        
        search_results = await self.search_manager.search(query, platform, num_results)
        
        references = self.reference_formatter.extract_references(search_results)
        logging.info(f"提取到 {len(references)} 条参考文献")
        
        if report_type == "literature_review":
            report = await self.report_generator.generate_literature_review(search_results, query)
        elif report_type == "corporate_research":
            report = await self.report_generator.generate_corporate_research_report(search_results, {"query": query}, query)
        elif report_type == "popular_science":
            report = await self.report_generator.generate_popular_science_report(search_results, {"query": query}, query)
        else:
            report = await self.report_generator.generate_report(search_results)
        return report
        
    async def process_query_stream(self, query: str, platform: str = "arxiv", 
                           num_results: int = 5, report_type: str = "standard") -> AsyncIterator[str]:
        """
        流式处理用户查询，执行搜索并生成报告
        """
        logging.info(f"流式处理查询: '{query}', 平台: {platform}, 结果数量: {num_results}, 报告类型: {report_type}")
        
        yield f"正在搜索 '{query}'，平台: {platform}，请稍候...\n\n"
        
        search_results = await self.search_manager.search(query, platform, num_results)
        
        found_count = 0
        for source, data in search_results.items():
            if isinstance(data, dict) and 'results' in data:
                found_count += len(data['results'])
        
        yield f"已找到 {found_count} 条相关结果，正在生成报告...\n\n"
        
        user_input = {"query": query}
        if report_type == "literature_review":
            async for chunk in self.report_generator.generate_literature_review_stream(search_results, query):
                yield chunk
        elif report_type == "corporate_research":
            async for chunk in self.report_generator.generate_corporate_research_report_stream(search_results, user_input, query):
                yield chunk
        elif report_type == "popular_science":
            async for chunk in self.report_generator.generate_popular_science_report_stream(search_results, user_input, query):
                yield chunk
        else:
            async for chunk in self.report_generator.generate_report_stream(search_results, user_input):
                yield chunk
    
    async def generate_enhanced_literature_review(self, query: str, platform: str = "arxiv", 
                                         num_results: int = 8) -> str:
        """
        使用多模型并行处理生成增强版文献综述
        
        Args:
            query: 用户查询
            platform: 搜索平台
            num_results: 结果数量
            
        Returns:
            生成的增强版文献综述
        """
        logging.info(f"使用多模型生成增强版文献综述: '{query}', 平台: {platform}, 结果数量: {num_results}")
        
        # 执行搜索，获取更多结果用于文献综述
        search_results = await self.search_manager.search(query, platform, num_results)
        
        # 使用增强版文献综述生成器
        report_chunks = []
        async for chunk in self.report_generator.generate_enhanced_literature_review_stream(search_results, query):
            report_chunks.append(chunk)
        
        return "".join(report_chunks)
        
    async def generate_enhanced_literature_review_stream(self, query: str, platform: str = "arxiv", 
                                               num_results: int = 8) -> AsyncIterator[str]:
        """
        流式生成增强版文献综述
        
        Args:
            query: 用户查询
            platform: 搜索平台 
            num_results: 结果数量
            
        Yields:
            流式的文献综述内容
        """
        logging.info(f"流式生成增强版文献综述: '{query}', 平台: {platform}, 结果数量: {num_results}")
        
        # 执行搜索
        search_results = await self.search_manager.search(query, platform, num_results)
        
        # 统计检索结果
        found_count = 0
        for source, data in search_results.items():
            if isinstance(data, dict) and 'results' in data:
                found_count += len(data['results'])
        
        yield f"🔍 **搜索完成**：在 {platform} 平台检索到 {found_count} 篇相关文献\n\n"
        yield "📖 **开始生成增强版文献综述**：确保所有文献都被分析并保持内容一致性...\n\n"
        
        # 使用增强版生成器
        async for chunk in self.report_generator.generate_enhanced_literature_review_stream(search_results, query):
            yield chunk 