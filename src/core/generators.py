"""
报告生成模块
提供各种类型报告生成功能
"""
import logging
import asyncio
from typing import Dict, AsyncIterator, List, Any, Optional
from src.core.llm_interface import LLMInterface
from src.core.reference_formatter import ReferenceFormatter
import hashlib
from collections import OrderedDict

# 简单LRU缓存用于合并结果，减少重复LLM调用
_MERGE_CACHE_MAX_ITEMS = 64
_MERGE_CACHE: OrderedDict[str, str] = OrderedDict()

def _merge_cache_get(key: str) -> Optional[str]:
    try:
        val = _MERGE_CACHE.pop(key)
        _MERGE_CACHE[key] = val
        return val
    except KeyError:
        return None

def _merge_cache_set(key: str, value: str) -> None:
    _MERGE_CACHE[key] = value
    if len(_MERGE_CACHE) > _MERGE_CACHE_MAX_ITEMS:
        _MERGE_CACHE.popitem(last=False)

class ReportGenerator:
    """报告生成器类"""
    
    def __init__(self, llm=None):
        self.llm = llm or LLMInterface()
        self.reference_formatter = ReferenceFormatter()
        
        self.report_templates = {
            "standard": {
                "title": "综合报告",
                "system_message": "你是一个知识渊博且表达能力优秀的AI助手，擅长整理和归纳信息，为用户生成全面、客观、详尽的综合报告。",
                "prompt_template": """
                请基于以下搜索结果，生成一份结构清晰、内容丰富、分析深入的综合报告。
                ---搜索结果---
                {context_str}
                ---搜索结果结束---
                """
            },
            "literature_review": {
                "title": "文献综述报告",
                "system_message": "你是一名严格的学术综述撰写者，输出必须是学术风格、信息密度高、避免空洞套话。不要写引导性或总结性套话，不要复述题目，不要写“本节将…”之类的开场。只写本节需要的内容。",
            },
            "corporate_research": {
                "title": "企业调研报告",
                "system_message": "你是专业的企业调研分析师，擅长从商业角度分析技术发展、市场机会和企业战略。注重实用性、可操作性和商业价值评估。",
                "prompt_template": """
                请基于以下搜索结果，生成一份专业的企业调研报告。报告应从企业经营和商业价值角度分析相关技术或行业发展情况。

                **报告要求：**
                1. 以企业视角分析市场机会和技术应用前景
                2. 重点关注商业价值、投资回报和风险评估
                3. 提供具体的业务建议和战略方向
                4. 分析竞争格局和企业定位机会
                5. 评估技术成熟度和产业化可行性

                **请按以下结构组织报告：**
                - **执行摘要**：核心发现和建议
                - **市场概况**：行业现状和发展趋势
                - **技术分析**：关键技术及其商业应用潜力
                - **竞争分析**：主要参与者和竞争格局  
                - **商业机会**：具体的业务机会和市场切入点
                - **风险评估**：技术风险、市场风险和竞争风险
                - **战略建议**：针对企业的具体行动建议

                ---搜索结果---
                {context_str}
                ---搜索结果结束---

                查询主题：{original_query}
                """
            },
            "popular_science": {
                "title": "科普知识报告",
                "system_message": "你是一位优秀的科普作家和知识传播者，擅长将复杂的学术知识转化为通俗易懂、生动有趣的科普内容。你的写作风格亲切自然，善于用比喻、类比和生活化的例子来解释抽象概念。你有着深厚的学术功底，能够准确理解前沿研究，同时具备出色的文字表达能力，让读者既能学到知识又感到有趣。请确保内容结构完整、格式规范、信息丰富详实，每个章节都要有足够的深度和广度。",
                "prompt_template": """
                请基于以下搜索结果，生成一份生动有趣、内容丰富的科普知识报告。报告应该通俗易懂，让普通读者也能轻松理解复杂的学术概念。

                **科普报告写作要求：**
                1. **内容充实详细**：整篇报告不少于2000字，每个章节内容丰富具体
                2. **语言通俗易懂**：避免过多专业术语，必要时要进行简单解释
                3. **结构清晰完整**：必须包含所有指定章节，每个章节至少3-4段内容
                4. **善用比喻类比**：大量使用生活中常见的事物来解释抽象概念
                5. **增加互动性**：适当提出问题引发读者思考，使用"你知道吗？"等表达
                6. **突出实用性**：详细说明知识与日常生活的联系和应用价值
                7. **保持科学严谨**：确保科学事实的准确性，引用具体的研究和数据

                **严格格式要求（必须遵守）：**
                - 使用标准Markdown格式，不要使用HTML标签
                - 避免生成重复的词句或乱码字符
                - 列表请使用Markdown格式的"-"或"*"，不要使用HTML标签
                - 段落之间用单个空行分隔，保持清晰的结构
                - 所有英文术语请用中文解释或提供中英对照

                **请严格按照以下结构组织科普报告（必须包含所有章节）：**

                ## 🤔 引言：你是否想过...
                用引人入胜的问题或现象引入主题，激发读者兴趣。要求：
                - 从日常生活的观察或有趣现象入手
                - 提出2-3个引人思考的问题
                - 简要预告报告将解答的内容
                - 内容不少于300字
                
                ## 📚 知识背景：让我们先了解基础
                用简单语言介绍必要的背景知识，为后续深入讲解做铺垫。要求：
                - 详细解释相关的基础概念和原理
                - 使用大量类比和比喻让抽象概念变得具体
                - 介绍领域的发展历史和重要里程碑
                - 内容不少于400字
                
                ## 🔍 深入探索：核心内容解析
                详细解释主要概念和原理，这是报告的核心部分。要求：
                - 分层次讲解，从简单到复杂，循序渐进
                - 大量使用生活化的例子和具体场景
                - 适当插入"你知道吗？"、"举个例子"等互动元素
                - 用简单的步骤描述复杂过程（如：首先、然后、最后）
                - 包含具体的数据和研究发现
                - 内容不少于600字
                
                ## 💡 实际应用：这些知识如何改变我们的生活
                详细说明实际应用和影响，让读者了解这些知识的价值。要求：
                - 列举多个具体的应用实例，涵盖不同领域
                - 详细描述对日常生活的影响和改变
                - 说明解决了哪些具体问题，带来了什么便利
                - 包含真实的案例和统计数据
                - 内容不少于400字
                
                ## 🌟 前沿发展：未来会怎样？
                介绍最新进展和发展趋势，展望未来可能性。要求：
                - 详细介绍最新的研究进展和突破
                - 分析未来3-5年可能的发展方向
                - 讨论对社会的潜在影响和变革
                - 包含专家观点和预测
                - 提及相关的挑战和争议
                - 内容不少于400字
                
                ## 🎯 小结：关键要点回顾
                系统总结要点，加深理解。要求：
                - 用简洁明了的语言重申核心概念
                - 总结主要的应用价值和影响
                - 强调记忆要点，帮助读者掌握精髓
                - 内容不少于200字
                
                ## 💭 思考题：拓展你的思维
                提供有深度的延伸思考问题，鼓励读者进一步探索。要求：
                - 设计3-5个开放性思考题
                - 问题要有一定深度，能引发思考和讨论
                - 可以结合时事热点或社会现象
                - 提供思考的方向和建议

                **特别注意：**
                - 多使用表情符号和视觉元素增加可读性
                - 适当加入"你知道吗？"、"有趣的是..."、"据研究表明..."等表达
                - 用具体数字、百分比和统计数据来支撑观点
                - 大量使用比喻，如"就像...一样"、"可以想象成..."、"类似于..."
                - 确保每个章节内容完整详细，不要出现内容截断
                - 在适当位置加入小标题来组织内容结构
                - 使用具体的例子和场景，避免抽象的描述

                ---搜索结果---
                {context_str}
                ---搜索结果结束---

                科普主题：{original_query}
                
                请开始创作这份内容丰富、生动有趣的科普报告！记住要包含所有章节，每个章节都要内容详实，整体篇幅要充实！
                """
            }
        }
        logging.info("报告生成器初始化完成")
    
    def _prepare_context_from_search_results(self, search_results: Dict, max_items_per_source: int = 5, max_chars: int = 12000) -> str:
        """为文献综述等其他报告类型准备上下文（保持原有逻辑）"""
        context_parts = []
        total_chars = 0
        for source, data in search_results.items():
            if data and isinstance(data, dict) and 'results' in data:
                items = data['results'][:max_items_per_source]
                for item in items:
                    title = item.get('title', '')
                    abstract = item.get('abstract', item.get('snippet', ''))
                    if not title or not abstract:
                        continue
                    authors = ", ".join(item.get('authors', []))
                    published = item.get('published', '')
                    journal = item.get('journal', '')
                    citation_count = item.get('citation_count', '')
                    item_text = f"来源: {source}\n标题: {title}\n"
                    if authors: item_text += f"作者: {authors}\n"
                    if published: item_text += f"发表时间: {published}\n"
                    if journal: item_text += f"期刊/会议: {journal}\n"
                    if citation_count: item_text += f"引用次数: {citation_count}\n"
                    item_text += f"摘要: {abstract}\n------\n"
                    if total_chars + len(item_text) > max_chars:
                        remaining = max_chars - total_chars
                        if remaining > 200:
                            context_parts.append(item_text[:remaining] + "...")
                        total_chars = max_chars
                        break
                    context_parts.append(item_text)
                    total_chars += len(item_text)
            if total_chars >= max_chars:
                break
        return "\n".join(context_parts) or "未从搜索结果中提取到足够的上下文信息。"

    def _extract_all_articles(self, search_results: Dict) -> List[Dict[str, Any]]:
        """提取所有检索到的文章，不限制数量"""
        articles = []
        for source, data in search_results.items():
            if data and isinstance(data, dict) and 'results' in data:
                for item in data['results']:
                    title = item.get('title', '').strip()
                    abstract = item.get('abstract', item.get('snippet', '')).strip()
                    
                    # 基本信息验证
                    if not title or not abstract:
                        continue
                    
                    # 构建完整的文章信息
                    article = {
                        'source': source,
                        'title': title,
                        'abstract': abstract,
                        'authors': item.get('authors', []),
                        'published': item.get('published', ''),
                        'journal': item.get('journal', ''),
                        'citation_count': item.get('citation_count', ''),
                        'url': item.get('url', item.get('link', '')),
                        'similarity': item.get('similarity', 0),
                        'categories': item.get('categories', [])
                    }
                    articles.append(article)
        
        logging.info(f"提取到 {len(articles)} 篇文章用于标准报告生成")
        return articles

    def _create_article_analysis_prompt(self, article: Dict[str, Any], index: int, total: int) -> str:
        """为单篇文章创建分析提示词"""
        return f"""
请对以下第{index}/{total}篇文章进行深入解读和分析：

**文章信息：**
- 标题：{article['title']}
- 作者：{', '.join(article['authors']) if article['authors'] else '未知'}
- 发表时间：{article['published'] or '未知'}
- 期刊/会议：{article['journal'] or '未知'}
- 引用次数：{article['citation_count'] or '未知'}
- 来源：{article['source']}

**摘要：**
{article['abstract']}

**分析要求：**
1. **核心贡献**：这篇文章的主要创新点和贡献是什么？
2. **技术方法**：采用了什么技术方法或理论框架？
3. **实验结果**：主要的实验结果或发现是什么？
4. **应用价值**：这项研究的实际应用价值和意义？
5. **研究局限**：存在哪些局限性或未来改进空间？

**输出格式：**
## 第{index}篇：{article['title']}

**基本信息**
- 作者：{', '.join(article['authors']) if article['authors'] else '未知'}
- 发表：{article['published'] or '未知'} | {article['journal'] or '未知'}
- 引用：{article['citation_count'] or '未知'}次

**核心贡献**
[详细分析文章的主要创新点和贡献]

**技术方法**
[分析采用的技术方法和理论框架]

**主要发现**
[总结关键的实验结果和发现]

**应用价值**
[评估研究的实际应用价值]

**研究局限**
[指出存在的局限性和改进方向]

---

请确保分析深入、客观、准确，避免空泛的描述。
"""

    async def _analyze_single_article(self, article: Dict[str, Any], index: int, total: int) -> str:
        """分析单篇文章"""
        try:
            prompt = self._create_article_analysis_prompt(article, index, total)
            system_message = """你是一位资深的学术研究分析专家，擅长深入解读学术论文。
你的任务是对每篇文章进行全面、深入、客观的分析，帮助读者快速理解文章的核心价值。

分析要求：
1. 保持客观中立的学术态度
2. 分析要深入具体，避免泛泛而谈
3. 突出文章的创新点和实际价值
4. 指出研究的局限性和改进空间
5. 使用清晰的结构化输出格式
6. 确保内容准确性和专业性"""

            result = await self.llm.call_llm(prompt, system_message)
            return result.strip()
        except Exception as e:
            logging.error(f"分析第{index}篇文章失败: {e}")
            return f"## 第{index}篇：{article['title']}\n\n**分析失败**：无法完成该文章的深入分析。\n\n---\n"

    async def _parallel_analyze_articles(self, articles: List[Dict[str, Any]], batch_size: int = 5) -> List[str]:
        """并行分析多篇文章"""
        all_analyses = []
        total_articles = len(articles)
        
        # 分批处理避免过多并发
        for i in range(0, total_articles, batch_size):
            batch = articles[i:i + batch_size]
            logging.info(f"正在分析第 {i+1}-{min(i+batch_size, total_articles)} 篇文章（共{total_articles}篇）")
            
            # 并行处理当前批次
            tasks = []
            for j, article in enumerate(batch):
                article_index = i + j + 1
                task = self._analyze_single_article(article, article_index, total_articles)
                tasks.append(task)
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果和异常
                for result in batch_results:
                    if isinstance(result, Exception):
                        logging.error(f"文章分析异常: {result}")
                        all_analyses.append("**分析失败**：该文章分析过程中出现错误。\n\n---\n")
                    else:
                        all_analyses.append(result)
                        
            except Exception as e:
                logging.error(f"批次分析失败: {e}")
                # 为失败的批次添加占位符
                for _ in batch:
                    all_analyses.append("**分析失败**：批次处理过程中出现错误。\n\n---\n")
        
        return all_analyses

    async def _generate_report_summary(self, articles: List[Dict[str, Any]], platform_type: str) -> str:
        """生成报告总结"""
        summary_prompt = f"""
基于以下 {len(articles)} 篇文章的检索结果，生成一个简要的总结报告：

**检索信息：**
- 平台：{platform_type}
- 文章数量：{len(articles)} 篇
- 来源分布：{self._get_source_distribution(articles)}

**总结要求：**
1. 概述检索到的文章主要涵盖哪些研究方向
2. 识别当前研究的热点和趋势
3. 指出主要的技术方法和应用领域
4. 总结整体研究现状和发展方向

请生成一个简洁而全面的总结（200-400字）。
"""
        
        system_message = """你是一位学术研究总结专家，擅长从多篇文章中提取关键信息并形成整体观点。
请基于提供的文章信息生成客观、准确、有洞察力的研究总结。"""
        
        try:
            summary = await self.llm.call_llm(summary_prompt, system_message)
            return summary.strip()
        except Exception as e:
            logging.error(f"生成报告总结失败: {e}")
            return f"**总结生成失败**：无法生成整体总结，但以下是 {len(articles)} 篇文章的详细分析。"

    def _get_source_distribution(self, articles: List[Dict[str, Any]]) -> str:
        """获取来源分布统计"""
        source_counts = {}
        for article in articles:
            source = article.get('source', '未知')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        distribution = []
        for source, count in source_counts.items():
            distribution.append(f"{source}({count}篇)")
        
        return ", ".join(distribution)

    async def _verify_completeness(self, articles: List[Dict[str, Any]], analyses: List[str]) -> Dict[str, Any]:
        """校对功能：验证所有文章都被分析了"""
        total_articles = len(articles)
        total_analyses = len(analyses)
        
        verification_result = {
            "complete": total_articles == total_analyses,
            "total_articles": total_articles,
            "total_analyses": total_analyses,
            "missing_count": max(0, total_articles - total_analyses),
            "extra_count": max(0, total_analyses - total_articles)
        }
        
        if not verification_result["complete"]:
            logging.warning(f"文章分析不完整: 应分析{total_articles}篇，实际分析{total_analyses}篇")
        else:
            logging.info(f"文章分析完整性校验通过: 成功分析{total_articles}篇文章")
        
        return verification_result

    async def generate_enhanced_standard_report_stream(self, search_results: Dict, user_input: Dict) -> AsyncIterator[str]:
        """生成增强版标准报告的流式输出"""
        logging.info("开始生成增强版标准行业报告...")
        
        platform_type = user_input.get('platform_type', '通用平台')
        
        # 提取所有文章
        articles = self._extract_all_articles(search_results)
        
        if not articles:
            yield "# 标准行业报告\n\n"
            yield "**错误**：未找到可分析的文章。请检查搜索结果或调整搜索条件。\n"
            return
        
        # 输出报告头部
        yield f"# 【{platform_type}】标准行业报告\n\n"
        yield f"**报告概览**\n"
        yield f"- 检索平台：{platform_type}\n"
        yield f"- 文章数量：{len(articles)} 篇\n"
        yield f"- 来源分布：{self._get_source_distribution(articles)}\n"
        yield f"- 生成时间：{self._get_current_time()}\n\n"
        
        # 生成总结
        yield "## 📊 研究现状总结\n\n"
        try:
            summary = await self._generate_report_summary(articles, platform_type)
            yield f"{summary}\n\n"
        except Exception as e:
            logging.error(f"生成总结失败: {e}")
            yield "总结生成失败，但以下是详细的文章分析。\n\n"
        
        yield "## 📚 详细文章解读\n\n"
        yield "以下是对每篇检索到的文章的深入分析：\n\n"
        
        # 并行分析所有文章
        try:
            analyses = await self._parallel_analyze_articles(articles)
            
            # 校对验证
            verification = await self._verify_completeness(articles, analyses)
            
            # 输出分析结果
            for i, analysis in enumerate(analyses):
                yield analysis
                yield "\n"
                
                # 每5篇文章后添加进度提示
                if (i + 1) % 5 == 0:
                    remaining = len(analyses) - (i + 1)
                    if remaining > 0:
                        yield f"*已完成 {i + 1} 篇分析，还有 {remaining} 篇...*\n\n"
            
            # 输出校对结果
            yield "\n## ✅ 分析完整性校验\n\n"
            if verification["complete"]:
                yield f"**校验通过**：成功分析了全部 {verification['total_articles']} 篇文章。\n\n"
            else:
                yield f"**校验警告**：应分析 {verification['total_articles']} 篇文章，实际分析 {verification['total_analyses']} 篇。\n"
                if verification["missing_count"] > 0:
                    yield f"缺失 {verification['missing_count']} 篇文章的分析。\n"
                yield "\n"
                
        except Exception as e:
            logging.error(f"文章分析过程失败: {e}")
            yield f"**错误**：文章分析过程中出现问题：{str(e)}\n\n"
        
        # 添加参考文献
        yield "\n## 📖 参考文献\n\n"
        references_section = self.reference_formatter.format_references(search_results, "markdown")
        yield references_section
        
        logging.info("增强版标准行业报告生成完成")

    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def generate_report_stream(self, search_results: Dict, user_input: Dict) -> AsyncIterator[str]:
        """生成标准报告流式输出 - 重定向到增强版本"""
        async for chunk in self.generate_enhanced_standard_report_stream(search_results, user_input):
            yield chunk

    # ===== 文献综述（严格章节化、分批并行、去套话合并） =====
    def _flatten_results(self, search_results: Dict) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for source, data in search_results.items():
            if isinstance(data, dict) and data.get('results'):
                for it in data['results']:
                    e = it.copy()
                    e['source'] = source
                    items.append(e)
        # 去重以减少上下文冗余与后续LLM调用
        return self._deduplicate_items(items)

    def _deduplicate_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于标题/链接做简单去重，优先保留首次出现的条目。"""
        seen_keys = set()
        deduped: List[Dict[str, Any]] = []
        for it in items:
            title = (it.get('title') or '').strip().lower()
            link = (it.get('link') or it.get('url') or '').strip().lower()
            key = link or title
            if not key:
                # 对于无标题无链接的记录直接跳过
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(it)
        return deduped

    def _section_prompt(self, original_query: str, section_name: str, context_str: str) -> str:
        rules = (
            "禁止写开场白、过渡句、总结性套话；"
            "禁止复述“本节/本文/该领域/近年来/取得进展”之类的泛化语句；"
            "不得重复题目或重复上一节内容；"
            "必须引用具体文献（使用其标题或明确标识），指出方法/数据/发现/结果；"
            "内容以事实和比较为主，避免空泛描述；"
            "仅撰写本节要求的内容，不得包含其他节的内容；"
            "使用中文，段落紧凑；优先使用多段落与要点列表以便阅读；适度使用小标题。"
        )
        if section_name == "1. 引言":
            task = "背景与动机、主题边界、综述目标与读者预期（不写总结语）。建议包含2-4段，必要时使用要点列表；字数≥500。"
        elif section_name == "2. 主要研究方向和核心概念":
            task = "列出并定义核心概念；按主题归纳主要研究方向，给出现有代表性工作标题并概括其要点；建议使用小标题分组与项目符号列表；字数≥800。"
        elif section_name == "3. 关键文献回顾与贡献":
            task = "选择代表性文献（≥5），逐篇给出：问题、方法、数据/实验、结论、创新点与局限；建议按每篇文献为一个小段或小标题；字数≥1200。"
        elif section_name == "4. 研究方法的演进与比较":
            task = "按时间或范式梳理方法演进；给出方法间对比表述（适用场景、复杂度、性能、资源需求），点名具体文献；适度用表格样式的列表；字数≥800。"
        elif section_name == "5. 现有研究的局限性与未来研究空白":
            task = "基于文献证据提炼具体局限与未覆盖问题；提出可操作的未来研究问题与可能路径；用条列清晰呈现；字数≥600。"
        elif section_name == "6. 结论与展望":
            task = "汇聚关键共识与稳健发现；指出近期可预期的方向与值得复现的结果；不写套话；字数≥500。"
        else:
            task = "撰写本节内容。"
        return f"""
你正在撰写关于“{original_query}”的学术文献综述的“{section_name}”。
严格遵循以下规则：{rules}
写作任务：{task}
可用文献信息：
{context_str}
只输出“{section_name}”正文内容，不要输出标题，不要输出分节以外的任何文字。
"""

    def _merge_prompt(self, original_query: str, section_name: str, drafts: List[str]) -> str:
        joined = "\n\n---\n\n".join(drafts)
        return (
            f"请将以下多段同一章节“{section_name}”的草稿内容进行去重合并：\n"
            f"- 删除冗余、套话、空洞总结与开场；\n"
            f"- 保留具体事实、数据、对比与文献指称（标题/标识）；\n"
            f"- 合并冲突点时给出更精确的表述；\n"
            f"- 仅输出“{section_name}”的最终正文，不要输出标题或其他说明；\n\n"
            f"草稿：\n{joined}\n"
        )

    async def _build_context_for_batch(self, batch: List[Dict[str, Any]]) -> str:
        temp = { 'batch': { 'results': batch } }
        return self._prepare_context_from_search_results({'batch': temp['batch']}, max_items_per_source=len(batch), max_chars=10000)

    async def _generate_section(self, original_query: str, section_name: str, all_items: List[Dict[str, Any]]) -> str:
        # 更保守的分批策略：控制批次数量，减少LLM并发调用
        n = len(all_items)
        if n > 24:
            batch_size = 12
        elif n > 12:
            batch_size = 8
        else:
            batch_size = max(1, n)
        batches = [all_items[i:i+batch_size] for i in range(0, n, batch_size)]
        tasks = []
        for batch in batches:
            context_str = await self._build_context_for_batch(batch)
            prompt = self._section_prompt(original_query, section_name, context_str)
            tasks.append({
                'task_id': f"{section_name}",
                'prompt': prompt,
                'system_message': self.report_templates['literature_review']['system_message'],
            })
        results = await self.llm.parallel_process(tasks)
        drafts = [r.get('result', '') for r in results if r.get('result')]
        if len(drafts) == 1:
            return drafts[0]
        if not drafts:
            return ""
        # 尝试使用缓存
        key_src = section_name + "\n" + "\n\n---\n\n".join(drafts)
        cache_key = hashlib.sha256(key_src.encode('utf-8')).hexdigest()
        cached = _merge_cache_get(cache_key)
        if cached:
            return cached
        merge_prompt = self._merge_prompt(original_query, section_name, drafts)
        merged = await self.llm.call_llm(merge_prompt, self.report_templates['literature_review']['system_message'])
        if merged:
            _merge_cache_set(cache_key, merged)
        return merged

    async def generate_literature_review(self, search_results: Dict, original_query: str) -> str:
        logging.info(f"开始为查询 '{original_query}' 生成文献综述（严格章节版）...")
        all_items = self._flatten_results(search_results)
        if not all_items:
            return f"# 文献综述报告: {original_query}\n\n未找到可用检索结果。"
        ordered_sections = [
            "1. 引言",
            "2. 主要研究方向和核心概念",
            "3. 关键文献回顾与贡献",
            "4. 研究方法的演进与比较",
            "5. 现有研究的局限性与未来研究空白",
            "6. 结论与展望",
        ]
        sections_text = []
        section_coros = [self._generate_section(original_query, sec, all_items) for sec in ordered_sections]
        generated = await asyncio.gather(*section_coros)
        for sec, body in zip(ordered_sections, generated):
            sections_text.append(f"## {sec}\n\n{body.strip()}\n")
        final_review = f"# 文献综述报告: {original_query}\n\n" + "\n".join(sections_text)
        references_section = self.reference_formatter.format_references(search_results, "markdown")
        final_review += references_section
        return final_review

    async def generate_literature_review_stream(self, search_results: Dict, original_query: str) -> AsyncIterator[str]:
        logging.info(f"流式生成文献综述（严格章节版）: {original_query}")
        all_items = self._flatten_results(search_results)
        if not all_items:
            yield f"# 文献综述报告: {original_query}\n\n未找到可用检索结果。"
            return
        ordered_sections = [
            "1. 引言",
            "2. 主要研究方向和核心概念",
            "3. 关键文献回顾与贡献",
            "4. 研究方法的演进与比较",
            "5. 现有研究的局限性与未来研究空白",
            "6. 结论与展望",
        ]
        yield f"# 文献综述报告: {original_query}\n\n"
        for sec in ordered_sections:
            # 更保守的分批策略：控制批次数量，减少LLM并发调用
            n = len(all_items)
            if n > 24:
                batch_size = 12
            elif n > 12:
                batch_size = 8
            else:
                batch_size = max(1, n)
            batches = [all_items[i:i+batch_size] for i in range(0, n, batch_size)]
            tasks = []
            for batch in batches:
                context_str = await self._build_context_for_batch(batch)
                prompt = self._section_prompt(original_query, sec, context_str)
                tasks.append({
                    'task_id': f"{sec}",
                    'prompt': prompt,
                    'system_message': self.report_templates['literature_review']['system_message'],
                })
            results = await self.llm.parallel_process(tasks)
            drafts = [r.get('result', '') for r in results if r.get('result')]
            yield f"## {sec}\n\n"
            if len(drafts) == 1:
                yield drafts[0] + "\n\n"
                continue
            if not drafts:
                yield "\n\n"
                continue
            key_src = sec + "\n" + "\n\n---\n\n".join(drafts)
            cache_key = hashlib.sha256(key_src.encode('utf-8')).hexdigest()
            cached = _merge_cache_get(cache_key)
            if cached:
                yield cached + "\n\n"
                continue
            merge_prompt = self._merge_prompt(original_query, sec, drafts)
            acc: List[str] = []
            async for chunk in self.llm.call_llm_stream(merge_prompt, self.report_templates['literature_review']['system_message']):
                acc.append(chunk)
                yield chunk
            merged_text = "".join(acc)
            if merged_text:
                _merge_cache_set(cache_key, merged_text)
            yield "\n\n"
        references_section = self.reference_formatter.format_references(search_results, "markdown")
        yield references_section
        
    async def generate_corporate_research_report(self, search_results: Dict, user_input: Dict, original_query: str) -> str:
        """生成企业调研报告"""
        logging.info(f"开始生成企业调研报告: {original_query}")
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=5)
        
        # 获取报告模板
        template = self.report_templates["corporate_research"]
        system_message = template["system_message"]
        
        # 构建提示词
        prompt = template["prompt_template"].format(context_str=context_str, original_query=original_query)
        
        # 调用LLM生成报告
        report = await self.llm.call_llm(prompt, system_message)
        
        # 添加页眉和页脚
        user_name = user_input.get("user_name", "尊敬的用户")
        company_info = f" ({user_input.get('occupation', '企业用户')})" if user_input.get('occupation') != '企业用户' else ""
        report = f"# 企业调研报告: {original_query}\n\n{report}\n\n---\n\n*此报告由KnowlEdge智能引擎为 {user_name}{company_info} 生成，基于 {user_input.get('day', 7)} 天内的最新市场数据*"
        
        # 添加参考文献
        report += self.reference_formatter.format_references(search_results, "markdown")
        
        return report
        
    async def generate_corporate_research_report_stream(self, search_results: Dict, user_input: Dict, original_query: str) -> AsyncIterator[str]:
        """流式生成企业调研报告"""
        logging.info(f"开始流式生成企业调研报告: {original_query}")
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=5)
        
        # 获取报告模板
        template = self.report_templates["corporate_research"]
        system_message = template["system_message"]
        
        # 构建提示词
        prompt = template["prompt_template"].format(context_str=context_str, original_query=original_query)
        
        # 先输出标题和说明
        title = f"# 企业调研报告: {original_query}\n\n"
        yield title
        
        # 添加报告说明
        yield "**报告说明**：本报告从企业经营和商业价值角度分析相关技术或行业发展情况，为企业决策提供参考。\n\n"
        yield "---\n\n"
        
        # 流式生成报告内容
        async for chunk in self.llm.call_llm_stream(prompt, system_message):
            yield chunk
            
        # 输出页脚
        user_name = user_input.get("user_name", "尊敬的用户")
        company_info = f" ({user_input.get('occupation', '企业用户')})" if user_input.get('occupation') != '企业用户' else ""
        yield f"\n\n---\n\n*此报告由KnowlEdge智能引擎为 {user_name}{company_info} 生成，基于 {user_input.get('day', 7)} 天内的最新市场数据*"
        
        # 添加参考文献
        references_section = self.reference_formatter.format_references(search_results, "markdown")
        yield references_section
        
    async def generate_popular_science_report(self, search_results: Dict, user_input: Dict, original_query: str) -> str:
        """生成科普知识报告"""
        logging.info(f"开始生成科普知识报告: {original_query}")
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=8, max_chars=20000)
        
        # 获取报告模板
        template = self.report_templates["popular_science"]
        system_message = template["system_message"]
        
        # 构建提示词
        prompt = template["prompt_template"].format(context_str=context_str, original_query=original_query)
        
        # 调用LLM生成报告
        report = await self.llm.call_llm(prompt, system_message)
        
        # 添加页眉和页脚
        user_name = user_input.get("user_name", "读者朋友")
        report = f"# 科普知识报告: {original_query}\n\n{report}\n\n---\n\n*此科普报告由KnowlEdge智能引擎为 {user_name} 精心生成，希望能增长您的知识*"
        
        # 添加参考文献
        report += self.reference_formatter.format_references(search_results, "markdown")
        
        return report
        
    async def generate_popular_science_report_stream(self, search_results: Dict, user_input: Dict, original_query: str) -> AsyncIterator[str]:
        """流式生成科普知识报告"""
        logging.info(f"开始流式生成科普知识报告: {original_query}")
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=8, max_chars=20000)
        
        # 获取报告模板 
        template = self.report_templates["popular_science"]
        system_message = template["system_message"]
        
        # 构建提示词
        prompt = template["prompt_template"].format(context_str=context_str, original_query=original_query)
        
        # 先输出标题
        title = f"# 科普知识报告: {original_query}\n\n"
        yield title
        
        # 流式生成报告内容
        async for chunk in self.llm.call_llm_stream(prompt, system_message):
            yield chunk
            
        # 输出页脚
        user_name = user_input.get("user_name", "读者朋友")
        yield f"\n\n---\n\n*此科普报告由KnowlEdge智能引擎为 {user_name} 精心生成，希望能增长您的知识*"
        
        # 添加参考文献
        references_section = self.reference_formatter.format_references(search_results, "markdown")
        yield references_section
    
    # --- 多模型文献综述生成方法（已禁用：回退单模型） ---
    
    async def _generate_multi_model_literature_review(self, context_str: str, original_query: str) -> str:
        logging.info("多模型综述已禁用，回退到单模型实现")
        try:
            # 直接复用单模型严格章节实现
            return await self.generate_literature_review({'batch': {'results': []}}, original_query)
        except Exception:
            return ""
    
    async def _generate_multi_model_literature_review_stream(self, context_str: str, original_query: str) -> AsyncIterator[str]:
        logging.info("多模型流式综述已禁用，回退到单模型实现")
        async for chunk in self.generate_literature_review_stream({'batch': {'results': []}}, original_query):
                        yield chunk

    async def generate_enhanced_literature_review_stream(self, search_results: Dict, original_query: str) -> AsyncIterator[str]:
        """
        生成增强版文献综述流式输出
        - 确保所有检索到的文章都被分析
        - 保持整体一致性和前后文关联性
        - 使用多模型并行处理提高效率
        """
        logging.info(f"开始生成增强版文献综述: {original_query}")
        
        # 获取所有文章，确保不遗漏
        all_articles = self._extract_all_articles(search_results)
        
        if not all_articles:
            yield f"# 文献综述报告: {original_query}\n\n未找到可分析的文献。"
            return
        
        # 输出报告头部和概览
        yield f"# 【增强版】文献综述报告: {original_query}\n\n"
        yield f"**综述概览**\n"
        yield f"- 检索文献数量：{len(all_articles)} 篇\n"
        yield f"- 来源分布：{self._get_source_distribution(all_articles)}\n"
        yield f"- 生成时间：{self._get_current_time()}\n\n"
        
        # 阶段1：预处理 - 并行分析所有文章以获得整体理解
        yield "## 📊 文献预处理分析\n\n"
        yield "正在对所有检索文献进行预处理分析，以确保综述的完整性和一致性...\n\n"
        
        try:
            # 并行预分析所有文章
            article_analyses = await self._parallel_preanalyze_articles(all_articles, original_query)
            
            yield f"✅ 完成 {len(article_analyses)} 篇文献的预处理分析\n\n"
            
            # 阶段2：按章节生成综述内容，基于完整的文章分析
            ordered_sections = [
                "1. 引言与研究背景",
                "2. 文献分布与研究现状",
                "3. 核心理论与方法综述", 
                "4. 主要研究发现与贡献",
                "5. 研究方法的演进与比较",
                "6. 存在问题与未来展望"
            ]
            
            # 基于所有文章分析生成各章节
            for section in ordered_sections:
                yield f"## {section}\n\n"
                
                # 为每个章节生成内容，确保基于所有文章的分析
                section_content = await self._generate_comprehensive_section(
                    original_query, section, all_articles, article_analyses
                )
                
                yield section_content
                yield "\n\n"
            
            # 阶段3：文献完整性验证
            yield "## ✅ 文献覆盖度验证\n\n"
            coverage_report = await self._verify_literature_coverage(all_articles, article_analyses)
            yield coverage_report
            yield "\n\n"
            
        except Exception as e:
            logging.error(f"增强版文献综述生成失败: {e}")
            yield f"**错误**：文献综述生成过程中出现问题：{str(e)}\n\n"
        
        # 添加完整的参考文献
        yield "## 📖 参考文献\n\n"
        references_section = self.reference_formatter.format_references(search_results, "markdown")
        yield references_section
        
        logging.info("增强版文献综述生成完成")

    async def _parallel_preanalyze_articles(self, articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """并行预分析所有文章，为后续综述生成提供基础"""
        logging.info(f"开始并行预分析 {len(articles)} 篇文章")
        
        # 创建预分析任务
        tasks = []
        for i, article in enumerate(articles):
            prompt = self._create_preanalysis_prompt(article, i + 1, len(articles), query)
            tasks.append({
                'task_id': f"preanalysis_{i+1}",
                'prompt': prompt,
                'system_message': "你是专业的学术文献分析专家，擅长快速提取文献的核心信息和学术价值。",
                'metadata': {'article_index': i, 'article': article}
            })
        
        # 并行处理，分批执行避免过载
        batch_size = 8  # 控制并发数量
        analyses = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await self.llm.parallel_process(batch)
            
            for task, result in zip(batch, batch_results):
                if result.get('result'):
                    analysis = {
                        'article': task['metadata']['article'],
                        'analysis': result['result'],
                        'index': task['metadata']['article_index']
                    }
                    analyses.append(analysis)
                else:
                    logging.warning(f"文章 {task['task_id']} 预分析失败")
        
        logging.info(f"完成 {len(analyses)} 篇文章的预分析")
        return analyses

    def _create_preanalysis_prompt(self, article: Dict[str, Any], index: int, total: int, query: str) -> str:
        """创建文章预分析提示词"""
        return f"""
请对以下文献进行快速而全面的学术分析，为文献综述 "{query}" 提供基础信息：

**文献信息 ({index}/{total})：**
- 标题：{article['title']}
- 作者：{', '.join(article['authors']) if article['authors'] else '未知'}
- 发表时间：{article['published'] or '未知'}
- 来源：{article['source']}

**摘要：**
{article['abstract']}

**分析要求：**
请从以下几个维度分析这篇文献：

1. **核心主题**：这篇文献的核心研究主题是什么？与查询"{query}"的关联度如何？
2. **研究方法**：采用了什么研究方法或技术路线？
3. **主要贡献**：这项研究的主要创新点和学术贡献是什么？
4. **关键发现**：有哪些重要的研究发现或结论？
5. **应用价值**：这项研究的实际应用价值和理论意义？
6. **研究地位**：在相关领域中的重要性和影响力如何？

**输出格式：**
保持简洁但信息丰富，重点突出与主题"{query}"相关的核心内容。
"""

    async def _generate_comprehensive_section(self, query: str, section: str, all_articles: List[Dict[str, Any]], 
                                            article_analyses: List[Dict[str, Any]]) -> str:
        """为特定章节生成综合性内容，基于所有文章的分析"""
        
        # 构建包含所有文章分析的上下文
        context_parts = []
        for analysis in article_analyses:
            article = analysis['article']
            analysis_text = analysis['analysis']
            
            context_part = f"""
文献 {analysis['index'] + 1}: {article['title']}
作者: {', '.join(article['authors']) if article['authors'] else '未知'}
发表: {article['published'] or '未知'}
分析: {analysis_text}
---
"""
            context_parts.append(context_part)
        
        full_context = "\n".join(context_parts)
        
        # 为不同章节定制不同的生成策略
        section_prompts = {
            "1. 引言与研究背景": f"""
基于所有检索到的文献，为文献综述 "{query}" 撰写引言与研究背景部分。

要求：
1. 概述研究领域的发展背景和重要性
2. 明确研究问题和综述目标
3. 说明文献选择的标准和范围
4. 为后续章节做好铺垫

基于文献：
{full_context}
""",
            "2. 文献分布与研究现状": f"""
基于所有检索到的文献，分析 "{query}" 领域的文献分布与研究现状。

要求：
1. 分析文献的时间分布和发展趋势
2. 概述主要研究机构和学者的贡献
3. 总结研究热点和发展阶段
4. 识别研究的地域分布特点

基于文献：
{full_context}
""",
            "3. 核心理论与方法综述": f"""
基于所有检索到的文献，综述 "{query}" 领域的核心理论与方法。

要求：
1. 梳理主要的理论框架和概念体系
2. 分析核心研究方法的发展演进
3. 比较不同理论和方法的优劣势
4. 识别理论和方法的创新点

基于文献：
{full_context}
""",
            "4. 主要研究发现与贡献": f"""
基于所有检索到的文献，总结 "{query}" 领域的主要研究发现与贡献。

要求：
1. 总结重要的研究发现和结论
2. 分析研究成果的学术价值和影响
3. 识别突破性进展和创新成果
4. 评估研究贡献的重要性

基于文献：
{full_context}
""",
            "5. 研究方法的演进与比较": f"""
基于所有检索到的文献，分析 "{query}" 领域研究方法的演进与比较。

要求：
1. 梳理研究方法的发展历程
2. 比较不同方法的适用性和有效性
3. 分析方法学的创新和改进
4. 评估方法的局限性和改进空间

基于文献：
{full_context}
""",
            "6. 存在问题与未来展望": f"""
基于所有检索到的文献，分析 "{query}" 领域存在的问题与未来展望。

要求：
1. 识别当前研究的局限性和不足
2. 分析未解决的关键问题
3. 提出未来研究的重要方向
4. 展望领域发展的前景和机遇

基于文献：
{full_context}
"""
        }
        
        prompt = section_prompts.get(section, f"""
请为文献综述 "{query}" 的 "{section}" 部分撰写内容。

基于文献：
{full_context}
""")
        
        # 调用LLM生成章节内容
        try:
            section_content = await self.llm.call_llm(
                prompt,
                "你是专业的学术综述撰写者，擅长基于大量文献生成结构清晰、逻辑严密的综述内容。要求：内容全面深入，逻辑清晰，避免空洞套话。"
            )
            return section_content
        except Exception as e:
            logging.error(f"章节 {section} 生成失败: {e}")
            return f"章节内容生成失败：{str(e)}"

    async def _verify_literature_coverage(self, all_articles: List[Dict[str, Any]], 
                                        article_analyses: List[Dict[str, Any]]) -> str:
        """验证文献覆盖度，确保所有文章都被分析"""
        total_articles = len(all_articles)
        analyzed_articles = len(article_analyses)
        
        coverage_rate = (analyzed_articles / total_articles * 100) if total_articles > 0 else 0
        
        coverage_report = f"""
**文献覆盖度统计：**
- 检索文献总数：{total_articles} 篇
- 成功分析文献：{analyzed_articles} 篇  
- 覆盖率：{coverage_rate:.1f}%

**覆盖状态：**
"""
        
        if coverage_rate >= 95:
            coverage_report += "✅ **优秀**：文献覆盖度达到95%以上，综述质量有保障。\n"
        elif coverage_rate >= 80:
            coverage_report += "⚠️ **良好**：文献覆盖度达到80%以上，综述基本完整。\n"
        else:
            coverage_report += "❌ **待改进**：文献覆盖度低于80%，建议检查分析过程。\n"
        
        # 列出未分析的文章（如果有）
        analyzed_titles = {analysis['article']['title'] for analysis in article_analyses}
        missing_articles = [article for article in all_articles if article['title'] not in analyzed_titles]
        
        if missing_articles:
            coverage_report += f"\n**未分析文献 ({len(missing_articles)} 篇)：**\n"
            for i, article in enumerate(missing_articles[:5], 1):  # 只显示前5篇
                coverage_report += f"{i}. {article['title']}\n"
            if len(missing_articles) > 5:
                coverage_report += f"... 还有 {len(missing_articles) - 5} 篇未列出\n"
        else:
            coverage_report += "\n✅ **所有检索文献均已完成分析**\n"
        
        return coverage_report