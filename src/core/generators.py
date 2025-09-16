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
            }
        }
        logging.info("报告生成器初始化完成")
    
    def _prepare_context_from_search_results(self, search_results: Dict, max_items_per_source: int = 5, max_chars: int = 12000) -> str:
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

    async def generate_report_stream(self, search_results: Dict, user_input: Dict) -> AsyncIterator[str]:
        logging.info("流式生成最终的搜索报告...")
        platform_type = user_input.get('platform_type') or search_results.get('platform_type', '通用平台')
        context_str = self._prepare_context_from_search_results(search_results)
        template = self.report_templates["standard"]
        system_message = template["system_message"]
        prompt = template["prompt_template"].format(context_str=context_str, platform_type=platform_type)
        title = f"# 【{platform_type}】{template['title']}\n\n"
        yield title
        async for chunk in self.llm.call_llm_stream(prompt, system_message):
            yield chunk
        yield f"\n\n---\n\n根据您选择的【{platform_type}】平台搜索结果生成的报告\n\n"
        references_section = self.reference_formatter.format_references(search_results, "markdown")
        yield references_section

    # ===== 文献综述（严格章节化、分批并行、去套话合并） =====
    def _flatten_results(self, search_results: Dict) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for source, data in search_results.items():
            if isinstance(data, dict) and data.get('results'):
                for it in data['results']:
                    e = it.copy()
                    e['source'] = source
                    items.append(e)
        return items

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
        batch_size = 10 if len(all_items) > 10 else max(1, len(all_items))
        batches = [all_items[i:i+batch_size] for i in range(0, len(all_items), batch_size)]
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
            batch_size = 10 if len(all_items) > 10 else max(1, len(all_items))
            batches = [all_items[i:i+batch_size] for i in range(0, len(all_items), batch_size)]
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
        
    async def generate_industry_research_report(self, search_results: Dict, user_input: Dict, original_query: str) -> str:
        """生成行业研究报告"""
        logging.info(f"开始生成行业研究报告: {original_query}")
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=5)
        
        # 获取报告模板
        template = self.report_templates["industry_research"]
        system_message = template["system_message"]
        
        # 构建提示词
        prompt = template["prompt_template"].format(context_str=context_str, original_query=original_query)
        
        # 调用LLM生成报告
        report = await self.llm.call_llm(prompt, system_message)
        
        # 添加页眉和页脚
        user_name = user_input.get("user_name", "尊敬的用户")
        report = f"# 行业研究报告: {original_query}\n\n{report}\n\n---\n\n*此报告由KnowlEdge智能引擎为 {user_name} 生成于 {user_input.get('day', 7)} 天内的最新行业数据*"
        
        # 添加参考文献
        report += self.reference_formatter.format_references(search_results, "markdown")
        
        return report
        
    async def generate_industry_research_report_stream(self, search_results: Dict, user_input: Dict, original_query: str) -> AsyncIterator[str]:
        """流式生成行业研究报告"""
        logging.info(f"开始流式生成行业研究报告: {original_query}")
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=5)
        
        # 获取报告模板
        template = self.report_templates["industry_research"]
        system_message = template["system_message"]
        
        # 构建提示词
        prompt = template["prompt_template"].format(context_str=context_str, original_query=original_query)
        
        # 先输出标题
        title = f"# 行业研究报告: {original_query}\n\n"
        yield title
        
        # 流式生成报告内容
        async for chunk in self.llm.call_llm_stream(prompt, system_message):
            yield chunk
            
        # 输出页脚
        user_name = user_input.get("user_name", "尊敬的用户")
        yield f"\n\n---\n\n*此报告由KnowlEdge智能引擎为 {user_name} 生成于 {user_input.get('day', 7)} 天内的最新行业数据*"
        
        # 添加参考文献
        references_section = self.reference_formatter.format_references(search_results, "markdown")
        yield references_section
        
    async def generate_popular_science_report(self, search_results: Dict, user_input: Dict, original_query: str) -> str:
        """生成科普知识报告"""
        logging.info(f"开始生成科普知识报告: {original_query}")
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=5)
        
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
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=5)
        
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
    
    # --- 多模型文献综述生成方法 ---
    
    async def _generate_multi_model_literature_review(self, context_str: str, original_query: str) -> str:
        """使用多模型并行处理生成文献综述"""
        logging.info(f"使用多模型并行生成文献综述: {original_query}")
        try:
            # 按顺序生成各章节内容
            ordered_sections = [
                "1. 引言",
                "2. 主要研究方向和核心概念",
                "3. 关键文献回顾与贡献",
                "4. 研究方法的演进与比较",
                "5. 现有研究的局限性与未来研究空白",
                "6. 结论与展望"
            ]
            
            # 构建基础上下文提示词
            base_prompt = f"""
            请基于主题 "{original_query}" 和以下提供的相关学术文献信息，撰写文献综述报告的相应章节。
            
            ---文献信息参考---
            {context_str}
            ---文献信息参考结束---
            
            请确保内容：
            1. 深入分析文献，而非简单摘要堆砌
            2. 内容详实充分，字数充足（800-1200字/章节）
            3. 使用专业学术风格，但表述清晰
            4. 适当引用文献中的关键观点和发现
            5. 提供批判性分析和见解，而不仅是描述性内容
            """
            
            # 准备并行任务列表
            section_tasks = []
            for section_name in ordered_sections:
                # 获取该章节对应的模型
                model_names = self.multi_model_config["section_assignments"].get(section_name, ["主模型"])
                model = next((m for m in self.multi_model_config["models"] if m["name"] == model_names[0]), None)
                
                if model:
                    # 构建章节特定的提示词
                    section_prompt = f"""
                    你需要为文献综述报告撰写 "## {section_name}" 章节的内容。
                    
                    请基于以下信息，撰写内容丰富、分析深入的章节内容。确保内容专业、全面，并使用学术风格。
                    
                    {base_prompt}
                    
                    请只生成 "{section_name}" 章节的内容，不要包含章节标题，直接开始正文内容。
                    确保内容详尽、充实，字数在800-1200字之间。
                    """
                    
                    # 添加到任务列表
                    section_tasks.append({
                        "prompt": section_prompt,
                        "system_message": model["system_message"],
                        "task_id": section_name
                    })
            
            # 并行执行所有章节生成任务
            sections_results = await self.llm.parallel_process(section_tasks)
            
            # 按顺序整合结果
            full_review = ""
            for section_name in ordered_sections:
                section_result = next((r for r in sections_results if r["task_id"] == section_name), None)
                if section_result and "result" in section_result:
                    full_review += f"## {section_name}\n\n{section_result['result']}\n\n"
                else:
                    logging.warning(f"未能获取章节 '{section_name}' 的生成结果")
            
            return full_review
            
        except Exception as e:
            logging.error(f"多模型文献综述生成失败: {e}")
            return ""  # 返回空字符串，让调用方回退到单模型生成
    
    async def _generate_multi_model_literature_review_stream(self, context_str: str, original_query: str) -> AsyncIterator[str]:
        """使用多模型并行处理流式生成文献综述"""
        logging.info(f"使用多模型流式生成文献综述: {original_query}")
        
        # 按顺序生成各章节内容
        ordered_sections = [
            "1. 引言",
            "2. 主要研究方向和核心概念",
            "3. 关键文献回顾与贡献",
            "4. 研究方法的演进与比较",
            "5. 现有研究的局限性与未来研究空白",
            "6. 结论与展望"
        ]
        
        # 构建基础上下文提示词
        base_prompt = f"""
        请基于主题 "{original_query}" 和以下提供的相关学术文献信息，撰写文献综述报告的相应章节。
        
        ---文献信息参考---
        {context_str}
        ---文献信息参考结束---
        
        请确保内容：
        1. 深入分析文献，而非简单摘要堆砌
        2. 内容详实充分，字数充足（800-1200字）
        3. 使用专业学术风格，但表述清晰
        4. 适当引用文献中的关键观点和发现
        5. 提供批判性分析和见解，而不仅是描述性内容
        """
        
        # 按顺序流式生成并输出各章节
        for section_name in ordered_sections:
            try:
                # 获取该章节对应的模型
                model_names = self.multi_model_config["section_assignments"].get(section_name, ["主模型"])
                model = next((m for m in self.multi_model_config["models"] if m["name"] == model_names[0]), None)
                
                if model:
                    # 输出章节标题
                    yield f"## {section_name}\n\n"
                    
                    # 构建章节特定的提示词
                    section_prompt = f"""
                    你需要为文献综述报告撰写 "## {section_name}" 章节的内容。
                    
                    请基于以下信息，撰写内容丰富、分析深入的章节内容。确保内容专业、全面，并使用学术风格。
                    
                    {base_prompt}
                    
                    请只生成 "{section_name}" 章节的内容，不要包含章节标题，直接开始正文内容。
                    确保内容详尽、充实，字数在800-1200字之间。
                    """
                    
                    # 流式生成章节内容
                    async for chunk in self.llm.call_llm_stream(section_prompt, model["system_message"]):
                        yield chunk
                    
                    # 章节之间添加换行
                    yield "\n\n"
            except Exception as e:
                logging.error(f"流式生成章节 '{section_name}' 时出错: {e}")
                yield f"*生成此章节时出现错误，请见谅。*\n\n" 