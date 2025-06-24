"""
报告生成模块
提供各种类型报告生成功能
"""
import logging
from typing import Dict, AsyncIterator, List
from src.core.llm_interface import LLMInterface

class ReportGenerator:
    """报告生成器类"""
    
    def __init__(self, llm=None):
        """初始化报告生成器"""
        self.llm = llm or LLMInterface()
        
        # 定义不同报告类型的模板和配置
        self.report_templates = {
            "standard": {
                "title": "综合报告",
                "system_message": "你是一个知识渊博且表达能力优秀的AI助手，擅长整理和归纳信息，为用户生成全面、客观的综合报告。",
                "prompt_template": """
                请根据以下搜索结果，生成一份结构清晰、信息丰富的综合报告。
                报告中应该包含对重要信息的提取、归纳总结，并按照逻辑关系组织内容。
                
                ---搜索结果---
                {context_str}
                ---搜索结果结束---
                
                请确保报告：
                1. 有明确的结构和标题
                2. 逻辑清晰，内容连贯
                3. 提炼关键信息，避免冗余
                4. 使用Markdown格式进行排版
                
                根据用户选择的【{platform_type}】平台特点，请特别关注该领域的最新进展和重要信息。
                """
            },
            "literature_review": {
                "title": "文献综述报告",
                "system_message": "你是一名资深的学术研究员和文献综述撰写专家。请严格按照用户要求的结构和格式生成内容。确保所有输出均为结构良好、干净的Markdown格式，段落间使用双换行符分隔，列表使用标准的Markdown语法。不要在段落中随意插入不必要的换行。用户在Prompt中章节标题后用括号 () 或 （） 包含的文字是对您生成该章节内容的引导和提示，这些括号及其内部的文字绝对不能出现在最终的输出中。您只需要生成这些括号提示之外的、针对该章节的实际内容。所有章节标题和结构由用户在Prompt中指定，请严格遵循。",
                "prompt_template": """
                请基于主题 "{original_query}" 和以下提供的相关学术文献信息，撰写一份全面且结构清晰的文献综述报告。
                报告应包含以下Markdown结构和内容。请直接在每个章节标题下撰写对应的内容，不要重复标题，也不要输出括号中的提示文字。
                
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

                ---文献信息参考---\n{context_str}\n---文献信息参考结束---\n

                请严格按照以上Markdown结构和要求生成完整的文献综述内容，每段话之间至多一句换行，标题和下面内容之间不换行，别部分也不要多余的换行。
                """
            },
            "industry_research": {
                "title": "行业研究报告",
                "system_message": "你是一位资深的行业分析师，擅长撰写专业的行业研究报告。请基于提供的信息，生成结构严谨、数据支撑充分、洞察深入的行业研究报告。报告应当客观中立，同时提供有价值的见解和预测。",
                "prompt_template": """
                请基于以下信息和数据，为主题"{original_query}"撰写一份专业的行业研究报告。
                
                ## 行业研究报告: {original_query}
                
                报告需包含以下章节结构：
                
                ## 1. 行业概况与背景
                (请提供行业的基本情况，发展历程，当前市场规模和增长率等关键信息)
                
                ## 2. 市场结构与竞争格局
                (分析行业的市场结构，主要参与者及其市场份额，竞争态势等)
                
                ## 3. 核心技术与创新趋势
                (详述行业中的关键技术，最新创新方向，技术演进路线图等)
                
                ## 4. 商业模式与价值链分析
                (剖析行业主要商业模式，价值链构成，以及各环节价值分配情况)
                
                ## 5. 规范与政策环境
                (概述影响行业的主要政策法规，监管框架，以及潜在的政策变化)
                
                ## 6. 行业挑战与机遇
                (指出行业面临的主要挑战，以及未来可能出现的机遇窗口)
                
                ## 7. 未来发展预测
                (基于数据和趋势，对行业未来3-5年的发展做出合理预测)
                
                ## 8. 结论与建议
                (总结报告要点，并为行业参与者提供战略性建议)

                ---信息来源---
                {context_str}
                ---信息来源结束---
                
                本报告的目标读者是对"{original_query}"相关行业感兴趣的专业人士、投资者和决策者。请确保报告内容专业、客观，数据引用准确，分析见解深入。如信息来源中缺乏某些必要数据，可以适当标注"数据缺失"，但应尽量基于已有信息做出合理分析。

                请使用规范的Markdown格式输出，保证报告结构清晰，便于阅读。
                """
            },
            "popular_science": {
                "title": "科普知识报告",
                "system_message": "你是一位优秀的科普作家，擅长将复杂的学术知识转化为通俗易懂、生动有趣的科普内容。请基于提供的研究信息，创作一篇既准确又引人入胜的科普文章，适合大众读者阅读。",
                "prompt_template": """
                请基于以下研究信息，为主题"{original_query}"创作一篇科普知识报告。
                
                ## 科普知识报告: {original_query}
                
                报告应包含以下内容结构：
                
                ## 1. 引人入胜的开篇
                (请创作一个吸引人的开头，可以是一个有趣的问题、一个令人惊讶的事实、一个生动的场景描述，或者与日常生活的联系，目的是激发读者的好奇心和阅读兴趣)
                
                ## 2. 核心概念简明解释
                (用通俗易懂的语言解释主题中的核心概念和基础知识，避免使用过多专业术语，如必须使用，则应当给出清晰解释)
                
                ## 3. 最新研究发现与突破
                (介绍该领域的最新研究成果和突破性进展，强调其意义和潜在影响)
                
                ## 4. 实际应用与生活关联
                (说明这些研究和知识如何应用于实际生活或产业，以及它们如何影响或改变我们的日常生活)
                
                ## 5. 趣味知识与小故事
                (穿插一些与主题相关的趣味事实、轶事或小故事，增加可读性和趣味性)
                
                ## 6. 未来展望与思考
                (探讨该领域未来可能的发展方向，以及它可能带来的社会变化和思考)
                
                ## 7. 延伸阅读与资源推荐
                (为对这一主题感兴趣的读者推荐一些易于理解的延伸阅读资料或资源)

                ---研究信息来源---
                {context_str}
                ---研究信息来源结束---
                
                本科普报告面向普通大众读者，他们可能没有相关的专业背景。请确保内容：
                1. 准确传达科学知识，不夸大或误导
                2. 语言生动有趣，避免枯燥
                3. 使用贴切的比喻、类比和例子帮助理解
                4. 适当使用问答形式或对话形式增强互动感
                5. 整体结构清晰，逻辑流畅

                请使用规范的Markdown格式输出，保证报告结构清晰，便于阅读。
                """
            }
        }
        logging.info("报告生成器初始化完成")
    
    def _prepare_context_from_search_results(self, search_results: Dict, max_items_per_source: int = 3, max_chars: int = 8000) -> str:
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
        
    async def generate_report(self, search_results: Dict) -> str:
        """生成标准搜索报告"""
        logging.info("生成最终的搜索报告...")
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results)
        
        # 获取报告模板
        template = self.report_templates["standard"]
        system_message = template["system_message"]
        
        # 构建提示词
        platform_type = search_results.get('platform_type', '通用平台')
        prompt = template["prompt_template"].format(context_str=context_str, platform_type=platform_type)
        
        # 调用LLM生成报告
        report = await self.llm.call_llm(prompt, system_message)
        return report
        
    async def generate_report_stream(self, search_results: Dict) -> AsyncIterator[str]:
        """流式生成标准搜索报告"""
        logging.info("流式生成最终的搜索报告...")
        platform_type = search_results.get('platform_type', '通用平台')
        
        # 从搜索结果中准备上下文
        context_str = self._prepare_context_from_search_results(search_results)
        
        # 获取报告模板
        template = self.report_templates["standard"]
        system_message = template["system_message"]
        
        # 构建提示词
        prompt = template["prompt_template"].format(context_str=context_str, platform_type=platform_type)
        
        # 先输出标题
        title = f"# 【{platform_type}】{template['title']}\n\n"
        yield title
        
        # 流式生成报告内容
        async for chunk in self.llm.call_llm_stream(prompt, system_message):
            yield chunk
            
        # 输出结尾
        yield f"\n\n---\n\n根据您选择的【{platform_type}】平台，近几日的行业内最新进展已整理好，请查收！"
        
    async def generate_literature_review(self, search_results: Dict, original_query: str) -> str:
        """生成文献综述"""
        logging.info(f"开始为查询 '{original_query}' 生成文献综述...")
        
        # 为文献综述提供更多的搜索结果上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=8)
        
        # 获取报告模板
        template = self.report_templates["literature_review"]
        system_message = template["system_message"]
        
        # 构建提示词
        prompt = template["prompt_template"].format(context_str=context_str, original_query=original_query)
        
        # 收集参考文献
        references_list = []
        for source, data in search_results.items():
            if data and isinstance(data, dict) and 'results' in data and data['results']:
                for item in data['results']:
                    if isinstance(item, dict):
                        title = item.get('title', '未知标题')
                        link = item.get('link', '#')
                        if link and link != '#':
                            references_list.append({"title": title, "link": link, "source": source})
        
        # 去重
        unique_references = []
        seen_links = set()
        for ref in references_list:
            if ref["link"] not in seen_links:
                unique_references.append(ref)
                seen_links.add(ref["link"])
               
        # 调用LLM生成文献综述内容
        review_text = await self.llm.call_llm(prompt, system_message)
        
        # 添加主标题
        final_report = f"# 文献综述报告: {original_query}\n\n{review_text}"
        
        # 添加参考文献
        if unique_references:
            references_section = "\n\n---\n## 参考文献\n\n"
            for i, ref in enumerate(unique_references):
                title = ref.get('title', '未知标题')
                source = ref.get('source', '未知来源')
                link = ref.get('link', '#')
                authors = ref.get('authors', [])
                published = ref.get('published', '')
                updated = ref.get('updated', '')
                
                # 格式化作者列表
                authors_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_str += " 等"
                
                # 构建参考文献条目
                references_section += f"{i+1}. **{title}**\n"
                references_section += f"   - 作者: {authors_str}\n"
                if published:
                    references_section += f"   - 发表时间: {published}\n"
                if updated and updated != published:
                    references_section += f"   - 更新时间: {updated}\n"
                references_section += f"   - 来源: {source}\n"
                references_section += f"   - 链接: [{link}]({link})\n"
                
                # 每两篇论文之间空一行，最后一篇不加空行
                if i < len(unique_references) - 1:
                    references_section += "\n"
                
            final_report += references_section
        else:
            final_report += "\n\n---\n未找到可引用的文献链接。"
            
        return final_report
        
    async def generate_literature_review_stream(self, search_results: Dict, original_query: str) -> AsyncIterator[str]:
        """流式生成文献综述"""
        logging.info(f"流式生成文献综述: {original_query}")
        
        # 为文献综述提供更多的搜索结果上下文
        context_str = self._prepare_context_from_search_results(search_results, max_items_per_source=8)
        
        # 先生成标题
        title = f"# 文献综述报告: {original_query}\n\n"
        yield title
        
        # 使用多模型并行处理生成文献综述的各个章节
        base_prompt = f"""
        请基于主题 "{original_query}" 和以下提供的相关学术文献信息，撰写文献综述报告的相应章节。
        
        ---文献信息参考---
        {context_str}
        ---文献信息参考结束---
        
        请确保内容：
        1. 深入分析文献，而非简单摘要堆砌
        2. 内容详实充分，字数充足
        3. 使用专业学术风格，但表述清晰
        4. 适当引用文献中的关键观点和发现
        """
        
        # 按顺序生成并输出各章节
        ordered_sections = [
            "1. 引言",
            "2. 主要研究方向和核心概念",
            "3. 关键文献回顾与贡献",
            "4. 研究方法的演进与比较",
            "5. 现有研究的局限性与未来研究空白",
            "6. 结论与展望"
        ]
        
        for section_name in ordered_sections:
            # 获取该章节对应的模型
            model_names = self.multi_model_config["section_assignments"].get(section_name, ["主模型"])
            model = next((m for m in self.multi_model_config["models"] if m["name"] == model_names[0]), None)
            
            if model:
                # 输出章节标题
                yield f"## {section_name}\n\n"
                
                # 构建章节特定的提示词
                section_prompt = f"""
                你需要为文献综述报告撰写 "{section_name}" 章节的内容。
                
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
        
        # 收集参考文献
        references_list = []
        for source, data in search_results.items():
            if data and isinstance(data, dict) and 'results' in data and data['results']:
                for item in data['results']:
                    if isinstance(item, dict):
                        title = item.get('title', '未知标题')
                        link = item.get('link', '#')
                        authors = item.get('authors', [])
                        published = item.get('published', '')
                        updated = item.get('updated', '')
                        if link and link != '#':
                            references_list.append({
                                "title": title, 
                                "link": link, 
                                "source": source,
                                "authors": authors,
                                "published": published,
                                "updated": updated
                            })
        
        # 去重
        unique_references = []
        seen_links = set()
        for ref in references_list:
            if ref["link"] not in seen_links:
                unique_references.append(ref)
                seen_links.add(ref["link"])
        
        # 输出参考文献
        if unique_references:
            references_section = "\n\n---\n## 参考文献\n\n"
            for i, ref in enumerate(unique_references):
                title = ref.get('title', '未知标题')
                source = ref.get('source', '未知来源')
                link = ref.get('link', '#')
                authors = ref.get('authors', [])
                published = ref.get('published', '')
                updated = ref.get('updated', '')
                
                # 格式化作者列表
                authors_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_str += " 等"
                
                # 构建参考文献条目
                references_section += f"{i+1}. **{title}**\n"
                references_section += f"   - 作者: {authors_str}\n"
                if published:
                    references_section += f"   - 发表时间: {published}\n"
                if updated and updated != published:
                    references_section += f"   - 更新时间: {updated}\n"
                references_section += f"   - 来源: {source}\n"
                references_section += f"   - 链接: [{link}]({link})\n"
                
                # 每两篇论文之间空一行，最后一篇不加空行
                if i < len(unique_references) - 1:
                    references_section += "\n"
                
            yield references_section
        else:
            yield "\n\n---\n未找到可引用的文献链接。"
            
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