#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的arXiv搜索脚本（基于 arxiv 库）
- 使用 arxiv.Search 进行检索
- 支持：中文查询翻译、类别过滤、时间范围过滤、排序
- 提取论文信息（含官方代码链接探测，可配置）
"""

import requests
import logging
import json
import os
import argparse
import sys
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import arxiv
import asyncio

class ArxivSearcher:
    """arXiv搜索器，用于搜索和提取论文信息"""
    
    def __init__(self, max_results: int = 20):
        """初始化arXiv搜索器"""
        logging.info(f"初始化ArxivSearcher: max_results={max_results}")
        self.max_results = max_results
        self.paperswithcode_api = "https://arxiv.paperswithcode.com/api/v0/papers/"
        self.github_api_url = "https://api.github.com/search/repositories"
        self.translate_api_url = "https://translate.googleapis.com/translate_a/single"
        # 通过环境变量控制是否抓取代码链接（默认关闭提速）
        self.enable_code_link = os.getenv("ENABLE_CODE_LINK", "0") in ("1", "true", "True")
    
    async def _translate_query(self, query: str) -> str:
        """将中文查询翻译成英文，优先 LLM，退回传统API，再退回简单映射"""
        if not re.search(r'[\u4e00-\u9fff]', query):
            return query
        logging.info(f"检测到中文查询，开始翻译: '{query}'")
        try:
            from src.core.llm_interface import LLMInterface
            llm = LLMInterface()
            prompt = f"请将以下中文学术查询翻译成英文关键词，只返回英文关键词：\n{query}"
            translated = await llm.call_llm(prompt)
            translated = translated.strip()
            if translated and not re.search(r'[\u4e00-\u9fff]', translated):
                logging.info(f"LLM翻译成功: '{query}' -> '{translated}'")
                return translated
        except Exception as e:
            logging.warning(f"LLM翻译失败，回退传统API: {e}")
        try:
            translated = await self._traditional_translate(query)
            if translated:
                logging.info(f"传统API翻译成功: '{query}' -> '{translated}'")
                return translated
        except Exception as e:
            logging.warning(f"传统API翻译失败，回退简单映射: {e}")
        mapped = self._simple_translate(query)
        logging.info(f"使用简单映射翻译: '{query}' -> '{mapped}'")
        return mapped

    async def _traditional_translate(self, text: str, source_lang="zh-CN", target_lang="en") -> str:
        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text
        }
        r = requests.get(self.translate_api_url, params=params, timeout=5)
        if r.status_code == 200:
            result = r.json()
            translations = [s[0] for s in result[0] if s and isinstance(s, list) and s[0]]
            return " ".join(translations)
        return ""

    def _simple_translate(self, query: str) -> str:
        translations = {
            "深度学习": "deep learning",
            "机器学习": "machine learning",
            "人工智能": "artificial intelligence",
            "自然语言处理": "natural language processing",
            "计算机视觉": "computer vision",
        }
        out = query
        for zh, en in translations.items():
            if zh in out:
                out = out.replace(zh, en)
        if out == query:
            english_terms = re.findall(r'[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)*', query)
            return " ".join(english_terms) if english_terms else "recent research"
        return out

    def _build_query(self, cleaned_query: str, categories: Optional[List[str]]) -> str:
        # 使用字段限定提升召回：标题/摘要/全文任一匹配
        phrase = cleaned_query.replace('"', '\\"').strip()
        term_clause = f'(ti:"{phrase}" OR abs:"{phrase}" OR all:"{phrase}")'
        if categories:
            cats = [c.strip() for c in categories if c and c.strip()]
            if cats:
                cat_expr = " OR ".join([f"cat:{c}" for c in cats])
                return f"({term_clause}) AND ({cat_expr})"
        return term_clause

    def _choose_sort(self, sort_by: str, time_range: Optional[Dict]) -> arxiv.SortCriterion:
        # 当有时间范围且未指定提交/更新排序时，默认用提交时间降序，保证第一页更可能是近期结果
        if time_range and sort_by in (None, "", "relevance"):
            return arxiv.SortCriterion.SubmittedDate
        if sort_by == "submittedDate":
            return arxiv.SortCriterion.SubmittedDate
        if sort_by == "lastUpdatedDate":
            return arxiv.SortCriterion.LastUpdatedDate
        return arxiv.SortCriterion.Relevance

    def search(self, 
               query: str, 
               sort_by: str = "relevance", 
               time_range: Optional[Dict[str, Union[str, int]]] = None,
               categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        # 翻译（如需）
        if re.search(r'[\u4e00-\u9fff]', query):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            cleaned_query = loop.run_until_complete(self._translate_query(query))
        else:
            cleaned_query = query
        if not cleaned_query:
            cleaned_query = "recent research"
        
        final_query = self._build_query(cleaned_query, categories)
        sort_criterion = self._choose_sort(sort_by, time_range)
        logging.info(f"使用ArxivSearcher搜索: {cleaned_query}, 排序方式: {sort_by or 'relevance'} -> {sort_criterion.name}, 时间范围: {time_range}, 类别: {categories}")
        logging.info(f"最终搜索查询: '{final_query}'")

        search_engine = arxiv.Search(
            query=final_query,
            max_results=max(self.max_results, 50) if time_range else self.max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # 执行并过滤
        results: List[Dict[str, Any]] = []
        date_filter: Optional[Tuple[datetime, datetime]] = None
        if time_range:
            if 'days' in time_range:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=int(time_range['days']))
                date_filter = (start_date, end_date)
            elif 'from' in time_range and 'to' in time_range:
                try:
                    start_date = datetime.strptime(time_range['from'], '%Y-%m-%d')
                    end_date = datetime.strptime(time_range['to'], '%Y-%m-%d')
                    date_filter = (start_date, end_date)
                except ValueError:
                    logging.warning(f"日期格式错误，应为YYYY-MM-DD: {time_range}")
        
        try:
            for result in search_engine.results():
                if date_filter:
                    paper_date = result.published.replace(tzinfo=None)
                    if paper_date < date_filter[0] or paper_date > date_filter[1]:
                        continue
                paper_info = self._extract_paper_info(result)
                if self.enable_code_link:
                    try:
                        code_url = self._get_code_link(paper_info["arxiv_id"])
                        if code_url:
                            paper_info["code_url"] = code_url
                    except Exception as e:
                        logging.debug(f"获取代码链接失败: {e}")
                results.append(paper_info)
                if len(results) >= self.max_results:
                    break
            logging.info(f"搜索完成，找到 {len(results)} 篇论文")
            return results
        except Exception as e:
            logging.error(f"arXiv搜索出错: {e}")
            return []

    def _extract_paper_info(self, result: Any) -> Dict[str, Any]:
        paper_id = result.get_short_id()
        ver_pos = paper_id.find('v')
        arxiv_id = paper_id if ver_pos == -1 else paper_id[:ver_pos]
        return {
            "title": result.title,
            "authors": [str(author) for author in result.authors],
            "abstract": result.summary.replace("\n", " "),
            "published": result.published.strftime("%Y-%m-%d"),
            "updated": result.updated.strftime("%Y-%m-%d"),
            "year": result.published.year,
            "month": result.published.month,
            "day": result.published.day,
            "categories": result.categories,
            "primary_category": result.primary_category,
            "arxiv_id": arxiv_id,
            "url": f"http://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}.pdf",
            "comment": result.comment,
            "source": "arXiv"
        }

    def _get_code_link(self, paper_id: str) -> Optional[str]:
        # Papers With Code
        try:
            code_url = self.paperswithcode_api + paper_id
            r = requests.get(code_url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if "official" in data and data["official"]:
                    return data["official"]["url"]
        except Exception as e:
            logging.debug(f"从Papers With Code获取代码链接失败: {e}")
        # GitHub 兜底
        try:
            params = {"q": f"arxiv:{paper_id}", "sort": "stars", "order": "desc"}
            r = requests.get(self.github_api_url, params=params, timeout=10)
            if r.status_code == 200:
                results = r.json()
                if results.get("total_count", 0) > 0:
                    return results["items"][0]["html_url"]
        except Exception as e:
            logging.debug(f"从GitHub获取代码链接失败: {e}")
        return None

def save_results_to_json(results: List[Dict[str, Any]], filename: str) -> None:
    """
    将搜索结果保存为JSON文件
    
    Args:
        results: 搜索结果列表
        filename: 输出文件名
    """
    print(f"保存结果到JSON: {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到 {filename}")
    except Exception as e:
        print(f"保存JSON时出错: {e}", file=sys.stderr)

def save_results_to_markdown(results: List[Dict[str, Any]], filename: str, topic: str = "arXiv搜索结果") -> None:
    """
    将搜索结果保存为Markdown文件
    
    Args:
        results: 搜索结果列表
        filename: 输出文件名
        topic: 主题名称
    """
    print(f"保存结果到Markdown: {filename}")
    try:
        markdown = ArxivSearcher().format_results_as_markdown(results, topic)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"结果已保存到 {filename}")
    except Exception as e:
        print(f"保存Markdown时出错: {e}", file=sys.stderr)

def main():
    """主函数"""
    print("开始执行arXiv搜索脚本")
    parser = argparse.ArgumentParser(description="优化的arXiv搜索工具")
    parser.add_argument("query", help="搜索查询字符串")
    parser.add_argument("--max", type=int, default=10, help="最大返回结果数 (默认: 10)")
    parser.add_argument("--sort-by", choices=["relevance", "lastUpdatedDate", "submittedDate"], 
                        default="relevance", help="排序方式 (默认: relevance)")
    parser.add_argument("--categories", help="限制搜索的类别，多个类别用逗号分隔")
    parser.add_argument("--year-start", type=int, help="开始年份")
    parser.add_argument("--year-end", type=int, help="结束年份")
    parser.add_argument("--output", help="输出文件名 (不包含扩展名)")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both", 
                        help="输出格式 (默认: both)")
    parser.add_argument("--topic", default="arXiv搜索结果", help="Markdown输出的主题名称")
    
    args = parser.parse_args()
    print(f"命令行参数: {args}")
    
    # 解析类别
    categories = None
    if args.categories:
        categories = args.categories.split(",")
        print(f"解析类别: {categories}")
    
    # 创建搜索器
    searcher = ArxivSearcher(max_results=args.max)
    
    # 执行搜索
    print(f"正在搜索: {args.query}")
    if categories:
        print(f"类别: {', '.join(categories)}")
    if args.year_start or args.year_end:
        year_range = f"{args.year_start or '不限'} - {args.year_end or '不限'}"
        print(f"年份范围: {year_range}")
    
    results = searcher.search(args.query, args.sort_by, {"from": args.year_start, "to": args.year_end} if args.year_start and args.year_end else None, categories)
    
    # 输出结果数量
    print(f"找到 {len(results)} 篇论文")
    
    # 保存结果
    if args.output:
        output_base = args.output
    else:
        # 使用当前日期和查询字符串作为默认文件名
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        query_str = args.query.replace(" ", "_")[:20]  # 限制长度
        output_base = f"arxiv_{date_str}_{query_str}"
    
    if args.format in ["json", "both"]:
        json_filename = f"{output_base}.json"
        save_results_to_json(results, json_filename)
    
    if args.format in ["markdown", "both"]:
        md_filename = f"{output_base}.md"
        save_results_to_markdown(results, md_filename, args.topic)
    
    print("脚本执行完成")

if __name__ == "__main__":
    main() 