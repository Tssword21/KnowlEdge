#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的arXiv搜索脚本
- 使用arXiv API搜索论文
- 提取论文信息，包括发表时间
- 格式化输出结果
"""

import requests
import logging
import json
import os
import argparse
import sys
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import arxiv
import asyncio

class ArxivSearcher:
    """arXiv搜索器，用于搜索和提取论文信息"""
    
    def __init__(self, max_results: int = 20):
        """
        初始化arXiv搜索器
        
        Args:
            max_results: 最大返回结果数
        """
        print(f"初始化ArxivSearcher: max_results={max_results}")
        self.max_results = max_results
        self.base_url = "https://export.arxiv.org/api/query"
        self.github_api_url = "https://api.github.com/search/repositories"
        self.paperswithcode_api = "https://arxiv.paperswithcode.com/api/v0/papers/"
        self.translate_api_url = "https://translate.googleapis.com/translate_a/single"
        
    async def _translate_query(self, query: str) -> str:
        """将中文查询翻译成英文，使用LLM服务和传统翻译API作为备份"""
        if not re.search(r'[\u4e00-\u9fff]', query):  # 如果不包含中文字符
            return query
            
        print(f"检测到中文查询，开始翻译: '{query}'")
        
        # 方法1: 使用LLM翻译
        try:
            from src.core.llm_interface import LLMInterface
            llm = LLMInterface()
            
            prompt = f"""
            请将以下中文学术查询翻译成英文关键词，只保留核心术语和概念，不要添加任何解释。
            格式应该是简洁的英文关键词组合，适合在学术搜索引擎中使用。
            
            中文查询: {query}
            
            英文关键词:
            """
            
            translated = await llm.call_llm(prompt)
            translated = translated.strip()
            if translated and len(translated) > 3:  # 确保翻译结果有效
                print(f"LLM翻译成功: '{query}' -> '{translated}'")
                return translated
            else:
                print(f"LLM翻译结果无效: '{translated}'，尝试其他方法")
        except Exception as e:
            print(f"LLM查询翻译失败: {e}，尝试其他方法")
        
        # 方法2: 使用传统翻译API
        try:
            translated = await self._traditional_translate(query)
            if translated and len(translated) > 3:  # 确保翻译结果有效
                print(f"传统API翻译成功: '{query}' -> '{translated}'")
                return translated
            else:
                print(f"传统API翻译结果无效: '{translated}'，尝试其他方法")
        except Exception as e:
            print(f"传统API翻译失败: {e}，尝试其他方法")
        
        # 方法3: 使用简单的映射表
        translated = self._simple_translate(query)
        print(f"使用简单映射翻译: '{query}' -> '{translated}'")
        return translated
    
    async def _traditional_translate(self, text: str, source_lang="zh-CN", target_lang="en") -> str:
        """使用传统翻译API翻译文本"""
        try:
            # 尝试使用Google翻译API
            params = {
                "client": "gtx",
                "sl": source_lang,
                "tl": target_lang,
                "dt": "t",
                "q": text
            }
            
            response = requests.get(self.translate_api_url, params=params, timeout=5)
            if response.status_code == 200:
                # 解析响应
                result = response.json()
                if result and isinstance(result, list) and len(result) > 0:
                    translations = []
                    for sentence in result[0]:
                        if sentence and isinstance(sentence, list) and len(sentence) > 0:
                            translations.append(sentence[0])
                    return " ".join(translations)
            
            # 如果Google翻译失败，尝试使用备用翻译API
            # 这里可以添加其他翻译API的调用
            
            return ""  # 如果所有API都失败，返回空字符串
        except Exception as e:
            print(f"传统翻译API调用失败: {e}")
            return ""
    
    def _simple_translate(self, query: str) -> str:
        """简单的中文关键词映射"""
        # 常见AI/ML术语的中英文映射
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
        
        # 检查查询中是否包含已知的中文术语，如果有则替换
        translated_query = query
        for zh, en in translations.items():
            if zh in query:
                translated_query = translated_query.replace(zh, en)
        
        # 如果没有任何替换发生，提取英文单词
        if translated_query == query:
            english_terms = re.findall(r'[a-zA-Z0-9]+(?:\s+[a-zA-Z0-9]+)*', query)
            if english_terms:
                return " ".join(english_terms)
            else:
                # 如果没有提取到英文单词，返回一个通用查询
                return "recent research papers"
            
        return translated_query

    def search(self, 
               query: str, 
               sort_by: str = "relevance", 
               time_range: Optional[Dict[str, Union[str, int]]] = None,
               categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        搜索arXiv论文
        
        Args:
            query: 搜索查询字符串
            sort_by: 排序方式，可选值：relevance、lastUpdatedDate、submittedDate
            time_range: 时间范围，格式为 {'from': '2023-01-01', 'to': '2023-12-31'} 或 {'days': 30}
            categories: 限制搜索的类别列表，如 ['cs.AI', 'cs.CL']
            
        Returns:
            包含论文信息的字典列表
        """
        # 检测是否包含中文字符，如果包含则进行翻译
        if re.search(r'[\u4e00-\u9fff]', query):
            # 创建一个异步运行时环境来执行异步翻译函数
            try:
                # 尝试获取当前事件循环
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # 如果没有事件循环（例如在非异步环境中），创建一个新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 执行异步翻译，等待翻译完成再继续
            cleaned_query = loop.run_until_complete(self._translate_query(query))
            print(f"翻译后的查询: '{cleaned_query}'")
            
            # 确保查询已被翻译成英文
            if not cleaned_query or re.search(r'[\u4e00-\u9fff]', cleaned_query):
                print(f"警告: 翻译结果仍包含中文或为空，使用备用翻译")
                cleaned_query = self._simple_translate(query)
                print(f"备用翻译结果: '{cleaned_query}'")
        else:
            cleaned_query = query
            
        # 确保查询不为空
        if not cleaned_query or cleaned_query.strip() == "":
            cleaned_query = "recent research papers"
            print(f"警告: 查询为空，使用默认查询: '{cleaned_query}'")
            
        print(f"最终搜索查询: '{cleaned_query}'")
        
        # 处理时间范围
        date_filter = None
        if time_range:
            if 'days' in time_range:
                # 如果指定了天数，计算从现在到过去N天的范围
                days = int(time_range['days'])
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                date_filter = (start_date, end_date)
            elif 'from' in time_range and 'to' in time_range:
                # 如果指定了具体日期范围
                try:
                    start_date = datetime.strptime(time_range['from'], '%Y-%m-%d')
                    end_date = datetime.strptime(time_range['to'], '%Y-%m-%d')
                    date_filter = (start_date, end_date)
                except ValueError:
                    print(f"日期格式错误，应为YYYY-MM-DD: {time_range}")
        
        # 转换排序方式为arxiv库的格式
        sort_criterion = arxiv.SortCriterion.Relevance
        if sort_by == "submittedDate":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        elif sort_by == "lastUpdatedDate":
            sort_criterion = arxiv.SortCriterion.LastUpdatedDate
            
        # 构建搜索过滤器
        search_filters = []
        
        # 添加类别过滤
        if categories and len(categories) > 0:
            for category in categories:
                search_filters.append(arxiv.SortCriterion.Category(category))
        
        # 创建搜索引擎
        print(f"执行搜索: query='{cleaned_query}', sort_by={sort_by}")
        search_engine = arxiv.Search(
            query=cleaned_query,
            max_results=self.max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # 执行搜索并处理结果
        results = []
        try:
            for result in search_engine.results():
                # 检查日期过滤
                if date_filter:
                    paper_date = result.published.replace(tzinfo=None)  # 移除时区信息以便比较
                    if paper_date < date_filter[0] or paper_date > date_filter[1]:
                        continue
                
                # 提取论文信息
                paper_info = self._extract_paper_info(result)
                
                # 尝试获取代码链接
                try:
                    code_url = self._get_code_link(paper_info["arxiv_id"])
                    if code_url:
                        paper_info["code_url"] = code_url
                except Exception as e:
                    print(f"获取代码链接失败: {e}")
                
                results.append(paper_info)
                print(f"处理结果: {paper_info['title']}")
            
            print(f"搜索完成，找到 {len(results)} 篇论文")
            return results
            
        except Exception as e:
            logging.error(f"arXiv搜索出错: {e}")
            print(f"arXiv搜索出错: {e}")
            return []
    
    def _extract_paper_info(self, result: Any) -> Dict[str, Any]:
        """从arxiv结果中提取论文信息"""
        paper_id = result.get_short_id()
        # 移除版本号，如 2108.09112v1 -> 2108.09112
        ver_pos = paper_id.find('v')
        if ver_pos != -1:
            arxiv_id = paper_id[0:ver_pos]
        else:
            arxiv_id = paper_id
            
        # 构建论文信息字典
        paper_info = {
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
        return paper_info
            
    def _get_code_link(self, paper_id: str) -> Optional[str]:
        """尝试获取论文相关的代码库链接"""
        # 首先尝试从Papers With Code获取官方代码链接
        try:
            code_url = self.paperswithcode_api + paper_id
            r = requests.get(code_url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if "official" in data and data["official"]:
                    return data["official"]["url"]
        except Exception as e:
            print(f"从Papers With Code获取代码链接失败: {e}")
        
        # 如果没有找到官方链接，尝试在GitHub搜索
        try:
            query = f"arxiv:{paper_id}"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc"
            }
            r = requests.get(self.github_api_url, params=params, timeout=10)
            if r.status_code == 200:
                results = r.json()
                if results["total_count"] > 0:
                    return results["items"][0]["html_url"]
        except Exception as e:
            print(f"从GitHub获取代码链接失败: {e}")
        
        return None
        
    def format_results_as_markdown(self, results: List[Dict[str, Any]], topic: str = "arXiv搜索结果") -> str:
        """
        将搜索结果格式化为Markdown表格
        
        Args:
            results: 搜索结果列表
            topic: 主题名称
        
        Returns:
            Markdown格式的表格字符串
        """
        print(f"格式化为Markdown: {len(results)} 篇论文")
        if not results:
            return f"## {topic}\n\n没有找到相关论文。\n"
        
        # 构建表头
        markdown = f"## {topic}\n\n"
        markdown += "| 发表日期 | 标题 | 作者 | 类别 | 链接 |\n"
        markdown += "|---------|------|------|------|------|\n"
        
        # 添加每篇论文的信息
        for paper in results:
            title = paper["title"].replace("|", "\\|").replace("\n", " ")  # 转义表格分隔符并移除换行
            authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors += " et al."
            authors = authors.replace("|", "\\|")  # 转义表格分隔符
            
            published = paper.get("published", "N/A")
            categories = paper.get("primary_category", "N/A")
            url = paper.get("url", "#")
            pdf_url = paper.get("pdf_url", "#")
            
            # 添加代码链接（如果有）
            links = f"[arXiv]({url}) | [PDF]({pdf_url})"
            if "code_url" in paper and paper["code_url"]:
                links += f" | [Code]({paper['code_url']})"
            
            markdown += f"| {published} | {title} | {authors} | {categories} | {links} |\n"
        
        return markdown

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