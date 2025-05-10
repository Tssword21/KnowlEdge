#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的arXiv搜索脚本
- 使用arXiv API搜索论文
- 提取论文信息，包括发表时间
- 格式化输出结果
"""

import arxiv
import datetime
import json
import os
import argparse
import sys
from typing import List, Dict, Any, Optional

class ArxivSearcher:
    """arXiv搜索器，用于搜索和提取论文信息"""
    
    def __init__(self, max_results: int = 20, sort_by: str = "relevance"):
        """
        初始化arXiv搜索器
        
        Args:
            max_results: 最大返回结果数
            sort_by: 排序方式，可选值：relevance、lastUpdatedDate、submittedDate
        """
        print(f"初始化ArxivSearcher: max_results={max_results}, sort_by={sort_by}")
        self.max_results = max_results
        self.sort_by = sort_by
        
        # 映射排序方式到arXiv API的排序参数
        self.sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate
        }
    
    def search(self, query: str, year_start: Optional[int] = None, year_end: Optional[int] = None, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        搜索arXiv论文
        
        Args:
            query: 搜索查询字符串
            year_start: 开始年份
            year_end: 结束年份
            categories: 限制搜索的类别列表
        
        Returns:
            包含论文信息的字典列表
        """
        # 构建查询
        search_query = query
        
        # 添加类别过滤
        if categories:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query}) AND ({category_filter})"
        
        print(f"执行搜索: query='{search_query}'")
        
        # 执行搜索
        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=self.max_results,
            sort_by=self.sort_map.get(self.sort_by, arxiv.SortCriterion.Relevance)
        )
        
        # 获取结果
        results = []
        try:
            print("开始获取搜索结果...")
            for result in client.results(search):
                print(f"处理结果: {result.title}")
                # 提取论文信息
                paper_info = self._extract_paper_info(result)
                
                # 根据年份筛选
                if year_start or year_end:
                    pub_year = paper_info.get("year")
                    if pub_year:
                        if year_start and pub_year < year_start:
                            print(f"跳过 {result.title} (年份 {pub_year} < {year_start})")
                            continue
                        if year_end and pub_year > year_end:
                            print(f"跳过 {result.title} (年份 {pub_year} > {year_end})")
                            continue
                
                results.append(paper_info)
                print(f"已添加结果: {len(results)}/{self.max_results}")
                
                if len(results) >= self.max_results:
                    print(f"已达到最大结果数 {self.max_results}")
                    break
        except Exception as e:
            print(f"搜索过程中出错: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        
        print(f"搜索完成，找到 {len(results)} 篇论文")
        return results
    
    def search_by_ids(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        通过论文ID搜索
        
        Args:
            paper_ids: arXiv论文ID列表
        
        Returns:
            包含论文信息的字典列表
        """
        print(f"通过ID搜索: {paper_ids}")
        client = arxiv.Client()
        search = arxiv.Search(id_list=paper_ids)
        
        results = []
        try:
            for result in client.results(search):
                print(f"处理ID结果: {result.title}")
                paper_info = self._extract_paper_info(result)
                results.append(paper_info)
        except Exception as e:
            print(f"ID搜索过程中出错: {e}", file=sys.stderr)
        
        print(f"ID搜索完成，找到 {len(results)} 篇论文")
        return results
    
    def _extract_paper_info(self, result: arxiv.Result) -> Dict[str, Any]:
        """
        从arXiv结果中提取论文信息
        
        Args:
            result: arXiv搜索结果
        
        Returns:
            包含论文信息的字典
        """
        try:
            # 提取作者信息
            authors = [author.name for author in result.authors]
            
            # 提取发表时间
            published = result.published
            updated = result.updated
            
            # 提取类别
            categories = [category for category in result.categories]
            
            # 构建论文信息字典
            paper_info = {
                "title": result.title,
                "authors": authors,
                "abstract": result.summary,
                "published": published.strftime("%Y-%m-%d"),
                "updated": updated.strftime("%Y-%m-%d"),
                "year": published.year,
                "month": published.month,
                "day": published.day,
                "categories": categories,
                "primary_category": result.primary_category,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "doi": result.doi,
                "journal_ref": result.journal_ref,
                "comment": result.comment,
                "citations": 0,  # arXiv API不提供引用信息
                "source": "arXiv"
            }
            
            return paper_info
        except Exception as e:
            print(f"提取论文信息时出错: {e}", file=sys.stderr)
            # 返回一个最小的论文信息字典
            return {
                "title": getattr(result, "title", "Unknown Title"),
                "authors": [],
                "abstract": "Error extracting information",
                "source": "arXiv"
            }

def format_results_as_markdown(results: List[Dict[str, Any]], topic: str = "arXiv搜索结果") -> str:
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
        
        # 创建链接
        links = f"[arXiv]({url}) | [PDF]({pdf_url})"
        
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
        markdown = format_results_as_markdown(results, topic)
        
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
    searcher = ArxivSearcher(max_results=args.max, sort_by=args.sort_by)
    
    # 执行搜索
    print(f"正在搜索: {args.query}")
    if categories:
        print(f"类别: {', '.join(categories)}")
    if args.year_start or args.year_end:
        year_range = f"{args.year_start or '不限'} - {args.year_end or '不限'}"
        print(f"年份范围: {year_range}")
    
    results = searcher.search(args.query, args.year_start, args.year_end, categories)
    
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