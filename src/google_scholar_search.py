#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Scholar搜索脚本
- 使用scholarly库搜索Google Scholar上的论文
- 提取论文信息，包括发表时间
- 格式化输出结果
"""

import scholarly
import datetime
import json
import os
import time
import argparse
from typing import List, Dict, Any, Optional

class GoogleScholarSearcher:
    """Google Scholar搜索器，用于搜索和提取论文信息"""
    
    def __init__(self, max_results: int = 20):
        """
        初始化Google Scholar搜索器
        
        Args:
            max_results: 最大返回结果数
        """
        self.max_results = max_results
    
    def search(self, query: str, year_start: Optional[int] = None, year_end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        搜索Google Scholar论文
        
        Args:
            query: 搜索查询字符串
            year_start: 开始年份
            year_end: 结束年份
        
        Returns:
            包含论文信息的字典列表
        """
        # 构建查询
        search_query = scholarly.search_pubs(query)
        
        # 执行搜索并处理结果
        results = []
        count = 0
        
        for result in search_query:
            # 提取论文信息
            paper_info = self._extract_paper_info(result)
            
            # 根据年份筛选
            if year_start or year_end:
                pub_year = paper_info.get("year")
                if pub_year:
                    if year_start and pub_year < year_start:
                        continue
                    if year_end and pub_year > year_end:
                        continue
            
            results.append(paper_info)
            count += 1
            
            # 添加延迟，避免被Google封禁
            time.sleep(1)
            
            # 达到最大结果数时停止
            if count >= self.max_results:
                break
        
        return results
    
    def search_by_author(self, author_name: str, max_papers: int = 10) -> List[Dict[str, Any]]:
        """
        通过作者名搜索论文
        
        Args:
            author_name: 作者名
            max_papers: 每位作者最多返回的论文数
        
        Returns:
            包含论文信息的字典列表
        """
        # 搜索作者
        search_query = scholarly.search_author(author_name)
        
        try:
            # 获取第一个匹配的作者
            author = next(search_query)
            
            # 获取作者的完整信息
            author = scholarly.fill(author)
            
            # 获取作者的论文
            results = []
            count = 0
            
            for pub in author['publications']:
                if count >= max_papers:
                    break
                
                # 获取论文的完整信息
                try:
                    pub_filled = scholarly.fill(pub)
                    paper_info = self._extract_paper_info(pub_filled)
                    results.append(paper_info)
                    count += 1
                    
                    # 添加延迟，避免被Google封禁
                    time.sleep(1)
                except Exception as e:
                    print(f"获取论文信息时出错: {e}")
                    continue
            
            return results
        except StopIteration:
            print(f"未找到作者: {author_name}")
            return []
        except Exception as e:
            print(f"搜索作者时出错: {e}")
            return []
    
    def _extract_paper_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        从scholarly结果中提取论文信息
        
        Args:
            result: scholarly搜索结果
        
        Returns:
            包含论文信息的字典
        """
        # 提取基本信息
        title = result.get('bib', {}).get('title', 'Unknown Title')
        authors = result.get('bib', {}).get('author', [])
        first_author = authors[0] if authors else "Unknown"
        abstract = result.get('bib', {}).get('abstract', '')
        year = result.get('bib', {}).get('pub_year')
        
        # 提取URL
        url = result.get('pub_url', '')
        
        # 提取引用次数
        citations = result.get('num_citations', 0)
        
        # 提取期刊/会议信息
        venue = result.get('bib', {}).get('venue', '')
        
        # 构建论文信息字典
        paper_info = {
            "title": title,
            "authors": authors,
            "first_author": first_author,
            "abstract": abstract,
            "year": year,
            "url": url,
            "citations": citations,
            "venue": venue,
            "source": "Google Scholar"
        }
        
        return paper_info

def format_results_as_markdown(results: List[Dict[str, Any]], topic: str = "搜索结果") -> str:
    """
    将搜索结果格式化为Markdown表格
    
    Args:
        results: 搜索结果列表
        topic: 主题名称
    
    Returns:
        Markdown格式的表格字符串
    """
    if not results:
        return f"## {topic}\n\n没有找到相关论文。\n"
    
    # 构建表头
    markdown = f"## {topic}\n\n"
    markdown += "| 发表年份 | 标题 | 作者 | 期刊/会议 | 引用次数 |\n"
    markdown += "|---------|------|------|-----------|----------|\n"
    
    # 添加每篇论文的信息
    for paper in results:
        title = paper["title"].replace("|", "\\|")  # 转义表格分隔符
        authors = paper["first_author"] + " et al." if len(paper["authors"]) > 1 else paper["first_author"]
        authors = authors.replace("|", "\\|")  # 转义表格分隔符
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "N/A").replace("|", "\\|")
        citations = paper.get("citations", 0)
        url = paper.get("url", "")
        
        if url:
            title_cell = f"[{title}]({url})"
        else:
            title_cell = title
        
        markdown += f"| {year} | {title_cell} | {authors} | {venue} | {citations} |\n"
    
    return markdown

def save_results_to_json(results: List[Dict[str, Any]], filename: str) -> None:
    """
    将搜索结果保存为JSON文件
    
    Args:
        results: 搜索结果列表
        filename: 输出文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {filename}")

def save_results_to_markdown(results: List[Dict[str, Any]], filename: str, topic: str = "搜索结果") -> None:
    """
    将搜索结果保存为Markdown文件
    
    Args:
        results: 搜索结果列表
        filename: 输出文件名
        topic: 主题名称
    """
    markdown = format_results_as_markdown(results, topic)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"结果已保存到 {filename}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Google Scholar搜索工具")
    parser.add_argument("query", help="搜索查询字符串")
    parser.add_argument("--max", type=int, default=10, help="最大返回结果数 (默认: 10)")
    parser.add_argument("--year-start", type=int, help="开始年份")
    parser.add_argument("--year-end", type=int, help="结束年份")
    parser.add_argument("--author", action="store_true", help="按作者名搜索")
    parser.add_argument("--output", help="输出文件名 (不包含扩展名)")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both", 
                        help="输出格式 (默认: both)")
    parser.add_argument("--topic", default="Google Scholar搜索结果", help="Markdown输出的主题名称")
    
    args = parser.parse_args()
    
    # 创建搜索器
    searcher = GoogleScholarSearcher(max_results=args.max)
    
    # 执行搜索
    print(f"正在搜索: {args.query}")
    if args.year_start or args.year_end:
        year_range = f"{args.year_start or '不限'} - {args.year_end or '不限'}"
        print(f"年份范围: {year_range}")
    
    if args.author:
        print("按作者名搜索")
        results = searcher.search_by_author(args.query, max_papers=args.max)
    else:
        results = searcher.search(args.query, args.year_start, args.year_end)
    
    # 输出结果数量
    print(f"找到 {len(results)} 篇论文")
    
    # 保存结果
    if args.output:
        output_base = args.output
    else:
        # 使用当前日期和查询字符串作为默认文件名
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        query_str = args.query.replace(" ", "_")[:20]  # 限制长度
        output_base = f"scholar_{date_str}_{query_str}"
    
    if args.format in ["json", "both"]:
        json_filename = f"{output_base}.json"
        save_results_to_json(results, json_filename)
    
    if args.format in ["markdown", "both"]:
        md_filename = f"{output_base}.md"
        save_results_to_markdown(results, md_filename, args.topic)

if __name__ == "__main__":
    main() 