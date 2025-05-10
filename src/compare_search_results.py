#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学术搜索结果比较工具
- 比较arXiv和Google Scholar的搜索结果
- 合并结果并生成综合报告
- 识别重复论文并突出显示差异
"""

import json
import os
import argparse
import datetime
from typing import List, Dict, Any, Tuple
import difflib
from improved_arxiv_search import ArxivSearcher, format_results_as_markdown as arxiv_format_markdown
from google_scholar_search import GoogleScholarSearcher, format_results_as_markdown as scholar_format_markdown

def load_results_from_json(filename: str) -> List[Dict[str, Any]]:
    """
    从JSON文件加载搜索结果
    
    Args:
        filename: JSON文件名
    
    Returns:
        搜索结果列表
    """
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return []

def find_similar_papers(arxiv_results: List[Dict[str, Any]], 
                        scholar_results: List[Dict[str, Any]], 
                        similarity_threshold: float = 0.8) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """
    查找两个结果集中相似的论文
    
    Args:
        arxiv_results: arXiv搜索结果
        scholar_results: Google Scholar搜索结果
        similarity_threshold: 相似度阈值
    
    Returns:
        相似论文对的列表，每对包含arXiv论文、Scholar论文和相似度
    """
    similar_papers = []
    
    for arxiv_paper in arxiv_results:
        arxiv_title = arxiv_paper.get("title", "").lower()
        if not arxiv_title:
            continue
        
        for scholar_paper in scholar_results:
            scholar_title = scholar_paper.get("title", "").lower()
            if not scholar_title:
                continue
            
            # 计算标题相似度
            similarity = difflib.SequenceMatcher(None, arxiv_title, scholar_title).ratio()
            
            if similarity >= similarity_threshold:
                similar_papers.append((arxiv_paper, scholar_paper, similarity))
    
    # 按相似度降序排序
    similar_papers.sort(key=lambda x: x[2], reverse=True)
    
    return similar_papers

def merge_results(arxiv_results: List[Dict[str, Any]], 
                 scholar_results: List[Dict[str, Any]], 
                 similarity_threshold: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], Dict[str, Any], float]]]:
    """
    合并两个结果集，去除重复项
    
    Args:
        arxiv_results: arXiv搜索结果
        scholar_results: Google Scholar搜索结果
        similarity_threshold: 相似度阈值
    
    Returns:
        合并后的结果列表和相似论文对列表
    """
    # 查找相似论文
    similar_papers = find_similar_papers(arxiv_results, scholar_results, similarity_threshold)
    
    # 创建已匹配论文的集合
    matched_arxiv_indices = set()
    matched_scholar_indices = set()
    
    for i, (arxiv_paper, scholar_paper, _) in enumerate(similar_papers):
        # 获取论文在原始列表中的索引
        arxiv_index = next((i for i, p in enumerate(arxiv_results) if p.get("title") == arxiv_paper.get("title")), None)
        scholar_index = next((i for i, p in enumerate(scholar_results) if p.get("title") == scholar_paper.get("title")), None)
        
        if arxiv_index is not None and scholar_index is not None:
            matched_arxiv_indices.add(arxiv_index)
            matched_scholar_indices.add(scholar_index)
    
    # 合并结果
    merged_results = []
    
    # 添加匹配的论文（使用arXiv的信息，但补充Google Scholar的引用信息）
    for i, (arxiv_paper, scholar_paper, _) in enumerate(similar_papers):
        merged_paper = arxiv_paper.copy()
        
        # 补充Google Scholar的信息
        if "citations" not in merged_paper or merged_paper["citations"] == 0:
            merged_paper["citations"] = scholar_paper.get("citations", 0)
        
        if "venue" not in merged_paper or not merged_paper["venue"]:
            merged_paper["venue"] = scholar_paper.get("venue", "")
        
        # 标记来源
        merged_paper["source"] = "arXiv & Google Scholar"
        
        merged_results.append(merged_paper)
    
    # 添加仅在arXiv中找到的论文
    for i, paper in enumerate(arxiv_results):
        if i not in matched_arxiv_indices:
            paper_copy = paper.copy()
            paper_copy["source"] = "arXiv"
            merged_results.append(paper_copy)
    
    # 添加仅在Google Scholar中找到的论文
    for i, paper in enumerate(scholar_results):
        if i not in matched_scholar_indices:
            paper_copy = paper.copy()
            paper_copy["source"] = "Google Scholar"
            merged_results.append(paper_copy)
    
    # 按发表年份和引用次数排序
    merged_results.sort(key=lambda x: (-(x.get("year") or 0), -(x.get("citations") or 0)))
    
    return merged_results, similar_papers

def format_merged_results_as_markdown(merged_results: List[Dict[str, Any]], topic: str = "合并搜索结果") -> str:
    """
    将合并的搜索结果格式化为Markdown表格
    
    Args:
        merged_results: 合并的搜索结果列表
        topic: 主题名称
    
    Returns:
        Markdown格式的表格字符串
    """
    if not merged_results:
        return f"## {topic}\n\n没有找到相关论文。\n"
    
    # 构建表头
    markdown = f"## {topic}\n\n"
    markdown += "| 发表年份 | 标题 | 作者 | 期刊/会议 | 引用次数 | 来源 |\n"
    markdown += "|---------|------|------|-----------|----------|------|\n"
    
    # 添加每篇论文的信息
    for paper in merged_results:
        title = paper.get("title", "Unknown Title").replace("|", "\\|")  # 转义表格分隔符
        
        # 处理作者
        if "authors" in paper and paper["authors"]:
            if isinstance(paper["authors"], list):
                first_author = paper["authors"][0] if paper["authors"] else "Unknown"
                authors = first_author + " et al." if len(paper["authors"]) > 1 else first_author
            else:
                authors = paper["authors"]
        else:
            authors = paper.get("author", "Unknown")
        
        authors = str(authors).replace("|", "\\|")  # 转义表格分隔符
        
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "N/A").replace("|", "\\|")
        citations = paper.get("citations", 0)
        source = paper.get("source", "Unknown")
        url = paper.get("url", "")
        
        if url:
            title_cell = f"[{title}]({url})"
        else:
            title_cell = title
        
        markdown += f"| {year} | {title_cell} | {authors} | {venue} | {citations} | {source} |\n"
    
    return markdown

def format_similar_papers_as_markdown(similar_papers: List[Tuple[Dict[str, Any], Dict[str, Any], float]], 
                                     topic: str = "相似论文比较") -> str:
    """
    将相似论文对格式化为Markdown表格
    
    Args:
        similar_papers: 相似论文对列表
        topic: 主题名称
    
    Returns:
        Markdown格式的表格字符串
    """
    if not similar_papers:
        return f"## {topic}\n\n没有找到相似论文。\n"
    
    # 构建表头
    markdown = f"## {topic}\n\n"
    markdown += "| 相似度 | arXiv标题 | Google Scholar标题 | arXiv年份 | Scholar年份 | arXiv引用 | Scholar引用 |\n"
    markdown += "|--------|-----------|-------------------|-----------|------------|-----------|-------------|\n"
    
    # 添加每对相似论文的信息
    for arxiv_paper, scholar_paper, similarity in similar_papers:
        arxiv_title = arxiv_paper.get("title", "Unknown").replace("|", "\\|")
        scholar_title = scholar_paper.get("title", "Unknown").replace("|", "\\|")
        
        arxiv_year = arxiv_paper.get("year", "N/A")
        scholar_year = scholar_paper.get("year", "N/A")
        
        arxiv_citations = arxiv_paper.get("citations", 0)
        scholar_citations = scholar_paper.get("citations", 0)
        
        arxiv_url = arxiv_paper.get("url", "")
        scholar_url = scholar_paper.get("url", "")
        
        if arxiv_url:
            arxiv_title_cell = f"[{arxiv_title}]({arxiv_url})"
        else:
            arxiv_title_cell = arxiv_title
            
        if scholar_url:
            scholar_title_cell = f"[{scholar_title}]({scholar_url})"
        else:
            scholar_title_cell = scholar_title
        
        markdown += f"| {similarity:.2f} | {arxiv_title_cell} | {scholar_title_cell} | {arxiv_year} | {scholar_year} | {arxiv_citations} | {scholar_citations} |\n"
    
    return markdown

def generate_comprehensive_report(arxiv_results: List[Dict[str, Any]], 
                                 scholar_results: List[Dict[str, Any]], 
                                 merged_results: List[Dict[str, Any]],
                                 similar_papers: List[Tuple[Dict[str, Any], Dict[str, Any], float]],
                                 query: str,
                                 output_filename: str) -> None:
    """
    生成综合报告
    
    Args:
        arxiv_results: arXiv搜索结果
        scholar_results: Google Scholar搜索结果
        merged_results: 合并的搜索结果
        similar_papers: 相似论文对列表
        query: 搜索查询
        output_filename: 输出文件名
    """
    # 生成报告标题和摘要
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"# 学术搜索综合报告\n\n"
    report += f"**查询**: {query}  \n"
    report += f"**生成时间**: {now}  \n"
    report += f"**arXiv结果数**: {len(arxiv_results)}  \n"
    report += f"**Google Scholar结果数**: {len(scholar_results)}  \n"
    report += f"**合并结果数**: {len(merged_results)}  \n"
    report += f"**相似论文对数**: {len(similar_papers)}  \n\n"
    
    # 添加统计信息
    arxiv_only = sum(1 for paper in merged_results if paper.get("source") == "arXiv")
    scholar_only = sum(1 for paper in merged_results if paper.get("source") == "Google Scholar")
    both_sources = sum(1 for paper in merged_results if paper.get("source") == "arXiv & Google Scholar")
    
    report += "## 统计信息\n\n"
    report += f"- 仅在arXiv中找到: {arxiv_only}篇论文\n"
    report += f"- 仅在Google Scholar中找到: {scholar_only}篇论文\n"
    report += f"- 两个来源都有: {both_sources}篇论文\n\n"
    
    # 添加合并结果
    report += format_merged_results_as_markdown(merged_results, "合并搜索结果")
    report += "\n\n"
    
    # 添加相似论文比较
    report += format_similar_papers_as_markdown(similar_papers, "相似论文比较")
    report += "\n\n"
    
    # 添加各来源的原始结果
    report += arxiv_format_markdown(arxiv_results, "arXiv原始结果")
    report += "\n\n"
    
    report += scholar_format_markdown(scholar_results, "Google Scholar原始结果")
    
    # 保存报告
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"综合报告已保存到 {output_filename}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="学术搜索结果比较工具")
    parser.add_argument("query", help="搜索查询字符串")
    parser.add_argument("--arxiv-file", help="arXiv搜索结果JSON文件")
    parser.add_argument("--scholar-file", help="Google Scholar搜索结果JSON文件")
    parser.add_argument("--max", type=int, default=20, help="每个来源的最大结果数 (默认: 20)")
    parser.add_argument("--similarity", type=float, default=0.8, help="标题相似度阈值 (默认: 0.8)")
    parser.add_argument("--output", help="输出文件名 (不包含扩展名)")
    parser.add_argument("--year-start", type=int, help="开始年份")
    parser.add_argument("--year-end", type=int, help="结束年份")
    
    args = parser.parse_args()
    
    # 设置输出文件名
    if args.output:
        output_base = args.output
    else:
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        query_str = args.query.replace(" ", "_")[:20]
        output_base = f"compare_{date_str}_{query_str}"
    
    report_filename = f"{output_base}.md"
    
    # 获取arXiv结果
    if args.arxiv_file:
        print(f"从文件加载arXiv结果: {args.arxiv_file}")
        arxiv_results = load_results_from_json(args.arxiv_file)
    else:
        print(f"正在搜索arXiv: {args.query}")
        arxiv_searcher = ArxivSearcher(max_results=args.max)
        arxiv_results = arxiv_searcher.search(args.query, args.year_start, args.year_end)
    
    # 获取Google Scholar结果
    if args.scholar_file:
        print(f"从文件加载Google Scholar结果: {args.scholar_file}")
        scholar_results = load_results_from_json(args.scholar_file)
    else:
        print(f"正在搜索Google Scholar: {args.query}")
        scholar_searcher = GoogleScholarSearcher(max_results=args.max)
        scholar_results = scholar_searcher.search(args.query, args.year_start, args.year_end)
    
    print(f"找到 {len(arxiv_results)} 篇arXiv论文")
    print(f"找到 {len(scholar_results)} 篇Google Scholar论文")
    
    # 合并结果
    print("正在合并和比较结果...")
    merged_results, similar_papers = merge_results(arxiv_results, scholar_results, args.similarity)
    
    print(f"合并后共有 {len(merged_results)} 篇论文")
    print(f"找到 {len(similar_papers)} 对相似论文")
    
    # 生成报告
    print("正在生成综合报告...")
    generate_comprehensive_report(
        arxiv_results, 
        scholar_results, 
        merged_results, 
        similar_papers, 
        args.query, 
        report_filename
    )

if __name__ == "__main__":
    main() 