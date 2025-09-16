"""
搜索模块
提供各种搜索功能，如Google搜索、ArXiv搜索等
"""
import logging
import requests
from xml.etree import ElementTree
import json
import asyncio
import os
import sys
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 修复导入路径问题
try:
    # 当在项目根目录运行时
    from src.config import Config
    from src.improved_arxiv_search import ArxivSearcher
except ImportError:
    # 当在src目录下运行时
    from config import Config
    from improved_arxiv_search import ArxivSearcher

# 初始化配置
config = Config()

_REQUEST_TIMEOUT = 15

class SearchEngine:
    """基础搜索引擎类"""
    
    def __init__(self, config: Config):
        """初始化搜索引擎
        
        Args:
            config: 配置对象
        """
        self.config = config
        
    async def search(self, query: str, max_results: int = 10) -> Dict:
        """执行搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            搜索结果字典
        """
        # 基类方法，子类需要重写
        raise NotImplementedError("子类必须实现search方法")

class WebSearch(SearchEngine):
    """Web搜索引擎"""
    
    async def search(self, query: str, max_results: int = 10) -> Dict:
        """执行Web搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            搜索结果字典
        """
        return await asyncio.to_thread(google_search, query, max_results)

class ArxivSearch(SearchEngine):
    """ArXiv搜索引擎"""
    
    async def search(self, query: str, max_results: int = 10, sort_by: str = "relevance", 
                    time_range: Dict = None, categories: List = None) -> Dict:
        """执行ArXiv搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            sort_by: 排序方式，可选值：relevance、lastUpdatedDate、submittedDate
            time_range: 时间范围，格式为 {'from': '2023-01-01', 'to': '2023-12-31'} 或 {'days': 30}
            categories: 限制搜索的类别列表，如 ['cs.AI', 'cs.CL']
            
        Returns:
            搜索结果字典
        """
        try:
            logging.info(f"使用ArxivSearcher搜索: {query}, 排序方式: {sort_by}, 时间范围: {time_range}, 类别: {categories}")
            arxiv_searcher = ArxivSearcher(max_results=max_results)
            results = await asyncio.to_thread(
                arxiv_searcher.search, 
                query, 
                sort_by=sort_by,
                time_range=time_range,
                categories=categories
            )
            formatted_results = []
            for paper in results:
                result = {
                    "title": paper.get("title", ""),
                    "link": paper.get("url", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors": paper.get("authors", []),
                    "published": paper.get("published", ""),
                    "categories": paper.get("categories", []),
                    "source": "ArXiv",
                    "pdf_url": paper.get("pdf_url", ""),
                    "arxiv_id": paper.get("arxiv_id", "")
                }
                if "code_url" in paper:
                    result["code_url"] = paper["code_url"]
                formatted_results.append(result)
            return {
                "query": query,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "sort_by": sort_by,
                "time_range": time_range,
                "categories": categories
            }
        except Exception as e:
            logging.error(f"ArXiv真实搜索出错: {e}")
            return await asyncio.to_thread(arxiv_search, query, max_results)

class GoogleScholarSearch(SearchEngine):
    """Google学术搜索引擎"""
    
    async def search(self, query: str, max_results: int = 10) -> Dict:
        logging.info(f"执行Google Scholar搜索: {query}")
        scholarly_query = f"{query} academic research paper"
        return await asyncio.to_thread(google_search, scholarly_query, max_results)

class PatentSearch(SearchEngine):
    """专利搜索引擎"""
    
    async def search(self, query: str, max_results: int = 10) -> Dict:
        logging.info(f"执行专利搜索: {query}")
        patent_query = f"{query} patent"
        return await asyncio.to_thread(google_search, patent_query, max_results)

class NewsSearch(SearchEngine):
    """新闻搜索引擎"""
    
    async def search(self, query: str, max_results: int = 10) -> Dict:
        logging.info(f"执行新闻搜索: {query}")
        news_query = f"{query} news recent"
        return await asyncio.to_thread(google_search, news_query, max_results)

def google_search(query: str, max_results: int = 10) -> Dict:
    logging.info(f"执行Google搜索: {query}")
    try:
        if not config.serper_api_key:
            logging.warning("SERPER_API_KEY 未配置，跳过Google搜索")
            return {"query": query, "results": [], "result_count": 0, "warning": "SERPER_API_KEY 未配置"}
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": max_results
        })
        headers = {
            'X-API-KEY': config.serper_api_key,
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return parse_google_results(data, query)
    except requests.Timeout:
        logging.error("Google搜索超时")
        return {"error": "timeout", "query": query, "results": []}
    except Exception as e:
        logging.error(f"Google搜索出错: {e}")
        return {"error": str(e), "query": query, "results": []}

def parse_google_results(data: Dict, query: str) -> Dict:
    try:
        results = []
        if "organic" in data:
            for item in data["organic"]:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position", 0),
                    "source": "Google搜索"
                }
                results.append(result)
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            result = {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "link": kg.get("website", ""),
                "snippet": kg.get("description", ""),
                "source": "Google知识图谱"
            }
            results.append(result)
        return {
            "query": query,
            "results": results,
            "result_count": len(results)
        }
    except Exception as e:
        logging.error(f"解析Google搜索结果时出错: {e}")
        return {"error": str(e), "query": query, "results": []}

def arxiv_search(query: str, max_results: int = 10) -> Dict:
    logging.info(f"执行arXiv搜索: {query}")
    try:
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        response = requests.get(base_url, params=params, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        return parse_arxiv_response(response.text, query)
    except requests.Timeout:
        logging.error("arXiv搜索超时")
        return {"error": "timeout", "query": query, "results": []}
    except Exception as e:
        logging.error(f"arXiv搜索出错: {e}")
        return {"error": str(e), "query": query, "results": []}

def parse_arxiv_response(xml_data: str, query: str) -> Dict:
    try:
        root = ElementTree.fromstring(xml_data)
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"
        }
        results = []
        for entry in root.findall(".//atom:entry", namespaces):
            title_elem = entry.find("atom:title", namespaces)
            title = title_elem.text if title_elem is not None else ""
            link = ""
            for link_elem in entry.findall("atom:link", namespaces):
                if link_elem.get("title") == "pdf":
                    link = link_elem.get("href", "")
                    break
            if not link:
                link_elem = entry.find("atom:id", namespaces)
                if link_elem is not None:
                    link = link_elem.text
            summary_elem = entry.find("atom:summary", namespaces)
            summary = summary_elem.text if summary_elem is not None else ""
            published_elem = entry.find("atom:published", namespaces)
            published = published_elem.text if published_elem is not None else ""
            authors = []
            for author_elem in entry.findall(".//atom:author/atom:name", namespaces):
                if author_elem.text:
                    authors.append(author_elem.text)
            result = {
                "title": title,
                "link": link,
                "snippet": summary,
                "abstract": summary,
                "published": published,
                "authors": authors,
                "source": "arXiv"
            }
            results.append(result)
        return {
            "query": query,
            "results": results,
            "result_count": len(results)
        }
    except Exception as e:
        logging.error(f"解析arXiv响应时出错: {e}")
        return {"error": str(e), "query": query, "results": []}

def google_arxiv_search(query: str, max_results: int = 10) -> Dict:
    logging.info(f"执行组合搜索 (Google + arXiv): {query}")
    google_results = google_search(query, max_results)
    arxiv_results = arxiv_search(query, max_results)
    return {
        "google": google_results,
        "arxiv": arxiv_results,
        "query": query
    }

class SearchManager:
    """搜索管理器，用于管理不同平台的搜索"""
    
    def __init__(self):
        self.config = Config()
        self.search_engines = {
            "arxiv": ArxivSearch(self.config),
            "google_scholar": GoogleScholarSearch(self.config),
            "patent": PatentSearch(self.config),
            "web": WebSearch(self.config),
            "news": NewsSearch(self.config)
        }
        logging.info("搜索管理器初始化完成")
    
    async def search(self, query: str, platform: str = "arxiv", 
                    max_results: int = 10, **kwargs) -> Dict:
        logging.info(f"执行搜索: '{query}', 平台: {platform}, 最大结果数: {max_results}")
        if isinstance(platform, str):
            platforms = [platform]
        else:
            platforms = platform
        for p in platforms:
            if p not in self.search_engines:
                logging.warning(f"不支持的搜索平台: {p}，将被跳过")
        supported_platforms = [p for p in platforms if p in self.search_engines]
        if not supported_platforms:
            logging.error(f"没有支持的搜索平台")
            return {"error": "没有支持的搜索平台", "query": query, "results": []}
        results = {}
        for p in supported_platforms:
            try:
                engine = self.search_engines[p]
                if p == "arxiv":
                    sort_by = kwargs.get("sort_by", "relevance")
                    time_range = kwargs.get("time_range", None)
                    categories = kwargs.get("categories", None)
                    results[p] = await engine.search(
                        query, 
                        max_results, 
                        sort_by=sort_by,
                        time_range=time_range,
                        categories=categories
                    )
                else:
                    results[p] = await engine.search(query, max_results)
            except Exception as e:
                logging.error(f"在平台 {p} 上搜索时出错: {e}")
                results[p] = {"error": str(e), "query": query, "results": []}
        return {
            "query": query,
            "platform_type": platform,
            "max_results": max_results,
            **results
        } 