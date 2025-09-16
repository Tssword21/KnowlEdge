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
        # 这里需要使用异步操作，但目前google_search是同步的
        # 在实际实现中应该改为异步方法
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
        # 使用improved_arxiv_search.py中的ArxivSearcher进行真实搜索
        try:
            logging.info(f"使用ArxivSearcher搜索: {query}, 排序方式: {sort_by}, 时间范围: {time_range}, 类别: {categories}")
            arxiv_searcher = ArxivSearcher(max_results=max_results)
            
            # 使用异步线程运行同步方法
            results = await asyncio.to_thread(
                arxiv_searcher.search, 
                query, 
                sort_by=sort_by,
                time_range=time_range,
                categories=categories
            )
            
            # 格式化结果以匹配期望的格式
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
                # 如果有代码链接，也添加进来
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
            # 如果真实搜索失败，回退到原有的搜索方法
            return await asyncio.to_thread(arxiv_search, query, max_results)

class GoogleScholarSearch(SearchEngine):
    """Google学术搜索引擎"""
    
    async def search(self, query: str, max_results: int = 10) -> Dict:
        """执行Google Scholar搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            搜索结果字典
        """
        # 使用现有的google_search函数，但可以在后续实现专门的Google Scholar搜索
        logging.info(f"执行Google Scholar搜索: {query}")
        # 添加学术关键词以提高相关性
        scholarly_query = f"{query} academic research paper"
        return await asyncio.to_thread(google_search, scholarly_query, max_results)

class PatentSearch(SearchEngine):
    """专利搜索引擎"""
    
    async def search(self, query: str, max_results: int = 10) -> Dict:
        """执行专利搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            搜索结果字典
        """
        # 暂时使用Google搜索代替专利搜索
        logging.info(f"执行专利搜索: {query}")
        patent_query = f"{query} patent"
        return await asyncio.to_thread(google_search, patent_query, max_results)

class NewsSearch(SearchEngine):
    """新闻搜索引擎"""
    
    async def search(self, query: str, max_results: int = 10) -> Dict:
        """执行新闻搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            搜索结果字典
        """
        # 暂时使用Google搜索代替新闻搜索
        logging.info(f"执行新闻搜索: {query}")
        news_query = f"{query} news recent"
        return await asyncio.to_thread(google_search, news_query, max_results)

def google_search(query: str, max_results: int = 10) -> Dict:
    """
    使用Google搜索API进行搜索
    
    Args:
        query: 搜索查询
        max_results: 最大结果数量
        
    Returns:
        搜索结果字典
    """
    logging.info(f"执行Google搜索: {query}")
    
    try:
        if not config.serper_api_key:
            logging.warning("SERPER_API_KEY 未配置，跳过Google搜索")
            return {"query": query, "results": [], "result_count": 0, "warning": "SERPER_API_KEY 未配置"}
        # 构建API请求
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": max_results
        })
        
        headers = {
            'X-API-KEY': config.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        # 发送请求
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()  # 如果请求失败，会抛出异常
        
        # 解析返回结果
        data = response.json()
        return parse_google_results(data, query)
        
    except Exception as e:
        logging.error(f"Google搜索出错: {e}")
        return {"error": str(e), "query": query, "results": []}

def parse_google_results(data: Dict, query: str) -> Dict:
    """
    解析Google搜索结果
    
    Args:
        data: Google API返回的JSON数据
        query: 原始查询
        
    Returns:
        解析后的结果字典
    """
    try:
        results = []
        
        # 处理有机搜索结果
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
        
        # 处理知识面板结果
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
    """
    搜索arXiv文章
    
    Args:
        query: 搜索查询
        max_results: 最大结果数量
        
    Returns:
        搜索结果字典
    """
    logging.info(f"执行arXiv搜索: {query}")
    
    try:
        # 构建API请求URL
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        # 发送请求
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # 解析XML响应
        return parse_arxiv_response(response.text, query)
        
    except Exception as e:
        logging.error(f"arXiv搜索出错: {e}")
        return {"error": str(e), "query": query, "results": []}

def parse_arxiv_response(xml_data: str, query: str) -> Dict:
    """
    解析arXiv API的XML响应
    
    Args:
        xml_data: arXiv API返回的XML数据
        query: 原始查询
        
    Returns:
        解析后的结果字典
    """
    try:
        root = ElementTree.fromstring(xml_data)
        
        # 定义命名空间
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom"
        }
        
        results = []
        
        # 遍历所有条目
        for entry in root.findall(".//atom:entry", namespaces):
            # 提取标题
            title_elem = entry.find("atom:title", namespaces)
            title = title_elem.text if title_elem is not None else ""
            
            # 提取链接
            link = ""
            for link_elem in entry.findall("atom:link", namespaces):
                if link_elem.get("title") == "pdf":
                    link = link_elem.get("href", "")
                    break
            
            # 如果没有找到PDF链接，使用摘要页面链接
            if not link:
                link_elem = entry.find("atom:id", namespaces)
                if link_elem is not None:
                    link = link_elem.text
            
            # 提取摘要
            summary_elem = entry.find("atom:summary", namespaces)
            summary = summary_elem.text if summary_elem is not None else ""
            
            # 提取发布日期
            published_elem = entry.find("atom:published", namespaces)
            published = published_elem.text if published_elem is not None else ""
            
            # 提取作者
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
    """
    组合Google和arXiv搜索
    
    Args:
        query: 搜索查询
        max_results: 每个搜索的最大结果数
        
    Returns:
        组合的搜索结果
    """
    logging.info(f"执行组合搜索 (Google + arXiv): {query}")
    
    # 并行执行两个搜索
    google_results = google_search(query, max_results)
    arxiv_results = arxiv_search(query, max_results)
    
    # 合并结果
    return {
        "google": google_results,
        "arxiv": arxiv_results,
        "query": query
    }

class SearchManager:
    """搜索管理器，用于管理不同平台的搜索"""
    
    def __init__(self):
        """初始化搜索管理器"""
        self.config = Config()
        
        # 初始化各搜索引擎
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
        """
        执行搜索
        
        Args:
            query: 搜索查询
            platform: 搜索平台，可以是单个平台名称或平台列表
            max_results: 最大结果数量
            **kwargs: 额外的搜索参数
            
        Returns:
            搜索结果字典
        """
        logging.info(f"执行搜索: '{query}', 平台: {platform}, 最大结果数: {max_results}")
        
        # 转换单个平台为列表
        if isinstance(platform, str):
            platforms = [platform]
        else:
            platforms = platform
            
        # 检查平台是否支持
        for p in platforms:
            if p not in self.search_engines:
                logging.warning(f"不支持的搜索平台: {p}，将被跳过")
                
        # 过滤掉不支持的平台
        supported_platforms = [p for p in platforms if p in self.search_engines]
        
        # 如果没有支持的平台，返回空结果
        if not supported_platforms:
            logging.error(f"没有支持的搜索平台")
            return {"error": "没有支持的搜索平台", "query": query, "results": []}
            
        # 执行搜索
        results = {}
        for p in supported_platforms:
            try:
                # 获取对应的搜索引擎
                engine = self.search_engines[p]
                
                # 根据平台类型传递不同的参数
                if p == "arxiv":
                    # 对于arxiv搜索，支持更多参数
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
                    # 其他平台使用标准参数
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