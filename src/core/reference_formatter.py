"""
引用格式化模块
处理论文引用信息的标准化输出
"""
import logging
from typing import List, Dict, Any, Optional, Union
import re
from datetime import datetime

class ReferenceFormatter:
    """论文引用信息格式化类"""
    
    def __init__(self):
        """初始化引用格式化器"""
        logging.info("引用格式化器初始化")
        
    def format_references(self, search_results: Dict, style: str = "markdown") -> str:
        """
        从搜索结果中提取并格式化参考文献
        
        Args:
            search_results: 搜索结果字典
            style: 输出格式，支持 'markdown', 'html', 'text'
            
        Returns:
            格式化后的参考文献字符串
        """
        # 收集参考文献
        references_list = self.extract_references(search_results)
        
        # 根据指定格式输出
        if style == "html":
            return self.format_references_html(references_list)
        elif style == "text":
            return self.format_references_text(references_list)
        else:  # 默认使用markdown
            return self.format_references_markdown(references_list)
    
    def extract_references(self, search_results: Dict) -> List[Dict]:
        """
        从搜索结果中提取参考文献信息
        
        Args:
            search_results: 搜索结果字典
            
        Returns:
            参考文献列表，每项包含标题、链接、作者等信息
        """
        references_list = []
        
        # 遍历搜索结果的各个平台
        for source, data in search_results.items():
            if data and isinstance(data, dict) and 'results' in data and data['results']:
                for item in data['results']:
                    if isinstance(item, dict):
                        # 提取必要信息
                        reference = {
                            "title": item.get('title', '未知标题'),
                            "link": item.get('link', '#'),
                            "source": source,
                            "authors": item.get('authors', []),
                            "published": self._format_date(item.get('published')),
                            "updated": self._format_date(item.get('updated')),
                            "doi": item.get('doi', ''),
                            "journal": item.get('journal', ''),
                            "abstract": item.get('abstract', item.get('snippet', '')),
                            "citation_count": item.get('citation_count'),
                            "year": self._extract_year(item.get('published'), item.get('year'))
                        }
                        
                        # 只添加有链接的引用
                        if reference["link"] and reference["link"] != '#':
                            references_list.append(reference)
        
        # 去重
        unique_references = self._deduplicate_references(references_list)
        
        return unique_references
    
    def _deduplicate_references(self, references: List[Dict]) -> List[Dict]:
        """去除重复的引用"""
        unique_references = []
        seen_links = set()
        seen_titles = set()
        
        for ref in references:
            # 使用链接和标题作为去重依据
            link_key = ref["link"] if ref["link"] and ref["link"] != '#' else None
            title_key = ref["title"].lower() if ref["title"] else None
            
            # 如果链接或标题都没见过，就添加
            if (link_key and link_key not in seen_links) or \
               (not link_key and title_key and title_key not in seen_titles):
                unique_references.append(ref)
                if link_key:
                    seen_links.add(link_key)
                if title_key:
                    seen_titles.add(title_key)
        
        return unique_references
    
    def _format_date(self, date_str: Optional[str]) -> str:
        """格式化日期字符串"""
        if not date_str:
            return ""
            
        # 尝试几种常见的日期格式
        date_formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m"  # 年月格式
        ]
        
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
                
        # 如果所有格式都不匹配，尝试提取年份
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if year_match:
            return year_match.group(0)
            
        # 如果无法解析，原样返回
        return str(date_str)
    
    def _extract_year(self, date_str: Optional[str], year: Optional[Union[int, str]]) -> int:
        """从日期或年份字段提取年份"""
        # 如果有明确的年份字段，直接使用
        if year:
            try:
                return int(year)
            except (ValueError, TypeError):
                pass
        
        # 从日期字符串中提取年份
        if date_str:
            year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
            if year_match:
                return int(year_match.group(0))
        
        # 默认返回当前年份
        return datetime.now().year
    
    def format_references_markdown(self, references: List[Dict]) -> str:
        """使用Markdown格式输出参考文献（优化版，减少过度换行但保持必要结构）"""
        if not references:
            return "\n\n## 参考文献\n\n未找到可引用的文献。"
            
        references_section = "\n\n## 参考文献\n\n"
        
        # 按年份排序，最新的在前面
        sorted_references = sorted(references, key=lambda x: x.get('year', 0), reverse=True)
        
        for i, ref in enumerate(sorted_references):
            title = ref.get('title', '未知标题')
            source = ref.get('source', '未知来源')
            link = ref.get('link', '#')
            authors = ref.get('authors', [])
            published = ref.get('published', '')
            updated = ref.get('updated', '')
            journal = ref.get('journal', '')
            doi = ref.get('doi', '')
            citation_count = ref.get('citation_count')
            year = ref.get('year', '')
            
            # 格式化作者列表
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " 等"
                
            if not authors_str:
                authors_str = "未知作者"
            
            # 构建参考文献条目 - 使用紧凑的嵌套列表格式
            references_section += f"{i+1}. **{title}**\n"
            
            # 使用单个嵌套列表，避免多层嵌套
            info_items = []
            info_items.append(f"作者: {authors_str}")
            if journal:
                info_items.append(f"期刊/会议: {journal}")
            if published:
                info_items.append(f"发表时间: {published}")
            if updated and updated != published:
                info_items.append(f"更新时间: {updated}")
            info_items.append(f"来源: {source}")
            if doi:
                info_items.append(f"DOI: {doi}")
            if citation_count:
                info_items.append(f"引用次数: {citation_count}")
            info_items.append(f"链接: [{link}]({link})")
            
            # 将所有信息放在一个嵌套列表中，减少层级
            for item in info_items:
                references_section += f"   - {item}\n"
            
            # 每篇论文之间只留一个空行
            if i < len(sorted_references) - 1:
                references_section += "\n"
        
        return references_section
    
    def format_references_html(self, references: List[Dict]) -> str:
        """使用HTML格式输出参考文献"""
        if not references:
            return "<h2>参考文献</h2><p>未找到可引用的文献。</p>"
            
        references_section = "<h2>参考文献</h2><div class='references'>"
        
        # 按年份排序，最新的在前面
        sorted_references = sorted(references, key=lambda x: x.get('year', 0), reverse=True)
        
        for i, ref in enumerate(sorted_references):
            title = ref.get('title', '未知标题')
            source = ref.get('source', '未知来源')
            link = ref.get('link', '#')
            authors = ref.get('authors', [])
            published = ref.get('published', '')
            updated = ref.get('updated', '')
            journal = ref.get('journal', '')
            doi = ref.get('doi', '')
            citation_count = ref.get('citation_count')
            
            # 格式化作者列表
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " 等"
                
            if not authors_str:
                authors_str = "未知作者"
            
            # 构建参考文献条目
            references_section += f"<div class='reference-item'>"
            references_section += f"<p><strong>{i+1}. {title}</strong></p>"
            references_section += f"<p>作者: {authors_str}</p>"
            if journal:
                references_section += f"<p>期刊/会议: {journal}</p>"
            if published:
                references_section += f"<p>发表时间: {published}</p>"
            if updated and updated != published:
                references_section += f"<p>更新时间: {updated}</p>"
            references_section += f"<p>来源: {source}</p>"
            if doi:
                references_section += f"<p>DOI: {doi}</p>"
            if citation_count:
                references_section += f"<p>引用次数: {citation_count}</p>"
            references_section += f"<p>链接: <a href='{link}' target='_blank'>{link}</a></p>"
            references_section += "</div>"
            
        references_section += "</div>"
        return references_section
    
    def format_references_text(self, references: List[Dict]) -> str:
        """使用纯文本格式输出参考文献"""
        if not references:
            return "\n\n参考文献\n===========\n\n未找到可引用的文献。"
            
        references_section = "\n\n参考文献\n===========\n\n"
        
        # 按年份排序，最新的在前面
        sorted_references = sorted(references, key=lambda x: x.get('year', 0), reverse=True)
        
        for i, ref in enumerate(sorted_references):
            title = ref.get('title', '未知标题')
            source = ref.get('source', '未知来源')
            link = ref.get('link', '#')
            authors = ref.get('authors', [])
            published = ref.get('published', '')
            updated = ref.get('updated', '')
            journal = ref.get('journal', '')
            doi = ref.get('doi', '')
            citation_count = ref.get('citation_count')
            
            # 格式化作者列表
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " 等"
                
            if not authors_str:
                authors_str = "未知作者"
            
            # 构建参考文献条目
            references_section += f"{i+1}. {title}\n"
            references_section += f"   作者: {authors_str}\n"
            if journal:
                references_section += f"   期刊/会议: {journal}\n"
            if published:
                references_section += f"   发表时间: {published}\n"
            if updated and updated != published:
                references_section += f"   更新时间: {updated}\n"
            references_section += f"   来源: {source}\n"
            if doi:
                references_section += f"   DOI: {doi}\n"
            if citation_count:
                references_section += f"   引用次数: {citation_count}\n"
            references_section += f"   链接: {link}\n"
            
            # 每两篇论文之间空一行，最后一篇不加空行
            if i < len(sorted_references) - 1:
                references_section += "\n"
        
        return references_section
        
    def get_formatted_citation(self, reference: Dict, style: str = "apa") -> str:
        """
        生成指定格式的引用文本
        
        Args:
            reference: 引用信息字典
            style: 引用格式，支持 'apa', 'mla', 'chicago', 'harvard'
            
        Returns:
            格式化的引用文本
        """
        if style == "mla":
            return self._format_citation_mla(reference)
        elif style == "chicago":
            return self._format_citation_chicago(reference)
        elif style == "harvard":
            return self._format_citation_harvard(reference)
        else:  # 默认APA格式
            return self._format_citation_apa(reference)
    
    def _format_citation_apa(self, reference: Dict) -> str:
        """APA引用格式"""
        authors = reference.get('authors', [])
        title = reference.get('title', '未知标题')
        journal = reference.get('journal', '')
        year = reference.get('year', '')
        doi = reference.get('doi', '')
        
        # 作者格式化
        if not authors:
            author_text = "未知作者. "
        else:
            author_text = ""
            for i, author in enumerate(authors):
                if i == len(authors) - 1 and i > 0:
                    author_text += "& "
                last_name_parts = author.split()
                if last_name_parts:
                    # 尝试提取姓氏
                    last_name = last_name_parts[-1]
                    first_name = " ".join(last_name_parts[:-1])
                    if first_name:
                        initials = "".join([name[0] + "." for name in first_name.split() if name])
                        author_text += f"{last_name}, {initials}"
                    else:
                        author_text += last_name
                else:
                    author_text += author
                
                if i < len(authors) - 1 and len(authors) > 1:
                    author_text += ", "
            author_text += " "
        
        # 年份
        year_text = f"({year}). " if year else ""
        
        # 标题和期刊
        if journal:
            title_text = f"{title}. {journal}"
        else:
            title_text = f"{title}."
            
        # DOI
        doi_text = f" doi:{doi}" if doi else ""
        
        return f"{author_text}{year_text}{title_text}{doi_text}"
    
    def _format_citation_mla(self, reference: Dict) -> str:
        """MLA引用格式"""
        # MLA格式实现
        authors = reference.get('authors', [])
        title = reference.get('title', '未知标题')
        journal = reference.get('journal', '')
        year = reference.get('year', '')
        
        # 作者格式化
        if not authors:
            author_text = "未知作者. "
        else:
            author_text = ""
            for i, author in enumerate(authors):
                if i == 0:
                    # 第一个作者，姓在前，名在后
                    last_name_parts = author.split()
                    if len(last_name_parts) > 1:
                        last_name = last_name_parts[-1]
                        first_name = " ".join(last_name_parts[:-1])
                        author_text += f"{last_name}, {first_name}"
                    else:
                        author_text += author
                else:
                    # 其他作者，名在前，姓在后
                    author_text += f", {author}"
                
            author_text += ". "
        
        # 标题加引号
        title_text = f'"{title}." '
        
        # 期刊斜体
        journal_text = f"{journal}, " if journal else ""
        
        # 年份
        year_text = f"{year}." if year else ""
        
        return f"{author_text}{title_text}{journal_text}{year_text}"
    
    def _format_citation_chicago(self, reference: Dict) -> str:
        """Chicago引用格式"""
        # Chicago格式实现
        authors = reference.get('authors', [])
        title = reference.get('title', '未知标题')
        journal = reference.get('journal', '')
        year = reference.get('year', '')
        
        # 作者格式化
        if not authors:
            author_text = "未知作者. "
        else:
            author_text = ""
            for i, author in enumerate(authors):
                if i == 0:
                    # 第一个作者，姓在前，名在后
                    last_name_parts = author.split()
                    if len(last_name_parts) > 1:
                        last_name = last_name_parts[-1]
                        first_name = " ".join(last_name_parts[:-1])
                        author_text += f"{last_name}, {first_name}"
                    else:
                        author_text += author
                else:
                    # 从第二个作者开始，名在前，姓在后
                    if i == 1:
                        author_text += ", and "
                    else:
                        author_text += ", "
                    author_text += author
                
            author_text += ". "
        
        # 标题加引号
        title_text = f'"{title}." '
        
        # 期刊斜体
        journal_text = f"{journal} " if journal else ""
        
        # 年份
        year_text = f"({year})." if year else ""
        
        return f"{author_text}{title_text}{journal_text}{year_text}"
    
    def _format_citation_harvard(self, reference: Dict) -> str:
        """Harvard引用格式"""
        # Harvard格式实现
        authors = reference.get('authors', [])
        title = reference.get('title', '未知标题')
        journal = reference.get('journal', '')
        year = reference.get('year', '')
        
        # 作者格式化
        if not authors:
            author_text = "未知作者 "
        else:
            author_text = ""
            for i, author in enumerate(authors):
                last_name_parts = author.split()
                if last_name_parts:
                    # 尝试提取姓氏
                    last_name = last_name_parts[-1]
                    first_name = " ".join(last_name_parts[:-1])
                    if first_name:
                        initials = "".join([name[0] + "." for name in first_name.split() if name])
                        author_text += f"{last_name}, {initials}"
                    else:
                        author_text += last_name
                else:
                    author_text += author
                
                if i < len(authors) - 2:
                    author_text += ", "
                elif i == len(authors) - 2:
                    author_text += " and "
                
            author_text += " "
        
        # 年份
        year_text = f"({year}) " if year else ""
        
        # 标题
        title_text = f"{title}. "
        
        # 期刊斜体
        journal_text = f"{journal}." if journal else ""
        
        return f"{author_text}{year_text}{title_text}{journal_text}" 