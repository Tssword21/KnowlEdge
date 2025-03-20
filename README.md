# 学术搜索工具集

这个工具集提供了一系列用于学术论文搜索和比较的Python脚本，可以帮助研究人员更高效地查找和分析相关文献。

## 功能特点

- **arXiv搜索**：搜索arXiv上的论文，提取标题、作者、摘要、发表时间等信息
- **Google Scholar搜索**：搜索Google Scholar上的论文，获取引用次数和期刊信息
- **结果比较与合并**：比较两个来源的搜索结果，识别相似论文，合并信息
- **综合报告生成**：生成包含统计信息和详细结果的Markdown格式报告

## 安装依赖

在使用这些工具前，请先安装必要的依赖：

```bash
pip install arxiv scholarly difflib argparse
```

## 使用方法

### arXiv搜索

使用`improved_arxiv_search.py`脚本搜索arXiv上的论文：

```bash
python improved_arxiv_search.py "机器学习" --max 20 --sort-by relevance --categories cs.AI,cs.LG
```

参数说明：
- 第一个参数是搜索查询字符串
- `--max`：最大返回结果数（默认：20）
- `--sort-by`：排序方式，可选值：relevance（相关性）、lastUpdatedDate（最近更新）、submittedDate（提交日期）
- `--categories`：限制搜索的类别，多个类别用逗号分隔
- `--output`：输出文件名（不包含扩展名）
- `--format`：输出格式，可选值：json、markdown、both（默认：both）

### Google Scholar搜索

使用`google_scholar_search.py`脚本搜索Google Scholar上的论文：

```bash
python google_scholar_search.py "深度学习" --max 20 --year-start 2020 --year-end 2023
```

参数说明：
- 第一个参数是搜索查询字符串
- `--max`：最大返回结果数（默认：20）
- `--year-start`：开始年份
- `--year-end`：结束年份
- `--author`：按作者名搜索（添加此参数表示按作者搜索）
- `--output`：输出文件名（不包含扩展名）
- `--format`：输出格式，可选值：json、markdown、both（默认：both）
- `--topic`：Markdown输出的主题名称

### 结果比较与合并

使用`compare_search_results.py`脚本比较和合并arXiv和Google Scholar的搜索结果：

```bash
python compare_search_results.py "自然语言处理" --max 20 --similarity 0.8
```

参数说明：
- 第一个参数是搜索查询字符串
- `--arxiv-file`：arXiv搜索结果JSON文件（如果已有结果文件）
- `--scholar-file`：Google Scholar搜索结果JSON文件（如果已有结果文件）
- `--max`：每个来源的最大结果数（默认：20）
- `--similarity`：标题相似度阈值（默认：0.8）
- `--output`：输出文件名（不包含扩展名）
- `--year-start`：开始年份
- `--year-end`：结束年份

## 输出示例

### 综合报告

综合报告包含以下几个部分：

1. **报告摘要**：包含查询信息、结果数量等基本信息
2. **统计信息**：显示仅在arXiv中找到的论文数、仅在Google Scholar中找到的论文数、两个来源都有的论文数
3. **合并搜索结果**：按发表年份和引用次数排序的合并结果表格
4. **相似论文比较**：显示在两个来源中找到的相似论文及其相似度
5. **原始结果**：分别显示arXiv和Google Scholar的原始搜索结果

## 注意事项

1. Google Scholar搜索可能会受到Google的访问限制，脚本中已添加延迟以减少被封禁的风险，但仍建议不要短时间内进行大量查询
2. 对于大量查询，建议先保存结果文件，然后使用`--arxiv-file`和`--scholar-file`参数进行比较，以避免重复查询
3. 相似度阈值可以根据需要调整，值越高表示要求标题越相似才会被认为是同一篇论文

## 示例工作流程

1. 首先搜索arXiv获取最新论文：
   ```bash
   python improved_arxiv_search.py "transformer models" --max 30 --categories cs.CL,cs.AI
   ```

2. 然后搜索Google Scholar获取引用信息：
   ```bash
   python google_scholar_search.py "transformer models" --max 30
   ```

3. 最后比较和合并结果：
   ```bash
   python compare_search_results.py "transformer models" --arxiv-file arxiv_20230501_transformer_models.json --scholar-file scholar_20230501_transformer_models.json
   ```

4. 查看生成的综合报告（Markdown格式）：
   ```bash
   compare_20230501_transformer_models.md
   ```

## 贡献与改进

欢迎提出建议和改进意见，或者提交代码贡献。可能的改进方向包括：

- 添加更多学术搜索源（如IEEE Xplore、ACM Digital Library等）
- 改进相似度匹配算法
- 添加更多过滤和排序选项
- 实现Web界面 