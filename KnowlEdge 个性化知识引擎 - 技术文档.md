# **KnowlEdge 系统技术报告**

总结：

开发并优化基于FastAPI的KnowlEdge个性化知识引擎，通过整合大型语言模型（LLM）与多源API（Google Search、ArXiv），实现用户画像驱动的智能报告生成。主导后端架构设计，构建用户画像模块（技能/兴趣提取、动态权重更新）及多平台搜索策略，支持文献综述、行业研究、科普文章等结构化报告生成；采用SSE技术实现前端实时进度反馈，优化交互体验。技术栈涵盖Python/FastAPI、SQLite、BERT文本相似度计算及LLM（DeepSeek）内容生成，完成从数据检索、分析到多格式报告输出的全流程自动化。

开发并优化基于FastAPI的KnowlEdge个性化知识引擎，通过整合LLM与多源API（Google Search、ArXiv），实现用户画像驱动的智能报告生成。主导后端架构设计，构建用户画像模块（技能/兴趣提取、动态权重更新）及多平台搜索策略，支持文献综述、行业研究、科普文章等结构化报告生成；技术栈涵盖Python/FastAPI、SQLite、BERT文本相似度计算及LLM（DeepSeek）内容生成，完成从数据检索、分析到多格式报告输出的全流程自动化。

## **1. 系统概述**

KnowlEdge 是一个旨在为用户提供个性化知识更新的智能引擎。用户可以通过 Web 界面输入其职业、关注领域、期望的知识更新周期、报告类型、所需文献数量以及可选的简历文本。系统会基于这些输入，结合用户画像分析（如果提供了简历或存在历史数据），在用户选择的平台（如学术期刊、新闻）上搜索相关信息，并最终生成一份结构化的、针对特定需求的报告（如标准行业报告、文献综述、行业调研报告或知识科普文章）。该系统利用大型语言模型 (LLM) 进行内容分析、翻译、摘要、推荐和报告撰写，并通过服务器发送事件 (SSE) 向前端实时反馈处理进度。

## **2. 系统架构**

系统主要由以下几个核心组件构成：

* **Web 应用层 (FastAPI - `app.py`)**: 提供前端用户界面和后端 API 接口。
* **核心逻辑层 (KnowledgeFlow - `KnowlEdge.py`)**: 编排整个知识获取和报告生成的工作流。
* **用户画像管理模块 (UserProfileManager - `KnowlEdge.py`)**: 负责创建、存储和更新用户画像信息。
* **数据库模块 (SQLite - `db_utils.py`)**: 持久化存储用户信息、兴趣、技能、搜索记录等。
* **前端界面 (HTML/CSS/JavaScript - `templates/index.html`)**: 用户与系统交互的界面。
* **外部服务集成**:
  * 大型语言模型 (如 DeepSeek API): 用于文本分析、翻译、摘要、推荐、报告生成。
  * 搜索引擎 (Google via Serper API, ArXiv API): 用于信息检索。
  * 翻译服务 (GoogleTranslator, 计划中的 BaiduTranslator, DeepLTranslator)。

除了先前列出的组件，系统还包含以下辅助脚本，用于系统的初始化、维护和数据查看：

- **系统初始化脚本 (init_system.py)**: 用于创建必要的目录结构和默认配置文件。
- **数据库查看工具 (view_database.py)**: 提供命令行界面来查看和导出数据库中的用户数据和统计信息。
- **数据库清理脚本 (clean_database.py)**: 用于定期清理旧的（如超过90天）搜索和交互记录。
- **用户画像验证脚本 (verify_profile.py)**: 用于检查数据库表结构、用户数据完整性和兴趣分类文件的存在性。

## **3. 核心组件详解**

### **3.1. Web 应用层 (`app.py`)**

* **技术栈**: Python, FastAPI, Jinja2。
* **主要功能**:

  * **`GET /`**: 使用 Jinja2 模板引擎渲染并提供 `index.html` 页面。
  * **`POST /process`**:
    * 接收来自前端表单的用户输入数据（姓名、职业、周期、平台、领域、邮箱、简历文本、报告类型、文献数量）。
    * 初始化 KnowledgeFlow 实例。
    * 通过服务器发送事件 (SSE) 实现流式响应，knowledge_flow_sse_generator 负责将处理进度和最终报告（根据请求的 report_type 生成特定类型的报告）逐步发送到前端。
    * 包含详细的日志记录和错误处理机制。
* **初始化**: 配置日志、模板目录、确保数据目录存在，并调用 `db_utils.verify_database()` 验证数据库。
* SSE 进度反馈: 定义了与前端一致的 TOTAL_STEPS 和 STEP_DEFINITIONS。get_step_name 函数会根据报告类型动态调整第五步的描述，用于在处理过程中向用户显示清晰的进度信息。

### **3.2. 核心逻辑层 (`KnowlEdge.py`)**

此文件包含三个主要的类：`KnowledgeFlow`、`UserProfileManager` 和 `ResumeReader`。

**3.2.1. `KnowledgeFlow` 类**

*   **职责**: 编排整个业务流程，从接收用户输入到生成最终报告。
*   **关键方法与流程**:
    1.  **`start_node(user_input)`**: 初始化上下文，收集用户输入（包括 `num_papers`），创建用户 ID (如果需要)，计算信息更新的时间范围。
    2.  **`build_user_profile(user_input, cv_text)`**:
        *   如果提供了简历文本 (`cv_text`)，则调用 `UserProfileManager` 的方法从简历中提取技能和兴趣，并存入数据库。
        *   如果未提供简历，则根据用户输入创建基本用户档案，并可能从用户提供的 `content_type` 初始化基本兴趣。
        *   确保每个用户都有一个唯一的 ID（即使是临时用户）。
        *   完成后调用 `display_profile_summary()` 在控制台输出画像摘要。
    3.  **`_call_llm(prompt, system_message)` (async)**:
        *   通用的LLM调用方法，接收用户Prompt和系统角色提示 (System Message)。
        *   执行LLM调用，并对返回内容进行后处理，包括去除首尾空白、规范化换行符（将多个换行符合并为最多两个，确保段落间距）。
    4.  **`_prepare_llm_context_from_search_results(search_results, max_items_per_source, max_chars)`**:
        *   从多源搜索结果中提取关键信息（来源、标题、摘要），格式化为文本字符串，作为LLM生成报告时的上下文参考。
        *   限制每个来源提取的条目数和总字符数，以适应LLM的输入限制。
    5.  **`_update_interests_from_query(query, weight_adjustment)`**:
        *   使用 LLM 从用户当前的搜索查询中提取最多3个主要兴趣领域或主题。
        *   调用 `UserProfileManager.update_interest_weights()` 更新用户画像中对应主题的权重。
    6.  **`translate_query(query)` (async)**:
        *   使用 `GoogleTranslator` (以及计划中的其他翻译API) 将用户查询翻译成英文。
        *   使用 LLM (DeepSeek) 对多个翻译结果进行评估和选择（或直接由LLM给出最佳翻译），以提高翻译质量。
        *   在翻译查询词的同时，会调用 `_update_interests_from_query` 更新用户兴趣。
    7.  **`build_search_query()` (async)**:
        *   获取用户关注的内容类型 (`content_type`) 作为查询基础。
        *   调用 `translate_query()` 将查询词翻译成英文。
        *   根据计算出的时间范围和翻译后的查询词，构建针对 Google、ArXiv 的搜索查询字符串。
        *   记录用户搜索行为到数据库 (通过 `UserProfileManager`)。
    8.  **`execute_search(queries)`**:
        *   根据用户选择的平台类型（学术期刊、新闻类、综合类）和指定的文献数量 (`num_papers`) 执行相应的搜索策略。
        *   文献数量会根据平台类型进行调整（例如，“综合类”可能会为每个子源分配 `num_papers // 2` 的数量）。
        *   调用 `google_search()` (使用 Serper API)、`arxiv_search()` (使用 ArXiv API) 或 `google_arxiv_search()`。
        *   对搜索结果进行解析 (`parse_google_results`, `parse_arxiv_response`)，并使用 BERT (`compute_similarity`) 计算结果与查询的余弦相似度。
    9.  **报告生成 (Report Generation - 多个专项方法)**:
        *   **`generate_literature_review(search_results, original_query)` (async)**:
            *   准备LLM上下文 (`_prepare_llm_context_from_search_results`)。
            *   使用特定的系统角色提示 (`system_message_lit_review`) 指导LLM扮演学术研究员和文献综述专家。
            *   根据预设的Markdown结构（引言、主要研究方向、关键文献回顾、研究方法、局限性与空白、结论）调用 `_call_llm` 生成文献综述。
            *   自动从搜索结果中提取并格式化参考文献列表追加到报告末尾。
        *   **`generate_industry_research_report(search_results, user_input, original_query)` (async)**:
            *   准备LLM上下文。
            *   使用特定的系统角色提示 (`system_message_industry_analyst`) 指导LLM扮演行业分析师。
            *   根据预设的Markdown结构（执行摘要、技术概览、技术对比、成熟度评估、应用前景、近期趋势与风险、初步战略建议）调用 `_call_llm` 生成行业调研报告。
        *   **`generate_popular_science_report(search_results, user_input, original_query)` (async)**:
            *   准备LLM上下文。
            *   使用特定的系统角色提示 (`system_msg_science_writer`) 指导LLM扮演科普作家。
            *   根据预设的Markdown结构（生活化类比解释、关心理由、核心概念三连击、学习指南、快问快答、结语，可能包含Mermaid图表提示）调用 `_call_llm` 生成知识科普文章。
        *   **`generate_report(search_results)` (旧方法，主要被专项报告取代)**:
            *   主要调用 `integrate_with_large_model()` 使用 LLM (DeepSeek) 对各来源的搜索结果进行整合、摘要和翻译（确保输出为中文）。LLM 的提示词指导其以特定格式（来源、标题、摘要、原文网址、相似度）呈现信息。此方法生成的报告更偏向于原始搜索结果的罗列和初步整合。
    10.  **`send_email(report)`**: 预留的发送邮件功能（当前为打印到控制台）。
*   **其他辅助方法**: `calculate_update_cycle`, `display_profile_summary`, `_extract_interest_from_content` (从文献内容提取兴趣并更新画像), `process_user_feedback`。

**3.2.2. `UserProfileManager` 类**

*   **职责**: 管理用户的画像数据，包括创建、存储、更新和分析。
*   **核心功能**:
    *   **用户创建与识别**: `_generate_user_id` (基于姓名、邮箱、职业的 MD5 哈希), `create_user` (存入 `users` 表)。
    *   **画像信息提取 (使用 LLM)**:
        *   `extract_skills_from_resume()`: 使用LLM从简历中提取技能、熟练程度和类别，进行JSON解析和清理，然后存入 `user_skills` 表。包含详细的进度打印和错误处理。
        *   `extract_interests_from_resume()`: 使用LLM从简历中提取兴趣主题，并根据预定义的 `interest_categories.json` (若不存在则创建默认分类，包括技术、科学、商业等大类及其子主题) 进行分类，计算初始权重，进行JSON解析和清理，然后存入 `user_interests` 表。包含详细的进度打印和错误处理，以及在解析失败时尝试创建基本兴趣的逻辑。
    *   **行为记录**:
        *   `record_search()`: 记录用户的搜索查询和平台到 `user_searches` 表。
        *   `record_interaction()`: 记录用户与特定内容的交互行为（如点击、收藏）到 `user_interactions` 表。
    *   **兴趣模型更新**:
        *   `update_interest_weights()`: 根据用户行为（如搜索、内容交互、反馈）调整特定兴趣主题的权重。如果兴趣不存在，则创建新的兴趣项，并尝试使用 `compute_similarity` 将其与预定义类别关联。
        *   `apply_time_decay()`: 定期降低旧兴趣的权重（基于`decay_factor`和`days_threshold`），以反映用户兴趣的动态变化。
    *   **画像分析与应用**:
        *   `get_top_interests()`: 获取用户当前权重最高的兴趣（每个主题只取最新记录）。
        *   `analyze_search_patterns()`: 使用 LLM 分析用户近期的搜索记录，识别主导主题和搜索模式，返回JSON格式的分析结果。
        *   `generate_recommendations()`: 基于用户顶级兴趣，使用 LLM 生成具体的、前沿的研究或学习主题推荐，返回包含`topic`和`reason`的JSON数组。
        *   `get_user_profile_summary()`: 提供用户画像的全面摘要，包括基本信息、顶级兴趣、技能、活动统计（搜索和交互次数）、最近搜索记录等。
*   **依赖**: OpenAI API (DeepSeek), `db_utils`, BERT (`compute_similarity`)。

**3.2.3. `ResumeReader` 类**

*   **职责**: 读取多种格式的简历文件内容。
*   **支持格式**: .txt, .pdf (PyPDF2), .docx (python-docx), .doc (提示转换), .xlsx/.xls (pandas), .jpg/.jpeg/.png (Pillow, Pytesseract OCR - 支持中英文 `chi_sim+eng`)。
*   **功能**: 自动检测文件类型并调用相应的解析库。如果未提供文件路径，会提示用户选择输入方式（文本或文件），并处理路径不存在的情况。

### **3.3. 数据库模块 (`db_utils.py`)**

*   **技术栈**: Python, SQLite。
*   **主要功能**:
    *   **`get_db_connection()`**:
        *   建立到 SQLite 数据库 (`./user_data/user_profiles.db`) 的连接。
        *   如果数据库文件或表不存在，则自动创建。
        *   使用 `conn.row_factory = sqlite3.Row` 以便通过列名访问数据。
    *   **表结构**:
        *   `users`: 用户基本信息 (ID 主键, name, occupation, email, created_at)。
        *   `user_interests`: 用户兴趣 (ID 自增主键, user_id 外键, topic, category, weight, timestamp)。
        *   `user_searches`: 用户搜索历史 (ID 自增主键, user_id 外键, query, platform, timestamp)。
        *   `user_interactions`: 用户内容交互记录 (ID 自增主键, user_id 外键, content_id, action_type, timestamp)。
        *   `user_skills`: 用户技能 (ID 自增主键, user_id 外键, skill, level, category, timestamp)。
    *   **`verify_database()`**: 检查数据库连接和读写权限，并打印表信息。（注意：文件中存在两个同名函数定义，后一个会覆盖前一个，文档中已指出此问题）。
    *   **`initialize_database()`**: 确保数据库和表已创建。

### **3.4. 前端界面 (`templates/index.html`)**

*   **技术栈**: HTML, CSS, JavaScript, Marked.js, Mermaid.js。
*   **用户界面**:
    *   提供一个表单供用户输入个人信息、偏好（包括报告类型 `report_type` 和文献数量 `num_papers`）和可选的简历文本。
    *   默认填充了一些示例值，方便测试。
*   **交互逻辑 (JavaScript)**:
    *   **表单提交**: 异步提交表单数据到后端的 `/process` 接口。
    *   **SSE 处理**: 监听并处理从服务器发送的 SSE 事件。
        *   `progress` 事件: 更新加载状态文本（根据 `app.py` 中 `get_step_name` 动态生成，区分不同报告类型的步骤描述）、进度条和分步处理状态列表。
        *   `complete` 事件: 使用 `marked.parse()` 将返回的 Markdown 报告渲染为 HTML 并显示。
        *   `error` 事件: 显示错误信息。
    *   **UI 更新**: 动态显示加载覆盖层、进度信息、处理结果或错误。
    *   `linkifyReport()`: 将报告文本中的 URL (以 "原文网址：" 或 "链接：" 开头) 转换为可点击的超链接。
    *   引入 `Mermaid.js`，理论上如果报告中包含 Mermaid 格式的图表代码，会被渲染。
*   **样式**: 使用内联 CSS 提供了一个简洁、现代的外观。

### **3.5. 辅助脚本**

**3.5.1. 系统初始化脚本 (`init_system.py`)**

*   **职责**: 确保系统运行所需的基础设施已准备就绪。
*   **主要功能**:
    *   **`create_directory_structure()`**: 创建 `./user_data` (主数据目录), `./user_data/logs` (日志目录), `./user_data/cache` (缓存目录)。
    *   **`initialize_database()`**: 调用 `db_utils.initialize_database()` 来创建或验证数据库表结构。
    *   **`create_interest_categories()`**: 创建 `./user_data/interest_categories.json` 文件，包含预定义的兴趣分类体系（技术、科学、商业、艺术、教育、健康等及其子主题）。这个文件会被 `UserProfileManager` 加载。
    *   **`verify_system()`**: 检查所有必要的目录和文件是否已成功创建。
*   **执行流程**: 在系统首次部署或需要重置时运行，为系统提供一个干净的初始状态。

**3.5.2. 数据库查看工具 (`view_database.py`)**

*   **职责**: 提供一个命令行界面，方便开发者或管理员查看数据库内容。
*   **主要功能**:
    *   **`check_database()`**: 检查数据库文件是否存在。
    *   **`view_all_users()`**: 显示所有用户的基本信息、技能和兴趣。
    *   **`view_user_by_id(user_id)`**: 显示特定用户的详细信息，包括基本信息、技能、兴趣和最近的搜索历史。
    *   **`view_database_stats()`**: 显示各表的记录数以及数据库文件的大小。
    *   **`export_user_data(user_id, output_file)`**: 将特定用户的数据（基本信息、技能、兴趣、搜索历史）导出到文本文件。
*   **交互**: 通过命令行菜单提供操作选项。

**3.5.3. 数据库清理脚本 (`clean_database.py`)**

*   **职责**: 定期清理数据库中的过期数据，以控制数据库大小和维护性能。
*   **主要功能**:
    *   **`clean_old_data(days=90)`**: 删除 `user_searches` 和 `user_interactions` 表中时间戳早于指定天数（默认为90天）的记录。
*   **执行**: 可通过计划任务（如 cron job）定期运行。

**3.5.4. 用户画像验证脚本 (`verify_profile.py`)**

*   **职责**: 验证用户画像系统的核心组件是否配置正确并包含预期数据。
*   **主要功能**:
    *   **`check_database_tables()`**:
        *   检查所有核心表 (`users`, `user_interests`, `user_searches`, `user_interactions`, `user_skills`) 是否存在。
        *   如果存在用户数据，则打印用户总数，并逐个显示用户的基本信息、前5项兴趣、前5项技能和最近3条搜索历史。
    *   **`check_interest_categories()`**: 检查 `interest_categories.json` 文件是否存在，并打印各主分类下的主题数量。
*   **用途**: 用于诊断系统配置问题或数据完整性问题。

## **4. 数据流与核心流程**

1.  **用户访问**: 用户打开 `/` 路径，`app.py` 返回 `index.html`。
2.  **信息输入**: 用户在 `index.html` 表单中填写信息（职业、关注领域、周期、报告类型、文献数量、邮箱等，可选简历）并提交。
3.  **请求处理 (`/process`)**:
    *   JavaScript 将表单数据 POST 到 `app.py` 的 `/process` 端点。
    *   `app.py` 调用 `knowledge_flow_sse_generator`，传入用户输入和选择的 `report_type`。
4.  **SSE 流开始**: `knowledge_flow_sse_generator` 开始执行，并通过 SSE 向前端发送进度。
5.  **`KnowledgeFlow` 执行**:
    *   **初始化**: `workflow.start_node()` 处理输入，创建用户 ID（如果不存在，则调用 `UserProfileManager.create_user()`）。
    *   **用户画像构建**: `workflow.build_user_profile()`。如果提供了简历，`UserProfileManager` 会调用 LLM 提取技能和兴趣，并存入数据库。
    *   **搜索参数构建**: `workflow.build_search_query()` 将用户关注领域翻译成英文，并结合时间范围构建搜索查询。
    *   **执行搜索**: `workflow.execute_search()` 根据用户选择的平台和文献数量调用 Google Search API (Serper) 和/或 ArXiv API。结果会进行相似度计算。
    *   **专项报告生成**:
        *   根据前端传入的 `report_type`，`knowledge_flow_sse_generator` 会调用 `KnowledgeFlow` 类中对应的专项报告生成方法 (如 `generate_literature_review`, `generate_industry_research_report`, `generate_popular_science_report`)。
        *   这些方法会使用 `_prepare_llm_context_from_search_results` 准备上下文，并使用 `_call_llm` 结合特定的系统角色提示和用户Prompt模板来调用LLM，生成结构化的Markdown报告。
        *   如果 `report_type` 是 "standard_report" 或未匹配到其他类型，则会调用旧的 `generate_report` 方法（主要进行搜索结果整合）。
    *   （可选）用户兴趣更新：在查询、内容提取等环节，会调用 `UserProfileManager` 的方法更新用户兴趣权重。
6.  **SSE 流结束**: `knowledge_flow_sse_generator` 发送包含最终报告（Markdown格式）的 `complete` 事件，或在出错时发送 `error` 事件。
7.  **前端展示**: JavaScript 接收到 `complete` 事件后，使用 `Marked.js` 将 Markdown 报告渲染为 HTML 并在页面上展示；接收到 `error` 事件则显示错误信息。

## **5. 技术栈总结**

*   **后端**: Python, FastAPI, Uvicorn
*   **前端**: HTML, CSS, JavaScript, Marked.js, Mermaid.js
*   **数据库**: SQLite
*   **AI/ML**:
    *   OpenAI API (DeepSeek): 用于文本分析、摘要、翻译、推荐、技能/兴趣提取、结构化报告生成。
    *   Transformers (BERT `bert-base-multilingual-cased`): 用于文本嵌入和余弦相似度计算。
*   **外部 API**:
    *   Serper API (Google Search)
    *   ArXiv API
    *   GoogleTranslate (deep-translator)
*   **数据处理**: PyPDF2, python-docx, pandas, Pillow, Pytesseract (OCR for `chi_sim+eng`)
*   **版本控制/依赖**: (未明确，但通常为 Git, pip/requirements.txt)

## **6. 潜在的改进与未来方向**

*   **错误处理与健壮性**: 进一步增强对外部 API 调用失败、LLM响应格式意外、数据解析错误等的处理。
*   **安全性**: 对用户输入进行更严格的校验和清理，防止潜在的安全风险（如注入）。API 密钥管理应遵循最佳实践（如使用环境变量，当前已部分实现）。
*   **用户反馈闭环**: `process_user_feedback` 方法已存在，可以进一步在前端集成反馈按钮（喜欢/不喜欢/收藏），让用户可以直接对报告中的条目进行反馈，从而更精确地调整用户画像。
*   **LLM Prompt管理与优化**: 将 `KnowledgeFlow` 中用于不同报告生成的System Prompt和User Prompt模板外部化到配置文件或专门的模块中，便于管理、迭代和A/B测试。
*   **异步任务队列**: 对于耗时较长的 LLM 调用或外部 API 请求，可以考虑使用如 Celery 之类的异步任务队列，以提高 `/process` 端点的响应速度和并发处理能力，尽管当前 SSE 已经缓解了部分阻塞问题。
*   **测试**: 增加单元测试和集成测试，确保代码质量和系统稳定性，特别是针对LLM输出的校验。
*   **前端优化**:
    *   可以将 CSS 和 JavaScript 分离到单独的文件。
    *   考虑使用现代前端框架 (如 Vue, React, Svelte) 以提升开发效率和可维护性。
*   **多语言支持**: 目前报告强制输出中文（通过LLM Prompt），未来可以根据用户偏好或浏览器设置支持更多语言的输入和输出。
*   **数据库迁移与扩展**: 随着用户量和数据量的增长，未来可能需要考虑更强大的数据库系统。
*   **`db_utils.py` 中的 `verify_database` 函数冗余**: 文件中存在两个同名函数，后者会覆盖前者。应予清理。
*   **日志管理 (`init_system.py` 创建了 `logs` 目录，但实际日志配置在各模块中)**:
    *   可以考虑将所有模块的日志统一配置输出到 `./user_data/logs` 目录下的不同文件或按日期轮换的日志文件中，便于集中管理和分析。
*   **缓存机制 (`init_system.py` 创建了 `cache` 目录)**:
    *   当前代码中未明确看到缓存目录的使用。未来可以考虑对耗时操作的结果（如 LLM 调用、频繁的数据库查询结果、已翻译的查询词）进行缓存，以提高性能和减少 API 调用成本。
*   **辅助脚本的健壮性**: 增加更详细的错误处理和用户提示，例如在 `view_database.py` 中处理无效的用户 ID 输入。
*   **配置统一**: `init_system.py` 和 `verify_profile.py` 中有部分重复的 `CONFIG` 定义，可以考虑将通用配置抽取到单独的配置文件或模块中，供所有脚本和主应用共享。
*   **`KnowlEdge.py` 中 LLM 调用后处理**: `_call_llm` 方法中的后处理逻辑（去除空白、规范换行）是通用性的，但如果不同类型的报告需要更精细的后处理，可能需要进一步细化。

## **7. 总结**

KnowlEdge 系统是一个功能丰富且设计精巧的个性化知识更新引擎。它有效地整合了用户画像构建、多源信息检索、LLM驱动的内容生成和实时反馈机制。特别是其针对不同报告类型的专项生成流程（文献综述、行业报告、科普文章），通过精细的Prompt工程，展现了LLM在结构化内容创作方面的强大潜力。系统架构清晰，模块化程度较高，为未来的功能扩展和优化打下了良好基础。通过持续迭代和完善，KnowlEdge有望成为用户获取专业领域知识的得力助手。

## 8. 附录

### 8.1 函数详细讲解

#### 8.1.1 app.py

`app.py` 文件定义了一个 FastAPI 应用，它提供了两个主要端点：

1.  `GET /`: 用于提供 `index.html` 页面。
2.  `POST /process`: 用于处理用户通过表单提交的数据。这个端点接收用户姓名、职业、知识更新周期、关注平台、关注领域、邮箱、**报告类型**、**所需文献数量**以及可选的简历文本。它使用这些数据初始化一个 `KnowledgeFlow` 实例。核心处理逻辑封装在异步生成器 `knowledge_flow_sse_generator` 中，该生成器会：
    *   调用 `KnowledgeFlow` 实例的各个方法执行处理步骤（初始化、用户画像分析、构建搜索参数、执行搜索）。
    *   根据用户请求的 `report_type`，调用 `KnowledgeFlow` 中对应的专项报告生成方法（如 `generate_literature_review`, `generate_industry_research_report`, `generate_popular_science_report`）或标准的 `generate_report` 方法。
    *   通过服务器发送事件 (SSE) 将每一步的处理进度和最终生成的报告（Markdown格式）流式传输回客户端。

该文件还包括：

*   日志配置。
*   模板目录配置 (用于 `index.html`)。
*   数据目录和数据库验证 (`db_utils.verify_database()`)。
*   SSE 进度步骤的定义 (`STEP_DEFINITIONS`)，以及一个辅助函数 `get_step_name(step_id, report_type)`，该函数会根据报告类型动态调整第五步（报告生成阶段）的描述文本，确保前端能准确展示当前操作。
*   错误处理机制，用于在 SSE 流处理期间捕获和发送错误事件给前端。

#### 8.1.2 KnowlEdge.py

`KnowlEdge.py` 是项目的核心，包含了大部分业务逻辑。主要内容包括：

*   **配置 (CONFIG)**: 存储 API 密钥 (DeepSeek, Serper, 计划中的 Baidu Translate), 模型名称 (BERT `bert-base-multilingual-cased`, LLM `deepseek-chat`), 数据目录和数据库路径。
*   **BERT 工具函数**:
    *   `get_bert_embeddings(text)`: 获取文本的 BERT 嵌入。
    *   `compute_similarity(text1, text2)`: 计算两个文本间的余弦相似度。
*   **`UserProfileManager` 类**: 负责用户画像的创建、更新和存储。
    *   初始化 OpenAI 客户端 (针对 DeepSeek)。
    *   `_load_interest_categories()`: 加载预定义的兴趣分类 (`interest_categories.json`)；如果文件不存在或加载失败，则调用 `_create_default_categories()` 创建包含技术、科学、商业等大类的默认分类体系。
    *   `create_user(user_info)`: 创建新用户，在数据库中存储基本信息，并根据用户信息生成用户 ID。
    *   `extract_skills_from_resume(user_id, resume_text, max_skills)`: 使用 LLM (DeepSeek) 从简历中提取技能，Prompt 指导 LLM 以特定JSON格式返回技能名称、熟练程度和类别。方法包含对返回JSON的解析、清理和错误处理逻辑，并将提取的技能存入数据库。
    *   `extract_interests_from_resume(user_id, resume_text, max_interests)`: 使用 LLM (DeepSeek) 从简历中提取兴趣，并根据预定义分类进行归类，计算初始权重。Prompt 指导 LLM 以特定JSON格式返回兴趣主题、类别和权重。方法包含对返回JSON的解析、清理和错误处理逻辑（包括在解析失败时尝试基于关键词创建基本兴趣），并将提取的兴趣存入数据库。
    *   `record_search(user_id, query, platform)`: 记录用户搜索行为。
    *   `record_interaction(user_id, content_id, action_type)`: 记录用户与内容的交互。
    *   `update_interest_weights(user_id, topic, adjustment)`: 更新用户特定兴趣主题的权重。如果主题为新，则创建并尝试使用 `compute_similarity` 与现有兴趣类别进行关联。
    *   `apply_time_decay(user_id, decay_factor, days_threshold)`: 对旧兴趣应用时间衰减。
    *   `get_top_interests(user_id, limit)`: 获取用户权重最高的兴趣（每个主题取最新记录）。
    *   `analyze_search_patterns(user_id, days)`: 使用 LLM 分析用户近期的搜索模式，期望返回包含主导主题和模式描述的JSON结果。
    *   `generate_recommendations(user_id, count)`: 基于用户画像（顶级兴趣）使用 LLM 生成具体的内容推荐，期望返回包含`topic`和`reason`的JSON数组。
    *   `get_user_profile_summary(user_id)`: 获取用户画像的综合摘要信息，包括基本信息、技能、顶级兴趣、活动统计和最近搜索。
*   **`KnowledgeFlow` 类**: 定义了整个知识获取和报告生成的工作流程。
    *   初始化 OpenAI 客户端、Serper API 密钥和 `UserProfileManager`。
    *   `start_node(user_input)`: 收集并处理用户初始输入（包括 `num_papers`），计算更新周期，创建用户ID。
    *   `calculate_update_cycle(days)`: 根据天数计算起始和结束日期。
    *   `build_user_profile(user_input, cv_text)`: 构建用户画像。如果提供了简历文本，则调用 `UserProfileManager` 的方法提取技能和兴趣；否则，创建基本用户档案，并可能基于用户输入的`content_type`初始化基本兴趣。确保用户 ID 存在。
    *   `display_profile_summary()`: 在控制台打印用户画像摘要。
    *   `_call_llm(prompt, system_message)`: (异步) 核心的LLM调用辅助函数。接收用户Prompt和系统角色Prompt，向配置的LLM（DeepSeek）发起请求，并对返回的文本内容进行后处理（如去除多余空白、规范化段落间的换行符）。
    *   `_prepare_llm_context_from_search_results(search_results, max_items_per_source, max_chars)`: 从多源搜索结果中提取标题、摘要等信息，格式化为一段文本，作为上下文供LLM在生成报告时参考。会限制每个来源的条目数和总字符数。
    *   `_update_interests_from_query(query, weight_adjustment)`: 使用 LLM 从用户当前的搜索查询中提取最多3个主要兴趣点，并调用`UserProfileManager`更新用户画像中的权重。
    *   `translate_query(query)`: (异步) 使用 `GoogleTranslator` 翻译查询到英文，然后使用 LLM (DeepSeek) 对翻译结果进行验证或选择最佳翻译。此过程也会触发 `_update_interests_from_query`。
    *   `build_search_query()`: (异步) 根据用户输入和关注领域构建搜索查询，包括Google搜索查询、ArXiv直接查询和针对ArXiv的Google搜索查询。查询词会通过 `translate_query` 进行翻译。
    *   `execute_search(queries)`: 根据用户选择的平台类型 (学术期刊, 新闻类, 综合类) 和指定的 `num_papers`（文献数量）执行相应的搜索 (Google, ArXiv)。文献数量会根据平台类型和来源数量进行智能分配。
    *   `google_search(query, max_results)`: 调用 Serper API 执行 Google 搜索，并限制结果数量。
    *   `parse_google_results(data, query)`: 解析 Google 搜索结果，提取标题、摘要、链接等，并使用 `compute_similarity` 计算与原始查询的相似度进行排序。
    *   `arxiv_search(query, max_results)`: 调用 ArXiv API 执行搜索，并限制结果数量。
    *   `parse_arxiv_response(xml_data, query)`: 解析 ArXiv XML 响应，提取元数据，并使用 `compute_similarity` 计算与原始查询的相似度进行排序。
    *   `google_arxiv_search(query, max_results)`: 通过 Google 搜索 ArXiv 文献，复用 `google_search` 和 `parse_google_results`。
    *   `_extract_interest_from_content(title, snippet, weight_adjustment)`: 从搜索结果的内容（标题和摘要）中使用LLM提取核心主题，并调用`UserProfileManager`更新用户模型。
    *   **`generate_literature_review(search_results, original_query)` (async)**: 准备LLM上下文，使用特定的系统角色提示 (`system_message_lit_review`) 指导LLM扮演学术研究员，按照预设Markdown结构生成文献综述，并自动附加参考文献列表。
    *   **`generate_industry_research_report(search_results, user_input, original_query)` (async)**: 准备LLM上下文，使用特定的系统角色提示 (`system_message_industry_analyst`) 指导LLM扮演行业分析师，按照预设Markdown结构生成行业调研报告。
    *   **`generate_popular_science_report(search_results, user_input, original_query)` (async)**: 准备LLM上下文，使用特定的系统角色提示 (`system_msg_science_writer`) 指导LLM扮演科普作家，按照预设Markdown结构（可能包含Mermaid图表提示）生成知识科普文章。
    *   **`generate_report(search_results)`**: (标准报告或回退方法) 调用 `integrate_with_large_model()` 使用 LLM (DeepSeek) 对各来源的搜索结果进行初步整合、摘要和翻译（确保输出为中文），LLM的提示词指导其以特定格式（来源、标题、摘要、原文网址、相似度）呈现信息。
    *   `integrate_with_large_model(search_results)`: (由 `generate_report` 调用) 使用 LLM (DeepSeek) 整合搜索结果并生成摘要。
    *   `send_email(report)`: （示例）发送邮件，当前为打印到控制台。
    *   `process_user_feedback(content_id, feedback_type, feedback_text)`: 处理用户对内容的反馈（喜欢、不喜欢等），使用LLM从反馈内容中提取主题，并相应调用`UserProfileManager`调整兴趣权重。
*   **`ResumeReader` 类**: 用于读取多种格式的简历文件。
    *   支持 `.txt`, `.pdf` (PyPDF2), `.docx` (python-docx), `.doc` (提示转换), `.xlsx`/`.xls` (pandas), `.jpg`/`.jpeg`/`.png` (Pillow, Pytesseract OCR - 使用 `lang='chi_sim+eng'` 支持中英文)。
    *   `read_resume(file_path)`: 根据文件扩展名调用相应的读取方法。如果未提供路径，则提示用户选择输入方式（文本或文件），并包含对文件不存在的错误处理。
*   **`collect_user_input()` 函数**: （主要用于本地测试）收集用户输入，目前硬编码了一些默认值。
*   **`main()` 函数**: (异步) 主执行流程，用于本地测试，模拟 `app.py` 中的 SSE 流程，调用 `KnowledgeFlow` 的核心方法。

#### 8.1.3 db_utils.py

`db_utils.py` 文件负责数据库的初始化和连接管理。它使用 SQLite 作为数据库。

主要功能包括：

*   **数据库配置 (`DB_CONFIG`)**: 定义数据目录 (`./user_data`) 和数据库文件路径 (`./user_data/user_profiles.db`)。
*   **`get_db_connection()`**:
    *   获取数据库连接。
    *   确保数据目录存在。
    *   如果数据库文件不存在，则创建该文件并初始化表结构。
    *   设置 `conn.row_factory = sqlite3.Row` 以便按列名访问查询结果。
    *   定义的表结构包括：
        *   `users`: 存储用户信息 (id, name, occupation, email, created_at)。
        *   `user_interests`: 存储用户兴趣 (id, user_id, topic, category, weight, timestamp)。
        *   `user_searches`: 存储用户搜索记录 (id, user_id, query, platform, timestamp)。
        *   `user_interactions`: 存储用户与内容的交互 (id, user_id, content_id, action_type, timestamp)。
        *   `user_skills`: 存储用户技能 (id, user_id, skill, level, category, timestamp)。
*   **`verify_database()`**:
    *   验证数据库是否正确创建并且可读写。
    *   通过写入和读取测试数据来进行验证。
    *   打印数据库中的表名和各表的记录数。
    *   注意：这个函数名在文件中定义了两次，一次使用 `logger`，一次使用 `print`。第二个定义会覆盖第一个。
*   **`initialize_database()`**:
    *   通过调用 `get_db_connection()` 来确保数据库和表已创建。

这个文件为应用程序提供了持久化存储用户数据的基础。

#### 8.1.4 index.html

`templates/index.html` 文件定义了用户界面的结构和交互。

**主要组成部分**：

1.  **HTML 结构**:
    *   一个主容器 (`<div class="container">`)。
    *   标题 "KnowlEdge 个性化知识更新"。
    *   一个表单 (`<form id="knowledgeForm">`)，包含以下输入字段：
        *   用户名 (`user_name`, text, 默认 "Tssword")
        *   职业 (`occupation`, text, 默认 "算法工程师")
        *   知识更新周期 (`day`, number, 默认 7)
        *   消息来源平台 (`platform`, select, 选项: 学术期刊 (默认), 新闻类, 综合类)
        *   关注领域 (`content_type`, text, 默认 "自然语言处理")
        *   邮箱 (`email`, email, 默认 "114514@qq.com")
        *   **报告类型 (`report_type`, select, 选项: 标准行业报告 (默认), 文献综述报告, 行业调研报告, 知识科普报告)**
        *   **文献/报告数量 (`num_papers`, number, 默认 10, min 5, max 20)**
        *   简历文本 (`cv_text`, textarea, 可选)
        *   提交按钮 ("生成报告")
    *   一个加载覆盖层 (`<div id="loadingOverlay">`)，包含：
        *   加载动画 (`spinner`)。
        *   加载状态文本 (`loadingStatus`)。
        *   进度条 (`progressBarFill` 在 `progress-bar-container` 内)。
        *   处理步骤列表 (`progressSteps`)。
    *   结果显示区域 (`<div class="results-area">`)，包含：
        *   报告容器 (`reportContainer` -> `reportContent`)。
        *   错误信息容器 (`errorContainer` -> `errorContent`)。

2.  **CSS 样式**:
    *   提供了现代化的、响应式的页面布局和元素样式。
    *   包括表单元素、按钮、加载覆盖层、进度条、步骤列表、报告和错误消息区域的样式。
    *   加载动画 (`@keyframes spin`)。

3.  **JavaScript 逻辑**:
    *   引入 `Marked.js` (用于将Markdown渲染为HTML) 和 `Mermaid.js` (用于渲染Mermaid图表)。
    *   `linkifyReport(reportText)`: 将报告文本中的 URL (以 "原文网址：" 或 "链接：" 开头) 转换为可点击的链接。
    *   `TOTAL_STEPS` 和 `STEP_DEFINITIONS`: 定义前端展示的步骤总数和基础名称。
    *   **表单提交处理 (`knowledgeForm.addEventListener('submit', ...)`):**
        *   阻止默认表单提交。
        *   获取表单数据 (`FormData`)。
        *   **UI 初始化**: 显示加载覆盖层，重置进度条和步骤列表，隐藏旧的报告/错误。
        *   **发起 Fetch 请求**: 向 `/process` 端点发送 POST 请求，携带表单数据。
        *   **处理 HTTP 响应**:
            *   如果响应不 `ok` (例如服务器错误)，则尝试解析错误信息 (JSON 或文本) 并显示。
            *   如果响应 `ok`，则开始处理 SSE (Server-Sent Events) 流。
        *   **SSE 流处理**:
            *   使用 `response.body.getReader()` 和 `TextDecoder` 读取和解码流数据。
            *   解析每一条 SSE 消息 (`data: {...}\n\n`)。
            *   根据事件类型 (`progress`, `complete`, `error`) 更新 UI：
                *   `progress`: 更新进度条百分比，使用从SSE事件中获取的 `eventData.message` 更新加载状态文本（该消息由后端`app.py`中的`get_step_name`动态生成，能反映不同报告类型的具体步骤），并标记当前/已完成的处理步骤。
                *   `complete`: 将进度条设置为 100%，使用 `marked.parse()` 将返回的Markdown报告渲染为HTML并显示，更新状态为 "处理完成！"。
                *   `error` (从流中发出的): 抛出错误，由外层 `catch` 处理。
        *   **错误处理 (`catch (error)`)**: 捕获 fetch 错误、非 `ok` 响应或 SSE 处理中抛出的错误。在 UI 中显示错误信息，并将加载状态设置为 "处理失败"。
        *   **最终处理 (`finally`)**:
            *   在处理完成后 (成功或失败)，隐藏加载覆盖层。
            *   如果成功生成报告，会显示一个 `alert`。
            *   重置进度条状态以备下次使用。

该 HTML 文件提供了一个用户友好的界面，允许用户输入他们的偏好（包括细化的报告类型和文献数量），并通过 SSE 实时查看处理进度和最终生成的Markdown报告（由前端渲染为HTML）。



​    uvicorn app:app --reload --port 5001
