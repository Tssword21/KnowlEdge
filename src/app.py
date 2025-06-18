# app.py
import asyncio
import logging
import os
import json
from fastapi import FastAPI, Request, Form # FastAPI 的核心组件
from fastapi.responses import HTMLResponse, StreamingResponse # 用于 HTML 和流式 SSE
from fastapi.templating import Jinja2Templates # 用于渲染 HTML 模板
# from fastapi.staticfiles import StaticFiles # 如果您有单独的 CSS/JS 文件，可以取消注释

# --- 从您的 KnowlEdge 项目导入 ---
# 确保 KnowlEdge.py 和相关依赖在 Python 路径中
from KnowlEdge import KnowledgeFlow, CONFIG, verify_database, UserProfileManager

# --- 应用配置和初始化 ---

# 1. 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# 2. 创建 FastAPI 应用实例
app = FastAPI(title="KnowlEdge 智能引擎", version="1.0.0") # 这是 Uvicorn 将运行的 app 实例

# 3. 配置模板目录 (假设 index.html 在 ./templates/ 文件夹中)
templates_dir = os.path.join(os.path.dirname(__file__), "../templates")
if not os.path.exists(templates_dir) or not os.path.isfile(os.path.join(templates_dir, "index.html")):
    logging.error(f"模板目录 '{templates_dir}' 或 'index.html' 未找到。HTML 页面可能无法加载。")
    # 如果没有模板，根路径将无法正常工作。
    # 您可以考虑在这里引发一个更严重的错误或提供一个默认的文本响应。
templates = Jinja2Templates(directory=templates_dir)

# 如果您有静态文件 (如分离的 CSS/JS)，可以取消注释下面这行并创建 'static' 文件夹
# app.mount("/static", StaticFiles(directory="static"), name="static")

# 4. 确保数据目录存在并验证数据库
try:
    if not verify_database():
        logging.warning("数据库验证失败。系统可能无法按预期工作。请考虑运行 init_system.py 进行初始化。")
    os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
    logging.info(f"数据目录 '{CONFIG['DATA_DIR']}' 已确保存在。")
except Exception as e:
    logging.error(f"初始化设置错误 (数据库/目录检查): {e}", exc_info=True)

# --- SSE 进度步骤定义 (与前端 JavaScript 中的定义一致) ---
TOTAL_STEPS = 6
STEP_DEFINITIONS = [
    {"id": 1, "name": "接收和初始化"}, {"id": 2, "name": "用户画像分析"},
    {"id": 3, "name": "构建搜索参数"}, {"id": 4, "name": "执行搜索"},
    {"id": 5, "name": "整合结果并生成报告/综述"}, # 将根据类型动态调整
    {"id": 6, "name": "处理完成"}
]

def get_step_name(step_id, report_type="standard_report"):
    """辅助函数：根据步骤 ID 获取描述性的步骤名称。"""
    if step_id == 5:
        if report_type == "literature_review":
            return "整合结果并生成文献综述"
        elif report_type == "industry_research_report":
            return "整合结果并生成行业调研报告"
        elif report_type == "popular_science_report":
            return "整合结果并生成知识科普报告"
        else: # standard_report 及其他未知类型
            return "整合结果并生成报告"
    
    for step_def in STEP_DEFINITIONS:
        if step_def["id"] == step_id: return step_def["name"]
    return "未知处理阶段"

# --- 核心异步生成器 (用于 SSE 事件流) ---
async def knowledge_flow_sse_generator(user_input_data: dict, cv_text_data: str, report_type: str, original_query: str):
    """
    异步生成器，为知识流处理过程生成服务器发送事件 (SSE)。
    它 yield SSE 格式的、UTF-8 编码的字节串。
    """
    current_step_id = 0
    try:
        async def yield_progress_bytes_inner(step_id_inner, message_override=None):
            nonlocal current_step_id
            current_step_id = step_id_inner
            message = message_override if message_override else get_step_name(step_id_inner, report_type)
            logging.info(f"SSE 进度 ({report_type}): 第 {step_id_inner}/{TOTAL_STEPS} 步 - {message}")
            event_data = {
                'type': 'progress',
                'step': step_id_inner,
                'total_steps': TOTAL_STEPS,
                'message': message
            }
            yield f"data: {json.dumps(event_data)}\n\n".encode('utf-8')
            await asyncio.sleep(0.05)

        # --- 开始处理流程 ---

        # 步骤 0 (初始化)
        # 我们需要从 yield_progress_bytes_inner 中 yield 数据
        async for sse_event in yield_progress_bytes_inner(1, "正在初始化处理流程..."):
            yield sse_event
        workflow = KnowledgeFlow()
        logging.info("KnowledgeFlow 引擎已为流式传输初始化。")

        # 步骤 1: 处理用户输入 (start_node)
        async for sse_event in yield_progress_bytes_inner(1): # 步骤名称 "接收和初始化"
            yield sse_event
        await asyncio.to_thread(workflow.start_node, user_input_data)
        logging.info(f"用户输入已处理: {user_input_data.get('user_name')}")

        # 步骤 2: 用户画像分析 (build_user_profile)
        async for sse_event in yield_progress_bytes_inner(2): # 步骤名称 "用户画像分析"
            yield sse_event
        await asyncio.to_thread(workflow.build_user_profile, user_input_data, cv_text_data)
        logging.info("用户画像已构建或跳过。")

        # 步骤 3: 构建搜索参数 (build_search_query)
        async for sse_event in yield_progress_bytes_inner(3): # 步骤名称 "构建搜索参数"
            yield sse_event
        queries = await workflow.build_search_query()
        logging.info(f"搜索查询已构建: {list(queries.keys())}")

        # 步骤 4: 执行搜索 (execute_search)
        async for sse_event in yield_progress_bytes_inner(4): # 步骤名称 "执行搜索"
            yield sse_event
        search_results = await asyncio.to_thread(workflow.execute_search, queries)
        logging.info(f"搜索执行完成。结果来源: {list(search_results.keys())}")

        # 步骤 5: 生成报告或文献综述
        step5_message_override = get_step_name(5, report_type)
        async for sse_event in yield_progress_bytes_inner(5, step5_message_override):
            yield sse_event
        
        report_text = ""
        if report_type == "literature_review":
            logging.info(f"开始生成文献综述，查询: {original_query}")
            report_text = await workflow.generate_literature_review(search_results, original_query)
        elif report_type == "industry_research_report":
            logging.info(f"开始生成行业调研报告，查询: {original_query}")
            report_text = await workflow.generate_industry_research_report(search_results, user_input_data, original_query)
        elif report_type == "popular_science_report":
            logging.info(f"开始生成知识科普报告，查询: {original_query}")
            report_text = await workflow.generate_popular_science_report(search_results, user_input_data, original_query)
        else: # standard_report
            logging.info("开始生成标准报告")
            report_text = await asyncio.to_thread(workflow.generate_report, search_results)
        
        logging.info(f"{report_type} 已生成，长度: {len(report_text)}")
        
        # 步骤 6: 完成处理
        async for sse_event in yield_progress_bytes_inner(6): # 步骤名称 "处理完成"
            yield sse_event
        await asyncio.sleep(0.2)

        # --- 发送处理完成的 SSE 事件 ---
        logging.info(f"所有步骤完成。通过 SSE 发送最终 {report_type}。")
        complete_event = {'type': 'complete', 'report': report_text}
        yield f"data: {json.dumps(complete_event)}\n\n".encode('utf-8') # 直接 yield 最终事件

    except Exception as e:
        error_step_name = get_step_name(current_step_id, report_type) if current_step_id > 0 else "初始化"
        logging.error(f"SSE 流处理错误 ({report_type}, 步骤 {current_step_id} - {error_step_name}): {e}", exc_info=True)
        error_message = f"在步骤 '{error_step_name}' 处理时发生错误: {str(e)}"
        error_event = {'type': 'error', 'message': error_message, 'step': current_step_id, 'total_steps': TOTAL_STEPS}
        yield f"data: {json.dumps(error_event)}\n\n".encode('utf-8') # 直接 yield 错误事件

# --- FastAPI 路径操作 (路由) ---

@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    """提供主 HTML 页面。"""
    # 确保 templates/index.html 存在
    index_html_path = os.path.join(templates_dir, "index.html")
    if not os.path.isfile(index_html_path):
        logging.error(f"index.html 未在目录 '{templates_dir}' 中找到。")
        return HTMLResponse(content="<html><body><h1>错误 500：未找到主页面模板。请检查服务器配置。</h1></body></html>", status_code=500)
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def handle_process_submission(
    # 使用 FastAPI 的 Form(...) 从 POST 请求中提取表单数据
    user_name: str = Form(default="未知用户"),
    occupation: str = Form(default="未知职业"),
    day: int = Form(default=7),
    platform: str = Form(default="学术期刊"),
    content_type: str = Form(default=""),
    email: str = Form(default=""),
    cv_text: str = Form(default=""),
    report_type: str = Form(default="standard_report"),
    num_papers: int = Form(default=10, ge=5, le=20) # 新增文献数量参数
):
    """处理表单提交并发起 SSE 事件流。"""
    logging.info(f"'/process' POST 路由命中。请求报告类型: {report_type}, 文献数量: {num_papers}。准备流式传输事件。")
    try:
        user_input_data = {
            "user_name": user_name, "occupation": occupation, "day": day,
            "platform": platform, "content_type": content_type, "email": email,
            "num_papers": num_papers # 将文献数量添加到用户输入数据中
        }
        logging.info(f"接收到用户数据: {user_name}, 职业: {occupation}, 简历长度: {len(cv_text)}, 关注领域: {content_type}, 请求文献数: {num_papers}")

        return StreamingResponse(knowledge_flow_sse_generator(user_input_data, cv_text, report_type, content_type),
                                 media_type="text/event-stream")
    except Exception as e:
        logging.error(f"'/process' 路由在流式传输开始前发生错误 ({report_type}): {e}", exc_info=True)
        # FastAPI 会自动将字典转换为 JSON 响应，并根据状态码推断。
        # 如果要明确设置状态码，可以使用 Response(content=json.dumps(...), status_code=500, media_type="application/json")
        return {"status": "error", "message": f"启动流程失败: {str(e)}"}, 500

# --- Uvicorn 运行说明 ---
# 要运行此 FastAPI 应用:
# 1. 在您的终端 (激活了 Python 虚拟环境 .venv 的情况下（.venv\Scripts\activate.ps1）)，导航到项目根目录。
# 2. 执行命令:
#    cd src
#    uvicorn app:app --reload --port 5001
