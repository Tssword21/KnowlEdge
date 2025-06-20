# app.py - 修改后的流式输出版本
import asyncio
import logging
import os
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.templating import Jinja2Templates
from KnowlEdge import KnowledgeFlow, CONFIG, verify_database, UserProfileManager

# --- 应用配置和初始化 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
app = FastAPI(title="KnowlEdge 智能引擎", version="1.0.0")

templates_dir = os.path.join(os.path.dirname(__file__), "../templates")
if not os.path.exists(templates_dir) or not os.path.isfile(os.path.join(templates_dir, "index.html")):
    logging.error(f"模板目录 '{templates_dir}' 或 'index.html' 未找到。HTML 页面可能无法加载。")
templates = Jinja2Templates(directory=templates_dir)

try:
    if not verify_database():
        logging.warning("数据库验证失败。系统可能无法按预期工作。请考虑运行 init_system.py 进行初始化。")
    os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
    logging.info(f"数据目录 '{CONFIG['DATA_DIR']}' 已确保存在。")
except Exception as e:
    logging.error(f"初始化设置错误 (数据库/目录检查): {e}", exc_info=True)

# --- SSE 进度步骤定义 ---
TOTAL_STEPS = 6
STEP_DEFINITIONS = [
    {"id": 1, "name": "接收和初始化"}, {"id": 2, "name": "用户画像分析"},
    {"id": 3, "name": "构建搜索参数"}, {"id": 4, "name": "执行搜索"},
    {"id": 5, "name": "整合结果并生成报告/综述"},
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
        else:
            return "整合结果并生成报告"
    
    for step_def in STEP_DEFINITIONS:
        if step_def["id"] == step_id: return step_def["name"]
    return "未知处理阶段"

# --- 核心异步生成器 (用于 SSE 事件流) ---
async def knowledge_flow_sse_generator(user_input_data: dict, cv_text_data: str, report_type: str, original_query: str):
    """
    异步生成器，为知识流处理过程生成服务器发送事件 (SSE)。
    现在支持流式输出报告内容。
    """
    current_step_id = 0
    try:
        async def yield_progress_dict(step_id_inner, message_override=None):
            nonlocal current_step_id
            current_step_id = step_id_inner
            message = message_override if message_override else get_step_name(step_id_inner, report_type)
            logging.info(f"SSE 进度 ({report_type}): 第 {step_id_inner}/{TOTAL_STEPS} 步 - {message}")
            yield {
                "event": "progress",
                "data": json.dumps({
                    "step": step_id_inner,
                    "total_steps": TOTAL_STEPS,
                    "message": message
                })
            }
            await asyncio.sleep(0.05)

        # --- 开始处理流程 ---

        # 步骤 1: 初始化
        async for sse_event in yield_progress_dict(1, "正在初始化处理流程..."):
            yield sse_event
        workflow = KnowledgeFlow()
        logging.info("KnowledgeFlow 引擎已为流式传输初始化。")

        # 步骤 2: 处理用户输入
        async for sse_event in yield_progress_dict(1):
            yield sse_event
        await asyncio.to_thread(workflow.start_node, user_input_data)
        logging.info(f"用户输入已处理: {user_input_data.get('user_name')}")

        # 步骤 3: 用户画像分析
        async for sse_event in yield_progress_dict(2):
            yield sse_event
        await asyncio.to_thread(workflow.build_user_profile, user_input_data, cv_text_data)
        logging.info("用户画像已构建或跳过。")

        # 步骤 4: 构建搜索参数
        async for sse_event in yield_progress_dict(3):
            yield sse_event
        queries = await workflow.build_search_query()
        logging.info(f"搜索查询已构建: {list(queries.keys())}")

        # 步骤 5: 执行搜索
        async for sse_event in yield_progress_dict(4):
            yield sse_event
        search_results = await asyncio.to_thread(workflow.execute_search, queries)
        logging.info(f"搜索执行完成。结果来源: {list(search_results.keys())}")

        # 步骤 6: 开始生成报告（这里开始流式输出）
        step5_message_override = get_step_name(5, report_type)
        async for sse_event in yield_progress_dict(5, step5_message_override):
            yield sse_event

        # 发送报告开始事件
        yield {
            "event": "report_start",
            "data": json.dumps({"message": "开始生成报告内容"})
        }
        
        # 根据报告类型调用相应的流式生成方法
        if report_type == "literature_review":
            logging.info(f"开始流式生成文献综述，查询: {original_query}")
            async for chunk in workflow.generate_literature_review_stream(search_results, original_query):
                yield {
                    "event": "report_chunk",
                    "data": chunk
                }
                await asyncio.sleep(0.01)  # 小延迟确保流畅输出
        elif report_type == "industry_research_report":
            logging.info(f"开始流式生成行业调研报告，查询: {original_query}")
            async for chunk in workflow.generate_industry_research_report_stream(search_results, user_input_data, original_query):
                yield {
                    "event": "report_chunk",
                    "data": chunk
                }
                await asyncio.sleep(0.01)
        elif report_type == "popular_science_report":
            logging.info(f"开始流式生成知识科普报告，查询: {original_query}")
            async for chunk in workflow.generate_popular_science_report_stream(search_results, user_input_data, original_query):
                yield {
                    "event": "report_chunk",
                    "data": chunk
                }
                await asyncio.sleep(0.01)
        else:  # standard_report
            logging.info("开始流式生成标准报告")
            async for chunk in workflow.generate_report_stream(search_results):
                yield {
                    "event": "report_chunk",
                    "data": chunk
                }
                await asyncio.sleep(0.01)

        # 步骤 7: 完成处理
        async for sse_event in yield_progress_dict(6):
            yield sse_event
        await asyncio.sleep(0.2)

        # 发送最终完成事件
        logging.info(f"所有步骤完成。{report_type} 流式输出完毕。")
        yield {
            "event": "complete",
            "data": json.dumps({"message": "报告生成完成"})
        }

    except Exception as e:
        error_step_name = get_step_name(current_step_id, report_type) if current_step_id > 0 else "初始化"
        logging.error(f"SSE 流处理错误 ({report_type}, 步骤 {current_step_id} - {error_step_name}): {e}", exc_info=True)
        error_message = f"在步骤 '{error_step_name}' 处理时发生错误: {str(e)}"
        yield {
            "event": "error",
            "data": json.dumps({
                "message": error_message,
                "step": current_step_id,
                "total_steps": TOTAL_STEPS
            })
        }

# --- FastAPI 路径操作 ---

@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    """提供主 HTML 页面。"""
    index_html_path = os.path.join(templates_dir, "index.html")
    if not os.path.isfile(index_html_path):
        logging.error(f"index.html 未在目录 '{templates_dir}' 中找到。")
        return HTMLResponse(content="<html><body><h1>错误 500：未找到主页面模板。请检查服务器配置。</h1></body></html>", status_code=500)
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def handle_process_submission(
    user_name: str = Form(default="未知用户"),
    occupation: str = Form(default="未知职业"),
    day: int = Form(default=7),
    platform: str = Form(default="学术期刊"),
    content_type: str = Form(default=""),
    email: str = Form(default=""),
    cv_text: str = Form(default=""),
    report_type: str = Form(default="standard_report"),
    num_papers: int = Form(default=10, ge=5, le=20)
):
    """处理表单提交并发起 SSE 事件流。"""
    logging.info(f"'/process' POST 路由命中。请求报告类型: {report_type}, 文献数量: {num_papers}。准备流式传输事件。")
    try:
        user_input_data = {
            "user_name": user_name, "occupation": occupation, "day": day,
            "platform": platform, "content_type": content_type, "email": email,
            "num_papers": num_papers
        }
        logging.info(f"接收到用户数据: {user_name}, 职业: {occupation}, 简历长度: {len(cv_text)}, 关注领域: {content_type}, 请求文献数: {num_papers}")

        return EventSourceResponse(knowledge_flow_sse_generator(user_input_data, cv_text, report_type, content_type))
    except Exception as e:
        logging.error(f"'/process' 路由在流式传输开始前发生错误 ({report_type}): {e}", exc_info=True)
        return {"status": "error", "message": f"启动流程失败: {str(e)}"}, 500

# --- Uvicorn 运行说明 ---
# 要运行此 FastAPI 应用:
# 1. 在您的终端 (激活了 Python 虚拟环境 .venv 的情况下（.venv\Scripts\activate.ps1)，导航到项目根目录。
# 2. 执行命令:
#    cd src
#    uvicorn app:app --reload --port 5001（或者 python -m uvicorn app:app --reload --port 5001）
