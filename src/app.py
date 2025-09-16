# app.py - 修改后的流式输出版本
import os
import sys
import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入路径问题
try:
    # 当在项目根目录运行时
    from src.core.knowledge_flow import KnowledgeFlow
    from src.config import Config
    from src.utils import setup_logging, verify_database, get_user_data_path
except ImportError:
    # 当在src目录下运行时
    from core.knowledge_flow import KnowledgeFlow
    from config import Config
    from utils import setup_logging, verify_database, get_user_data_path

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tempfile

# --- 应用配置和初始化 ---
setup_logging()
app = FastAPI(title="KnowlEdge 智能引擎", version="1.0.0")

# 获取配置
config = Config()

# 优雅关闭全局HTTP会话
try:
    from src.core.llm_interface import _GLOBAL_SESSION
except Exception:
    from core.llm_interface import _GLOBAL_SESSION

@app.on_event("shutdown")
async def shutdown_event():
    try:
        if _GLOBAL_SESSION and not _GLOBAL_SESSION.closed:
            await _GLOBAL_SESSION.close()
            logging.info("已关闭全局HTTP会话")
    except Exception as e:
        logging.warning(f"关闭全局HTTP会话失败: {e}")

# 启动时做关键配置自检与提示
@app.on_event("startup")
async def startup_event():
    try:
        cfg = Config()
        missing = []
        if not (cfg.llm_api_key and cfg.llm_api_key.strip()):
            missing.append("DEEPSEEK_API_KEY")
        if not (cfg.serper_api_key and cfg.serper_api_key.strip()):
            logging.warning("SERPER_API_KEY 未设置：谷歌相关平台将不可用。")
        if missing:
            logging.warning(f"缺少必要配置: {', '.join(missing)}")
    except Exception as e:
        logging.warning(f"启动配置自检失败: {e}")

# 启动时自动初始化（数据库与兴趣分类）
try:
    # 数据目录
    os.makedirs(get_user_data_path(), exist_ok=True)
    # 初始化数据库（自动补建缺表）
    try:
        from src.db_utils import initialize_database
    except ImportError:
        from db_utils import initialize_database
    if initialize_database():
        logging.info("数据库初始化/校验完成。")
    else:
        logging.warning("数据库初始化/校验失败，后续操作可能异常。")

    # 初始化兴趣分类文件（若不存在）
    interest_file = os.path.join(get_user_data_path(), "interest_categories.json")
    if not os.path.exists(interest_file):
        try:
            # 复用脚本中的默认创建逻辑
            from src.scripts.init_system import create_interest_categories as _create_ic
        except Exception:
            try:
                from scripts.init_system import create_interest_categories as _create_ic
            except Exception:
                _create_ic = None
        if _create_ic:
            ok = _create_ic()
            if ok:
                logging.info("兴趣分类文件已创建。")
            else:
                logging.warning("兴趣分类文件创建失败。")
        else:
            logging.warning("未找到兴趣分类创建函数，跳过创建。")

    # 最后再做一次校验输出
    if not verify_database():
        logging.warning("数据库验证失败。系统可能无法按预期工作。请考虑运行 init_system.py 进行初始化。")
    logging.info(f"数据目录 '{get_user_data_path()}' 已确保存在。")
except Exception as e:
    logging.error(f"初始化设置错误 (数据库/目录检查): {e}", exc_info=True)

# 设置静态文件和模板
templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
if not os.path.exists(templates_dir) or not os.path.isfile(os.path.join(templates_dir, "index.html")):
    logging.error(f"模板目录 '{templates_dir}' 或 'index.html' 未找到。HTML 页面可能无法加载。")
templates = Jinja2Templates(directory=templates_dir)

# 静态文件服务
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    logging.warning(f"静态文件目录 '{static_dir}' 未找到。静态资源可能无法加载。")
    os.makedirs(static_dir, exist_ok=True)

# --- SSE 进度步骤定义 ---
TOTAL_STEPS = 6
STEP_DEFINITIONS = [
    {"id": 1, "name": "接收和初始化"}, {"id": 2, "name": "用户画像分析"},
    {"id": 3, "name": "构建搜索参数"}, {"id": 4, "name": "执行搜索"},
    {"id": 5, "name": "整合结果并生成报告/综述"},
    {"id": 6, "name": "处理完成"}
]

# 定义报告类型映射
REPORT_TYPE_MAP = {
    "standard": "standard",                    # 标准报告
    "literature_review": "literature_review",  # 文献综述
    "industry_research_report": "industry_research",  # 行业研究报告
    "popular_science_report": "popular_science"      # 科普知识报告
}

def get_step_name(step_id, report_type="standard"):
    """辅助函数：根据步骤 ID 获取描述性的步骤名称。"""
    if step_id == 5:
        # 根据报告类型映射显示不同的第5步描述
        normalized_type = REPORT_TYPE_MAP.get(report_type, "standard")
        if normalized_type == "literature_review":
            return "整合结果并生成文献综述"
        elif normalized_type == "industry_research":
            return "整合结果并生成行业研究报告"
        elif normalized_type == "popular_science":
            return "整合结果并生成科普知识报告"
        else:
            return "整合结果并生成标准报告"
    
    for step_def in STEP_DEFINITIONS:
        if step_def["id"] == step_id: return step_def["name"]
    return "未知处理阶段"

# --- 核心异步生成器 (用于 SSE 事件流) ---
async def knowledge_flow_sse_generator(user_input_data: dict, resume_file_path: Optional[str] = None, report_type: str = "standard", original_query: str = ""):
    """
    异步生成器，为知识流处理过程生成服务器发送事件 (SSE)。
    支持流式输出报告内容。
    """
    current_step_id = 0
    temp_file_path = None
    
    # 规范化报告类型
    normalized_report_type = REPORT_TYPE_MAP.get(report_type, "standard")
    
    # 轻量清洗SSE分片，避免非UTF字符
    def _sanitize_chunk(text: Any) -> str:
        try:
            s = text if isinstance(text, str) else str(text)
        except Exception:
            s = ""
        return s.replace("\x00", "").replace("\r\n", "\n")
    
    try:
        # 初始化KnowledgeFlow
        workflow = KnowledgeFlow(llm_model=user_input_data.get('llm_model'))
        
        # 为进度状态更新创建辅助函数
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

        # --- 开始处理流程 ---

        # 步骤 1: 初始化
        async for sse_event in yield_progress_dict(1, "正在初始化处理流程..."):
            yield sse_event
        
        # 生成用户ID，这里简单使用用户名
        user_id = user_input_data.get("user_name", "anonymous").lower().replace(" ", "_")
        user_name = user_input_data.get("user_name", "未知用户")
        logging.info(f"处理用户: {user_name}, ID: {user_id}")

        # 步骤 2: 用户画像分析
        async for sse_event in yield_progress_dict(2):
            yield sse_event
            
        # 如果有简历文件路径，直接分析；否则创建/获取基本画像
        if resume_file_path and os.path.exists(resume_file_path):
            try:
                temp_file_path = resume_file_path
                analyze_result = await workflow.analyze_resume(user_id, temp_file_path)
                if analyze_result and analyze_result.get("success"):
                    logging.info("简历分析完成（已保存的临时文件）")
                else:
                    logging.warning(f"简历分析未成功：{(analyze_result or {}).get('message', '未知原因')}")
                    await workflow.get_or_create_user_profile(user_id, user_name)
            except Exception as e:
                logging.error(f"处理简历文件时出错: {str(e)}")
                await workflow.get_or_create_user_profile(user_id, user_name)
        else:
            # 如果没有简历，只确保用户画像存在
            await workflow.get_or_create_user_profile(user_id, user_name)
            logging.info("无简历文件，已创建或获取基本用户画像。")

        # 步骤 3: 构建搜索参数
        async for sse_event in yield_progress_dict(3):
            yield sse_event
            
        # 准备平台列表
        platform_map = {
            "arXiv论文": ["arxiv"],
            "谷歌学术": ["google_scholar"],
            "混合搜索": ["google_scholar", "arxiv"], 
            "综合资讯": ["web", "news"],
            "全平台": ["google_scholar", "arxiv", "patent", "web", "news"]
        }
        platform_type = user_input_data.get("platform", "arXiv论文")
        platforms = platform_map.get(platform_type, ["web"])
        
        # 获取查询内容
        query = original_query or user_input_data.get("content_type", "")
        if not query:
            raise ValueError("搜索内容不能为空")
            
        logging.info(f"搜索平台: {platforms}, 查询内容: {query}")

        # 步骤 4: 执行搜索
        async for sse_event in yield_progress_dict(4):
            yield sse_event
            
        # 设置搜索结果数量
        max_results = user_input_data.get("num_papers", 15)  # 增加默认结果数量
        
        # 如果是混合搜索，设置每个平台的结果数量
        platform_max_results = {}
        if platform_type == "混合搜索":
            # 对于混合搜索，平均分配结果数量
            per_platform_results = max(1, max_results // len(platforms))
            platform_max_results = {
                "google_scholar": per_platform_results,
                "arxiv": max_results - per_platform_results  # 确保总数不超过用户要求
            }
        else:
            # 对于其他平台，每个平台使用相同的最大结果数
            for platform in platforms:
                platform_max_results[platform] = max_results
        
        # 获取排序方式
        sort_by = user_input_data.get("sort_by", "relevance")
        
        # 获取时间范围
        time_range = user_input_data.get("time_range", None)
        
        # 获取类别列表
        categories = user_input_data.get("categories", None)
        
        # 执行个性化搜索
        search_result_data = await workflow.personalized_search(
            user_id, 
            query, 
            platforms, 
            platform_max_results,
            sort_by,
            time_range,
            categories
        )
        
        search_results = search_result_data["results"]
        enhanced_query = search_result_data["enhanced_query"]
        logging.info(f"搜索执行完成。查询: {query} -> {enhanced_query}. 平台: {platforms}")

        # 步骤 5: 开始生成报告（流式输出）
        step5_message_override = get_step_name(5, report_type)
        async for sse_event in yield_progress_dict(5, step5_message_override):
            yield sse_event

        # 发送报告开始事件
        yield {
            "event": "report_start",
            "data": json.dumps({"message": "开始生成报告内容"})
        }
        
        # 构建用户输入数据用于报告生成
        user_report_input = {
            "query": query,
            "enhanced_query": enhanced_query,
            "platform_type": platform_type,
            "user_name": user_name,
            "occupation": user_input_data.get("occupation", ""),
            "day": user_input_data.get("day", 7),
            "email": user_input_data.get("email", ""),
        }
        
        # 根据报告类型调用相应的流式生成方法
        if normalized_report_type == "literature_review":
            async for chunk in workflow.generate_literature_review_stream(search_results, original_query):
                yield {
                    "event": "report_chunk",
                    "data": _sanitize_chunk(chunk)
                }
        elif normalized_report_type == "industry_research":
            async for chunk in workflow.generate_industry_research_report_stream(search_results, user_report_input, original_query):
                yield {
                    "event": "report_chunk",
                    "data": _sanitize_chunk(chunk)
                }
        elif normalized_report_type == "popular_science":
            async for chunk in workflow.generate_popular_science_report_stream(search_results, user_report_input, original_query):
                yield {
                    "event": "report_chunk",
                    "data": _sanitize_chunk(chunk)
                }
        else:
            # 默认使用标准报告生成器
            async for chunk in workflow.generate_report_stream(search_results, user_report_input):
                yield {
                    "event": "report_chunk",
                    "data": _sanitize_chunk(chunk)
                }

        # 步骤 6: 完成处理
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
                "code": "INTERNAL_ERROR",
                "step": current_step_id,
                "total_steps": TOTAL_STEPS
            })
        }
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logging.info(f"临时文件已删除: {temp_file_path}")
            except Exception as e:
                logging.warning(f"删除临时文件失败: {temp_file_path}, 错误: {str(e)}")

# --- FastAPI 路径操作 ---

@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    """提供主 HTML 页面。"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def handle_process_submission(
    request: Request,
    user_name: str = Form(default="未知用户"),
    occupation: str = Form(default="未知职业"),
    day: int = Form(default=7),
    platform: str = Form(default="arXiv论文"),
    content_type: str = Form(default=""),
    email: str = Form(default=""),
    resume_file: UploadFile = File(None),
    report_type: str = Form(default="standard"),
    num_papers: int = Form(default=10, ge=5, le=20),
    sort_by: str = Form(default="relevance"),
    time_filter: str = Form(default="recent"),
    categories: str = Form(default=""),
    time_from: Optional[str] = Form(None),
    time_to: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None)
):
    """处理表单提交并发起 SSE 事件流。"""
    logging.info(f"'/process' POST 路由命中。请求报告类型: {report_type}, 文献数量: {num_papers}。准备流式传输事件。")
    try:
        # 关键配置校验（早期失败更友好）
        cfg = Config()
        missing_keys = []
        if not (cfg.llm_api_key and cfg.llm_api_key.strip()):
            missing_keys.append("DEEPSEEK_API_KEY")
        if not (cfg.serper_api_key and cfg.serper_api_key.strip()):
            # 仅当选择含 web/google_scholar/全平台/综合资讯/混合搜索时强制
            needs_serper = platform in ["谷歌学术", "混合搜索", "综合资讯", "全平台"]
            if needs_serper:
                missing_keys.append("SERPER_API_KEY")
        if missing_keys:
            return JSONResponse(content={"error": f"缺少必要配置: {', '.join(missing_keys)}", "code": "MISSING_CONFIG"}, status_code=400)
        
        # 处理类别列表
        category_list = None
        if categories and categories.strip():
            category_list = [cat.strip() for cat in categories.split(',') if cat.strip()]
        
        # 处理时间过滤
        time_range = None
        if time_filter == "recent":
            time_range = {"days": day}
        elif time_filter == "custom" and time_from and time_to:
            time_range = {"from": time_from, "to": time_to}
                
        user_input_data = {
            "user_name": user_name, 
            "occupation": occupation, 
            "day": day,
            "platform": platform, 
            "content_type": content_type, 
            "email": email,
            "num_papers": num_papers,
            "sort_by": sort_by,
            "time_range": time_range,
            "categories": category_list,
            "llm_model": llm_model
        }
        
        # 在开始流之前读取保存简历到临时文件，避免流式响应导致上传句柄关闭
        resume_temp_path = None
        resume_file_info = ", 无简历文件"
        try:
            if resume_file and getattr(resume_file, "filename", None) and str(resume_file.filename).strip():
                # 读取全部内容
                try:
                    await resume_file.seek(0)
                except Exception:
                    pass
                content = await resume_file.read()
                if content:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.filename)[1])
                    tmp.write(content)
                    tmp.flush()
                    tmp.close()
                    resume_temp_path = tmp.name
                    resume_file_info = f", 简历文件: {resume_file.filename}, 大小: {len(content)}B"
                else:
                    resume_file_info = f", 简历文件: {resume_file.filename}, 但内容为空"
        except Exception as e:
            logging.warning(f"读取上传简历失败：{e}")
        logging.info(f"接收到用户数据: {user_name}, 职业: {occupation}{resume_file_info}, 关注领域: {content_type}, 请求文献数: {num_papers}")

        return EventSourceResponse(knowledge_flow_sse_generator(
            user_input_data,
            resume_temp_path,
            report_type,
            content_type
        ))
    except Exception as e:
        logging.error(f"'/process' 路由在流式传输开始前发生错误 ({report_type}): {e}", exc_info=True)
        return {"status": "error", "message": f"启动流程失败: {str(e)}"}, 500

@app.post('/api/enhanced_literature_review')
async def enhanced_literature_review(request: Request):
    """
    使用多模型并行处理生成增强版文献综述
    """
    try:
        data = await request.json()
        query = data.get('query', '')
        platform = data.get('platform', 'arxiv')
        num_results = int(data.get('num_results', 8))
        llm_model = data.get('llm_model')
        
        if not query:
            return JSONResponse(content={"error": "查询不能为空"}, status_code=400)
            
        knowledge_flow = KnowledgeFlow(llm_model=llm_model)
        report = await knowledge_flow.generate_enhanced_literature_review(query, platform, num_results)
        return JSONResponse(content={"result": report})
        
    except Exception as e:
        logging.error(f"生成增强版文献综述时出错: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post('/api/enhanced_literature_review_stream')
async def enhanced_literature_review_stream(request: Request):
    """
    使用多模型并行处理流式生成增强版文献综述
    """
    try:
        data = await request.json()
        query = data.get('query', '')
        platform = data.get('platform', 'arxiv')
        num_results = int(data.get('num_results', 8))
        llm_model = data.get('llm_model')
        citation_style = data.get('citation_style', 'markdown')
        
        if not query:
            return JSONResponse(content={"error": "查询不能为空"}, status_code=400)
            
        knowledge_flow = KnowledgeFlow(llm_model=llm_model)
        
        async def generate():
            try:
                async for chunk in knowledge_flow.generate_enhanced_literature_review_stream(query, platform, num_results):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            except Exception as e:
                logging.error(f"流式生成增强版文献综述时出错: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                yield f"data: {json.dumps({'done': True})}\n\n"
                
        return EventSourceResponse(generate())
    except Exception as e:
        logging.error(f"处理增强版文献综述流请求时出错: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post('/api/get_references')
async def get_references(request: Request):
    """
    从搜索结果获取格式化的参考文献列表
    """
    try:
        data = await request.json()
        query = data.get('query', '')
        platform = data.get('platform', 'arxiv') 
        num_results = int(data.get('num_results', 10))
        citation_style = data.get('style', 'markdown')  # 支持 markdown, html, text
        
        if not query:
            return JSONResponse(content={"error": "查询不能为空"}, status_code=400)
        
        # 创建知识流处理器和引用格式化器
        knowledge_flow = KnowledgeFlow()
        reference_formatter = knowledge_flow.reference_formatter
        
        # 执行搜索
        search_results = await knowledge_flow.search_manager.search(query, platform, num_results)
        
        # 获取格式化的引用
        references = reference_formatter.format_references(search_results, citation_style)
        
        # 获取独立的引用列表
        reference_list = reference_formatter.extract_references(search_results)
        
        return JSONResponse(content={
            "formatted_references": references,
            "reference_count": len(reference_list),
            "references": reference_list
        })
        
    except Exception as e:
        logging.error(f"获取参考文献时出错: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/health")
async def health():
    try:
        cfg = Config()
        ok_llm = bool(cfg.llm_api_key and cfg.llm_api_key.strip())
        ok_serper = bool(cfg.serper_api_key and cfg.serper_api_key.strip())
        missing = []
        if not ok_llm:
            missing.append("DEEPSEEK_API_KEY")
        if not ok_serper:
            missing.append("SERPER_API_KEY")
        return JSONResponse(content={
            "ok": True,
            "llm": ok_llm,
            "serper": ok_serper,
            "missing": missing,
            "platforms": {
                "arxiv": True,
                "google_scholar": ok_serper,
                "web": ok_serper,
                "news": ok_serper,
                "patent": ok_serper
            }
        })
    except Exception as e:
        logging.error(f"健康检查失败: {e}")
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)

# --- Uvicorn 运行说明 ---
# 要运行此 FastAPI 应用:
# 1. 在您的终端 (激活了 Python 虚拟环境 .venv 的情况下（.venv\Scripts\activate.ps1)，导航到项目根目录。
# 2. 执行命令:
#    cd src
#    uvicorn app:app --reload --port 5001
# 或者使用 Python 模块运行:
#    cd src
#    python -m uvicorn app:app --reload --port 5001
