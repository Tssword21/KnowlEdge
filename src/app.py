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

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tempfile
from starlette.middleware.sessions import SessionMiddleware
import bcrypt

# --- 应用配置和初始化 ---
setup_logging()
app = FastAPI(title="KnowlEdge 智能引擎", version="1.0.0")

# 会话中间件（用于登录态）
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "change-me"))

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

def _is_admin(request: Request) -> bool:
    """简单管理员判定：优先会话 is_admin，其次用户名在 ADMIN_USERS 或 user_auth 表。"""
    try:
        user = request.session.get("user")
        if user and isinstance(user, dict) and user.get("is_admin"):
            return True
        if not (user and isinstance(user, dict) and user.get("username")):
            return False
        admins_env = os.getenv("ADMIN_USERS", "admin")
        admin_list = [u.strip() for u in admins_env.split(',') if u.strip()]
        if user["username"] in admin_list:
            return True
        # 数据库兜底
        try:
            from src.db_utils import get_db_connection
        except Exception:
            from db_utils import get_db_connection
        conn = get_db_connection()
        row = conn.execute("SELECT is_admin FROM user_auth WHERE user_id=?", (user.get("user_id"),)).fetchone()
        return bool(row and (row[0] == 1 or row[0] == '1'))
    except Exception:
        return False

# --- FastAPI 路径操作 ---

@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    """提供主 HTML 页面。"""
    user = request.session.get("user")
    return templates.TemplateResponse("index.html", {"request": request, "current_user": user})

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
        # 会话用户覆盖用户名/邮箱（若已登录）
        try:
            sess_user = request.session.get("user")
            if sess_user and isinstance(sess_user, dict):
                if sess_user.get("username"):
                    user_name = sess_user["username"]
                if not email and sess_user.get("email"):
                    email = sess_user["email"]
        except Exception:
            pass
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

# --- 认证相关接口 ---

@app.get("/auth", response_class=HTMLResponse)
async def get_auth_page(request: Request):
    return templates.TemplateResponse("auth.html", {"request": request})

@app.post("/auth/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...), email: str = Form(None), occupation: str = Form(None), admin_code: str = Form(None)):
    try:
        from src.db_utils import get_db_connection
    except Exception:
        from db_utils import get_db_connection
    conn = get_db_connection()
    try:
        # 检查是否存在
        exists = conn.execute("SELECT username FROM user_auth WHERE username=?", (username,)).fetchone()
        if exists:
            return JSONResponse(content={"error": "用户名已存在", "code": "USERNAME_TAKEN"}, status_code=400)
        # 创建基础 user 记录
        user_id = username.lower()
        conn.execute("INSERT OR IGNORE INTO users (id, name, occupation, email) VALUES (?, ?, ?, ?)", (user_id, username, occupation or "", email or ""))
        # 判断管理员（邀请码优先，其次 ADMIN_USERS 列表）
        admins_env = os.getenv("ADMIN_USERS", "admin")
        invite = os.getenv("ADMIN_SETUP_CODE", "123456")
        is_admin = 1 if (admin_code and invite and admin_code == invite) else 0
        if not is_admin:
            if username in [u.strip() for u in admins_env.split(',') if u.strip()]:
                is_admin = 1
        # 写入 auth
        pwd_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn.execute("INSERT INTO user_auth (user_id, username, email, password_hash, is_admin) VALUES (?, ?, ?, ?, ?)", (user_id, username, email or "", pwd_hash, is_admin))
        conn.commit()
        request.session["user"] = {"user_id": user_id, "username": username, "email": email or "", "is_admin": bool(is_admin)}
        return JSONResponse(content={"ok": True, "is_admin": bool(is_admin)})
    except Exception as e:
        logging.error(f"注册失败: {e}")
        return JSONResponse(content={"error": "注册失败", "code": "REGISTER_FAILED"}, status_code=500)
    finally:
        try:
            conn.close()
        except Exception:
            pass

@app.post("/auth/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    try:
        from src.db_utils import get_db_connection
    except Exception:
        from db_utils import get_db_connection
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT user_id, password_hash, email, is_admin FROM user_auth WHERE username=?", (username,)).fetchone()
        if not row:
            return JSONResponse(content={"error": "用户不存在", "code": "USER_NOT_FOUND"}, status_code=400)
        ok = bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8"))
        if not ok:
            return JSONResponse(content={"error": "密码错误", "code": "INVALID_CREDENTIALS"}, status_code=400)
        request.session["user"] = {"user_id": row["user_id"], "username": username, "email": row["email"] or "", "is_admin": bool(row["is_admin"])}
        return JSONResponse(content={"ok": True, "is_admin": bool(row["is_admin"])})
    except Exception as e:
        logging.error(f"登录失败: {e}")
        return JSONResponse(content={"error": "登录失败", "code": "LOGIN_FAILED"}, status_code=500)
    finally:
        try:
            conn.close()
        except Exception:
            pass

@app.post("/auth/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return JSONResponse(content={"ok": True})

@app.get("/admin", response_class=HTMLResponse)
async def get_admin_page(request: Request):
    """管理员界面。"""
    if not _is_admin(request):
        return RedirectResponse(url="/auth?next=/admin", status_code=302)
    user = request.session.get("user")
    return templates.TemplateResponse("admin.html", {"request": request, "current_user": user})

@app.get("/api/admin/users")
async def admin_list_users(request: Request, q: Optional[str] = None, limit: int = 50, offset: int = 0):
    if not _is_admin(request):
        return JSONResponse(content={"error": "FORBIDDEN"}, status_code=403)
    try:
        try:
            from src.db_utils import get_db_connection
        except Exception:
            from db_utils import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        where = ""
        params: List[Any] = []
        if q and q.strip():
            where = "WHERE (u.id LIKE ? OR u.name LIKE ? OR u.email LIKE ?)"
            like = f"%{q.strip()}%"
            params.extend([like, like, like])
        sql = f"""
            SELECT
              u.id, u.name, u.occupation, u.email, u.created_at,
              (SELECT COUNT(1) FROM user_interests ui WHERE ui.user_id = u.id) AS interests_count,
              (SELECT COUNT(1) FROM user_skills us WHERE us.user_id = u.id) AS skills_count,
              (SELECT COUNT(1) FROM search_history sh WHERE sh.user_id = u.id) AS searches_count,
              (SELECT is_admin FROM user_auth ua WHERE ua.user_id = u.id LIMIT 1) AS is_admin
            FROM users u
            {where}
            ORDER BY u.created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([max(1, min(limit, 200)), max(0, offset)])
        rows = cursor.execute(sql, params).fetchall()
        users = []
        for r in rows:
            users.append({
                "id": r[0],
                "name": r[1],
                "occupation": r[2],
                "email": r[3],
                "created_at": r[4],
                "is_admin": bool(r[8]) if len(r) > 8 else False,
                "counts": {
                    "interests": r[5],
                    "skills": r[6],
                    "searches": r[7]
                }
            })
        return JSONResponse(content={"items": users, "limit": limit, "offset": offset, "q": q or ""})
    except Exception as e:
        logging.error(f"admin_list_users 失败: {e}", exc_info=True)
        return JSONResponse(content={"error": "INTERNAL_ERROR"}, status_code=500)
    finally:
        try:
            conn.close()
        except Exception:
            pass

@app.get("/api/admin/users/{user_id}")
async def admin_user_detail(request: Request, user_id: str):
    if not _is_admin(request):
        return JSONResponse(content={"error": "FORBIDDEN"}, status_code=403)
    try:
        try:
            from src.db_utils import get_db_connection
        except Exception:
            from db_utils import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        # 基本信息
        user_row = cursor.execute("SELECT id, name, occupation, email, created_at, updated_at FROM users WHERE id=?", (user_id,)).fetchone()
        if not user_row:
            return JSONResponse(content={"error": "NOT_FOUND"}, status_code=404)
        # is_admin
        admin_row = cursor.execute("SELECT is_admin FROM user_auth WHERE user_id=?", (user_id,)).fetchone()
        # 画像概要（如存在）
        profile_row = cursor.execute(
            "SELECT username, profile_data, created_at, updated_at FROM user_profiles WHERE user_id=?",
            (user_id,)
        ).fetchone()
        # 兴趣（按权重）
        interests = cursor.execute(
            """
            SELECT topic, category, weight, reason, last_updated
            FROM user_interests
            WHERE user_id=?
            ORDER BY weight DESC, last_updated DESC
            LIMIT 200
            """,
            (user_id,)
        ).fetchall()
        # 技能
        skills = cursor.execute(
            """
            SELECT skill, level, category, timestamp
            FROM user_skills
            WHERE user_id=?
            ORDER BY timestamp DESC
            LIMIT 200
            """,
            (user_id,)
        ).fetchall()
        # 教育与工作
        education = cursor.execute(
            "SELECT institution, major, degree, time_period, timestamp FROM user_education WHERE user_id=? ORDER BY timestamp DESC",
            (user_id,)
        ).fetchall()
        work = cursor.execute(
            "SELECT company, position, time_period, description, timestamp FROM user_work_experience WHERE user_id=? ORDER BY timestamp DESC",
            (user_id,)
        ).fetchall()
        # 最近搜索
        searches = cursor.execute(
            "SELECT query, platform, timestamp FROM search_history WHERE user_id=? ORDER BY timestamp DESC LIMIT 50",
            (user_id,)
        ).fetchall()
        # 交互统计
        interactions = cursor.execute(
            "SELECT COUNT(1) FROM user_interactions WHERE user_id=?",
            (user_id,)
        ).fetchone()
        def row_to_dict(row):
            try:
                return {k: row[k] for k in row.keys()}
            except Exception:
                return dict(row) if isinstance(row, dict) else {}
        user_dict = row_to_dict(user_row)
        user_dict["is_admin"] = bool(admin_row[0]) if admin_row else False
        profile_data = None
        if profile_row:
            try:
                # 尝试解析 JSON
                import json as _json
                profile_data = {
                    "username": profile_row[0],
                    "profile": _json.loads(profile_row[1]) if profile_row[1] else {},
                    "created_at": profile_row[2],
                    "updated_at": profile_row[3]
                }
            except Exception:
                profile_data = {
                    "username": profile_row[0],
                    "profile_raw": profile_row[1],
                    "created_at": profile_row[2],
                    "updated_at": profile_row[3]
                }
        resp = {
            "user": user_dict,
            "profile": profile_data,
            "interests": [row_to_dict(r) for r in interests],
            "skills": [row_to_dict(r) for r in skills],
            "education": [row_to_dict(r) for r in education],
            "work": [row_to_dict(r) for r in work],
            "recent_searches": [row_to_dict(r) for r in searches],
            "interactions_count": interactions[0] if interactions else 0
        }
        return JSONResponse(content=resp)
    except Exception as e:
        logging.error(f"admin_user_detail 失败: {e}", exc_info=True)
        return JSONResponse(content={"error": "INTERNAL_ERROR"}, status_code=500)
    finally:
        try:
            conn.close()
        except Exception:
            pass

@app.post("/api/admin/users/{user_id}/role")
async def admin_set_role(request: Request, user_id: str):
    if not _is_admin(request):
        return JSONResponse(content={"error": "FORBIDDEN"}, status_code=403)
    try:
        body = await request.json()
        is_admin = 1 if (body.get("is_admin") in (True, 1, "1", "true", "True")) else 0
        try:
            from src.db_utils import get_db_connection
        except Exception:
            from db_utils import get_db_connection
        conn = get_db_connection()
        conn.execute("UPDATE user_auth SET is_admin=? WHERE user_id=?", (is_admin, user_id))
        conn.commit()
        return JSONResponse(content={"ok": True, "is_admin": bool(is_admin)})
    except Exception as e:
        logging.error(f"admin_set_role 失败: {e}", exc_info=True)
        return JSONResponse(content={"error": "INTERNAL_ERROR"}, status_code=500)
    finally:
        try:
            conn.close()
        except Exception:
            pass

# --- Uvicorn 运行说明 ---
# 要运行此 FastAPI 应用:
# 1. 在您的终端 (激活了 Python 虚拟环境 .venv 的情况下（.venv\Scripts\activate.ps1)，导航到项目根目录。
# 2. 执行命令:
#    cd src
#    uvicorn app:app --reload --port 5001
# 或者使用 Python 模块运行:
#    cd src
#    python -m uvicorn app:app --reload --port 5001
