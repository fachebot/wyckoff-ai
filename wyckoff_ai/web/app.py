"""
威科夫分析 Web 应用主入口

FastAPI 应用，提供 REST API 和前端页面
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from wyckoff_ai.logging import get_logger, setup_logging
from wyckoff_ai.web.api import router as api_router

logger = get_logger("web.app")

# 路径配置
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# 创建 FastAPI 应用
app = FastAPI(
    title="威科夫分析系统",
    description="基于威科夫方法的加密货币技术分析系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 模板引擎
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 注册 API 路由
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chart/{symbol}", response_class=HTMLResponse)
async def chart_page(request: Request, symbol: str):
    """图表页面"""
    symbol = symbol.replace("-", "/").upper()
    return templates.TemplateResponse(
        "chart.html",
        {"request": request, "symbol": symbol}
    )


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    setup_logging(level="INFO")
    logger.info("威科夫分析 Web 服务启动")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("威科夫分析 Web 服务关闭")


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """运行服务器"""
    import uvicorn
    uvicorn.run(
        "wyckoff_ai.web.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server(reload=True)
