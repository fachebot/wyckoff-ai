"""
Wyckoff-AI Web API 主入口

FastAPI 应用程序，提供 REST API 和 Web 界面
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from wyckoff_ai.api.routes import analysis, backtest
from wyckoff_ai.logging import get_logger, setup_logging

# 初始化日志
setup_logging(level="INFO")
logger = get_logger("api.main")

# 创建 FastAPI 应用
app = FastAPI(
    title="Wyckoff-AI",
    description="威科夫 K 线分析 Web 服务 - 提供威科夫事件检测、状态机分析和策略回测",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(analysis.router)
app.include_router(backtest.router)

# 静态文件目录
STATIC_DIR = Path(__file__).parent.parent / "web" / "static"
WEB_DIR = Path(__file__).parent.parent / "web"

# 挂载静态文件
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页"""
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
    <html>
        <head><title>Wyckoff-AI</title></head>
        <body>
            <h1>Wyckoff-AI Web Service</h1>
            <p>API 文档: <a href="/api/docs">/api/docs</a></p>
        </body>
    </html>
    """)


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "wyckoff-ai"}


@app.on_event("startup")
async def startup_event():
    logger.info("Wyckoff-AI Web 服务启动")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Wyckoff-AI Web 服务关闭")


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """启动服务器"""
    import uvicorn
    uvicorn.run(
        "wyckoff_ai.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_server(reload=True)

