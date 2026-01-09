from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.runnables import RunnableLambda, RunnableSequence
from pydantic import BaseModel

from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
from wyckoff_ai.features import compute_features
from wyckoff_ai.regime import RegimeConfig, add_regime_columns
from wyckoff_ai.report.render import render_markdown
from wyckoff_ai.schemas import WyckoffAnalysis
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff


class PipelineOutput(BaseModel):
    analysis: WyckoffAnalysis
    report_md: str
    ohlcv_rows: int
    gaps: int


def _maybe_llm_enhance_report(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Optional LLM step:
    - If OPENAI_API_KEY is present, ask the model to rewrite report_md in Chinese,
      but keep structure stable (do not change analysis JSON).
    - If not present, no-op.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return payload

    try:
        # Delayed import to keep "no-key" path lightweight
        from langchain_openai import ChatOpenAI
    except Exception:
        return payload

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    analysis: WyckoffAnalysis = payload["analysis"]

    prompt = (
        "你是一位威科夫/结构分析研究员。"
        "请基于给定的结构化 JSON 结论，生成一份更专业但简洁的中文 Markdown 报告。"
        "要求：\n"
        "1) 必须遵循“结构定位→证据→关键位→剧本→风险”的顺序\n"
        "2) 不要编造不存在的事件/价位；只能解释 JSON 里已有内容\n"
        "3) 报告中给出关键价位时，直接引用 JSON levels/range 中的数值\n"
        "4) 语气像研究员，不要夸张\n"
        "5) 报告中所有时间请统一换算为北京时间（UTC+8）再展示\n"
        "\n"
        "结构化JSON如下：\n"
        f"{json.dumps(analysis.model_dump(), ensure_ascii=False)}\n"
    )

    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None)
        if isinstance(text, str) and text.strip():
            payload["report_md"] = text.strip()
    except Exception:
        # LLM failure should not break pipeline
        pass

    return payload


def build_pipeline() -> RunnableSequence:
    """
    LangChain LCEL pipeline:
      input -> fetch -> features -> wyckoff -> render -> (optional llm) -> output
    """

    def step_fetch(inp: dict[str, Any]) -> dict[str, Any]:
        symbol = inp["symbol"]
        timeframe = inp["timeframe"]
        limit = int(inp.get("limit", 720))
        fr = fetch_ohlcv_binance_spot(
            symbol=symbol, timeframe=timeframe, limit=limit)
        return {**inp, "ohlcv": fr.df, "gaps": fr.gaps}

    def step_features(inp: dict[str, Any]) -> dict[str, Any]:
        feats = compute_features(inp["ohlcv"])
        # Allow CLI to override regime method/k; compute_features() already adds default kmeans,
        # this re-applies only if user asks for cusum/none or a different k.
        method = str(inp.get("regime", "kmeans"))
        k = int(inp.get("regime_k", 4))
        if method != "kmeans" or k != 4:
            feats = add_regime_columns(feats, RegimeConfig(method=method, k=k))
        return {**inp, "features": feats}

    def step_detect(inp: dict[str, Any]) -> dict[str, Any]:
        cfg = DetectionConfig(
            strict=bool(inp.get("strict", False)),
            lookback_bars=int(inp.get("lookback_bars", 220)),
        )
        analysis = detect_wyckoff(
            inp["features"],
            symbol=inp["symbol"],
            exchange=inp.get("exchange", "binance"),
            timeframe=inp["timeframe"],
            cfg=cfg,
        )
        return {**inp, "analysis": analysis}

    def step_render(inp: dict[str, Any]) -> dict[str, Any]:
        report_md = render_markdown(inp["analysis"])
        return {**inp, "report_md": report_md}

    def step_finalize(inp: dict[str, Any]) -> PipelineOutput:
        return PipelineOutput(
            analysis=inp["analysis"],
            report_md=inp["report_md"],
            ohlcv_rows=int(len(inp["ohlcv"])),
            gaps=int(inp.get("gaps", 0)),
        )

    return RunnableSequence(
        RunnableLambda(step_fetch),
        RunnableLambda(step_features),
        RunnableLambda(step_detect),
        RunnableLambda(step_render),
        RunnableLambda(_maybe_llm_enhance_report),
        RunnableLambda(step_finalize),
    )
