from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from wyckoff_ai.schemas import WyckoffAnalysis

if TYPE_CHECKING:
    from wyckoff_ai.mtf import MTFAnalysisResult
    from wyckoff_ai.wyckoff.sequence import SequenceAnalysis


_BJT = timezone(timedelta(hours=8))


def _to_bjt_str(iso_ts: str) -> str:
    """
    把 ISO 时间字符串转换为北京时间（UTC+8）用于报告展示。
    """
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(_BJT)
        return dt.isoformat(timespec="seconds").replace("T", " ")
    except Exception:
        return iso_ts


def render_markdown(analysis: WyckoffAnalysis) -> str:
    rng = analysis.range
    lines: list[str] = []
    lines.append(
        f"# 威科夫结构报告（{analysis.symbol} / {analysis.timeframe} / {analysis.exchange}）")
    lines.append("")
    lines.append(f"- **截至时间（北京时间）**：{_to_bjt_str(analysis.asof_ts)}")
    lines.append(f"- **市场结构**：{analysis.market_structure}")
    if analysis.regime_hint:
        lines.append(
            f"- **状态提示**：{analysis.regime_hint}（{analysis.regime_method or 'n/a'}）")
    if rng.duration_bars > 0 and rng.low is not None and rng.high is not None:
        lines.append(
            f"- **最新区间**：[{rng.low:.2f}, {rng.high:.2f}]  中轴 {rng.mid:.2f}  持续 {rng.duration_bars} 根"
        )
    else:
        lines.append("- **最新区间**：未检测到稳定横盘区间（或持续不足）")

    lines.append("")
    lines.append("## 事件（规则识别）")
    if not analysis.events:
        lines.append("- 暂无高置信度事件。")
    else:
        for e in analysis.events[-12:]:
            ev = "；".join(e.evidence[:5])
            lines.append(
                f"- **{e.type}** | {_to_bjt_str(e.ts)} | 价格 {e.price:.2f} | 置信度 {e.confidence:.2f}\n  - 证据：{ev}"
            )

        # If user wants "range 段里每一次 UT/SPRING"，单独把这两类全量列出（避免被 -12 截断）
        ut_spring = [e for e in analysis.events if e.type in (
            "UT", "UTAD", "SPRING")]
        if ut_spring:
            lines.append("")
            lines.append("## 区间内 UT/SPRING 全量列表")
            for e in ut_spring:
                lines.append(
                    f"- **{e.type}** | {_to_bjt_str(e.ts)} | 价格 {e.price:.2f} | 置信度 {e.confidence:.2f}")

    if analysis.event_forward_stats:
        lines.append("")
        lines.append("## 历史事件 -> 后验表现（同一段K线内统计）")
        lines.append(
            "- 说明：对本次拉取的K线窗口内，同类事件发生后未来 12/24/48 根的收益做统计；仅作条件参考，不构成预测。")
        # Only show stats for event types that appeared recently to keep report compact
        recent_types = []
        for e in analysis.events[-12:]:
            if e.type not in recent_types:
                recent_types.append(e.type)
        for t in recent_types[:8]:
            s = analysis.event_forward_stats.get(t) or {}
            if not s:
                continue
            n = int(s.get("n", 0))
            r12 = s.get("r12_med")
            r24 = s.get("r24_med")
            r48 = s.get("r48_med")
            w12 = s.get("win12")
            parts = [f"n={n}"]
            if isinstance(r12, (int, float)):
                parts.append(f"12根中位={r12*100:.2f}%")
            if isinstance(r24, (int, float)):
                parts.append(f"24根中位={r24*100:.2f}%")
            if isinstance(r48, (int, float)):
                parts.append(f"48根中位={r48*100:.2f}%")
            if isinstance(w12, (int, float)):
                parts.append(f"12根胜率={w12*100:.0f}%")
            lines.append(f"- **{t}**：{'，'.join(parts)}")

    lines.append("")
    lines.append("## 关键价位")
    if analysis.levels.support:
        lines.append(
            f"- **支撑**：{', '.join(f'{x:.2f}' for x in analysis.levels.support[:6])}")
    if analysis.levels.resistance:
        lines.append(
            f"- **阻力**：{', '.join(f'{x:.2f}' for x in analysis.levels.resistance[:6])}")
    if analysis.levels.pivot:
        lines.append(
            f"- **Pivot**：{', '.join(f'{x:.2f}' for x in analysis.levels.pivot[:12])}")
    if not (analysis.levels.support or analysis.levels.resistance or analysis.levels.pivot):
        lines.append("- 暂无。")

    lines.append("")
    lines.append("## 剧本（确认 / 失效）")
    for s in analysis.scenarios:
        lines.append(f"- **偏向**：{s.bias}")
        lines.append(f"  - **确认**：{s.confirmation}")
        lines.append(f"  - **失效**：{s.invalidation}")
        if s.notes:
            lines.append(f"  - **备注**：{s.notes}")
    
    # 概率化剧本
    if analysis.probabilistic_scenarios:
        lines.append("")
        lines.append("## 概率化交易剧本")
        lines.append("")
        lines.append("> 基于状态机、事件序列、量价分析等多维度信息计算的概率化剧本，提供实用的交易指导。")
        lines.append("")
        
        for idx, ps in enumerate(analysis.probabilistic_scenarios[:5], 1):  # 只显示前5个
            prob_bar = "=" * int(ps.probability * 20) + "-" * (20 - int(ps.probability * 20))
            lines.append(f"### 剧本 {idx}: {ps.name}")
            lines.append("")
            lines.append(f"- **概率**：[{prob_bar}] {ps.probability*100:.0f}%")
            lines.append(f"- **置信度**：{ps.confidence*100:.0f}%")
            lines.append(f"- **偏向**：{_bias_emoji(ps.bias)} {ps.bias}")
            lines.append(f"- **风险等级**：{_risk_emoji(ps.risk_level)} {ps.risk_level}")
            lines.append(f"- **描述**：{ps.description}")
            
            # 概率分解
            if ps.probability_breakdown:
                lines.append("")
                lines.append("#### 概率来源")
                for source, value in ps.probability_breakdown.items():
                    source_name = {
                        "state_machine": "状态机",
                        "sequence": "事件序列",
                        "history": "历史统计",
                        "structure": "市场结构",
                        "breakout_signal": "突破信号",
                        "duration": "持续时间",
                        "no_breakout": "无突破信号",
                    }.get(source, source)
                    lines.append(f"  - {source_name}: {value*100:.0f}%")
            
            # 交易信号
            if ps.signal:
                lines.append("")
                lines.append("#### 交易信号")
                
                if ps.signal.entry_price:
                    lines.append(f"- **入场价**：{ps.signal.entry_price:.2f}")
                if ps.signal.entry_condition:
                    lines.append(f"- **入场条件**：{ps.signal.entry_condition}")
                
                if ps.signal.stop_loss:
                    lines.append(f"- **止损位**：{ps.signal.stop_loss:.2f}")
                
                if ps.signal.targets:
                    lines.append("- **目标位**：")
                    for i, target in enumerate(ps.signal.targets, 1):
                        lines.append(f"  - 目标{i}: {target:.2f}")
                
                if ps.signal.risk_reward_ratio > 0:
                    lines.append(f"- **风险收益比**：1:{ps.signal.risk_reward_ratio:.2f}")
                
                if ps.signal.position_size_pct > 0:
                    lines.append(f"- **建议仓位**：{ps.signal.position_size_pct:.0f}%")
                
                if ps.signal.time_horizon:
                    lines.append(f"- **时间窗口**：{ps.signal.time_horizon}")
                
                if ps.signal.confirmation_signals:
                    lines.append("")
                    lines.append("- **确认信号**：")
                    for sig in ps.signal.confirmation_signals:
                        lines.append(f"  - ✓ {sig}")
                
                if ps.signal.invalidation_signals:
                    lines.append("")
                    lines.append("- **失效条件**：")
                    for sig in ps.signal.invalidation_signals:
                        lines.append(f"  - ✗ {sig}")
            
            # 关键事件
            if ps.key_events:
                lines.append("")
                lines.append(f"- **关键事件**：{', '.join(ps.key_events)}")
            
            # 证据
            if ps.evidence:
                lines.append("")
                lines.append("- **证据**：")
                for ev in ps.evidence:
                    lines.append(f"  - {ev}")
            
            # 风险因素
            if ps.risk_factors:
                lines.append("")
                lines.append("- **风险因素**：")
                for rf in ps.risk_factors:
                    lines.append(f"  - [!] {rf}")
            
            lines.append("")

    # 状态机分析
    if analysis.state_machine:
        sm = analysis.state_machine
        lines.append("")
        lines.append("## 状态机分析")
        lines.append("")
        lines.append(f"- **当前状态**：{sm.state_description}")
        lines.append(f"- **状态置信度**：{sm.state_confidence*100:.0f}%")
        lines.append(f"- **阶段进度**：{_progress_bar(sm.phase_progress)} {sm.phase_progress*100:.0f}%")
        lines.append(f"- **偏向判断**：{_bias_emoji(sm.bias)} {sm.bias} ({sm.bias_confidence*100:.0f}%)")
        lines.append(f"- **在阶段停留**：{sm.time_in_phase_bars} 根K线")
        
        if sm.events_in_phase:
            lines.append(f"- **阶段内事件**：{', '.join(sm.events_in_phase)}")
        if sm.missing_events:
            lines.append(f"- **缺失事件**：{', '.join(sm.missing_events)}")
        if sm.next_expected_events:
            lines.append(f"- **下一期望事件**：{', '.join(sm.next_expected_events)}")
        
        # 转换历史
        if sm.recent_transitions:
            lines.append("")
            lines.append("### 状态转换历史")
            for t in sm.recent_transitions[-3:]:
                lines.append(f"- {t['from']} → {t['to']} | 触发: {t['trigger']} | {t['reason']}")
        
        # 预测
        if sm.next_probable_states:
            lines.append("")
            lines.append("### 下一状态预测")
            for p in sm.next_probable_states[:3]:
                lines.append(f"- {p['state']}: {p['probability']*100:.0f}%")
        
        if sm.predicted_events:
            lines.append("")
            lines.append("### 事件预测")
            for p in sm.predicted_events[:3]:
                lines.append(f"- {p['type']}: {p['probability']*100:.0f}% - {p['description']}")
        
        # 交易建议
        if sm.action_suggestion:
            lines.append("")
            lines.append("### 交易建议")
            lines.append(f"- {sm.action_suggestion}")
        
        # 状态机风险
        lines.append("")
        lines.append(f"### 状态机风险评估")
        lines.append(f"- **风险等级**：{_risk_emoji(sm.risk_level)} {sm.risk_level}")
        if sm.risk_factors:
            for f in sm.risk_factors:
                lines.append(f"  - [!] {f}")
        
        if sm.phase_notes:
            lines.append("")
            lines.append("### 阶段备注")
            for note in sm.phase_notes:
                lines.append(f"- {note}")

    lines.append("")
    lines.append("## 风险提示")
    if analysis.risk_notes:
        for r in analysis.risk_notes:
            lines.append(f"- {r}")
    else:
        lines.append("- 暂无额外风险提示。")

    lines.append("")
    lines.append("> 说明：本工具优先使用规则引擎稳定输出；若配置 LLM，将在不改变结构化 JSON 的前提下强化中文解释。")
    return "\n".join(lines)


def render_sequence_section(sequence: "SequenceAnalysis") -> list[str]:
    """渲染序列分析部分"""
    lines = []
    lines.append("## 序列分析")
    lines.append("")
    lines.append(f"- **主要偏向**：{sequence.primary_bias}")
    lines.append(f"- **当前阶段**：{sequence.current_stage}")
    lines.append(f"- **阶段描述**：{sequence.stage_description}")
    
    if sequence.accumulation_match:
        acc = sequence.accumulation_match
        lines.append(f"- **吸筹匹配度**：{acc.completeness*100:.0f}% (置信度 {acc.confidence*100:.0f}%)")
        if acc.matched_events:
            lines.append(f"  - 已匹配事件：{', '.join(acc.matched_events)}")
    
    if sequence.distribution_match:
        dist = sequence.distribution_match
        lines.append(f"- **派发匹配度**：{dist.completeness*100:.0f}% (置信度 {dist.confidence*100:.0f}%)")
        if dist.matched_events:
            lines.append(f"  - 已匹配事件：{', '.join(dist.matched_events)}")
    
    if sequence.next_expected_events:
        lines.append(f"- **下一预期事件**：{', '.join(sequence.next_expected_events)}")
    
    if sequence.sequence_notes:
        lines.append("")
        lines.append("### 序列备注")
        for note in sequence.sequence_notes:
            lines.append(f"- {note}")
    
    return lines


def render_mtf_markdown(result: "MTFAnalysisResult") -> str:
    """
    渲染多时间框架分析报告
    """
    lines: list[str] = []
    
    # 标题
    lines.append(f"# 多时间框架威科夫分析（{result.symbol} / {result.exchange}）")
    lines.append("")
    
    # 概览
    lines.append("## 综合结论")
    lines.append("")
    lines.append(f"- **分析时间框架**：{', '.join(result.timeframes)}")
    lines.append(f"- **主导方向**：{_bias_emoji(result.overall_bias)} {result.overall_bias}")
    lines.append(f"- **整体置信度**：{result.overall_confidence*100:.0f}%")
    lines.append(f"- **结构阶段**：{result.structure_phase}")
    lines.append(f"- **风险等级**：{_risk_emoji(result.risk_level)} {result.risk_level}")
    lines.append("")
    
    # 共振分析
    lines.append("## 级别共振")
    lines.append("")
    res = result.resonance
    alignment_bar = "=" * int(res.alignment_score * 10) + "-" * (10 - int(res.alignment_score * 10))
    lines.append(f"- **共振得分**：[{alignment_bar}] {res.alignment_score*100:.0f}%")
    lines.append(f"- **是否一致**：{'[OK] 是' if res.aligned else '[!] 否'}")
    
    if res.conflicts:
        lines.append("- **冲突**：")
        for conflict in res.conflicts:
            lines.append(f"  - [!] {conflict}")
    
    if res.notes:
        lines.append("- **共振备注**：")
        for note in res.notes:
            lines.append(f"  - {note}")
    lines.append("")
    
    # 各时间框架分析
    lines.append("## 各时间框架分析")
    lines.append("")
    
    for tfa in result.tf_analyses:
        lines.append(f"### {tfa.timeframe_name} ({tfa.timeframe})")
        lines.append("")
        lines.append(f"- **偏向**：{_bias_emoji(tfa.bias)} {tfa.bias} ({tfa.bias_strength*100:.0f}%)")
        lines.append(f"- **市场结构**：{tfa.analysis.market_structure}")
        
        # 区间信息
        rng = tfa.analysis.range
        if rng.duration_bars > 0 and rng.low is not None:
            lines.append(f"- **区间**：[{rng.low:.2f}, {rng.high:.2f}] 中轴 {rng.mid:.2f}")
        
        # 序列信息
        if tfa.sequence.current_stage:
            lines.append(f"- **序列阶段**：{tfa.sequence.current_stage}")
        
        # 关键事件
        if tfa.key_events:
            lines.append("- **关键事件**：")
            for e in tfa.key_events[-3:]:
                lines.append(f"  - {e.type} @ {_to_bjt_str(e.ts)} | {e.price:.2f} | {e.confidence*100:.0f}%")
        
        lines.append("")
    
    # 交易计划
    lines.append("## 交易计划")
    lines.append("")
    lines.append(f"- **入场时间框架**：{result.entry_timeframe}")
    if result.entry_events:
        lines.append(f"- **关注入场信号**：{', '.join(result.entry_events[:4])}")
    if result.stop_reference:
        lines.append(f"- **止损参考**：{result.stop_reference:.2f}")
    if result.target_reference:
        lines.append(f"- **目标参考**：{result.target_reference:.2f}")
    lines.append("")
    
    # 行动计划
    if result.action_plan:
        lines.append("### 行动建议")
        for i, action in enumerate(result.action_plan, 1):
            lines.append(f"{i}. {action}")
        lines.append("")
    
    # 风险评估
    lines.append("## 风险评估")
    lines.append("")
    lines.append(f"- **风险等级**：{_risk_emoji(result.risk_level)} {result.risk_level}")
    if result.risk_factors:
        lines.append("- **风险因素**：")
        for factor in result.risk_factors:
            lines.append(f"  - [!] {factor}")
    else:
        lines.append("- 暂无明显风险因素")
    lines.append("")
    
    # 总结
    lines.append("## 总结")
    lines.append("")
    lines.append(f"> {result.summary}")
    lines.append("")
    lines.append("> 说明：多时间框架分析遵循大级别定方向、小级别找入场原则。级别共振度越高，信号可靠性越强。")
    
    return "\n".join(lines)


def _bias_emoji(bias: str) -> str:
    """获取偏向对应的标记"""
    return {"bullish": "[+]", "bearish": "[-]", "neutral": "[o]"}.get(bias, "[o]")


def _risk_emoji(risk: str) -> str:
    """获取风险等级对应的标记"""
    return {"low": "[L]", "medium": "[M]", "high": "[H]", "extreme": "[X]"}.get(risk, "[?]")


def _progress_bar(progress: float, length: int = 10) -> str:
    """生成进度条"""
    filled = int(progress * length)
    return "[" + "=" * filled + "-" * (length - filled) + "]"
