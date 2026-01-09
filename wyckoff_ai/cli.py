from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from wyckoff_ai.chains.pipeline import build_pipeline
from wyckoff_ai.config import WyckoffConfig, get_config, load_config, set_config
from wyckoff_ai.data.binance import fetch_ohlcv_binance_spot
from wyckoff_ai.exceptions import WyckoffError
from wyckoff_ai.features import compute_features
from wyckoff_ai.logging import LogLevel, get_logger, setup_logging
from wyckoff_ai.mtf import MTF_PRESETS, analyze_mtf
from wyckoff_ai.report.render import render_mtf_markdown
from wyckoff_ai.wyckoff.rules import DetectionConfig, detect_wyckoff
from wyckoff_ai.wyckoff.state_machine import WyckoffStateMachine

logger = get_logger("cli")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False,
                    indent=2), encoding="utf-8")


def cmd_analyze(args: argparse.Namespace) -> int:
    config = get_config()
    console = Console()
    
    # 合并配置：CLI 参数 > 配置文件
    symbol = args.symbol or config.analysis.default_symbol
    timeframe = args.timeframe or config.analysis.default_timeframe
    limit = args.limit or config.analysis.default_limit
    out_dir = Path(args.out or config.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始分析 {symbol} {timeframe}")

    try:
        chain = build_pipeline()
        
        # 使用配置值，CLI 参数优先
        exchange = args.exchange or config.exchange.name
        lookback = args.lookback_bars if args.lookback_bars is not None else (config.analysis.lookback_bars or limit)
        strict = args.strict if args.strict else config.analysis.strict_mode
        regime = args.regime or config.analysis.regime_method
        regime_k = args.regime_k or config.analysis.regime_k
        
        payload = {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "lookback_bars": lookback,
            "strict": strict,
            "regime": regime,
            "regime_k": regime_k,
        }

        console.print(
            f"[bold]分析[/bold] exchange={exchange} symbol={symbol} timeframe={timeframe} limit={limit}"
        )
        result = chain.invoke(payload)

        # Save artifacts
        result.analysis.model_dump()  # ensure pydantic object is realized
        _write_json(out_dir / "analysis.json", result.analysis.model_dump())
        _write_text(out_dir / "report.md", result.report_md)

        # Optional exports
        export_csv = args.export_csv or config.output.export_csv
        if export_csv:
            fr = fetch_ohlcv_binance_spot(
                symbol=symbol, timeframe=timeframe, limit=int(limit)
            )
            df_feat = compute_features(fr.df)
            fr.df.to_csv(out_dir / "ohlcv.csv", index=False, encoding="utf-8")
            df_feat.to_csv(out_dir / "features.csv", index=False, encoding="utf-8")

        console.print(f"[green]完成[/green] 行数={result.ohlcv_rows} 缺口={result.gaps}")
        console.print(f"- JSON: {out_dir / 'analysis.json'}")
        console.print(f"- 报告: {out_dir / 'report.md'}")
        if export_csv:
            console.print(f"- OHLCV: {out_dir / 'ohlcv.csv'}")
            console.print(f"- 特征: {out_dir / 'features.csv'}")
        
        logger.info(f"分析完成，检测到 {len(result.analysis.events)} 个事件")
        return 0
        
    except WyckoffError as e:
        logger.error(f"分析失败: {e}")
        console.print(f"[red]错误[/red]: {e.message}")
        if e.details:
            for k, v in e.details.items():
                console.print(f"  {k}: {v}")
        return 1
    except Exception as e:
        logger.exception("未预期的错误")
        console.print(f"[red]未预期的错误[/red]: {e}")
        if args.verbose:
            console.print(traceback.format_exc())
        return 1


def cmd_mtf(args: argparse.Namespace) -> int:
    """多时间框架分析命令"""
    console = Console()
    out_dir = Path(args.out or os.getenv("WYCKOFF_OUT_DIR", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析时间框架
    timeframes = None
    if args.timeframes:
        timeframes = [tf.strip() for tf in args.timeframes.split(",")]

    console.print(
        f"[bold]多时间框架分析[/bold] symbol={args.symbol} "
        f"preset={args.preset or 'custom'} timeframes={timeframes or MTF_PRESETS.get(args.preset, [])}"
    )

    # 执行 MTF 分析
    result = analyze_mtf(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframes=timeframes,
        preset=args.preset,
        limit=args.limit,
        strict=args.strict,
    )

    # 保存结果
    # 构建 JSON 输出
    mtf_json = {
        "symbol": result.symbol,
        "exchange": result.exchange,
        "timeframes": result.timeframes,
        "overall_bias": result.overall_bias,
        "overall_confidence": result.overall_confidence,
        "resonance": {
            "aligned": result.resonance.aligned,
            "alignment_score": result.resonance.alignment_score,
            "dominant_bias": result.resonance.dominant_bias,
            "conflicts": result.resonance.conflicts,
            "notes": result.resonance.notes,
        },
        "structure_phase": result.structure_phase,
        "entry_timeframe": result.entry_timeframe,
        "entry_events": result.entry_events,
        "stop_reference": result.stop_reference,
        "target_reference": result.target_reference,
        "risk_level": result.risk_level,
        "risk_factors": result.risk_factors,
        "action_plan": result.action_plan,
        "tf_analyses": [
            {
                "timeframe": tfa.timeframe,
                "timeframe_name": tfa.timeframe_name,
                "bias": tfa.bias,
                "bias_strength": tfa.bias_strength,
                "market_structure": tfa.analysis.market_structure,
                "sequence_stage": tfa.sequence.current_stage,
                "key_events": [
                    {"type": e.type, "ts": e.ts, "price": e.price, "confidence": e.confidence}
                    for e in tfa.key_events
                ],
                "summary": tfa.summary,
            }
            for tfa in result.tf_analyses
        ],
    }
    _write_json(out_dir / "mtf_analysis.json", mtf_json)

    # 生成报告
    report_md = render_mtf_markdown(result)
    _write_text(out_dir / "mtf_report.md", report_md)

    console.print(f"[green]完成[/green] 分析了 {len(result.tf_analyses)} 个时间框架")
    console.print(f"- 主导方向: {result.overall_bias}")
    console.print(f"- 共振得分: {result.resonance.alignment_score*100:.0f}%")
    console.print(f"- JSON: {out_dir / 'mtf_analysis.json'}")
    console.print(f"- 报告: {out_dir / 'mtf_report.md'}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """回测命令"""
    from wyckoff_ai.backtest import (
        BacktestConfig as BtConfig,
        BacktestEngine,
        generate_backtest_report,
        calculate_metrics,
    )
    
    app_config = get_config()
    console = Console()
    
    # 合并配置：CLI 参数 > 配置文件
    symbol = args.symbol or app_config.analysis.default_symbol
    timeframe = args.timeframe or app_config.analysis.default_timeframe
    limit = args.limit or 1000
    exchange = args.exchange or app_config.exchange.name
    out_dir = Path(args.out or app_config.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始回测 {symbol} {timeframe}")

    try:
        console.print(
            f"[bold]回测[/bold] symbol={symbol} timeframe={timeframe} limit={limit}"
        )

        # 获取数据
        console.print("[dim]获取K线数据...[/dim]")
        fr = fetch_ohlcv_binance_spot(
            symbol=symbol, timeframe=timeframe, limit=limit
        )
        df = compute_features(fr.df)
        console.print(f"[dim]获取到 {len(df)} 根K线[/dim]")

        # 检测事件
        console.print("[dim]检测威科夫事件...[/dim]")
        cfg = DetectionConfig(lookback_bars=min(limit, 500))
        analysis = detect_wyckoff(
            df,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            cfg=cfg,
        )
        console.print(f"[dim]检测到 {len(analysis.events)} 个事件[/dim]")

        if not analysis.events:
            console.print("[yellow]未检测到事件，无法回测[/yellow]")
            return 1

        # 配置回测 - 从配置文件获取默认值
        bt_cfg = app_config.backtest
        allowed_events = None
        if args.events:
            allowed_events = [e.strip().upper() for e in args.events.split(",")]
        elif bt_cfg.allowed_events:
            allowed_events = bt_cfg.allowed_events

        config = BtConfig(
            initial_capital=args.capital or bt_cfg.initial_capital,
            position_size_pct=args.position_size or bt_cfg.position_size_pct,
            stop_loss_atr=args.stop_atr or bt_cfg.stop_loss_atr,
            take_profit_atr=args.target_atr or bt_cfg.take_profit_atr,
            min_confidence=args.min_confidence or app_config.analysis.min_confidence,
            max_bars_in_trade=args.max_bars or bt_cfg.max_bars_in_trade,
            allowed_events=allowed_events,
        )

        # 运行回测
        console.print("[dim]运行回测...[/dim]")
        engine = BacktestEngine(config)
        result = engine.run(df, analysis.events)

        # 生成报告
        report_md = generate_backtest_report(result, title=f"威科夫策略回测报告 - {args.symbol}")
        _write_text(out_dir / "backtest_report.md", report_md)

        # 保存回测结果JSON
        metrics = calculate_metrics(result)
        backtest_json = {
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "total_bars": result.total_bars,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": metrics.trade_metrics.win_rate,
            "total_return_pct": metrics.total_return_pct,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "profit_factor": metrics.trade_metrics.profit_factor,
            "avg_mfe": metrics.trade_metrics.avg_mfe,
            "avg_mae": metrics.trade_metrics.avg_mae,
            "config": {
                "initial_capital": config.initial_capital,
                "position_size_pct": config.position_size_pct,
                "stop_loss_atr": config.stop_loss_atr,
                "take_profit_atr": config.take_profit_atr,
                "min_confidence": config.min_confidence,
            },
            "stats_by_event": result.stats_by_event,
        }
        _write_json(out_dir / "backtest_result.json", backtest_json)

        # 输出摘要
        console.print("")
        console.print("[bold green]回测完成[/bold green]")
        console.print(f"- 总交易: {result.total_trades}")
        console.print(f"- 胜率: {metrics.trade_metrics.win_rate*100:.1f}%")
        console.print(f"- 总收益: {metrics.total_return_pct:.2f}%")
        console.print(f"- 最大回撤: {metrics.max_drawdown_pct:.2f}%")
        console.print(f"- Sharpe比率: {metrics.sharpe_ratio:.2f}")
        console.print(f"- 盈亏比: {metrics.trade_metrics.profit_factor:.2f}")
        console.print("")
        console.print(f"- 报告: {out_dir / 'backtest_report.md'}")
        console.print(f"- JSON: {out_dir / 'backtest_result.json'}")
        
        logger.info(f"回测完成，共 {result.total_trades} 笔交易")
        return 0
        
    except WyckoffError as e:
        logger.error(f"回测失败: {e}")
        console.print(f"[red]错误[/red]: {e.message}")
        if e.details:
            for k, v in e.details.items():
                console.print(f"  {k}: {v}")
        return 1
    except Exception as e:
        logger.exception("回测时发生未预期的错误")
        console.print(f"[red]未预期的错误[/red]: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            console.print(traceback.format_exc())
        return 1


def cmd_chart(args: argparse.Namespace) -> int:
    """可视化命令"""
    from wyckoff_ai.visualization import (
        create_analysis_chart,
        create_state_diagram,
        create_phase_progress_chart,
        create_timeline_chart,
        save_chart,
    )
    from wyckoff_ai.visualization.state_diagram import create_combined_state_view
    
    app_config = get_config()
    console = Console()
    
    # 合并配置：CLI 参数 > 配置文件
    symbol = args.symbol or app_config.analysis.default_symbol
    timeframe = args.timeframe or app_config.analysis.default_timeframe
    limit = args.limit or app_config.analysis.default_limit
    exchange = args.exchange or app_config.exchange.name
    out_dir = Path(args.out or app_config.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始生成可视化 {symbol} {timeframe}")

    try:
        console.print(
            f"[bold]生成可视化[/bold] symbol={symbol} timeframe={timeframe} limit={limit}"
        )

        # 获取数据
        console.print("[dim]获取K线数据...[/dim]")
        fr = fetch_ohlcv_binance_spot(
            symbol=symbol, timeframe=timeframe, limit=limit
        )
        df = compute_features(fr.df)
        console.print(f"[dim]获取到 {len(df)} 根K线[/dim]")

        # 检测事件
        console.print("[dim]检测威科夫事件...[/dim]")
        cfg = DetectionConfig(lookback_bars=min(limit, 500))
        analysis = detect_wyckoff(
            df,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            cfg=cfg,
        )
        console.print(f"[dim]检测到 {len(analysis.events)} 个事件[/dim]")

        # 运行状态机
        console.print("[dim]分析状态机...[/dim]")
        sm = WyckoffStateMachine()
        sm_result = sm.analyze(analysis.events, df)

        output_format = args.format or "html"
        min_confidence = args.min_confidence or app_config.analysis.min_confidence
        
        # 1. K线图表（带事件标注）
        console.print("[dim]生成K线图表...[/dim]")
        kline_chart = create_analysis_chart(
            df=df,
            analysis=analysis,
            show_volume=not args.no_volume,
            show_ema=not args.no_ema,
            show_events=True,
            show_levels=True,
            show_range=True,
            min_confidence=min_confidence,
            height=args.height or 900,
        )
        kline_path = save_chart(
            kline_chart,
            out_dir / "kline_chart",
            format=output_format,
            width=args.width or 1600,
            height=args.height or 900,
        )
        console.print(f"  - K线图表: {kline_path}")

        # 2. 状态机图
        if sm_result:
            console.print("[dim]生成状态机图...[/dim]")
            state_chart = create_state_diagram(
                result=sm_result,
                title=f"威科夫状态机 - {symbol}",
            )
            state_path = save_chart(
                state_chart,
                out_dir / "state_diagram",
                format=output_format,
                width=1000,
                height=700,
            )
            console.print(f"  - 状态机图: {state_path}")

            # 3. 阶段进度图
            progress_chart = create_phase_progress_chart(
                result=sm_result,
                title=f"阶段进度 - {symbol}",
            )
            progress_path = save_chart(
                progress_chart,
                out_dir / "phase_progress",
                format=output_format,
                width=800,
                height=300,
            )
            console.print(f"  - 阶段进度: {progress_path}")

            # 4. 状态转换时间线
            if sm_result.transition_history:
                timeline_chart = create_timeline_chart(
                    transitions=sm_result.transition_history,
                    title=f"状态转换时间线 - {symbol}",
                )
                timeline_path = save_chart(
                    timeline_chart,
                    out_dir / "state_timeline",
                    format=output_format,
                    width=1200,
                    height=400,
                )
                console.print(f"  - 时间线: {timeline_path}")

        console.print("")
        console.print("[bold green]可视化生成完成[/bold green]")
        console.print(f"- 输出目录: {out_dir}")
        
        logger.info(f"可视化完成，输出目录: {out_dir}")
        return 0
        
    except WyckoffError as e:
        logger.error(f"可视化生成失败: {e}")
        console.print(f"[red]错误[/red]: {e.message}")
        if e.details:
            for k, v in e.details.items():
                console.print(f"  {k}: {v}")
        return 1
    except Exception as e:
        logger.exception("可视化生成时发生未预期的错误")
        console.print(f"[red]未预期的错误[/red]: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            console.print(traceback.format_exc())
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """启动 Web 服务"""
    import uvicorn
    
    console = Console()
    
    console.print(f"[bold]启动威科夫分析 Web 服务[/bold]")
    console.print(f"- 地址: http://{args.host}:{args.port}")
    console.print(f"- API 文档: http://{args.host}:{args.port}/docs")
    console.print(f"- 开发模式: {'是' if args.reload else '否'}")
    console.print("")
    console.print("[dim]按 Ctrl+C 停止服务[/dim]")
    console.print("")
    
    try:
        uvicorn.run(
            "wyckoff_ai.web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]服务已停止[/yellow]")
        return 0
    except Exception as e:
        logger.exception("启动服务失败")
        console.print(f"[red]启动失败[/red]: {e}")
        return 1


def cmd_eval_events(args: argparse.Namespace) -> int:
    """事件评估命令"""
    from wyckoff_ai.backtest import (
        evaluate_events,
        generate_backtest_report,
    )
    from wyckoff_ai.backtest.report import generate_event_stats_report
    
    console = Console()
    out_dir = Path(args.out or os.getenv("WYCKOFF_OUT_DIR", "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始事件评估 {args.symbol} {args.timeframe}")

    try:
        console.print(
            f"[bold]事件评估[/bold] symbol={args.symbol} timeframe={args.timeframe} limit={args.limit}"
        )

        # 获取数据
        console.print("[dim]获取K线数据...[/dim]")
        fr = fetch_ohlcv_binance_spot(
            symbol=args.symbol, timeframe=args.timeframe, limit=args.limit
        )
        df = compute_features(fr.df)
        console.print(f"[dim]获取到 {len(df)} 根K线[/dim]")

        # 检测事件
        console.print("[dim]检测威科夫事件...[/dim]")
        cfg = DetectionConfig(lookback_bars=min(args.limit, 500))
        analysis = detect_wyckoff(
            df,
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            cfg=cfg,
        )
        console.print(f"[dim]检测到 {len(analysis.events)} 个事件[/dim]")

        if not analysis.events:
            console.print("[yellow]未检测到事件，无法评估[/yellow]")
            return 1

        # 评估事件
        console.print("[dim]评估事件表现...[/dim]")
        performances, stats = evaluate_events(df, analysis.events)

        # 生成报告
        report_md = generate_event_stats_report(stats, title=f"威科夫事件后验分析 - {args.symbol}")
        _write_text(out_dir / "event_eval_report.md", report_md)

        # 保存JSON
        eval_json = {
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "total_events": len(analysis.events),
            "evaluated_events": len(performances),
            "stats_by_type": {
                et: {
                    "count": s.count,
                    "win_rate_24": s.win_rate_24,
                    "median_return_24": s.median_return_24,
                    "avg_mfe": s.avg_mfe,
                    "avg_mae": s.avg_mae,
                    "profit_factor": s.profit_factor,
                    "direction_accuracy": s.direction_accuracy,
                }
                for et, s in stats.items()
            },
        }
        _write_json(out_dir / "event_eval_result.json", eval_json)

        # 输出摘要
        console.print("")
        console.print("[bold green]评估完成[/bold green]")
        console.print(f"- 评估事件数: {len(performances)}")
        console.print("")
        console.print("[bold]各事件类型表现:[/bold]")
        
        # 按胜率排序输出
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].win_rate_24, reverse=True)
        for et, s in sorted_stats:
            icon = "✅" if s.win_rate_24 > 0.5 else "❌" if s.win_rate_24 < 0.5 else "➖"
            console.print(
                f"  {icon} {et}: 样本={s.count}, 胜率={s.win_rate_24*100:.1f}%, "
                f"中位收益={s.median_return_24:.2f}%"
            )
        
        console.print("")
        console.print(f"- 报告: {out_dir / 'event_eval_report.md'}")
        console.print(f"- JSON: {out_dir / 'event_eval_result.json'}")
        
        logger.info(f"事件评估完成，评估了 {len(performances)} 个事件")
        return 0
        
    except WyckoffError as e:
        logger.error(f"事件评估失败: {e}")
        console.print(f"[red]错误[/red]: {e.message}")
        if e.details:
            for k, v in e.details.items():
                console.print(f"  {k}: {v}")
        return 1
    except Exception as e:
        logger.exception("事件评估时发生未预期的错误")
        console.print(f"[red]未预期的错误[/red]: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            console.print(traceback.format_exc())
        return 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wyckoff-ai", 
        description="威科夫K线分析工具（Binance + LangChain）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置文件支持：
  配置文件按以下顺序查找：
  1. --config 指定的路径
  2. 当前目录 wyckoff.toml
  3. ~/.wyckoff/config.toml
  4. 环境变量 WYCKOFF_CONFIG_FILE

  配置优先级：CLI 参数 > 环境变量 > 配置文件 > 默认值
  
示例：
  wyckoff-ai analyze --symbol BTC/USDT
  wyckoff-ai --config my-config.toml analyze --symbol ETH/USDT
  wyckoff-ai backtest --symbol BTC/USDT --capital 50000
        """
    )
    
    # 全局选项
    p.add_argument(
        "-c", "--config",
        default=None,
        metavar="FILE",
        help="配置文件路径（TOML 格式）"
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出（等同于 --log-level DEBUG）"
    )
    p.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认从配置文件读取，否则 INFO）"
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="日志文件路径（可选）"
    )
    p.add_argument(
        "--show-config",
        action="store_true",
        help="显示当前配置并退出"
    )
    
    sub = p.add_subparsers(dest="cmd")

    # 单时间框架分析
    a = sub.add_parser("analyze", help="拉取K线并输出威科夫结构分析（JSON+Markdown）")
    a.add_argument("--exchange", default=None, choices=["binance", "okx", "bybit"],
                   help="交易所（默认从配置文件读取）")
    a.add_argument("--symbol", "-s", default=None, help="交易对，如 BTC/USDT（默认从配置文件读取）")
    a.add_argument("--timeframe", "-t", default=None, help="时间周期，如 1h/4h/15m（默认从配置文件读取）")
    a.add_argument("--limit", "-l", type=int, default=None, help="拉取K线根数（默认从配置文件读取）")
    a.add_argument(
        "--lookback-bars",
        type=int,
        default=None,
        help="用于识别/统计的回看根数（默认=limit）",
    )
    a.add_argument("--strict", action="store_true", help="严格模式：低置信度不标注事件")
    a.add_argument(
        "--regime",
        default=None,
        choices=["kmeans", "cusum", "none"],
        help="状态识别方法（默认从配置文件读取）",
    )
    a.add_argument("--regime-k", type=int, default=None,
                   help="KMeans 聚类数（默认从配置文件读取）")
    a.add_argument("--out", "-o", default=None,
                   help="输出目录（默认从配置文件读取）")
    a.add_argument("--export-csv", action="store_true",
                   help="额外导出 ohlcv.csv/features.csv")
    a.set_defaults(func=cmd_analyze)

    # 多时间框架分析
    m = sub.add_parser("mtf", help="多时间框架威科夫分析（级别共振）")
    m.add_argument("--exchange", default=None, choices=["binance", "okx", "bybit"])
    m.add_argument("--symbol", "-s", default=None, help="交易对（默认从配置文件读取）")
    m.add_argument(
        "--preset",
        default="swing",
        choices=list(MTF_PRESETS.keys()),
        help="预设时间框架组合：swing(1d/4h/1h), intraday(4h/1h/15m), scalp(1h/15m/5m), position(1w/1d/4h)",
    )
    m.add_argument(
        "--timeframes",
        default=None,
        help="自定义时间框架，逗号分隔，如 '1d,4h,1h'（优先于 preset）",
    )
    m.add_argument("--limit", type=int, default=300, help="每个时间框架拉取的K线根数")
    m.add_argument("--strict", action="store_true", help="严格模式")
    m.add_argument("--out", default=None, help="输出目录")
    m.set_defaults(func=cmd_mtf)

    # 回测命令
    b = sub.add_parser("backtest", help="威科夫事件策略回测")
    b.add_argument("--exchange", default=None, choices=["binance", "okx", "bybit"])
    b.add_argument("--symbol", "-s", default=None, help="交易对（默认从配置文件读取）")
    b.add_argument("--timeframe", "-t", default=None, help="时间周期（默认从配置文件读取）")
    b.add_argument("--limit", "-l", type=int, default=None, help="拉取K线根数（默认 1000）")
    b.add_argument("--capital", type=float, default=None, help="初始资金（默认从配置文件读取）")
    b.add_argument("--position-size", type=float, default=None, help="单笔仓位百分比（默认从配置文件读取）")
    b.add_argument("--stop-atr", type=float, default=None, help="止损 ATR 倍数（默认从配置文件读取）")
    b.add_argument("--target-atr", type=float, default=None, help="止盈 ATR 倍数（默认从配置文件读取）")
    b.add_argument("--min-confidence", type=float, default=None, help="最小置信度（默认从配置文件读取）")
    b.add_argument("--max-bars", type=int, default=None, help="最大持仓K线数（默认从配置文件读取）")
    b.add_argument(
        "--events",
        default=None,
        help="允许交易的事件类型，逗号分隔，如 'SOS,SPRING,LPS'（默认从配置文件读取）",
    )
    b.add_argument("--out", "-o", default=None, help="输出目录（默认从配置文件读取）")
    b.set_defaults(func=cmd_backtest)

    # 事件评估命令
    e = sub.add_parser("eval-events", help="评估威科夫事件的后验表现")
    e.add_argument("--exchange", default=None, choices=["binance", "okx", "bybit"])
    e.add_argument("--symbol", "-s", default=None, help="交易对（默认从配置文件读取）")
    e.add_argument("--timeframe", "-t", default=None, help="时间周期（默认从配置文件读取）")
    e.add_argument("--limit", "-l", type=int, default=None, help="拉取K线根数（默认 1000）")
    e.add_argument("--out", "-o", default=None, help="输出目录（默认从配置文件读取）")
    e.set_defaults(func=cmd_eval_events)

    # 可视化命令
    c = sub.add_parser("chart", help="生成威科夫分析可视化图表")
    c.add_argument("--exchange", default=None, choices=["binance", "okx", "bybit"])
    c.add_argument("--symbol", "-s", default=None, help="交易对（默认从配置文件读取）")
    c.add_argument("--timeframe", "-t", default=None, help="时间周期（默认从配置文件读取）")
    c.add_argument("--limit", "-l", type=int, default=None, help="拉取K线根数（默认从配置文件读取）")
    c.add_argument(
        "--format", "-f",
        default=None,
        choices=["html", "png", "svg", "pdf"],
        help="输出格式（默认 html）"
    )
    c.add_argument("--width", type=int, default=None, help="图片宽度（默认 1600）")
    c.add_argument("--height", type=int, default=None, help="图片高度（默认 900）")
    c.add_argument("--min-confidence", type=float, default=None, help="最小事件置信度（默认从配置文件读取）")
    c.add_argument("--no-volume", action="store_true", help="不显示成交量")
    c.add_argument("--no-ema", action="store_true", help="不显示 EMA 均线")
    c.add_argument("--out", "-o", default=None, help="输出目录（默认从配置文件读取）")
    c.set_defaults(func=cmd_chart)

    # Web 服务命令
    w = sub.add_parser("serve", help="启动 Web 服务（REST API + 前端界面）")
    w.add_argument("--host", default="0.0.0.0", help="绑定地址（默认 0.0.0.0）")
    w.add_argument("--port", "-p", type=int, default=8000, help="端口（默认 8000）")
    w.add_argument("--reload", action="store_true", help="开发模式，自动重载代码")
    w.set_defaults(func=cmd_serve)

    return p


def main() -> int:
    load_dotenv(override=False)
    # Windows/CI 里经常出现 cp936 等导致的中文乱码；强制 UTF-8 输出更稳
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # 加载配置文件
    try:
        config = load_config(config_file=args.config)
        set_config(config)
    except Exception as e:
        console = Console()
        console.print(f"[red]配置加载失败[/red]: {e}")
        return 1
    
    # 配置日志 - 优先级：CLI > 配置文件
    log_level: LogLevel = "INFO"
    if args.verbose:
        log_level = "DEBUG"
    elif args.log_level:
        log_level = args.log_level
    else:
        log_level = config.logging.level
    
    log_file = args.log_file or config.logging.file
    
    setup_logging(
        level=log_level,
        log_file=log_file,
        show_path=args.verbose or config.logging.show_path,
        rich_traceback=config.logging.rich_traceback,
    )
    
    # 显示配置并退出
    if args.show_config:
        console = Console()
        console.print("[bold]当前配置:[/bold]")
        import json as json_mod
        console.print(json_mod.dumps(config.model_dump(), indent=2, ensure_ascii=False))
        return 0
    
    # 检查是否指定了命令
    if not args.cmd:
        parser.print_help()
        return 0
    
    logger.debug(f"CLI 启动，命令: {args.cmd}")
    logger.debug(f"配置文件: {args.config or '默认'}")
    
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
