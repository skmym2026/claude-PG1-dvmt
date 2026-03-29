"""
cwh_backtest_engine.py - バックテストエンジン

CWH パターンを検出し、トレードルールに従いシミュレーションを実行する。
Intel Core i3 環境での動作保証のため、逐次処理を採用する。
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from cwh_data_fetcher import fetch_ohlcv
from cwh_pattern_analyzer import PatternResult, PatternType, analyze_patterns
from cwh_performance_metrics import PerformanceMetrics, compute_metrics
from cwh_screener import detect_cwh_candidates
from cwh_trade_rules import ExitReason, TradeRecord, apply_exit_rules, compute_entry_signal

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """バックテスト結果の集約。"""

    ticker: str
    trades: list[TradeRecord]
    metrics: PerformanceMetrics
    pattern_results: list[PatternResult]


def run_backtest_single(
    ticker: str,
    start: str,
    end: str,
    cfg: dict,
) -> Optional[BacktestResult]:
    """1 銘柄のバックテストを実行する。

    Args:
        ticker: ティッカーシンボル
        start: バックテスト開始日
        end: バックテスト終了日
        cfg: cwh_config.yaml の全設定（辞書）

    Returns:
        BacktestResult。データ取得失敗時は None。
    """
    df = fetch_ohlcv(ticker, start, end)
    if df.empty:
        logger.warning("スキップ: %s（データなし）", ticker)
        return None

    # CWH 候補を検出
    candidates = detect_cwh_candidates(df, ticker, cfg["screener"])
    if not candidates:
        logger.info("パターン検出なし: %s", ticker)
        return BacktestResult(
            ticker=ticker,
            trades=[],
            metrics=compute_metrics([]),
            pattern_results=[],
        )

    # パターン分類
    pattern_results = analyze_patterns(candidates, cfg["patterns"])

    trades: list[TradeRecord] = []

    for pattern in pattern_results:
        cand = pattern.candidate

        # エントリー価格の決定（右リム高値をピボット価格とする）
        pivot_price = df["High"].iloc[cand.idx_pr]

        entry_bar = compute_entry_signal(
            df,
            cand.idx_pv,
            pivot_price,
            cfg["backtest"],
        )
        if entry_bar is None:
            continue

        entry_price = df["Close"].iloc[entry_bar]
        entry_date = df.index[entry_bar]

        # 出口判定
        exit_bar, exit_price, exit_reason = apply_exit_rules(
            df,
            entry_bar,
            entry_price,
            cfg["backtest"],
        )

        exit_date = df.index[exit_bar]

        record = TradeRecord(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=entry_price,
        )
        record.exit_date = exit_date
        record.exit_price = exit_price
        record.exit_reason = exit_reason

        trades.append(record)
        logger.info(
            "%s: エントリー %s@%.1f → 出口 %s@%.1f (%s) pnl=%.1f%%",
            ticker,
            entry_date.date(),
            entry_price,
            exit_date.date(),
            exit_price,
            exit_reason.value,
            record.pnl_pct,
        )

    pnl_list = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    metrics = compute_metrics(pnl_list)

    return BacktestResult(
        ticker=ticker,
        trades=trades,
        metrics=metrics,
        pattern_results=pattern_results,
    )


def run_backtest_all(
    tickers: list[str],
    start: str,
    end: str,
    cfg: dict,
) -> list[BacktestResult]:
    """複数銘柄を逐次バックテストする（並列処理なし）。

    Args:
        tickers: ティッカーシンボルのリスト
        start: バックテスト開始日
        end: バックテスト終了日
        cfg: cwh_config.yaml の全設定

    Returns:
        BacktestResult のリスト
    """
    results: list[BacktestResult] = []
    for ticker in tickers:
        logger.info("バックテスト開始: %s", ticker)
        result = run_backtest_single(ticker, start, end, cfg)
        if result is not None:
            results.append(result)
    return results


def export_trades_csv(
    results: list[BacktestResult],
    output_path: str = "backtest_results.csv",
) -> None:
    """トレード一覧を CSV に出力する。

    Args:
        results: BacktestResult のリスト
        output_path: 出力 CSV ファイルパス
    """
    rows: list[dict] = []
    for result in results:
        for trade in result.trades:
            rows.append(
                {
                    "ticker": trade.ticker,
                    "entry_date": trade.entry_date.date() if trade.entry_date else "",
                    "entry_price": round(trade.entry_price, 2),
                    "exit_date": trade.exit_date.date() if trade.exit_date else "",
                    "exit_price": round(trade.exit_price, 2) if trade.exit_price else "",
                    "exit_reason": trade.exit_reason.value if trade.exit_reason else "",
                    "pnl_pct": round(trade.pnl_pct, 2) if trade.pnl_pct is not None else "",
                }
            )

    if not rows:
        logger.warning("出力するトレードがありません")
        return

    fieldnames = ["ticker", "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "pnl_pct"]
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("CSV 出力完了: %s (%d 行)", output_path, len(rows))
