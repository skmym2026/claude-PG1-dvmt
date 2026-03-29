"""
cwh_realtime_monitor.py - リアルタイム監視モジュール

毎営業日の引け後に CWH パターン候補と保有銘柄の
ピーク検知アラートを確認するモニタリング機能。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from cwh_data_fetcher import fetch_ohlcv
from cwh_pattern_analyzer import PatternResult, analyze_patterns
from cwh_screener import detect_cwh_candidates
from cwh_trade_rules import ExitReason

logger = logging.getLogger(__name__)


def check_peak_signals(
    df: pd.DataFrame,
    entry_price: float,
    cfg: dict,
) -> list[str]:
    """直近バーのピーク検知アラートを返す。

    Args:
        df: OHLCV DataFrame（直近分を含む）
        entry_price: エントリー価格
        cfg: cwh_config.yaml の backtest セクション

    Returns:
        アラートメッセージのリスト（空なら正常）
    """
    alerts: list[str] = []
    if df.empty:
        return alerts

    vol_period: int = 25
    peak_vol_ratio: float = cfg.get("peak_detection_volume_ratio", 2.0)

    # 最新バー
    last = df.iloc[-1]
    vol_avg = df["Volume"].iloc[-vol_period - 1 : -1].mean()
    body = abs(last["Close"] - last["Open"])
    upper_wick = last["High"] - max(last["Close"], last["Open"])

    # 大商い上ヒゲ
    if last["Volume"] >= peak_vol_ratio * vol_avg and upper_wick > body:
        alerts.append(
            f"⚠️ 大商い上ヒゲ検知: Vol={last['Volume']:.0f} (avg={vol_avg:.0f}), "
            f"上ヒゲ={upper_wick:.1f} > 実体={body:.1f}"
        )

    # 窓開け陰線
    if len(df) >= 2:
        prev_close = df["Close"].iloc[-2]
        if last["Close"] < last["Open"] and last["Open"] > prev_close:
            alerts.append(
                f"⚠️ 窓開け陰線検知: Open={last['Open']:.1f} > PrevClose={prev_close:.1f}, "
                f"Close={last['Close']:.1f}"
            )

    return alerts


def monitor_watchlist(
    tickers: list[str],
    cfg: dict,
    lookback_days: int = 200,
) -> dict[str, list[PatternResult]]:
    """監視銘柄リストについて最新のパターンを確認する。

    Args:
        tickers: 監視ティッカーリスト
        cfg: cwh_config.yaml の全設定
        lookback_days: 過去何日分を取得するか

    Returns:
        {ticker: PatternResult リスト}
    """
    today = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    result: dict[str, list[PatternResult]] = {}
    for ticker in tickers:
        df = fetch_ohlcv(ticker, start, today)
        if df.empty:
            continue
        candidates = detect_cwh_candidates(df, ticker, cfg["screener"])
        patterns = analyze_patterns(candidates, cfg["patterns"])
        result[ticker] = patterns
        if patterns:
            for p in patterns:
                logger.info(
                    "監視銘柄 %s: %s パターン検出 (%s)",
                    ticker,
                    p.pattern_type.value,
                    p.reason,
                )
        else:
            logger.info("監視銘柄 %s: パターン検出なし", ticker)

    return result


def run_daily_monitor(cfg: dict) -> None:
    """毎営業日の引け後監視を実行するエントリーポイント。

    Args:
        cfg: cwh_config.yaml の全設定
    """
    monitoring_tickers: list[str] = cfg.get("stocks", {}).get("monitoring", [])
    logger.info("=== デイリー監視開始 ===")
    patterns = monitor_watchlist(monitoring_tickers, cfg)
    for ticker, plist in patterns.items():
        if plist:
            print(f"[{ticker}] {len(plist)} パターン検出:")
            for p in plist:
                print(f"  {p.pattern_type.value}: {p.reason}")
        else:
            print(f"[{ticker}] パターンなし")
    logger.info("=== デイリー監視完了 ===")
