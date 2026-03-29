"""
cwh_growth_market_screener.py - 東証グロース市場全銘柄スクリーニング

yfinance から東証グロース市場銘柄を取得し、CWH パターンを一括スクリーニングする。
Intel Core i3 環境での動作保証のため、逐次処理を採用する。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from cwh_data_fetcher import fetch_ohlcv, get_market_cap
from cwh_pattern_analyzer import PatternResult, analyze_patterns
from cwh_screener import detect_cwh_candidates

logger = logging.getLogger(__name__)

# 東証グロース市場の代表的銘柄コード例（実運用時は適宜更新）
GROWTH_MARKET_SAMPLE: list[str] = [
    "153A.T",   # カウリス
    "485A.T",   # パワーエックス
    "5253.T",   # カバー
    "6228.T",   # JET
    "5586.T",   # ラボロ
    "5242.T",   # アイズ
    "5885.T",   # ジーディープ
    "4894.T",   # クオリプス
]


def screen_growth_market(
    tickers: Optional[list[str]],
    cfg: dict,
    lookback_days: int = 400,
    top_n: int = 10,
) -> list[PatternResult]:
    """東証グロース市場の銘柄をスクリーニングし、CWH 候補の上位を返す。

    Args:
        tickers: スクリーニング対象ティッカーリスト（None の場合はサンプルリストを使用）
        cfg: cwh_config.yaml の全設定
        lookback_days: 過去何日分を取得するか
        top_n: 上位何件を返すか

    Returns:
        PatternResult のリスト（最大 top_n 件）
    """
    if tickers is None:
        tickers = GROWTH_MARKET_SAMPLE

    today = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    all_patterns: list[PatternResult] = []
    market_cap_min: float = cfg.get("filters", {}).get("market_cap_min", 0)

    for ticker in tickers:
        # 時価総額フィルター
        mc = get_market_cap(ticker)
        if mc is not None and mc < market_cap_min:
            logger.info("時価総額フィルター除外: %s (%.0f 円)", ticker, mc)
            continue

        df = fetch_ohlcv(ticker, start, today)
        if df.empty:
            continue

        candidates = detect_cwh_candidates(df, ticker, cfg["screener"])
        patterns = analyze_patterns(candidates, cfg["patterns"])
        all_patterns.extend(patterns)
        logger.info("%s: %d パターン検出", ticker, len(patterns))

    # 直近のパターンを優先（idx_pv が大きいほど直近）
    all_patterns.sort(key=lambda p: p.candidate.idx_pv, reverse=True)
    return all_patterns[:top_n]
