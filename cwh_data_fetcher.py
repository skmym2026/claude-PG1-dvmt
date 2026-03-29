"""
cwh_data_fetcher.py - データ取得モジュール（yfinance）

日本株（.T シンボル）の日足OHLCVデータを yfinance から取得する。
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """指定銘柄の日足 OHLCV データを取得する。

    Args:
        ticker: ティッカーシンボル（例: "5253.T"）
        start: 取得開始日（YYYY-MM-DD 形式）
        end: 取得終了日（YYYY-MM-DD 形式）
        auto_adjust: 分割・配当調整済みフラグ

    Returns:
        DataFrame（列: Open, High, Low, Close, Volume）。
        取得失敗時は空の DataFrame を返す。
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        df: pd.DataFrame = ticker_obj.history(
            start=start,
            end=end,
            auto_adjust=auto_adjust,
        )
        if df.empty:
            logger.warning("データが取得できませんでした: %s", ticker)
            return pd.DataFrame()
        # 必要な列だけ残す
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        logger.info("%s: %d 行取得 (%s ～ %s)", ticker, len(df), start, end)
        return df
    except Exception as exc:
        logger.error("取得エラー %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_multiple(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """複数銘柄の OHLCV データを順次取得する（逐次処理）。

    Intel Core i3 環境での動作保証のため、並列処理は使用しない。

    Args:
        tickers: ティッカーシンボルのリスト
        start: 取得開始日（YYYY-MM-DD 形式）
        end: 取得終了日（YYYY-MM-DD 形式）

    Returns:
        {ticker: DataFrame} の辞書
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = fetch_ohlcv(ticker, start, end)
        if not df.empty:
            result[ticker] = df
    return result


def get_market_cap(ticker: str) -> Optional[float]:
    """時価総額（円）を取得する。

    Args:
        ticker: ティッカーシンボル

    Returns:
        時価総額（円）。取得不可の場合は None。
    """
    try:
        info = yf.Ticker(ticker).info
        market_cap = info.get("marketCap")
        return float(market_cap) if market_cap else None
    except Exception as exc:
        logger.error("時価総額取得エラー %s: %s", ticker, exc)
        return None
