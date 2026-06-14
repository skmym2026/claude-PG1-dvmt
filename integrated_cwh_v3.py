"""
integrated_cwh_v3.py
CWH統合型スキャニング＆出口戦略：メイン統合エンジン
"""

import pandas as pd
from typing import Optional, Dict

from cwh_integrated_filters import apply_all_filters, check_market_cap, check_credit_ratio, check_liquidity
from cwh_peak_detection import detect_peak
from cwh_exit_timer import calc_days_held, calc_unrealized_gain, check_time_cutoff
from cwh_sbi_order_output import format_sbi_order


def _find_cup_params(df: pd.DataFrame):
    """
    カップパラメータを計算する。

    Returns:
        (h_l, b_1, gap_cup) — 左頂点高値、第1ボトム安値、カップの深さ
    """
    lookback = df.iloc[-60:] if len(df) >= 60 else df
    h_l: float = float(lookback['High'].max())
    hl_idx = lookback['High'].idxmax()

    # HL以降の安値
    after_hl = df.loc[hl_idx:] if hl_idx in df.index else df
    b_1: float = float(after_hl['Low'].min())
    gap_cup: float = h_l - b_1
    return h_l, b_1, gap_cup


def cwh_integrated_strategy(df: pd.DataFrame, stock_info: Dict) -> Optional[Dict]:
    """
    統合CWH戦略エンジン。

    Args:
        df: OHLCV + EPS データフレーム (columns: Open, High, Low, Close, Volume[, EPS])
        stock_info: {code, market_cap, credit_ratio[, entry_date, entry_price]}

    Returns:
        {'ACTION': 'SELL_NOW'|'BUY'|'HOLD', 'CODE': str, 'TARGET_1': float,
         'TRAILING_STOP_WIDTH': float, 'REASON': str, 'ENTRY_PRICE': float,
         'DAYS_HELD': int} または None
    """
    if len(df) < 2:
        return None

    # ==================== 0. 基本フィルタ ====================
    if not check_market_cap(stock_info):
        return None
    if not check_credit_ratio(stock_info):
        return None
    if not check_liquidity(df):
        return None

    # ==================== 1. カップパラメータ ====================
    h_l, b_1, gap_cup = _find_cup_params(df)

    # ==================== 2. 全フィルター適用 ====================
    if not apply_all_filters(df, stock_info, h_l, b_1):
        return None

    # ==================== 3. ブレイク判定（エントリー） ====================
    curr = df.iloc[-1]
    v_25_avg: float = float(df['Volume'].rolling(25).mean().iloc[-1])

    is_entry: bool = (
        float(curr['Close']) >= 1.03 * h_l
        and float(curr['Volume']) >= 1.5 * v_25_avg
    )

    if not is_entry:
        return None

    # ==================== 4. ピーク検知 ====================
    is_peak_out, peak_reason = detect_peak(df, v_25_avg)

    # ==================== 5. 経過日数・足切り ====================
    entry_date = stock_info.get('entry_date')
    entry_price = stock_info.get('entry_price', float(curr['Close']))
    days_held: int = calc_days_held(entry_date)

    if not is_peak_out and entry_date is not None:
        unrealized_gain = calc_unrealized_gain(float(curr['Close']), entry_price)
        cutoff, cutoff_reason = check_time_cutoff(days_held, unrealized_gain)
        if cutoff:
            is_peak_out = True
            peak_reason = cutoff_reason

    # ==================== 6. 結果出力 ====================
    target_1: float = h_l + (gap_cup * 0.618)
    trailing_stop_width: float = float(curr['Close']) * 0.03

    return {
        "ACTION": "SELL_NOW" if is_peak_out else "BUY",
        "CODE": stock_info['code'],
        "ENTRY_PRICE": float(curr['Close']),
        "TARGET_1": target_1,
        "TRAILING_STOP": float(curr['Close']) * 0.92,
        "TRAILING_STOP_WIDTH": trailing_stop_width,
        "DAYS_HELD": days_held,
        "REASON": peak_reason if is_peak_out else "Breakout signal",
        "MARKET_CAP": stock_info['market_cap'],
        "CREDIT_RATIO": stock_info['credit_ratio'],
        "LIQUIDITY_2B_YEN": float(curr['Close']) * float(curr['Volume']) / 100_000_000,
    }


if __name__ == "__main__":
    import numpy as np

    rng = pd.date_range("2025-01-01", periods=70, freq="B")
    close_prices = [1000.0] * 10 + [1100.0] + [950.0] * 30 + [1050.0] * 28 + [1140.0]
    df_demo = pd.DataFrame(
        {
            "Open": [p * 0.99 for p in close_prices],
            "High": [p * 1.02 for p in close_prices],
            "Low": [p * 0.97 for p in close_prices],
            "Close": close_prices,
            "Volume": [500_000] * 69 + [800_000],
            "EPS": list(range(1, 71)),
        },
        index=rng,
    )

    info = {"code": "9999.T", "market_cap": 2000, "credit_ratio": 2.5}
    result = cwh_integrated_strategy(df_demo, info)
    print(result)
    if result:
        order = format_sbi_order(result)
        print(order)
