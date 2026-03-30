"""
cwh_integrated_filters.py
フィルター群：時価総額・信用残・流動性・EPS成長性
"""

import pandas as pd
from typing import Dict, Optional


def check_market_cap(stock_info: Dict) -> bool:
    """
    時価総額フィルター。
    MC >= 1,000億円（機関投資家の参入可能性）
    """
    return stock_info.get('market_cap', 0) >= 1000


def check_credit_ratio(stock_info: Dict) -> bool:
    """
    信用倍率フィルター。
    Credit_Ratio <= 5.0（「上値のしこり」を排除）
    """
    return stock_info.get('credit_ratio', 0) <= 5.0


def check_liquidity(df: pd.DataFrame) -> bool:
    """
    流動性チェック。
    LQ = Close × Volume >= 2億円
    """
    curr = df.iloc[-1]
    lq: float = curr['Close'] * curr['Volume']
    return lq >= 200_000_000


def check_eps_growth(df: pd.DataFrame) -> bool:
    """
    EPS成長性チェック。
    EPS_{t-1} > EPS_{t-2} > EPS_{t-3}（連続増益）
    """
    if 'EPS' not in df.columns or len(df) < 3:
        return True  # データ不足の場合はスキップ（フィルタしない）
    eps = df['EPS'].dropna()
    if len(eps) < 3:
        return True
    return float(eps.iloc[-1]) > float(eps.iloc[-2]) > float(eps.iloc[-3])


def check_cup_depth(h_l: float, b_1: float) -> bool:
    """
    調整幅（カップの深さ）チェック。
    0.15 <= Gap_cup / HL <= 0.35
    """
    if h_l <= 0:
        return False
    gap_cup = h_l - b_1
    depth_ratio = gap_cup / h_l
    return 0.15 <= depth_ratio <= 0.35


def check_midpoint_break(df: pd.DataFrame, h_l: float, b_1: float) -> bool:
    """
    中点突破判定。
    P_t > (2·HL + B_1) / 3
    """
    curr_close = float(df.iloc[-1]['Close'])
    midpoint = (2 * h_l + b_1) / 3
    return curr_close > midpoint


def apply_all_filters(df: pd.DataFrame, stock_info: Dict, h_l: float, b_1: float) -> bool:
    """
    全フィルターを一括チェック。
    Falseが返ればエントリー不可。
    """
    return (
        check_market_cap(stock_info)
        and check_credit_ratio(stock_info)
        and check_liquidity(df)
        and check_eps_growth(df)
        and check_cup_depth(h_l, b_1)
        and check_midpoint_break(df, h_l, b_1)
    )
