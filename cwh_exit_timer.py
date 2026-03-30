"""
cwh_exit_timer.py
経過日数・足切り管理モジュール
"""

import pandas as pd
from typing import Optional, Tuple, Dict


def calc_days_held(
    entry_date: Optional[pd.Timestamp],
    as_of: Optional[pd.Timestamp] = None,
) -> int:
    """
    エントリー日からの経過日数を返す。

    Args:
        entry_date: エントリー日
        as_of: 基準日（省略時は今日）。テストや履歴分析で任意の日付を指定可能。
    """
    if entry_date is None:
        return 0
    reference = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now().normalize()
    return max(0, (reference - pd.Timestamp(entry_date).normalize()).days)


def calc_unrealized_gain(current_price: float, entry_price: float) -> float:
    """
    含み益率を返す（例: 0.12 → +12%）。
    """
    if entry_price <= 0:
        return 0.0
    return (current_price - entry_price) / entry_price


def check_time_cutoff(
    days_held: int,
    unrealized_gain: float,
) -> Tuple[bool, str]:
    """
    経過日数による足切り判定。

    Rules:
    - 75日以上経過 → 無条件売却
    - 25日以上経過 かつ 含み益 +10% 未満 → 売却
    - 利益 +20% 以上の場合はトレール継続（75日ルールに含む）

    Returns:
        (is_cutoff, reason)
    """
    if days_held >= 75:
        if unrealized_gain >= 0.20:
            return False, ""  # +20%以上ならトレール継続
        return True, f"75-day cutoff (time-axis limit), gain={unrealized_gain*100:.1f}%"

    if days_held >= 25 and unrealized_gain < 0.10:
        return True, f"25-day cutoff with {unrealized_gain*100:.1f}% gain"

    return False, ""


def evaluate_exit_timer(
    stock_info: Dict,
    current_price: float,
    as_of: Optional[pd.Timestamp] = None,
) -> Tuple[bool, str]:
    """
    stock_info から entry_date・entry_price を読み取り足切り判定を行う。

    Args:
        stock_info: {'entry_date': pd.Timestamp, 'entry_price': float, ...}
        current_price: 現在の終値
        as_of: 基準日（省略時は今日）。テストや履歴分析で任意の日付を指定可能。

    Returns:
        (should_sell, reason)
    """
    entry_date = stock_info.get('entry_date')
    entry_price = stock_info.get('entry_price', current_price)

    days_held = calc_days_held(entry_date, as_of=as_of)
    unrealized_gain = calc_unrealized_gain(current_price, entry_price)

    return check_time_cutoff(days_held, unrealized_gain)
