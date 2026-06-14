"""
cwh_peak_detection.py
ピーク検知モジュール：上ヒゲ・大商い、窓開け陰線
"""

import pandas as pd
from typing import Tuple


def detect_upper_wick_large_volume(df: pd.DataFrame, v_25_avg: float) -> Tuple[bool, str]:
    """
    上ヒゲ・大商い判定。
    条件: V_t >= 2.0 * V_25 かつ 上ヒゲの長さ > 実体の長さ
    """
    curr = df.iloc[-1]
    upper_wick: float = float(curr['High']) - max(float(curr['Open']), float(curr['Close']))
    body_size: float = abs(float(curr['Open']) - float(curr['Close']))

    if float(curr['Volume']) >= 2.0 * v_25_avg and upper_wick > body_size:
        return True, "Upper wick + large volume"
    return False, ""


def detect_gap_up_bearish(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    窓開け陰線判定。
    条件: 寄り付きが前日終値の+3%以上 かつ 当日終値 < 当日始値（陰線）
    """
    if len(df) < 2:
        return False, ""
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    is_gap_up: bool = float(curr['Open']) >= float(prev['Close']) * 1.03
    is_bearish: bool = float(curr['Close']) < float(curr['Open'])

    if is_gap_up and is_bearish:
        return True, "Gap up + bearish candle"
    return False, ""


def detect_peak(df: pd.DataFrame, v_25_avg: float) -> Tuple[bool, str]:
    """
    ピーク検知メイン関数。
    いずれかのシグナルが発生した場合 (True, reason) を返す。
    """
    triggered, reason = detect_upper_wick_large_volume(df, v_25_avg)
    if triggered:
        return True, reason

    triggered, reason = detect_gap_up_bearish(df)
    if triggered:
        return True, reason

    return False, ""
