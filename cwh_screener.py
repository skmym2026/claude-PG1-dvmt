"""
cwh_screener.py - CWH パターン検出エンジン（条件1〜10）

前回指示書の判定条件1〜10 に基づき、OHLCV データから
Cup with Handle パターンの候補を検出する。

アルゴリズム概要（O(n²) 設計）:
  1. 左リム（PL）候補：局所高値バーを列挙
  2. 各 PL に対し、右方向でカップ底（BC）= 実際の最安値を確定
  3. BC の右側でカップ長さ・リム対称条件を満たす右リム（PR）を探索
  4. PR の右側でハンドル底（BH）= 実際の最安値を確定
  5. 全条件を評価して CWHCandidate に格納
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CWHCandidate:
    """CWH パターン候補の検出結果を格納するデータクラス。"""

    ticker: str
    # キーポイントのインデックス位置
    idx_pl: int        # PL：左リム
    idx_bc: int        # BC：カップ底（PL〜PR 間の最安値バー）
    idx_pr: int        # PR：右リム
    idx_bh: int        # BH：ハンドル底（PR〜PV 間の最安値バー）
    idx_pv: int        # PV：ピボット（ブレイクアウト点）

    # 判定値
    r_prior: float     # 条件1：事前上昇率
    d_cup: float       # 条件2：カップ深さ
    n_cup: int         # 条件3：カップ長さ（営業日）
    d_handle: float    # 条件4：ハンドル深さ
    n_handle: int      # 条件5：ハンドル長さ（営業日）
    v_handle_avg: float  # 条件6：ハンドル平均出来高
    v_cup_avg: float     # 条件6：カップ平均出来高
    s_rim: float       # 条件7：リム対称性
    ma50_bh: float     # 条件8：BH 時点の MA50
    close_bh: float    # 条件8：BH の終値
    e_pivot: float     # 条件9：ピボット有効性
    v_pv: float        # 条件10：PV の出来高
    v20_pv: float      # 条件10：PV 時点の 20 日平均出来高

    # 各条件の合否フラグ
    cond_flags: dict[str, bool] = field(default_factory=dict)

    @property
    def all_pass(self) -> bool:
        """全条件を満たしているか。"""
        return all(self.cond_flags.values())


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """単純移動平均を計算する。"""
    return series.rolling(window=window, min_periods=window).mean()


def _find_local_highs(high: pd.Series, window: int = 5) -> list[int]:
    """局所高値（周辺 window バーの中で最大値）のインデックス一覧を返す。

    Args:
        high: High 価格の Series
        window: 左右それぞれの比較ウィンドウ幅

    Returns:
        局所高値バーのインデックスリスト
    """
    n = len(high)
    peaks: list[int] = []
    for i in range(window, n - window):
        left_max = high.iloc[i - window : i].max()
        right_max = high.iloc[i + 1 : i + window + 1].max()
        if high.iloc[i] >= left_max and high.iloc[i] >= right_max:
            peaks.append(i)
    return peaks


def detect_cwh_candidates(
    df: pd.DataFrame,
    ticker: str,
    cfg: dict,
) -> list[CWHCandidate]:
    """OHLCV データから CWH パターン候補を全件検出する。

    アルゴリズム（O(n²) 以下）:
      ① 局所高値を列挙 → PL 候補
      ② 各 PL で条件1（事前上昇）確認
      ③ PL から cup_search_range バー内の最安値バー → BC を確定
      ④ BC の右側で条件2〜9 を満たす PR を探索
      ⑤ PR の右側でハンドル範囲の最安値バー → BH を確定
      ⑥ 条件4〜10 を評価して CWHCandidate を生成

    Args:
        df: OHLCV DataFrame（列: Open, High, Low, Close, Volume）
        ticker: 銘柄コード
        cfg: cwh_config.yaml の screener セクション

    Returns:
        CWHCandidate のリスト（all_pass=True のものと False のものを含む）
    """
    min_bars_needed = cfg["cup_bars_min"] + cfg["handle_bars_max"] + cfg["prior_trend_bars"]
    if len(df) < min_bars_needed:
        return []

    # 事前計算：移動平均
    ma50: pd.Series = _rolling_mean(df["Close"], cfg["ma_period"])
    v20: pd.Series = _rolling_mean(df["Volume"], cfg["breakout_volume_period"])

    n = len(df)
    cup_search_range: int = cfg.get("cup_search_range", 120)
    local_high_window: int = cfg.get("local_high_window", 5)

    # ① 局所高値を PL 候補として列挙
    pl_candidates = _find_local_highs(df["High"], window=local_high_window)

    candidates: list[CWHCandidate] = []

    for i_pl in pl_candidates:
        # PL が探索範囲外なら skip
        if i_pl < cfg["prior_trend_bars"]:
            continue
        if i_pl >= n - cfg["cup_bars_min"] - cfg["handle_bars_max"] - 5:
            continue

        h_pl = df["High"].iloc[i_pl]
        if h_pl <= 0:
            continue

        # ② 条件1：事前上昇トレンド確認
        # Rprior = (C(PL) - min_C_prior) / min_C_prior >= prior_trend_min
        prior_start = max(0, i_pl - cfg["prior_trend_bars"])
        min_prior = df["Close"].iloc[prior_start:i_pl].min()
        if min_prior <= 0:
            continue
        r_prior = (df["Close"].iloc[i_pl] - min_prior) / min_prior
        cond1 = r_prior >= cfg["prior_trend_min"]
        if not cond1:
            continue

        # カップ探索の右端（PL から cup_search_range バー以内）
        cup_right_bound = min(i_pl + cup_search_range, n - cfg["handle_bars_min"] - 3)

        # ③ BC = PL〜cup_right_bound の範囲で実際の最安値バーを確定
        #    （カップ中央付近から探索してカップ形状を確認）
        cup_segment_lows = df["Low"].iloc[i_pl + 1 : cup_right_bound]
        if cup_segment_lows.empty:
            continue
        # BC は PL〜PR の間の最安値バー（後で PR 確定後に再検証）
        i_bc_candidate = int(cup_segment_lows.values.argmin()) + i_pl + 1

        # ④ PR を探索：BC の右側、かつ cup_bars_min を満たす範囲
        pr_start = i_bc_candidate + int(cfg["cup_bars_min"] // 2)
        pr_end = min(cup_right_bound + cfg["handle_bars_max"], n - cfg["handle_bars_min"] - 3)

        for i_pr in range(pr_start, pr_end):
            if i_pr >= n - cfg["handle_bars_min"] - 2:
                break

            # 条件3：カップ長さ
            # Ncup = i_pr - i_pl
            n_cup = i_pr - i_pl
            if n_cup < cfg["cup_bars_min"]:
                continue

            h_pr = df["High"].iloc[i_pr]
            if h_pr <= 0:
                continue

            # 条件9：ピボット有効性
            # Epivot = |H(PR) - H(PL)| / H(PL) <= pivot_tolerance
            e_pivot = abs(h_pr - h_pl) / h_pl
            cond9 = e_pivot <= cfg["pivot_tolerance"]
            if not cond9:
                continue

            # 条件7：左右リムの対称性
            # Srim = |H(PL) - H(PR)| / H(PL) <= rim_symmetry_max
            s_rim = abs(h_pl - h_pr) / h_pl
            cond7 = s_rim <= cfg["rim_symmetry_max"]
            if not cond7:
                continue

            # BC の確定：PL〜PR 間の実際の最安値バー
            cup_inner = df["Low"].iloc[i_pl + 1 : i_pr]
            if cup_inner.empty:
                continue
            i_bc = int(cup_inner.values.argmin()) + i_pl + 1
            l_bc = df["Low"].iloc[i_bc]

            # 条件2：カップ深さ
            # Dcup = (H(PL) - L(BC)) / H(PL)
            d_cup = (h_pl - l_bc) / h_pl
            cond2 = cfg["cup_depth_min"] <= d_cup <= cfg["cup_depth_max"]
            if not cond2:
                continue

            # カップ期の平均出来高
            v_cup_avg = df["Volume"].iloc[i_pl : i_pr + 1].mean()

            # ⑤ BH を確定：PR〜(PR + handle_bars_max) の範囲の最安値バー
            handle_end = min(i_pr + cfg["handle_bars_max"] + 1, n - 2)
            if i_pr + cfg["handle_bars_min"] >= handle_end:
                continue
            handle_segment = df["Low"].iloc[i_pr + 1 : handle_end]
            if handle_segment.empty:
                continue
            i_bh = int(handle_segment.values.argmin()) + i_pr + 1
            l_bh = df["Low"].iloc[i_bh]

            # 条件4：ハンドル深さ
            # Dhandle = (H(PR) - L(BH)) / H(PR)
            d_handle = (h_pr - l_bh) / h_pr
            cond4 = cfg["handle_depth_min"] <= d_handle <= cfg["handle_depth_max"]

            # 条件5：ハンドル長さ
            n_handle = i_bh - i_pr
            cond5 = cfg["handle_bars_min"] <= n_handle <= cfg["handle_bars_max"]

            # 条件6：ハンドル中の出来高縮小
            # V_handle_avg < V_cup_avg * handle_volume_ratio
            v_handle_avg = df["Volume"].iloc[i_pr : i_bh + 1].mean()
            cond6 = v_handle_avg < v_cup_avg * cfg["handle_volume_ratio"]

            # 条件8：50 日移動平均との関係
            # C(BH) >= MA50(BH)
            ma50_val = float(ma50.iloc[i_bh]) if not np.isnan(ma50.iloc[i_bh]) else 0.0
            close_bh = float(df["Close"].iloc[i_bh])
            cond8 = close_bh >= ma50_val if ma50_val > 0 else True

            # ピボット（PV）= BH の翌営業日
            i_pv = i_bh + 1
            if i_pv >= n:
                break

            # 条件10：ブレイクアウト時の出来高急増
            # V(PV) >= breakout_volume_ratio * V20(PV)
            v_pv = float(df["Volume"].iloc[i_pv])
            v20_val = float(v20.iloc[i_pv]) if not np.isnan(v20.iloc[i_pv]) else 0.0
            cond10 = v_pv >= cfg["breakout_volume_ratio"] * v20_val if v20_val > 0 else False

            cond_flags = {
                "cond1_prior_trend": cond1,
                "cond2_cup_depth": cond2,
                "cond3_cup_bars": n_cup >= cfg["cup_bars_min"],
                "cond4_handle_depth": cond4,
                "cond5_handle_bars": cond5,
                "cond6_volume": cond6,
                "cond7_symmetry": cond7,
                "cond8_ma50": cond8,
                "cond9_pivot": cond9,
                "cond10_breakout_vol": cond10,
            }

            candidate = CWHCandidate(
                ticker=ticker,
                idx_pl=i_pl,
                idx_bc=i_bc,
                idx_pr=i_pr,
                idx_bh=i_bh,
                idx_pv=i_pv,
                r_prior=r_prior,
                d_cup=d_cup,
                n_cup=n_cup,
                d_handle=d_handle,
                n_handle=n_handle,
                v_handle_avg=v_handle_avg,
                v_cup_avg=v_cup_avg,
                s_rim=s_rim,
                ma50_bh=ma50_val,
                close_bh=close_bh,
                e_pivot=e_pivot,
                v_pv=v_pv,
                v20_pv=v20_val,
                cond_flags=cond_flags,
            )

            if candidate.all_pass:
                candidates.append(candidate)

    logger.info("%s: %d 候補検出", ticker, len(candidates))
    return candidates
