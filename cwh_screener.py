"""
cwh_screener.py - CWH パターン検出エンジン（条件1〜10）

前回指示書の判定条件1〜10 に基づき、OHLCV データから
Cup with Handle パターンの候補を検出する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CWHCandidate:
    """CWH パターン候補の検出結果を格納するデータクラス。"""

    ticker: str
    # キーポイントのインデックス位置
    idx_pl: int        # PL：左リム
    idx_bc: int        # BC：カップ底
    idx_pr: int        # PR：右リム
    idx_bh: int        # BH：ハンドル底
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

    @property
    def pivot_price(self) -> float:
        """ピボット価格（右リム高値）は呼び出し元で付与する。"""
        return 0.0


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """単純移動平均を計算する。"""
    return series.rolling(window=window, min_periods=window).mean()


def detect_cwh_candidates(
    df: pd.DataFrame,
    ticker: str,
    cfg: dict,
) -> list[CWHCandidate]:
    """OHLCV データから CWH パターン候補を全件検出する。

    Args:
        df: OHLCV DataFrame（列: Open, High, Low, Close, Volume）
        ticker: 銘柄コード
        cfg: cwh_config.yaml の screener セクション

    Returns:
        CWHCandidate のリスト
    """
    if len(df) < cfg["cup_bars_min"] + cfg["handle_bars_max"] + cfg["prior_trend_bars"]:
        return []

    # 移動平均の事前計算
    ma50: pd.Series = _rolling_mean(df["Close"], cfg["ma_period"])
    v20: pd.Series = _rolling_mean(df["Volume"], cfg["breakout_volume_period"])

    candidates: list[CWHCandidate] = []
    n = len(df)

    # 左リム（PL）を探索
    for i_pl in range(cfg["prior_trend_bars"], n - cfg["cup_bars_min"] - cfg["handle_bars_max"] - 5):
        h_pl = df["High"].iloc[i_pl]

        # 条件1：事前上昇トレンド確認
        # Rprior = (C(PL) - min_C_prior) / min_C_prior >= prior_trend_min
        prior_start = max(0, i_pl - cfg["prior_trend_bars"])
        min_prior = df["Close"].iloc[prior_start:i_pl].min()
        if min_prior <= 0:
            continue
        r_prior = (df["Close"].iloc[i_pl] - min_prior) / min_prior
        cond1 = r_prior >= cfg["prior_trend_min"]
        if not cond1:
            continue

        # カップ底（BC）を探索：PL の右側
        cup_end_max = min(i_pl + cfg.get("cup_search_range", 120), n - cfg["handle_bars_max"] - 5)
        for i_bc in range(i_pl + int(cfg["cup_bars_min"] // 2), cup_end_max):
            l_bc = df["Low"].iloc[i_bc]

            # 条件2：カップ深さ
            # Dcup = (H(PL) - L(BC)) / H(PL)
            if h_pl <= 0:
                continue
            d_cup = (h_pl - l_bc) / h_pl
            cond2 = cfg["cup_depth_min"] <= d_cup <= cfg["cup_depth_max"]
            if not cond2:
                continue

            # 右リム（PR）を探索：BC の右側
            for i_pr in range(i_bc + int(cfg["cup_bars_min"] // 2), cup_end_max + cfg["handle_bars_max"]):
                if i_pr >= n - cfg["handle_bars_min"]:
                    break

                # 条件3：カップ長さ
                # Ncup = D(PR) - D(PL)
                n_cup = i_pr - i_pl
                cond3 = n_cup >= cfg["cup_bars_min"]
                if not cond3:
                    continue

                h_pr = df["High"].iloc[i_pr]

                # 条件9：ピボット有効性（右リムが左リム高値に近い）
                # Epivot = |H(PR) - H(PL)| / H(PL) <= pivot_tolerance
                e_pivot = abs(h_pr - h_pl) / h_pl if h_pl > 0 else 1.0
                cond9 = e_pivot <= cfg["pivot_tolerance"]
                if not cond9:
                    continue

                # 条件7：左右リムの対称性
                # Srim = |H(PL) - H(PR)| / H(PL)
                s_rim = abs(h_pl - h_pr) / h_pl if h_pl > 0 else 1.0
                cond7 = s_rim <= cfg["rim_symmetry_max"]
                if not cond7:
                    continue

                # カップ期の平均出来高
                v_cup_avg = df["Volume"].iloc[i_pl:i_pr + 1].mean()

                # ハンドル底（BH）を探索：PR の右側
                for i_bh in range(i_pr + cfg["handle_bars_min"], i_pr + cfg["handle_bars_max"] + 1):
                    if i_bh >= n - 2:
                        break

                    # 条件4：ハンドル深さ
                    # Dhandle = (H(PR) - L(BH)) / H(PR)
                    l_bh = df["Low"].iloc[i_bh]
                    if h_pr <= 0:
                        continue
                    d_handle = (h_pr - l_bh) / h_pr
                    cond4 = cfg["handle_depth_min"] <= d_handle <= cfg["handle_depth_max"]
                    if not cond4:
                        continue

                    # 条件5：ハンドル長さ
                    n_handle = i_bh - i_pr
                    cond5 = cfg["handle_bars_min"] <= n_handle <= cfg["handle_bars_max"]
                    if not cond5:
                        continue

                    # 条件6：ハンドル中の出来高縮小
                    # V_handle_avg < V_cup_avg
                    v_handle_avg = df["Volume"].iloc[i_pr:i_bh + 1].mean()
                    cond6 = v_handle_avg < v_cup_avg * cfg["handle_volume_ratio"]

                    # 条件8：50日移動平均との関係
                    # C(BH) >= MA50(BH)
                    ma50_val = ma50.iloc[i_bh] if not np.isnan(ma50.iloc[i_bh]) else 0.0
                    close_bh = df["Close"].iloc[i_bh]
                    cond8 = close_bh >= ma50_val if ma50_val > 0 else True

                    # ピボット（PV）を探索：BH の翌日
                    i_pv = i_bh + 1
                    if i_pv >= n:
                        break

                    # 条件10：ブレイクアウト時の出来高急増
                    # V(PV) >= 1.5 * V20(PV)
                    v_pv = df["Volume"].iloc[i_pv]
                    v20_val = v20.iloc[i_pv] if not np.isnan(v20.iloc[i_pv]) else 0.0
                    cond10 = v_pv >= cfg["breakout_volume_ratio"] * v20_val if v20_val > 0 else False

                    cond_flags = {
                        "cond1_prior_trend": cond1,
                        "cond2_cup_depth": cond2,
                        "cond3_cup_bars": cond3,
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
