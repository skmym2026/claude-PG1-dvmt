"""
CWH/C-NH 統合スクリーニング・バックテストシステム v3.3（最適化版）

主要変更点（v3.2 → v3.3）:
  ★修正1: cup_bars_min   35 → 30日    （品質と検出率の中間値）
  ★修正2: prior_trend_min 0.30 → 0.28  （現実的な緩和）
  ★修正3: trailing_stop_min_gain 0.05 → 0.15（TP1到達後のみ発動）
  ★修正4: max_hold_bars  120 → 90日    （長期保有リスク低減）

実行例:
  python cwh_v33.py --mode=real --stocks=5253.T,6228.T,5586.T
  python cwh_v33.py --mode=demo
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# yfinance はオプション（インストールされていない場合はデモデータを使用）
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

# ============================================================
# v3.3 確定パラメータ
# ============================================================
_SEED_RANGE = 2**31  # 32-bit signed integer max; used as seed range for the RNG

CONFIG_V33 = {
    # ストップロス
    "stop_loss_ratio":              0.92,   # SL = P_entry × 0.92  (-8%)

    # 分割利確
    "take_profit_ratio_1":          1.15,   # TP1 = P_entry × 1.15 (+15%)
    "take_profit_qty_1":            0.50,   # TP1 で 50% 売却
    "take_profit_ratio_2":          1.28,   # TP2 = P_entry × 1.28 (+28%)

    # 保有期間
    "max_hold_bars":                90,     # ★修正4: 120 → 90日

    # トレーリングストップ（TP1 到達後のみ発動）
    "trailing_stop_ratio":          0.92,   # TS = P_max × 0.92
    "trailing_stop_min_gain":       0.15,   # ★修正3: 0.05 → 0.15

    # カップ形状（CWH）
    "cwh_cup_bars_min":             30,     # ★修正1: 35 → 30日
    "cwh_cup_bars_max":             200,
    "cwh_cup_depth_min":            0.15,
    "cwh_cup_depth_max":            0.35,
    "cwh_prior_trend_min":          0.28,   # ★修正2: 0.30 → 0.28
    "cwh_prior_trend_bars":         40,

    # ハンドル（CWH）
    "cwh_handle_bars_min":          4,
    "cwh_handle_bars_max":          25,
    "cwh_handle_depth_min":         0.05,
    "cwh_handle_depth_max":         0.15,

    # カップ形状（C-NH）
    "cnh_cup_bars_min":             20,
    "cnh_cup_bars_max":             200,
    "cnh_cup_depth_min":            0.15,
    "cnh_cup_depth_max":            0.40,
    "cnh_prior_trend_min":          0.20,
    "cnh_prior_trend_bars":         60,
    "cnh_handle_bars_max":          3,      # ノーハンドル判定

    # 共通
    "rim_symmetry_max":             0.15,
    "bo_volume_ratio_cwh":          1.5,    # CWH BO 出来高倍率
    "bo_volume_ratio_cnh":          1.3,    # C-NH BO 出来高倍率
    "vol_ma_period":                20,     # 出来高移動平均期間
    "price_ma_period":              50,     # 価格移動平均期間
}


# ============================================================
# データ取得
# ============================================================

def get_stock_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    yfinance から実データを取得する。
    取得失敗時は None を返す（呼び出し元でデモデータへフォールバック）。

    Args:
        ticker: 銘柄コード（"5253.T" など東証サフィックス付き）
        start:  開始日（"YYYY-MM-DD"）
        end:    終了日（"YYYY-MM-DD"）

    Returns:
        OHLCV の DataFrame、または None
    """
    if not _YFINANCE_AVAILABLE:
        print(f"Warning: yfinance が未インストールです。デモデータを使用します。")
        return None

    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"Warning: {ticker} のデータが空です。デモデータを使用します。")
            return None
        # カラム名を統一
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:
        print(f"Warning: {ticker} 取得失敗（{e}）。デモデータを使用します。")
        return None


def generate_demo_data(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    テスト・デモ用のサンプル株価データを生成する。

    Args:
        n_bars: 生成するバー数
        seed:   乱数シード

    Returns:
        OHLCV の DataFrame
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=n_bars)
    close = np.cumprod(1 + rng.normal(0.0003, 0.015, n_bars)) * 1000.0
    high = close * (1 + rng.uniform(0.001, 0.02, n_bars))
    low = close * (1 - rng.uniform(0.001, 0.02, n_bars))
    open_ = low + rng.uniform(0, 1, n_bars) * (high - low)
    volume = rng.integers(100_000, 2_000_000, n_bars).astype(float)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


# ============================================================
# 出口判定ロジック（v3.3）
# ============================================================

def check_exit_v33(
    closes: np.ndarray,
    lows: np.ndarray,
    buy_price: float,
    cfg: dict = CONFIG_V33,
) -> dict:
    """
    v3.3 出口判定ロジック（分割 TP + 条件付き TS）

    優先順位:
      1. SL : L(t) <= P_entry × SL_ratio
      2. TS : L(t) <= P_max(t) × TS_ratio  かつ  P_max(t) > P_entry × TP1_ratio
      3. TP2: C(t) >= P_entry × TP2_ratio
      4. EoP: t > max_hold_bars

    Args:
        closes:    エントリー翌日以降の終値配列
        lows:      エントリー翌日以降の安値配列
        buy_price: エントリー価格
        cfg:       パラメータ辞書（デフォルト CONFIG_V33）

    Returns:
        {
          "exit_bar":    int,    # 出口バーインデックス（0 始まり）
          "exit_reason": str,    # "SL" / "TS" / "TP2" / "EoP"
          "exit_price":  float,  # 出口価格
          "tp1_reached": bool,   # TP1 到達フラグ
          "return_pct":  float,  # リターン（%）
        }
    """
    sl_price  = buy_price * cfg["stop_loss_ratio"]
    tp1_price = buy_price * cfg["take_profit_ratio_1"]
    tp2_price = buy_price * cfg["take_profit_ratio_2"]
    ts_ratio  = cfg["trailing_stop_ratio"]
    max_bars  = cfg["max_hold_bars"]

    tp1_reached = False
    p_max = buy_price

    n = min(len(closes), max_bars)

    for i in range(n):
        c = closes[i]
        lo = lows[i]

        # TP1 到達チェック（フラグ更新のみ、部分売却は呼び出し元で処理）
        if c >= tp1_price:
            tp1_reached = True

        # 高値更新
        if c > p_max:
            p_max = c

        # 優先順位 1: SL
        if lo <= sl_price:
            exit_price = sl_price
            return {
                "exit_bar":    i,
                "exit_reason": "SL",
                "exit_price":  exit_price,
                "tp1_reached": tp1_reached,
                "return_pct":  (exit_price / buy_price - 1) * 100,
            }

        # 優先順位 2: TS（TP1 到達後のみ）
        if tp1_reached:
            ts_threshold = p_max * ts_ratio
            if lo <= ts_threshold:
                exit_price = ts_threshold
                return {
                    "exit_bar":    i,
                    "exit_reason": "TS",
                    "exit_price":  exit_price,
                    "tp1_reached": tp1_reached,
                    "return_pct":  (exit_price / buy_price - 1) * 100,
                }

        # 優先順位 3: TP2
        if c >= tp2_price:
            exit_price = tp2_price
            return {
                "exit_bar":    i,
                "exit_reason": "TP2",
                "exit_price":  exit_price,
                "tp1_reached": tp1_reached,
                "return_pct":  (exit_price / buy_price - 1) * 100,
            }

    # 優先順位 4: EoP（保有期間終了）
    exit_price = closes[n - 1] if n > 0 else buy_price
    return {
        "exit_bar":    n - 1,
        "exit_reason": "EoP",
        "exit_price":  exit_price,
        "tp1_reached": tp1_reached,
        "return_pct":  (exit_price / buy_price - 1) * 100,
    }


# ============================================================
# CWH/C-NH パターン検出
# ============================================================

def detect_cup_pattern(df: pd.DataFrame, cfg: dict = CONFIG_V33) -> list:
    """
    CWH および C-NH パターンを検出する。

    Args:
        df:  OHLCV DataFrame（インデックスは日付）
        cfg: パラメータ辞書

    Returns:
        検出されたパターンのリスト。各要素は dict:
        {
          "pattern":     "CWH" or "C-NH",
          "entry_bar":   int,    # エントリーバーインデックス
          "entry_date":  date,
          "pivot_price": float,
          "cup_depth":   float,
          "prior_trend": float,
          "bo_volume_ok": bool,
        }
    """
    closes  = df["Close"].values
    highs   = df["High"].values
    lows    = df["Low"].values
    volumes = df["Volume"].values
    n       = len(closes)

    vol_ma = pd.Series(volumes).rolling(cfg["vol_ma_period"]).mean().values

    patterns = []

    # カップ検索の開始インデックス（十分なデータが必要）
    min_lookback = max(
        cfg["cwh_cup_bars_max"],
        cfg["cwh_prior_trend_bars"],
        cfg["cnh_cup_bars_max"],
        cfg["cnh_prior_trend_bars"],
    )

    for i in range(min_lookback, n - 1):
        # --- CWH スクリーニング ---
        for cup_len in range(cfg["cwh_cup_bars_min"], cfg["cwh_cup_bars_max"] + 1):
            cup_start = i - cup_len
            if cup_start < cfg["cwh_prior_trend_bars"]:
                break

            left_pivot  = highs[cup_start]
            cup_low_idx = np.argmin(lows[cup_start:i]) + cup_start
            cup_low     = lows[cup_low_idx]
            right_rim   = highs[i]

            # カップ深さ
            cup_depth = (left_pivot - cup_low) / left_pivot
            if not (cfg["cwh_cup_depth_min"] <= cup_depth <= cfg["cwh_cup_depth_max"]):
                continue

            # リム対称性
            rim_diff = abs(left_pivot - right_rim) / left_pivot
            if rim_diff > cfg["rim_symmetry_max"]:
                continue

            # 事前上昇
            prior_start = cup_start - cfg["cwh_prior_trend_bars"]
            prior_low   = np.min(lows[prior_start:cup_start])
            prior_trend = (left_pivot - prior_low) / prior_low
            if prior_trend < cfg["cwh_prior_trend_min"]:
                continue

            # BO 出来高
            bo_vol_ok = (
                vol_ma[i] > 0 and
                volumes[i] >= vol_ma[i] * cfg["bo_volume_ratio_cwh"]
            )

            # ハンドル検索（CWH）
            for h_len in range(cfg["cwh_handle_bars_min"], cfg["cwh_handle_bars_max"] + 1):
                h_start = i - h_len
                if h_start <= cup_start:
                    break
                h_high = np.max(highs[h_start:i])
                h_low  = np.min(lows[h_start:i])
                h_depth = (h_high - h_low) / h_high
                if not (cfg["cwh_handle_depth_min"] <= h_depth <= cfg["cwh_handle_depth_max"]):
                    continue

                # CWH 検出
                pivot_price = left_pivot
                patterns.append({
                    "pattern":      "CWH",
                    "entry_bar":    i + 1,
                    "entry_date":   df.index[i + 1] if i + 1 < n else df.index[i],
                    "pivot_price":  pivot_price,
                    "cup_depth":    cup_depth,
                    "prior_trend":  prior_trend,
                    "bo_volume_ok": bo_vol_ok,
                })
                break  # ハンドル長さはひとつ見つかれば OK

        # --- C-NH スクリーニング ---
        for cup_len in range(cfg["cnh_cup_bars_min"], cfg["cnh_cup_bars_max"] + 1):
            cup_start = i - cup_len
            if cup_start < cfg["cnh_prior_trend_bars"]:
                break

            left_pivot  = highs[cup_start]
            cup_low_idx = np.argmin(lows[cup_start:i]) + cup_start
            cup_low     = lows[cup_low_idx]
            right_rim   = highs[i]

            cup_depth = (left_pivot - cup_low) / left_pivot
            if not (cfg["cnh_cup_depth_min"] <= cup_depth <= cfg["cnh_cup_depth_max"]):
                continue

            rim_diff = abs(left_pivot - right_rim) / left_pivot
            if rim_diff > cfg["rim_symmetry_max"]:
                continue

            prior_start = cup_start - cfg["cnh_prior_trend_bars"]
            prior_low   = np.min(lows[prior_start:cup_start])
            prior_trend = (left_pivot - prior_low) / prior_low
            if prior_trend < cfg["cnh_prior_trend_min"]:
                continue

            # ノーハンドル判定（直近 handle_bars_max 日以内に深いハンドルなし）
            recent_high = np.max(highs[i - cfg["cnh_handle_bars_max"]:i + 1])
            recent_low  = np.min(lows[i - cfg["cnh_handle_bars_max"]:i + 1])
            if recent_high > 0:
                recent_retrace = (recent_high - recent_low) / recent_high
                if recent_retrace > cfg["cwh_handle_depth_max"]:
                    continue  # 深い押し入りがある = ハンドルあり → C-NH 除外

            bo_vol_ok = (
                vol_ma[i] > 0 and
                volumes[i] >= vol_ma[i] * cfg["bo_volume_ratio_cnh"]
            )

            patterns.append({
                "pattern":      "C-NH",
                "entry_bar":    i + 1,
                "entry_date":   df.index[i + 1] if i + 1 < n else df.index[i],
                "pivot_price":  left_pivot,
                "cup_depth":    cup_depth,
                "prior_trend":  prior_trend,
                "bo_volume_ok": bo_vol_ok,
            })
            break

    return patterns


# ============================================================
# バックテスト実行
# ============================================================

def run_backtest_v33(
    df: pd.DataFrame,
    ticker: str = "Unknown",
    cfg: dict = CONFIG_V33,
) -> dict:
    """
    v3.3 バックテストを実行する。

    Args:
        df:     OHLCV DataFrame
        ticker: 銘柄コード（レポート用）
        cfg:    パラメータ辞書

    Returns:
        バックテスト結果 dict
    """
    patterns = detect_cup_pattern(df, cfg)
    closes  = df["Close"].values
    lows    = df["Low"].values
    n       = len(closes)

    trades = []
    for p in patterns:
        entry_bar = p["entry_bar"]
        if entry_bar >= n:
            continue

        buy_price = closes[entry_bar]
        future_closes = closes[entry_bar + 1:]
        future_lows   = lows[entry_bar + 1:]

        if len(future_closes) == 0:
            continue

        result = check_exit_v33(future_closes, future_lows, buy_price, cfg)
        trades.append({
            "ticker":       ticker,
            "pattern":      p["pattern"],
            "entry_date":   p["entry_date"],
            "entry_price":  buy_price,
            "exit_reason":  result["exit_reason"],
            "exit_price":   result["exit_price"],
            "return_pct":   result["return_pct"],
            "tp1_reached":  result["tp1_reached"],
            "bo_volume_ok": p["bo_volume_ok"],
            "cup_depth":    p["cup_depth"],
            "prior_trend":  p["prior_trend"],
        })

    # サマリー統計
    if trades:
        returns = [t["return_pct"] for t in trades]
        wins    = [r for r in returns if r > 0]
        tp1_cnt = sum(1 for t in trades if t["tp1_reached"])
        tp2_cnt = sum(1 for t in trades if t["exit_reason"] == "TP2")
        bo_ok   = sum(1 for t in trades if t["bo_volume_ok"])

        equity = np.cumprod([1 + r / 100 for r in returns])
        drawdowns = []
        peak = equity[0]
        for e in equity:
            if e > peak:
                peak = e
            drawdowns.append((peak - e) / peak)
        mdd = max(drawdowns) * 100 if drawdowns else 0.0

        summary = {
            "ticker":        ticker,
            "n_patterns":    len(patterns),
            "n_trades":      len(trades),
            "win_rate_pct":  len(wins) / len(trades) * 100,
            "avg_return_pct": np.mean(returns),
            "cum_return_pct": (equity[-1] - 1) * 100,
            "mdd_pct":       mdd,
            "tp1_count":     tp1_cnt,
            "tp2_count":     tp2_cnt,
            "bo_vol_pass_pct": bo_ok / len(trades) * 100,
        }
    else:
        summary = {
            "ticker":        ticker,
            "n_patterns":    len(patterns),
            "n_trades":      0,
            "win_rate_pct":  0.0,
            "avg_return_pct": 0.0,
            "cum_return_pct": 0.0,
            "mdd_pct":       0.0,
            "tp1_count":     0,
            "tp2_count":     0,
            "bo_vol_pass_pct": 0.0,
        }

    return {"trades": trades, "summary": summary}


# ============================================================
# レポート出力
# ============================================================

def print_report(results: list) -> None:
    """バックテスト結果を標準出力に表示する。"""
    print("\n" + "=" * 70)
    print("  CWH/C-NH v3.3 バックテスト結果")
    print("=" * 70)

    all_trades = []
    for r in results:
        s = r["summary"]
        print(f"\n【{s['ticker']}】")
        print(f"  パターン検出数 : {s['n_patterns']}")
        print(f"  トレード数     : {s['n_trades']}")
        if s["n_trades"] > 0:
            print(f"  勝率           : {s['win_rate_pct']:.1f}%")
            print(f"  平均リターン   : {s['avg_return_pct']:.2f}%")
            print(f"  累積リターン   : {s['cum_return_pct']:.2f}%")
            print(f"  MDD            : {s['mdd_pct']:.2f}%")
            print(f"  TP1 到達件数   : {s['tp1_count']}")
            print(f"  TP2 到達件数   : {s['tp2_count']}")
            print(f"  BO 出来高通過率: {s['bo_vol_pass_pct']:.1f}%")
        all_trades.extend(r["trades"])

    # 全体サマリー
    if all_trades:
        returns = [t["return_pct"] for t in all_trades]
        wins    = [r for r in returns if r > 0]
        equity  = np.cumprod([1 + r / 100 for r in returns])
        peak = equity[0]
        drawdowns = []
        for e in equity:
            if e > peak:
                peak = e
            drawdowns.append((peak - e) / peak)
        mdd = max(drawdowns) * 100 if drawdowns else 0.0

        n_detected = sum(1 for r in results if r["summary"]["n_trades"] > 0)
        n_total    = len(results)

        print("\n" + "=" * 70)
        print("  全体サマリー")
        print("=" * 70)
        print(f"  検出率       : {n_detected}/{n_total} 銘柄")
        print(f"  全トレード数 : {len(all_trades)}")
        print(f"  全体勝率     : {len(wins) / len(all_trades) * 100:.1f}%")
        print(f"  全体平均 RET : {np.mean(returns):.2f}%")
        print(f"  全体累積 RET : {(equity[-1] - 1) * 100:.2f}%")
        print(f"  全体 MDD     : {mdd:.2f}%")
        tp1_total = sum(t["tp1_reached"] for t in all_trades)
        tp2_total = sum(1 for t in all_trades if t["exit_reason"] == "TP2")
        print(f"  TP1 到達件数 : {tp1_total}")
        print(f"  TP2 到達件数 : {tp2_total}")
    print()


# ============================================================
# CLI エントリーポイント
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CWH/C-NH v3.3 バックテストシステム"
    )
    parser.add_argument(
        "--mode",
        choices=["real", "demo"],
        default="demo",
        help="データソース: real=yfinance実データ, demo=生成データ（デフォルト: demo）",
    )
    parser.add_argument(
        "--stocks",
        default="5253.T,6228.T,5586.T,5242.T,5885.T,4894.T,5803.T,7011.T,6586.T",
        help="カンマ区切りの銘柄コード（例: 5253.T,6228.T）",
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="バックテスト開始日（YYYY-MM-DD）",
    )
    parser.add_argument(
        "--end",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="バックテスト終了日（YYYY-MM-DD）",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.stocks.split(",") if t.strip()]
    results = []

    print(f"\nCWH/C-NH v3.3 バックテスト開始")
    print(f"  モード  : {args.mode}")
    print(f"  銘柄数  : {len(tickers)}")
    print(f"  期間    : {args.start} ～ {args.end}")

    for ticker in tickers:
        print(f"  処理中  : {ticker} ...", end=" ", flush=True)

        df = None
        if args.mode == "real":
            df = get_stock_data(ticker, args.start, args.end)

        if df is None:
            seed = abs(hash(ticker)) % _SEED_RANGE
            df = generate_demo_data(n_bars=500, seed=seed)
            print("(デモデータ使用)", end=" ")

        result = run_backtest_v33(df, ticker=ticker)
        results.append(result)
        print(f"→ パターン{result['summary']['n_patterns']}件 / トレード{result['summary']['n_trades']}件")

    print_report(results)


if __name__ == "__main__":
    main()
