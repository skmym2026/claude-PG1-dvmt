# 2026年1月～3月 実データバックテストエンジン
#
# 使用方法:
#   python backtest_jan_mar_2026.py
#
# 出力ファイル:
#   daily_detection_20260106_20260331.csv
#   monthly_summary_202601-202603.csv
#   survival_rate_analysis.csv
#   tier_performance.csv

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

from cwh_v33 import CWHScreener, CONFIG_V33, check_exit_v33


# ---------------------------------------------------------------------------
# 銘柄ユニバース (フィルター済みサンプル)
# 本番実行時は東証プライム/スタンダード/グロースの全銘柄リストに差し替える
# ---------------------------------------------------------------------------
SAMPLE_TICKERS = [
    # 東証プライム (主要銘柄)
    "7203", "6758", "9984", "8306", "6861", "4063", "9433", "8035",
    "6098", "4519", "6954", "7267", "6501", "8316", "4502", "9432",
    "6702", "7751", "6367", "5108", "8411", "4661", "6869", "4901",
    "3382", "2914", "8801", "8802", "6503", "7011",
    # 東証スタンダード・グロース (中小型成長株)
    "5253", "6228", "5586", "5242", "5885", "4894",
    "5803", "6586", "5301", "3659", "4385", "6095",
]


def get_all_tickers(use_sample: bool = True) -> list:
    """
    東証3市場の全銘柄コードを返す。

    Parameters
    ----------
    use_sample : True の場合はサンプル銘柄リストを返す (開発・テスト用)。
                 False の場合は東証の公式一覧から全銘柄を取得する (本番用)。

    Returns
    -------
    list of str : 銘柄コード (例: ["7203", "6758", ...])
    """
    if use_sample:
        return SAMPLE_TICKERS

    # 本番: J-Quants / 東証公式CSVから取得するロジックをここに実装
    # 例: pd.read_csv("https://www.jpx.co.jp/.../data_j.xls", ...)
    # フィルター: 時価総額 >= 100億円, 流動性 >= 2.0億円
    raise NotImplementedError(
        "本番モードの銘柄取得は未実装です。"
        "J-Quants API または東証公式CSVから取得するロジックを実装してください。"
    )


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------
class BacktestEngine:
    """
    2026年1月6日～3月31日の全銘柄 CWH/C-NH スクリーニングと収益シミュレーション。
    """

    def __init__(
        self,
        start_date: str = "2026-01-06",
        end_date: str = "2026-03-31",
        use_sample_tickers: bool = True,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.config = CONFIG_V33
        self.use_sample_tickers = use_sample_tickers
        self.all_detections: list = []

    # ── 公開 API ──────────────────────────────────────────────────────────

    def run_full_backtest(self) -> pd.DataFrame:
        """
        start_date ～ end_date の全営業日を順番にスクリーニングし、
        検出結果を結合した DataFrame を返す。
        """
        current_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

        all_detections: list = []

        while current_date <= end_date:
            if current_date.weekday() < 5:  # 月曜～金曜
                date_str = current_date.strftime("%Y-%m-%d")
                print(f"[{date_str}] Screening...", flush=True)
                day_results = self.run_daily_screening(date_str)
                if not day_results.empty:
                    all_detections.append(day_results)
                    print(
                        f"  → {len(day_results)} pattern(s) detected.",
                        flush=True,
                    )
            current_date += timedelta(days=1)

        if not all_detections:
            return pd.DataFrame()

        result = pd.concat(all_detections, ignore_index=True)
        self.all_detections = all_detections
        return result

    def run_daily_screening(self, date_str: str) -> pd.DataFrame:
        """
        指定日付での全銘柄スクリーニングを実行し、
        検出パターンの DataFrame を返す。

        Parameters
        ----------
        date_str : "YYYY-MM-DD" 形式の対象日付
        """
        tickers = get_all_tickers(use_sample=self.use_sample_tickers)
        daily_results: list = []

        for ticker in tickers:
            try:
                df = yf.download(
                    f"{ticker}.T",
                    start="2025-10-01",   # 前年データも含める (MA200用)
                    end=date_str,
                    progress=False,
                    auto_adjust=True,
                )

                if df is None or len(df) < self.config["min_data_bars"]:
                    continue

                screener = CWHScreener(df, ticker, self.config)
                patterns = screener.detect_all_patterns()

                for pattern in patterns:
                    if pattern.all_conditions_met:
                        # エントリー価格 = 当日終値 (翌営業日始値に近似)
                        entry_price = float(df["Close"].iloc[-1])
                        daily_results.append({
                            "date": date_str,
                            "ticker": ticker,
                            "pattern_type": pattern.pattern_type,
                            "score": pattern.score,
                            "entry_price": entry_price,
                            "cup_depth": round(pattern.cup_depth, 4),
                            "rim_symmetry": round(pattern.rim_symmetry, 4),
                            "bo_volume_ratio": round(pattern.bo_volume_ratio, 2),
                            "has_handle": pattern.has_handle,
                            "tier": _classify_tier(pattern.score, self.config),
                        })

            except Exception:
                pass  # データ取得失敗時はスキップ

            # API 負荷軽減
            time.sleep(0.05)

        return pd.DataFrame(daily_results)

    # ── 内部ユーティリティ ────────────────────────────────────────────────

    @staticmethod
    def _classify_tier(score: float, config: dict) -> str:
        return _classify_tier(score, config)


# ---------------------------------------------------------------------------
# _classify_tier: スコアから Tier を返す
# ---------------------------------------------------------------------------
def _classify_tier(score: float, config: dict = None) -> str:
    if config is None:
        config = CONFIG_V33
    if score >= config["tier1_score"]:
        return "I"
    elif score >= config["tier2_score"]:
        return "II"
    elif score >= config["tier3_score"]:
        return "III"
    else:
        return "IV"


# ---------------------------------------------------------------------------
# backtest_survival_rate: 検出銘柄の生存率シミュレーション
# ---------------------------------------------------------------------------
def backtest_survival_rate(
    detection_df: pd.DataFrame,
    config: dict = None,
    fetch_end: str = "2026-07-31",
) -> pd.DataFrame:
    """
    検出銘柄について、検出日から最大 max_hold_bars 営業日後まで追跡し、
    TP1/TP2/SL/TS/EoP のいずれに到達したかを記録する。

    Parameters
    ----------
    detection_df : run_daily_screening / run_full_backtest の出力 DataFrame
    config       : パラメータ辞書。省略時は CONFIG_V33
    fetch_end    : yfinance データ取得の終了日 (max_hold_bars をカバーできる日付)

    Returns
    -------
    pd.DataFrame : 個別銘柄の成績
    """
    if config is None:
        config = CONFIG_V33

    survival_results: list = []

    for _, row in detection_df.iterrows():
        ticker = row["ticker"]
        entry_date = row["date"]
        entry_price = float(row["entry_price"])

        try:
            df_future = yf.download(
                f"{ticker}.T",
                start=entry_date,
                end=fetch_end,
                progress=False,
                auto_adjust=True,
            )

            if df_future is None or len(df_future) < 2:
                continue

            close_arr = df_future["Close"].values.astype(float)
            low_arr = df_future["Low"].values.astype(float)
            dates_arr = df_future.index.to_list()

            exit_result = check_exit_v33(
                close_arr,
                low_arr,
                entry_price,
                config,
                entry_date=entry_date,
                dates=dates_arr,
            )

            exit_price = exit_result["price"]
            exit_date = exit_result["date"]
            hold_days = (
                (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
                if exit_date is not None
                else config["max_hold_bars"]
            )

            return_pct = (exit_price - entry_price) / entry_price * 100

            survival_results.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_reason": exit_result["reason"],
                "return_pct": round(return_pct, 2),
                "hold_days": hold_days,
                "tier": row.get("tier", _classify_tier(row["score"], config)),
                "score": row["score"],
                "pattern_type": row.get("pattern_type", ""),
            })

        except Exception:
            pass

        time.sleep(0.05)

    return pd.DataFrame(survival_results)


# ---------------------------------------------------------------------------
# calculate_performance_summary: Tier別・月別の成績集計
# ---------------------------------------------------------------------------
def calculate_performance_summary(survival_df: pd.DataFrame) -> tuple:
    """
    Tier別・月別の成績を集計して (tier_stats, monthly_stats) を返す。

    Parameters
    ----------
    survival_df : backtest_survival_rate の出力 DataFrame

    Returns
    -------
    tuple:
        tier_stats    (pd.DataFrame) : Tier別集計
        monthly_stats (pd.DataFrame) : 月別集計
    """
    if survival_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Tier別集計
    tier_stats = (
        survival_df.groupby("tier")
        .agg(
            count=("ticker", "count"),
            mean_return=("return_pct", "mean"),
            std_return=("return_pct", "std"),
            min_return=("return_pct", "min"),
            max_return=("return_pct", "max"),
            mean_hold_days=("hold_days", "mean"),
            win_rate=("return_pct", lambda x: (x > 0).mean() * 100),
        )
        .round(2)
    )

    # 出口理由の割合 (Tier別)
    exit_pivot = (
        survival_df.groupby(["tier", "exit_reason"])
        .size()
        .unstack(fill_value=0)
    )
    tier_stats = tier_stats.join(exit_pivot, how="left")

    # 月別集計
    survival_df = survival_df.copy()
    survival_df["entry_month"] = pd.to_datetime(survival_df["entry_date"]).dt.month
    monthly_stats = (
        survival_df.groupby("entry_month")
        .agg(
            count=("ticker", "count"),
            mean_return=("return_pct", "mean"),
            mean_hold_days=("hold_days", "mean"),
            win_rate=("return_pct", lambda x: (x > 0).mean() * 100),
        )
        .round(2)
    )

    return tier_stats, monthly_stats


# ---------------------------------------------------------------------------
# simulate_investment: 投資シミュレーション (3戦略)
# ---------------------------------------------------------------------------
def simulate_investment(
    survival_df: pd.DataFrame,
    initial_capital: float = 10_000_000,
) -> pd.DataFrame:
    """
    3つの投資戦略のシミュレーションを実行して結果 DataFrame を返す。

    戦略1: Tier I のみ
    戦略2: Tier I+II (推奨)
    戦略3: 全 Tier

    Parameters
    ----------
    survival_df     : backtest_survival_rate の出力 DataFrame
    initial_capital : 初期資金 (円)

    Returns
    -------
    pd.DataFrame : 各戦略の最終資産・ROI
    """
    if survival_df.empty:
        return pd.DataFrame()

    strategies = {
        "Strategy1_TierI": {
            "tiers": ["I"],
            "weights": {"I": 1.0},
        },
        "Strategy2_TierI_II": {
            "tiers": ["I", "II"],
            "weights": {"I": 0.60, "II": 0.40},
        },
        "Strategy3_All": {
            "tiers": ["I", "II", "III", "IV"],
            "weights": {"I": 0.60, "II": 0.25, "III": 0.10, "IV": 0.05},
        },
    }

    results = []
    for strategy_name, cfg in strategies.items():
        subset = survival_df[survival_df["tier"].isin(cfg["tiers"])].copy()
        if subset.empty:
            results.append({
                "strategy": strategy_name,
                "trade_count": 0,
                "final_capital": initial_capital,
                "roi_pct": 0.0,
            })
            continue

        capital = initial_capital
        tier_counts = subset["tier"].value_counts().to_dict()
        for _, row in subset.iterrows():
            tier = row["tier"]
            weight = cfg["weights"].get(tier, 0.0)
            position = capital * weight / max(tier_counts.get(tier, 1), 1)
            pnl = position * (row["return_pct"] / 100)
            capital += pnl

        roi_pct = (capital - initial_capital) / initial_capital * 100
        results.append({
            "strategy": strategy_name,
            "trade_count": len(subset),
            "final_capital": round(capital, 0),
            "roi_pct": round(roi_pct, 2),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("CWH/C-NH v3.3 バックテスト — 2026年1月～3月")
    print("=" * 60)

    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ── Phase 1: 全銘柄スクリーニング ─────────────────────────────────────
    print("\n[Phase 1] 全銘柄スクリーニング開始 ...")
    engine = BacktestEngine(
        start_date="2026-01-06",
        end_date="2026-03-31",
        use_sample_tickers=True,   # 本番: False に切り替え
    )

    detection_df = engine.run_full_backtest()

    detection_csv = os.path.join(OUTPUT_DIR, "daily_detection_20260106_20260331.csv")
    if not detection_df.empty:
        detection_df.to_csv(detection_csv, index=False, encoding="utf-8-sig")
        print(f"\n  → 検出ログを保存: {detection_csv}")
        print(f"  → 総検出数: {len(detection_df)} パターン")
    else:
        print("  → 検出パターンなし")

    # ── Phase 2: 生存率シミュレーション ───────────────────────────────────
    if not detection_df.empty:
        print("\n[Phase 2] 生存率シミュレーション開始 ...")
        survival_df = backtest_survival_rate(detection_df)

        survival_csv = os.path.join(OUTPUT_DIR, "survival_rate_analysis.csv")
        survival_df.to_csv(survival_csv, index=False, encoding="utf-8-sig")
        print(f"  → 生存率分析を保存: {survival_csv}")

        tier_stats, monthly_stats = calculate_performance_summary(survival_df)

        tier_csv = os.path.join(OUTPUT_DIR, "tier_performance.csv")
        tier_stats.to_csv(tier_csv, encoding="utf-8-sig")
        print(f"  → Tier別成績を保存: {tier_csv}")

        monthly_csv = os.path.join(OUTPUT_DIR, "monthly_summary_202601-202603.csv")
        monthly_stats.to_csv(monthly_csv, encoding="utf-8-sig")
        print(f"  → 月別集計を保存: {monthly_csv}")

        # ── Phase 3: 投資シミュレーション ─────────────────────────────────
        print("\n[Phase 3] 投資シミュレーション開始 ...")
        sim_df = simulate_investment(survival_df, initial_capital=10_000_000)
        print(sim_df.to_string(index=False))

        # サマリー表示
        print("\n" + "=" * 60)
        print("バックテスト完了サマリー")
        print("=" * 60)
        if not survival_df.empty:
            overall_win_rate = (survival_df["return_pct"] > 0).mean() * 100
            overall_avg_return = survival_df["return_pct"].mean()
            print(f"  全体勝率          : {overall_win_rate:.1f}%")
            print(f"  平均リターン      : {overall_avg_return:+.2f}%")
            print(f"  総トレード数      : {len(survival_df)}")
            print(f"  Tier I 生存率     : "
                  f"{(survival_df[survival_df['tier']=='I']['exit_reason']!='SL').mean()*100:.1f}%")
    else:
        print("\n検出銘柄がないため Phase 2/3 をスキップします。")

    print("\n✅ バックテスト終了")
