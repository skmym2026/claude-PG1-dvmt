# CWH/C-NH Pattern Screener v3.3
#
# 変更履歴:
#   v3.1: max_hold_bars: 120, 分割TP導入, rim_symmetry_max: 0.15, yfinance実データ切替
#   v3.2: 東証3市場対応, Tier分類追加
#   v3.3: スクリーニング精度向上, 出口判定強化

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG_V33: v3.3 確定パラメータ設定
# ---------------------------------------------------------------------------
CONFIG_V33 = {
    # ── カップ形状 ────────────────────────────────────────────────────────
    "cup_min_bars": 30,          # カップ最小期間 (営業日)
    "cup_max_bars": 150,         # カップ最大期間 (営業日)
    "cup_depth_min": 0.10,       # カップ深さ最小 (10%)
    "cup_depth_max": 0.50,       # カップ深さ最大 (50%)
    "rim_symmetry_max": 0.15,    # 左右リム高さ差の許容率 (v3.1: 0.12→0.15)

    # ── ハンドル ──────────────────────────────────────────────────────────
    "handle_min_bars": 5,        # ハンドル最小期間 (営業日)
    "handle_max_bars": 30,       # ハンドル最大期間 (営業日)
    "handle_depth_max": 0.12,    # ハンドル深さ最大 (カップ深さ比)
    "handle_position_min": 0.50, # ハンドル位置最小 (カップ高の50%以上)
    "handle_drift_max": 0.05,    # ハンドル上昇ドリフト許容率

    # ── 移動平均・トレンド ────────────────────────────────────────────────
    "ma_short": 10,              # 短期移動平均 (日)
    "ma_medium": 50,             # 中期移動平均 (日)
    "ma_long": 200,              # 長期移動平均 (日)
    "trend_slope_min": 0.0,      # トレンド傾き最小値
    "ma_alignment_required": True, # MA整列チェック (短期 > 中期 > 長期)

    # ── 出来高 ────────────────────────────────────────────────────────────
    "volume_ma_period": 50,      # 出来高移動平均期間
    "volume_bo_ratio_min": 1.5,  # BO時の出来高倍率最小 (r_bo)
    "volume_handle_ratio_max": 0.8, # ハンドル期間の出来高比率最大
    "volume_dry_up": True,       # ハンドルでの出来高枯渇チェック

    # ── トレード実行 ──────────────────────────────────────────────────────
    "entry_bo_pct": 0.01,        # BO認定の価格上昇率 (+1%)
    "tp1_ratio": 1.15,           # 第1利確目標 (+15%) — 50%売却
    "tp2_ratio": 1.28,           # 第2利確目標 (+28%) — 残50%売却
    "sl_ratio": 0.92,            # ストップロス (-8%)
    "trailing_stop_pct": 0.05,   # トレイリングストップ幅 (5%)

    # ── ポジション・時間軸 ────────────────────────────────────────────────
    "max_hold_bars": 120,        # 最大保有期間 (営業日) (v3.1: 60→120)
    "tp1_sell_ratio": 0.50,      # TP1到達時の売却比率 (50%)
    "position_size_tier1": 0.10, # Tier I 1銘柄あたりポジションサイズ
    "position_size_tier2": 0.07, # Tier II 1銘柄あたりポジションサイズ

    # ── フィルター ────────────────────────────────────────────────────────
    "min_market_cap_bn": 100,    # 最小時価総額 (億円)
    "min_liquidity_bn": 2.0,     # 最小流動性 (億円/日)
    "min_data_bars": 100,        # スクリーニングに必要な最小データ数
    "score_threshold": 60,       # スクリーニング通過最小スコア

    # ── Tier スコア閾値 ───────────────────────────────────────────────────
    "tier1_score": 90,           # Tier I 閾値 (スコア >= 90)
    "tier2_score": 80,           # Tier II 閾値 (スコア >= 80)
    "tier3_score": 70,           # Tier III 閾値 (スコア >= 70)
}


# ---------------------------------------------------------------------------
# CWHPattern: 検出パターンの情報を保持するデータクラス
# ---------------------------------------------------------------------------
@dataclass
class CWHPattern:
    ticker: str
    pattern_type: str            # 'CWH' or 'C-NH'
    has_handle: bool
    score: float
    all_conditions_met: bool

    # カップ形状
    cup_start_idx: int = 0
    cup_end_idx: int = 0
    cup_depth: float = 0.0
    rim_symmetry: float = 0.0
    left_rim_price: float = 0.0
    right_rim_price: float = 0.0
    cup_low_price: float = 0.0

    # ハンドル (CWH のみ)
    handle_start_idx: int = 0
    handle_end_idx: int = 0
    handle_depth: float = 0.0

    # ブレイクアウト
    bo_volume_ratio: float = 0.0

    # スコア内訳
    score_breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CWHScreener: CWH/C-NH パターン検出クラス
# ---------------------------------------------------------------------------
class CWHScreener:
    def __init__(self, df: pd.DataFrame, ticker: str, config: dict = None):
        """
        Parameters
        ----------
        df     : yfinance 形式の OHLCV DataFrame (インデックス: DatetimeIndex)
                 カラム名は大文字・小文字いずれも許容 (Close/close, Volume/volume)
        ticker : 銘柄コード (例: "5253")
        config : パラメータ辞書。省略時は CONFIG_V33 を使用
        """
        self.ticker = ticker
        self.config = config if config is not None else CONFIG_V33

        # カラム名を小文字に統一
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        self.df = df.dropna(subset=["close", "volume"]).reset_index(drop=True)

    # ── 公開 API ──────────────────────────────────────────────────────────

    def detect_all_patterns(self) -> List[CWHPattern]:
        """全候補カップを検出してパターンリストを返す"""
        if len(self.df) < self.config["min_data_bars"]:
            return []

        patterns: List[CWHPattern] = []
        close = self.df["close"].values
        n = len(close)

        cup_min = self.config["cup_min_bars"]
        cup_max = self.config["cup_max_bars"]

        # スライディングウィンドウでカップ候補を探索
        for right in range(cup_min, n):
            for cup_len in range(cup_min, min(cup_max + 1, right + 1)):
                left = right - cup_len
                pattern = self._evaluate_cup(left, right)
                if pattern is not None and pattern.all_conditions_met:
                    patterns.append(pattern)
                    break  # この right で最長カップを採用したら次の right へ

        # 重複削除 (right が重なるものは score 上位を残す)
        patterns = self._deduplicate(patterns)
        return patterns

    # ── 内部ロジック ──────────────────────────────────────────────────────

    def _evaluate_cup(self, left_idx: int, right_idx: int) -> Optional[CWHPattern]:
        """
        [left_idx, right_idx] のウィンドウでカップを評価し、
        条件を満たす場合は CWHPattern を返す。
        """
        close = self.df["close"].values
        volume = self.df["volume"].values
        n = len(close)

        window = close[left_idx:right_idx + 1]
        if len(window) < self.config["cup_min_bars"]:
            return None

        left_rim = close[left_idx]
        right_rim = close[right_idx]
        cup_low = window.min()
        cup_low_idx = left_idx + int(window.argmin())

        # ── 条件1: カップ深さ ─────────────────────────────────────────────
        higher_rim = max(left_rim, right_rim)
        cup_depth = (higher_rim - cup_low) / higher_rim
        if not (self.config["cup_depth_min"] <= cup_depth <= self.config["cup_depth_max"]):
            return None

        # ── 条件2: リム対称性 ─────────────────────────────────────────────
        rim_symmetry = abs(left_rim - right_rim) / higher_rim
        if rim_symmetry > self.config["rim_symmetry_max"]:
            return None

        # ── 条件3: カップ底の位置 (ウィンドウ中央付近) ──────────────────
        relative_low_pos = (cup_low_idx - left_idx) / max(right_idx - left_idx, 1)
        if not (0.25 <= relative_low_pos <= 0.75):
            return None

        # ── 条件4: 移動平均整列 ───────────────────────────────────────────
        if not self._check_ma_alignment(right_idx):
            return None

        # ── 条件5: 出来高 BO チェック ─────────────────────────────────────
        bo_vol_ratio = self._calc_bo_volume_ratio(right_idx)

        # ── ハンドル検出 ──────────────────────────────────────────────────
        handle_info = None
        has_handle = False
        handle_depth = 0.0
        handle_start = right_idx
        handle_end = right_idx

        remaining = n - right_idx - 1
        if remaining >= self.config["handle_min_bars"]:
            handle_info = self._detect_handle(right_idx)
            if handle_info is not None:
                has_handle = True
                handle_start, handle_end, handle_depth = handle_info

        pattern_type = "CWH" if has_handle else "C-NH"

        # ── スコア計算 ────────────────────────────────────────────────────
        score, breakdown = self._calc_score(
            cup_depth, rim_symmetry, bo_vol_ratio,
            has_handle, handle_depth
        )

        all_conditions_met = score >= self.config["score_threshold"]

        return CWHPattern(
            ticker=self.ticker,
            pattern_type=pattern_type,
            has_handle=has_handle,
            score=score,
            all_conditions_met=all_conditions_met,
            cup_start_idx=left_idx,
            cup_end_idx=right_idx,
            cup_depth=cup_depth,
            rim_symmetry=rim_symmetry,
            left_rim_price=left_rim,
            right_rim_price=right_rim,
            cup_low_price=cup_low,
            handle_start_idx=handle_start,
            handle_end_idx=handle_end,
            handle_depth=handle_depth,
            bo_volume_ratio=bo_vol_ratio,
            score_breakdown=breakdown,
        )

    def _check_ma_alignment(self, idx: int) -> bool:
        """短期 MA > 中期 MA > 長期 MA の整列を確認"""
        if not self.config["ma_alignment_required"]:
            return True
        close = self.df["close"]
        ma_s = self.config["ma_short"]
        ma_m = self.config["ma_medium"]
        ma_l = self.config["ma_long"]
        if idx < ma_l:
            return False  # データ不足
        ma_short_val = close.iloc[max(0, idx - ma_s + 1):idx + 1].mean()
        ma_medium_val = close.iloc[max(0, idx - ma_m + 1):idx + 1].mean()
        ma_long_val = close.iloc[max(0, idx - ma_l + 1):idx + 1].mean()
        return ma_short_val > ma_medium_val > ma_long_val

    def _calc_bo_volume_ratio(self, right_idx: int) -> float:
        """右リム付近の出来高倍率を計算"""
        volume = self.df["volume"].values
        vol_ma_period = self.config["volume_ma_period"]
        if right_idx < vol_ma_period:
            return 0.0
        avg_vol = volume[right_idx - vol_ma_period:right_idx].mean()
        if avg_vol == 0:
            return 0.0
        return volume[right_idx] / avg_vol

    def _detect_handle(self, cup_end_idx: int) -> Optional[tuple]:
        """
        カップ右リム以降のハンドルを検出。
        戻り値: (handle_start, handle_end, handle_depth) or None
        """
        close = self.df["close"].values
        n = len(close)
        h_min = self.config["handle_min_bars"]
        h_max = self.config["handle_max_bars"]
        right_rim = close[cup_end_idx]

        for handle_len in range(h_min, min(h_max + 1, n - cup_end_idx)):
            h_start = cup_end_idx + 1
            h_end = cup_end_idx + handle_len
            if h_end >= n:
                break
            handle_slice = close[h_start:h_end + 1]

            # ハンドル深さ
            handle_low = handle_slice.min()
            handle_depth = (right_rim - handle_low) / right_rim
            if handle_depth > self.config["handle_depth_max"]:
                continue

            # ハンドル位置: カップ高の50%以上
            cup_close_values = self.df["close"].values[:cup_end_idx + 1]
            cup_low_value = cup_close_values[cup_close_values.argmin()]
            cup_range = right_rim - cup_low_value
            if cup_range > 0:
                handle_pos = (handle_low - (right_rim - cup_range)) / cup_range
                if handle_pos < self.config["handle_position_min"]:
                    continue

            return (h_start, h_end, handle_depth)

        return None

    def _calc_score(
        self,
        cup_depth: float,
        rim_symmetry: float,
        bo_vol_ratio: float,
        has_handle: bool,
        handle_depth: float,
    ) -> tuple:
        """
        0～100 のスコアを計算して (score, breakdown) を返す。
        各評価項目の重みは以下の通り:
          - カップ深さ適正度   : 25点
          - リム対称性        : 25点
          - BO出来高倍率      : 25点
          - ハンドル品質      : 25点
        """
        breakdown = {}

        # カップ深さ (理想: 0.25～0.35)
        ideal_depth = 0.30
        depth_score = max(0.0, 25.0 - abs(cup_depth - ideal_depth) * 100)
        breakdown["cup_depth"] = round(depth_score, 1)

        # リム対称性 (小さいほど良い)
        sym_score = max(0.0, 25.0 * (1.0 - rim_symmetry / self.config["rim_symmetry_max"]))
        breakdown["rim_symmetry"] = round(sym_score, 1)

        # BO出来高倍率 (r_bo >= 1.5 で満点)
        min_ratio = self.config["volume_bo_ratio_min"]
        if bo_vol_ratio >= min_ratio:
            vol_score = min(25.0, 25.0 * (bo_vol_ratio / (min_ratio * 2)))
        else:
            vol_score = 25.0 * (bo_vol_ratio / min_ratio) * 0.5
        breakdown["bo_volume"] = round(vol_score, 1)

        # ハンドル品質
        if has_handle:
            handle_score = max(0.0, 25.0 * (1.0 - handle_depth / self.config["handle_depth_max"]))
        else:
            handle_score = 10.0  # C-NH はハンドルなしで基礎点のみ
        breakdown["handle"] = round(handle_score, 1)

        total = depth_score + sym_score + vol_score + handle_score
        return round(total, 1), breakdown

    def _deduplicate(self, patterns: List[CWHPattern]) -> List[CWHPattern]:
        """同じ right_idx を持つパターンのうち score 最大のものを残す"""
        best: dict = {}
        for p in patterns:
            key = p.cup_end_idx
            if key not in best or p.score > best[key].score:
                best[key] = p
        return sorted(best.values(), key=lambda p: p.score, reverse=True)


# ---------------------------------------------------------------------------
# check_exit_v33: v3.3 出口判定ロジック
# ---------------------------------------------------------------------------
def check_exit_v33(
    close_prices: np.ndarray,
    low_prices: np.ndarray,
    entry_price: float,
    config: dict,
    entry_date=None,
    dates=None,
) -> dict:
    """
    エントリー後の出口を判定する。

    Parameters
    ----------
    close_prices : エントリー日以降の終値配列
    low_prices   : エントリー日以降の安値配列
    entry_price  : エントリー価格
    config       : CONFIG_V33 相当のパラメータ辞書
    entry_date   : エントリー日 (datetime)。省略可
    dates        : 各バーの日付配列。省略可

    Returns
    -------
    dict with keys:
        reason      : 'TP1' | 'TP2' | 'SL' | 'TS' | 'EoP' (End of Period)
        bar_index   : 出口バーインデックス
        price       : 出口価格
        date        : 出口日 (dates 提供時)
    """
    tp1 = entry_price * config["tp1_ratio"]
    tp2 = entry_price * config["tp2_ratio"]
    sl = entry_price * config["sl_ratio"]
    ts_pct = config["trailing_stop_pct"]
    max_bars = config["max_hold_bars"]
    tp1_partial = config["tp1_sell_ratio"]

    trailing_stop_price = sl
    peak_price = entry_price
    tp1_hit = False

    effective_bars = min(len(close_prices), max_bars)

    for i in range(effective_bars):
        c = close_prices[i]
        lo = low_prices[i] if i < len(low_prices) else c

        # ストップロス
        if lo <= sl and not tp1_hit:
            return _exit_result("SL", i, sl, dates)

        # 第1利確
        if c >= tp1 and not tp1_hit:
            tp1_hit = True
            # TP1 到達後はトレイリングストップ開始
            peak_price = c
            trailing_stop_price = c * (1 - ts_pct)

        # 第2利確
        if tp1_hit and c >= tp2:
            return _exit_result("TP2", i, tp2, dates)

        # トレイリングストップ (TP1 到達後)
        if tp1_hit:
            if c > peak_price:
                peak_price = c
                trailing_stop_price = peak_price * (1 - ts_pct)
            if lo <= trailing_stop_price:
                return _exit_result("TS", i, trailing_stop_price, dates)

    # 最大保有期間到達
    last_i = effective_bars - 1
    last_price = close_prices[last_i] if effective_bars > 0 else entry_price
    return _exit_result("EoP", last_i, last_price, dates)


def _exit_result(reason: str, bar_index: int, price: float, dates) -> dict:
    result = {
        "reason": reason,
        "bar_index": bar_index,
        "price": round(price, 2),
        "date": None,
    }
    if dates is not None and bar_index < len(dates):
        result["date"] = dates[bar_index]
    return result
