"""
CWH / C-NH 統合検出・バックテストエンジン v3.0
- CWH (Cup with Handle) パターン検出
- C-NH (Cup with No Handle) パターン検出
- 分割TP出口戦略（TP1 +15% / TP2 +28%）
- yfinance 実データ対応
"""

from __future__ import annotations

import yaml
from typing import Dict, List, Optional, Tuple


def load_config(config_file: str = "cwh_config.yaml") -> Dict:
    """YAML設定ファイルを読み込む。"""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


# ===== C-NH 検出パラメータ（デフォルト値） =====
CNH_PARAMS: Dict = {
    "cup_depth_min": 0.15,
    "cup_depth_max": 0.40,
    "cup_bars_min": 20,
    "cup_bars_max": 60,
    "handle_bars_max": 3,
    "rim_symmetry_max": 0.15,   # v3.1: 0.12 → 0.15 に緩和（検出率向上）
    "breakout_multiplier": 1.03,
    "breakout_volume_ratio": 1.5,
}

# ===== CWH 検出パラメータ（デフォルト値） =====
CWH_PARAMS: Dict = {
    "cup_depth_min": 0.15,
    "cup_depth_max": 0.35,
    "cup_bars_min": 20,
    "cup_bars_max": 60,
    "handle_depth_max": 0.50,
    "handle_bars_max": 20,
    "rim_symmetry_max": 0.15,
    "breakout_multiplier": 1.03,
    "breakout_volume_ratio": 1.5,
}


# ---------------------------------------------------------------------------
# 出口判定関数
# ---------------------------------------------------------------------------

def check_exit_with_split_tp(
    prices: List[float],
    lows: List[float],
    buy_price: float,
    cfg: Dict,
    position_size: int = 100,
) -> Tuple[bool, Optional[str], Optional[float], Optional[int], int]:
    """
    分割TP機能付き出口判定

    代数式:
      SL  = P_buy × 0.92
      TP1 = P_buy × 1.15  （P > TP1 で50%売却、即時リターン）
      TP2 = P_buy × 1.28  （P ≥ TP2 で残全売却）
      TS  = max_P × 0.92  （最高値から-8%のトレーリングストップ）

    優先順位（各バーで上から評価）:
      1. SL  (-8%)         → 全売却
      2. TP2 (+28%)        → 全売却
      2b. TP1 (+15% 超過)  → 50%売却、即時リターン
      3. TrailingStop      → 全売却

    Args:
        prices:        終値リスト（保有開始日から順）
        lows:          安値リスト（prices と同長）
        buy_price:     エントリー価格
        cfg:           設定辞書（未使用パラメータの拡張用）
        position_size: 初期ポジションサイズ（%）

    Returns:
        (exit_flag, exit_reason, exit_price, exit_index, qty_remaining)
    """
    sl = buy_price * 0.92
    tp1 = buy_price * 1.15
    tp2 = buy_price * 1.28
    max_p = buy_price
    ts_p = max_p * 0.92
    qty = position_size

    for i, (c, lo) in enumerate(zip(prices, lows)):
        if c > max_p:
            max_p = c
            ts_p = max_p * 0.92

        # 優先1: SL (-8%)
        if lo <= sl:
            return True, "SL(-8%)", sl, i, 0

        # 優先2: TP2 (+28%) → 全売却
        if c >= tp2:
            return True, "TP2(+28%)", tp2, i, 0

        # 優先2b: TP1 (+15% 超過) → 50%売却、即時リターン
        # TP1ラインを「超過（厳密な >）」した時点でシグナル発行
        if c > tp1 and qty == position_size:
            return True, "TP1(+15%)", tp1, i, position_size // 2

        # 優先3: トレーリングストップ（利益+5%以上の時のみ発動）
        if lo <= ts_p and max_p > buy_price * 1.05 and qty > 0:
            return True, "TrailingStop", ts_p, i, 0

    return False, None, None, None, qty


# ---------------------------------------------------------------------------
# パターン検出関数
# ---------------------------------------------------------------------------

def _is_cwh_pattern(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    params: Dict,
) -> bool:
    """
    CWH パターン判定（内部ヘルパー）

    条件:
      1. カップの深さ: cup_depth_min ≤ (H_L - B_1)/H_L ≤ cup_depth_max
      2. カップ期間:   cup_bars_min ≤ N_cup ≤ cup_bars_max
      3. ハンドル深さ: (H_L - B_handle)/H_L ≤ handle_depth_max
      4. ハンドル期間: N_handle ≤ handle_bars_max
      5. リム対称性:   |H_left - H_right|/H_L ≤ rim_symmetry_max
      6. ブレイクアウト: P_close ≥ breakout_multiplier × H_L
                        かつ V ≥ breakout_volume_ratio × V_25avg
    """
    n = len(closes)
    if n < params["cup_bars_min"] + params["handle_bars_max"]:
        return False

    # 最高値（カップ左リム）の探索
    hl = max(highs[: -params["handle_bars_max"]])
    hl_idx = highs.index(hl)

    # カップ底（最安値）
    cup_section_lows = lows[hl_idx : n - params["handle_bars_max"]]
    if not cup_section_lows:
        return False
    b1 = min(cup_section_lows)
    b1_rel_idx = cup_section_lows.index(b1)
    b1_idx = hl_idx + b1_rel_idx

    # カップ期間
    n_cup = b1_idx - hl_idx
    if not (params["cup_bars_min"] <= n_cup <= params["cup_bars_max"]):
        return False

    # カップ深さ
    depth_ratio = (hl - b1) / hl
    if not (params["cup_depth_min"] <= depth_ratio <= params["cup_depth_max"]):
        return False

    # 右リム（カップ底〜ハンドル開始前の最高値）
    right_section = highs[b1_idx: n - params["handle_bars_max"]]
    if not right_section:
        return False
    h_right = max(right_section)

    # リム対称性
    if hl > 0 and abs(hl - h_right) / hl > params["rim_symmetry_max"]:
        return False

    # ハンドル領域
    handle_section_lows = lows[n - params["handle_bars_max"]:]
    if not handle_section_lows:
        return False
    b_handle = min(handle_section_lows)
    handle_depth = (h_right - b_handle) / h_right if h_right > 0 else 1.0
    if handle_depth > params["handle_depth_max"]:
        return False

    # ブレイクアウト判定
    pivot = h_right
    v25_avg = sum(volumes[-25:]) / min(25, len(volumes))
    if (
        closes[-1] >= params["breakout_multiplier"] * pivot
        and volumes[-1] >= params["breakout_volume_ratio"] * v25_avg
    ):
        return True

    return False


def _is_cnh_pattern(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    params: Dict,
) -> bool:
    """
    C-NH パターン判定（内部ヘルパー）

    代数式:
      H_L = max{P_t | t ∈ [T-cup_bars_max, T-handle_bars_max_check]}
      B_1 = min{P_t | t ∈ [T_HL+1, T]}
      N_handle = {i | P_i < breakout_multiplier × H_L, i ∈ [B_1, T]}

    条件:
      1. N_handle ≤ handle_bars_max  （ハンドル期間が短い）
      2. cup_depth_min ≤ (H_L - B_1)/H_L ≤ cup_depth_max
      3. P_close ≥ breakout_multiplier × H_L
         かつ V ≥ breakout_volume_ratio × V_25avg
      4. リム対称性: |H_L - H_right|/H_L ≤ rim_symmetry_max（v3.1: 0.15）
    """
    # v3.1: C-NH の rim_symmetry_max を設定から適用
    cnh_params = dict(params)
    cnh_params["rim_symmetry_max"] = params.get("rim_symmetry_max", 0.15)

    n = len(closes)
    if n < cnh_params["cup_bars_min"]:
        return False

    # 直近 cup_bars_max 内の最高値（左リム）
    search_window = min(cnh_params["cup_bars_max"], n)
    hl = max(highs[-search_window:])
    hl_rel_idx = highs[-search_window:].index(hl)
    hl_idx = n - search_window + hl_rel_idx

    # カップ底
    after_hl = lows[hl_idx + 1:]
    if not after_hl:
        return False
    b1 = min(after_hl)
    b1_idx = hl_idx + 1 + after_hl.index(b1)

    # カップ深さ
    depth_ratio = (hl - b1) / hl if hl > 0 else 0.0
    if not (cnh_params["cup_depth_min"] <= depth_ratio <= cnh_params["cup_depth_max"]):
        return False

    # カップ期間
    n_cup = b1_idx - hl_idx
    if not (cnh_params["cup_bars_min"] <= n_cup <= cnh_params["cup_bars_max"]):
        return False

    # ハンドル期間（HL価格の breakout_multiplier 倍未満の日数）
    breakout_line = cnh_params["breakout_multiplier"] * hl
    handle_bars = sum(1 for h in highs[b1_idx:] if h < breakout_line)
    if handle_bars > cnh_params["handle_bars_max"]:
        return False

    # 右リム対称性（C-NH ではカップ底以降の最高値を右リムとする）
    right_section = highs[b1_idx:]
    if not right_section:
        return False
    h_right = max(right_section)
    if hl > 0 and abs(hl - h_right) / hl > cnh_params["rim_symmetry_max"]:
        return False

    # ブレイクアウト判定
    v25_avg = sum(volumes[-25:]) / min(25, len(volumes))
    if (
        closes[-1] >= breakout_line
        and volumes[-1] >= cnh_params["breakout_volume_ratio"] * v25_avg
    ):
        return True

    return False


def find_patterns(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    cfg: Optional[Dict] = None,
) -> Dict[str, bool]:
    """
    CWH / C-NH パターンを検出し、結果を返す。

    Args:
        highs, lows, closes, volumes: OHLCVデータのリスト
        cfg: cwh_config.yaml から読み込んだ設定辞書

    Returns:
        {"cwh": bool, "cnh": bool}
    """
    cfg = cfg or {}

    cwh_params = dict(CWH_PARAMS)
    cnh_params = dict(CNH_PARAMS)

    # 設定ファイルからパラメータを上書き
    if "cwh" in cfg:
        cwh_params.update(cfg["cwh"])
    if "cnh" in cfg:
        # v3.1: C-NH の rim_symmetry_max を設定から適用（0.12 → 0.15）
        cnh_params.update(cfg["cnh"])
        cnh_params["rim_symmetry_max"] = cfg["cnh"].get("rim_symmetry_max", 0.15)

    return {
        "cwh": _is_cwh_pattern(highs, lows, closes, volumes, cwh_params),
        "cnh": _is_cnh_pattern(highs, lows, closes, volumes, cnh_params),
    }


# ---------------------------------------------------------------------------
# バックテストエンジン
# ---------------------------------------------------------------------------

class CWHCNHBacktest:
    """
    CWH / C-NH 統合バックテストエンジン

    使用例:
        engine = CWHCNHBacktest(config_file="cwh_config.yaml")
        results = engine.run_all({"cwh": ["5253.T"], "cnh": ["5803.T"]})
    """

    def __init__(
        self,
        config_file: str = "cwh_config.yaml",
        start_date: str = "2020-01-01",
        end_date: str = "2026-03-29",
        use_real_data: bool = True,
    ) -> None:
        self.cfg = load_config(config_file)
        self.start_date = self.cfg.get("backtest_start", start_date)
        self.end_date = self.cfg.get("backtest_end", end_date)
        self.use_real_data = use_real_data
        # v3.1: max_hold_bars を設定から読み込み（デフォルト120）
        self.max_hold_days: int = self.cfg.get("max_hold_bars", 120)
        self.trades: List[Dict] = []

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def _fetch_data(self, code: str):
        """yfinance から実データを取得する。"""
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance が見つかりません。`pip install yfinance>=0.2.28` を実行してください。"
            ) from exc

        df = yf.download(code, start=self.start_date, end=self.end_date, progress=False)
        if df.empty:
            raise ValueError(f"{code}: データが取得できませんでした。")
        return df

    # ------------------------------------------------------------------
    # 単一銘柄バックテスト
    # ------------------------------------------------------------------

    def _backtest_single(self, code: str, pattern_type: str) -> List[Dict]:
        """1銘柄のバックテストを実行し、トレードリストを返す。"""
        df = self._fetch_data(code)

        highs = df["High"].tolist()
        lows = df["Low"].tolist()
        closes = df["Close"].tolist()
        volumes = df["Volume"].tolist()
        dates = df.index.tolist()

        trades = []
        window = self.cfg.get("cwh" if pattern_type == "cwh" else "cnh", {}).get(
            "cup_bars_max", 60
        ) + self.cfg.get("cwh" if pattern_type == "cwh" else "cnh", {}).get(
            "handle_bars_max", 20
        )

        for end_idx in range(window, len(closes)):
            h_win = highs[:end_idx]
            lo_win = lows[:end_idx]
            c_win = closes[:end_idx]
            v_win = volumes[:end_idx]

            pattern_result = find_patterns(h_win, lo_win, c_win, v_win, self.cfg)

            if not pattern_result.get(pattern_type, False):
                continue

            # エントリー
            buy_price = closes[end_idx]
            entry_date = dates[end_idx]

            # 保有期間の終値・安値
            hold_end = min(end_idx + self.max_hold_days + 1, len(closes))
            hold_prices = closes[end_idx + 1: hold_end]
            hold_lows = lows[end_idx + 1: hold_end]

            if not hold_prices:
                continue

            exit_flag, reason, exit_price, exit_idx_rel, qty = check_exit_with_split_tp(
                hold_prices, hold_lows, buy_price, self.cfg
            )

            if not exit_flag:
                # max_hold_bars 経過後に強制終了
                reason = "MaxHold"
                exit_price = hold_prices[-1]
                exit_idx_rel = len(hold_prices) - 1

            actual_exit_price = exit_price if exit_price is not None else hold_prices[-1]
            ret = (actual_exit_price - buy_price) / buy_price

            trades.append(
                {
                    "code": code,
                    "pattern": pattern_type.upper(),
                    "entry_date": entry_date,
                    "entry_price": buy_price,
                    "exit_reason": reason,
                    "exit_price": actual_exit_price,
                    "return": ret,
                }
            )

        return trades

    # ------------------------------------------------------------------
    # パフォーマンス計算
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_metrics(trades: List[Dict]) -> Dict:
        """バックテスト結果からパフォーマンス指標を計算する。"""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_return": 0.0,
                "max_drawdown": 0.0,
            }

        returns = [t["return"] for t in trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        total_win = sum(wins) if wins else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

        # 最大ドローダウン（累積リターンの最大落込み）
        cumulative = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            cumulative *= 1 + r
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "profit_factor": profit_factor,
            "avg_return": sum(returns) / len(returns),
            "max_drawdown": max_dd,
        }

    # ------------------------------------------------------------------
    # 全銘柄実行
    # ------------------------------------------------------------------

    def run_all(self, stocks: Dict[str, List[str]]) -> Dict:
        """
        全銘柄・全パターンのバックテストを実行する。

        Args:
            stocks: {"cwh": ["5253.T", ...], "cnh": ["5803.T", ...]}

        Returns:
            パフォーマンス指標と全トレードリスト
        """
        all_trades: List[Dict] = []

        for pattern_type, codes in stocks.items():
            for code in codes:
                try:
                    trades = self._backtest_single(code, pattern_type)
                    all_trades.extend(trades)
                    print(f"[{pattern_type.upper()}] {code}: {len(trades)} トレード")
                except Exception as exc:
                    print(f"[ERROR] {code}: {exc}")

        self.trades = all_trades
        metrics = self._calc_metrics(all_trades)
        metrics["trades"] = all_trades
        return metrics
