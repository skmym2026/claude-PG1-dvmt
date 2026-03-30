"""
CWH/C-NH v3.3 出口判定ロジック テスト

テスト対象: cwh_v33.check_exit_v33()

テストケース:
  - test_exit_v33_sl         : SL が最優先で発動することを確認
  - test_exit_v33_ts_after_tp1: TS が TP1 到達後のみ発動することを確認
  - test_exit_v33_ts_not_before_tp1: TP1 未到達では TS が発動しないことを確認
  - test_exit_v33_tp2        : TP2 到達で全売却
  - test_exit_v33_eop        : EoP（保有期間終了）で退出
  - test_exit_v33_sl_overrides_ts: SL は TS より優先されることを確認
  - test_exit_v33_config_params: CONFIG_V33 の v3.3 確定パラメータを確認
"""

import sys
import os

import numpy as np
import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwh_v33 import check_exit_v33, generate_demo_data, CONFIG_V33


# ============================================================
# テスト用ヘルパー
# ============================================================

def make_flat(buy_price: float, n: int):
    """全バーで価格が横ばいのシンプルな配列を生成する。"""
    closes = np.full(n, buy_price)
    lows   = np.full(n, buy_price)
    return closes, lows


def make_rising(buy_price: float, n: int, final_price: float):
    """buy_price から final_price まで線形に上昇する配列を生成する。"""
    closes = np.linspace(buy_price, final_price, n)
    lows   = closes * 0.995  # 安値は終値の -0.5%
    return closes, lows


# ============================================================
# テスト本体
# ============================================================

class TestExitV33SL:
    """SL テスト: L(t) <= P_entry × 0.92 で最優先発動"""

    def test_sl_triggers_immediately(self):
        """最初のバーで SL を下回る場合に SL で退出することを確認。"""
        buy_price = 1000.0
        sl_price  = buy_price * CONFIG_V33["stop_loss_ratio"]  # 920

        closes = np.array([900.0, 950.0, 1100.0])  # 最初のバーは低い終値
        lows   = np.array([sl_price - 1, 930.0, 1050.0])  # 安値が SL を下回る

        result = check_exit_v33(closes, lows, buy_price)

        assert result["exit_reason"] == "SL", f"期待: SL、実際: {result['exit_reason']}"
        assert result["exit_bar"] == 0
        assert result["exit_price"] == pytest.approx(sl_price, rel=1e-6)
        assert result["return_pct"] < 0

    def test_sl_triggers_mid_series(self):
        """中間バーで SL を下回る場合に SL で退出することを確認。"""
        buy_price = 1000.0
        sl_price  = buy_price * CONFIG_V33["stop_loss_ratio"]

        # 最初 3 バーは上昇、4 バー目で SL
        closes = np.array([1010.0, 1020.0, 1030.0, 910.0, 1100.0])
        lows   = np.array([1005.0, 1015.0, 1025.0, sl_price - 5, 1080.0])

        result = check_exit_v33(closes, lows, buy_price)

        assert result["exit_reason"] == "SL"
        assert result["exit_bar"] == 3


class TestExitV33TS:
    """TS テスト: TP1 到達後のみ発動"""

    def test_ts_triggers_after_tp1(self):
        """TP1 を超えた後に TS が発動することを確認。"""
        buy_price = 1000.0
        tp1_price = buy_price * CONFIG_V33["take_profit_ratio_1"]  # 1150
        ts_ratio  = CONFIG_V33["trailing_stop_ratio"]               # 0.92

        # バー 0: TP1 超え（tp1_reached=True）、バー 1: 高値更新、バー 2: TS 下回る
        p_max_after_rise = 1200.0
        ts_threshold = p_max_after_rise * ts_ratio  # 1104

        closes = np.array([tp1_price + 1, p_max_after_rise, ts_threshold - 1])
        lows   = np.array([tp1_price,     p_max_after_rise - 5, ts_threshold - 1])

        result = check_exit_v33(closes, lows, buy_price)

        assert result["exit_reason"] == "TS", f"期待: TS、実際: {result['exit_reason']}"
        assert result["tp1_reached"] is True
        assert result["return_pct"] > 0  # TP1 超えているのでプラス

    def test_ts_does_not_trigger_before_tp1(self):
        """TP1 未到達では TS が発動しないことを確認（SL または EoP になる）。"""
        buy_price = 1000.0
        tp1_price = buy_price * CONFIG_V33["take_profit_ratio_1"]  # 1150
        ts_ratio  = CONFIG_V33["trailing_stop_ratio"]

        # 高値は TP1 未満（1100）→ P_max=1100 → TS=1012
        # 安値を TS 以下にしても TP1 未達なので TS は発動しない
        p_max     = tp1_price - 50  # 1100
        ts_thresh = p_max * ts_ratio  # 1012

        closes = np.array([p_max, ts_thresh - 5, buy_price + 10])
        lows   = np.array([p_max - 5, ts_thresh - 5, buy_price + 5])

        result = check_exit_v33(closes, lows, buy_price)

        assert result["exit_reason"] != "TS", (
            "TP1 未到達なのに TS が発動した（誤動作）"
        )
        assert result["tp1_reached"] is False


class TestExitV33TP2:
    """TP2 テスト: C(t) >= P_entry × 1.28 で全売却"""

    def test_tp2_triggers(self):
        """TP2 価格に到達したら TP2 で退出することを確認。"""
        buy_price = 1000.0
        tp2_price = buy_price * CONFIG_V33["take_profit_ratio_2"]  # 1280

        closes = np.array([1100.0, 1200.0, tp2_price])
        lows   = np.array([1090.0, 1190.0, tp2_price - 5])

        result = check_exit_v33(closes, lows, buy_price)

        assert result["exit_reason"] == "TP2", f"期待: TP2、実際: {result['exit_reason']}"
        assert result["exit_price"] == pytest.approx(tp2_price, rel=1e-6)
        assert result["return_pct"] == pytest.approx(28.0, abs=0.1)

    def test_tp2_requires_close_not_low(self):
        """TP2 は終値判定（安値が TP2 を超えても終値未達なら発動しない）。"""
        buy_price = 1000.0
        tp2_price = buy_price * CONFIG_V33["take_profit_ratio_2"]

        # 終値は TP2 未満だが安値が TP2 以上（不整合データでロジックを確認）
        closes = np.array([tp2_price - 1])
        lows   = np.array([tp2_price + 10])

        result = check_exit_v33(closes, lows, buy_price)

        assert result["exit_reason"] != "TP2"


class TestExitV33EoP:
    """EoP テスト: max_hold_bars 経過後に退出"""

    def test_eop_triggers_at_max_hold(self):
        """max_hold_bars バー経過後に EoP で退出することを確認。"""
        buy_price = 1000.0
        max_bars  = CONFIG_V33["max_hold_bars"]  # 90

        # どの出口条件も発動しない価格帯（横ばい）
        n = max_bars + 20  # max_hold_bars を超えるデータ
        flat_price = buy_price * 1.05  # +5%（SL/TP どちらにも届かない）
        closes = np.full(n, flat_price)
        lows   = np.full(n, flat_price * 0.99)

        result = check_exit_v33(closes, lows, buy_price)

        assert result["exit_reason"] == "EoP", f"期待: EoP、実際: {result['exit_reason']}"
        assert result["exit_bar"] == max_bars - 1  # 0 始まり

    def test_eop_respects_max_hold_bars_v33(self):
        """v3.3 の max_hold_bars は 90 日であることを確認。"""
        assert CONFIG_V33["max_hold_bars"] == 90, (
            f"v3.3 max_hold_bars は 90 のはずだが {CONFIG_V33['max_hold_bars']} になっている"
        )


class TestExitV33Priority:
    """出口優先順位テスト: SL > TS > TP2 > EoP"""

    def test_sl_overrides_ts(self):
        """同一バーで SL と TS が重なった場合 SL が優先されることを確認。"""
        buy_price = 1000.0
        sl_price  = buy_price * CONFIG_V33["stop_loss_ratio"]       # 920
        tp1_price = buy_price * CONFIG_V33["take_profit_ratio_1"]   # 1150
        tp2_price = buy_price * CONFIG_V33["take_profit_ratio_2"]   # 1280
        ts_ratio  = CONFIG_V33["trailing_stop_ratio"]

        # p_max は TP2 を超えないよう TP2 価格より 10 低い値に設定
        p_max    = tp2_price - 10
        ts_thresh = p_max * ts_ratio  # noqa: F841  (使用しないが可読性のため保持)

        # バー 0: TP1 超え、バー 1: 高値更新（TP2 未満）、バー 2: SL と TS の両方を下回る
        closes = np.array([tp1_price + 1, p_max, sl_price - 10])
        lows   = np.array([tp1_price,     p_max,  sl_price - 10])

        result = check_exit_v33(closes, lows, buy_price)

        assert result["exit_reason"] == "SL", (
            f"SL が TS より優先されるべき。実際: {result['exit_reason']}"
        )


class TestConfigV33Params:
    """CONFIG_V33 パラメータ検証テスト"""

    def test_cup_bars_min_is_30(self):
        """★修正1: cup_bars_min が 30 になっていることを確認。"""
        assert CONFIG_V33["cwh_cup_bars_min"] == 30, (
            f"v3.3 cwh_cup_bars_min は 30 のはずだが {CONFIG_V33['cwh_cup_bars_min']}"
        )

    def test_prior_trend_min_is_028(self):
        """★修正2: prior_trend_min が 0.28 になっていることを確認。"""
        assert CONFIG_V33["cwh_prior_trend_min"] == pytest.approx(0.28), (
            f"v3.3 cwh_prior_trend_min は 0.28 のはずだが {CONFIG_V33['cwh_prior_trend_min']}"
        )

    def test_trailing_stop_min_gain_is_015(self):
        """★修正3: trailing_stop_min_gain が 0.15 になっていることを確認。"""
        assert CONFIG_V33["trailing_stop_min_gain"] == pytest.approx(0.15), (
            f"v3.3 trailing_stop_min_gain は 0.15 のはずだが {CONFIG_V33['trailing_stop_min_gain']}"
        )

    def test_max_hold_bars_is_90(self):
        """★修正4: max_hold_bars が 90 になっていることを確認。"""
        assert CONFIG_V33["max_hold_bars"] == 90, (
            f"v3.3 max_hold_bars は 90 のはずだが {CONFIG_V33['max_hold_bars']}"
        )

    def test_stop_loss_ratio(self):
        """SL 比率が 0.92（-8%）であることを確認。"""
        assert CONFIG_V33["stop_loss_ratio"] == pytest.approx(0.92)

    def test_take_profit_ratio_2(self):
        """TP2 比率が 1.28（+28%）であることを確認。"""
        assert CONFIG_V33["take_profit_ratio_2"] == pytest.approx(1.28)


class TestGenerateDemoData:
    """デモデータ生成テスト"""

    def test_demo_data_shape(self):
        """generate_demo_data が正しい列を持つ DataFrame を返すことを確認。"""
        df = generate_demo_data(n_bars=100)
        assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)
        assert len(df) == 100

    def test_demo_data_ohlc_consistency(self):
        """High >= Close >= Low の OHLC 整合性を確認。"""
        df = generate_demo_data(n_bars=200)
        assert (df["High"] >= df["Close"]).all(), "High >= Close 違反"
        assert (df["Close"] >= df["Low"]).all(), "Close >= Low 違反"
        assert (df["High"] >= df["Low"]).all(), "High >= Low 違反"

    def test_demo_data_positive_volume(self):
        """Volume が正の値であることを確認。"""
        df = generate_demo_data(n_bars=50)
        assert (df["Volume"] > 0).all()
