"""
tests/test_sbi_output.py
SBI注文フォーマットのテスト
"""

import os
import csv
import tempfile
import pytest
from cwh_sbi_order_output import format_sbi_order, save_orders_to_csv


def _buy_result(**kwargs):
    base = {
        "ACTION": "BUY",
        "CODE": "9999.T",
        "ENTRY_PRICE": 1250.0,
        "TARGET_1": 1465.0,
        "TRAILING_STOP": 1150.0,
        "TRAILING_STOP_WIDTH": 37.5,
        "DAYS_HELD": 0,
        "REASON": "Breakout signal",
        "MARKET_CAP": 2000,
        "CREDIT_RATIO": 2.5,
        "LIQUIDITY_2B_YEN": 5.0,
    }
    base.update(kwargs)
    return base


def _sell_result(**kwargs):
    base = {
        "ACTION": "SELL_NOW",
        "CODE": "9999.T",
        "REASON": "Upper wick + large volume",
    }
    base.update(kwargs)
    return base


# ---- Noneハンドリング ----

def test_format_none_returns_none():
    """入力がNoneの場合はNoneを返す。"""
    assert format_sbi_order(None) is None


# ---- BUY注文 ----

def test_buy_order_type_is_oco():
    """BUYシグナルはOCO注文タイプになる。"""
    order = format_sbi_order(_buy_result())
    assert order is not None
    assert order['ORDER_TYPE'] == 'OCO'


def test_buy_order_has_required_fields():
    """BUY注文に必要なフィールドが含まれる。"""
    order = format_sbi_order(_buy_result())
    assert order is not None
    for field in ('TICKER', 'SIDE', 'ENTRY_PRICE', 'PROFIT_TARGET', 'STOP_LOSS', 'TIME_LIMIT'):
        assert field in order, f"Missing field: {field}"


def test_buy_order_ticker():
    """BUY注文のTICKERが正しい。"""
    order = format_sbi_order(_buy_result(CODE="7203.T"))
    assert order['TICKER'] == '7203.T'


def test_buy_order_trail_width():
    """トレール幅が3.0%に設定されている。"""
    order = format_sbi_order(_buy_result())
    assert order['TRAIL_WIDTH_PERCENT'] == 3.0


def test_buy_order_stop_loss_equals_trailing_stop():
    """ストップロスがTRAILING_STOPの値と一致する。"""
    result = _buy_result(TRAILING_STOP=1150.0)
    order = format_sbi_order(result)
    assert order['STOP_LOSS'] == 1150.0


def test_buy_order_profit_target_equals_target_1():
    """利確目標がTARGET_1と一致する。"""
    result = _buy_result(TARGET_1=1465.0)
    order = format_sbi_order(result)
    assert order['PROFIT_TARGET'] == 1465.0


# ---- SELL注文 ----

def test_sell_order_type_is_market():
    """SELL_NOWシグナルは成行き売り注文タイプになる。"""
    order = format_sbi_order(_sell_result())
    assert order is not None
    assert order['ORDER_TYPE'] == 'SELL_AT_MARKET'


def test_sell_order_ticker():
    """SELL注文のTICKERが正しい。"""
    order = format_sbi_order(_sell_result(CODE="9984.T"))
    assert order['TICKER'] == '9984.T'


def test_sell_order_has_reason():
    """SELL注文にREASONフィールドが含まれる。"""
    order = format_sbi_order(_sell_result(REASON="75-day cutoff"))
    assert order['REASON'] == "75-day cutoff"


# ---- CSV出力 ----

def test_save_orders_to_csv_creates_file():
    """有効な注文リストがある場合にCSVファイルが作成される。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_orders.csv")
        orders = [format_sbi_order(_buy_result())]
        save_orders_to_csv(orders, path)
        assert os.path.exists(path)


def test_save_orders_to_csv_row_count():
    """注文数分の行がCSVに書き込まれる。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_orders.csv")
        orders = [
            format_sbi_order(_buy_result(CODE="7203.T")),
            format_sbi_order(_sell_result(CODE="9984.T")),
        ]
        save_orders_to_csv(orders, path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2


def test_save_orders_to_csv_empty_list():
    """注文リストが空の場合はファイルを作成しない。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty_orders.csv")
        save_orders_to_csv([], path)
        assert not os.path.exists(path)


def test_save_orders_to_csv_skips_none():
    """Noneを含む注文リストはNoneをスキップする。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_orders.csv")
        orders = [None, format_sbi_order(_buy_result()), None]
        save_orders_to_csv(orders, path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
