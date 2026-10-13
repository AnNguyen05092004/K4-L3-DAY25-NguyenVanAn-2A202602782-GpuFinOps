"""Test cho các phần mở rộng "Your Turn" (file mới, không đụng tới test gốc)."""

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from finops import pricing
from missions import m2_inference_levers


def test_cache_break_even_reads():
    # phụ phí ghi 25%, đọc giảm 90% -> cần 0.25 / 0.9 = 0.2778 lần đọc
    assert abs(pricing.cache_break_even_reads(0.25, 0.10) - 0.25 / 0.9) < 1e-9
    assert pricing.cache_break_even_reads(0.0) == 0.0  # ghi miễn phí -> hòa vốn ngay
    assert pricing.cache_break_even_reads(0.25, 1.0) == float(
        "inf"
    )  # đọc không rẻ hơn -> vô nghĩa


def test_cache_is_worth_it_rules():
    assert pricing.cache_is_worth_it(0.3, 0.25) is True  # trên ngưỡng 0.2778
    assert pricing.cache_is_worth_it(0.2, 0.25) is False  # dưới ngưỡng
    assert pricing.cache_is_worth_it(0, 0.25) is False  # không đọc lại lần nào -> lỗ
    assert pricing.cache_is_worth_it(100, 0.25, 1.0) is False


def test_m2_cache_gate_not_cheaper_than_no_gate():
    r = m2_inference_levers.run(verbose=False)
    # bỏ chiết khấu cache ở nơi không đáng thì chi phí chỉ có thể bằng hoặc cao hơn
    assert r["optimized_daily_no_gate"] <= r["optimized_daily"] + 1e-6
    assert 60 <= r["savings_pct"] <= 95
