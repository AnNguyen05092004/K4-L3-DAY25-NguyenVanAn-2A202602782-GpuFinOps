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


def test_reasoning_budget_shares_and_scenarios():
    rs = m2_inference_levers.run(verbose=False)["reasoning"]
    # reasoning chiếm ít traffic nhưng nhiều chi phí hơn, và nhiều năng lượng hơn nữa
    assert rs["traffic_share"] < rs["cost_share"] < rs["energy_share"]
    assert 0 < rs["traffic_share"] < 1
    # trần 10% lớn hơn tỷ lệ hiện tại -> không tiết kiệm; trần càng chặt tiết kiệm càng nhiều
    saved = [s["saved_usd"] for s in rs["scenarios"]]
    assert saved[0] == 0.0
    assert saved == sorted(saved)
    wh = [s["saved_wh"] for s in rs["scenarios"]]
    assert wh == sorted(wh) and wh[-1] > 0


def test_region_picks_and_carbon_scheduling():
    from finops import sustainability
    from missions import m3_carbon_aware
    rows = sustainability.region_comparison(1000)
    picks = sustainability.pick_regions(rows)
    assert picks["cheapest"] == "us-east-wa" and picks["cleanest"] == "europe-north1"
    # giá carbon = 0 -> cân bằng trùng rẻ nhất; giá carbon rất lớn -> trùng sạch nhất
    assert sustainability.pick_regions(sustainability.region_comparison(1000, 0.0))["balanced"] == picks["cheapest"]
    assert sustainability.pick_regions(sustainability.region_comparison(1000, 1e6))["balanced"] == picks["cleanest"]
    r = m3_carbon_aware.run(verbose=False)
    assert len(r["jobs"]) == 5 and r["co2_saved_kg"] > 0
    assert r["cleanest_co2_kg"] < r["baseline_co2_kg"]


def test_rightsize_checks_bandwidth_and_vram_not_just_price():
    from missions import m1_efficiency_audit as m1
    from missions._common import catalog_by_type
    cat = catalog_by_type()
    # nhu cầu rất nhỏ -> GPU rẻ nhất (L4) là đủ
    small = {"tflops": 10, "bw_tbs": 0.05, "mem_gb": 5}
    pick, _ = m1.recommend_rightsize(small, "H100", cat)
    assert pick["gpu_type"] == "L4"
    # cần băng thông 0.5 TB/s: L4 (0.3) và A10G (0.6 nhưng phải ≤ 70%) bị loại -> chọn A100
    need_bw = {"tflops": 10, "bw_tbs": 0.5, "mem_gb": 5}
    pick, options = m1.recommend_rightsize(need_bw, "H100", cat)
    assert pick["gpu_type"] == "A100"
    assert all("bandwidth" in o["blocked_by"] for o in options if o["gpu_type"] in ("L4", "A10G"))
    # cần 60 GB VRAM: A10G/L4 (24 GB) bị loại vì VRAM
    need_vram = {"tflops": 10, "bw_tbs": 0.05, "mem_gb": 60}
    pick, _ = m1.recommend_rightsize(need_vram, "H100", cat)
    assert pick["gpu_type"] == "A100"


def test_rightsize_amd_only_when_allowed_and_no_downgrade_when_blocked():
    from missions import m1_efficiency_audit as m1
    from missions._common import catalog_by_type
    cat = catalog_by_type()
    heavy = {"tflops": 400, "bw_tbs": 1.5, "mem_gb": 60}   # H100 train khỏe: A100 không đủ compute
    pick, _ = m1.recommend_rightsize(heavy, "H100", cat)
    assert pick is None                                    # giữ nguyên, không ép hạ cấp
    pick_amd, _ = m1.recommend_rightsize(heavy, "H100", cat, allow_amd=True)
    assert pick_amd["gpu_type"] == "MI300X"


def test_m1_rightsizing_report_is_consistent():
    from missions import m1_efficiency_audit as m1
    r = m1.run(verbose=False)
    assert len(r["rightsizing"]) == 11
    assert r["rightsize_monthly_savings"] >= 0
    assert r["rightsize_monthly_savings_with_amd"] >= r["rightsize_monthly_savings"]
