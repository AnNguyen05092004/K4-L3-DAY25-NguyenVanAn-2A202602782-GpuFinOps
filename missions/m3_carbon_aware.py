"""M3b — Ext 5: lập lịch nhận biết carbon (carbon-aware scheduling).

Run: python missions/m3_carbon_aware.py

Vai trò: với các job có thể bị ngắt (interruptible=1) — chạy ở đâu cũng được vì không
cần phản hồi realtime — tính điện năng, tiền điện và carbon nếu chạy ở us-east-1 so với
các vùng khác, rồi chọn vùng tối ưu theo 3 tiêu chí (rẻ nhất, sạch nhất, cân bằng).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

BASELINE_REGION = "us-east-1"   # nơi các job đang chạy giả định


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    # 1) Điện năng của từng job interruptible: kWh = công suất (W) x GPU-giờ / 1000.
    #    GPU-giờ dùng cột `days` thật của từng job (M3 gốc dùng cứng 30 ngày).
    #    Giả định: GPU chạy ở công suất định mức (TDP), chưa tính PUE của datacenter.
    sched = []
    for j in jobs:
        if not int(num(j["interruptible"])):
            continue
        gpu_hours = num(j["hours_per_day"]) * num(j["days"]) * int(num(j["num_gpus"]))
        kwh = num(cat[j["gpu_type"]]["watts"]) * gpu_hours / 1000.0
        sched.append({"job_id": j["job_id"], "gpu_type": j["gpu_type"],
                      "gpu_hours": gpu_hours, "kwh": kwh})
    total_kwh = sum(s["kwh"] for s in sched)

    # 2) So sánh 5 vùng cho cùng tổng điện năng, chọn vùng theo 3 tiêu chí.
    table = sustainability.region_comparison(total_kwh)
    picks = sustainability.pick_regions(table)
    by_region = {t["region"]: t for t in table}
    base, clean = by_region[BASELINE_REGION], by_region[picks["cleanest"]]
    cheap, balanced = by_region[picks["cheapest"]], by_region[picks["balanced"]]

    # 3) Carbon từng job: ở vùng gốc vs vùng sạch nhất.
    for s in sched:
        s["co2_base_kg"] = sustainability.carbon_g(s["kwh"] * 1000, BASELINE_REGION) / 1000.0
        s["co2_clean_kg"] = sustainability.carbon_g(s["kwh"] * 1000, picks["cleanest"]) / 1000.0

    saved_kg = base["co2_kg"] - clean["co2_kg"]
    saved_pct = saved_kg / base["co2_kg"] * 100 if base["co2_kg"] else 0.0

    # Giá carbon (USD/tấn) mà tại đó vùng sạch nhất bắt đầu thắng vùng rẻ nhất:
    # p_sạch + g_sạch*c/1e6 = p_rẻ + g_rẻ*c/1e6  ->  c = (p_sạch - p_rẻ) * 1e6 / (g_rẻ - g_sạch)
    if cheap["g_per_kwh"] > clean["g_per_kwh"]:
        breakeven_carbon_price = ((clean["usd_per_kwh"] - cheap["usd_per_kwh"]) * 1e6
                                  / (cheap["g_per_kwh"] - clean["g_per_kwh"]))
    else:
        breakeven_carbon_price = 0.0

    if verbose:
        print("== Ext 5: Carbon-aware scheduling ==")
        print(f"{'job':18}{'GPU':7}{'GPU-giờ':>9}{'kWh':>9}{'CO2@' + BASELINE_REGION + ' (kg)':>20}"
              f"{'CO2@' + picks['cleanest'] + ' (kg)':>24}{'tiết kiệm':>11}")
        for s in sched:
            print(f"{s['job_id']:18}{s['gpu_type']:7}{s['gpu_hours']:>9.0f}{s['kwh']:>9.1f}"
                  f"{s['co2_base_kg']:>20.1f}{s['co2_clean_kg']:>24.1f}{s['co2_base_kg'] - s['co2_clean_kg']:>11.1f}")
        print(f"\nTổng: {total_kwh:,.0f} kWh  |  carbon {base['co2_kg']:,.1f} kg -> {clean['co2_kg']:,.1f} kg"
              f"  (tiết kiệm {saved_kg:,.1f} kg CO2e = {saved_pct:.0f}%)")

        print(f"\nSo sánh 5 vùng cho {total_kwh:,.0f} kWh (giá carbon nội bộ ${sustainability.CARBON_PRICE_PER_TONNE:.0f}/tấn):")
        print(f"{'vùng':17}{'$/kWh':>8}{'gCO2/kWh':>10}{'tiền điện $':>13}{'CO2 (kg)':>10}{'tổng $':>9}  nhãn")
        labels = {}   # một vùng có thể mang nhiều nhãn (ví dụ vừa rẻ nhất vừa cân bằng)
        for key, name in (("cheapest", "rẻ nhất"), ("cleanest", "sạch nhất"), ("balanced", "cân bằng")):
            labels.setdefault(picks[key], []).append(name)
        for t in sorted(table, key=lambda r: r["total_usd"]):
            label = ", ".join(labels.get(t["region"], []))
            print(f"{t['region']:17}{t['usd_per_kwh']:>8.3f}{t['g_per_kwh']:>10}{t['energy_usd']:>13.2f}"
                  f"{t['co2_kg']:>10.1f}{t['total_usd']:>9.2f}  {label}")
        print(f"\nRẻ nhất: {picks['cheapest']} | Sạch nhất: {picks['cleanest']} | Cân bằng: {picks['balanced']}")
        print(f"Chuyển {BASELINE_REGION} -> {picks['balanced']}: tiền điện ${base['energy_usd']:,.2f} -> ${balanced['energy_usd']:,.2f}, "
              f"carbon {base['co2_kg']:,.1f} -> {balanced['co2_kg']:,.1f} kg")
        print(f"{picks['cleanest']} chỉ thắng {picks['cheapest']} khi giá carbon > ${breakeven_carbon_price:,.0f}/tấn")

    return {
        "jobs": sched, "total_kwh": total_kwh, "regions": table, "picks": picks,
        "baseline_co2_kg": base["co2_kg"], "cleanest_co2_kg": clean["co2_kg"],
        "co2_saved_kg": saved_kg, "co2_saved_pct": saved_pct,
        "breakeven_carbon_price": breakeven_carbon_price,
    }


if __name__ == "__main__":
    run()
