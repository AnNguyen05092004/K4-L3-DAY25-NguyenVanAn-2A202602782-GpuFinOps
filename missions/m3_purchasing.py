"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

Ext 1 (chính sách mua nâng cao): so sánh quy tắc đơn giản cũ với chính sách mới trong
finops/pricing.py (tỷ lệ thu hồi spot theo loại GPU, reserved 1 năm vs 3 năm theo vòng
đời workload, duty cycle theo số ngày chạy thật). Kết quả gốc của M3 giữ nguyên để
verify.py/M5 không đổi; phần Ext 1 chỉ thêm phân tích và khóa "improved" trong kết quả.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30

# --- Ext 1: giả định vòng đời workload (tháng) theo loại job — dùng để chọn reserved 1yr vs 3yr ---
LIFETIME_MONTHS = {"infer": 36}      # dịch vụ inference chạy lâu dài
DEFAULT_LIFETIME_MONTHS = 12         # train/dev: ngắn hạn hơn
MATRIX_GPUS = ("H100", "A100", "A10G", "L4")
MATRIX_DUTIES = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
SENSITIVITY_LIFETIMES = (6, 12, 24, 36)


def _prices(c: dict) -> dict:
    """Bảng giá số của một GPU lấy từ catalog (CSV đọc ra là chuỗi)."""
    return {"on_demand_hr": num(c["on_demand_hr"]), "spot_hr": num(c["spot_hr"]),
            "reserved_1yr_hr": num(c["reserved_1yr_hr"]), "reserved_3yr_hr": num(c["reserved_3yr_hr"])}


def _price_job(gpu_hours: float, ngpu: int, prices: dict, choice: dict, gpu_type: str,
               lifetime: float, spot_rate: float | None = None) -> float:
    """Tiền/tháng của một job theo gói đã chọn, tính ĐÚNG: reserved bị tính 24/7 cả tháng."""
    tier = choice["tier"]
    if tier == "spot":
        rate = spot_rate if spot_rate is not None else pricing.SPOT_INTERRUPT_RATE.get(
            gpu_type, pricing.DEFAULT_SPOT_INTERRUPT_RATE)
        return gpu_hours * prices["spot_hr"] * pricing.spot_overhead_factor(rate)
    if tier == "reserved":
        term = choice["term"] or "3yr"
        eff = pricing.reserved_effective_rate(prices[f"reserved_{term}_hr"],
                                              pricing.RESERVED_TERMS[term], lifetime)
        return eff * 24 * DAYS * ngpu
    return gpu_hours * prices["on_demand_hr"]


def _label(choice: dict) -> str:
    return choice["tier"] if not choice["term"] else f"{choice['tier']}-{choice['term']}"


def compare_policies(jobs: list, cat: dict) -> dict:
    """Ext 1: chấm cùng một bộ job bằng quy tắc cũ và chính sách mới với cách tính tiền đã sửa."""
    rows = []
    od_total = old_total = new_total = 0.0
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd, days = num(j["hours_per_day"]), num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        lifetime = LIFETIME_MONTHS.get(j["kind"], DEFAULT_LIFETIME_MONTHS)
        prices = _prices(cat[gtype])
        gpu_hours = hpd * days * ngpu
        od_cost = gpu_hours * prices["on_demand_hr"]

        # quy tắc cũ: tier theo hours_per_day, spot luôn 5%/giờ, reserved luôn 3 năm
        old_tier = pricing.recommend_tier(hpd, interruptible)
        old_choice = {"tier": old_tier, "term": "3yr" if old_tier == "reserved" else None}
        old_cost = _price_job(gpu_hours, ngpu, prices, old_choice, gtype, lifetime,
                              spot_rate=pricing.DEFAULT_SPOT_INTERRUPT_RATE)
        # chính sách mới: rẻ nhất mỗi giờ-dùng-thật, xét thu hồi theo GPU, số ngày chạy, vòng đời
        new_choice = pricing.best_tier(hpd, interruptible, prices, gtype, days, lifetime)
        new_cost = _price_job(gpu_hours, ngpu, prices, new_choice, gtype, lifetime)

        od_total += od_cost
        old_total += old_cost
        new_total += new_cost
        rows.append({"job_id": j["job_id"], "gpu_type": gtype, "old": _label(old_choice), "new": _label(new_choice),
                     "on_demand": od_cost, "old_cost": old_cost, "new_cost": new_cost})
    return {
        "rows": rows, "on_demand": od_total, "old_cost": old_total, "new_cost": new_total,
        "old_pct": (1 - old_total / od_total) * 100 if od_total else 0.0,
        "new_pct": (1 - new_total / od_total) * 100 if od_total else 0.0,
    }


def tier_matrix(cat: dict, lifetime: float = 36) -> dict:
    """Ma trận khuyến nghị GPU x duty cycle x interruptible (dấu * = khác quy tắc cũ)."""
    short = {"on_demand": "OD", "spot": "SPOT"}
    matrix, diffs = {}, 0
    for gtype in MATRIX_GPUS:
        prices = _prices(cat[gtype])
        for interruptible in (False, True):
            cells = []
            for duty in MATRIX_DUTIES:
                hpd = duty * 24
                best = pricing.best_tier(hpd, interruptible, prices, gtype, DAYS, lifetime)
                code = short.get(best["tier"]) or "R" + best["term"][0]
                old = pricing.recommend_tier(hpd, interruptible)
                differs = (best["tier"] != old)
                diffs += differs
                cells.append(code + ("*" if differs else ""))
            matrix[(gtype, interruptible)] = cells
    return {"matrix": matrix, "diffs": diffs}


def lifetime_sensitivity(jobs: list, cat: dict) -> list:
    """Với job inference (chạy dài), gói mua đổi thế nào khi vòng đời workload ngắn lại."""
    out = []
    for j in jobs:
        if j["kind"] != "infer":
            continue
        prices = _prices(cat[j["gpu_type"]])
        picks = {}
        for life in SENSITIVITY_LIFETIMES:
            picks[life] = _label(pricing.best_tier(num(j["hours_per_day"]), bool(int(num(j["interruptible"]))),
                                                   prices, j["gpu_type"], num(j["days"]), life))
        out.append({"job_id": j["job_id"], "gpu_type": j["gpu_type"], "picks": picks})
    return out


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    improved = compare_policies(jobs, cat)
    matrix = tier_matrix(cat)
    sens = lifetime_sensitivity(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n== Ext 1: chính sách mua nâng cao ==")
        print(f"(1) M3 gốc (cứng 30 ngày, reserved tính theo giờ dùng): ${on_demand_monthly:,.0f} -> ${optimized_monthly:,.0f}  ({savings_pct:.1f}%)")
        print(f"(2) Quy tắc cũ, tính tiền đúng (cột days, reserved trả 24/7): ${improved['on_demand']:,.0f} -> ${improved['old_cost']:,.0f}  ({improved['old_pct']:.1f}%)")
        print(f"(3) Chính sách mới (thu hồi theo GPU, 1yr vs 3yr, days)    : ${improved['on_demand']:,.0f} -> ${improved['new_cost']:,.0f}  ({improved['new_pct']:.1f}%)")
        print(f"\n{'job':18}{'gpu':7}{'tier cũ':14}{'tier mới':14}{'on-demand':>11}{'cũ ($)':>9}{'mới ($)':>9}")
        for r in improved["rows"]:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['old']:14}{r['new']:14}{r['on_demand']:>11,.0f}{r['old_cost']:>9,.0f}{r['new_cost']:>9,.0f}")

        print("\nMa trận khuyến nghị (hàng = GPU/interruptible, cột = duty cycle; OD = on-demand, R1/R3 = reserved 1/3 năm; * = khác quy tắc cũ):")
        print(f"{'GPU':8}{'ngắt được':>10}" + "".join(f"{d:>7.0%}" for d in MATRIX_DUTIES))
        for (gtype, interruptible), cells in matrix["matrix"].items():
            print(f"{gtype:8}{'có' if interruptible else 'không':>10}" + "".join(f"{c:>7}" for c in cells))
        print(f"Số ô khác quy tắc cũ: {matrix['diffs']}/{len(matrix['matrix']) * len(MATRIX_DUTIES)}")

        print("\nGói mua thay đổi theo vòng đời workload (tháng):")
        print(f"{'job':18}{'gpu':7}" + "".join(f"{str(life) + ' th':>14}" for life in SENSITIVITY_LIFETIMES))
        for s in sens:
            print(f"{s['job_id']:18}{s['gpu_type']:7}" + "".join(f"{s['picks'][life]:>14}" for life in SENSITIVITY_LIFETIMES))

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "improved": improved, "tier_matrix": matrix, "lifetime_sensitivity": sens}


if __name__ == "__main__":
    run()
