"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py

Ext 2 (right-sizing theo MBU/băng thông): ngoài việc phát hiện GPU kém hiệu quả, đề xuất
GPU rẻ hơn CHỈ KHI nó vẫn đủ tính toán, băng thông bộ nhớ và VRAM cho nhu cầu đỉnh của
GPU hiện tại. Rẻ hơn theo $/giờ chưa đủ — phải đáp ứng cả ba ràng buộc.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

# --- Ext 2: tham số right-sizing ---
TARGET_UTIL = 0.70     # GPU mới chỉ nên chạy ≤ 70% công suất đỉnh (chừa headroom, thực tế khó đạt MFU > 60%)
VRAM_MARGIN = 0.90     # VRAM đỉnh phải ≤ 90% dung lượng GPU mới (chừa chỗ cho overhead)
AMD_TYPES = {"MI300X"}  # khác nhà cung cấp -> phải đổi stack phần mềm, chỉ xét khi bật allow_amd
DAYS_PER_MONTH = 30


def cheaper_gpu_options(demand: dict, current: str, cat: dict, target_util: float = TARGET_UTIL) -> list:
    """Với mỗi loại GPU RẺ HƠN GPU hiện tại (theo $/giờ), kiểm tra có đáp ứng nhu cầu đỉnh không.

    demand = {"tflops", "bw_tbs", "mem_gb"} là nhu cầu đỉnh trong 24 giờ.
    Một GPU mới bị loại ("blocked_by") nếu thiếu: tính toán (compute), băng thông bộ nhớ (bandwidth)
    hoặc dung lượng VRAM. Kết quả sắp theo giá tăng dần.
    """
    cur_price = num(cat[current]["on_demand_hr"])
    options = []
    for gtype, c in cat.items():
        price = num(c["on_demand_hr"])
        if price >= cur_price:
            continue
        blocked = []
        if demand["tflops"] / target_util > num(c["peak_tflops_fp16"]):
            blocked.append("compute")
        if demand["bw_tbs"] / target_util > num(c["peak_bw_tbs"]):
            blocked.append("bandwidth")
        if demand["mem_gb"] > VRAM_MARGIN * num(c["hbm_gb"]):
            blocked.append("VRAM")
        options.append({"gpu_type": gtype, "price": price, "blocked_by": blocked, "ok": not blocked})
    return sorted(options, key=lambda o: o["price"])


def recommend_rightsize(demand: dict, current: str, cat: dict,
                        target_util: float = TARGET_UTIL, allow_amd: bool = False):
    """Trả về (GPU rẻ nhất vẫn đủ nhu cầu hoặc None, danh sách mọi lựa chọn đã xét)."""
    options = [o for o in cheaper_gpu_options(demand, current, cat, target_util)
               if allow_amd or o["gpu_type"] not in AMD_TYPES]
    feasible = [o for o in options if o["ok"]]
    return (feasible[0] if feasible else None), options


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
                               "peak_tflops": 0.0, "peak_bw": 0.0, "peak_mem": 0.0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        # Ext 2: nhu cầu ĐỈNH trong 24 giờ (size theo đỉnh, không theo trung bình)
        a["peak_tflops"] = max(a["peak_tflops"], num(r["achieved_tflops"]))
        a["peak_bw"] = max(a["peak_bw"], num(r["achieved_bw_tbs"]))
        a["peak_mem"] = max(a["peak_mem"], num(r["mem_used_gb"]))
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "demand": {"tflops": a["peak_tflops"], "bw_tbs": a["peak_bw"], "mem_gb": a["peak_mem"]},
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(cat[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Ext 2: right-sizing theo nhu cầu đỉnh ---
    rightsizing = []
    for s in sorted(summary, key=lambda x: x["gpu_id"]):
        cur_price = num(cat[s["gpu_type"]]["on_demand_hr"])
        pick, options = recommend_rightsize(s["demand"], s["gpu_type"], cat)
        pick_amd, _ = recommend_rightsize(s["demand"], s["gpu_type"], cat, allow_amd=True)
        monthly = lambda p: (cur_price - p["price"]) * 24 * DAYS_PER_MONTH if p else 0.0
        rightsizing.append({
            "gpu_id": s["gpu_id"], "gpu_type": s["gpu_type"], "demand": s["demand"],
            "pick": pick["gpu_type"] if pick else None, "monthly_saving": monthly(pick),
            "pick_with_amd": pick_amd["gpu_type"] if pick_amd else None,
            "monthly_saving_with_amd": monthly(pick_amd),
            "saving_pct": (cur_price - pick["price"]) / cur_price * 100 if pick else 0.0,
            "options": options,
        })
    fleet_monthly = sum(num(cat[s["gpu_type"]]["on_demand_hr"]) for s in summary) * 24 * DAYS_PER_MONTH
    rs_total = sum(r["monthly_saving"] for r in rightsizing)
    rs_total_amd = sum(r["monthly_saving_with_amd"] for r in rightsizing)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n== Ext 2: Right-sizing theo băng thông (MBU) và VRAM ==")
        print("Bảng giá theo dung lượng và băng thông (catalog):")
        print(f"{'GPU':8}{'$/giờ':>7}{'VRAM GB':>9}{'$/GB-VRAM':>11}{'BW TB/s':>9}{'$/(TB/s)':>10}")
        for g, c in sorted(cat.items(), key=lambda x: num(x[1]["on_demand_hr"]) / num(x[1]["hbm_gb"])):
            p, hbm, bw = num(c["on_demand_hr"]), num(c["hbm_gb"]), num(c["peak_bw_tbs"])
            print(f"{g:8}{p:>7.2f}{hbm:>9.0f}{p / hbm:>11.4f}{bw:>9.2f}{p / bw:>10.2f}")
        print(f"\nNhu cầu ĐỈNH 24h của từng GPU; GPU mới phải chạy ≤ {TARGET_UTIL:.0%} công suất đỉnh, VRAM ≤ {VRAM_MARGIN:.0%}.")
        print("(✗ = bị loại: TF = thiếu TFLOPs, BW = thiếu băng thông, VRAM = thiếu dung lượng)")
        print(f"{'GPU':12}{'type':7}{'TFLOPs':>8}{'BW':>6}{'VRAM':>6}  {'đề xuất':9}{'tiết kiệm/tháng':>17}  các GPU rẻ hơn đã xét")
        for r in rightsizing:
            d = r["demand"]
            pick = r["pick"] or "giữ nguyên"
            save = f"${r['monthly_saving']:,.0f} ({r['saving_pct']:.0f}%)" if r["pick"] else "-"
            short = {"compute": "TF", "bandwidth": "BW", "VRAM": "VRAM"}
            notes = " ".join(o["gpu_type"] + ("✓" if o["ok"] else "✗" + "+".join(short[b] for b in o["blocked_by"]))
                             for o in r["options"] if o["gpu_type"] not in AMD_TYPES)
            amd = f" | {r['pick_with_amd']}✓(AMD)" if r["pick_with_amd"] in AMD_TYPES else ""
            print(f"{r['gpu_id']:12}{r['gpu_type']:7}{d['tflops']:>8.0f}{d['bw_tbs']:>6.2f}{d['mem_gb']:>6.0f}  {pick:9}{save:>17}  {notes}{amd}")
        print(f"\nRight-size (chỉ NVIDIA): tiết kiệm ${rs_total:,.0f}/tháng = {rs_total / fleet_monthly:.1%} chi phí đội GPU (${fleet_monthly:,.0f}/tháng)")
        print(f"Nếu chấp nhận đổi sang AMD MI300X: tiết kiệm ${rs_total_amd:,.0f}/tháng = {rs_total_amd / fleet_monthly:.1%} (cần chuyển stack phần mềm)")
        for l in lies:
            r = next(x for x in rightsizing if x["gpu_id"] == l["gpu_id"])
            verdict = f"hạ xuống {r['pick']}" if r["pick"] else "KHÔNG hạ cấp được"
            why = "; ".join(f"{o['gpu_type']} {'+'.join(o['blocked_by'])}" for o in r["options"]
                            if not o["ok"] and o["gpu_type"] not in AMD_TYPES)
            print(f"GPU 'lie' {l['gpu_id']} ({l['gpu_type']}): {verdict}" + (f" — bị loại: {why}" if why else ""))

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsizing": rightsizing, "rightsize_monthly_savings": round(rs_total, 2),
            "rightsize_monthly_savings_with_amd": round(rs_total_amd, 2), "fleet_monthly": round(fleet_monthly, 2)}


if __name__ == "__main__":
    run()
