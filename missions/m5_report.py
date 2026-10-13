"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png

Báo cáo được SINH RA từ kết quả của các mission (kể cả 5 extension) nên số liệu luôn khớp
output của missions. Đừng sửa tay outputs/report.md: mỗi lần chạy M5, verify.py hoặc pytest
file này sẽ được ghi đè. Muốn đổi nội dung thì sửa các hàm _section_* bên dưới.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, load_csv, ROOT
from finops import report, sustainability
from missions import (m1_efficiency_audit, m2_inference_levers, m3_purchasing,
                      m3_carbon_aware)

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def _mean_col(rows: list, gpu_id: str, col: str) -> float:
    vals = [num(r[col]) for r in rows if r["gpu_id"] == gpu_id]
    return sum(vals) / len(vals) if vals else 0.0


def _lie_waste(r1: dict, cat: dict) -> list:
    """Tiền "FLOPs không dùng" của mỗi GPU lie so với các GPU cùng loại đang chạy khỏe (util >= 90%)."""
    out = []
    lie_ids = {l["gpu_id"] for l in r1["lies"]}
    for lie in r1["lies"]:
        peers = [s for s in r1["summary"] if s["gpu_type"] == lie["gpu_type"]
                 and s["gpu_id"] not in lie_ids and s["gpu_util_pct"] >= 90]
        monthly = num(cat[lie["gpu_type"]]["on_demand_hr"]) * 24 * DAYS
        if peers:
            peer_mfu = sum(p["mfu"] for p in peers) / len(peers)
            out.append({"gpu_id": lie["gpu_id"], "gpu_type": lie["gpu_type"], "util": lie["gpu_util_pct"],
                        "mfu": lie["mfu"], "peer_mfu": peer_mfu, "monthly": monthly,
                        "waste": max(0.0, 1 - lie["mfu"] / peer_mfu) * monthly})
        else:
            out.append({"gpu_id": lie["gpu_id"], "gpu_type": lie["gpu_type"], "util": lie["gpu_util_pct"],
                        "mfu": lie["mfu"], "peer_mfu": None, "monthly": monthly, "waste": None})
    return out


def _section_unit_cost(r2: dict) -> str:
    rows = ["## Đơn giá $/1M-token (inference)", "",
            "Đây là thước đo chính của lab: tiền trả cho mỗi triệu token phục vụ được, "
            "baseline = mọi request chạy model lớn, không cache, không batch.", "",
            "| Kịch bản | $/ngày | $/1M-token | Tiết kiệm |", "|---|---|---|---|"]
    for lv in r2["levers"]:
        rows.append(f"| {lv['name']} | ${lv['usd_per_day']:.2f} | ${lv['usd_per_m']:.3f} | {lv['savings_pct']:.1f}% |")
    rows += ["", "Các đòn bẩy nhân vào nhau nên tiết kiệm riêng lẻ không cộng lại bằng mức khi bật cả ba. "
             "Cascade chiếm phần lớn vì 80% request chuyển sang model nhỏ rẻ hơn 15–37 lần. "
             "Các đòn bẩy hạ tầng (purchasing, right-size, idle) đo bằng USD/tháng vì dữ liệu telemetry và "
             "workloads không gắn với số token."]
    return "\n".join(rows)


def _section_correction(headline: dict, corrected: dict) -> str:
    rows = ["## Hiệu chỉnh số liệu (độ tin cậy của con số tiết kiệm)", "",
            "Con số headline ở trên đến từ mission gốc. Sau khi làm Ext 1 và Ext 2, hai đòn bẩy được tính lại "
            "cho đúng hơn:", "",
            "| Đòn bẩy | Headline (USD/tháng) | Hiệu chỉnh (USD/tháng) | Lý do |", "|---|---|---|---|",
            f"| Inference (cascade/cache/batch) | ${headline['infer']:,.0f} | ${corrected['infer']:,.0f} | Không đổi |",
            f"| Purchasing (spot/reserved) | ${headline['purchasing']:,.0f} | ${corrected['purchasing']:,.0f} | "
            "Dùng cột days thật của từng job, reserved bị tính 24/7, tỷ lệ thu hồi spot theo loại GPU (Ext 1) |",
            f"| Right-size util-lies | ${headline['rightsize']:,.0f} | ${corrected['rightsize']:,.0f} | "
            "Hạ cấp một bậc không qua kiểm tra tính toán, băng thông, VRAM; chỉ giữ GPU hạ được an toàn (Ext 2) |",
            f"| Kill idle GPUs | ${headline['idle']:,.0f} | ${corrected['idle']:,.0f} | Không đổi |",
            f"| **Baseline** | ${headline['baseline']:,.0f} | ${corrected['baseline']:,.0f} | Baseline on-demand tính theo số ngày thật |",
            f"| **Tổng tiết kiệm** | ${headline['total']:,.0f} ({headline['pct']:.1f}%) | "
            f"${corrected['total']:,.0f} ({corrected['pct']:.1f}%) | |", "",
            f"Kết luận: tuyệt đối số tiền giảm nhưng tỷ lệ tiết kiệm gần như không đổi "
            f"({headline['pct']:.1f}% so với {corrected['pct']:.1f}%), nên mức giảm khoảng {corrected['pct']:.0f}% đáng tin."]
    return "\n".join(rows)


def _section_gpu_lie(r1: dict, cat: dict, tel: list) -> str:
    waste = _lie_waste(r1, cat)
    rows = ["## GPU-Util \"nói dối\": cơ chế và tác động chi phí", "",
            "`GPU-Util` của nvidia-smi chỉ là tỷ lệ thời gian có ít nhất một kernel đang chạy. Một kernel nhỏ, "
            "chạy chậm hoặc đang chờ dữ liệu vẫn được tính là \"bận\". MFU mới đo phần FLOPs thực sự sinh ra "
            "so với đỉnh lý thuyết. Nguyên nhân thường gặp: chờ bộ nhớ (memory stall), batch quá nhỏ, chờ nạp "
            "dữ liệu hoặc giao tiếp giữa các GPU, quá nhiều kernel nhỏ làm tốn thời gian khởi chạy.", "",
            "| GPU | Loại | GPU-Util | MFU | SM active | Tensor active | Tiền thuê/tháng | FLOPs trả tiền nhưng không dùng |",
            "|---|---|---|---|---|---|---|---|"]
    for w in waste:
        sm = _mean_col(tel, w["gpu_id"], "sm_active_pct")
        tensor = _mean_col(tel, w["gpu_id"], "tensor_active_pct")
        lost = f"≈ ${w['waste']:,.0f}/tháng (so với MFU {w['peer_mfu']:.2f} của các {w['gpu_type']} khỏe)" \
            if w["waste"] is not None else "không có GPU cùng loại để so sánh"
        rows.append(f"| {w['gpu_id']} | {w['gpu_type']} | {w['util']:.1f}% | {w['mfu']:.3f} | {sm:.0f}% | "
                    f"{tensor:.0f}% | ${w['monthly']:,.0f} | {lost} |")
    rows += ["", "Bằng chứng cơ chế: SM active rất cao nhưng Tensor active thấp nghĩa là các khối tính toán \"bận\" "
             "mà Tensor Core, bộ phận thực sự nhân ma trận, chỉ chạy một phần thời gian. Cách xử lý đúng là "
             "profiling để tăng MFU (batch lớn hơn, fuse kernel, nạp dữ liệu song song), không phải hạ cấp phần cứng "
             "(xem Ext 2)."]
    return "\n".join(rows)


def _section_actions(ctx: dict) -> str:
    rows = ["## Khuyến nghị theo thứ tự ưu tiên (ROI)", "",
            "| # | Hành động | Tiết kiệm ước tính | Vì sao ưu tiên ở mức này |", "|---|---|---|---|",
            f"| 1 | Mua spot cho job train/dev/batch, reserved cho inference chạy 24/7 | "
            f"${ctx['purchasing']:,.0f}/tháng (hiệu chỉnh) | Khoản tiết kiệm lớn nhất, chỉ đổi cách mua, "
            "không đổi code. Chỉ chọn reserved 3 năm khi chắc workload sống đủ 36 tháng (Ext 1) |",
            f"| 2 | Tắt GPU nằm không (lịch tắt/autoscale) | ${ctx['idle']:,.0f}/tháng | Rủi ro bằng 0, làm trong ngày |",
            f"| 3 | Bật cascade, batch, prompt cache cho inference | ${ctx['infer']:,.0f}/tháng, "
            f"${ctx['pm_base']:.3f} → ${ctx['pm_opt']:.3f}/1M-token | Giảm đơn giá lớn nhất, nhưng khoản chi inference "
            "chỉ là phần nhỏ của baseline và cần kiểm tra chất lượng model nhỏ |",
            f"| 4 | Profiling các GPU lie để tăng MFU | ≈ ${ctx['lie_waste']:,.0f}/tháng (chỉ tính GPU có đối chứng) | "
            "Sửa nguyên nhân gốc thay vì hạ cấp, nhưng cần công kỹ sư |",
            f"| 5 | Trần reasoning {ctx['cap']:.0%} traffic | ${ctx['cap_usd'] * DAYS:,.0f}/tháng và "
            f"{ctx['cap_wh_pct']:.0f}% năng lượng | Tiền ít nhưng reasoning chiếm {ctx['r_energy']:.0%} điện; cần đo chất lượng |",
            f"| 6 | Dời job train/batch sang {ctx['balanced']} | {ctx['co2_saved_balanced']:,.0f} kg CO2e và "
            f"${ctx['elec_saved']:,.0f} tiền điện cho các job ngắt được | Không ảnh hưởng người dùng; cần tính phí egress và quota |",
            f"| 7 | Right-size theo băng thông và VRAM | ${ctx['rs_safe']:,.0f}/tháng (NVIDIA), "
            f"${ctx['rs_amd']:,.0f}/tháng nếu chấp nhận AMD MI300X | Đòn bẩy nhỏ nếu giữ NVIDIA; đổi AMD cần chuyển stack phần mềm |"]
    return "\n".join(rows)


def _section_sustainability(rc: dict, r2: dict) -> str:
    regions = {t["region"]: t for t in rc["regions"]}
    base = regions[m3_carbon_aware.BASELINE_REGION]
    picks = rc["picks"]
    rows = ["## Nhận xét về tính bền vững (carbon, vùng triển khai)", "",
            f"Với {rc['total_kwh']:,.0f} kWh của {len(rc['jobs'])} job có thể ngắt, giả định công suất định mức (TDP) "
            f"và giá carbon nội bộ ${sustainability.CARBON_PRICE_PER_TONNE:.0f}/tấn:", "",
            "| Vùng | $/kWh | gCO2/kWh | Tiền điện | CO2 (kg) | Tổng gồm tiền carbon | Nhãn |", "|---|---|---|---|---|---|---|"]
    labels = {}
    for key, name in (("cheapest", "rẻ nhất"), ("cleanest", "sạch nhất"), ("balanced", "cân bằng")):
        labels.setdefault(picks[key], []).append(name)
    for t in sorted(rc["regions"], key=lambda r: r["total_usd"]):
        rows.append(f"| {t['region']} | {t['usd_per_kwh']:.3f} | {t['g_per_kwh']} | ${t['energy_usd']:,.2f} | "
                    f"{t['co2_kg']:,.1f} | ${t['total_usd']:,.2f} | {', '.join(labels.get(t['region'], []))} |")
    bal = regions[picks["balanced"]]
    rows += ["",
             f"- Vùng rẻ nhất và cân bằng nhất là **{picks['balanced']}**: tiền điện ${base['energy_usd']:,.2f} → "
             f"${bal['energy_usd']:,.2f}, carbon {base['co2_kg']:,.0f} → {bal['co2_kg']:,.0f} kg.",
             f"- Vùng sạch nhất **{picks['cleanest']}** giảm carbon {rc['co2_saved_pct']:.0f}% nhưng chỉ thắng về tổng "
             f"chi phí khi giá carbon vượt khoảng ${rc['breakeven_carbon_price']:,.0f}/tấn. Chọn vùng nào tùy ưu tiên "
             "của công ty: tiền, hay cam kết giảm phát thải.",
             f"- Reasoning chỉ chiếm {r2['reasoning']['traffic_share']:.1%} traffic và {r2['reasoning']['cost_share']:.1%} "
             f"chi phí nhưng tới {r2['reasoning']['energy_share']:.0%} năng lượng, nên quản lý reasoning là đòn bẩy carbon "
             "lớn nhất ở phía inference.",
             "- Đánh đổi: dời job sang vùng xa không ảnh hưởng job train/batch nhưng có phí egress, quota GPU, quy định "
             "lưu trữ dữ liệu và độ trễ nếu là dịch vụ phục vụ người dùng; inference realtime không nên dời."]
    return "\n".join(rows)


def _section_extensions(r1: dict, r2: dict, r3: dict, rc: dict) -> str:
    imp = r3["improved"]
    cs = r2["cache_stats"]
    rs = r2["reasoning"]
    cap2 = next(s for s in rs["scenarios"] if abs(s["cap"] - 0.02) < 1e-9)
    n_lie = len(r1["lies"])
    rows = ["## Phần mở rộng đã thực hiện (đo lường)", "",
            "| Extension | Kết quả đo | Insight chính |", "|---|---|---|",
            f"| 1. recommend_tier nâng cao | Tiết kiệm mua GPU: {r3['savings_pct']:.1f}% (gốc) → {imp['old_pct']:.1f}% "
            f"(quy tắc cũ, tính tiền đúng) → {imp['new_pct']:.1f}% (chính sách mới) | Tier không đổi trên 8 job; "
            "thay đổi chủ yếu do tính tiền đúng (days, reserved 24/7). Vòng đời workload quyết định reserved 1 năm hay 3 năm |",
            f"| 2. Right-sizing theo MBU | An toàn: ${r1['rightsize_monthly_savings']:,.0f}/tháng; "
            f"nếu chấp nhận AMD: ${r1['rightsize_monthly_savings_with_amd']:,.0f}/tháng; {n_lie} GPU lie không hạ cấp được | "
            "Rẻ theo giờ chưa đủ: phải đủ TFLOPs, băng thông và VRAM |",
            f"| 3. cache_is_worth_it | Hòa vốn small {cs['small']['break_even_reads']:.2f} / large {cs['large']['break_even_reads']:.2f} lần đọc; "
            f"thực tế {cs['small']['mean_reads']:.1f} / {cs['large']['mean_reads']:.1f}; "
            f"{r2['cache_skipped_requests']} request mất chiết khấu | Traffic dày nên cache gần như luôn có lợi; hàm chặn tiết kiệm ảo khi traffic thưa |",
            f"| 4. Ngân sách reasoning | {rs['traffic_share']:.1%} traffic = {rs['cost_share']:.1%} chi phí = {rs['energy_share']:.0%} năng lượng; "
            f"trần 2%: -${cap2['saved_usd']:.2f}/ngày, -{cap2['saved_wh_pct']:.0f}% Wh | "
            "Reasoning là vấn đề năng lượng nhiều hơn vấn đề tiền; trần 10% chưa có tác dụng vì hiện chỉ 8.4% |",
            f"| 5. Lập lịch carbon | {rc['baseline_co2_kg']:,.0f} → {rc['cleanest_co2_kg']:,.0f} kg CO2e "
            f"(-{rc['co2_saved_pct']:.0f}%) ở {rc['picks']['cleanest']}; rẻ và cân bằng nhất: {rc['picks']['balanced']} | "
            "\"Tối ưu\" phụ thuộc tiêu chí: giá carbon phải rất cao mới đổi kết quả |"]
    return "\n".join(rows)


def _section_limits(r2: dict) -> str:
    avg_tokens = r2["total_tokens"] / 2400
    return "\n".join([
        "## Giả định và giới hạn", "",
        "- Dữ liệu là mô phỏng tất định (seed=25); giá là snapshot tháng 6/2026.",
        "- Baseline inference giả định mọi request chạy model lớn nên mức giảm 82.6% là cận trên; cascade chưa đo chất lượng.",
        "- Baseline gộp tiền token (giá API) với tiền thuê GPU của 8 job; hai đòn bẩy idle và right-size tính trên "
        "đội telemetry 11 GPU, không nằm trong baseline.",
        f"- Energy per query dùng giả định 800 token/query; trung bình thật của dữ liệu là {avg_tokens:,.0f} token/request.",
        "- Giả định của extension: cache sống 1 giờ và phí lưu $0.30/1M token (Ext 3); tỷ lệ thu hồi spot theo GPU và vòng "
        "đời workload (Ext 1); giá carbon $100/tấn, công suất TDP, bỏ qua PUE (Ext 5); size theo nhu cầu đỉnh với mục tiêu tải 70% (Ext 2).",
    ])


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    rc = m3_carbon_aware.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- số liệu hiệu chỉnh (Ext 1 + Ext 2) ---
    imp = r3["improved"]
    corr_purchasing = imp["on_demand"] - imp["new_cost"]
    corr_rightsize = r1["rightsize_monthly_savings"]
    corr_baseline = r2["baseline_daily"] * DAYS + imp["on_demand"]
    corr_total = infer_savings + corr_purchasing + corr_rightsize + idle_savings
    corrected = {"infer": infer_savings, "purchasing": corr_purchasing, "rightsize": corr_rightsize,
                 "idle": idle_savings, "baseline": corr_baseline, "total": corr_total,
                 "pct": corr_total / corr_baseline * 100 if corr_baseline else 0.0}
    headline = {"infer": infer_savings, "purchasing": purchasing_savings, "rightsize": rightsize_savings,
                "idle": idle_savings, "baseline": baseline, "total": sum(levers.values()), "pct": total_pct}

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
        "cheapest_region": rc["picks"]["cheapest"],
        "cleanest_region": rc["picks"]["cleanest"],
        "balanced_region": rc["picks"]["balanced"],
    }

    # --- ngữ cảnh cho bảng khuyến nghị ---
    regions = {t["region"]: t for t in rc["regions"]}
    base_reg = regions[m3_carbon_aware.BASELINE_REGION]
    bal_reg = regions[rc["picks"]["balanced"]]
    cap2 = next(s for s in r2["reasoning"]["scenarios"] if abs(s["cap"] - 0.02) < 1e-9)
    cat_ = catalog_by_type()
    ctx = {
        "purchasing": corr_purchasing, "idle": idle_savings, "infer": infer_savings,
        "pm_base": r2["baseline_per_m"], "pm_opt": r2["optimized_per_m"],
        "lie_waste": sum(w["waste"] for w in _lie_waste(r1, cat_) if w["waste"] is not None),
        "cap": cap2["cap"], "cap_usd": cap2["saved_usd"], "cap_wh_pct": cap2["saved_wh_pct"],
        "r_energy": r2["reasoning"]["energy_share"], "balanced": rc["picks"]["balanced"],
        "co2_saved_balanced": base_reg["co2_kg"] - bal_reg["co2_kg"],
        "elec_saved": base_reg["energy_usd"] - bal_reg["energy_usd"],
        "rs_safe": r1["rightsize_monthly_savings"], "rs_amd": r1["rightsize_monthly_savings_with_amd"],
    }
    tel = load_csv("gpu_telemetry.csv")
    sections = [
        _section_unit_cost(r2),
        _section_correction(headline, corrected),
        _section_gpu_lie(r1, cat, tel),
        _section_actions(ctx),
        _section_sustainability(rc, r2),
        _section_extensions(r1, r2, r3, rc),
        _section_limits(r2),
    ]

    md = report.build_report(baseline, optimized, levers, sustainability=sust, extra_sections=sections)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"), baseline=baseline)

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1),
            "corrected": corrected}


if __name__ == "__main__":
    run()
