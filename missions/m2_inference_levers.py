"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

Ext 3 (cache_is_worth_it): chỉ tính tiết kiệm từ cache khi prefix được đọc lại đủ
nhiều để bù phí ghi cache. Số lần đọc lại được ước lượng từ chính token_usage.csv.

Ext 4 (ngân sách reasoning): tách chi phí $ và năng lượng Wh của traffic
is_reasoning=1, rồi ước lượng tiết kiệm nếu giới hạn (cap) tỷ lệ reasoning.
"""

from __future__ import annotations
import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# --- Ext 3: tham số kinh tế của prompt cache (giả định minh họa, ghi rõ trong báo cáo) ---
CACHE_READ_DISCOUNT = 0.10  # đọc từ cache chỉ tính 10% giá input (giảm 90%)
CACHE_STORAGE_FEE_PER_M = (
    0.30  # kiểu Gemini: phí ghi/lưu 1 triệu token cache trong 1 giờ (USD)
)
ANTHROPIC_WRITE_SURCHARGE = (
    0.25  # kiểu Anthropic: ghi cache đắt hơn input thường 25% (chỉ để so sánh)
)

# --- Ext 4: tham số ngân sách reasoning ---
REASONING_OUTPUT_MULTIPLIER = (
    6  # data/generate.py sinh output của reasoning = output thường x 6
)
REASONING_CAP_SCENARIOS = (
    0.10,
    0.05,
    0.02,
    0.01,
)  # trần reasoning tính theo % TỔNG traffic


def estimate_cache_reads(rows) -> dict:
    """Ước lượng số lần đọc lại cache của mỗi 'prefix'.

    Giả định (nêu rõ trong báo cáo): cache sống 1 giờ; mỗi nhóm (team, route_tier, giờ)
    dùng chung một prefix -> request đầu tiên của nhóm GHI cache, các request sau ĐỌC.
    Vậy số lần đọc của nhóm = số request trong nhóm - 1.
    """
    count = defaultdict(int)
    for r in rows:
        count[
            (r["team"], r["route_tier"], r["ts"][11:13])
        ] += 1  # ts[11:13] = giờ trong ngày
    return {key: n - 1 for key, n in count.items()}


def optimized_request_cost(r, out_tokens, reads_by_group, write_cost) -> float:
    """Chi phí 1 request sau tối ưu (cascade + cache có điều kiện + batch) với số output cho trước.

    Cho phép truyền out_tokens khác thực tế để hỏi "nếu request này không dùng reasoning thì tốn bao nhiêu?".
    """
    tier = r["route_tier"]
    pin, pout = MODEL_PRICES[tier]
    key = (r["team"], tier, r["ts"][11:13])
    worth = pricing.cache_is_worth_it(
        reads_by_group[key], write_cost[tier], CACHE_READ_DISCOUNT
    )
    cached = int(num(r["cached_input_tokens"])) if worth else 0
    return pricing.request_cost(
        int(num(r["input_tokens"])),
        out_tokens,
        pin,
        pout,
        cached_in=cached,
        batch=bool(int(num(r["is_batch"]))),
    )


def lever_breakdown(rows, opt_cost_all: float) -> list:
    """Chi phí/ngày và $/1M-token khi bật TỪNG đòn bẩy riêng lẻ (so với baseline: model lớn, không cache, không batch).

    Các đòn bẩy nhân vào nhau nên phần tiết kiệm riêng lẻ KHÔNG cộng lại bằng mức tiết kiệm khi bật cả ba.
    Dòng "cả ba" lấy đúng chi phí tối ưu của mission (đã có cache_is_worth_it); cache riêng lẻ ở đây chưa lọc điều kiện.
    """
    def total(cascade: bool, cache: bool, batch: bool) -> float:
        usd = 0.0
        for r in rows:
            inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
            pin, pout = MODEL_PRICES[r["route_tier"] if cascade else "large"]
            usd += pricing.request_cost(
                inp, out, pin, pout,
                cached_in=int(num(r["cached_input_tokens"])) if cache else 0,
                batch=bool(int(num(r["is_batch"]))) if batch else False,
            )
        return usd

    tokens = sum(int(num(r["input_tokens"])) + int(num(r["output_tokens"])) for r in rows)
    base = total(False, False, False)
    scenarios = [
        ("Baseline (model lớn, không cache, không batch)", base),
        ("Chỉ cascade", total(True, False, False)),
        ("Chỉ prompt cache", total(False, True, False)),
        ("Chỉ batch", total(False, False, True)),
        ("Cả ba (có cache_is_worth_it)", opt_cost_all),
    ]
    return [
        {"name": name, "usd_per_day": usd, "usd_per_m": pricing.dollars_per_million(usd, tokens),
         "savings_pct": (1 - usd / base) * 100 if base else 0.0}
        for name, usd in scenarios
    ]


def analyze_reasoning(rows, reads_by_group, write_cost) -> dict:
    """Ext 4: tách $ và Wh của traffic reasoning, rồi mô phỏng việc giới hạn reasoning.

    Mô phỏng "hạ cấp": một request reasoning bị chuyển thành request thường thì output
    giảm REASONING_OUTPUT_MULTIPLIER lần và hệ số năng lượng x80 biến mất. Giả định các
    request bị hạ cấp được chọn ngẫu nhiên nên tiết kiệm tỷ lệ tuyến tính với số bị hạ cấp.
    """
    grp = {
        False: {"n": 0, "usd": 0.0, "wh": 0.0, "tokens": 0},
        True: {"n": 0, "usd": 0.0, "wh": 0.0, "tokens": 0},
    }
    demote_usd = demote_wh = (
        0.0  # tổng tiền/điện tiết kiệm nếu HẠ CẤP TOÀN BỘ reasoning
    )
    for r in rows:
        is_r = bool(int(num(r["is_reasoning"])))
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        usd = optimized_request_cost(r, out, reads_by_group, write_cost)
        wh = sustainability.wh_per_query(inp + out, is_reasoning=is_r)
        g = grp[is_r]
        g["n"] += 1
        g["usd"] += usd
        g["wh"] += wh
        g["tokens"] += inp + out
        if is_r:
            out_plain = out // REASONING_OUTPUT_MULTIPLIER
            demote_usd += usd - optimized_request_cost(
                r, out_plain, reads_by_group, write_cost
            )
            demote_wh += wh - sustainability.wh_per_query(
                inp + out_plain, is_reasoning=False
            )

    total = {k: grp[False][k] + grp[True][k] for k in ("n", "usd", "wh", "tokens")}
    r_, p_ = grp[True], grp[False]
    summary = {
        "reasoning_requests": r_["n"],
        "traffic_share": r_["n"] / total["n"],
        "cost_share": r_["usd"] / total["usd"],
        "energy_share": r_["wh"] / total["wh"],
        "usd_per_req_reasoning": r_["usd"] / r_["n"],
        "usd_per_req_plain": p_["usd"] / p_["n"],
        "wh_per_req_reasoning": r_["wh"] / r_["n"],
        "wh_per_req_plain": p_["wh"] / p_["n"],
        "total_usd": total["usd"],
        "total_wh": total["wh"],
        "reasoning_usd": r_["usd"],
        "reasoning_wh": r_["wh"],
        "plain_usd": p_["usd"],
        "plain_wh": p_["wh"],
        "plain_requests": p_["n"],
    }
    # kịch bản trần: giữ lại tối đa cap * tổng request cho reasoning, phần dư hạ cấp
    scenarios = []
    for cap in REASONING_CAP_SCENARIOS:
        keep = min(1.0, cap * total["n"] / r_["n"])
        scenarios.append(
            {
                "cap": cap,
                "keep_frac": keep,
                "saved_usd": (1 - keep) * demote_usd,
                "saved_wh": (1 - keep) * demote_wh,
                "saved_usd_pct": (1 - keep) * demote_usd / total["usd"] * 100,
                "saved_wh_pct": (1 - keep) * demote_wh / total["wh"] * 100,
            }
        )
    summary["scenarios"] = scenarios
    return summary


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    reads_by_group = estimate_cache_reads(rows)
    # phụ phí ghi quy về "bội số giá input" của từng tier: model rẻ thì phí cố định chiếm tỷ lệ lớn hơn
    write_cost = {
        tier: CACHE_STORAGE_FEE_PER_M / price[0] for tier, price in MODEL_PRICES.items()
    }

    base_cost = opt_cost = opt_cost_no_gate = 0.0
    total_tokens = 0
    skipped_requests = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching (chỉ khi đáng), batch API
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        key = (r["team"], tier, r["ts"][11:13])
        if not pricing.cache_is_worth_it(
            reads_by_group[key], write_cost[tier], CACHE_READ_DISCOUNT
        ):
            skipped_requests += 1
        # bản cũ (luôn tính cache) để so sánh trước/sau; bản mới chỉ tính cache khi đáng
        opt_cost_no_gate += pricing.request_cost(
            inp, out, pin, pout, cached_in=cached, batch=is_batch
        )
        opt_cost += optimized_request_cost(r, out, reads_by_group, write_cost)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # thống kê cache theo tier: ngưỡng hòa vốn vs số lần đọc thực tế
    cache_stats = {}
    for tier in MODEL_PRICES:
        reads = [n for (_, t, _), n in reads_by_group.items() if t == tier]
        be = pricing.cache_break_even_reads(write_cost[tier], CACHE_READ_DISCOUNT)
        cache_stats[tier] = {
            "write_cost": round(write_cost[tier], 3),
            "break_even_reads": round(be, 3),
            "mean_reads": round(sum(reads) / len(reads), 2),
            "groups": len(reads),
            "groups_ok": sum(
                1
                for n in reads
                if pricing.cache_is_worth_it(n, write_cost[tier], CACHE_READ_DISCOUNT)
            ),
        }

    reasoning = analyze_reasoning(rows, reads_by_group, write_cost)
    levers = lever_breakdown(rows, opt_cost)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print("\nĐóng góp từng đòn bẩy (bật riêng lẻ):")
        print(f"{'kịch bản':48}{'$/ngày':>9}{'$/1M-token':>12}{'tiết kiệm':>11}")
        for lv in levers:
            print(f"{lv['name']:48}{lv['usd_per_day']:>9.2f}{lv['usd_per_m']:>12.3f}{lv['savings_pct']:>10.1f}%")
        print()
        print(
            f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive"
        )
        print("\n== Ext 3: cache_is_worth_it ==")
        print(
            f"{'tier':7}{'giá in':>8}{'write_cost':>12}{'hòa vốn':>10}{'đọc TB':>9}{'nhóm đạt':>11}"
        )
        for tier, s in cache_stats.items():
            print(
                f"{tier:7}{'$' + format(MODEL_PRICES[tier][0], '.2f'):>8}{s['write_cost']:>12}"
                f"{s['break_even_reads']:>10}{s['mean_reads']:>9}{str(s['groups_ok']) + '/' + str(s['groups']):>11}"
            )
        be_a = pricing.cache_break_even_reads(
            ANTHROPIC_WRITE_SURCHARGE, CACHE_READ_DISCOUNT
        )
        print(
            f"(tham chiếu kiểu Anthropic, phụ phí ghi {ANTHROPIC_WRITE_SURCHARGE:.0%}: hòa vốn {be_a:.2f} lần đọc, mọi tier như nhau)"
        )
        print(f"request bị bỏ chiết khấu cache: {skipped_requests}/{len(rows)}")
        print(
            f"chi phí/ngày: luôn tính cache ${opt_cost_no_gate:.4f} -> chỉ khi đáng ${opt_cost:.4f}  "
            f"(+${opt_cost - opt_cost_no_gate:.4f})"
        )

        rs = reasoning
        print("\n== Ext 4: ngân sách reasoning ==")
        print(
            f"{'nhóm':11}{'request':>9}{'%traffic':>10}{'$/ngày':>9}{'%chi phí':>10}{'Wh/ngày':>10}{'%năng lượng':>13}{'Wh/req':>9}"
        )
        print(
            f"{'thường':11}{rs['plain_requests']:>9}{(1 - rs['traffic_share']):>10.1%}{rs['plain_usd']:>9.2f}"
            f"{(1 - rs['cost_share']):>10.1%}{rs['plain_wh']:>10.1f}{(1 - rs['energy_share']):>13.1%}{rs['wh_per_req_plain']:>9.2f}"
        )
        print(
            f"{'reasoning':11}{rs['reasoning_requests']:>9}{rs['traffic_share']:>10.1%}{rs['reasoning_usd']:>9.2f}"
            f"{rs['cost_share']:>10.1%}{rs['reasoning_wh']:>10.1f}{rs['energy_share']:>13.1%}{rs['wh_per_req_reasoning']:>9.2f}"
        )
        print(
            f"1 request reasoning đắt gấp {rs['usd_per_req_reasoning'] / rs['usd_per_req_plain']:.1f}x về $ "
            f"và gấp {rs['wh_per_req_reasoning'] / rs['wh_per_req_plain']:.0f}x về Wh so với request thường"
        )
        print(
            "\nGiới hạn (cap) reasoning theo % tổng traffic, phần vượt trần bị hạ cấp thành request thường:"
        )
        print(
            f"{'trần':>6}{'giữ lại':>10}{'tiết kiệm $/ngày':>19}{'(%)':>8}{'tiết kiệm Wh/ngày':>20}{'(%)':>8}"
        )
        for s in rs["scenarios"]:
            print(
                f"{s['cap']:>6.0%}{s['keep_frac']:>10.0%}{s['saved_usd']:>19.4f}{s['saved_usd_pct']:>7.1f}%"
                f"{s['saved_wh']:>20.1f}{s['saved_wh_pct']:>7.1f}%"
            )
        print(
            f"(hiện reasoning chiếm {rs['traffic_share']:.1%} traffic nên trần 10% chưa có tác dụng)"
        )

    return {
        "baseline_daily": round(base_cost, 2),
        "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3),
        "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1),
        "total_tokens": total_tokens,
        "optimized_daily_no_gate": round(opt_cost_no_gate, 4),
        "cache_stats": cache_stats,
        "cache_skipped_requests": skipped_requests,
        "reasoning": reasoning, "levers": levers,
    }


if __name__ == "__main__":
    run()
