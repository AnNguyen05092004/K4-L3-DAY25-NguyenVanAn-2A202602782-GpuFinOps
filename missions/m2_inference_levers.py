"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

Ext 3 (cache_is_worth_it): chỉ tính tiết kiệm từ cache khi prefix được đọc lại đủ
nhiều để bù phí ghi cache. Số lần đọc lại được ước lượng từ chính token_usage.csv.
"""

from __future__ import annotations
import os as _os, sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing

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
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        key = (r["team"], tier, r["ts"][11:13])
        worth = pricing.cache_is_worth_it(
            reads_by_group[key], write_cost[tier], CACHE_READ_DISCOUNT
        )
        if not worth:
            skipped_requests += 1
        # bản cũ (luôn tính cache) để so sánh trước/sau; bản mới chỉ tính cache khi đáng
        opt_cost_no_gate += pricing.request_cost(
            inp, out, pin, pout, cached_in=cached, batch=is_batch
        )
        opt_cost += pricing.request_cost(
            inp, out, pin, pout, cached_in=cached if worth else 0, batch=is_batch
        )

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

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
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
    }


if __name__ == "__main__":
    run()
