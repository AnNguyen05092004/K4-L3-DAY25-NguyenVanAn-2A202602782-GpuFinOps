"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""

from __future__ import annotations
import math


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,  # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,  # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    prices: dict | None = None,
    gpu_type: str | None = None,
    active_days: float = 30,
    lifetime_months: float = 36,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    Ext 1: nếu truyền `prices` (bảng giá của GPU đó) thì dùng chính sách NÂNG CAO —
    chọn gói có chi phí mỗi giờ-dùng-thật thấp nhất, xét tỷ lệ bị thu hồi spot theo
    loại GPU, duty cycle theo số ngày chạy thật, và reserved 1 năm vs 3 năm theo vòng
    đời workload (xem best_tier). Không truyền `prices` thì giữ đúng quy tắc đơn giản cũ.
    """
    if prices is not None:
        return best_tier(hours_per_day, interruptible, prices, gpu_type,
                         active_days, lifetime_months)["tier"]
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,  # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (
        (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    )
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }


def cache_break_even_reads(write_cost: float, read_discount: float = 0.10) -> float:
    """Số lần ĐỌC LẠI tối thiểu để một lần ghi cache bắt đầu có lời.

    Đơn vị: giá input thường của model = 1.0.
      - write_cost    : phụ phí ghi cache trả MỘT lần (0.25 = đắt hơn input thường 25%).
      - read_discount : mỗi lần đọc từ cache chỉ trả phần này (0.10 = giảm 90%).
    Mỗi lần đọc tiết kiệm (1 - read_discount) so với đọc input thường, nên cần
    write_cost / (1 - read_discount) lần đọc để bù lại phụ phí ghi.
    """
    saving_per_read = 1.0 - read_discount
    if (
        saving_per_read <= 0
    ):  # đọc cache không rẻ hơn input thường -> không bao giờ có lời
        return float("inf")
    return max(0.0, write_cost) / saving_per_read


def cache_is_worth_it(
    avg_reads: float, write_cost: float, read_discount: float = 0.10
) -> bool:
    """True nếu trung bình mỗi prefix được đọc lại ĐỦ NHIỀU để cache có lời (so sánh nghiêm ngặt)."""
    return avg_reads > cache_break_even_reads(write_cost, read_discount)


# --- Ext 1: chính sách mua GPU nâng cao (tỷ lệ thu hồi theo GPU + reserved 1yr vs 3yr) ---

# Xác suất bị thu hồi spot mỗi giờ theo loại GPU — GIẢ ĐỊNH minh họa (GPU lớn, khan hiếm
# thường ít bị thu hồi hơn GPU nhỏ). Loại không có trong bảng dùng DEFAULT_SPOT_INTERRUPT_RATE.
SPOT_INTERRUPT_RATE = {
    "H100": 0.03, "H200": 0.03, "B200": 0.04, "A100": 0.05,
    "MI300X": 0.06, "A10G": 0.08, "L4": 0.10,
}
DEFAULT_SPOT_INTERRUPT_RATE = 0.05
RESERVED_TERMS = {"1yr": 12, "3yr": 36}   # kỳ hạn cam kết reserved (tháng)
DAYS_PER_MONTH = 30


def spot_overhead_factor(interrupt_rate: float, ckpt_overhead_frac: float = 0.03,
                         rework_hours_per_interrupt: float = 0.5) -> float:
    """Hệ số giờ-GPU thực tế trên mỗi giờ công việc khi chạy spot (checkpoint + chạy lại khi bị thu hồi).

    Cùng công thức với spot_checkpoint_cost: 1 + overhead checkpoint + tỷ lệ thu hồi x giờ chạy lại.
    """
    return 1.0 + ckpt_overhead_frac + interrupt_rate * rework_hours_per_interrupt


def reserved_effective_rate(rate_hr: float, term_months: float, lifetime_months: float) -> float:
    """Giá reserved hiệu dụng mỗi giờ khi workload chỉ sống `lifetime_months` tháng.

    Cam kết luôn phải trả đủ kỳ hạn (làm tròn lên bội số của kỳ hạn). Ví dụ workload sống 12 tháng
    mà mua 3 năm: trả 36 tháng nhưng chỉ dùng 12 -> giá hiệu dụng gấp 3 lần giá niêm yết.
    """
    if lifetime_months <= 0:
        return float("inf")
    terms = math.ceil(lifetime_months / term_months)
    return rate_hr * terms * term_months / lifetime_months


def tier_options(hours_per_day: float, interruptible: bool, prices: dict, gpu_type: str | None = None,
                 active_days: float = DAYS_PER_MONTH, lifetime_months: float = 36) -> list:
    """Chi phí mỗi GIỜ-DÙNG-THẬT của từng gói mua có thể áp dụng cho workload.

    prices cần có: on_demand_hr, spot_hr, reserved_1yr_hr, reserved_3yr_hr (số, USD/GPU-giờ).
    Reserved bị tính tiền 24/7 nên chi phí mỗi giờ-dùng = giá hiệu dụng / duty cycle
    (duty cycle = giờ dùng thật trong tháng / tổng giờ trong tháng).
    """
    duty = min(1.0, max(0.0, hours_per_day) * max(0.0, active_days) / (24.0 * DAYS_PER_MONTH))
    options = [{"tier": "on_demand", "term": None, "cost_per_used_hr": prices["on_demand_hr"]}]
    if interruptible:
        rate = SPOT_INTERRUPT_RATE.get(gpu_type, DEFAULT_SPOT_INTERRUPT_RATE)
        options.append({"tier": "spot", "term": None,
                        "cost_per_used_hr": prices["spot_hr"] * spot_overhead_factor(rate)})
    if duty > 0:
        for term, months in RESERVED_TERMS.items():
            eff = reserved_effective_rate(prices[f"reserved_{term}_hr"], months, lifetime_months)
            options.append({"tier": "reserved", "term": term, "cost_per_used_hr": eff / duty})
    return options


def best_tier(hours_per_day: float, interruptible: bool, prices: dict, gpu_type: str | None = None,
              active_days: float = DAYS_PER_MONTH, lifetime_months: float = 36) -> dict:
    """Gói mua rẻ nhất mỗi giờ-dùng-thật: {'tier', 'term', 'cost_per_used_hr'} (hòa thì ưu tiên on_demand)."""
    options = tier_options(hours_per_day, interruptible, prices, gpu_type, active_days, lifetime_months)
    return min(options, key=lambda o: o["cost_per_used_hr"])
