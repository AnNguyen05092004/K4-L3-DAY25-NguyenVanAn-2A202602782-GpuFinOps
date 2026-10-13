"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


# Giá carbon nội bộ (USD / tấn CO2) — giả định minh họa để quy carbon ra tiền khi so sánh vùng.
CARBON_PRICE_PER_TONNE = 100.0


def region_comparison(kwh: float, carbon_price_per_tonne: float = CARBON_PRICE_PER_TONNE) -> list:
    """Bảng so sánh mọi vùng cho cùng một lượng điện (kWh).

    Mỗi dòng: giá điện, cường độ carbon, tiền điện thực tế, carbon thực tế (kg CO2),
    và tổng chi phí đã cộng 'tiền carbon' (carbon_kg x giá carbon) để tìm vùng cân bằng.
    """
    rows = []
    for region in REGION_CARBON:
        wh = kwh * 1000.0
        energy_usd = energy_cost_usd(wh, region)
        co2_kg = carbon_g(wh, region) / 1000.0
        rows.append({
            "region": region,
            "usd_per_kwh": REGION_PRICE_KWH[region],
            "g_per_kwh": REGION_CARBON[region],
            "energy_usd": energy_usd,
            "co2_kg": co2_kg,
            "total_usd": energy_usd + co2_kg * carbon_price_per_tonne / 1000.0,
        })
    return rows


def pick_regions(rows: list) -> dict:
    """Chọn vùng tối ưu theo 3 tiêu chí: rẻ nhất (tiền điện), sạch nhất (CO2), cân bằng (tổng gồm tiền carbon)."""
    return {
        "cheapest": min(rows, key=lambda r: r["energy_usd"])["region"],
        "cleanest": min(rows, key=lambda r: r["co2_kg"])["region"],
        "balanced": min(rows, key=lambda r: r["total_usd"])["region"],
    }
