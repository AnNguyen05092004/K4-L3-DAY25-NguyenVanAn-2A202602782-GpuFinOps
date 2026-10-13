"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 extra_sections: list | None = None) -> str:
    """Return a markdown cost-optimization report.

    extra_sections: danh sách khối markdown (chuỗi) chèn sau phần Sustainability — dùng cho
    phần phân tích, $/1M-token và các extension do M5 sinh ra (không bị ghi đè tay).
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
        ]
        if "cheapest_region" in sustainability:
            # 3 tiêu chí tách riêng: rẻ nhất, sạch nhất, cân bằng (Ext 5)
            lines += [
                f"- Cheapest region (electricity $): {sustainability['cheapest_region']}",
                f"- Cleanest region (gCO2/kWh): {sustainability['cleanest_region']}",
                f"- Best balanced region (cost + carbon price): {sustainability['balanced_region']}",
            ]
        else:
            lines.append(f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}")
    for section in extra_sections or []:
        lines += ["", section.strip("\n")]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline: float | None = None) -> str:
    """Write a savings chart PNG. Returns the path. No-op if matplotlib absent.

    Có `baseline` -> vẽ biểu đồ waterfall thật (baseline, từng lever trừ dần, optimized).
    Không có -> biểu đồ cột đơn giản như bản gốc.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    if baseline is None:
        ax.bar(names, vals, color="#2e548a")
        ax.set_title("GPU cost savings by FinOps lever")
    else:
        optimized = baseline - sum(vals)
        labels = ["Baseline"] + names + ["Optimized"]
        ax.bar(0, baseline, color="#7a869a")
        running = baseline
        for i, v in enumerate(vals, start=1):
            ax.bar(i, v, bottom=running - v, color="#2e8b57")      # cột tiết kiệm "treo" từ mức hiện tại đi xuống
            if v >= baseline * 0.05:      # cột đủ cao: ghi số bên trong; cột thấp: ghi ngay bên dưới để không đè lên nhau
                ax.text(i, running - v / 2, f"-${v:,.0f}", ha="center", va="center", color="white", fontsize=9)
            else:
                ax.text(i, running - v - baseline * 0.01, f"-${v:,.0f}", ha="center", va="top", color="#2e8b57", fontsize=9)
            running -= v
        ax.bar(len(vals) + 1, optimized, color="#2e548a")
        ax.text(0, baseline, f"${baseline:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(len(vals) + 1, optimized, f"${optimized:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        # dấu $ phải được escape (\\$), nếu không matplotlib hiểu cặp $...$ là công thức toán
        ax.set_title(f"GPU cost waterfall: \\${baseline:,.0f} to \\${optimized:,.0f} per month "
                     f"({sum(vals) / baseline * 100:.0f}% saved)")
    ax.set_ylabel("USD / month")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
