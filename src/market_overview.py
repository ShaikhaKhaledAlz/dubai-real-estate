"""
Dubai residential market overview from official DLD sales (2008 - today).

- Monthly number of sales and median price per sqft (apartments and villas)
- Off-plan share of sales by year
- Most active areas in the latest full year

Run:  python src/market_overview.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from area_labels import short

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
IMAGES = ROOT / "images"
LATEST_FULL_YEAR = 2025
TOP_N = 15


def load_monthly() -> pd.DataFrame:
    m = pd.read_csv(PROCESSED / "monthly_market.csv")
    last = m["month"].max()
    m = m[m["month"] < last]  # the newest month is still in progress when the data is extracted
    m["month"] = pd.PeriodIndex(m["month"], freq="M").to_timestamp()
    return m


def yearly_summary(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(year=m["month"].dt.year)
    y = m.groupby("year").agg(deals=("deals", "sum"), value_aed=("value_aed", "sum"))
    off = m[m["off_plan"] == 1].groupby("year")["deals"].sum()
    y["off_plan_share"] = (off / y["deals"]).fillna(0)
    y["value_aed_bn"] = (y.pop("value_aed") / 1e9).round(1)
    return y


def chart_monthly(m: pd.DataFrame) -> None:
    deals = m.groupby("month")["deals"].sum()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.bar(deals.index, deals.values, width=25, color="#2a6f97")
    ax1.set_title("Residential sales registered with DLD per month")
    ax1.set_ylabel("Sales")
    ax1.grid(axis="y", alpha=0.3)
    # deal-weighted median of ready + off-plan medians is not exact, so plot each segment
    for (ptype, off), g in m.groupby(["property_type", "off_plan"]):
        g = g.set_index("month")["median_ppsf"].where(g.set_index("month")["deals"] >= 30)
        g = g.reindex(pd.date_range(m["month"].min(), m["month"].max(), freq="MS"))  # gaps stay gaps
        label = f"{ptype} - {'off-plan' if off else 'ready'}"
        ax2.plot(g.index, g.rolling(3, min_periods=2).mean(), label=label)
    ax2.set_title("Median price per sqft (AED, 3-month average)")
    ax2.set_ylabel("AED per sqft")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "market_monthly.png", dpi=120)
    plt.close(fig)


def chart_offplan(y: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(y.index, y["off_plan_share"] * 100, color="#e07a5f")
    ax.set_title("Off-plan share of residential sales (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "offplan_share.png", dpi=120)
    plt.close(fig)


def top_areas(year: int = LATEST_FULL_YEAR) -> pd.DataFrame:
    s = pd.read_csv(PROCESSED / "sales_by_area_year.csv")
    s = s[(s["year"] == year)]
    allm = s[s["market"] == "All"].groupby("area").agg(deals=("deals", "sum"))
    apt = s[(s["market"] == "All") & (s["property_type"] == "Apartment")].set_index("area")
    off = s[s["market"] == "Off-plan"].groupby("area")["deals"].sum()
    t = allm.assign(off_plan_share=(off / allm["deals"]).fillna(0),
                    apt_median_ppsf=apt["median_ppsf"].round(0),
                    apt_median_price=apt["median_price_aed"].round(-3))
    return t.sort_values("deals", ascending=False).head(TOP_N)


def chart_top_areas(t: pd.DataFrame, year: int = LATEST_FULL_YEAR) -> None:
    t = t.sort_values("deals")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([short(a) for a in t.index], t["deals"], color="#2a6f97")
    for i, (n, p) in enumerate(zip(t["deals"], t["apt_median_ppsf"])):
        ax.text(n, i, f"  {n:,}  (apt AED {p:,.0f}/sqft)", va="center", fontsize=7)
    ax.set_title(f"Most active areas by residential sales, {year}")
    ax.set_xlim(0, t["deals"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(IMAGES / "top_areas.png", dpi=120)
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    m = load_monthly()
    y = yearly_summary(m)
    print("Residential sales per year\n")
    print(y.assign(off_plan_share=(y["off_plan_share"] * 100).round(0)).to_string(), "\n")
    chart_monthly(m)
    chart_offplan(y)
    t = top_areas()
    t.to_csv(PROCESSED / "top_areas.csv")
    print(f"Most active areas in {LATEST_FULL_YEAR}\n")
    print(t.assign(off_plan_share=(t["off_plan_share"] * 100).round(0)).to_string())
    chart_top_areas(t)


if __name__ == "__main__":
    main()
