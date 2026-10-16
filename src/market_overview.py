"""
STEP 2 - Market overview: how big is the market and where is it going?

What this script does, in plain words:
    1. Counts residential sales per month and per year since 2008.
    2. Shows the typical (median) price per sqft over time for four groups:
       ready apartments, off-plan apartments, ready villas, off-plan villas.
    3. Shows what share of sales are off-plan (bought before the building is finished).
    4. Lists the 15 busiest areas in the latest full year.

Outputs: images/market_monthly.png, images/offplan_share.png, images/top_areas.png,
         data/processed/yearly_market.csv, data/processed/top_areas.csv

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


def load_monthly():
    """Monthly sales, value and median price per sqft, split by type and off-plan."""
    m = pd.read_csv(PROCESSED / "monthly_market.csv")
    # The newest month was still in progress when the data was downloaded, so drop it.
    m = m[m["month"] < m["month"].max()]
    m["month"] = pd.to_datetime(m["month"] + "-01")
    return m


def yearly_summary(m):
    m = m.assign(year=m["month"].dt.year)
    y = m.groupby("year").agg(sales=("deals", "sum"), value_aed=("value_aed", "sum"))
    off_plan_sales = m[m["off_plan"] == 1].groupby("year")["deals"].sum()
    y["off_plan_share"] = (off_plan_sales / y["sales"]).fillna(0).round(3)
    y["value_aed_bn"] = (y["value_aed"] / 1e9).round(1)
    y["months_of_data"] = m.groupby("year")["month"].nunique()
    return y.drop(columns="value_aed")


def chart_monthly(m):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # top: number of sales per month
    sales = m.groupby("month")["deals"].sum()
    top.bar(sales.index, sales.values, width=25, color="#2a6f97")
    top.set_title("Residential sales registered with DLD per month")
    top.set_ylabel("Sales")
    top.grid(axis="y", alpha=0.3)

    # bottom: median price per sqft for each group (3-month average to smooth noise)
    all_months = pd.date_range(m["month"].min(), m["month"].max(), freq="MS")
    for (ptype, off_plan), g in m.groupby(["property_type", "off_plan"]):
        g = g.set_index("month")
        price = g["median_ppsf"].where(g["deals"] >= 30)   # skip months with < 30 sales
        price = price.reindex(all_months)                  # missing months stay empty
        name = f"{ptype} - {'off-plan' if off_plan else 'ready'}"
        bottom.plot(price.index, price.rolling(3, min_periods=2).mean(), label=name)
    bottom.set_title("Median price per sqft (AED, 3-month average)")
    bottom.set_ylabel("AED per sqft")
    bottom.legend(fontsize=8)
    bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(IMAGES / "market_monthly.png", dpi=120)
    plt.close(fig)


def chart_offplan(y):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(y.index, y["off_plan_share"] * 100, color="#e07a5f")
    ax.set_title("Off-plan share of residential sales (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "offplan_share.png", dpi=120)
    plt.close(fig)


def top_areas(year=LATEST_FULL_YEAR):
    s = pd.read_csv(PROCESSED / "sales_by_area_year.csv")
    s = s[s["year"] == year]
    every_sale = s[s["market"] == "All"]
    sales = every_sale.groupby("area")["deals"].sum()
    off_plan = s[s["market"] == "Off-plan"].groupby("area")["deals"].sum()
    apartments = every_sale[every_sale["property_type"] == "Apartment"].set_index("area")

    t = pd.DataFrame({
        "common_name": [short(a) for a in sales.index],
        "sales": sales,
        "off_plan_share": (off_plan / sales).fillna(0).round(3),
        "apt_median_ppsf": apartments["median_ppsf"].round(0),
        "apt_median_price": apartments["median_price_aed"].round(-3),
    })
    return t.sort_values("sales", ascending=False).head(TOP_N)


def chart_top_areas(t, year=LATEST_FULL_YEAR):
    t = t.sort_values("sales")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(t["common_name"], t["sales"], color="#2a6f97")
    for i, (n, p) in enumerate(zip(t["sales"], t["apt_median_ppsf"])):
        ax.text(n, i, f"  {n:,} sales  (apartments AED {p:,.0f}/sqft)", va="center", fontsize=7)
    ax.set_title(f"Busiest areas by number of residential sales, {year}")
    ax.set_xlim(0, t["sales"].max() * 1.5)
    fig.tight_layout()
    fig.savefig(IMAGES / "top_areas.png", dpi=120)
    plt.close(fig)


def main():
    IMAGES.mkdir(exist_ok=True)
    m = load_monthly()
    y = yearly_summary(m)
    y.to_csv(PROCESSED / "yearly_market.csv")
    print("Residential sales per year\n")
    print(y.to_string(), "\n")
    chart_monthly(m)
    chart_offplan(y)

    t = top_areas()
    t.to_csv(PROCESSED / "top_areas.csv")
    print(f"Busiest areas in {LATEST_FULL_YEAR}\n")
    print(t.to_string())
    chart_top_areas(t)


if __name__ == "__main__":
    main()
