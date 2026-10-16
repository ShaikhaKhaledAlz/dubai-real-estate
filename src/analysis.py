"""
STEP 3 - Investment analysis: which areas gave the best return for the risk?

What this script does, in plain words:
    1. For each area and each year, take the typical (median) price per sqft
       of READY apartments, and the typical rent per sqft of NEW rent contracts.
    2. Keep only areas with plenty of sales and rents in EVERY year from 2015 to 2025,
       so every number is based on a large sample.
    3. For each area, calculate:
         - price growth per year
         - rental yield (rent / price)
         - total return per year (price growth + rental yield)
         - volatility (how much the yearly return jumps around = risk)
         - Sharpe ratio (extra return per unit of risk)
    4. Save a table (data/processed/area_metrics.csv) and three charts (images/).

Why only READY apartments?
    Off-plan homes are new launches and their mix changes a lot from year to year.
    Using ready apartments compares like with like.

Run:  python src/analysis.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # lets matplotlib save charts without opening a window
import matplotlib.pyplot as plt
import pandas as pd

from area_labels import short

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]   # the project folder
PROCESSED = ROOT / "data" / "processed"
IMAGES = ROOT / "images"

START, END = 2015, 2025      # study period (full calendar years only)
MIN_SALES = 50               # ready apartment sales needed in EVERY year
MIN_RENTS = 100              # new rent contracts needed in EVERY year

# Risk-free rate = what you could earn with almost no risk (cash in US Treasury bills).
# The UAE dirham is pegged to the US dollar, so US rates are a fair benchmark.
# Yearly average 3-month T-bill rate in %, source: FRED series RIFSGFSM03NA.
TBILL_RATE = {2016: 0.32, 2017: 0.93, 2018: 1.94, 2019: 2.06, 2020: 0.35,
              2021: 0.05, 2022: 2.02, 2023: 5.07, 2024: 4.97, 2025: 4.07}
RISK_FREE_RATE = sum(TBILL_RATE.values()) / len(TBILL_RATE) / 100   # = 2.2%


# ---------------------------------------------------------------- load data
def load_sales():
    """One row per year x area x property type x market (Ready / Off-plan / All)."""
    return pd.read_csv(PROCESSED / "sales_by_area_year.csv")


def load_rents():
    """One row per year x area x property type x contract (New / Renewal)."""
    return pd.read_csv(PROCESSED / "rents_by_area_year.csv")


def make_tables(sales, rents):
    """Turn the long tables into 'wide' tables: one row per year, one column per area."""
    s = sales[(sales["property_type"] == "Apartment")
              & (sales["market"] == "Ready")
              & (sales["year"].between(START, END))]
    r = rents[(rents["property_type"] == "Apartment")
              & (rents["contract"] == "New")
              & (rents["year"].between(START, END))]

    price = s.pivot_table(index="year", columns="area", values="median_ppsf")
    n_sales = s.pivot_table(index="year", columns="area", values="deals", aggfunc="sum")
    rent = r.pivot_table(index="year", columns="area", values="median_rent_psf")
    n_rents = r.pivot_table(index="year", columns="area", values="contracts", aggfunc="sum")
    return price, n_sales, rent, n_rents


def areas_with_enough_data(n_sales, n_rents):
    """Areas with at least MIN_SALES sales and MIN_RENTS rents in every single year."""
    years = range(START, END + 1)
    enough_sales = (n_sales.reindex(years).fillna(0) >= MIN_SALES).all()  # True/False per area
    enough_rents = (n_rents.reindex(years).fillna(0) >= MIN_RENTS).all()
    return sorted(a for a in enough_sales.index
                  if enough_sales[a] and enough_rents.get(a, False))


# ---------------------------------------------------------------- metrics
def area_metrics(sales, rents):
    price, n_sales, rent, n_rents = make_tables(sales, rents)
    areas = areas_with_enough_data(n_sales, n_rents)
    price, rent = price[areas], rent[areas]

    rental_yield = rent / price                      # e.g. 0.07 = 7% a year
    price_change = price.pct_change()                # % change from last year
    # Total return in a year = price change that year + the rent you earned
    # (rent as a % of what you paid at the start of the year, i.e. last year's price)
    total_return = (price_change + rental_yield.shift(1)).loc[START + 1:]

    years = END - START
    m = pd.DataFrame({
        "median_ppsf_latest": price.loc[END].round(0),
        "sales_latest": n_sales.loc[END, areas].astype(int),
        "price_cagr": (price.loc[END] / price.loc[START]) ** (1 / years) - 1,
        "price_cagr_5y": (price.loc[END] / price.loc[END - 5]) ** (1 / 5) - 1,
        "gross_yield_latest": rental_yield.loc[END],
        "avg_gross_yield": rental_yield.mean(),
        "avg_total_return": total_return.mean(),
        "volatility": total_return.std(),
    })
    m["sharpe_ratio"] = (m["avg_total_return"] - RISK_FREE_RATE) / m["volatility"]
    m = m.sort_values("sharpe_ratio", ascending=False)
    return m, price


# ---------------------------------------------------------------- charts
def chart_price_index(price, m):
    """Price per sqft for the 8 busiest areas, rebased so START = 100."""
    busiest = m.sort_values("sales_latest", ascending=False).index[:8]
    index = price[busiest] / price[busiest].loc[START] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    for area in busiest:
        ax.plot(index.index, index[area], marker="o", markersize=3, label=short(area))
    ax.axhline(100, color="grey", linewidth=0.8)
    ax.set_title(f"Ready apartment price per sqft, indexed ({START} = 100)")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Index (100 = price in {START})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "price_trends.png", dpi=120)
    plt.close(fig)


def chart_risk_return(m):
    """Each dot is an area. Higher = more return. Further right = more risk."""
    fig, ax = plt.subplots(figsize=(10, 7))
    x = m["volatility"] * 100
    y = m["avg_total_return"] * 100
    ax.scatter(x, y, s=60, color="#2a6f97")

    # Write the area name next to each dot. If two names would overlap,
    # move the second one up a little and draw a thin line to its dot.
    used = []
    for area in m.sort_values("avg_total_return").index:
        px, py = x[area], y[area]
        lift = 3
        while any(abs(px - ux) < 2.5 and abs(py + lift / 25 - uy) < 0.18 for ux, uy in used):
            lift += 9
        used.append((px, py + lift / 25))
        line = dict(arrowstyle="-", color="grey", lw=0.5) if lift > 3 else None
        ax.annotate(short(area), (px, py), xytext=(4, lift), textcoords="offset points",
                    fontsize=7, arrowprops=line)

    ax.axhline(RISK_FREE_RATE * 100, linestyle="--", color="grey",
               label=f"Risk-free rate ({RISK_FREE_RATE:.1%})")
    ax.set_title(f"Risk vs return by area, ready apartments {START + 1}-{END}")
    ax.set_xlabel("Risk: volatility of yearly total return (%)")
    ax.set_ylabel("Average yearly total return (%) = price growth + rental yield")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "risk_return.png", dpi=120)
    plt.close(fig)


def chart_yields(m):
    s = (m["gross_yield_latest"] * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(s))))
    ax.barh([short(a) for a in s.index], s.values, color="#2a6f97")
    for i, v in enumerate(s.values):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=7)
    ax.set_title(f"Gross rental yield by area, ready apartments {END}")
    ax.set_xlabel("Yearly rent as % of price")
    ax.set_xlim(0, s.max() * 1.12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "rental_yields.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    IMAGES.mkdir(exist_ok=True)
    m, price = area_metrics(load_sales(), load_rents())
    chart_price_index(price, m)
    chart_risk_return(m)
    chart_yields(m)
    m.round(4).to_csv(PROCESSED / "area_metrics.csv")

    # print a readable version of the table
    pretty = m.copy()
    pretty.index = [short(a) for a in pretty.index]
    for col in ["price_cagr", "price_cagr_5y", "gross_yield_latest", "avg_gross_yield",
                "avg_total_return", "volatility"]:
        pretty[col] = (pretty[col] * 100).round(1).astype(str) + "%"
    pretty["sharpe_ratio"] = pretty["sharpe_ratio"].round(2)
    print(f"Risk-free rate used: {RISK_FREE_RATE:.1%}")
    print(f"{len(m)} areas had {MIN_SALES}+ ready apartment sales and {MIN_RENTS}+ new "
          f"rent contracts in every year {START}-{END}\n")
    print(pretty.to_string())


if __name__ == "__main__":
    main()
