"""
Investment analysis of Dubai residential property by area, on official DLD data.

Question: "Which Dubai areas gave investors the best return for the risk taken?"

For every area with enough sales in every year of the study window we measure:
- Median price per sqft (ready apartments)
- Price CAGR: compound annual growth of that median (full period and last 5 years)
- Gross rental yield: median rent per sqft (new Ejari contracts) / median sale price per sqft
- Yearly total return = price change that year + rental yield
- Volatility: standard deviation of the yearly total return
- Sharpe ratio: (average total return - risk-free rate) / volatility

Ready (secondary-market) apartments are used so that the price index is not
distorted by the changing mix of off-plan launches.

Run:  python src/analysis.py   (after build_dataset.py, or with the committed data)
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

START, END = 2015, 2025          # full calendar years
PROPERTY_TYPE = "Apartment"
MIN_SALES = 50                   # ready apartment sales needed in EVERY year
MIN_RENTS = 100                  # new rent contracts needed in EVERY year
RISK_FREE_RATE = 0.022           # approx. average 3-month US T-bill 2016-2025 (AED is pegged to USD)


def load_sales() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "sales_by_area_year.csv")


def load_rents() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "rents_by_area_year.csv")


def panel(sales: pd.DataFrame, rents: pd.DataFrame, ptype: str = PROPERTY_TYPE) -> dict:
    """Year x area tables of price per sqft, rent per sqft and counts."""
    s = sales[(sales["property_type"] == ptype) & (sales["market"] == "Ready")
              & sales["year"].between(START, END)]
    r = rents[(rents["property_type"] == ptype) & (rents["contract"] == "New")
              & rents["year"].between(START, END)]
    return {
        "ppsf": s.pivot_table(index="year", columns="area", values="median_ppsf"),
        "sales": s.pivot_table(index="year", columns="area", values="deals", aggfunc="sum"),
        "rent_psf": r.pivot_table(index="year", columns="area", values="median_rent_psf"),
        "rents": r.pivot_table(index="year", columns="area", values="contracts", aggfunc="sum"),
    }


def eligible_areas(p: dict) -> list[str]:
    years = range(START, END + 1)
    ok_sales = p["sales"].reindex(years).fillna(0).ge(MIN_SALES).all()
    ok_rents = p["rents"].reindex(years).fillna(0).ge(MIN_RENTS).all()
    both = ok_sales[ok_sales].index.intersection(ok_rents[ok_rents].index)
    return sorted(both)


def area_metrics(sales: pd.DataFrame, rents: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = panel(sales, rents)
    areas = eligible_areas(p)
    price = p["ppsf"][areas]
    yld = (p["rent_psf"][areas] / price)
    # total return in year t = price growth in t + rental yield earned on last year's price
    total = price.pct_change() + yld.shift(1)
    total = total.iloc[1:]
    n_years = END - START

    m = pd.DataFrame({
        "median_ppsf_latest": price.iloc[-1].round(0),
        "sales_latest": p["sales"].loc[END, areas].astype(int),
        "price_cagr": (price.iloc[-1] / price.iloc[0]) ** (1 / n_years) - 1,
        "price_cagr_5y": (price.loc[END] / price.loc[END - 5]) ** (1 / 5) - 1,
        "gross_yield_latest": yld.iloc[-1],
        "avg_gross_yield": yld.mean(),
        "avg_total_return": total.mean(),
        "volatility": total.std(),
    })
    m["sharpe_ratio"] = (m["avg_total_return"] - RISK_FREE_RATE) / m["volatility"]
    return m.sort_values("sharpe_ratio", ascending=False), {"price": price, "yield": yld, "total": total}


def plot_price_index(price: pd.DataFrame, m: pd.DataFrame) -> None:
    """Price per sqft rebased to 100 in START, for the 8 biggest markets."""
    top = m.sort_values("sales_latest", ascending=False).index[:8]
    idx = price[top] / price[top].iloc[0] * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    for area in top:
        ax.plot(idx.index, idx[area], marker="o", ms=3, label=short(area))
    ax.axhline(100, color="grey", lw=0.8)
    ax.set_title(f"Ready apartment price per sqft, indexed ({START} = 100)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "price_trends.png", dpi=120)
    plt.close(fig)


def plot_risk_return(m: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(m["volatility"] * 100, m["avg_total_return"] * 100, s=60, color="#2a6f97")
    placed = []  # simple label nudging so close points don't overlap
    for area, row in m.sort_values("avg_total_return").iterrows():
        x, y = row["volatility"] * 100, row["avg_total_return"] * 100
        dy = 3
        while any(abs(x - px) < 2.5 and abs((y + dy / 25) - py) < 0.18 for px, py in placed):
            dy += 9
        placed.append((x, y + dy / 25))
        ax.annotate(short(area), (x, y), xytext=(4, dy), textcoords="offset points", fontsize=7,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5) if dy > 3 else None)
    ax.axhline(RISK_FREE_RATE * 100, linestyle="--", color="grey", label="Risk-free rate")
    ax.set_title(f"Risk vs return by area, ready apartments {START + 1}-{END}")
    ax.set_xlabel("Volatility of yearly total return (%)")
    ax.set_ylabel("Average yearly total return (%) = price growth + rental yield")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "risk_return.png", dpi=120)
    plt.close(fig)


def plot_yields(m: pd.DataFrame) -> None:
    s = (m["gross_yield_latest"] * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(s))))
    ax.barh([short(a) for a in s.index], s.values, color="#2a6f97")
    for i, v in enumerate(s.values):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=7)
    ax.set_title(f"Gross rental yield by area, ready apartments {END}")
    ax.set_xlabel("%")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "rental_yields.png", dpi=120)
    plt.close(fig)


def main() -> pd.DataFrame:
    IMAGES.mkdir(exist_ok=True)
    m, series = area_metrics(load_sales(), load_rents())
    plot_price_index(series["price"], m)
    plot_risk_return(m)
    plot_yields(m)

    pretty = m.copy()
    for col in ["price_cagr", "price_cagr_5y", "gross_yield_latest", "avg_gross_yield", "avg_total_return", "volatility"]:
        pretty[col] = (pretty[col] * 100).round(1).astype(str) + "%"
    pretty["sharpe_ratio"] = pretty["sharpe_ratio"].round(2)
    print(f"{len(m)} areas with {MIN_SALES}+ ready apartment sales and {MIN_RENTS}+ new "
          f"rent contracts in every year {START}-{END}\n")
    print(pretty.to_string())
    m.round(4).to_csv(PROCESSED / "area_metrics.csv")
    return m


if __name__ == "__main__":
    main()
