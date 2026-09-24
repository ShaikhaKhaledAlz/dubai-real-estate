"""
Investment analysis of Dubai residential property by area.

Answers the question an investment analyst would ask:
"Which Dubai areas gave the best return for the risk taken?"

Metrics per area
- Median price per sqft (latest year)
- Price CAGR: compound annual growth rate of median price per sqft
- Gross rental yield: annual rent / price
- Total return: CAGR + rental yield (simple approximation)
- Volatility: standard deviation of yearly price changes
- Sharpe ratio: (total return - risk-free rate) / volatility

Run:  python src/analysis.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sample_transactions.csv"
IMAGES = ROOT / "images"
RISK_FREE_RATE = 0.04  # rough proxy for a USD/AED risk-free rate


def load_data(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    df = df.dropna(subset=["price_aed", "size_sqft"])
    df = df[(df["price_aed"] > 0) & (df["size_sqft"] > 0)]
    df["year"] = df["transaction_date"].dt.year
    df["price_per_sqft"] = df["price_aed"] / df["size_sqft"]
    df["gross_yield"] = df["annual_rent_aed"] / df["price_aed"]
    return df


def yearly_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Median price per sqft for each area and year (rows = year, columns = area)."""
    return df.pivot_table(index="year", columns="area", values="price_per_sqft", aggfunc="median")


def area_metrics(df: pd.DataFrame) -> pd.DataFrame:
    prices = yearly_prices(df)
    n_years = prices.index.max() - prices.index.min()
    cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1
    volatility = prices.pct_change().std()
    gross_yield = df.groupby("area")["gross_yield"].median()

    m = pd.DataFrame({
        "median_ppsf_latest": prices.iloc[-1].round(0),
        "price_cagr": cagr,
        "gross_yield": gross_yield,
        "volatility": volatility,
    })
    m["total_return"] = m["price_cagr"] + m["gross_yield"]
    m["sharpe_ratio"] = (m["total_return"] - RISK_FREE_RATE) / m["volatility"]
    return m.sort_values("sharpe_ratio", ascending=False)


def plot_price_trends(prices: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for area in prices.columns:
        ax.plot(prices.index, prices[area], marker="o", label=area)
    ax.set_title("Median price per sqft by area (AED)")
    ax.set_xlabel("Year")
    ax.set_ylabel("AED per sqft")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "price_trends.png", dpi=120)
    plt.close(fig)


def plot_risk_return(m: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(m["volatility"] * 100, m["total_return"] * 100, s=90)
    for area, row in m.iterrows():
        ax.annotate(area, (row["volatility"] * 100, row["total_return"] * 100),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(RISK_FREE_RATE * 100, linestyle="--", color="grey", label="Risk-free rate")
    ax.set_title("Risk vs return by area")
    ax.set_xlabel("Volatility of yearly price change (%)")
    ax.set_ylabel("Total return per year (%) = growth + rental yield")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "risk_return.png", dpi=120)
    plt.close(fig)


def plot_yields(m: pd.DataFrame) -> None:
    s = (m["gross_yield"] * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(s.index, s.values)
    ax.set_title("Median gross rental yield by area (%)")
    ax.set_xlabel("%")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "rental_yields.png", dpi=120)
    plt.close(fig)


def main() -> pd.DataFrame:
    IMAGES.mkdir(exist_ok=True)
    df = load_data()
    m = area_metrics(df)
    plot_price_trends(yearly_prices(df))
    plot_risk_return(m)
    plot_yields(m)

    pretty = m.copy()
    for col in ["price_cagr", "gross_yield", "volatility", "total_return"]:
        pretty[col] = (pretty[col] * 100).round(1).astype(str) + "%"
    pretty["sharpe_ratio"] = pretty["sharpe_ratio"].round(2)
    print(pretty.to_string())
    m.to_csv(ROOT / "data" / "area_metrics.csv")
    return m


if __name__ == "__main__":
    main()
