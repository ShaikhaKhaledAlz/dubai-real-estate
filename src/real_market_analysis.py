"""
Analysis of REAL Dubai Land Department transactions.

Works for any date range: the more data in data/raw/, the richer the results.
If the data covers 3+ months, a monthly price trend chart is added automatically.

Run (after prepare_dld_data.py):  python src/real_market_analysis.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real_transactions.csv"
IMAGES = ROOT / "images"
MIN_DEALS = 15  # only rank areas with enough sales to be meaningful


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["transaction_date"])
    df["price_per_sqft"] = df["price_aed"] / df["size_sqft"]
    return df


def area_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("area")
    s = pd.DataFrame({
        "deals": g.size(),
        "median_price_aed": g["price_aed"].median().round(-3),
        "median_ppsf_aed": g["price_per_sqft"].median().round(0),
        "off_plan_share": g["off_plan"].mean(),
    })
    return s[s["deals"] >= MIN_DEALS].sort_values("median_ppsf_aed", ascending=False)


def chart_areas(s: pd.DataFrame) -> None:
    s = s.sort_values("median_ppsf_aed")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(s))))
    ax.barh(s.index, s["median_ppsf_aed"], color="#2a6f97")
    for i, (v, n) in enumerate(zip(s["median_ppsf_aed"], s["deals"])):
        ax.text(v + 15, i, f"{v:,.0f}  (n={n})", va="center", fontsize=8)
    ax.set_title(f"Median price per sqft by area (AED), areas with {MIN_DEALS}+ sales")
    ax.set_xlabel("AED per sqft")
    ax.set_xlim(0, s["median_ppsf_aed"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(IMAGES / "real_price_by_area.png", dpi=120)
    plt.close(fig)


def chart_bedrooms(df: pd.DataFrame) -> None:
    d = df[df["bedrooms"] <= 5]
    groups = sorted(d["bedrooms"].unique())
    data = [d.loc[d["bedrooms"] == b, "price_aed"] / 1e6 for b in groups]
    labels = ["Studio" if b == 0 else f"{int(b)} BR" for b in groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_title("Sale price by number of bedrooms")
    ax.set_ylabel("Price (AED millions)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "real_price_by_bedrooms.png", dpi=120)
    plt.close(fig)


def chart_offplan(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(df["off_plan"].map({True: "Off-plan", False: "Ready"})).agg(
        deals=("price_aed", "size"),
        median_price_aed=("price_aed", "median"),
        median_ppsf_aed=("price_per_sqft", "median"),
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(t.index, t["deals"], color=["#e07a5f", "#2a6f97"])
    for i, n in enumerate(t["deals"]):
        ax.text(i, n, f"{n / t['deals'].sum():.0%}", ha="center", va="bottom")
    ax.set_title("Residential sales: off-plan vs ready")
    ax.set_ylabel("Number of sales")
    fig.tight_layout()
    fig.savefig(IMAGES / "real_offplan_vs_ready.png", dpi=120)
    plt.close(fig)
    return t


def chart_monthly_trend(df: pd.DataFrame) -> None:
    months = df["transaction_date"].dt.to_period("M")
    if months.nunique() < 3:
        return
    m = df.groupby(months)["price_per_sqft"].median()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(m.index.to_timestamp(), m.values, marker="o")
    ax.set_title("Monthly median price per sqft, all residential sales (AED)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMAGES / "real_monthly_trend.png", dpi=120)
    plt.close(fig)


def price_model(df: pd.DataFrame) -> dict:
    """5-fold cross-validated price model on the real data."""
    d = df.copy()
    top = d["area"].value_counts()
    d["area_grouped"] = d["area"].where(d["area"].map(top) >= 5, "Other")
    d["off_plan"] = d["off_plan"].astype(int)
    cat, num = ["area_grouped", "property_type"], ["size_sqft", "bedrooms", "off_plan"]
    model = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                                    ("num", "passthrough", num)])),
        ("gbr", GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          random_state=42)),
    ])
    y = np.log(d["price_aed"])
    pred = np.exp(cross_val_predict(model, d[cat + num], y,
                                    cv=KFold(5, shuffle=True, random_state=42)))
    actual = d["price_aed"].values
    r2 = 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    mape = np.median(np.abs(pred - actual) / actual)
    return {"r2": r2, "median_abs_pct_error": mape}


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    df = load()
    print(f"{len(df):,} residential sales, {df['transaction_date'].min():%d %b %Y} "
          f"to {df['transaction_date'].max():%d %b %Y}")
    print(f"Total value: AED {df['price_aed'].sum() / 1e9:.2f} billion\n")

    s = area_summary(df)
    s.to_csv(ROOT / "data" / "real_area_summary.csv")
    pretty = s.copy()
    pretty["off_plan_share"] = (pretty["off_plan_share"] * 100).round(0).astype(int).astype(str) + "%"
    print(pretty.to_string(), "\n")

    print(chart_offplan(df).round(0).to_string(), "\n")
    chart_areas(s)
    chart_bedrooms(df)
    chart_monthly_trend(df)

    m = price_model(df)
    print(f"Price model (5-fold cross-validation): R² = {m['r2']:.2f}, "
          f"median error = {m['median_abs_pct_error'] * 100:.1f}%")


if __name__ == "__main__":
    main()
