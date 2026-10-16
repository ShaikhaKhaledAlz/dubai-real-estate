"""
Machine-learning price model + "value finder", trained on real DLD sales.

1. Train a Gradient Boosting model on 2023-2025 sales to predict the price of a home
   from its area, type, size, bedrooms, off-plan status and sale date.
2. Test it "out of time" on 2026 sales it has never seen. This is a harder and more
   honest test than a random split, because the model has to work on the future.
3. Flag 2026 sales priced 20-50% BELOW the model's estimate: possible bargains to research.
   (Gaps above 50% are almost always non-market deals such as partial-share transfers.)

Data: data/processed/sales_sample.csv, a random 3% sample of all 2023+ residential sales.

Run:  python src/model.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sales_sample.csv"
IMAGES = ROOT / "images"
TEST_YEAR = 2026
DISCOUNT = 0.20     # flag sales priced 20%+ below the model's estimate
MAX_DISCOUNT = 0.50 # 50%+ below is almost always a non-market deal (share transfer, family sale)
MIN_PPSF = 300      # below AED 300/sqft is not a normal arm's-length sale in today's Dubai

CATEGORICAL = ["area", "property_type"]
NUMERIC = ["size_sqft", "bedrooms", "off_plan", "months"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df[df["price_aed"] / df["size_sqft"] >= MIN_PPSF].copy()
    df["months"] = (df["date"].dt.year - 2023) * 12 + df["date"].dt.month  # time trend
    return df


def build_model() -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               min_frequency=20), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    gbr = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                                        categorical_features=[0, 1], random_state=42)
    return Pipeline([("prep", prep), ("model", gbr)])


def main() -> None:
    IMAGES.mkdir(exist_ok=True)
    df = load()
    train, test = df[df["date"].dt.year < TEST_YEAR], df[df["date"].dt.year == TEST_YEAR].copy()
    X = CATEGORICAL + NUMERIC
    model = build_model().fit(train[X], np.log(train["price_aed"]))

    test["model_price"] = np.exp(model.predict(test[X])).round(-3)
    err = (test["model_price"] - test["price_aed"]).abs() / test["price_aed"]
    r2_log = r2_score(np.log(test["price_aed"]), np.log(test["model_price"]))
    print(f"Trained on {len(train):,} sales (2023-{TEST_YEAR - 1}), "
          f"tested on {len(test):,} unseen {TEST_YEAR} sales")
    print(f"R² (log price):        {r2_log:.3f}")
    print(f"Median absolute error: {err.median() * 100:.1f}%")
    print(f"Within ±20% of price:  {(err <= 0.20).mean() * 100:.0f}% of sales")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test["price_aed"] / 1e6, test["model_price"] / 1e6, s=4, alpha=0.3)
    lim = [0, np.percentile(test["price_aed"] / 1e6, 98)]
    ax.plot(lim, lim, color="red", linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual price (AED millions)")
    ax.set_ylabel("Predicted price (AED millions)")
    ax.set_title(f"Price model on unseen {TEST_YEAR} sales")
    fig.tight_layout()
    fig.savefig(IMAGES / "predicted_vs_actual.png", dpi=120)
    plt.close(fig)

    test["discount"] = (1 - test["price_aed"] / test["model_price"]).round(3)
    deals = test[test["discount"].between(DISCOUNT, MAX_DISCOUNT)].sort_values("discount", ascending=False)
    cols = ["date", "area", "property_type", "bedrooms", "size_sqft", "off_plan",
            "price_aed", "model_price", "discount"]
    deals[cols].to_csv(ROOT / "data" / "processed" / "possible_undervalued_deals.csv", index=False)
    print(f"\n{len(deals)} of {len(test):,} {TEST_YEAR} sales ({len(deals) / len(test):.0%}) "
          f"priced {DISCOUNT:.0%}-{MAX_DISCOUNT:.0%} below the model estimate. Top 5:")
    print(deals[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
