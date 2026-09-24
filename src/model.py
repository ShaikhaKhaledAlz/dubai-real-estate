"""
Machine-learning price model + "value finder".

1. Train a Gradient Boosting model to predict a property's price from
   its area, type, size, bedrooms, year and off-plan status.
2. Measure accuracy on data the model has never seen (test set).
3. Flag the latest-year deals priced well BELOW the model's estimate:
   possible undervalued opportunities.

Run:  python src/model.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from analysis import IMAGES, ROOT, load_data

CATEGORICAL = ["area", "property_type"]
NUMERIC = ["size_sqft", "bedrooms", "year", "off_plan"]


def build_model() -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    gbr = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    return Pipeline([("prep", prep), ("model", gbr)])


def main() -> None:
    df = load_data()
    df["off_plan"] = df["off_plan"].astype(int)
    X = df[CATEGORICAL + NUMERIC]
    y = np.log(df["price_aed"])  # log price: errors become percentage-like

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model().fit(X_train, y_train)

    pred = np.exp(model.predict(X_test))
    actual = np.exp(y_test)
    print(f"R² (test set):   {r2_score(actual, pred):.3f}")
    print(f"Mean abs % error: {mean_absolute_percentage_error(actual, pred) * 100:.1f}%")

    # Predicted vs actual chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual / 1e6, pred / 1e6, s=6, alpha=0.4)
    lim = [0, np.percentile(actual / 1e6, 99)]
    ax.plot(lim, lim, color="red", linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual price (AED millions)")
    ax.set_ylabel("Predicted price (AED millions)")
    ax.set_title("Model: predicted vs actual (test set)")
    fig.tight_layout()
    fig.savefig(IMAGES / "predicted_vs_actual.png", dpi=120)
    plt.close(fig)

    # Value finder: latest year, priced >15% below model estimate
    latest = df[df["year"] == df["year"].max()].copy()
    latest["model_price"] = np.exp(model.predict(latest[CATEGORICAL + NUMERIC])).round(-3)
    latest["discount"] = 1 - latest["price_aed"] / latest["model_price"]
    deals = latest[latest["discount"] > 0.15].sort_values("discount", ascending=False)
    cols = ["transaction_date", "area", "property_type", "bedrooms", "size_sqft",
            "price_aed", "model_price", "discount"]
    deals[cols].to_csv(ROOT / "data" / "possible_undervalued_deals.csv", index=False)
    print(f"\n{len(deals)} deals priced >15% below model estimate. Top 5:")
    print(deals[cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
