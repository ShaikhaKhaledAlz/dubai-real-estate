"""
STEP 4 - Machine-learning price model + "value finder".

What this script does, in plain words:
    1. TRAIN: show the model ~13,000 real sales from 2023-2025. For each sale it sees
       the area, apartment or villa, size, bedrooms, off-plan or ready, and the month.
       It learns how these things relate to the price.
    2. TEST: ask it to price ~3,200 sales from 2026 that it has NEVER seen,
       and compare its guesses with the real prices. Testing on a later year
       ("out-of-time") is a fair test, because in real life you predict the future.
    3. VALUE FINDER: list 2026 sales whose real price was 20-50% BELOW the model's
       estimate. These are leads worth a closer look, not recommendations.

The model is a Gradient Boosting Regressor: it builds hundreds of small decision
trees, each one fixing the mistakes of the ones before it.

Data: data/processed/sales_sample.csv (a random 3% of all residential sales since 2023).

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
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sales_sample.csv"
IMAGES = ROOT / "images"

TEST_YEAR = 2026        # train on years before this, test on this year
MIN_PPSF = 300          # sales below AED 300/sqft are not normal market sales -> removed
DISCOUNT_FROM = 0.20    # value finder: at least 20% below the model's estimate ...
DISCOUNT_TO = 0.50      # ... but not more than 50% (bigger gaps are almost always
                        #     non-market deals, e.g. selling a share of a home)

TEXT_COLUMNS = ["area", "property_type"]                        # words -> one-hot encoded
NUMBER_COLUMNS = ["size_sqft", "bedrooms", "off_plan", "months"]


# ---------------------------------------------------------------- data
def load():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df[df["price_aed"] / df["size_sqft"] >= MIN_PPSF].copy()
    # months since Jan 2023 (1, 2, 3 ...) so the model can learn the market trend
    df["months"] = (df["date"].dt.year - 2023) * 12 + df["date"].dt.month
    # a few sales have no bedroom info: use -1 so the model knows it is missing
    df["bedrooms"] = df["bedrooms"].fillna(-1)
    return df


# ---------------------------------------------------------------- model
def build_model():
    """Step 1 turns text into numbers (one column per area). Step 2 is the model."""
    prepare = ColumnTransformer([
        # areas with fewer than 20 sales are grouped together as "infrequent"
        ("text", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
         TEXT_COLUMNS),
        ("numbers", "passthrough", NUMBER_COLUMNS),
    ])
    model = GradientBoostingRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                      random_state=42)
    return Pipeline([("prepare", prepare), ("model", model)])


# ---------------------------------------------------------------- chart
def chart_predicted_vs_actual(test):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test["price_aed"] / 1e6, test["model_price"] / 1e6, s=4, alpha=0.3)
    top = np.percentile(test["price_aed"] / 1e6, 98)
    ax.plot([0, top], [0, top], color="red", linestyle="--", label="perfect prediction")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Actual price (AED millions)")
    ax.set_ylabel("Model's predicted price (AED millions)")
    ax.set_title(f"Price model on {len(test):,} unseen {TEST_YEAR} sales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "predicted_vs_actual.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    IMAGES.mkdir(exist_ok=True)
    df = load()
    train = df[df["date"].dt.year < TEST_YEAR]
    test = df[df["date"].dt.year == TEST_YEAR].copy()
    columns = TEXT_COLUMNS + NUMBER_COLUMNS

    # We predict log(price) instead of price, so a 10% mistake on a cheap flat counts
    # the same as a 10% mistake on an expensive villa. np.exp() turns it back into AED.
    model = build_model().fit(train[columns], np.log(train["price_aed"]))
    test["model_price"] = np.exp(model.predict(test[columns])).round(-3)

    error = (test["model_price"] - test["price_aed"]).abs() / test["price_aed"]
    r2 = r2_score(np.log(test["price_aed"]), np.log(test["model_price"]))
    print(f"Trained on {len(train):,} sales (2023-{TEST_YEAR - 1}), "
          f"tested on {len(test):,} unseen {TEST_YEAR} sales")
    print(f"R² (log price):            {r2:.2f}   (1.00 would be perfect)")
    print(f"Median error:              {error.median():.1%}")
    print(f"Predictions within ±20%:   {(error <= 0.20).mean():.0%}")
    chart_predicted_vs_actual(test)

    # value finder
    test["discount"] = (1 - test["price_aed"] / test["model_price"]).round(3)
    deals = test[test["discount"].between(DISCOUNT_FROM, DISCOUNT_TO)]
    deals = deals.sort_values("discount", ascending=False)
    show = ["date", "area", "property_type", "bedrooms", "size_sqft", "off_plan",
            "price_aed", "model_price", "discount"]
    deals[show].to_csv(ROOT / "data" / "processed" / "possible_undervalued_deals.csv", index=False)
    print(f"\nValue finder: {len(deals)} of {len(test):,} sales ({len(deals) / len(test):.0%}) "
          f"were {DISCOUNT_FROM:.0%}-{DISCOUNT_TO:.0%} below the model's estimate. Top 5:")
    print(deals[show].head().to_string(index=False))


if __name__ == "__main__":
    main()
