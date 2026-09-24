"""
Generate a SYNTHETIC sample dataset of Dubai property transactions.

Why synthetic? The real data (Dubai Land Department open data) is free but large,
so this repo ships a small, realistic-looking sample that lets the project run
out of the box. The numbers are simulated. They are NOT official figures.
To use the real data, see README -> "Using real DLD data".

Run:  python src/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"

# area: (approx. starting price per sqft in AED, yearly growth drift, gross rental yield, volatility)
AREAS = {
    "Palm Jumeirah":        (2300, 0.14, 0.050, 0.10),
    "Downtown Dubai":       (2000, 0.11, 0.055, 0.08),
    "Dubai Marina":         (1500, 0.10, 0.063, 0.07),
    "Business Bay":         (1350, 0.11, 0.065, 0.08),
    "Dubai Hills Estate":   (1300, 0.13, 0.058, 0.08),
    "Jumeirah Lake Towers": (1000, 0.08, 0.072, 0.06),
    "Arabian Ranches":      (1050, 0.10, 0.055, 0.06),
    "Jumeirah Village Circle": (800, 0.09, 0.078, 0.06),
    "Dubai South":          (650, 0.08, 0.075, 0.07),
    "International City":   (450, 0.06, 0.090, 0.07),
}

PROPERTY_TYPES = {"Apartment": 0.75, "Villa": 0.18, "Townhouse": 0.07}
VILLA_AREAS = {"Palm Jumeirah", "Dubai Hills Estate", "Arabian Ranches", "Dubai South"}
YEARS = range(2019, 2026)
# market-wide cycle: soft 2019-2020, strong boom from 2021
MARKET_CYCLE = {2019: -0.05, 2020: -0.06, 2021: 0.04, 2022: 0.08, 2023: 0.06, 2024: 0.03, 2025: 0.01}


def size_for(ptype: str, rooms: int) -> float:
    base = {"Apartment": 450, "Villa": 1800, "Townhouse": 1300}[ptype]
    return base + rooms * RNG.normal(420, 60) + RNG.normal(0, 80)


def main(n_per_area_year: int = 120) -> pd.DataFrame:
    rows = []
    for area, (ppsf0, drift, yld, vol) in AREAS.items():
        ppsf = ppsf0
        for year in YEARS:
            ppsf *= 1 + drift * 0.5 + MARKET_CYCLE[year] + RNG.normal(0, vol / 3)
            for _ in range(n_per_area_year):
                types = list(PROPERTY_TYPES)
                probs = np.array(list(PROPERTY_TYPES.values()))
                if area not in VILLA_AREAS:
                    probs = np.array([0.97, 0.0, 0.03])
                ptype = RNG.choice(types, p=probs / probs.sum())
                rooms = int(RNG.integers(0, 4) if ptype == "Apartment" else RNG.integers(3, 7))
                size = max(300.0, size_for(ptype, rooms))
                off_plan = RNG.random() < (0.25 + 0.05 * (year - 2019))
                unit_ppsf = ppsf * RNG.lognormal(0, 0.12) * (0.93 if off_plan else 1.0)
                price = unit_ppsf * size
                rent = price * yld * RNG.lognormal(0, 0.10)
                month = int(RNG.integers(1, 13))
                day = int(RNG.integers(1, 29))
                rows.append({
                    "transaction_date": f"{year}-{month:02d}-{day:02d}",
                    "area": area,
                    "property_type": ptype,
                    "bedrooms": rooms,
                    "size_sqft": round(size, 0),
                    "off_plan": off_plan,
                    "price_aed": round(price, -3),
                    "annual_rent_aed": round(rent, -2),
                })
    df = pd.DataFrame(rows).sort_values("transaction_date").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} synthetic transactions to {OUT}")
    return df


if __name__ == "__main__":
    main()
