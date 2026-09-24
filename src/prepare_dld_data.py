"""
Convert official Dubai Land Department (DLD) transaction CSVs into the
clean format used by this project.

How to use
1. Download CSV file(s) from https://dubailand.gov.ae/en/open-data/real-estate-data/
2. Put them in data/raw/  (you can add as many files as you like, e.g. one per month)
3. Run:  python src/prepare_dld_data.py
   -> writes data/real_transactions.csv

Cleaning steps
- Keep only Sales (drop mortgages and gifts, which are not market prices)
- Keep only Residential flats and villas
- Convert size from square metres to square feet
- Turn "Studio" / "2 B/R" into a number of bedrooms
- Remove duplicates and extreme outliers (price per sqft outside the 1st-99th percentile)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "real_transactions.csv"
SQM_TO_SQFT = 10.764

TYPE_MAP = {"Flat": "Apartment", "Villa": "Villa"}


def bedrooms(rooms: str) -> float:
    if not isinstance(rooms, str):
        return float("nan")
    if rooms.strip().lower() == "studio":
        return 0
    first = rooms.split()[0]
    return float(first) if first.isdigit() else float("nan")


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    df = pd.concat((pd.read_csv(f, encoding="utf-8-sig") for f in files), ignore_index=True)
    return df.drop_duplicates(subset="TRANSACTION_NUMBER")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["GROUP_EN"] == "Sales") & (df["USAGE_EN"] == "Residential")]
    df = df[df["PROP_SB_TYPE_EN"].isin(TYPE_MAP)]
    out = pd.DataFrame({
        "transaction_date": pd.to_datetime(df["INSTANCE_DATE"]).dt.date,
        "area": df["AREA_EN"].str.title().str.strip(),
        "project": df["PROJECT_EN"],
        "property_type": df["PROP_SB_TYPE_EN"].map(TYPE_MAP),
        "bedrooms": df["ROOMS_EN"].map(bedrooms),
        "size_sqft": (df["PROCEDURE_AREA"] * SQM_TO_SQFT).round(0),
        "off_plan": df["IS_OFFPLAN_EN"].eq("Off-Plan"),
        "price_aed": df["TRANS_VALUE"],
        "nearest_metro": df["NEAREST_METRO_EN"],
    })
    out = out.dropna(subset=["price_aed", "size_sqft", "bedrooms"])
    out = out[(out["price_aed"] > 0) & (out["size_sqft"] > 100)]
    ppsf = out["price_aed"] / out["size_sqft"]
    lo, hi = ppsf.quantile([0.01, 0.99])
    out = out[ppsf.between(lo, hi)]
    return out.sort_values("transaction_date").reset_index(drop=True)


def main() -> pd.DataFrame:
    raw = load_raw()
    df = clean(raw)
    df.to_csv(OUT, index=False)
    print(f"Read {len(raw):,} raw DLD records -> kept {len(df):,} residential sales")
    print(f"Period: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Saved to {OUT}")
    return df


if __name__ == "__main__":
    main()
