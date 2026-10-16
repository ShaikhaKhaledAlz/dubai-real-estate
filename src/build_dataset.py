"""
Build the project's real datasets from the official Dubai Land Department (DLD)
open data, published on Data Dubai (https://data.dubai).

Inputs (download yourself, they are too big for GitHub):
  - "Real Estate Transactions" -> https://data.dubai/en/l/470061
      files: transactions_*.csv(.gz)      (~1.1 GB, every DLD transaction since the 1960s)
  - "Rent Contracts" (Ejari)   -> https://data.dubai/en/l/468586
      files: rent_contracts_*.csv(.gz)    (~5 GB, every registered rent contract)
  Put them in data/raw/ and run:  python src/build_dataset.py
  Quick test on the 1,500-row samples already in data/raw/:  python src/build_dataset.py --test

Outputs (small, committed to the repo) in data/processed/:
  sales_by_area_year.csv   one row per year x area x property type x off-plan flag
  rents_by_area_year.csv   one row per year x area x property type x new/renewal
  monthly_market.csv       city-wide monthly sales count, value and median AED/sqft
  sales_sample.csv         random 3% sample of 2023+ sales (for the price model)

Cleaning rules (identical for every file)
  Sales : trans_group = Sales, usage = Residential, Flat or Villa,
          price > AED 50k, 15-5,000 sqm, AED 100-20,000 per sqft, year >= 2008
  Rents : usage = Residential, Flat/Studio/Penthouse or Villa, one property per contract,
          annual rent > AED 5k, 15-3,000 sqm, AED 5-1,000 per sqft per year, 2010-2026
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SQM_TO_SQFT = 10.764
CHUNK = 250_000
SAMPLE_FRAC, SAMPLE_FROM_YEAR = 0.03, 2023
SAMPLE_MODE = False  # --test: use the small samples in data/raw/

SALE_COLS = ["instance_date", "trans_group_en", "property_usage_en", "property_type_en",
             "property_sub_type_en", "reg_type_en", "area_id", "area_name_en",
             "actual_worth", "procedure_area", "rooms_en"]
RENT_COLS = ["contract_start_date", "property_usage_en", "ejari_property_type_en", "no_of_prop",
             "annual_amount", "actual_area", "area_id", "contract_reg_type_en"]
RENT_KIND = {"Flat": "Apartment", "Studio": "Apartment", "Penthouse": "Apartment", "Villa": "Villa"}


def bedrooms(rooms: pd.Series) -> pd.Series:
    beds = pd.to_numeric(rooms.str.extract(r"^(\d+)")[0], errors="coerce")
    return beds.mask(rooms.eq("Studio"), 0)


def clean_sales(c: pd.DataFrame) -> pd.DataFrame:
    c = c[c["trans_group_en"].eq("Sales") & c["property_usage_en"].eq("Residential")]
    sub, typ = c["property_sub_type_en"].fillna(""), c["property_type_en"]
    kind = np.select([sub.eq("Flat"), sub.eq("Villa") | (sub.eq("") & typ.eq("Villa"))],
                     ["Apartment", "Villa"], "")
    sqft = c["procedure_area"] * SQM_TO_SQFT
    d = pd.DataFrame({
        "date": pd.to_datetime(c["instance_date"], errors="coerce"),
        "area_id": c["area_id"],
        "area": c["area_name_en"],
        "property_type": kind,
        "bedrooms": bedrooms(c["rooms_en"].fillna("")),
        "size_sqft": sqft.round(0),
        "off_plan": c["reg_type_en"].fillna("").str.startswith("Off").astype(int),
        "price_aed": c["actual_worth"],
        "ppsf": c["actual_worth"] / sqft,
    })
    ok = (d["property_type"].ne("") & (d["price_aed"] > 50_000)
          & c["procedure_area"].between(15, 5_000, inclusive="neither")
          & d["ppsf"].between(100, 20_000) & (d["date"].dt.year >= 2008))
    return d[ok]


def clean_rents(c: pd.DataFrame) -> pd.DataFrame:
    c = c[c["property_usage_en"].eq("Residential") & c["no_of_prop"].eq(1)]
    sqft = c["actual_area"] * SQM_TO_SQFT
    d = pd.DataFrame({
        "year": pd.to_datetime(c["contract_start_date"], errors="coerce").dt.year,
        "area_id": c["area_id"],
        "property_type": c["ejari_property_type_en"].map(RENT_KIND),
        "contract": np.where(c["contract_reg_type_en"].eq("New"), "New", "Renewal"),
        "annual_rent_aed": c["annual_amount"],
        "rent_psf": c["annual_amount"] / sqft,
    })
    ok = (d["property_type"].notna() & (d["annual_rent_aed"] > 5_000)
          & c["actual_area"].between(15, 3_000, inclusive="neither")
          & d["rent_psf"].between(5, 1_000) & d["year"].between(2010, 2026))
    return d[ok]


def read_chunks(pattern: str, cols: list[str]):
    files = sorted(f for f in RAW.glob(pattern) if "sample" not in f.name or SAMPLE_MODE)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {RAW}. See the docstring.")
    for f in files:
        print(f"  reading {f.name}")
        yield from pd.read_csv(f, usecols=cols, chunksize=CHUNK, low_memory=False)


def build_sales() -> pd.DataFrame:
    sales = pd.concat(clean_sales(c) for c in read_chunks("transactions*.csv*", SALE_COLS))
    sales["year"] = sales["date"].dt.year
    names = sales.groupby("area_id")["area"].agg(lambda s: s.mode().iat[0]).str.title()

    # Medians for ready homes, off-plan homes, and both together ("All")
    keys = ["year", "area_id", "property_type"]
    stats = dict(deals=("price_aed", "size"), median_ppsf=("ppsf", "median"),
                 median_price_aed=("price_aed", "median"))
    split = sales.assign(market=sales["off_plan"].map({0: "Ready", 1: "Off-plan"}))
    by_year = pd.concat([
        split.groupby(keys + ["market"]).agg(**stats).reset_index(),
        sales.groupby(keys).agg(**stats).reset_index().assign(market="All"),
    ]).sort_values(keys + ["market"])
    by_year.insert(2, "area", by_year["area_id"].map(names))

    month = sales["date"].dt.to_period("M").astype(str).rename("month")
    monthly = sales.groupby([month, "property_type", "off_plan"]).agg(
        deals=("price_aed", "size"), value_aed=("price_aed", "sum"),
        median_ppsf=("ppsf", "median")).reset_index()

    recent = sales[sales["year"] >= SAMPLE_FROM_YEAR]
    sample = recent.sample(frac=SAMPLE_FRAC, random_state=42)
    sample = sample.assign(date=sample["date"].dt.date, area=sample["area_id"].map(names))
    sample = sample[["date", "area", "property_type", "bedrooms", "size_sqft", "off_plan", "price_aed"]]

    by_year.round(1).to_csv(OUT / "sales_by_area_year.csv", index=False)
    monthly.round(1).to_csv(OUT / "monthly_market.csv", index=False)
    sample.sort_values("date").to_csv(OUT / "sales_sample.csv", index=False)
    print(f"Sales: kept {len(sales):,} residential sales")
    return names


def build_rents(names: pd.Series) -> None:
    rents = pd.concat(clean_rents(c) for c in read_chunks("rent_contracts*.csv*", RENT_COLS))
    g = rents.groupby(["year", "area_id", "property_type", "contract"])
    out = g.agg(contracts=("rent_psf", "size"), median_rent_psf=("rent_psf", "median"),
                median_annual_rent_aed=("annual_rent_aed", "median")).reset_index()
    out.insert(2, "area", out["area_id"].map(names))
    out.round(1).to_csv(OUT / "rents_by_area_year.csv", index=False)
    print(f"Rents: kept {len(rents):,} residential contracts")


def main() -> None:
    global OUT, SAMPLE_MODE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--test", action="store_true",
                    help="run on the small 1,500-row samples in data/raw/ and write to data/test_output/")
    if ap.parse_args().test:
        SAMPLE_MODE, OUT = True, ROOT / "data" / "test_output"
    OUT.mkdir(parents=True, exist_ok=True)
    names = build_sales()
    build_rents(names)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
