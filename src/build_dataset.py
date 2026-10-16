"""
STEP 1 - Build the project's datasets from the official Dubai Land Department files.

You only need this script if you want to UPDATE the data. The results are already
saved in data/processed/, so the other scripts work without it.

Where the data comes from (free, official, open data on Data Dubai):
    "Real Estate Transactions"  https://data.dubai/en/l/470061
        files named transactions_*.csv.gz       (~1.1 GB, every DLD transaction)
    "Rent Contracts" (Ejari)    https://data.dubai/en/l/468586
        files named rent_contracts_*.csv.gz     (~5 GB, every registered rent contract)
    Download all parts into data/raw/ and run:   python src/build_dataset.py
    Quick test with the small samples already in data/raw/:   python src/build_dataset.py --test

What the script keeps (the "cleaning rules"):
    Sales: only Sales (not mortgages or gifts), only Residential flats and villas,
           price above AED 50,000, size 15-5,000 sqm, AED 100-20,000 per sqft, from 2008.
    Rents: only Residential flats/studios/penthouses and villas, one property per contract,
           yearly rent above AED 5,000, size 15-3,000 sqm, AED 5-1,000 per sqft, 2010-2026.
    These rules remove typing errors and deals that are not normal market sales.

What it saves in data/processed/ (small files, fine for GitHub):
    sales_by_area_year.csv  sales per year x area x type x market (Ready / Off-plan / All)
    rents_by_area_year.csv  rents per year x area x type x contract (New / Renewal)
    monthly_market.csv      sales per month for the whole city
    sales_sample.csv        a random 3% of sales since 2023 (used by the price model)
    area_names.json         area id -> area name
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SQM_TO_SQFT = 10.764        # 1 square metre = 10.764 square feet
CHUNK_SIZE = 250_000        # read the big files 250,000 rows at a time (saves memory)
SAMPLE_SHARE = 0.03         # keep 3% of recent sales for the price model
SAMPLE_FROM_YEAR = 2023

SALE_COLUMNS = ["instance_date", "trans_group_en", "property_usage_en", "property_type_en",
                "property_sub_type_en", "reg_type_en", "area_id", "area_name_en",
                "actual_worth", "procedure_area", "rooms_en"]
RENT_COLUMNS = ["contract_start_date", "property_usage_en", "ejari_property_type_en",
                "no_of_prop", "annual_amount", "actual_area", "area_id", "contract_reg_type_en"]
RENT_TYPES = {"Flat": "Apartment", "Studio": "Apartment", "Penthouse": "Apartment",
              "Villa": "Villa"}


# ---------------------------------------------------------------- helpers
def nice_name(name):
    """'AL BARSHA SOUTH FOURTH' -> 'Al Barsha South Fourth'."""
    name = str(name).strip().lower()
    return "".join(ch.upper() if i == 0 or name[i - 1] in " -(" else ch
                   for i, ch in enumerate(name))


def bedrooms(rooms):
    """'2 B/R' -> 2, 'Studio' -> 0, anything else -> empty."""
    beds = pd.to_numeric(rooms.str.extract(r"^(\d+)")[0], errors="coerce")
    return beds.mask(rooms == "Studio", 0)


def read_in_chunks(pattern, columns, test):
    """Read every file in data/raw/ that matches the pattern, a chunk at a time."""
    files = [f for f in sorted(RAW.glob(pattern)) if ("sample" in f.name) == test]
    if not files:
        raise FileNotFoundError(f"No files like '{pattern}' in {RAW}. Read the notes at the top.")
    for f in files:
        print(f"  reading {f.name}")
        yield from pd.read_csv(f, usecols=columns, chunksize=CHUNK_SIZE, low_memory=False)


# ---------------------------------------------------------------- cleaning
def clean_sales(chunk):
    c = chunk[(chunk["trans_group_en"] == "Sales") & (chunk["property_usage_en"] == "Residential")]
    sub_type = c["property_sub_type_en"].fillna("")
    is_flat = sub_type == "Flat"
    is_villa = (sub_type == "Villa") | ((sub_type == "") & (c["property_type_en"] == "Villa"))
    sqft = c["procedure_area"] * SQM_TO_SQFT

    sales = pd.DataFrame({
        "date": pd.to_datetime(c["instance_date"], errors="coerce"),
        "area_id": c["area_id"],
        "area": c["area_name_en"],
        "property_type": np.select([is_flat, is_villa], ["Apartment", "Villa"], ""),
        "bedrooms": bedrooms(c["rooms_en"].fillna("")),
        "size_sqft": sqft.round(0),
        "off_plan": c["reg_type_en"].fillna("").str.startswith("Off").astype(int),
        "price_aed": c["actual_worth"],
        "ppsf": c["actual_worth"] / sqft,          # price per square foot
    })
    keep = ((sales["property_type"] != "")
            & (sales["price_aed"] > 50_000)
            & (c["procedure_area"] > 15) & (c["procedure_area"] < 5_000)
            & sales["ppsf"].between(100, 20_000)
            & (sales["date"].dt.year >= 2008))
    return sales[keep]


def clean_rents(chunk):
    c = chunk[(chunk["property_usage_en"] == "Residential") & (chunk["no_of_prop"] == 1)]
    sqft = c["actual_area"] * SQM_TO_SQFT
    rents = pd.DataFrame({
        "year": pd.to_datetime(c["contract_start_date"], errors="coerce").dt.year,
        "area_id": c["area_id"],
        "property_type": c["ejari_property_type_en"].map(RENT_TYPES),
        "contract": np.where(c["contract_reg_type_en"] == "New", "New", "Renewal"),
        "annual_rent_aed": c["annual_amount"],
        "rent_psf": c["annual_amount"] / sqft,     # yearly rent per square foot
    })
    keep = (rents["property_type"].notna()
            & (rents["annual_rent_aed"] > 5_000)
            & (c["actual_area"] > 15) & (c["actual_area"] < 3_000)
            & rents["rent_psf"].between(5, 1_000)
            & rents["year"].between(2010, 2026))
    return rents[keep]


# ---------------------------------------------------------------- build files
def build_sales(test):
    sales = pd.concat(clean_sales(c) for c in read_in_chunks("transactions*.csv*", SALE_COLUMNS, test))
    sales["year"] = sales["date"].dt.year
    # the most common spelling of each area's name
    names = sales.groupby("area_id")["area"].agg(lambda s: nice_name(s.mode().iloc[0]))

    # yearly medians for ready homes, off-plan homes, and all homes together
    groups = ["year", "area_id", "property_type"]
    stats = dict(deals=("price_aed", "size"), median_ppsf=("ppsf", "median"),
                 median_price_aed=("price_aed", "median"))
    sales["market"] = sales["off_plan"].map({0: "Ready", 1: "Off-plan"})
    by_year = pd.concat([
        sales.groupby(groups + ["market"]).agg(**stats).reset_index(),
        sales.groupby(groups).agg(**stats).reset_index().assign(market="All"),
    ]).sort_values(groups + ["market"])
    by_year.insert(2, "area", by_year["area_id"].map(names))

    # monthly totals for the whole city
    month = sales["date"].dt.strftime("%Y-%m").rename("month")
    monthly = sales.groupby([month, "property_type", "off_plan"]).agg(
        deals=("price_aed", "size"), value_aed=("price_aed", "sum"),
        median_ppsf=("ppsf", "median")).reset_index()

    # random sample of recent sales for the price model
    recent = sales[sales["year"] >= SAMPLE_FROM_YEAR]
    sample = recent.sample(frac=SAMPLE_SHARE, random_state=42).sort_values("date")
    sample = sample.assign(date=sample["date"].dt.date, area=sample["area_id"].map(names))
    sample = sample[["date", "area", "property_type", "bedrooms", "size_sqft", "off_plan", "price_aed"]]

    by_year.round(1).to_csv(OUT / "sales_by_area_year.csv", index=False)
    monthly.round(1).to_csv(OUT / "monthly_market.csv", index=False)
    sample.to_csv(OUT / "sales_sample.csv", index=False)
    (OUT / "area_names.json").write_text(json.dumps({str(k): v for k, v in names.items()}))
    print(f"Sales: kept {len(sales):,} residential sales")
    return names


def build_rents(names, test):
    rents = pd.concat(clean_rents(c) for c in read_in_chunks("rent_contracts*.csv*", RENT_COLUMNS, test))
    out = rents.groupby(["year", "area_id", "property_type", "contract"]).agg(
        contracts=("rent_psf", "size"), median_rent_psf=("rent_psf", "median"),
        median_annual_rent_aed=("annual_rent_aed", "median")).reset_index()
    out.insert(2, "area", out["area_id"].map(names))
    out.round(1).to_csv(OUT / "rents_by_area_year.csv", index=False)
    print(f"Rents: kept {len(rents):,} residential rent contracts")


def main():
    global OUT
    parser = argparse.ArgumentParser(description="Build data/processed/ from the DLD files.")
    parser.add_argument("--test", action="store_true",
                        help="use the 1,500-row samples in data/raw/ and save to data/test_output/")
    test = parser.parse_args().test
    if test:
        OUT = ROOT / "data" / "test_output"
    OUT.mkdir(parents=True, exist_ok=True)
    names = build_sales(test)
    build_rents(names, test)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
