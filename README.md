# Dubai Real Estate Investment Analysis

![check](https://github.com/ShaikhaKhaledAlz/dubai-real-estate/actions/workflows/check.yml/badge.svg)

**Question:** if you bought an apartment in Dubai to rent out, which neighbourhoods gave the best return for the risk you took?

This project answers that question with **official Dubai Land Department (DLD) data only**. It does not use any made-up or sample data. It covers:

- **1,093,169 residential sales** registered from January 2008 to 15 October 2026
- **5,500,720 residential rent contracts** (Ejari) from 2010 to 2026

**Tools:** Python · pandas · scikit-learn · matplotlib · Streamlit · Plotly

---

## Key results at a glance

1. **Dubai went through a full boom-bust-boom cycle.** In the 18 areas studied, ready apartment prices fell by a median of 35% between 2015 and 2020. They then rose by a median of 80% between 2020 and 2025.
2. **Rent did most of the work for investors.** Over 2015–2025, prices grew between −1.5% and +4.8% a year. Gross rental yields of 5.4–9.5% a year made up most of the return.
3. **JVC and Dubai Marina gave the best return for the risk.** They had the highest Sharpe ratios (0.79 and 0.74).
4. **Cheaper areas pay higher rent yields.** International City yields 9.5%, while Palm Jumeirah yields 5.4% and Downtown Dubai 5.8%.
5. **The market is huge but cooling in 2026.** 2025 was a record year with 181,818 sales, 72% of them off-plan. January–September 2026 sales are 18% below the same months of 2025.
6. **A machine-learning model can price homes fairly well.** Tested on 2026 sales it had never seen, its typical error is 12%.

---

## Key terms explained

| Term | What it means |
|---|---|
| **Median** | The middle value when you sort all prices. Unlike the average, a few very expensive sales can't pull it up much. |
| **Price per sqft** | Price ÷ size in square feet. Lets you compare a small flat with a big one. |
| **Ready vs off-plan** | *Ready* = the building is finished. *Off-plan* = bought from the developer before it's built. |
| **Gross rental yield** | One year's rent ÷ the price of the home. Rent of 70,000 on a 1,000,000 flat = 7%. "Gross" means before costs such as service charges. |
| **Price growth (CAGR)** | The average yearly growth rate. +4% a year over 10 years turns 100 into 148. |
| **Total return** | Price growth + rental yield in the same year. It is what an investor earned. |
| **Volatility (risk)** | How much the yearly return jumps up and down. Higher = less predictable. |
| **Risk-free rate** | What you could earn with almost no risk. Here: US Treasury bills, averaging 2.2% a year over 2016–2025. The UAE dirham is pegged to the US dollar. |
| **Sharpe ratio** | (Total return − risk-free rate) ÷ volatility = extra return per unit of risk. Higher is better. |
| **R²** | How much of the price differences the model explains. 1.00 = perfect, 0 = useless. |

---

## 1. Market overview

![Monthly market](images/market_monthly.png)

| Year | Residential sales | Value (AED bn) | Off-plan share |
|---|---|---|---|
| 2015 | 31,902 | 51.6 | 38% |
| 2020 | 26,410 | 39.2 | 50% |
| 2023 | 108,210 | 234.1 | 60% |
| 2024 | 154,503 | 319.5 | 68% |
| 2025 | 181,818 | 390.1 | 72% |
| 2026 (Jan–Sep) | 106,780 | 221.0 | 75% |

*All years are in `data/processed/yearly_market.csv`.*

- **Record volumes, driven by off-plan.** 2025 had almost 7 times as many sales as 2020. Off-plan's share rose from 38% in 2015 to 72% in 2025.
- **2026 is slowing.** Monthly sales dropped from about 17,000 in late 2025 to about 10,000 in September 2026. Ready apartment prices have been flat at about AED 1,400 per sq ft.
- **The busiest areas are newer mid-market communities.** JVC led 2025 with 17,909 sales, followed by Business Bay, Dubailand Residence Complex & Arabian Ranches 3, and Dubai South.

![Off-plan share](images/offplan_share.png)
![Top areas](images/top_areas.png)

### About area names

DLD records sales under **official land-registry names**, which are often not the names people use. For example, *Marsa Dubai* is Dubai Marina and *Al Barsha South Fourth* is JVC.

The common names in this project were checked against the DLD data itself. For each area, I looked at which master project (community) its sales belong to. 100% of sales in *Al Barsha South Fourth* are in the master project "Jumeirah Village Circle". The full list, with its DLD names, is in `src/area_labels.py`.

---

## 2. Risk and return by area (ready apartments, 2015–2025)

To keep the results reliable, an area is only included if it had **at least 50 ready apartment sales and 100 new rent contracts in every year from 2015 to 2025**. 18 areas pass this test.

| Area (DLD name) | AED/sqft 2025 | Price growth / yr, 10 yrs | Price growth / yr, last 5 yrs | Rental yield 2025 | Avg total return / yr | Volatility | Sharpe |
|---|---|---|---|---|---|---|---|
| JVC (Al Barsha South Fourth) | 1,274 | 4.0% | 15.4% | 7.9% | 12.9% | 13.5% | 0.79 |
| Dubai Marina, JBR & Dubai Harbour (Marsa Dubai) | 1,900 | 3.3% | 13.3% | 6.2% | 10.6% | 11.3% | 0.74 |
| Al Furjan & Discovery Gardens (Jabal Ali First) | 1,040 | 2.4% | 13.0% | 8.6% | 11.9% | 14.1% | 0.69 |
| Dubai Silicon Oasis (Nadd Hessa) | 899 | 1.1% | 10.2% | 8.5% | 10.5% | 12.4% | 0.67 |
| Arjan (Al Barshaa South Third) | 1,284 | 2.1% | 9.6% | 7.2% | 10.5% | 12.7% | 0.66 |
| The Greens (Al Thanyah Third) | 1,738 | 2.7% | 15.6% | 6.7% | 11.3% | 14.5% | 0.63 |
| International City (Al Warsan First) | 629 | −1.1% | 8.1% | 9.5% | 9.1% | 11.1% | 0.62 |
| JLT (Al Thanyah Fifth) | 1,420 | 1.8% | 14.1% | 7.6% | 10.8% | 14.3% | 0.60 |
| Business Bay | 1,778 | 1.2% | 9.6% | 7.1% | 8.2% | 10.5% | 0.57 |
| Palm Jumeirah | 2,563 | 4.8% | 20.2% | 5.4% | 12.8% | 19.4% | 0.55 |
| Downtown Dubai (Burj Khalifa) | 2,539 | 2.5% | 15.3% | 5.8% | 9.4% | 13.4% | 0.54 |

*Top 11 of 18 areas, sorted by Sharpe ratio. The full table is in `data/processed/area_metrics.csv`.*

**How to read this table:** take JVC. A ready apartment there cost AED 1,274 per sq ft in 2025. Prices grew 4.0% a year over 10 years, and 15.4% a year over the last 5. Rent in 2025 was 7.9% of the price. Adding growth and rent, an investor earned about 12.9% a year on average.

**What it shows**

- **Rent was the steady part of the return.** The 2015–2020 fall cancelled much of the later price rise, so over 10 years most of the return came from rent.
- **Prime vs affordable.** Palm Jumeirah had the fastest price growth since 2020 (20% a year) but the lowest yield (5.4%). International City had the highest yield (9.5%) but the slowest growth (8.1% a year since 2020).
- **Every area beat cash.** All 18 areas averaged 8–13% a year before costs, compared with the 2.2% risk-free rate.

![Price trends](images/price_trends.png)
![Risk vs return](images/risk_return.png)
![Rental yields](images/rental_yields.png)

---

## 3. Price model and value finder

The model learns the price of a home from six facts: area, apartment or villa, size, bedrooms, off-plan or ready, and the month of the sale. It is a **Gradient Boosting** model, which builds hundreds of small decision trees, each one fixing the mistakes of the previous ones.

**A fair test:** the model learns from 13,323 sales in 2023–2025. It is then tested on 3,248 sales from 2026 that it has never seen. This is harder than a random split, because the model has to predict the future, just as it would in real life.

| Result on unseen 2026 sales | |
|---|---|
| R² (log price) | 0.90 |
| Median error | 12.0% |
| Predictions within ±20% of the real price | 71% |

**Value finder:** 335 of the 2026 sales (10%) were 20–50% below the model's estimate. They are saved in `data/processed/possible_undervalued_deals.csv`. These are leads to research, not recommendations: a low price can have reasons the data doesn't show, such as the view, floor or condition. Gaps above 50% are left out because they are almost always not normal sales, for example selling a share of a home.

![Predicted vs actual](images/predicted_vs_actual.png)

---

## How the project works

```
Official DLD files (6 GB)  ──►  1. build_dataset.py   cleans them into small summary files
                                       │
                                       ▼
                          data/processed/*.csv  (already in this repo)
                                       │
          ┌────────────────────────────┼─────────────────────────────┐
          ▼                            ▼                             ▼
 2. market_overview.py        3. analysis.py                 4. model.py
 sales, prices, off-plan      growth, yield, risk, Sharpe    price model + value finder
          └────────────────────────────┼─────────────────────────────┘
                                       ▼
                         5. app.py  interactive dashboard
```

| Folder / file | What's inside |
|---|---|
| `data/raw/` | 1,500-row samples of the two official DLD files (the full files are too big for GitHub) |
| `data/processed/` | Summary files built from the full DLD data, plus the results of each script |
| `images/` | The charts used in this README |
| `src/` | The Python scripts, numbered by step in their first line |
| `tests/` | Automatic checks that GitHub runs on every change (the green "check" badge at the top) |
| `app.py` | The dashboard |

---

## How to run it yourself

1. Install [Python](https://www.python.org/downloads/) (3.10 or newer).
2. Open a terminal and run:

```bash
git clone https://github.com/ShaikhaKhaledAlz/dubai-real-estate.git
cd dubai-real-estate
pip install -r requirements.txt

python src/market_overview.py   # step 2: market charts
python src/analysis.py          # step 3: risk/return table and charts
python src/model.py             # step 4: price model and value finder
streamlit run app.py            # step 5: dashboard opens in your browser
```

The summary data is already in the repo, so you don't need to download anything else.

### Updating with the latest DLD data (optional)

1. Download all CSV parts of [Real Estate Transactions](https://data.dubai/en/l/470061) (about 1.1 GB) and [Rent Contracts](https://data.dubai/en/l/468586) (about 5 GB) from Data Dubai. Put them in `data/raw/`. They are ignored by git because they are too big.
2. Run `python src/build_dataset.py`, then the scripts above.
3. To check the cleaning step without the big download, run `python src/build_dataset.py --test`. It uses the samples in `data/raw/`.

---

## Data cleaning rules

- **Sales kept:** sales only (not mortgages or gifts), residential flats and villas, price above AED 50,000, size 15–5,000 sq m, price AED 100–20,000 per sq ft, from 2008.
- **Rents kept:** residential flats, studios, penthouses and villas, one property per contract, yearly rent above AED 5,000, size 15–3,000 sq m, rent AED 5–1,000 per sq ft a year.
- **Price model only:** sales below AED 300 per sq ft are removed, because they are not normal market sales today.

## Limitations (what to keep in mind)

- **Gross, not net.** Yields ignore service charges, empty months, agent fees and mortgage costs, so real returns are lower.
- **Simple total return.** Total return adds price growth and yield. It is not a full IRR.
- **Medians move with the mix.** If an area sells more new or bigger buildings in one year, its median moves even if prices didn't. Using ready apartments and price per sq ft reduces this but doesn't remove it. Dubai Production City shows it clearly: its median jumped 75% in 2021 and fell 30% in 2022, so its high volatility is mostly a mix effect.
- **2008–2009 look unusual.** 2009 shows three times as many sales as 2008, half of them off-plan. The investment analysis starts in 2015, so it doesn't rely on those years.
- **Recent months can still change**, because some sales are registered late.
- **Area names.** Area boundaries are DLD's. A DLD area can contain more than one community, for example Dubai Marina, JBR and Dubai Harbour are all in *Marsa Dubai*.

## Next steps

- Net yield after service charges (DLD service charge index) and a full IRR with a mortgage
- A repeat-sales index (the same home sold twice) to remove mix effects completely
- Compare with Abu Dhabi and other Gulf markets; add drivers such as interest rates and population growth

---

*Built by Shaikha as a portfolio project in data analysis and investment research.*
*Data: © Dubai Land Department, published on [Data Dubai](https://data.dubai) under its open data licence. Risk-free rate: US 3-month T-bill yearly averages from [FRED](https://fred.stlouisfed.org/series/RIFSGFSM03NA).*
