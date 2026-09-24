# Dubai Real Estate Investment Analysis

**Which Dubai neighbourhoods give investors the best return for the risk they take?**

This project looks at Dubai residential property the way an investment analyst would. It measures price growth, rental yield, volatility and risk-adjusted return (Sharpe ratio) for 10 major areas. It also trains a machine-learning model that estimates property prices and flags deals that look undervalued.

**Tools:** Python · pandas · scikit-learn · matplotlib · Streamlit · Plotly

> ⚠️ **About the data:** The file `data/sample_transactions.csv` is **synthetic** (simulated) data built to look like Dubai's market. It lets anyone run the project straight away. The numbers are **not** official statistics. You can switch to real Dubai Land Department data using the steps in [Using real DLD data](#using-real-dld-data).

---

## Key findings (sample data)

| Area | Price growth / yr | Gross rental yield | Volatility | Total return / yr | Sharpe ratio |
|---|---|---|---|---|---|
| Dubai South | 7.0% | 7.5% | 4.6% | 14.5% | 2.29 |
| Palm Jumeirah | 9.4% | 5.0% | 5.5% | 14.4% | 1.89 |
| Jumeirah Lake Towers | 7.0% | 7.2% | 5.7% | 14.2% | 1.80 |
| Dubai Marina | 6.9% | 6.3% | 5.2% | 13.2% | 1.78 |
| Jumeirah Village Circle | 7.3% | 7.9% | 6.4% | 15.1% | 1.73 |
| Dubai Hills Estate | 8.1% | 5.9% | 6.0% | 14.0% | 1.65 |
| Arabian Ranches | 6.8% | 5.5% | 5.0% | 12.3% | 1.65 |
| Business Bay | 7.6% | 6.5% | 6.1% | 14.1% | 1.64 |
| Downtown Dubai | 6.0% | 5.5% | 5.7% | 11.5% | 1.31 |
| International City | 3.2% | 8.9% | 6.7% | 12.1% | 1.21 |

**What this shows**

1. **Prime and affordable areas make money in different ways.** Palm Jumeirah returns come mostly from price growth. International City and JVC depend more on rental income.
2. **The highest return isn't always the best investment.** JVC had the highest total return, but once volatility is counted, Dubai South and Palm Jumeirah come out ahead on a risk-adjusted basis.
3. **The price model is accurate enough to screen deals.** It explains about 97% of price variation (R² = 0.97) with a typical error of about 10%. It found 93 recent deals priced more than 15% below the model's estimate.

### Price trends
![Price trends](images/price_trends.png)

### Risk vs return
![Risk vs return](images/risk_return.png)

### Rental yields
![Rental yields](images/rental_yields.png)

### Price model accuracy
![Predicted vs actual](images/predicted_vs_actual.png)

---

## Methodology

| Metric | How it's calculated |
|---|---|
| Price per sqft | Transaction price ÷ size, median per area per year |
| Price CAGR | (latest median ÷ first median)^(1 ÷ years) − 1 |
| Gross rental yield | Annual rent ÷ purchase price (median) |
| Total return | Price CAGR + gross rental yield (simple approximation, before costs) |
| Volatility | Standard deviation of yearly % change in median price per sqft |
| Sharpe ratio | (Total return − 4% risk-free rate) ÷ volatility |

**Price model:** Gradient Boosting Regressor on log(price). Features: area, property type, size, bedrooms, year, off-plan flag. The data is split 80/20 into training and test sets, and accuracy is measured only on the unseen 20%.

**Limitations:** Gross yield ignores service charges, vacancy and fees. Total return is a simple sum, not a full IRR. Median prices can shift when the mix of properties sold changes from year to year.

---

## Project structure

```
dubai-real-estate/
├── app.py                        # Interactive Streamlit dashboard
├── data/
│   ├── sample_transactions.csv   # Synthetic sample data (8,400 transactions)
│   ├── area_metrics.csv          # Output: metrics per area
│   └── possible_undervalued_deals.csv  # Output: model's value finder
├── images/                       # Charts used in this README
├── src/
│   ├── generate_sample_data.py   # Creates the synthetic dataset
│   ├── analysis.py               # Investment metrics + charts
│   └── model.py                  # ML price model + value finder
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/ShaikhaKhaledAlz/dubai-real-estate.git
cd dubai-real-estate
pip install -r requirements.txt

python src/analysis.py      # metrics table + charts
python src/model.py         # price model + undervalued deals
streamlit run app.py        # interactive dashboard in your browser
```

## Using real DLD data

The Dubai Land Department publishes open transaction and rental data on the Dubai Pulse / DLD open-data portals.

1. Download the transactions (and rents) CSV from the DLD open-data portal.
2. Rename the columns to match the ones used here: `transaction_date, area, property_type, bedrooms, size_sqft, off_plan, price_aed, annual_rent_aed`. Official sizes are often in sq m. Multiply by 10.764 to get sq ft.
3. Save the file as `data/sample_transactions.csv` (or change the `DATA` path in `src/analysis.py`) and run the scripts again.

## Next steps

- Connect to live DLD data and refresh automatically
- Net yield after service charges and fees; a full IRR model with mortgage leverage
- Compare with other Gulf markets (Abu Dhabi, Riyadh) and global cities
- Add macro factors (interest rates, oil price, population growth)

---

*Built by Shaikha as a portfolio project in data analysis and investment research.*
