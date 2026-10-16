"""
STEP 5 - Interactive dashboard (opens in your web browser).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

# let Python find the scripts in the src/ folder
sys.path.append(str(Path(__file__).parent / "src"))
from analysis import END, RISK_FREE_RATE, START, area_metrics, load_rents, load_sales  # noqa: E402
from area_labels import label  # noqa: E402

st.set_page_config(page_title="Dubai Real Estate Investment Dashboard", layout="wide")
st.title("Dubai Real Estate Investment Dashboard")
st.caption("Source: official Dubai Land Department open data (sales and Ejari rent contracts) "
           "from data.dubai. Area names: common name (official DLD name).")

sales, rents = load_sales(), load_rents()

# ---------------------------------------------------------------- price history
st.header("Price history by area")
ptype = st.radio("Property type", ["Apartment", "Villa"], horizontal=True)
market = st.radio("Ready or off-plan?", ["All", "Ready", "Off-plan"], horizontal=True)

s = sales[(sales["property_type"] == ptype) & (sales["market"] == market)]
busiest = s[s["year"] == END].sort_values("deals", ascending=False)["area"].head(40).tolist()
areas = st.multiselect("Areas (the 40 busiest)", busiest, default=busiest[:6], format_func=label)
chosen = s[s["area"].isin(areas)]

if chosen.empty:
    st.warning("Pick at least one area.")
    st.stop()

latest = chosen[chosen["year"] == END]
c1, c2, c3 = st.columns(3)
c1.metric(f"Sales in {END} (selected areas)", f"{latest['deals'].sum():,}")
c2.metric(f"Typical price per sqft in {END}",
          f"AED {latest.sort_values('deals')['median_ppsf'].iloc[-1]:,.0f}",
          help="Median price per sqft in the busiest selected area")
c3.metric("Years of data", f"{chosen['year'].min()}-{chosen['year'].max()}")

chart = chosen.assign(area=chosen["area"].map(label))
st.plotly_chart(px.line(chart, x="year", y="median_ppsf", color="area", markers=True,
                        labels={"median_ppsf": "Median AED per sqft", "year": "Year"},
                        title=f"Median price per sqft, {ptype.lower()}s ({market.lower()})"),
                use_container_width=True)

# ---------------------------------------------------------------- risk and return
st.header(f"Risk and return, ready apartments {START + 1}-{END}")
st.write("Higher = more return. Further right = more risk (prices and returns jump around more). "
         f"The best areas sit top-left. Risk-free rate used: {RISK_FREE_RATE:.1%}.")
m, _ = area_metrics(sales, rents)
m = m.reset_index()
m["area"] = m["area"].map(label)
st.plotly_chart(px.scatter(m, x="volatility", y="avg_total_return", text="area",
                           size="median_ppsf_latest",
                           labels={"volatility": "Risk (volatility)",
                                   "avg_total_return": "Average yearly total return"},
                           title="Risk vs return (bubble size = price per sqft)"),
                use_container_width=True)
st.dataframe(m.style.format({"price_cagr": "{:.1%}", "price_cagr_5y": "{:.1%}",
                             "gross_yield_latest": "{:.1%}", "avg_gross_yield": "{:.1%}",
                             "avg_total_return": "{:.1%}", "volatility": "{:.1%}",
                             "sharpe_ratio": "{:.2f}", "median_ppsf_latest": "{:,.0f}"}))
