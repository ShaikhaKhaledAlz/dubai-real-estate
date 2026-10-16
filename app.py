"""
Interactive dashboard on official Dubai Land Department data.  Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from analysis import END, START, area_metrics, load_rents, load_sales  # noqa: E402
from area_labels import label  # noqa: E402

st.set_page_config(page_title="Dubai Real Estate Investment Dashboard", layout="wide")
st.title("Dubai Real Estate Investment Dashboard")
st.caption("Source: Dubai Land Department open data (sales and Ejari rent contracts) via data.dubai")

sales, rents = load_sales(), load_rents()
ptype = st.radio("Property type", ["Apartment", "Villa"], horizontal=True)
market = st.radio("Market", ["All", "Ready", "Off-plan"], horizontal=True)

s = sales[(sales["property_type"] == ptype) & (sales["market"] == market)]
top = (s[s["year"] == END].sort_values("deals", ascending=False)["area"].head(40).tolist())
areas = st.multiselect("Areas", top, default=top[:6], format_func=label)
f = s[s["area"].isin(areas)]

if f.empty:
    st.warning("No data for this selection.")
    st.stop()

latest = f[f["year"] == END]
c1, c2, c3 = st.columns(3)
c1.metric(f"Sales in {END}", f"{latest['deals'].sum():,}")
c2.metric(f"Median AED/sqft {END} (largest area)", f"{latest.sort_values('deals')['median_ppsf'].iloc[-1]:,.0f}")
c3.metric("Years of data", f"{f['year'].min()}-{f['year'].max()}")

st.plotly_chart(px.line(f.assign(area=f["area"].map(label)), x="year", y="median_ppsf", color="area",
                        markers=True, title=f"Median price per sqft, {ptype.lower()}s ({market.lower()})"),
                use_container_width=True)

st.subheader(f"Risk and return, ready apartments {START + 1}-{END}")
m, _ = area_metrics(sales, rents)
m = m.reset_index()
m["area"] = m["area"].map(label)
st.plotly_chart(px.scatter(m, x="volatility", y="avg_total_return", text="area", size="median_ppsf_latest",
                           title="Risk vs return (bubble size = price per sqft)"), use_container_width=True)
st.dataframe(m.style.format({"price_cagr": "{:.1%}", "price_cagr_5y": "{:.1%}",
                             "gross_yield_latest": "{:.1%}", "avg_gross_yield": "{:.1%}",
                             "avg_total_return": "{:.1%}", "volatility": "{:.1%}",
                             "sharpe_ratio": "{:.2f}", "median_ppsf_latest": "{:,.0f}"}))
