"""
Interactive dashboard.  Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from analysis import area_metrics, load_data, yearly_prices  # noqa: E402

st.set_page_config(page_title="Dubai Real Estate Investment Dashboard", layout="wide")
st.title("Dubai Real Estate Investment Dashboard")
st.caption("Sample data is synthetic. See README for how to plug in real Dubai Land Department data.")

df = load_data()
areas = st.multiselect("Areas", sorted(df["area"].unique()), default=sorted(df["area"].unique()))
ptypes = st.multiselect("Property type", sorted(df["property_type"].unique()),
                        default=sorted(df["property_type"].unique()))
f = df[df["area"].isin(areas) & df["property_type"].isin(ptypes)]

if f.empty:
    st.warning("No data for this selection.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Transactions", f"{len(f):,}")
c2.metric("Median price (AED)", f"{f['price_aed'].median():,.0f}")
c3.metric("Median gross yield", f"{f['gross_yield'].median() * 100:.1f}%")

prices = yearly_prices(f).reset_index().melt(id_vars="year", var_name="area", value_name="AED per sqft")
st.plotly_chart(px.line(prices, x="year", y="AED per sqft", color="area", markers=True,
                        title="Median price per sqft"), use_container_width=True)

m = area_metrics(f).reset_index()
st.plotly_chart(px.scatter(m, x="volatility", y="total_return", text="area", size="median_ppsf_latest",
                           title="Risk vs return (bubble size = price per sqft)"), use_container_width=True)
st.dataframe(m.style.format({"price_cagr": "{:.1%}", "gross_yield": "{:.1%}", "volatility": "{:.1%}",
                             "total_return": "{:.1%}", "sharpe_ratio": "{:.2f}",
                             "median_ppsf_latest": "{:,.0f}"}))
