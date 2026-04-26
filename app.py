# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altair as alt

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ineq_data.csv")
    df["Year"] = df["Year"].astype(int)
    return df

df = load_data()

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")

# Country multiselect (all countries available in the dataset)
all_countries = sorted(df["Entity"].unique())
selected_countries = st.sidebar.multiselect(
    "Countries to display",
    options=all_countries,
    default=all_countries[:10] if len(all_countries) > 10 else all_countries
)

# Optional region filter if you later add a 'Region' column to the CSV
# all_regions = ["All"] + sorted(df["Region"].dropna().unique())
# selected_region = st.sidebar.selectbox("Region", options=all_regions)

# Year range slider
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

year_min, year_max = st.sidebar.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# -----------------------
# Apply filters
# -----------------------
df_view = df.copy()

if selected_countries:
    df_view = df_view[df_view["Entity"].isin(selected_countries)]

# If you add a Region column later, uncomment this:
# if selected_region != "All":
#     df_view = df_view[df_view["Region"] == selected_region]

df_view = df_view[(df_view["Year"] >= year_min) & (df_view["Year"] <= year_max)]

# -----------------------
# Fit regression on filtered data
# -----------------------
reg_data = df_view[["Gini coefficient", "log_gdp_pc", "cons_pct_gdp"]].dropna()

if len(reg_data) > 10:
    X = reg_data[["log_gdp_pc", "cons_pct_gdp"]]
    X = sm.add_constant(X)
    y = reg_data["Gini coefficient"]
    model = sm.OLS(y, X).fit()
else:
    model = None

# -----------------------
# Layout: title, description, metrics
# -----------------------
st.title("Inequality, Growth and Consumption")

st.write(
    "Explore how income levels and final consumption expenditure "
    "(% of GDP) relate to the Gini coefficient. Use the filters on "
    "the left to select countries and years. Hover over points in "
    "the charts to see details for 
