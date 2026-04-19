# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("ineq_data.csv")

# Ensure Year is integer (just in case)
df["Year"] = df["Year"].astype(int)

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.header("Filters")

# Country multiselect
all_countries = sorted(df["Entity"].unique())
selected_countries = st.sidebar.multiselect(
    "Countries to display",
    options=all_countries,
    default=all_countries
)

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
if selected_countries:
    df_view = df[df["Entity"].isin(selected_countries)]
else:
    df_view = df.copy()

df_view = df_view[(df_view["Year"] >= year_min) & (df_view["Year"] <= year_max)]

# -----------------------
# Fit regression on filtered data
# (still using log_gdp_pc and cons_pct_gdp)
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
# Main title and description
# -----------------------
st.title("Inequality, Growth and Consumption")

st.write(
    "Explore how income levels and final consumption expenditure "
    "(% of GDP) relate to the Gini coefficient for a selected set "
    "of countries and years."
)

st.write(f"Showing data for **{year_min}–{year_max}**.")

# -----------------------
# Plot 1: GDP vs Gini
# -----------------------
st.subheader("GDP per capita vs inequality")

fig, ax = plt.subplots(figsize=(6, 4))

for c in sorted(df_view["Entity"].unique()):
    subset = df_view[df_view["Entity"] == c]
    ax.scatter(
        subset["log_gdp_pc"],
        subset["Gini coefficient"],
        alpha=0.6,
        label=c
    )

ax.set_xlabel("log(GDP per capita)")
ax.set_ylabel("Gini coefficient")
if df_view["Entity"].nunique() <= 10:
    ax.legend()
st.pyplot(fig)

# -----------------------
# Summary table
# -----------------------
st.subheader("Summary statistics for selected countries")

if not df_view.empty:
    summary = (
        df_view[["Entity", "Gini coefficient", "GDP per capita", "cons_pct_gdp"]]
        .groupby("Entity")
        .agg({
            "Gini coefficient": "mean",
            "GDP per capita": "mean",
            "cons_pct_gdp": "mean",
        })
        .rename(columns={
            "Gini coefficient": "Mean Gini",
            "GDP per capita": "Mean GDP pc",
            "cons_pct_gdp": "Mean cons % GDP",
        })
        .round(3)
    )
    st.dataframe(summary)
else:
    st.write("No data for the selected filters.")

# -----------------------
# Plot 2: Consumption vs Gini
# -----------------------
st.subheader("Consumption share vs inequality")

fig2, ax2 = plt.subplots(figsize=(6, 4))

for c in sorted(df_view["Entity"].unique()):
    subset = df_view[df_view["Entity"] == c]
    ax2.scatter(
        subset["cons_pct_gdp"],
        subset["Gini coefficient"],
        alpha=0.6,
        label=c
    )

ax2.set_xlabel("Final consumption expenditure (% of GDP)")
ax2.set_ylabel("Gini coefficient")
if df_view["Entity"].nunique() <= 10:
    ax2.legend()
st.pyplot(fig2)

# -----------------------
# Optional: brief model info
# -----------------------
if model is not None:
    st.subheader("Model snapshot (filtered data)")
    st.write(
        "Simple OLS regression of Gini on log(GDP per capita) and "
        "consumption share (% of GDP) for the currently selected "
        "countries and years."
    )
    coef_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "p_value": model.pvalues
    }).round(4)
    st.dataframe(coef_df)
else:
    st.write("Not enough data in the current filter to estimate the regression.")
