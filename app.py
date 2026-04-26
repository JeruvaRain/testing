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
    "the charts to see details for each country–year."
)

st.write(f"Currently showing data for **{year_min}–{year_max}**.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Countries in view", df_view["Entity"].nunique())
with col2:
    st.metric("Observations", len(df_view))
with col3:
    if model is not None:
        st.metric("R² (current filters)", f"{model.rsquared:.3f}")
    else:
        st.metric("R² (current filters)", "N/A")

# -----------------------
# Tabs for different views
# -----------------------
tab1, tab2, tab3 = st.tabs(["GDP vs Gini", "Consumption vs Gini", "Summary & Model"])

# -----------------------
# Tab 1: GDP vs Gini (Altair)
# -----------------------
with tab1:
    st.subheader("GDP per capita vs inequality")

    if not df_view.empty:
        scatter_gdp = (
            alt.Chart(df_view)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("log_gdp_pc:Q", title="log(GDP per capita)"),
                y=alt.Y("Gini coefficient:Q", title="Gini coefficient"),
                color=alt.Color("Entity:N", title="Country",
                                legend=alt.Legend(columns=2)),
                tooltip=[
                    alt.Tooltip("Entity:N", title="Country"),
                    alt.Tooltip("Year:O", title="Year"),
                    alt.Tooltip("Gini coefficient:Q", title="Gini", format=".3f"),
                    alt.Tooltip("GDP per capita:Q", title="GDP pc", format=".0f"),
                    alt.Tooltip("cons_pct_gdp:Q", title="Cons % GDP", format=".2f"),
                ],
            )
            .properties(height=400)
            .interactive()
        )

        # Optional regression line (visual only)
        reg_line = scatter_gdp.transform_regression(
            "log_gdp_pc", "Gini coefficient"
        ).mark_line(color="black")

        st.altair_chart(scatter_gdp + reg_line, use_container_width=True)
    else:
        st.write("No data for the selected filters.")

# -----------------------
# Tab 2: Consumption vs Gini (Altair)
# -----------------------
with tab2:
    st.subheader("Consumption share vs inequality")

    if not df_view.empty:
        scatter_cons = (
            alt.Chart(df_view)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(
                    "cons_pct_gdp:Q",
                    title="Final consumption expenditure (% of GDP)"
                ),
                y=alt.Y("Gini coefficient:Q", title="Gini coefficient"),
                color=alt.Color("Entity:N", title="Country",
                                legend=alt.Legend(columns=2)),
                tooltip=[
                    alt.Tooltip("Entity:N", title="Country"),
                    alt.Tooltip("Year:O", title="Year"),
                    alt.Tooltip("Gini coefficient:Q", title="Gini", format=".3f"),
                    alt.Tooltip("GDP per capita:Q", title="GDP pc", format=".0f"),
                    alt.Tooltip("cons_pct_gdp:Q", title="Cons % GDP", format=".2f"),
                ],
            )
            .properties(height=400)
            .interactive()
        )

        reg_line2 = scatter_cons.transform_regression(
            "cons_pct_gdp", "Gini coefficient"
        ).mark_line(color="black")

        st.altair_chart(scatter_cons + reg_line2, use_container_width=True)
    else:
        st.write("No data for the selected filters.")

# -----------------------
# Tab 3: Summary table & model snapshot
# -----------------------
with tab3:
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

    st.markdown("---")
    st.subheader("Regression snapshot (current filters)")

    if model is not None:
        st.write(
            "OLS regression of Gini on log(GDP per capita) and "
            "final consumption expenditure (% of GDP), estimated using "
            "the countries and years currently selected."
        )

        coef_df = pd.DataFrame({
            "coef": model.params,
            "std_err": model.bse,
            "p_value": model.pvalues
        }).round(4)

        st.dataframe(coef_df)
    else:
        st.write("Not enough data in the current filter to estimate the regression.")
