# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import altair as alt

# --------------- #
# 1. Cargar datos #
# --------------- #

@st.cache_data
def load_data():
    df = pd.read_csv("ineq_data.csv")
    df["Year"] = df["Year"].astype(int)

    df = df.rename(columns={
        "log_gdp_per_capita": "log_gdp_pc",
        "Final consumption expenditure (% of GDP)": "cons_pct_gdp",
        "gini": "Gini coefficient",
        "gdp_per_capita": "GDP per capita",
    })

    if "log_gdp_pc" not in df.columns and "GDP per capita" in df.columns:
        df["log_gdp_pc"] = np.log(df["GDP per capita"])

    return df

# -------------------------------------------- #
# 2. Cargar los datos UNA VEZ y usar df abajo  #
# -------------------------------------------- #

df = load_data()
has_region = "Region" in df.columns

# -------------------- #
# 3. Filtros laterales #
# -------------------- #

st.sidebar.header("Filtros")

# Selector de países (comienza sin ninguno seleccionado)
all_countries = sorted(df["Entity"].unique())
selected_countries = st.sidebar.multiselect(
    "Países a mostrar",
    options=all_countries,
    default=[],
)

# Selector múltiple de regiones (vacío = todas las regiones)
if has_region:
    all_regions = sorted(df["Region"].dropna().unique())
    selected_regions = st.sidebar.multiselect(
        "Regiones a mostrar",
        options=all_regions,
        default=[],
    )
else:
    selected_regions = []

# Rango de años
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
year_min, year_max = st.sidebar.slider(
    "Rango de años",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

# ------------------ #
# 4. Aplicar filtros #
# ------------------ #

df_view = df.copy()

# Filtrar por regiones seleccionadas (si las hay)
if has_region and selected_regions:
    df_view = df_view[df_view["Region"].isin(selected_regions)]

# Filtrar por países seleccionados (si los hay)
if selected_countries:
    df_view = df_view[df_view["Entity"].isin(selected_countries)]

# Filtrar por rango de años
df_view = df_view[(df_view["Year"] >= year_min) & (df_view["Year"] <= year_max)]

# Regla de colores:
# - Si hay más de una región visible -> color por Región
# - Si solo hay una región visible (o no hay columna Region) -> color por País
if has_region and not df_view.empty:
    n_regions = df_view["Region"].nunique()
else:
    n_regions = 0

color_field = "Region:N" if n_regions > 1 else "Entity:N"
color_title = "Región" if n_regions > 1 else "País"

# --------------------------------------- #
# 5. Regresión con los datos filtrados    #
# --------------------------------------- #

reg_data = df_view[["Gini coefficient", "log_gdp_pc", "cons_pct_gdp"]].dropna()

if len(reg_data) > 10:
    X = reg_data[["log_gdp_pc", "cons_pct_gdp"]]
    X = sm.add_constant(X)
    y = reg_data["Gini coefficient"]
    model = sm.OLS(y, X).fit()
else:
    model = None

# -------------------------------- #
# 6. ítulo, descripción y métricas #
# -------------------------------- #

st.title("Desigualdad, crecimiento y consumo")

st.write(
    "Explora cómo los niveles de ingreso y el gasto de consumo final "
    "(% del PIB) se relacionan con el coeficiente de Gini. Usa los filtros "
    "de la izquierda para seleccionar países, regiones y años. Pasa el ratón "
    "por los puntos de los gráficos para ver los detalles de cada país‑año. "
    "La línea azul representa los resultados de la regresión o la tendencia esperada."
)

st.write(f"Actualmente se muestran datos para **{year_min}–{year_max}**.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="Países en la vista", 
        value=df_view["Entity"].nunique(),
        help="Número de países distintos que aparecen en la base de datos"
    )
with col2:
    st.metric(
        label="Observaciones",
        value=len(df_view),
        help="Número total de filas (país-año) utilizadas"
        )
with col3:
    if model is not None:
        st.metric(
            label="R² (filtros actuales)",
            value=f"{model.rsquared:.3f}",
            help="R² (coeficiente de determinación): mide qué proporción de la variación del coeficiente de Gini\n"
            "es explicada por las variables del modelo (log(PIB per cápita) y consumo % PIB) con los filtros actuales".
            )
    else:
        st.metric(
            label="R² (filtros actuales)",
            value="N/D",
            help="No se muestra R² poque no hay suficientes datos con los filtros actuales para estimar la regresión."
            )

# ----------- #
# 7. Pestañas #
# ----------- #

tab1, tab2, tab3 = st.tabs(["PIB vs Gini", "Consumo vs Gini", "Resumen y modelo"])


# Pestaña 1: PIB vs Gini #

with tab1:
    st.subheader("PIB per cápita vs desigualdad")

    if not df_view.empty:
        scatter_gdp = (
            alt.Chart(df_view)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("log_gdp_pc:Q", title="log(PIB per cápita)"),
                y=alt.Y("Gini coefficient:Q", title="Coeficiente de Gini"),
                color=alt.Color(
                    color_field,
                    title=color_title,
                    legend=alt.Legend(columns=2),
                    scale=alt.Scale(scheme="tableau10"),
                ),
                tooltip=[
                    alt.Tooltip("Entity:N", title="País"),
                    alt.Tooltip("Region:N", title="Región") if has_region else None,
                    alt.Tooltip("Year:O", title="Año"),
                    alt.Tooltip("Gini coefficient:Q", title="Gini", format=".3f"),
                    alt.Tooltip("GDP per capita:Q", title="PIB per cápita", format=".0f"),
                    alt.Tooltip("cons_pct_gdp:Q", title="Consumo % PIB", format=".2f"),
                ],
            )
            .properties(height=400)
            .interactive()
        )

        reg_line = (
            alt.Chart(df_view)
            .transform_regression(
                "log_gdp_pc",
                "Gini coefficient",
                as_=["log_gdp_pc", "gini_pred"],
            )
            .mark_line(size=2)
            .encode(
                x="log_gdp_pc:Q",
                y="gini_pred:Q",
                color=alt.value("deepskyblue"),
            )
        )

        st.altair_chart(scatter_gdp + reg_line, width="stretch")
    else:
        st.write("No hay datos para los filtros seleccionados.")


# Pestaña 2: Consumo vs Gini   #

with tab2:
    st.subheader("Cuota de consumo vs desigualdad")

    if not df_view.empty:
        scatter_cons = (
            alt.Chart(df_view)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(
                    "cons_pct_gdp:Q",
                    title="Gasto de consumo final (% del PIB)",
                ),
                y=alt.Y("Gini coefficient:Q", title="Coeficiente de Gini"),
                color=alt.Color(
                    color_field,
                    title=color_title,
                    legend=alt.Legend(columns=2),
                    scale=alt.Scale(scheme="tableau10"),
                ),
                tooltip=[
                    alt.Tooltip("Entity:N", title="País"),
                    alt.Tooltip("Region:N", title="Región") if has_region else None,
                    alt.Tooltip("Year:O", title="Año"),
                    alt.Tooltip("Gini coefficient:Q", title="Gini", format=".3f"),
                    alt.Tooltip("GDP per capita:Q", title="PIB per cápita", format=".0f"),
                    alt.Tooltip("cons_pct_gdp:Q", title="Consumo % PIB", format=".2f"),
                ],
            )
            .properties(height=400)
            .interactive()
        )

        reg_line2 = (
            alt.Chart(df_view)
            .transform_regression(
                "cons_pct_gdp",
                "Gini coefficient",
                as_=["cons_pct_gdp", "gini_pred"],
            )
            .mark_line(size=2)
            .encode(
                x="cons_pct_gdp:Q",
                y="gini_pred:Q",
                color=alt.value("deepskyblue"),
            )
        )

        st.altair_chart(scatter_cons + reg_line2, width="stretch")
    else:
        st.write("No hay datos para los filtros seleccionados.")


# Pestaña 3: Resumen y modelo    #

with tab3:
    st.subheader("Estadísticos resumidos (filtros actuales)")

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
                "Gini coefficient": "Gini medio",
                "GDP per capita": "PIB pc medio",
                "cons_pct_gdp": "Consumo % PIB medio",
            })
            .round(3)
        )
        st.dataframe(summary)
    else:
        st.write("No hay datos para los filtros seleccionados.")

    st.markdown("---")
    st.subheader("Instantánea de la regresión (filtros actuales)")

    if model is not None:
        st.write(
            "Regresión OLS del coeficiente de Gini sobre log(PIB per cápita) y "
            "gasto de consumo final (% del PIB), estimada con los países y años "
            "seleccionados actualmente."
        )

        coef_df = pd.DataFrame({
            "coeficiente": model.params,
            "error estándar": model.bse,
            "valor p": model.pvalues
        }).round(4)

        st.dataframe(coef_df)
    else:
        st.write("No hay suficientes datos con los filtros actuales para estimar la regresión.")

# ---------------------------------- #
# 8. Módulo de ayuda y explicaciones #
# ---------------------------------- #

st.markdown("---")
st.header("Guía rápida para interpretar los gráficos")

with st.expander("¿Qué es el PIB per cápita?"):
    st.markdown(
        """
- El **PIB per cápita** es el valor total de bienes y servicios que produce un país en un año,
  dividido entre su población. Se usa como indicador aproximado del ingreso medio por persona.

- En nuestros gráficos usamos el **logaritmo del PIB per cápita** para que los países muy ricos
  no aplasten visualmente al resto y la nube de puntos sea más fácil de interpretar.

Referencias:
- [Indicadores de renta per cápita – Banco Mundial (World Bank Open Data)](https://datos.bancomundial.org/indicador/NY.GDP.PCAP.KD)
- [Renta per cápita – Naciones Unidas / estadísticas de desarrollo humano](https://hdr.undp.org/)
        """
    )

with st.expander("¿Qué es el coeficiente de Gini?"):
    st.markdown(
        """
- El **coeficiente de Gini** mide qué tan desigual es la distribución del ingreso (u otra variable) en una sociedad.
- Toma valores entre **0 y 1**: 0 = igualdad total; 1 = desigualdad máxima.
- En la práctica, la mayoría de países se sitúa entre aproximadamente **0,25 y 0,65**: valores más altos implican más desigualdad.

Referencias:
- [Coeficiente de Gini – Enciclopedia académica (por ejemplo, Wikipedia en español)](https://es.wikipedia.org/wiki/Coeficiente_de_Gini)
- [Medición de la desigualdad económica y Gini – CORE Econ (The Economy, sección 5.12)](https://books.core-econ.org/the-economy/microeconomics/es/05-the-rules-of-the-game-12-measuring-economic-inequality.html)
        """
    )

with st.expander("¿Cómo leer el gráfico PIB vs Gini?"):
    st.markdown(
        """
- Cada punto representa un **país en un año concreto**:
  - Eje horizontal: PIB per cápita (en logaritmos).
  - Eje vertical: desigualdad medida por el coeficiente de Gini.

- Si los puntos de los países con mayor PIB per cápita tienden a situarse con valores de Gini más bajos,
  se suele interpretar que **mayor nivel de ingreso medio se asocia con menor desigualdad**, aunque siempre con excepciones.

Referencia:
- [Materiales de CORE Econ sobre desarrollo desigualdad y pobreza](https://books.core-econ.org/the-economy/microeconomics/es/05-the-rules-of-the-game-12-measuring-economic-inequality.html)
        """
    )

with st.expander("¿Qué significa la línea de regresión?"):
    st.markdown(
        """
- La línea azul es una **línea de regresión**: resume la relación promedio entre las dos variables del gráfico
  (por ejemplo, PIB per cápita y Gini) mediante un modelo estadístico lineal.

- No implica que el PIB o el consumo *causen* directamente la desigualdad; simplemente muestra una **tendencia media**
  en los datos. Si la línea desciende al movernos hacia la derecha, indica que, en promedio,
  a mayor PIB per cápita, menor Gini.

- El estadístico **R²** que mostramos arriba indica qué parte de la variación del Gini se explica por las variables del modelo:
  valores cercanos a 1 implican un ajuste fuerte, valores cercanos a 0, un ajuste débil.

Referencias:
- [Diagramas de dispersión y regresión – OpenStax, *Introducción a la estadística*, sección 12.2](https://openstax.org/books/introducci%C3%B3n-estad%C3%ADstica/pages/12-2-diagramas-de-dispersion)
        """
    )

with st.expander("¿Cómo interpretar Consumo vs Gini?"):
    st.markdown(
        """
- En el gráfico de **Consumo vs Gini**, el eje horizontal muestra el **gasto de consumo final como porcentaje del PIB**,
  es decir, qué parte de la producción total se destina al consumo de los hogares y del sector público.

- Al compararlo con el Gini, observamos si los países donde el consumo tiene más peso en la economía
  tienden a ser **más o menos desiguales**.

- La línea de regresión resume si, en promedio, un mayor peso del consumo se asocia con mayor o menor desigualdad.

Referencias:
- [Consumo final y cuentas nacionales – Naciones Unidas / Sistema de Cuentas Nacionales](https://unstats.un.org/unsd/nationalaccount/)
- [Indicadores de gasto de consumo final – Banco Mundial](https://datos.bancomundial.org/indicador/NE.CON.TOTL.ZS)
        """
    )
