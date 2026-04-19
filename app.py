{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNT5LTsSHIL1vnWtnK30+Ae",
      "include_colab_link": True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JeruvaRain/testing/blob/master/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import statsmodels.api as sm\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Load data\n",
        "df = pd.read_csv(\"ineq_data.csv\")\n",
        "\n",
        "# Fit regression\n",
        "reg_data = df[[\"Gini coefficient\", \"log_gdp_pc\", \"cons_pct_gdp\"]].dropna()\n",
        "X = reg_data[[\"log_gdp_pc\", \"cons_pct_gdp\"]]\n",
        "X = sm.add_constant(X)\n",
        "y = reg_data[\"Gini coefficient\"]\n",
        "model = sm.OLS(y, X).fit()\n",
        "\n",
        "st.title(\"Inequality, Growth and Consumption\")\n",
        "\n",
        "st.write(\n",
        "    \"This app uses a simple regression model to explore how income \"\n",
        "    \"levels and final consumption expenditure (% of GDP) relate to \"\n",
        "    \"the Gini coefficient for a small set of countries.\"\n",
        ")\n",
        "\n",
        "st.sidebar.header(\"Choose parameters\")\n",
        "\n",
        "log_gdp = st.sidebar.slider(\n",
        "    \"log(GDP per capita)\",\n",
        "    float(df[\"log_gdp_pc\"].min()),\n",
        "    float(df[\"log_gdp_pc\"].max()),\n",
        "    float(df[\"log_gdp_pc\"].mean())\n",
        ")\n",
        "\n",
        "cons = st.sidebar.slider(\n",
        "    \"Final consumption expenditure (% of GDP)\",\n",
        "    float(df[\"cons_pct_gdp\"].min()),\n",
        "    float(df[\"cons_pct_gdp\"].max()),\n",
        "    float(df[\"cons_pct_gdp\"].mean())\n",
        ")\n",
        "\n",
        "X_new = pd.DataFrame({\n",
        "    \"const\": [1.0],\n",
        "    \"log_gdp_pc\": [log_gdp],\n",
        "    \"cons_pct_gdp\": [cons]\n",
        "})\n",
        "predicted_gini = model.predict(X_new)[0]\n",
        "\n",
        "st.subheader(\"Predicted inequality for chosen values\")\n",
        "st.write(f\"**Predicted Gini coefficient:** {predicted_gini:.3f}\")\n",
        "\n",
        "st.subheader(\"Observed data and your chosen point\")\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(6, 4))\n",
        "ax.scatter(df[\"log_gdp_pc\"], df[\"Gini coefficient\"], alpha=0.4, label=\"Observed\")\n",
        "ax.scatter([log_gdp], [predicted_gini], color=\"red\", label=\"Your choice\")\n",
        "ax.set_xlabel(\"log(GDP per capita)\")\n",
        "ax.set_ylabel(\"Gini coefficient\")\n",
        "ax.legend()\n",
        "st.pyplot(fig)"
      ],
      "metadata": {
        "id": "EzxlDZB6nlgA"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
