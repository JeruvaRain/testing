{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPoYMa9RYU9+Z1gZJZyQt4B",
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
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 383
        },
        "id": "hC1gLFwdg7hl",
        "outputId": "c4b0e211-bf40-4108-c653-c116ca8466b6"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'streamlit'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_16326/1499464890.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mstreamlit\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mstatsmodels\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mapi\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0msm\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmatplotlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpyplot\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
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
        "# Fit the same regression model as before\n",
        "reg_data = df[[\"Gini coefficient\", \"log_gdp_pc\", \"cons_pct_gdp\"]].dropna()\n",
        "X = reg_data[[\"log_gdp_pc\", \"cons_pct_gdp\"]]\n",
        "X = sm.add_constant(X)\n",
        "y = reg_data[\"Gini coefficient\"]\n",
        "model = sm.OLS(y, X).fit()\n",
        "\n",
        "st.title(\"Inequality, Growth and Consumption\")\n",
        "\n",
        "st.write(\"This app uses a simple regression model to explore how income \"\n",
        "         \"levels and final consumption expenditure (% of GDP) relate to \"\n",
        "         \"the Gini coefficient for a small set of countries.\")\n",
        "\n",
        "# Sidebar controls\n",
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
        "# Predict Gini for chosen values\n",
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
        "# Plot: observed vs predicted point\n",
        "st.subheader(\"Observed data and your chosen point\")\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(6, 4))\n",
        "\n",
        "# Scatter of actual data (size small, alpha for readability)\n",
        "ax.scatter(df[\"log_gdp_pc\"], df[\"Gini coefficient\"], alpha=0.4, label=\"Observed\")\n",
        "\n",
        "# Highlight the chosen (log_gdp, predicted_gini)\n",
        "ax.scatter([log_gdp], [predicted_gini], color=\"red\", label=\"Your choice\")\n",
        "\n",
        "ax.set_xlabel(\"log(GDP per capita)\")\n",
        "ax.set_ylabel(\"Gini coefficient\")\n",
        "ax.legend()\n",
        "\n",
        "st.pyplot(fig)"
      ]
    }
  ]
}
