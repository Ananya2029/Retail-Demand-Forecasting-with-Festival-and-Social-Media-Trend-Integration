import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide"
)

st.title(
    "📈 Retail Demand Forecasting with Festival and Social Media Trends"
)

@st.cache_data
def load_data():

    df = pd.read_csv(
        "../data/processed/final_featured_sales.csv"
    )

    return df

df = load_data()

# ====================================
# SIDEBAR CONTROLS
# ====================================

st.sidebar.title("Controls")

# Store Selection
store_id = st.sidebar.selectbox(
    "Select Store",
    sorted(df["Store"].unique()),
    format_func=lambda x: f"Store {x}"
)

# Forecast Horizon
forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [7, 30, 90]
)

# Festival Filter
festival_options = ["All"] + sorted(
    df["Festival_Name"].dropna().unique()
)

selected_festival = st.sidebar.selectbox(
    "Festival Filter",
    festival_options
)

# Trend Score Filter
trend_range = st.sidebar.slider(
    "Trend Score Range",
    min_value=int(df["Trend_Score"].min()),
    max_value=int(df["Trend_Score"].max()),
    value=(
        int(df["Trend_Score"].min()),
        int(df["Trend_Score"].max())
    )
)

# Sentiment Score Filter
sentiment_range = st.sidebar.slider(
    "Sentiment Score Range",
    min_value=-1.0,
    max_value=1.0,
    value=(-1.0, 1.0),
    step=0.1
)

# SHAP Toggle
show_shap = st.sidebar.checkbox(
    "Show Explainability",
    value=True
)

# ====================================
# APPLY FILTERS
# ====================================

filtered_df = df.copy()

# Store Filter
filtered_df = filtered_df[
    filtered_df["Store"] == store_id
]

# Festival Filter
if selected_festival != "All":

    filtered_df = filtered_df[
        filtered_df["Festival_Name"]
        ==
        selected_festival
    ]

# Trend Score Filter
filtered_df = filtered_df[

    (filtered_df["Trend_Score"] >= trend_range[0])

    &

    (filtered_df["Trend_Score"] <= trend_range[1])

]

# Sentiment Filter
filtered_df = filtered_df[

    (filtered_df["Sentiment_Score"] >= sentiment_range[0])

    &

    (filtered_df["Sentiment_Score"] <= sentiment_range[1])

]

st.header("Dataset Overview")

st.write(
    "Dataset Shape:",
    df.shape
)

st.dataframe(
    df.head()
)

st.header("Sales Trend")

store_df = filtered_df

fig, ax = plt.subplots(
    figsize=(12,5)
)

ax.plot(
    pd.to_datetime(
        store_df["Date"]
    ),
    store_df["Weekly_Sales"]
)

ax.set_title(
    f"Store {store_id} Sales Trend"
)

st.pyplot(fig)

st.header("Festival Analytics")

festival_sales = (

    filtered_df

    .groupby(
        "Festival_Name"
    )["Weekly_Sales"]

    .mean()

    .sort_values(
        ascending=False
    )

)

fig, ax = plt.subplots(
    figsize=(10,5)
)

festival_sales.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Average Sales by Festival"
)

st.pyplot(fig)

st.header(
    "Social Media Analytics"
)

fig, ax = plt.subplots(
    figsize=(10,5)
)

sns.scatterplot(

    data=filtered_df,

    x="Trend_Score",

    y="Weekly_Sales",

    ax=ax

)

ax.set_title(
    "Trend Score vs Sales"
)

st.pyplot(fig)

st.header(
    "Feature Correlation"
)

numeric_df = filtered_df.select_dtypes(
    include=np.number
)

fig, ax = plt.subplots(
    figsize=(12,8)
)

sns.heatmap(

    numeric_df.corr(),

    cmap="coolwarm",

    ax=ax

)

st.pyplot(fig)

forecast_df = pd.read_csv(
    "../outputs/forecast_results.csv"
)

st.header(
    "Demand Forecast"
)

forecast_view = forecast_df.head(
    forecast_horizon
)

st.dataframe(
    forecast_view
)

fig, ax = plt.subplots(
    figsize=(12,5)
)

ax.plot(

    pd.to_datetime(
        forecast_view["Date"]
    ),

    forecast_view[
        "Predicted_Sales"
    ]

)

ax.set_title(
    f"Next {forecast_horizon} Forecast Periods"
)

st.pyplot(fig)

st.header(
    "Demand Peaks"
)

peak_threshold = (

    forecast_df[
        "Predicted_Sales"
    ]

    .quantile(0.90)

)

peaks = forecast_df[

    forecast_df[
        "Predicted_Sales"
    ]

    >= peak_threshold

]

st.dataframe(
    peaks
)

if show_shap:

    shap_df = pd.read_csv(
        "../outputs/shap_feature_importance.csv"
    )

    st.header(
        "Model Explainability"
    )

    top20 = shap_df.head(20)

    fig, ax = plt.subplots(
        figsize=(10,8)
    )

    ax.barh(
        top20["Feature"],
        top20["Importance"]
    )

    ax.invert_yaxis()

    st.pyplot(fig)

st.header(
    "Download Forecast"
)

csv = forecast_df.to_csv(
    index=False
)

st.download_button(

    label="Download Forecast CSV",

    data=csv,

    file_name="forecast.csv",

    mime="text/csv"

)

