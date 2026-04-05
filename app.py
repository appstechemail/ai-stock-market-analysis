import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from src.predict import predict_today

# This version includes:

#  Equity Curve
#  PnL Curve
#  Drawdown Chart
#  Live Sharpe Ratio
#  Long–Short Portfolio (V6)
#  Clean formatting (2 decimals, no trailing zeros)
#  Dates without time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📈 AI Trading Dashboard (V6 Pro)")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    final_df = joblib.load("artifacts/final_df.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    models = joblib.load("artifacts/models.pkl")
    FEATURES = joblib.load("artifacts/features.pkl")
    return final_df, scaler, models, FEATURES

final_df, scaler, models, FEATURES = load_data()

# =========================
# GET LATEST DATA
# =========================
latest_data = final_df.groupby("Company").tail(1)

signals, portfolio = predict_today(
    latest_data,
    scaler,
    models,
    None,
    FEATURES
)

# =========================
# SAFETY FIX
# =========================
required_cols = [
    "Target", "StopLoss", "Expected_Return_%", "RR_Ratio",
    "Signal_Date", "Action_Date", "Exit_Date", "Side"
]

for col in required_cols:
    if col not in signals.columns:
        signals[col] = np.nan
    if col not in portfolio.columns:
        portfolio[col] = np.nan

# =========================
# FORMAT FUNCTION
# =========================
def format_df(df):
    df = df.copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(
            lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if pd.notnull(x) else x
        )

    for col in ["Signal_Date", "Action_Date", "Exit_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    return df

# =========================
# COLOR FUNCTIONS
# =========================
def color_signal(val):
    if val == "STRONG BUY":
        return "background-color: #2E7D32; color: white"
    elif val == "BUY":
        return "background-color: #66BB6A; color: white"
    elif val == "SELL":
        return "background-color: #E53935; color: white"
    elif val == "STRONG SELL":
        return "background-color: #B71C1C; color: white"
    return ""

def color_side(val):
    if val == "LONG":
        return "color: green"
    elif val == "SHORT":
        return "color: red"
    return ""

# =========================
# 📈 PERFORMANCE ENGINE
# =========================
def compute_performance(df):

    df = df.copy()
    df = df.sort_values(["Company", "Date"])

    df["Return"] = df.groupby("Company")["Close"].pct_change()

    # simple signal proxy
    df["Signal_Num"] = np.where(df["Target"] > df["Close"], 1,
                        np.where(df["Target"] < df["Close"], -1, 0))

    df["Strategy_Return"] = df["Signal_Num"] * df["Return"]

    daily_returns = df.groupby("Date")["Strategy_Return"].mean().fillna(0)

    equity = (1 + daily_returns).cumprod()

    pnl = equity.diff().fillna(0)

    # drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max

    # sharpe
    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() != 0 else 0
    )

    result = pd.DataFrame({
        "Date": equity.index,
        "Equity": equity.values,
        "PnL": pnl.values,
        "Drawdown": drawdown.values
    })

    result["Date"] = pd.to_datetime(result["Date"]).dt.date

    return result, sharpe

performance_df, sharpe_ratio = compute_performance(final_df)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔍 Filters")

signal_filter = st.sidebar.selectbox(
    "Signal Type",
    ["ALL", "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
)

prob_threshold = st.sidebar.slider("Min Probability", 0.0, 1.0, 0.0)

filtered_signals = signals.copy()

if signal_filter != "ALL":
    filtered_signals = filtered_signals[
        filtered_signals["Signal"] == signal_filter
    ]

filtered_signals = filtered_signals[
    filtered_signals["Probability"] >= prob_threshold
]

# =========================
# REFRESH
# =========================
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# =========================
# TOP METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Stocks", len(signals))
col2.metric("Long Trades", len(portfolio[portfolio["Side"] == "LONG"]))
col3.metric("Short Trades", len(portfolio[portfolio["Side"] == "SHORT"]))
col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

# =========================
# SIGNAL TABLE
# =========================
st.subheader("📊 Trading Signals")

st.dataframe(
    format_df(filtered_signals)
    .style.applymap(color_signal, subset=["Signal"]),
    use_container_width=True
)

# =========================
# PORTFOLIO
# =========================
st.subheader("💰 Portfolio Allocation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 LONG")
    long_port = portfolio[portfolio["Side"] == "LONG"]

    if len(long_port) > 0:
        st.dataframe(
            format_df(long_port)
            .style.applymap(color_side, subset=["Side"]),
            use_container_width=True
        )
    else:
        st.info("No LONG positions")

with col2:
    st.subheader("🔴 SHORT")
    short_port = portfolio[portfolio["Side"] == "SHORT"]

    if len(short_port) > 0:
        st.dataframe(
            format_df(short_port)
            .style.applymap(color_side, subset=["Side"]),
            use_container_width=True
        )
    else:
        st.info("No SHORT positions")

# =========================
# 📈 PERFORMANCE CHARTS
# =========================
st.subheader("📈 Portfolio Performance")

fig = go.Figure()

# Equity
fig.add_trace(go.Scatter(
    x=performance_df["Date"],
    y=performance_df["Equity"],
    name="Equity",
    mode="lines"
))

# PnL
fig.add_trace(go.Scatter(
    x=performance_df["Date"],
    y=performance_df["PnL"],
    name="PnL",
    yaxis="y2"
))

fig.update_layout(
    template="plotly_dark",
    yaxis=dict(title="Equity"),
    yaxis2=dict(title="PnL", overlaying="y", side="right"),
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# 📉 DRAWDOWN CHART
# =========================
st.subheader("📉 Drawdown")

fig_dd = go.Figure()

fig_dd.add_trace(go.Scatter(
    x=performance_df["Date"],
    y=performance_df["Drawdown"],
    name="Drawdown",
    fill='tozeroy'
))

fig_dd.update_layout(
    template="plotly_dark",
    yaxis_title="Drawdown",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig_dd, use_container_width=True)

# =========================
# 📈 STOCK CHART
# =========================
st.subheader("📈 Stock Chart")

company_list = final_df["Company"].unique()
selected_company = st.selectbox("Select Company", company_list)

def plot_chart(df, company):
    data = df[df["Company"] == company].copy()
    data = data.sort_values("Date")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    return fig

st.plotly_chart(plot_chart(final_df, selected_company), use_container_width=True)
