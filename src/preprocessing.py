import numpy as np


# =========================
# 1. CLEANING FUNCTION
# =========================

def clean_data(df):

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna()

    return df


# ===================================================
# 1. MARKET REGIME DETECTION (Bull / Bear / Sideways)
# ===================================================

def detect_market_regime(df):
    df = df.copy()

    # Trend (based on NSE index)
    df["Market_Trend"] = np.where(
        df["NSE_Close"] > df["NSE_Close"].rolling(50).mean(), 1, 0
    )

    # Volatility
    df["Market_Volatility"] = df["NSE_Return"].rolling(10).std()

    # Regime classification
    df["Regime"] = np.where(
        (df["Market_Trend"] == 1) & (df["Market_Volatility"] < df["Market_Volatility"].quantile(0.6)),
        "BULL",
        np.where(
            (df["Market_Trend"] == 0) & (df["Market_Volatility"] > df["Market_Volatility"].quantile(0.6)),
            "BEAR",
            "SIDEWAYS"
        )
    )

    return df
