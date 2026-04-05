import pandas as pd
import numpy as np
import yfinance as yf
from config.config import CONFIG


def add_basic_fundamentals(df, stocks=None):
    """
    Adds time-safe fundamental features (NO DATA LEAKAGE)

    Features added:
    - EPS
    - PE
    - EPS Growth
    - PE Change
    - Value Score
    - PE Rank (cross-sectional)
    - MarketCap (static snapshot - optional)
    """

    # =========================
    # 0. COPY DATA (SAFETY)
    # =========================
    df = df.copy()

    # =========================
    # 1. LOAD EPS FILE
    # =========================
    print("📂 Loading EPS file:", CONFIG["PATH_OF_ALL_STOCKS_EPS_FILE"])

    eps_df = pd.read_csv(CONFIG["PATH_OF_ALL_STOCKS_EPS_FILE"])

    # Clean columns
    eps_df.columns = eps_df.columns.str.strip().str.upper()

    # =========================
    # 2. VALIDATE
    # =========================
    required_cols = ["COMPANY", "YEAR", "EPS"]
    for col in required_cols:
        if col not in eps_df.columns:
            raise ValueError(f"❌ Missing column '{col}' in EPS file")

    # =========================
    # 3. CLEAN EPS DATA
    # =========================
    eps_df["COMPANY"] = eps_df["COMPANY"].astype(str).str.upper().str.strip()

    # Extract valid YEAR (handles dirty formats like "2022-23")
    eps_df["YEAR"] = (
        eps_df["YEAR"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
    )

    eps_df["YEAR"] = pd.to_numeric(eps_df["YEAR"], errors="coerce")
    eps_df = eps_df.dropna(subset=["YEAR"])

    eps_df["YEAR"] = eps_df["YEAR"].astype(int)

    # =========================
    # 4. PREPARE MAIN DF
    # =========================
    df["Date"] = pd.to_datetime(df["Date"])
    df["Company"] = df["Company"].astype(str).str.upper().str.strip()

    # =========================
    # 5. CREATE FINANCIAL YEAR
    # (APR–MAR logic for India)
    # =========================
    df["YEAR"] = np.where(
        df["Date"].dt.month >= 4,
        df["Date"].dt.year + 1,
        df["Date"].dt.year
    ).astype(int)

    # =========================
    # 6. MERGE EPS
    # =========================
    df = df.merge(
        eps_df[["COMPANY", "YEAR", "EPS"]],
        left_on=["Company", "YEAR"],
        right_on=["COMPANY", "YEAR"],
        how="left"
    )

    df.drop(columns=["COMPANY"], inplace=True)

    # =========================
    # 7. HANDLE MISSING EPS
    # =========================
    df["EPS"] = df.groupby("Company")["EPS"].ffill()

    # =========================
    # 8. CALCULATE PE
    # =========================
    df["PE"] = df["Close"] / df["EPS"]

    df["PE"] = df["PE"].replace([np.inf, -np.inf], np.nan)

    # =========================
    # 9. ADVANCED FEATURES (KEY 🔥)
    # =========================

    # EPS Growth (very important)
    df["EPS_Growth"] = df.groupby("Company")["EPS"].pct_change()

    # PE Change
    df["PE_Change"] = df.groupby("Company")["PE"].pct_change()

    # Value Score (alpha signal)
    df["Value_Score"] = df["EPS_Growth"] / df["PE"]

    # =========================
    # 10. CROSS-SECTIONAL FEATURES
    # =========================

    # Rank companies by PE each day
    df["PE_Rank"] = df.groupby("Date")["PE"].rank()

    # Rank by Value Score
    df["Value_Rank"] = df.groupby("Date")["Value_Score"].rank(ascending=False)

    # =========================
    # 11. OPTIONAL: MARKET CAP (SAFE USAGE)
    # =========================
    if stocks is not None:
        print("📊 Adding MarketCap (static snapshot)")

        for ticker, name in stocks.items():
            try:
                info = yf.Ticker(ticker).info
                market_cap = info.get("marketCap")

                df.loc[df["Company"] == name.upper(), "MarketCap"] = market_cap

            except Exception:
                print(f"⚠️ Failed MarketCap for {name}")

    # =========================
    # 12. CLEAN FINAL DATA
    # =========================

    df.drop(columns=["YEAR"], inplace=True)

    # Replace bad values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Optional: fill remaining NaNs smartly
    df["EPS_Growth"] = df["EPS_Growth"].fillna(0)
    df["PE_Change"] = df["PE_Change"].fillna(0)
    df["Value_Score"] = df["Value_Score"].fillna(0)

    print("✅ Fundamentals added successfully")

    return df
