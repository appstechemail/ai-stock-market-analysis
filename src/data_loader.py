import yfinance as yf
import pandas as pd
from datetime import datetime
from config.config import CONFIG, START_DATE, END_DATE, stocks




# =========================

def download_price_data():

    all_data = []

    for ticker, name in stocks.items():
        print(f"Downloading {ticker} --- {name}...")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"⚠️ No data for {name}")
            continue

        # Fix MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(col).capitalize() for col in df.columns]

        df["Company"] = name
        df["Ticker"] = ticker

        df = df.reset_index()

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)
