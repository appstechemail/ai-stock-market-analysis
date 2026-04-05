import numpy as np
from config.config import CONFIG

def add_technical_features(df):

    # ✅ ADD THIS LINE HERE
    df = df.sort_values(["Company", "Date"]).reset_index(drop=True)

    # Returns
    df["Return"] = df.groupby("Company")["Close"].pct_change()

    df["Log_Return"] = df.groupby("Company")["Close"].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Moving averages
    for window in CONFIG["MA_WINDOWS"]:
        df[f"MA{window}"] = df.groupby("Company")["Close"].transform(
            lambda x: x.rolling(window).mean()
        )

    # Volatility
    df["Volatility"] = df.groupby("Company")["Return"].transform(
        lambda x: x.rolling(CONFIG["VOL_WINDOW"]).std()
    )

    # Momentum
    df["Momentum"] = df.groupby("Company")["Close"].transform(
        lambda x: x - x.shift(CONFIG["MOMENTUM_WINDOW"])
    )

    # RSI
    def compute_rsi(series, window):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df["RSI"] = df.groupby("Company")["Close"].transform(
        lambda x: compute_rsi(x, CONFIG["RSI_WINDOW"])
    )

    # MACD
    def compute_macd(series):
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        return ema12 - ema26

    df["MACD"] = df.groupby("Company")["Close"].transform(compute_macd)

    # Bollinger Bands
    for window in CONFIG["MA_WINDOWS"]:
        df[f"BB_Upper_{window}"] = df.groupby("Company")["Close"].transform(
            lambda x: x.rolling(window).mean() + 2*x.rolling(window).std()
            )
        df[f"BB_Lower_{window}"] = df.groupby("Company")["Close"].transform(
            lambda x: x.rolling(window).mean() - 2*x.rolling(window).std()
            )

    # Exponential Moving Average
    for span in [10, 20, 50]:
        df[f"EMA_{span}"] = df.groupby("Company")["Close"].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
            )

    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag{lag}"] = df.groupby("Company")["Close"].shift(lag)

    # Lagged Returns
    for lag in [1, 2, 3, 5, 10]:
        df[f"Return_Lag{lag}"] = df.groupby("Company")["Return"].shift(lag)


    # Trend Feature
    df["Trend"] = (df["Close"] > df["MA50"]).astype(int)

    # Volume Change
    df["Volume_Change"] = df.groupby("Company")["Volume"].pct_change()

    # Rolling Returns
    df["Return_5"] = df.groupby("Company")["Return"].transform(lambda x: x.rolling(5).sum())
    df["Return_10"] = df.groupby("Company")["Return"].transform(lambda x: x.rolling(10).sum())

    df["Return_Rank"] = df.groupby("Date")["Return"].rank(pct=True)
    df["Momentum_Rank"] = df.groupby("Date")["Momentum"].rank(pct=True)
    df["Volatility_Rank"] = df.groupby("Date")["Volatility"].rank(pct=True)

    return df
