import numpy as np
from config.config import CONFIG

def add_target(df):

    days = CONFIG["PREDICT_DAYS_AHEAD"]


    # Today I buy stock →
    # What % profit/loss after N days? i.e.
    # Future Return = (Price after N days / Today's Price) - 1
    future_return = df.groupby("Company")["Close"].shift(-days) / df["Close"] - 1

    threshold = 1.5 * df["Volatility"]
    # 1 → BUY (price will go up > +2%)
    # 0 → SELL (price will go down < -2%)
    # NaN → ignore (small moves)
    # df["Target"] = np.where(
    # future_return > threshold, 1,
    # np.where(future_return < -threshold, 0, np.nan)
    # )

    df["Target"] = np.where(
    future_return > 0.02, 1,
    np.where(future_return < -0.02, 0, np.nan)
    )

    # It removes (drops) all rows from the DataFrame df where the column "Target" has a missing value (NaN). i.e.
    # - dropna() = remove missing values
    # - subset=["Target"] = check missing values only in the Target column
    # - df = = save the cleaned DataFrame back into df
    df = df.dropna(subset=["Target"])

    return df


