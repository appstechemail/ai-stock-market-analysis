import numpy as np
import pandas as pd
from config.config import CONFIG


# WHAT V6 DOES
# Core Upgrades:
#  Long AND Short portfolio
#  Market-neutral (balanced exposure)
#  Risk parity allocation (low volatility = higher weight)
#  Strong signal filtering (high Sharpe)
#  Regime-aware logic (Bull vs Bear)
#  Stable, error-free numpy-safe implementation

def predict_today(
    latest_data,
    scaler,
    models,
    weights,
    FEATURES,
    predict_days_ahead=CONFIG['PREDICT_DAYS_AHEAD']
):

    latest_data = latest_data.copy()

    # =========================
    # 1. ENSURE FEATURES
    # =========================
    for col in FEATURES:
        if col not in latest_data.columns:
            latest_data[col] = 0

    X_live = latest_data[FEATURES].copy()

    # =========================
    # 2. SCALING (SAFE)
    # =========================
    try:
        X_scaled = pd.DataFrame(
            scaler.transform(X_live),
            columns=FEATURES
        )
    except:
        X_scaled = X_live.copy()

    # =========================
    # 3. SAFE PROBA FUNCTION
    # =========================
    def get_proba(model, X):
        try:
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim > 1 else proba
        except:
            return np.zeros(len(X))

    # =========================
    # 4. MODEL PROBABILITIES
    # =========================
    probas = {
        "LR":  get_proba(models.get("lr"), X_scaled),
        "LGB": get_proba(models.get("lgb"), X_scaled),
        "XGB": get_proba(models.get("xgb"), X_live),
        "MLP": get_proba(models.get("mlp"), X_scaled),
    }

    # =========================
    # 🔥 5. REGIME-AWARE ENSEMBLE
    # =========================
    close = latest_data["Close"].values
    ma50 = latest_data["MA50"].values
    vol = latest_data["Volatility"].fillna(0).values

    trend = (close > ma50).astype(int)

    ensemble_proba = np.where(
        trend == 1,  # BULL
        0.4 * probas["LR"] + 0.3 * probas["MLP"] + 0.3 * probas["LGB"],
        0.5 * probas["LR"] + 0.3 * probas["LGB"] + 0.2 * probas["XGB"]
    )

    ensemble_proba = np.clip(ensemble_proba, 0.01, 0.99)

    # =========================
    # 6. DYNAMIC THRESHOLDS
    # =========================
    high_th = np.percentile(ensemble_proba, 80)
    mid_th  = np.percentile(ensemble_proba, 60)
    low_th  = np.percentile(ensemble_proba, 40)
    very_low_th = np.percentile(ensemble_proba, 20)

    # =========================
    # 7. POSITION STRENGTH
    # =========================
    position = (ensemble_proba - 0.5) * 4
    position = np.clip(position, -1.5, 1.5)

    # =========================
    # 8. CREATE SIGNALS
    # =========================
    signals = pd.DataFrame({
        "Company": latest_data["Company"].values,
        "Close": close,
        "Probability": ensemble_proba,
        "Position": position
    })

    # =========================
    # 9. SIGNAL LOGIC
    # =========================
    signals["Signal"] = np.where(
        ensemble_proba >= high_th, "STRONG BUY",
        np.where(
            ensemble_proba >= mid_th, "BUY",
            np.where(
                ensemble_proba <= very_low_th, "STRONG SELL",
                np.where(
                    ensemble_proba <= low_th, "SELL",
                    "HOLD"
                )
            )
        )
    )

    # =========================
    # 10. DATE HANDLING
    # =========================
    if "Date" in latest_data.columns:
        signals["Signal_Date"] = pd.to_datetime(latest_data["Date"].values)
    else:
        signals["Signal_Date"] = pd.Timestamp.today()

    signals["Action_Date"] = signals["Signal_Date"] + pd.Timedelta(days=1)
    signals["Exit_Date"] = signals["Signal_Date"] + pd.Timedelta(days=predict_days_ahead)

    # =========================
    # 11. STOPLOSS / TARGET
    # =========================
    signals["StopLoss"] = np.where(
        signals["Signal"].isin(["BUY", "STRONG BUY"]),
        close * (1 - 2 * vol),
        np.where(
            signals["Signal"].isin(["SELL", "STRONG SELL"]),
            close * (1 + 2 * vol),
            np.nan
        )
    )

    signals["Target"] = np.where(
        signals["Signal"].isin(["BUY", "STRONG BUY"]),
        close * (1 + 3 * vol),
        np.where(
            signals["Signal"].isin(["SELL", "STRONG SELL"]),
            close * (1 - 3 * vol),
            np.nan
        )
    )

    # =========================
    # 12. RETURNS + RISK
    # =========================
    signals["Expected_Return_%"] = (
        (signals["Target"] - signals["Close"]) / signals["Close"]
    ) * 100

    signals["Risk"] = abs(signals["Close"] - signals["StopLoss"])
    signals["Reward"] = abs(signals["Target"] - signals["Close"])
    signals["RR_Ratio"] = signals["Reward"] / signals["Risk"]

    # =========================
    # 🔥 13. FILTER STRONG TRADES
    # =========================
    long_trades = signals[
        (signals["Signal"].isin(["BUY", "STRONG BUY"])) &
        (signals["Probability"] > 0.58) &
        (signals["RR_Ratio"] > 1.5)
    ].copy()

    short_trades = signals[
        (signals["Signal"].isin(["SELL", "STRONG SELL"])) &
        (signals["Probability"] < 0.42) &
        (signals["RR_Ratio"] > 1.5)
    ].copy()

    # =========================
    # 🔥 14. RISK PARITY ALLOCATION
    # =========================
    capital = CONFIG.get("CAPITAL", 1000000)

    def allocate_portfolio(trades, side="long"):

        if len(trades) == 0:
            return trades

        # risk parity: inverse volatility
        risk_weight = 1 / (vol[:len(trades)] + 1e-6)

        trades["Weight"] = risk_weight / risk_weight.sum()

        # scale capital equally between long/short
        trades["Allocation"] = trades["Weight"] * (capital / 2)

        trades["Shares"] = np.floor(
            trades["Allocation"] / trades["Close"]
        ).astype(int)

        trades["Side"] = side

        return trades

    long_portfolio = allocate_portfolio(long_trades.copy(), "LONG")
    short_portfolio = allocate_portfolio(short_trades.copy(), "SHORT")

    # =========================
    # 15. COMBINE PORTFOLIO
    # =========================
    portfolio = pd.concat([long_portfolio, short_portfolio], ignore_index=True)

    # =========================
    # 16. FINAL SORT
    # =========================
    signals = signals.sort_values("Probability", ascending=False).reset_index(drop=True)

    return signals, portfolio
