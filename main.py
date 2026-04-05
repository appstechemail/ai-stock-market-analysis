# ================================
# IMPORTS
# ================================
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.data_loader import download_price_data
from src.features import add_technical_features
from src.target import add_target
from src.fundamentals import add_basic_fundamentals
from src.preprocessing import clean_data, detect_market_regime

from src.backtest import run_backtest
from src.predict import predict_today
from src.models import train_models
from src.evaluation import build_summary
from train import get_model_probabilities

from config.config import CONFIG


# ================================
# WALK-FORWARD ENSEMBLE
# ================================
def walk_forward_auto_weights(X_test, probas, final_df, window=400):

    print("\n🔄 Running Walk-Forward Auto Weights...")

    probas = {k: pd.Series(v, index=X_test.index) for k, v in probas.items()}

    all_ensemble = []

    for start in range(0, len(X_test), window):

        end = min(start + window, len(X_test))
        idx = X_test.index[start:end]

        if len(idx) < 100:
            continue

        sharpes = {}

        # =========================
        # Evaluate each model
        # =========================
        for model_name, proba_series in probas.items():

            res = run_backtest(
                proba_series.loc[idx],
                X_test.loc[idx],
                final_df
            )

            returns = res["Strategy"].dropna()

            if len(returns) < 20 or returns.std() == 0:
                continue

            sharpe = returns.mean() / returns.std() * np.sqrt(252)

            if sharpe > 0:
                sharpes[model_name] = sharpe

        if len(sharpes) == 0:
            continue

        # =========================
        # Convert to weights
        # =========================
        sharpes = pd.Series(sharpes)
        weights = sharpes / sharpes.sum()

        print(f"Window {start}-{end} Weights:", weights.to_dict())

        # =========================
        # Build ensemble
        # =========================
        ensemble_vals = np.zeros(len(idx))

        for model_name, weight in weights.items():
            ensemble_vals += weight * probas[model_name].loc[idx].values

        all_ensemble.append(pd.Series(ensemble_vals, index=idx))

    # =========================
    # Merge all windows
    # =========================
    final_ensemble = pd.concat(all_ensemble).sort_index()

    # 🔥 FIX: FULL ALIGNMENT
    final_ensemble = final_ensemble.reindex(X_test.index)

    # Fill gaps
    final_ensemble = final_ensemble.ffill().bfill()

    print("✅ Walk-forward ensemble ready")

    return final_ensemble


# ================================
# MAIN PIPELINE
# ================================
def main():

    # ================================
    # 1. DATA
    # ================================
    print("\n📥 Downloading data...")
    df = download_price_data()
    df = df.sort_values(["Company", "Date"]).reset_index(drop=True)

    # ================================
    # 2. MACRO
    # ================================
    print("📊 Adding macro data...")

    macro = yf.download(
        "^NSEI",
        start=CONFIG["START_DATE"],
        end=CONFIG["END_DATE"],
        progress=False
    )

    if isinstance(macro.columns, pd.MultiIndex):
        macro.columns = [col[0] for col in macro.columns]

    macro = macro[['Close']].rename(columns={'Close': 'NSE_Close'})
    macro['NSE_Return'] = macro['NSE_Close'].pct_change()
    macro = macro.reset_index()

    df['Date'] = pd.to_datetime(df['Date'])
    macro['Date'] = pd.to_datetime(macro['Date'])

    df = pd.merge_asof(
        df.sort_values('Date'),
        macro.sort_values('Date'),
        on='Date',
        direction='backward'
    )

    df = detect_market_regime(df)

    # ================================
    # 3. FEATURES
    # ================================
    if CONFIG["USE_TECHNICAL"]:
        df = add_technical_features(df)

    if CONFIG["USE_FUNDAMENTAL"]:
        df = add_basic_fundamentals(df)

    # ================================
    # 4. TARGET + CLEAN
    # ================================
    df = add_target(df)
    final_df = clean_data(df)

    print("✅ Final dataset:", final_df.shape)

    # ================================
    # 5. FEATURES
    # ================================
    FEATURES = [
        "Return","Log_Return","MA10","MA20","MA50",
        "Volatility","Momentum","RSI","MACD","NSE_Return",
        "PE","EPS","Close_Lag1","Close_Lag2","Close_Lag3","Close_Lag5",
        "Trend","Volume_Change","Return_5","Return_10",
        "EPS_Growth","PE_Change","Value_Score",
        "PE_Rank","Value_Rank","Return_Rank",
        "Momentum_Rank","Volatility_Rank"
    ]

    TARGET = "Target"

    X = final_df[FEATURES].dropna()
    y = final_df.loc[X.index, TARGET]

    # ================================
    # 6. SPLIT
    # ================================
    split = int(len(X) * CONFIG["TRAIN_SIZE"])

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ================================
    # 7. SCALING
    # ================================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ================================
    # 8. TRAIN MODELS
    # ================================
    print("🤖 Training models...")
    models = train_models(X_train, y_train, X_train_scaled)

    # ================================
    # 9. PROBABILITIES
    # ================================
    probas = get_model_probabilities(models, X_test, X_test_scaled)
    probas = {k.lower(): v for k, v in probas.items()}

    # ================================
    # 10. WALK-FORWARD ENSEMBLE
    # ================================
    ensemble_proba = walk_forward_auto_weights(
        X_test,
        probas,
        final_df,
        window=400
    )

    # ================================
    # 🔥 FIX: ALIGN EVERYTHING
    # ================================
    common_index = y_test.index.intersection(ensemble_proba.index)

    y_test_aligned = y_test.loc[common_index]
    ensemble_proba = ensemble_proba.loc[common_index]

    # ================================
    # 11. BACKTEST
    # ================================
    ensemble_results = run_backtest(
        ensemble_proba,
        X_test.loc[common_index],
        final_df
    )

    results = {
        "ENSEMBLE": ensemble_results,
        "XGB": run_backtest(probas["xgb"], X_test, final_df),
        "LR": run_backtest(probas["lr"], X_test, final_df),
        "LGB": run_backtest(probas["lgb"], X_test, final_df),
        "CAT": run_backtest(probas["cat"], X_test, final_df),
        "MLP": run_backtest(probas["mlp"], X_test, final_df),
    }

    # ================================
    # 12. ACCURACY (FIXED)
    # ================================
    accuracies = {
        "ENSEMBLE": accuracy_score(
            y_test_aligned,
            (ensemble_proba > CONFIG["THRESHOLD"]).astype(int)
        ),
        "XGB": accuracy_score(y_test, models["xgb"].predict(X_test)),
        "LR": accuracy_score(y_test, models["lr"].predict(X_test_scaled)),
        "LGB": accuracy_score(y_test, models["lgb"].predict(X_test)),
        "CAT": accuracy_score(y_test, models["cat"].predict(X_test)),
        "MLP": accuracy_score(y_test, models["mlp"].predict(X_test_scaled)),
    }

    # ================================
    # 13. SUMMARY
    # ================================
    summary = build_summary(results, accuracies)

    print("\n📊 MODEL PERFORMANCE SUMMARY")
    print(summary)

    # ================================
    # 14. LIVE SIGNALS
    # ================================
    print("\n📡 Generating signals...")

    latest_data = final_df.groupby("Company").tail(1)

    signals, portfolio = predict_today(
        latest_data,
        scaler,
        models,
        None,   # walk-forward used instead
        FEATURES
    )

    print(signals.head())

    # ================================
    # 15. SAVE ARTIFACTS
    # ================================
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(final_df, "artifacts/final_df.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(models, "artifacts/models.pkl")
    joblib.dump(FEATURES, "artifacts/features.pkl")

    print("✅ Artifacts saved successfully")


# ================================
# ENTRY
# ================================
if __name__ == "__main__":
    main()
