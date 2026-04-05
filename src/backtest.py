import pandas as pd
import numpy as np

from itertools import product



def run_backtest(proba, X_test, final_df, mlp_proba=None, span=4):

    test_results = X_test.copy()
    test_results["Company"] = final_df.loc[test_results.index, "Company"]

    # =========================
    # 1. MERGE REQUIRED DATA
    # =========================
    test_results["Return"] = final_df.loc[test_results.index, "Return"]
    test_results["Close"] = final_df.loc[test_results.index, "Close"]
    test_results["MA50"] = final_df.loc[test_results.index, "MA50"]
    test_results["Volatility"] = final_df.loc[test_results.index, "Volatility"]

    # =========================
    # 2. TREND FILTER
    # =========================
    # Trend
    trend = (test_results["Close"] > test_results["MA50"]).astype(int)

    # =============================================
    # 3. REGIME SWITCH (only if mlp_proba is provided)
    # =============================================
    if mlp_proba is not None:
        trend_strength = (test_results["Close"] - test_results["MA50"]) / test_results["MA50"]

        # Regime sensitivity
        # trend_strength > 0.01   # aggressive
        # trend_strength > 0.02   # balanced (recommended)
        # trend_strength > 0.03   # conservative
        regime = trend_strength > 0.02


        trend_strength = (test_results["Close"] - test_results["MA50"]) / test_results["MA50"]

        # Convert to smooth weight (0 → 1)
        regime_weight = np.clip(trend_strength * 20, 0, 1)

        # Blend instead of switching
        proba = (
            regime_weight * mlp_proba +
            (1 - regime_weight) * proba
        )



    # ML strength scaling
    # (proba - 0.5) * 2.5   # safer
    # (proba - 0.5) * 3     # current (good)
    # (proba - 0.5) * 4     # aggressive
    # ML strength
    ml_strength = np.clip((proba - 0.5) * 5, 0, 1) ** 1.5

    # =========================
    # 4. VOLATILITY ADJUSTMENT
    # =========================

    # Volatility adjustment
    vol_adj = 1 / (1 + test_results["Volatility"])

    # =========================
    # 5. RANK BASED FILTER (TOP STOCKS ONLY)
    # =========================
    rank = pd.Series(proba, index=test_results.index).rank(pct=True)

    # Only top 20% trades
    signal_filter = (rank > 0.75).astype(int)


    # =========================
    # BASE ML POSITION
    # =========================
    base_position = trend * (ml_strength ** 1.5) * vol_adj

    # =========================
    # FINAL POSITION
    # =========================
    test_results["Position"] = base_position * signal_filter

    # =========================
    # SMOOTHING
    # =========================
    test_results["Position"] = test_results["Position"].ewm(span=4
                                                            ).mean()

    # =========================
    # RISK CONTROL
    # =========================
    test_results["Position"] = np.clip(test_results["Position"], 0, 1).fillna(0)


    # =========================
    # 6. TRANSACTION COST
    # =========================

    # Costs
    transaction_cost = 0.001
    test_results["Trade"] = test_results["Position"].diff().abs()

    # =========================
    # 7. STRATEGY RETURNS
    # =========================

    test_results["Strategy"] = (
        test_results["Position"].shift(1) * test_results["Return"]
        - test_results["Trade"] * transaction_cost
    )

    # =========================
    # 8. CUMULATIVE RETURNS
    # =========================
    test_results["Cumulative_Strategy"] = test_results["Strategy"].cumsum()
    test_results["Cumulative_Market"] = test_results["Return"].cumsum()

    # =========================
    # 9. DATE INDEX
    # =========================

    test_results["Date"] = final_df.loc[test_results.index, "Date"]
    test_results["Date"] = pd.to_datetime(test_results["Date"])


    test_results["Date"] = pd.to_datetime(final_df.loc[test_results.index, "Date"])

    return test_results  # KEEP INTEGER INDEX



# ================================================
# ########## OPTIMIZER WEIGHTS ##################
# =================================================

import numpy as np
import pandas as pd
from itertools import product


def optimize_weights(
    X_test,
    final_df,
    xgb_proba,
    lr_proba,
    lgb_proba,
    cat_proba,
    mlp_proba
):
    """
    Optimize ensemble weights using Sharpe ratio
    Models used:
    XGB, LR, LGB, CAT, MLP
    """

    print("\n🔍 Optimizing ensemble weights...")

    # Convert to pandas Series (important for index alignment)
    xgb_proba = pd.Series(xgb_proba, index=X_test.index)
    lr_proba  = pd.Series(lr_proba,  index=X_test.index)
    lgb_proba = pd.Series(lgb_proba, index=X_test.index)
    cat_proba = pd.Series(cat_proba, index=X_test.index)
    mlp_proba = pd.Series(mlp_proba, index=X_test.index)

    best_sharpe = -np.inf
    best_weights = None
    best_proba = None

    # Smaller grid = faster + stable
    weight_range = [0.20, 0.15, 0.25, 0.30, 0.10]

    # =========================
    # GRID SEARCH
    # =========================
    for w in product(weight_range, repeat=5):

        weights = np.array(w)

        # Normalize weights
        weights = weights / weights.sum()

        # Optional: penalize extreme weights
        weights = weights ** 1.5
        weights = weights / weights.sum()

        # =========================
        # ENSEMBLE
        # =========================
        # ensemble_proba = (
        #     weights[0] * xgb_proba +
        #     weights[1] * lr_proba +
        #     weights[2] * lgb_proba +
        #     weights[3] * cat_proba +
        #     weights[4] * mlp_proba
        # )

        ensemble_proba = (
            0.30 * xgb_proba["xgb"] +
            0.25 * lr_proba["lr"] +
            0.20 * lgb_proba["lgb"] +
            0.15 * cat_proba["cat"] +
            0.10 * mlp_proba["mlp"]
        )


        # =========================
        # BACKTEST
        # =========================
        results = run_backtest(
            ensemble_proba,
            X_test,
            final_df,
            mlp_proba=mlp_proba
        )

        returns = results["Strategy"].dropna()

        # Skip bad cases
        if len(returns) < 30:
            continue

        if returns.std() == 0:
            continue

        # =========================
        # SHARPE
        # =========================
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

        # =========================
        # SAVE BEST
        # =========================
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
            best_proba = ensemble_proba

    # =========================
    # FALLBACK
    # =========================
    if best_weights is None:
        print("⚠️ No optimal weights found → using default")

        best_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

        best_proba = (
            best_weights[0] * xgb_proba +
            best_weights[1] * lr_proba +
            best_weights[2] * lgb_proba +
            best_weights[3] * cat_proba +
            best_weights[4] * mlp_proba
        )

    print("\n✅ Best Weights Found:")
    print(f"XGB: {best_weights[0]:.2f}, LR: {best_weights[1]:.2f}, "
          f"LGB: {best_weights[2]:.2f}, CAT: {best_weights[3]:.2f}, "
          f"MLP: {best_weights[4]:.2f}")

    print("📈 Best Sharpe:", round(best_sharpe, 3))

    return best_proba, best_weights
