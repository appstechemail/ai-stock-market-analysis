import numpy as np
import pandas as pd


def evaluate(results):
    # Strategy return
    strategy_return = results["Cumulative_Strategy"].iloc[-1]

    # Market return
    market_return = results["Cumulative_Market"].iloc[-1]

    # Sharpe ratio
    returns = results["Strategy"].dropna()

    downside = returns[returns < 0]

    if downside.std() == 0:
        sortino = 0
    else:
        sortino = returns.mean() / downside.std() * np.sqrt(252)

    # Max Drawdown
    cum = results["Cumulative_Strategy"]
    drawdown = cum - cum.cummax()
    max_drawdown = drawdown.min()

    # print("Strategy Return:", strategy_return)
    # print("Market Return:", market_return)
    # print("Sharpe Ratio:", sharpe_ratio)
    # print("Max Drawdown:", max_drawdown)

    return strategy_return, market_return, sortino, max_drawdown



# =========================
# 0. CALL EVALUATE
# =========================

def evaluate_full(results):

    sr, mr, sortino, mdd = evaluate(results)

    return {
        "Strategy Return": sr,
        "Market Return": mr,
        "Sortino Ratio": sortino,
        "Max Drawdown": mdd
    }


# =========================
# 0. BUILD SUMMARY
# =========================
def build_summary(results_dict, accuracies_dict):

    df_summary = []

    for name, result in results_dict.items():
        sr, mr, sharpe, mdd = evaluate(result)

        df_summary.append([
            name,
            mr,
            sr,
            sharpe,
            mdd,
            accuracies_dict[name] * 100
        ])

    results_df = pd.DataFrame(df_summary, columns=[
        "Model",
        "Market Return",
        "Strategy Return",
        "Sharpe Ratio",
        "Max Drawdown",
        "% Accuracy"
    ])

    return results_df

