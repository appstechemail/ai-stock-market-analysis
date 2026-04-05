# 🚀 AI Stock Market Analysis & Trading System

## 📌 Overview

End-to-end AI-powered stock prediction system with:
- Machine Learning models (XGB, LGB, LR, MLP, CAT, ET and RF)
- Ensemble strategy
- Backtesting engine
- Risk management (StopLoss, Target, RR Ratio)
- Portfolio allocation (Risk parity)
- Streamlit dashboard

---

## 🧠 Key Features

* 📊 Multi-stock analysis
* 🤖 ML Models:

  * Random Forest
  * Extra Trees
  * XGBoost
  * Neural Network (MLP)
* ⚖️ Ensemble strategy with optimized weights
* 📉 Backtesting engine
* 📈 Technical indicators:

  * RSI, MACD, Moving Averages
* 💰 Fundamental features:

  * EPS, PE Ratio, Market Cap
* 📊 Live dashboard using Streamlit

---

## 📂 END-to-end AI-powered stock prediction system Flow

End-to-end AI-powered stock prediction system with:
- Machine Learning models (XGB, LGB, LR, MLP, CAT)
- Ensemble strategy
- Backtesting engine
- Risk management (StopLoss, Target, RR Ratio)
- Portfolio allocation (Risk parity)
- Streamlit dashboard

---

## 📂 Project Structure

```
AI_Stock_Market_Analysis/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── artifacts/
│   ├── final_df.pkl
│   ├── scaler.pkl
│   ├── models.pkl
│   ├── features.pkl
│
├── src/
│   ├── predict.py
│   ├── models.py
│   ├── features.py
│   ├── backtest.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   ├── fundamentals.py
│   ├── data_loader.py
│   └── target.py
│
├── config/
│   └── config.py
│
└── train.py


```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ai-stock-market-analysis.git
cd ai-stock-market-analysis

pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Model       | Strategy Return | Sharpe Ratio |
| ----------- | --------------- | ------------ |
| Ensemble    | High            | Strong       |
| XGBoost     | Strong          | Stable       |
| Linear Reg  | High            | Aggressive   |

---

## 🧪 Data Sources

* Market data via `yfinance`
* Custom EPS dataset (`ALL_STOCKS_EPS.csv`)

---

## 📈 Strategy Logic

* Predict probability of stock movement
* Combine models using weighted ensemble
* Apply filters:

  * Trend (MA50)
  * Volatility adjustment
* Generate signals:

  * BUY / SELL / HOLD

---

## 🔮 Future Improvements

* Live trading integration
* Portfolio optimization
* Risk management module
* Deployment on cloud

---

## 👨‍💻 Author

Parmod Chaudhary

---

## ⭐ If you like this project, give it a star!

