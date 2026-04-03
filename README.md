# 🚀 AI Stock Market Analysis & Trading System

## 📌 Overview

This project is an **AI-powered stock market analysis and trading system** that uses:

* Machine Learning models
* Ensemble techniques
* Fundamental analysis (EPS, PE)
* Technical indicators

It also includes a **Streamlit dashboard** for live predictions and visualization.

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

## 📂 Project Structure

```
stock-market-analysis/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── ALL_STOCKS_EPS.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── fundamentals.py
│   ├── target.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── backtest.py
│   ├── optimizer.py
│   ├── predict.py
│   └── utils.py
│
├── artifacts/
│   ├── models.pkl
│   ├── scaler.pkl
│   ├── weights.pkl
│   └── features.pkl
│
├── app/
│   └── app.py   (FastAPI / Streamlit)
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

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
| Extra Trees | High            | Aggressive   |

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

