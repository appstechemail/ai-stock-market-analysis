CONFIG = {
    "START_DATE": "2015-01-01",
    "END_DATE": None,
    "STOCKS": {
        "HDBFS.NS": "HDB FINANCE",
        "TBOTEK.NS": "TBO TEK",
        "NSDL.BO": 'NSDL',
        "MAPMYINDIA.NS": "C. E. INFO",
        "SUNPHARMA.NS": "SUN PHARMA",
        "PFC.NS": 'POWER FINANCE',
        "DRREDDY.NS": "DR REDDY",
        "CIPLA.NS": "CIPLA",
        "TORNTPHARM.NS": "TORRENT PHARMA"
    },
    "USE_TECHNICAL": True,
    "USE_FUNDAMENTAL": True,
    "MA_WINDOWS": [10, 20, 50],
    "RSI_WINDOW": 14,
    "VOL_WINDOW": 10,
    "MOMENTUM_WINDOW": 10,
    "PREDICT_DAYS_AHEAD": 5,
    "THRESHOLD": 0.5,
    "TRAIN_SIZE": 0.8,
    "PATH_OF_ALL_STOCKS_EPS_FILE": "data/ALL_STOCKS_EPS.csv",
    "PATH_OF_FINAL_DF": "data/final_df.csv",
    "CAPITAL": 500000
}

from datetime import datetime

# =========================
# 1. DATA HANDLING
# =========================

START_DATE = CONFIG["START_DATE"]

END_DATE = CONFIG["END_DATE"]
if END_DATE is None:
    END_DATE = datetime.today().strftime("%Y-%m-%d")

stocks = CONFIG["STOCKS"]