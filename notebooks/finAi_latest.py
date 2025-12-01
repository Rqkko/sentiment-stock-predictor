#!/usr/bin/env python
# coding: utf-8

# Finnhub: d4m5t5hr01qjidhtok10d4m5t5hr01qjidhtok1g

# In[2]:


get_ipython().system('pip install finnhub')


# In[7]:


import os, time, hashlib
from datetime import datetime, timedelta
import pandas as pd, numpy as np
import finnhub, requests, yfinance as yf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from pandas.tseries.offsets import BDay
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[11]:


# ------------------- CONFIG -------------------

# FINNHUB_API_KEY = "d2arrjpr01qgk9ugd2ugd2arrjpr01qgk9ugd2v0"
# MARKETAUX_API_KEY = "3Ej6emn1jO1MzOjziTsrCacnbQvx6pjueugwiMb4"
SYMBOL = "NVDA"
CUTOVER_DATE = datetime(2024, 8, 28).date()  # use Finnhub on or after
YEAR_START = "2023-07-01"
YEAR_END = "2025-08-01"
LAG_DAYS = [1, 3, 7, 14]
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# API endpoints
MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"

# ------------------- INIT CLIENT -------------------

# finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


# # Building Data

# In[ ]:


# ------------------- 1) FETCH NEWS -------------------

all_news = []
start = datetime.strptime(YEAR_START, "%Y-%m-%d")
end = datetime.strptime(YEAR_END, "%Y-%m-%d")
delta = timedelta(days=7)

print("Fetching NVDA news using Marketaux (before cutover) and Finnhub (after)...")

while start <= end:
    batch_end = min(start + delta, end)
    date_from, date_to = start.date(), batch_end.date()

    #  if before cutoff date — use Marketaux
    if date_to < CUTOVER_DATE:
        # Marketaux request
        params = {
            "api_token": MARKETAUX_API_KEY,
            "symbols": SYMBOL,
            "published_after": f"{date_from}T00:00:00",
            "published_before": f"{date_to}T23:59:59",
            "limit": 1000
        }
        try:
            resp = requests.get(MARKETAUX_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            print(f"Marketaux: {len(data)} articles {date_from}→{date_to}")
            for a in data:
                all_news.append({
                    "id": a["uuid"],
                    "timestamp": a["published_at"],
                    "headline": a["title"],
                    "summary": a["description"],
                    "url": a["url"],
                    "source": "marketaux"
                })
        except Exception as e:
            print(f"Marketaux error {date_from}→{date_to}: {e}")

    #  if on/after cutoff date — use Finnhub
    else:
        try:
            fh = finnhub_client.company_news(SYMBOL, _from=str(date_from), to=str(date_to))
            print(f"Finnhub: {len(fh)} articles {date_from}→{date_to}")
            for a in fh:
                all_news.append({
                    "id": a["id"],
                    "timestamp": pd.to_datetime(a["datetime"], unit="s", utc=True),
                    "headline": a["headline"],
                    "summary": a.get("summary", ""),
                    "url": a.get("url", ""),
                    "source": "finnhub"
                })
        except Exception as e:
            print(f"Finnhub error {date_from}→{date_to}: {e}")

    start = batch_end + timedelta(days=1)
    time.sleep(0.3)

# ------------------- Deduplicate & Normalize -------------------

df = pd.DataFrame(all_news)
# hash headline+summary+timestamp
df["ts_str"] = df["timestamp"].astype(str)
df["hash"] = (df["headline"].fillna("") + df["summary"].fillna("") + df["ts_str"]).apply(
    lambda x: hashlib.sha1(x.encode()).hexdigest()
)
df = df.drop_duplicates(subset=["hash"]).reset_index(drop=True)
df["published_at"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["published_at"])
df["pub_date"] = df["published_at"].dt.date
df["text"] = (df["headline"] + ". " + df["summary"]).str.strip()

print(f"Total news after deduplication: {len(df)}")
print("Date range:", df['pub_date'].min(), "→", df['pub_date'].max())

df.to_csv(os.path.join(OUT_DIR, "news_combined.csv"), index=False)


# In[ ]:


# ------------------- 2) Sentiment Scoring -------------------

print("Scoring sentiment (FinBERT)...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(DEVICE)

def score(text):
    if not text:
        return (0,0,1,0)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    neg, neu, pos = softmax(out.logits.cpu().numpy()[0])
    return pos-neg, pos, neu, neg

sent = df["text"].apply(lambda t: pd.Series(score(t), index=["sent_net","sent_pos","sent_neu","sent_neg"]))
df = pd.concat([df, sent], axis=1)
df.to_csv(os.path.join(OUT_DIR, "news_with_sentiment.csv"), index=False)
print("Sentiment scoring done.")


# In[ ]:


# ------------------- 3) Fetch Prices -------------------

start_dt = (pd.to_datetime(YEAR_START) - timedelta(days=5)).strftime("%Y-%m-%d")
end_price = df["pub_date"].max() + BDay(max(LAG_DAYS)+5)
print("Fetching historical prices:", start_dt, "→", end_price.date())
prices = yf.download(SYMBOL, start=start_dt, end=end_price.strftime("%Y-%m-%d"), interval="1d")
prices.reset_index(inplace=True)
prices["date"] = pd.to_datetime(prices["Date"]).dt.date
prices = prices[["date","Open","High","Low","Close","Volume"]]
prices.to_csv(os.path.join(OUT_DIR, "price_data.csv"), index=False)
print("Price data saved.")

# Next: compute lags, directional labels, correlation, etc. (unchanged)


# In[6]:


# -------------------------
# 4) Compute lagged pct changes per news article
# -------------------------

# Use price DataFrame saved earlier
df_price = prices.copy()
df_price["date"] = pd.to_datetime(df_price["date"])
df_price = df_price.sort_values("date").reset_index(drop=True)
trading_dates = list(sorted(df_price["date"].unique()))

def get_trading_date_on_or_after(input_date):
    input_date = pd.Timestamp(input_date)
    for d in trading_dates:
        if d >= input_date:
            return d
    return None

def get_close_on_date(date):
    match = df_price.loc[df_price["date"] == date, "Close"]
    if len(match) == 0:
        return None
    return float(match.iloc[0])

def add_trading_days(start_date, days):
    if start_date not in trading_dates:
        start_date = get_trading_date_on_or_after(start_date)
    if start_date is None:
        return None
    try:
        start_idx = trading_dates.index(start_date)
    except ValueError:
        return None
    target_idx = start_idx + days
    if target_idx >= len(trading_dates):
        return None
    return trading_dates[target_idx]

records = []
price_min_date = df_price["date"].min()
price_max_date = df_price["date"].max()
print(f"Price data covers from {price_min_date.date()} to {price_max_date.date()}")

for _, r in df.iterrows():
    pub_date = pd.to_datetime(r["pub_date"])
    if pub_date < price_min_date:
        print(f"Warning: News date {pub_date.date()} before price data start {price_min_date.date()}. Skipping.")
        continue
    if pub_date > price_max_date:
        print(f"Warning: News date {pub_date.date()} after price data end {price_max_date.date()}. Skipping.")
        continue

    base_td = get_trading_date_on_or_after(pub_date)
    if base_td is None:
        print(f"Warning: No trading date found on/after news date {pub_date.date()}. Skipping.")
        continue
    base_close = get_close_on_date(base_td)
    if base_close is None:
        print(f"Warning: No close price for base trading date {base_td.date()}. Skipping.")
        continue

    rec = {
        "id": r.get("id", None),
        "published_at": r.get("published_at", None),
        "pub_date": pub_date,
        "base_trading_date": base_td,
        "base_close": base_close,
        "sent_net": r["sent_net"],
        "sent_pos": r["sent_pos"],
        "sent_neu": r["sent_neu"],
        "sent_neg": r["sent_neg"],
        "headline": r.get("headline", ""),
        "url": r.get("url", ""),
        "source": r.get("source", "")
    }

    for lag in LAG_DAYS:
        target_date = add_trading_days(base_td, lag)
        if target_date is None:
            print(f"Note: Lag {lag}d out of range for article on {pub_date.date()}.")
            rec[f"close_{lag}d"] = np.nan
            rec[f"pct_change_{lag}d"] = np.nan
            rec[f"trading_date_{lag}d"] = None
            continue

        close_price = get_close_on_date(target_date)
        if close_price is None:
            print(f"Warning: No price for lag {lag}d date {target_date.date()} (article {pub_date.date()}).")
            rec[f"close_{lag}d"] = np.nan
            rec[f"pct_change_{lag}d"] = np.nan
            rec[f"trading_date_{lag}d"] = None
            continue

        rec[f"close_{lag}d"] = close_price
        rec[f"pct_change_{lag}d"] = (close_price - base_close) / base_close
        rec[f"trading_date_{lag}d"] = target_date

    records.append(rec)

df_results = pd.DataFrame(records)
df_results.to_csv(os.path.join(OUT_DIR, "nvda_news_lagged_results.csv"), index=False)
print(f"Saved per-article lagged results. Processed articles: {len(df_results)}")


# In[29]:


# ================================
# Standalone Step 4: recompute lagged pct changes
# using existing price_data.csv and news_with_sentiment.csv
# ================================
import os
import numpy as np
import pandas as pd
from datetime import datetime

OUT_DIR = "./outputs"
LAG_DAYS = [1, 3, 7, 14, 30]

price_path = os.path.join(OUT_DIR, "price_data.csv")
news_sent_path = os.path.join(OUT_DIR, "news_with_sentiment.csv")

df_price = pd.read_csv(price_path)
df_price["date"] = pd.to_datetime(df_price["date"])
df_price = df_price.sort_values("date").reset_index(drop=True)

df = pd.read_csv(news_sent_path)
df["pub_date"] = pd.to_datetime(df["pub_date"])

trading_dates = list(sorted(df_price["date"].unique()))

def get_trading_date_on_or_after(input_date):
    input_date = pd.Timestamp(input_date)
    for d in trading_dates:
        if d >= input_date:
            return d
    return None

def get_close_on_date(date):
    match = df_price.loc[df_price["date"] == date, "Close"]
    if len(match) == 0:
        return None
    return float(match.iloc[0])

def add_trading_days(start_date, days):
    if start_date not in trading_dates:
        start_date = get_trading_date_on_or_after(start_date)
    if start_date is None:
        return None
    try:
        start_idx = trading_dates.index(start_date)
    except ValueError:
        return None
    target_idx = start_idx + days
    if target_idx >= len(trading_dates):
        return None
    return trading_dates[target_idx]

records = []
price_min_date = df_price["date"].min()
price_max_date = df_price["date"].max()
print(f"Price data covers from {price_min_date.date()} to {price_max_date.date()}")

for _, r in df.iterrows():
    pub_date = pd.to_datetime(r["pub_date"])
    if pub_date < price_min_date or pub_date > price_max_date:
        continue

    base_td = get_trading_date_on_or_after(pub_date)
    if base_td is None:
        continue
    base_close = get_close_on_date(base_td)
    if base_close is None:
        continue

    rec = {
        "id": r.get("id", None),
        "published_at": r.get("published_at", None),
        "pub_date": pub_date,
        "base_trading_date": base_td,
        "base_close": base_close,
        "sent_net": r["sent_net"],
        "sent_pos": r["sent_pos"],
        "sent_neu": r["sent_neu"],
        "sent_neg": r["sent_neg"],
        "headline": r.get("headline", ""),
        "url": r.get("url", ""),
        "source": r.get("source", "")
    }

    for lag in LAG_DAYS:
        target_date = add_trading_days(base_td, lag)
        if target_date is None:
            rec[f"close_{lag}d"] = np.nan
            rec[f"pct_change_{lag}d"] = np.nan
            rec[f"trading_date_{lag}d"] = None
            continue

        close_price = get_close_on_date(target_date)
        if close_price is None:
            rec[f"close_{lag}d"] = np.nan
            rec[f"pct_change_{lag}d"] = np.nan
            rec[f"trading_date_{lag}d"] = None
            continue

        rec[f"close_{lag}d"] = close_price
        rec[f"pct_change_{lag}d"] = (close_price - base_close) / base_close
        rec[f"trading_date_{lag}d"] = target_date

    records.append(rec)

df_results = pd.DataFrame(records)
out_path = os.path.join(OUT_DIR, "nvda_news_lagged_results.csv")
df_results.to_csv(out_path, index=False)
print(f"Saved per-article lagged results -> {out_path}, Processed articles: {len(df_results)}")


# # Analysis

# In[45]:


# ================================
# Standalone Step 5: Analysis using existing lag file
# (no need to rerun steps 1–3 if CSVs already exist)
# ================================
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

OUT_DIR = "./outputs"

# --- Load precomputed lagged results ---
lag_file = os.path.join(OUT_DIR, "nvda_news_lagged_results.csv")
if not os.path.exists(lag_file):
    raise FileNotFoundError(
        f"{lag_file} not found. You must run steps 1–4 at least once to create it."
    )

df_results = pd.read_csv(lag_file)

# Optional: ensure numeric types
df_results["sent_net"] = pd.to_numeric(df_results["sent_net"], errors="coerce")
for lag in LAG_DAYS:
    col = f"pct_change_{lag}d"
    if col in df_results.columns:
        df_results[col] = pd.to_numeric(df_results[col], errors="coerce")

print("Analyzing results by lag...")
summary_rows = []

for lag in LAG_DAYS:
    col = f"pct_change_{lag}d"

    if "sent_net" not in df_results.columns or col not in df_results.columns:
        print(f"Skipping lag {lag}d — missing 'sent_net' or '{col}' in df_results.")
        continue

    tmp = df_results[["sent_net", col]].dropna()
    n = len(tmp)

    if n >= 5:
        r_val, p_val = stats.pearsonr(tmp["sent_net"], tmp[col])
        X = tmp[["sent_net"]].values.reshape(-1, 1)
        y = tmp[col].values
        lr = LinearRegression().fit(X, y)
        coef = float(lr.coef_[0])
        intercept = float(lr.intercept_)
        r2 = float(lr.score(X, y))
    else:
        r_val = p_val = coef = intercept = r2 = np.nan

    tmp_dir = tmp.assign(
        sent_dir=np.sign(tmp["sent_net"]),
        price_dir=np.sign(tmp[col])
    )
    tmp_dir = tmp_dir[tmp_dir["sent_dir"] != 0]
    accuracy = (tmp_dir["sent_dir"] == tmp_dir["price_dir"]).mean() if len(tmp_dir) > 0 else np.nan

    summary_rows.append({
        "lag_days": lag,
        "n": n,
        "pearson_r": r_val,
        "p_value": p_val,
        "reg_coef": coef,
        "reg_intercept": intercept,
        "reg_r2": r2,
        "directional_accuracy": accuracy
    })

df_summary = pd.DataFrame(summary_rows).sort_values("lag_days")
if not df_summary.empty:
    summary_csv = os.path.join(OUT_DIR, "nvda_lag_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print("Saved lag summary ->", summary_csv)
    print(df_summary)
else:
    print("No valid lag results to save.")


# In[46]:


# -------------------------
# 6) Compute per-article best lag
# -------------------------
def best_lag_info(row):
    vals = row[[f"pct_change_{lag}d" for lag in LAG_DAYS]].abs()
    vals = vals.dropna()
    if vals.empty:
        return (None, np.nan)
    best_col = vals.idxmax()
    best_lag = int(best_col.split("_")[2].replace("d",""))
    best_val = row[best_col]
    return (best_lag, best_val)

df_results["best_lag_days"] = df_results.apply(lambda r: best_lag_info(r)[0], axis=1)
df_results["best_lag_pct_change"] = df_results.apply(lambda r: best_lag_info(r)[1], axis=1)
df_results.to_csv(os.path.join(OUT_DIR, "nvda_news_lagged_results_with_bestlag.csv"), index=False)
print("Saved per-article best-lag file.")


# In[47]:


# -------------------------
# 7) Plots
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(df_summary["lag_days"], df_summary["pearson_r"], marker="o")
plt.xlabel("Lag after news (days)")
plt.ylabel("Pearson correlation (sent_net vs pct_change)")
plt.title("NVDA: sentiment vs pct change correlation by lag")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "correlation_vs_lag.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(df_summary["lag_days"].astype(str), df_summary["directional_accuracy"])
plt.xlabel("Lag (days)")
plt.ylabel("Directional accuracy (fraction)")
plt.title("Direction accuracy of sentiment prediction by lag")
plt.grid(axis="y")
plt.savefig(os.path.join(OUT_DIR, "directional_accuracy_by_lag.png"), dpi=200)
plt.close()

print("Plots saved in", OUT_DIR)
print("Done.")


# In[48]:


# --------- load prerequisites from disk (so we don't need steps 1–3) ---------
df_results = pd.read_csv(os.path.join(OUT_DIR, "nvda_news_lagged_results.csv"))
df_price   = pd.read_csv(os.path.join(OUT_DIR, "price_data.csv"))


# In[49]:


# ================================
# 8) FEATURE ENGINEERING (standalone daily panel)
# ================================
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# --- Load from saved CSVs (no need to rerun steps 1–3) ---
price_file = os.path.join(OUT_DIR, "price_data.csv")
lags_file  = os.path.join(OUT_DIR, "nvda_news_lagged_results.csv")

df_price = pd.read_csv(price_file)
df_results = pd.read_csv(lags_file)

# --- Basic cleaning / typing ---
# Dates
df_price["date"] = pd.to_datetime(df_price["date"])
df_results["pub_date"] = pd.to_datetime(df_results["pub_date"])

# Numeric price columns
price_num_cols = ["Open", "High", "Low", "Close", "Volume"]
for c in price_num_cols:
    if c in df_price.columns:
        df_price[c] = pd.to_numeric(df_price[c], errors="coerce")

# Numeric sentiment columns (if present)
for c in ["sent_net", "sent_pos", "sent_neu", "sent_neg"]:
    if c in df_results.columns:
        df_results[c] = pd.to_numeric(df_results[c], errors="coerce")

# --- aggregate daily sentiment from articles ---
df_results["pub_date"] = df_results["pub_date"].dt.normalize()

news_day = (
    df_results
    .groupby("pub_date")
    .agg(
        sent_net_mean=("sent_net", "mean"),
        sent_net_sum=("sent_net", "sum"),
        sent_pos_mean=("sent_pos", "mean"),
        sent_neg_mean=("sent_neg", "mean"),
        n_articles=("id", "count")
    )
    .sort_index()
    .reset_index()
)

# --- price features ---
px = df_price.copy()
px["date"] = px["date"].dt.normalize()
px = px.sort_values("date").rename(columns={"Close": "close"}).set_index("date")

# returns & vol
px["ret_1d"]  = px["close"].pct_change(1)
px["ret_5d"]  = px["close"].pct_change(5)
px["ret_10d"] = px["close"].pct_change(10)
px["vol_10d"] = px["ret_1d"].rolling(10).std()
px["vol_20d"] = px["ret_1d"].rolling(20).std()

# flatten multiindex columns in px (defensive)
if isinstance(px.columns, pd.MultiIndex):
    px.columns = ["_".join([str(c) for c in col]).strip("_") for col in px.columns]

# ensure news_day index is 'date'
news_day = news_day.rename(columns={"pub_date": "date"}).set_index("date")

# merge features
feat = px.join(news_day, how="left")

# fill no-news days with zeros (no news signal that day)
for c in ["sent_net_mean", "sent_net_sum", "sent_pos_mean", "sent_neg_mean", "n_articles"]:
    if c in feat.columns:
        feat[c] = feat[c].fillna(0)

# --- future label horizons ---
HORIZONS = [1, 3, 5, 10]  # trading days ahead

# auto-detect close column (handles "close" or "close_NVDA" cases)
close_candidates = [c for c in feat.columns if c.startswith("close")]
if len(close_candidates) == 0:
    raise ValueError(f"No 'close*' column found in feat.columns: {feat.columns.tolist()}")
close_col = close_candidates[0]
print(f"Using close column: {close_col}")

for h in HORIZONS:
    # return over next h days
    feat[f"label_ret_{h}d"] = feat[close_col].shift(-h) / feat[close_col] - 1.0
    
    # 3-class direction label: -1, 0, +1 (with neutral band)
    band = 0.002  # +/- 0.2% as neutral band
    lab = feat[f"label_ret_{h}d"].apply(
        lambda r: 1 if r > band else (-1 if r < -band else 0)
    )
    feat[f"label_dir_{h}d"] = lab

feat = feat.dropna().copy()

print("Feature panel 'feat' ready. Shape:", feat.shape)
print("Columns:", feat.columns.tolist())


# In[ ]:


# # ================================
# # 9) Binary direction labels (UP / DOWN)
# # ================================

# # Choose the horizon (in trading days ahead)
# H_BIN = 5   # you can change to 3, 5, 10 later

# label_col_3class = f"label_dir_{H_BIN}d"
# if label_col_3class not in feat.columns:
#     raise ValueError(f"{label_col_3class} not found in feat. Run the feature-engineering cell first.")

# # Work on a copy so we don't mess up 'feat'
# bin_df = feat.copy()

# # Filter out neutral days (label_dir == 0)
# mask_non_neutral = bin_df[label_col_3class] != 0
# bin_df = bin_df.loc[mask_non_neutral].copy()

# print(f"Total days before neutral filter: {len(feat)}")
# print(f"Total days after removing neutral band: {len(bin_df)}")

# # Map -1 -> 0 (down), +1 -> 1 (up)
# bin_df["y_bin"] = bin_df[label_col_3class].map({-1: 0, 1: 1})

# # Sanity check
# print("Class distribution (0=down, 1=up):")
# print(bin_df["y_bin"].value_counts())


# In[59]:


# ================================
# 10) Feature matrix & time-based split
# ================================

# Feature columns to use
FEAT_BIN_COLS = [
    "ret_1d", "ret_5d", "ret_10d",
    "vol_10d", "vol_20d",
    "sent_net_mean", "sent_net_sum",
    "sent_pos_mean", "sent_neg_mean", "n_articles"
]

# Ensure all features exist
missing_cols = [c for c in FEAT_BIN_COLS if c not in bin_df.columns]
if missing_cols:
    raise ValueError(f"Missing feature columns for binary model: {missing_cols}")

# Sort by date index to respect time ordering
bin_df = bin_df.sort_index()

X = bin_df[FEAT_BIN_COLS].values
y = bin_df["y_bin"].values

# Time-based split: first 80% train, last 20% test
split_idx = int(len(bin_df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")


# In[ ]:


# ================================
# 11) Binary direction model (Logistic Regression)
# ================================
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

# --- 11.1: Choose prediction horizon and build binary labels ---

H = 1  # predict 1-day-ahead direction; change to 3,5,10 if you like
target_col = f"label_dir_{H}d"

if target_col not in feat.columns:
    raise ValueError(f"{target_col} not found in feat. Available label_dir_* columns: "
                     f"{[c for c in feat.columns if 'label_dir_' in c]}")

# label_dir_* is in {-1, 0, +1}; drop neutral 0 for pure up/down
df_bin = feat.loc[feat[target_col] != 0].copy()

# Binary target: 1 = UP, 0 = DOWN
y = (df_bin[target_col] == 1).astype(int)

# --- 11.2: Select features (price + sentiment + volume, etc.) ---

feature_cols = [
    "ret_1d", "ret_5d", "ret_10d",
    "vol_10d", "vol_20d",
    "sent_net_mean", "sent_net_sum",
    "sent_pos_mean", "sent_neg_mean",
    "n_articles",
]

missing_feats = [c for c in feature_cols if c not in df_bin.columns]
if missing_feats:
    raise ValueError(f"Missing feature columns in df_bin: {missing_feats}")

X = df_bin[feature_cols].copy()

# Safety: drop any remaining rows with NaNs in X or y
mask_valid = X.notna().all(axis=1) & y.notna()
X = X.loc[mask_valid]
y = y.loc[mask_valid]

# --- 11.3: Time-respecting train/test split (no shuffling) ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False  # keep chronological order to avoid look-ahead bias
)

# --- 11.4: Model: scale -> logistic regression ---

logreg_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # helpful if UP vs DOWN is imbalanced
        solver="lbfgs"
    ))
])

logreg_clf.fit(X_train, y_train)

y_pred = logreg_clf.predict(X_test)
y_proba = logreg_clf.predict_proba(X_test)[:, 1]  # prob of UP

# --- 11.5: Metrics ---

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_proba)

print(f"=== Binary Direction (H={H}d, UP=1 / DOWN=0) — Logistic Regression ===")
print(f"Samples train/test: {len(y_train)}/{len(y_test)}")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1 score : {f1:.3f}")
print(f"ROC AUC  : {auc:.3f}")
print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))


# In[63]:


# ================================
# 12) Helper to run binary logreg for any horizon
# ================================
from sklearn.model_selection import train_test_split

def run_logreg_direction(feat, H=1, feature_cols=None, test_size=0.2):
    """
    Train and evaluate a binary UP/DOWN model for horizon H (trading days).
    Returns:
        metrics: dict
        df_out: DataFrame with index=dates of test set and cols=[y_true, y_pred, proba_up]
        model: fitted Pipeline
    """
    target_col = f"label_dir_{H}d"
    if target_col not in feat.columns:
        raise ValueError(f"{target_col} not in feat.columns")

    # drop neutral
    df_bin = feat.loc[feat[target_col] != 0].copy()
    y = (df_bin[target_col] == 1).astype(int)

    if feature_cols is None:
        feature_cols = [
            "ret_1d", "ret_5d", "ret_10d",
            "vol_10d", "vol_20d",
            "sent_net_mean", "sent_net_sum",
            "sent_pos_mean", "sent_neg_mean",
            "n_articles",
        ]

    missing_feats = [c for c in feature_cols if c not in df_bin.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")

    X = df_bin[feature_cols].copy()

    # clean NaNs
    mask_valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask_valid]
    y = y.loc[mask_valid]

    # keep the date index for later joining
    dates = X.index

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates,
        test_size=test_size,
        shuffle=False
    )

    logreg_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    logreg_clf.fit(X_train, y_train)

    y_pred  = logreg_clf.predict(X_test)
    y_proba = logreg_clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    metrics = {
        "H": H,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }

    df_out = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "proba_up": y_proba,
    }, index=dates_test)

    print(f"=== Logistic Reg, H={H}d (UP=1/DOWN=0) ===")
    print(f"Train/Test size: {len(y_train)}/{len(y_test)}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1       : {f1:.3f}")
    print(f"ROC AUC  : {auc:.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    return metrics, df_out, logreg_clf


# # Evaluation

# In[64]:


metrics_1d, df_pred_1d, model_1d = run_logreg_direction(feat, H=1)


# In[56]:


# ================================
# 13) Compare horizons
# ================================
all_metrics = []
pred_store = {}

for H in [1, 3, 5, 10]:
    try:
        m, df_p, _ = run_logreg_direction(feat, H=H)
        all_metrics.append(m)
        pred_store[H] = df_p
    except Exception as e:
        print(f"H={H} failed: {e}")

df_horizon_metrics = pd.DataFrame(all_metrics)
df_horizon_metrics


# In[44]:


# ================================
# 14) Coefficients / feature importance
# ================================
H_best = 1  # or whichever you liked
_, _, model_best = run_logreg_direction(feat, H=H_best)

log_reg = model_best.named_steps["logreg"]
coef = log_reg.coef_[0]  # 1D array
feat_names = model_best.named_steps["scaler"].feature_names_in_

coef_df = pd.DataFrame({
    "feature": feat_names,
    "coef": coef
}).sort_values("coef", ascending=False)

print(coef_df)


# # Arima
