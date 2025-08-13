# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="aJxtNWIOTJoy" colab={"base_uri": "https://localhost:8080/"} outputId="56cf6283-a10d-48fb-c5ee-1846fe127365"
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime

tickers = ['NVDA']
data = yf.download(tickers, start="2015-07-21", end=datetime.now(), auto_adjust=False)
# Check the data
print(tabulate(data.tail(10), headers='keys', tablefmt='psql'))


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="J5vFgZ-rp3ta" outputId="2dfb8840-cd4a-447c-d414-143ec24fff85"
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[('Adj Close', 'NVDA')])
plt.title('NVDA Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.grid(True)
plt.show()

# Filter the date range
filtered_data = data.loc['2024-06-01':'2025-06-01']

# Plot
plt.figure(figsize=(12, 6))
plt.plot(filtered_data.index, filtered_data[('Adj Close', 'NVDA')])
plt.title('NVDA Adjusted Close Price (2024-06-01 to 2025-06-01)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.grid(True)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="4ZID2xVi5FG6" outputId="6913aa5b-3753-4d95-c842-1d4242c38888"
import requests
from datetime import datetime

api_key = 'd2ea8opr01qr1ro8v0v0d2ea8opr01qr1ro8v0vg'
symbol = 'NVDA'

# Fetch news
r = requests.get(
    f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-09-01&to=2024-10-01&token={api_key}"
)
news = r.json()

# Add readable_date field
for article in news:
    article['readable_date'] = datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d %H:%M:%S')

print("Length", len(news));
news

# %% colab={"base_uri": "https://localhost:8080/"} id="ADQu-6cl86fg" outputId="fba5ba03-6245-445d-c47a-434b8d9ab592"
print(f"First: {news[-1]}")
print(f"Last: {news[0]}")

# %% colab={"base_uri": "https://localhost:8080/"} id="dJJ8h7fGEEcC" outputId="dd7731f6-3652-4ede-ab10-d509c7cae383"
# !pip install vaderSentiment

# %% colab={"base_uri": "https://localhost:8080/"} id="HPsz8LobD0Tc" outputId="0163be9d-715b-4374-9c2b-94487c0514fb"
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
for article in news:
    sentiment = analyzer.polarity_scores(article['headline'])
    print(article['datetime'], article['headline'], sentiment['compound'])
