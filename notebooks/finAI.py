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

# %% id="aJxtNWIOTJoy" colab={"base_uri": "https://localhost:8080/"} outputId="7a632e20-a502-446b-c64f-f91ab3732c97"
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


# %% id="lC5_-ZuDYT3d" colab={"base_uri": "https://localhost:8080/"} outputId="5af6e26d-88d5-43db-ed3e-6ab40a8bd351"
print("Hello World")

# %% colab={"base_uri": "https://localhost:8080/"} id="LzIDJD5UpsgN" outputId="6d838575-0592-4447-ab77-6527063fda53"
print("I am testing Colab version control with GitHub")

# %% colab={"base_uri": "https://localhost:8080/", "height": 372} id="J5vFgZ-rp3ta" outputId="6df492a5-9f74-40cc-d580-0f8a14bc2553"
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[('Adj Close', 'NVDA')])
plt.title('NVDA Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.grid(True)
plt.show()
