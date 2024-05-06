import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

symbol = 'AAPL'

start_date = '2020-01-01'
end_date = '2021-01-01'

stock_data = yf.download(symbol, start=start_date, end=end_date)

stock_data.dropna(inplace=True)

stock_data.index = pd.to_datetime(stock_data.index)

plt.figure(figsize=(12, 6))
sns.barplot(x=stock_data.index, y=stock_data['Volume'])
plt.title('Stock Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()

plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Closing Price')
plt.plot(stock_data.index, stock_data['SMA_10'], label='SMA 10 Days')
plt.plot(stock_data.index, stock_data['SMA_20'], label='SMA 20 Days')
plt.title('Closing Price and Simple Moving Averages (SMA)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

acf_values = stock_data['Close'].autocorr()

print("Autocorrelation Function (ACF) Value:", acf_values)

correlation_matrix = stock_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heat Map of Correlation Matrix')
plt.show()

result = seasonal_decompose(stock_data['Close'], model='additive', period=20)

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(result.observed, label='Observed')
plt.legend()
plt.title('Time Series Decomposition')

plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.legend()

plt.tight_layout()
plt.show()
