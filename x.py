import pandas as pd

df = pd.read_csv("stock_data/AAPL.csv", skiprows=3)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
print(df.head())
