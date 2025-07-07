import yfinance as yf
from datetime import datetime, timedelta

def fetch_stock(ticker, name):
    print(f"📈 Fetching stock data for {name} ({ticker})...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.reset_index(inplace=True)
    df.to_csv(f"stock_data/{ticker}.csv", index=False)

if __name__ == "__main__":
    companies = {
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "Google": "GOOGL",
        "Meta": "META",
        "Amazon": "AMZN",
        "Microsoft": "MSFT"
    }

    for name, ticker in companies.items():
        fetch_stock(ticker, name)
