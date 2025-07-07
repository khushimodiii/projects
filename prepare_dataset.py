import pandas as pd
import os
from datetime import timedelta

# Company mapping: Display name → Stock ticker
companies = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Google": "GOOGL",
    "Meta": "META",
    "Amazon": "AMZN",
    "Microsoft": "MSFT"
}

all_data = []

for name, ticker in companies.items():
    print(f"\n🔍 Merging data for {name}...")

    news_path = f"scored_news/{name}_news_scored.csv"
    stock_path = f"stock_data/{ticker}.csv"

    if not os.path.exists(news_path) or not os.path.exists(stock_path):
        print(f"⚠️ Missing data for {name}, skipping.")
        continue

    # Load news and stock data
    news_df = pd.read_csv(news_path)
    stock_df = pd.read_csv(stock_path)

    # Ensure 'publishedAt' exists and rename to 'date'
    # Ensure 'publishedAt' exists and rename to 'date'
    if 'publishedAt' not in news_df.columns:
        print(f"⚠️ No 'publishedAt' column in {name}_news_scored.csv, skipping.")
        continue
    news_df['date'] = pd.to_datetime(news_df['publishedAt'], errors='coerce')
    news_df['date'] = news_df['date'].dt.tz_localize(None)  # <- this fixes the merge error
    news_df.dropna(subset=['date'], inplace=True)


    # Clean and rename stock date
    stock_df.columns = [col.lower() for col in stock_df.columns]  # normalize column names
    if 'date' not in stock_df.columns or 'close' not in stock_df.columns:
        print(f"⚠️ Required columns missing in {ticker}.csv, skipping.")
        continue
    stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
    stock_df = stock_df[['date', 'close']].dropna()

    # Logging
    print(f"📰 {name} - News dates: {news_df['date'].min()} → {news_df['date'].max()}")
    print(f"📈 {name} - Stock dates: {stock_df['date'].min()} → {stock_df['date'].max()}")
    print(f"📅 News dtype: {news_df['date'].dtype}")
    print(f"📅 Stock dtype: {stock_df['date'].dtype}")
    print(f"📊 News entries: {len(news_df)} | Stock entries: {len(stock_df)}")

    # Sort for merge_asof
    news_df = news_df.sort_values('date')
    stock_df = stock_df.sort_values('date')

    # Merge sentiment with stock price on closest *previous* date (price already known)
    merged = pd.merge_asof(news_df, stock_df, on='date', direction='backward')

    # Now shift stock price forward by 1 day for next_close
    stock_df_shifted = stock_df.copy()
    stock_df_shifted['date'] = stock_df_shifted['date'] - timedelta(days=1)
    stock_df_shifted.rename(columns={'close': 'next_close'}, inplace=True)

    # Merge to get next day's close
    merged = pd.merge_asof(merged, stock_df_shifted, on='date', direction='forward')

    # Calculate target
    merged['difference'] = merged['next_close'] - merged['close']
    merged['target'] = (merged['difference'] > 0).astype(int)

    # Add company name
    merged['company'] = name

    # Show target distribution
    print(f"📊 {name} target distribution:\n{merged['target'].value_counts()}")

    all_data.append(merged)

# Concatenate all data
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    print("\n📈 Sample price comparisons:")
    print(final_df[['date', 'company', 'close', 'next_close', 'difference', 'target']].head(10))

    final_df.to_csv("sentiment_merged.csv", index=False)
    print("\n✅ Final merged dataset saved as: sentiment_merged.csv")
else:
    print("❌ No data available after processing.")
