# score_sentiment.py
import os
import pandas as pd
from textblob import TextBlob
from config import COMPANIES

os.makedirs("scored_news", exist_ok=True)

def analyze_sentiment(text):
    if pd.isna(text):
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

for company in COMPANIES:
    print(f"🧠 Analyzing sentiment for {company}...")
    path = f"news_data/{company}_news.csv"
    if not os.path.exists(path):
        print(f"⚠️ Missing news file for {company}")
        continue

    df = pd.read_csv(path)
    df['polarity'] = df['title'].apply(analyze_sentiment)
    df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral')
    df.to_csv(f"scored_news/{company}_news_scored.csv", index=False)
    print(f"✅ Saved scored data to scored_news/{company}_news_scored.csv")
