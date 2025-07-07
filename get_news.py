# get_news.py
import os
import requests
import pandas as pd
from config import NEWS_API_KEY, COMPANIES

os.makedirs("news_data", exist_ok=True)

def fetch_news(company):
    print(f"📰 Fetching news for {company}...")
    url = f"https://newsapi.org/v2/everything?q={company}&sortBy=publishedAt&language=en&pageSize=100&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch news for {company}")
        return None

    articles = response.json().get("articles", [])
    if not articles:
        print(f"⚠️ No articles found for {company}")
        return None

    df = pd.DataFrame(articles)[['title', 'description', 'publishedAt']]
    df.to_csv(f"news_data/{company}_news.csv", index=False)
    print(f"✅ Saved {len(df)} articles to news_data/{company}_news.csv")
    return df

for company in COMPANIES:
    fetch_news(company)
