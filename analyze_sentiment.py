# analyze_sentiment.py

from textblob import TextBlob

def analyze_sentiment(df):
    def get_sentiment_label(polarity):
        if polarity > 0.1:
            return 'POSITIVE'
        elif polarity < -0.1:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'

    df = df.dropna(subset=['title'])
    df['title'] = df['title'].astype(str)
    df['polarity'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment'] = df['polarity'].apply(get_sentiment_label)

    # Optional: numeric sentiment score
    sentiment_map = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    return df
