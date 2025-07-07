import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load model and vectorizer
with open("stock_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Sample accuracy display (replace with actual test data if available)
try:
    test_df = pd.read_csv("sample_sentiment.csv")
    test_df = test_df.dropna(subset=['title', 'target'])
    X_test = vectorizer.transform(test_df['title'])
    y_test = test_df['target']
    acc = model.score(X_test, y_test)
except:
    acc = None

st.set_page_config(page_title="📈 Stock Sentiment Predictor", layout="centered")
st.title("📰 Stock News Sentiment → 📊 Stock Price Movement")

st.sidebar.header("📍 Model Overview")
if acc is not None:
    st.sidebar.metric("✅ Accuracy", f"{acc * 100:.2f}%")
else:
    st.sidebar.write("Accuracy data unavailable.")

# Input form
st.subheader("📌 Enter a news headline")
news_input = st.text_area("Headline", "Apple releases new AI-powered MacBooks")

# Store prediction history
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔮 Predict"):
    if news_input.strip():
        vector = vectorizer.transform([news_input])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        result = "📈 Predicted: Stock will go **UP**" if prediction == 1 else "📉 Predicted: Stock will go **DOWN**"
        st.success(result)
        st.write(f"🔵 Probability UP: `{proba[1]:.2f}` | 🔴 Probability DOWN: `{proba[0]:.2f}`")

        st.session_state.history.append((news_input, prediction))
    else:
        st.warning("Please enter a headline.")

# Show prediction history
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕓 Prediction History")
    for i, (headline, pred) in enumerate(reversed(st.session_state.history[-5:]), 1):
        label = "UP" if pred == 1 else "DOWN"
        st.write(f"{i}. **{label}** – {headline[:70]}...")

# Upload CSV
st.markdown("---")
st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with a 'title' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'title' in df.columns:
            df = df.dropna(subset=['title'])
            vectors = vectorizer.transform(df['title'])
            preds = model.predict(vectors)
            df['prediction'] = preds
            st.dataframe(df[['title', 'prediction']])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
        else:
            st.error("Your CSV must contain a 'title' column.")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Metric summary
st.markdown("---")
st.subheader("📊 Summary of Predictions")
col1, col2 = st.columns(2)
with col1:
    ups = sum(pred == 1 for _, pred in st.session_state.history)
    st.metric("🟢 Predicted UP", ups)
with col2:
    downs = sum(pred == 0 for _, pred in st.session_state.history)
    st.metric("🔴 Predicted DOWN", downs)
