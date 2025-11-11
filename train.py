import pandas as pd
from agents.data_fetcher_agent import DataFetcherAgent
from agents.data_preprocessor_agent import DataPreprocessorAgent
from agents.lstm_agent import LSTMAgent
from agents.llm_agent import LLMAgent # Needed for sentiment
import config

# --- THIS IS THE MASTER AGENT'S SENTIMENT LOGIC, COPIED HERE ---
# We need to create the full dataset for training

def _compute_daily_sentiment(llm, news_data):
    print("\n📰 Computing daily sentiment from news data...")
    sentiment_by_date = {}
    if not news_data:
        return sentiment_by_date
    df_news = pd.DataFrame(news_data)
    if "publishedAt" not in df_news.columns:
        return sentiment_by_date
    df_news["date"] = pd.to_datetime(df_news["publishedAt"]).dt.date
    for date, group in df_news.groupby("date"):
        texts = list(filter(None, group["title"].tolist() + group["description"].tolist()))
        combined_text = " ".join(texts)
        if combined_text.strip():
            sentiment_result = llm.analyze_sentiment(combined_text)
            if isinstance(sentiment_result, str):
                sentiment_lower = sentiment_result.lower()
                if "positive" in sentiment_lower: score = 0.8
                elif "negative" in sentiment_lower: score = -0.8
                else: score = 0.0
            else: score = 0.0
            sentiment_by_date[pd.Timestamp(date)] = score
    return sentiment_by_date

def _attach_daily_sentiment(df_prices, sentiment_by_date, sentiment_col="Sentiment"):
    print("\n🔗 Attaching sentiment to stock price data...")
    df = df_prices.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.index = pd.to_datetime(df.index).normalize()
    df[sentiment_col] = [ sentiment_by_date.get(pd.Timestamp(date), 0.0) for date in df.index ]
    return df
# --- END OF COPIED LOGIC ---


def main_train():
    TICKER = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE = "2024-01-01" # Use a fixed range for training

    print(f"--- Starting Training Pipeline for {TICKER} ---")

    # 1. Initialize Agents
    fetcher = DataFetcherAgent()
    preprocessor = DataPreprocessorAgent()
    llm = LLMAgent()
    lstm = LSTMAgent()

    # 2. Fetch Data
    fetched = fetcher.run(TICKER, START_DATE, END_DATE)
    stock_df = fetched.get("stock_data")
    news_data = fetched.get("news_data")
    if stock_df is None:
        print("❌ No stock data, stopping training.")
        return

    # 3. Create Sentiment Features
    sentiment_by_date = _compute_daily_sentiment(llm, news_data)
    stock_with_sentiment = _attach_daily_sentiment(stock_df, sentiment_by_date)

    # 4. Preprocess Data
    print("\n📊 Preprocessing data for training...")
    processed_data = preprocessor.preprocess(stock_with_sentiment)
    
    if processed_data is None or processed_data['X_train'].shape[0] == 0:
        print("❌ No training data generated. Stopping.")
        return

    # 5. Train Model
    lstm.train(processed_data)

    print("\n--- ✅ Training Pipeline Complete ---")

if __name__ == "__main__":
    main_train()