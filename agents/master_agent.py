import pandas as pd
import numpy as np
from .llm_agent import LLMAgent
from .lstm_agent import LSTMAgent
from .data_preprocessor_agent import DataPreprocessorAgent
from .data_fetcher_agent import DataFetcherAgent


class MasterAgent:
    """
    The Master Agent orchestrates the entire FinScope workflow
    """

    def __init__(self):
        self.fetcher = DataFetcherAgent()
        self.preprocessor = DataPreprocessorAgent()
        self.llm = LLMAgent()
        self.lstm = LSTMAgent()

    def run(self, stock_symbol: str, start_date: str, end_date: str):
        """
        Main pipeline execution
        """
        try:
            print(f"\n🎯 Starting FinScope workflow for {stock_symbol}...")
            
            # Step 1: Load the LSTM model and its scalers
            if not self.lstm.load():
                print("❌ Halting pipeline: Could not load LSTM model.")
                print("💡 Please run a training script to create 'lstm_model.h5'")
                return None

            # Step 2: Fetch stock and news data
            fetched = self.fetcher.run(stock_symbol, start_date, end_date)
            stock_df = fetched.get("stock_data")
            news_data = fetched.get("news_data")

            if stock_df is None or stock_df.empty:
                print("❌ No stock data available. Terminating pipeline.")
                return None

            print(f"✅ Stock data shape: {stock_df.shape}")
            print(f"✅ News articles: {len(news_data) if news_data else 0}")

            # Step 3: Run LLM sentiment analysis
            sentiment_by_date = self._compute_daily_sentiment(news_data)

            # Step 4: Attach sentiment to stock data
            stock_with_sentiment = self._attach_daily_sentiment(stock_df, sentiment_by_date)

            print(f"🔍 DEBUG: Data passed to preprocessor - Columns: {list(stock_with_sentiment.columns)}")

            # Step 5: Preprocess data for LSTM
            print(f"\n📊 Preprocessing data...")
            processed_data = self.preprocessor.preprocess(stock_with_sentiment)
            
            if processed_data is None or processed_data['X_test'].shape[0] == 0:
                print("❌ No test data generated after preprocessing. Cannot predict.")
                return None

            print(f"✅ Preprocessing completed successfully")
            print(f"   Processed data keys: {list(processed_data.keys())}")

            # Step 6: LSTM Prediction
            print(f"\n🤖 Running LSTM prediction...")
            predicted_price = self.lstm.predict(processed_data)

            if predicted_price is None:
                print(f"\n⚠️ Final prediction failed.")
                return None

            # --- NEW DECISION LOGIC STARTS HERE ---

            # 7. Get the last actual price for comparison
            # We must flatten the columns if they are MultiIndex, just in case
            stock_df_copy = stock_df.copy()
            if isinstance(stock_df_copy.columns, pd.MultiIndex):
                stock_df_copy.columns = [col[0] for col in stock_df_copy.columns]
                
            last_actual_price = stock_df_copy['Close'].iloc[-1]
            
            # 8. Generate a "Buy/Sell/Hold" signal
            signal = "Hold"
            percent_change = (predicted_price - last_actual_price) / last_actual_price
            
            # (You can adjust these thresholds)
            if percent_change > 0.02:  # If predicted price is > 2% higher
                signal = "Buy"
            elif percent_change < -0.02: # If predicted price is > 2% lower
                signal = "Sell"

            # 9. Get recent sentiment for reasoning
            recent_sentiment_scores = []
            for date, score in sorted(sentiment_by_date.items(), reverse=True)[:5]: # Get 5 most recent
                recent_sentiment_scores.append(f"On {date.date()}: score = {score}")
            recent_sentiment_text = "\n".join(recent_sentiment_scores)
            if not recent_sentiment_text:
                recent_sentiment_text = "No recent news sentiment available."

            # 10. Ask LLM for the final reasoning
            print(f"\n🧠 Generating final decision and reasoning...")
            reasoning_prompt = f"""
            You are FinScope, an AI financial analyst. Your LSTM model has made a prediction.
            Your job is to provide a final recommendation and a brief, easy-to-understand reasoning.

            Here is the data:
            - Stock Ticker: {stock_symbol}
            - Last Actual Close Price: ${last_actual_price:.2f}
            - Your Predicted Next Day Price: ${predicted_price:.2f}
            - Your Final Recommendation: {signal}
            - Recent News Sentiment:
            {recent_sentiment_text}

            Please provide a summary. Explain *why* the {signal} recommendation was made,
            linking the price prediction (e.g., "model predicts a rise/fall") and the recent news sentiment.
            Keep it concise (2-3 sentences).
            """
            
            # Use the generic 'generate_response' method from your LLMAgent
            # (assuming you've updated LLMAgent as advised)
            final_summary = self.llm.generate_response(reasoning_prompt) 

            # 11. Print the final output
            print("\n" + "="*40)
            print(f"🎉 FinScope Final Analysis for {stock_symbol}")
            print("="*40)
            print(f"   Last Actual Price:       ${last_actual_price:.2f}")
            print(f"   Predicted Next Day Price:  ${predicted_price:.2f}")
            print(f"   Recommendation:          **{signal}**")
            print("\n   Reasoning:")
            # Simple text wrap for the reasoning
            import textwrap
            wrapped_summary = "\n   > ".join(textwrap.wrap(final_summary, width=70))
            print(f"   > {wrapped_summary}")
            print("="*40)
            
            # Return the full analysis
            return {
                "ticker": stock_symbol,
                "last_price": last_actual_price,
                "predicted_price": predicted_price,
                "recommendation": signal,
                "reasoning": final_summary
            }
            # --- END OF NEW DECISION LOGIC ---

        except Exception as e:
            print(f"\n❌ Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _compute_daily_sentiment(self, news_data):
        """
        Groups news articles by date and analyzes sentiment
        """
        print("\n📰 Computing daily sentiment from news data...")

        sentiment_by_date = {}
        if not news_data:
            print("   No news data found for sentiment analysis.")
            return sentiment_by_date

        df_news = pd.DataFrame(news_data)
        if "publishedAt" not in df_news.columns:
            print("   Missing 'publishedAt' field in news data.")
            return sentiment_by_date

        df_news["date"] = pd.to_datetime(df_news["publishedAt"]).dt.date

        for date, group in df_news.groupby("date"):
            texts = list(filter(None, group["title"].tolist() + group["description"].tolist()))
            combined_text = " ".join(texts)

            if combined_text.strip():  # Only analyze if there's actual text
                sentiment_result = self.llm.analyze_sentiment(combined_text) # Use analyze_sentiment here
                sentiment_score = self._extract_sentiment_score(sentiment_result)
                sentiment_by_date[pd.Timestamp(date)] = sentiment_score
            else:
                pass # Don't print for empty text

        return sentiment_by_date

    def _extract_sentiment_score(self, sentiment_result):
        """
        Convert sentiment result to numerical score [-1, 1]
        """
        if sentiment_result is None:
            return 0.0

        # Handle string responses from LLM
        if isinstance(sentiment_result, str):
            sentiment_lower = sentiment_result.lower()
            if "positive" in sentiment_lower:
                return 0.8
            elif "negative" in sentiment_lower:
                return -0.8
            else:
                return 0.0
        else:
            return 0.0  # Default for unexpected types

    def _attach_daily_sentiment(self, df_prices, sentiment_by_date, sentiment_col="Sentiment"):
        """
        Adds sentiment column to price data
        """
        print("\n🔗 Attaching sentiment to stock price data...")

        df = df_prices.copy()
        
        # --- FIX for MultiIndex ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        # --- End Fix ---
            
        df.index = pd.to_datetime(df.index).normalize()
        df[sentiment_col] = [
            sentiment_by_date.get(pd.Timestamp(date), 0.0) for date in df.index
        ]

        print(f"✅ Sentiment column attached. Data shape: {df.shape}")
        return df