API_KEY_GEMINI = "45he9fubprwiuvbpiefbfiboubyVo"
API_KEY_NEWS = "TY1OOKQID545yt4LZR"
TICKER = "AAPL" # The stock ticker you want to analyze (e.g., "AAPL", "GOOGL", "MSFT")
START_DATE = "2020-01-01" # Start date for historical data
END_DATE = "2024-01-01" # End date for historical data

# --- Model Settings ---
LSTM_SEQUENCE_LENGTH = 60 # Number of past days' data to use for predicting the next day
LSTM_MODEL_PATH = "models/lstm_model.h5" # Path to save/load the trained LSTM model
ENABLE_TRAINING = True # Set to False to skip training and load an existing model