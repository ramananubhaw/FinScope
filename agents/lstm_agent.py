import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import traceback

class LSTMAgent:
    """
    The LSTM Agent for quantitative analysis.
    This model expects 3D sequence data with multiple features
    from the DataPreprocessorAgent.
    """
    def __init__(self, sequence_length=60, model_path="models/lstm_model.h5"):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        # This agent does *not* have its own scaler.
        # It relies on the 'target_scaler' passed from the preprocessor.

    def predict(self, processed_data):
        """
        Predicts the next day's closing price using the 6-feature model.
        
        Args:
            processed_data (dict): Dictionary from the preprocessor,
                                 containing 3D sequenced data and the target_scaler.
        
        Returns:
            float: The predicted (inverse-transformed) closing price.
        """
        if self.model is None:
            # This should not happen if master_agent calls load() first
            print(" Error: Model is not loaded. Call load() first.")
            return None
        
        print("🤖 Making prediction with multi-feature LSTM model...")
        
        try:
            X_test = processed_data['X_test']
            target_scaler = processed_data['target_scaler']
            
            if len(X_test) == 0:
                print("❌ No test data available for prediction")
                # Try to use X_train if X_test is empty
                X_train = processed_data.get('X_train')
                if X_train is None or len(X_train) == 0:
                    print("❌ No train data available either. Cannot predict.")
                    return None
                print("⚠️ No test data, using last sequence from TRAINING data.")
                last_sequence = X_train[-1]
            else:
                 # Get the last sequence from the test data
                last_sequence = X_test[-1] 
            
            # Reshape for LSTM: (1, sequence_length, num_features)
            # last_sequence is already (sequence_length, num_features)
            prediction_input = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))
            
            # Make scaled prediction
            predicted_scaled = self.model.predict(prediction_input)
            
            # Inverse transform the prediction
            predicted_price = target_scaler.inverse_transform(predicted_scaled)
            
            return float(predicted_price[0][0])

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            
            # Fallback: return the last known price
            y_test = processed_data.get('y_test', [])
            if len(y_test) > 0 and target_scaler:
                last_scaled_price = y_test[-1]
                fallback_price = target_scaler.inverse_transform([[last_scaled_price]])
                print(f"⚠️ Prediction failed. Returning last known price: ${fallback_price[0][0]:.2f}")
                return float(fallback_price[0][0])
                
            return 150.0 # Default fallback

    def build_model(self, input_shape):
        """
        Builds the LSTM model architecture.
        input_shape should be (sequence_length, num_features)
        e.g., (60, 6)
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        print(f"✅ LSTM model built successfully for input shape {input_shape}")

    def train(self, processed_data):
        """
        Trains the LSTM model on the preprocessed 3D data.
        """
        print("Training 6-feature LSTM model...")
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        if len(X_train) == 0:
            print("❌ Cannot train: No training data.")
            return

        # Reshape data for LSTM [samples, timesteps, features]
        # X_train is already in this shape from the preprocessor
        # Input shape is (timesteps, features)
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        self.build_model(input_shape)
        self.model.fit(X_train, y_train, epochs=25, batch_size=32)
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"✅ Model trained and saved to {self.model_path}")

    def load(self):
        """
        Loads a pre-trained model from the specified path.
        """
        if os.path.exists(self.model_path):
            print(f"Loading pre-trained model from {self.model_path}...")
            self.model = load_model(self.model_path)
            print("✅ Model loaded successfully.")
            return True
        else:
            print(f"❌ Error: Model file not found at {self.model_path}")
            return False