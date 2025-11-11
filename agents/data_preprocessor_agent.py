import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPreprocessorAgent:
    def __init__(self, standard_scaling=False, imputation_strategy='mean', test_size=0.2, target_column='Target', random_state=42, seq_length=60, lstm_preprocessing=True):
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        self.scaler = StandardScaler() if standard_scaling else MinMaxScaler()
        self.target_scaler = StandardScaler() if standard_scaling else MinMaxScaler()
        self.encoder = None
        self.target_encoder = None
        self.test_size = test_size
        self.target_column = target_column
        self.random_state = random_state
        self.seq_length = seq_length
        self.lstm_preprocessing = lstm_preprocessing
        # Store the actual scaler used for the target
        self.active_target_scaler = None

    def preprocess(self, data):
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # --- THIS IS THE FIX ---
        # Flatten the MultiIndex columns (e.g., ('Close', 'AAPL') -> 'Close')
        if isinstance(df.columns, pd.MultiIndex):
            print("🔧 Flattening MultiIndex columns...")
            # Takes the first part of each tuple: ('Close', 'AAPL') -> 'Close'
            df.columns = [col[0] for col in df.columns]
        # --- END OF FIX ---

        print(f"🔧 Preprocessing data - Initial shape: {df.shape}")
        print(f"   Initial columns: {list(df.columns)}") # Will now show clean names

        # --- CRITICAL: Check if 'Close' column exists FIRST ---
        if 'Close' not in df.columns:
            print(f"❌ Available columns: {list(df.columns)}")
            raise ValueError("The 'Close' column is missing, cannot create 'Target'.")
        
        # --- Create the 'Target' column properly ---
        df[self.target_column] = df['Close'].shift(-1)
        print(f"✅ Created '{self.target_column}' column by shifting 'Close' price")
        print(f"   Data shape after creating target: {df.shape}")

        # --- Check if Target column exists before dropna ---
        if self.target_column not in df.columns:
            raise KeyError(f"'{self.target_column}' column was not created successfully")

        # --- NOW this line will work ---
        initial_rows = len(df)
        df = df.dropna(subset=[self.target_column])
        print(f"✅ Dropped NaN targets. Rows: {initial_rows} → {len(df)}")

        # Separate features (X) and target (y)
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        print(f"   Features (X) shape: {X.shape}")
        print(f"   Target (y) shape: {y.shape}")
        print(f"   Feature columns: {list(X.columns)}")

        # If there's no data left after preprocessing, return early
        if len(X) == 0 or len(y) == 0:
            print("❌ No data available after preprocessing")
            return {
                'X_train': np.array([]),
                'X_test': np.array([]), 
                'y_train': np.array([]),
                'y_test': np.array([]),
                'target_scaler': self.active_target_scaler
            }

        # Identify numerical and categorical columns in features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"   Numerical columns: {numerical_cols}")
        print(f"   Categorical columns: {categorical_cols}")

        # Split the data
        if self.lstm_preprocessing:
            # For time series, we must not shuffle
            split_index = int(len(X) * (1 - self.test_size))
            X_train_df, X_test_df = X.iloc[:split_index], X.iloc[split_index:]
            y_train_df, y_test_df = y.iloc[:split_index], y.iloc[split_index:]
        else:
            # For non-time-series data, shuffling is okay
            X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
            )

        print(f"   Train set: {X_train_df.shape[0]} samples")
        print(f"   Test set: {X_test_df.shape[0]} samples")

        # Preprocess features
        if numerical_cols:
            X_train_df, X_test_df = self._preprocess_numerical_input(X_train_df, X_test_df, numerical_cols)
            print(f"✅ Preprocessed numerical columns")
        
        if categorical_cols:
            X_train_df, X_test_df = self._preprocess_categorical_input(X_train_df, X_test_df, categorical_cols)
            print(f"✅ Preprocessed categorical columns")

        # Preprocess the target variable (y)
        if pd.api.types.is_numeric_dtype(y_train_df):
            y_train_df, y_test_df = self._preprocess_numerical_output(y_train_df, y_test_df)
            print(f"✅ Preprocessed numerical target")
        else:
            y_train_df, y_test_df = self._preprocess_categorical_output(y_train_df, y_test_df)
            print(f"✅ Preprocessed categorical target")

        # --- Create 3D sequences for LSTM ---
        X_train_seq, y_train_seq = self._create_sequences(X_train_df.values, y_train_df.values)
        X_test_seq, y_test_seq = self._create_sequences(X_test_df.values, y_test_df.values)
        
        processed_data = {
            'X_train': X_train_seq,
            'X_test': X_test_seq,
            'y_train': y_train_seq,
            'y_test': y_test_seq,
            'target_scaler': self.active_target_scaler # <-- Pass the *active* scaler
        }
        
        print(f"🎯 Final processed data shapes (after sequencing):")
        print(f"   X_train: {processed_data['X_train'].shape}")
        print(f"   X_test: {processed_data['X_test'].shape}")
        print(f"   y_train: {processed_data['y_train'].shape}")
        print(f"   y_test: {processed_data['y_test'].shape}")
        
        return processed_data

    # --- HELPER METHOD for 3D sequences ---
    def _create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(self.seq_length, len(X) + 1):
             if i <= len(y):
                X_seq.append(X[i - self.seq_length : i, :])
                y_seq.append(y[i - 1])

        return np.array(X_seq), np.array(y_seq)

    def _preprocess_numerical_input(self, X_train_df, X_test_df, numerical_cols):
        X_train_df.loc[:, numerical_cols] = self.imputer.fit_transform(X_train_df.loc[:, numerical_cols])
        X_test_df.loc[:, numerical_cols] = self.imputer.transform(X_test_df.loc[:, numerical_cols])
        X_train_df.loc[:, numerical_cols] = self.scaler.fit_transform(X_train_df.loc[:, numerical_cols])
        X_test_df.loc[:, numerical_cols] = self.scaler.transform(X_test_df.loc[:, numerical_cols])
        return X_train_df, X_test_df
    
    def _preprocess_numerical_output(self, y_train_df, y_test_df):
        self.active_target_scaler = self.target_scaler
        
        y_train_scaled = self.active_target_scaler.fit_transform(y_train_df.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.active_target_scaler.transform(y_test_df.values.reshape(-1, 1)).ravel()
        
        y_train_df = pd.Series(y_train_scaled, index=y_train_df.index, name=y_train_df.name)
        y_test_df = pd.Series(y_test_scaled, index=y_test_df.index, name=y_test_df.name)
        return y_train_df, y_test_df

    def _preprocess_categorical_input(self, X_train_df, X_test_df, categorical_cols):
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(X_train_df[categorical_cols])
        encoded_train = pd.DataFrame(self.encoder.transform(X_train_df[categorical_cols]), index=X_train_df.index, columns=self.encoder.get_feature_names_out(categorical_cols))
        encoded_test = pd.DataFrame(self.encoder.transform(X_test_df[categorical_cols]), index=X_test_df.index, columns=self.encoder.get_feature_names_out(categorical_cols))
        X_train_df = X_train_df.drop(columns=categorical_cols).join(encoded_train)
        X_test_df = X_test_df.drop(columns=categorical_cols).join(encoded_test)
        return X_train_df, X_test_df
    
    def _preprocess_categorical_output(self, y_train_df, y_test_df):
        num_classes = y_train_df.nunique()
        if num_classes <= 2:
            self.target_encoder = LabelEncoder()
            self.active_target_scaler = self.target_encoder
            
            y_train_encoded = self.active_target_scaler.fit_transform(y_train_df)
            y_test_encoded = self.active_target_scaler.transform(y_test_df)
            
            y_train_df = pd.Series(y_train_encoded, index=y_train_df.index, name=self.target_column)
            y_test_df = pd.Series(y_test_encoded, index=y_test_df.index, name=self.target_column)
        else:
            self.target_encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
            self.active_target_scaler = self.target_encoder
            
            y_train_encoded = self.active_target_scaler.fit_transform(y_train_df.values.reshape(-1, 1))
            y_test_encoded = self.active_target_scaler.transform(y_test_df.values.reshape(-1, 1))
            
            y_train_df = pd.DataFrame(y_train_encoded, index=y_train_df.index, columns=self.active_target_scaler.get_feature_names_out([self.target_column]))
            y_test_df = pd.DataFrame(y_test_encoded, index=y_test_df.index, columns=self.active_target_scaler.get_feature_names_out([self.target_column]))
        return y_train_df, y_test_df