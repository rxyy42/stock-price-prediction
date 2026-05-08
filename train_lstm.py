import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import os

# --- Configuration (Match these with your main app script) ---
TICKER = 'AAPL'  # Choose the stock you want to train a model for
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2025-04-20' # Use a long period for training
LSTM_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume'] # Features for the model
TARGET_COLUMN = 'Close' # What we want to predict
LSTM_SEQUENCE_LENGTH = 1000 # How many past days to use for prediction
LSTM_MODEL_PATH = 'lstm_model.h5'
SCALER_PATH = 'scaler.pkl'

# --- 1. Fetch Data ---
print(f"Fetching data for {TICKER} from {TRAIN_START_DATE} to {TRAIN_END_DATE}...")
try:
    data = yf.download(TICKER, start=TRAIN_START_DATE, end=TRAIN_END_DATE)
    if data.empty:
        raise ValueError("No data fetched.")
    print(f"Data fetched successfully: {data.shape[0]} rows")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# --- 2. Preprocess Data ---
print("Preprocessing data...")
# Select features and handle missing values (simple forward fill)
data = data[LSTM_FEATURES].ffill().bfill()

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit scaler ONLY on the features, then transform
scaled_data = scaler.fit_transform(data)

# Save the fitted scaler!
print(f"Saving scaler to {SCALER_PATH}...")
try:
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")
except Exception as e:
    print(f"Error saving scaler: {e}")
    exit()

# --- 3. Create Sequences ---
print(f"Creating sequences with length {LSTM_SEQUENCE_LENGTH}...")
X = []
y = []
target_col_index = data.columns.get_loc(TARGET_COLUMN) # Find index of target column

for i in range(LSTM_SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-LSTM_SEQUENCE_LENGTH:i, :]) # All features for the sequence
    y.append(scaled_data[i, target_col_index])         # The target value at the end of the sequence

X, y = np.array(X), np.array(y)

print(f"Sequences created: X shape={X.shape}, y shape={y.shape}")

# --- 4. Split Data ---
# Splitting sequentially for time series is often preferred, but train_test_split is simpler here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Use random_state for reproducibility
print(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# --- 5. Build LSTM Model ---
print("Building LSTM model...")
model = Sequential([
    LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(units=256, return_sequences=False), # Second LSTM layer
    Dropout(0.3),
    Dense(units=1) # Output layer predicts one value (the target column)
])

# --- 6. Compile Model ---
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 7. Train Model ---
print("Training LSTM model (this may take some time)...")
# Adjust epochs and batch_size for performance/time trade-off
# Use validation_data to monitor performance on unseen test data during training
history = model.fit(
    X_train, y_train,
    epochs=25, # Start with fewer epochs, increase if needed
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1 # Show progress
)

print("Training complete.")

# --- 8. Evaluate Model (Optional) ---
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss}")

# --- 9. Save Trained Model ---
print(f"Saving trained model to {LSTM_MODEL_PATH}...")
try:
    model.save(LSTM_MODEL_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

print("\n--- Training Process Finished ---")
print(f"Files created: {LSTM_MODEL_PATH}, {SCALER_PATH}")
print("You can now run the main GUI application script.")