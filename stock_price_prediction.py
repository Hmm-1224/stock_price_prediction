import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Constants
API_KEY = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key
STOCK_SYMBOL = 'AAPL'  # Example: Apple Inc.
BASE_URL = 'https://www.alphavantage.co/query'

# Function to fetch live stock data from Alpha Vantage
def fetch_stock_data(symbol, interval='5min', outputsize='full'):
    params = {
        'function': 'TIME_SERIES_INTRADAY',
        'symbol': symbol,
        'interval': interval,
        'apikey': API_KEY,
        'outputsize': outputsize
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data

# Function to preprocess stock data
def preprocess_data(data):
    # Extracting the time series data
    time_series = data[f'Time Series ({interval})']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.astype(float)

    # Keeping only the 'close' prices
    df = df['4. close'].values.reshape(-1, 1)
    return df

# Function to create LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predicting the next closing price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fetch and preprocess stock data
data = fetch_stock_data(STOCK_SYMBOL)
interval = '5min'
prices = preprocess_data(data)

# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create training and test sets
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

# Create datasets for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create and train the LSTM model
model = create_model((X_train.shape[1], 1))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stopping])

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Inverse scaling

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(prices[len(prices) - len(predictions):], color='blue', label='Actual Stock Price')
plt.plot(np.arange(len(prices) - len(predictions), len(prices)), predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

print(f"Prediction accuracy: {100 * (1 - (np.abs(predictions - prices[train_size + time_step:]) / prices[train_size + time_step:])).mean():.2f}%")

