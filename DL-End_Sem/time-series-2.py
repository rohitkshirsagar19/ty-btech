import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("time_series_data.csv", parse_dates=[0], index_col=0)
df = df.iloc[1:].apply(pd.to_numeric, errors='coerce').dropna()

# Scale all features (input)
input_scaler = MinMaxScaler()
X_scaled = input_scaler.fit_transform(df)

# Scale only 'Close' for output target
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(df[['Close']])

# Create sequences
SEQ_LEN = 60
def create_multivariate_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X, y = create_multivariate_sequences(X_scaled, y_scaled, SEQ_LEN)

# Train/Test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.3),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Predictions
y_pred = model.predict(X_test)
y_pred_inv = target_scaler.inverse_transform(y_pred)
y_test_inv = target_scaler.inverse_transform(y_test)

# Plot Predicted vs Actual (Close)
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Close', linewidth=2)
plt.plot(y_pred_inv, label='Predicted Close', linewidth=2)
plt.title("📊 Multivariate Model - Actual vs Predicted Close")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# 🔮 Future Rolling Forecast
def rolling_forecast(model, recent_data, steps=10):
    forecast = []
    input_seq = recent_data[-SEQ_LEN:]  # shape: (SEQ_LEN, num_features)

    for _ in range(steps):
        pred = model.predict(input_seq.reshape(1, SEQ_LEN, X.shape[2]), verbose=0)
        forecast.append(pred[0])
        # Make new input with latest prediction + shift
        next_row = np.append(input_seq[1:], [input_seq[-1]], axis=0)
        input_seq = next_row

    forecast = np.array(forecast)
    return target_scaler.inverse_transform(forecast)

# Forecast next 10 days of Close
future_close = rolling_forecast(model, X_scaled, steps=10)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), future_close, marker='o', label='Forecasted Close Price')
plt.title("🔮 10-Day Future Close Forecast (Multivariate Input)")
plt.xlabel("Day Ahead")
plt.ylabel("Price")
plt.grid(True)
plt.legend()
plt.show()