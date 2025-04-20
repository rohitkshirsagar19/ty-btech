import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 🔹 1. Load the dataset (fix columns manually)
columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
data = pd.read_csv("time_series_data.csv", skiprows=2, names=columns)

# 🔹 2. Drop any rows with missing values
data = data.dropna()

# 🔹 3. Prepare date index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 🔹 4. Normalize the 'Close' column
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(data[['Close']])

# 🔹 5. Create sequences (60 past days → 1 next day)
X, y = [], []
for i in range(60, len(scaled_close)):
    X.append(scaled_close[i-60:i])
    y.append(scaled_close[i])

X, y = np.array(X), np.array(y)

# 🔹 6. Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

# 🔹 7. Compile & Train
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)

# 🔹 8. Predict next day
last_60 = scaled_close[-60:].reshape(1, 60, 1)
predicted_scaled = model.predict(last_60)
predicted_price = scaler.inverse_transform(predicted_scaled)

print("📈 Predicted next closing price:", predicted_price[0][0])
