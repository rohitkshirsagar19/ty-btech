import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset

df = pd.read_csv('letter-recognition.csv')

print("Dataset: \n",df.head())
print("Dataset columns: \n",df.columns)
print("Dataset Shape: \n" ,df.shape)

X = df.drop('letter', axis=1)
y = df['letter']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(26, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluating the Model
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy: ", acc * 100)


# Making Predictions
y_prob = model.predict(X_test)

y_pred = np.argmax(y_prob, axis=1)

y_letter = encoder.inverse_transform(y_pred)

print("Predicted letters: ", y_letter[:10])
print("Actual letters: ", encoder.inverse_transform(y_test[:10]))
