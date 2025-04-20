import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

# Load the dataset
df = pd.read_csv('sentiment_analysis_1.csv')

# Preprocess the data
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Tokenize the text
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences , maxlen=100)

# Split the data into training and testing sets
X_train , X_test , y_train , y_test = train_test_split(padded , df['label'] , test_size=0.2 , random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=10000,output_dim=64,input_length=100))
model.add(SimpleRNN(64))
model.add(Dense(64,activation='relu'))

model.add(Dense(3,activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(X_train,y_train,epochs=5,batch_size=32,validation_split=0.2,verbose=1)

# Evaluate the model
loss , acc = model.evaluate(X_test,y_test)
print("Test Accuracy: ", acc * 100)

# Make predictions
i = 0
sample = X_test[i].reshape(1,100)
pred = model.predict(sample)
predicted_class = np.argmax(pred)
print("Predicted class: ", predicted_class)
print("Actual class: ", y_test[i])
print("Text: ", df['text'][i])
