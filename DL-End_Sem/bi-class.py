import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Load the IMDB dataset
(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=10000)

X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=500))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,batch_size=512,validation_split=0.2,verbose=1)

loss,acc = model.evaluate(X_test,y_test)
print("Test Accuracy: ", acc * 100)

