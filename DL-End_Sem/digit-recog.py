import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten , Dense , Conv2D , MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the data to fit the model
X_train = X_train.reshape(-1,28,28,1)/255.0
X_test = X_test.reshape(-1,28,28,1)/255.0

# Convert the labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model
model = Sequential()
model.add(Conv2D(32 , (3,3),activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(X_train,y_train,epochs=5,batch_size=32,validation_split=0.2,verbose=1)

# Evaluate the model
loss , acc = model.evaluate(X_test,y_test)
print("Test Accuracy: ", acc * 100)

# Make predictions
i = 2
sample = X_test[i].reshape(-1,28,28,1)
pred = model.predict(sample)
predicted_class = np.argmax(pred)

# Display the image and prediction
plt.imshow(X_test[i].reshape(28,28),cmap='gray')
plt.title(f'Predicted: {predicted_class}')
plt.axis('off')
plt.show()