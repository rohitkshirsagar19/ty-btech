from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 🔹 1. Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(int)

# 🔹 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 3. Train a Gaussian Naive Bayes classifier
print("Training Gaussian Naive Bayes...")
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 🔹 4. Predict and evaluate
y_pred = gnb.predict(X_test)

# 🔹 5. Report results
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# 🔹 6. Show confusion matrix
import seaborn as sns
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - MNIST (Bayes Classifier)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
