import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
iris = load_iris()
X = iris.data[:, 2:4]  # Petal length and width only
y = iris.target
class_names = iris.target_names



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Try different values of K and evaluate
k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"\nK = {k}")
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

# Plot K vs Accuracy
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='blue')
plt.title("KNN Accuracy for Different K Values")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
