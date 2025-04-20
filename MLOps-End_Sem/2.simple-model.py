from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

C = [0.1, 1.0, 10.0]
accuracies = []  # Renamed to plural for clarity

# Training loop
for c in C:
    model = LogisticRegression(C=c, max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    current_acc = accuracy_score(y_test, y_pred)
    accuracies.append(current_acc)
    print(f"Accuracy for C={c}: {current_acc * 100:.2f}%")
    
    model_name = f"iris_model_C_{c}.joblib"
    joblib.dump(model, model_name)
    print(f"Model saved as {model_name}")

# Final report
print("\nFinal Accuracies:")
for c, acc in zip(C, accuracies):
    print(f"C={c}: {acc * 100:.2f}%")


