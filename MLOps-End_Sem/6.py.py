from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Train Random Forest for comparison
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Create directory
os.makedirs("results", exist_ok=True)

# Predict
y_pred_log = log_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig("results/confusion_matrix_logistic.png")
plt.close()
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

# Get probabilities
y_scores_log = log_model.predict_proba(X_test)[:, 1]

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_scores_log)
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
pr_display.plot()
plt.title("Precision-Recall Curve - Logistic Regression")
plt.savefig("results/precision_recall_logistic.png")
plt.close()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logistic
y_pred_log = log_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

# Random Forest
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("📊 Model Comparison")
print(f"Logistic Regression: Accuracy={acc_log:.2f}, F1 Score={f1_log:.2f}")
print(f"Random Forest: Accuracy={acc_rf:.2f}, F1 Score={f1_rf:.2f}")
