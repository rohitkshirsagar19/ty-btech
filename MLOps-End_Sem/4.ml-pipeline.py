# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load Dataset (e.g., Wine Dataset)
from sklearn.datasets import load_wine
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Step 3: Preprocessing
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Visualize Feature Importances
plt.figure(figsize=(10, 6))
plt.barh(data.feature_names, model.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Wine Dataset Feature Importance")
plt.show()
