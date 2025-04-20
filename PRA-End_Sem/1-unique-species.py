from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Split the dataset into training and testing sets
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

print("Predictions on the test set:")
print(classification_report(y_test, y_pred))

df['species'] = pd.Categorical.from_codes(y, iris.target_names)
print("Unique species in the dataset:")
print(df.groupby('species').mean())
