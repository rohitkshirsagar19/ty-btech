from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y

print("Missing Values:\n", iris_df.isnull().sum())

print("Data Types:\n", iris_df.dtypes)
print("Statistical Summary:\n", iris_df.describe())
print("Correlation Matrix:\n", iris_df.corr())
plt.figure(figsize=(10, 6))
sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix') 
plt.show()