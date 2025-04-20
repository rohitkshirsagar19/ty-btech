from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(X, columns=['text'])
df['category'] = y

# Split the dataset into training and testing sets
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create a pipeline that combines TfidfVectorizer and MultinomialNB
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Fit the model on the training data
model.fit(X_train,y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

print("Predictions on the test set:")
print(classification_report(y_test, y_pred))

