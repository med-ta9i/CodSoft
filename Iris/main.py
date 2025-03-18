import pandas as pd

# Load the dataset
df = pd.read_csv('D:\\python\\codsoft\\Iris\\IRIS.csv')

# Display the first few rows
print(df.head())

print(df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

# Pair plot to visualize relationships between features
sns.pairplot(df, hue='species')
plt.show()

X = df.drop('species', axis=1)
y = df['species']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()