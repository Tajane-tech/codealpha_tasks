import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("Iris.csv")

df.drop('Id', axis=1, inplace=True)

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

df['species'] = df['species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

print("Feature Names:", ['sepal_length','sepal_width','petal_length','petal_width'])
print("Target Names:", ['setosa','versicolor','virginica'])

print("\n--- Statistical Details ---")

stats = df.groupby('species')[['sepal_length','sepal_width','petal_length','petal_width']].mean()

for species in stats.index:
    print(f"\nStatistics for Species {species}:\n")
    print(stats.loc[[species]])

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))