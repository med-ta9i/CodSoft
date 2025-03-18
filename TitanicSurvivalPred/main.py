import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Chargement et Exploration des Données
df = pd.read_csv("D:\\python\\CodSoft\\TitanicSurvivalPred\\TitanicDataset.csv")
print(df.head())
print(df.info())
print(df.describe())

# 2. Prétraitement des Données
# Suppression des colonnes inutiles
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# Remplacement des valeurs manquantes
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# df["Age"].fillna(df["Age"].median(), inplace=True)
# df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encodage des variables catégoriques
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# Séparation des variables indépendantes et dépendantes
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Normalisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraînement du Modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Visualisation des Résultats
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrice de Confusion")
plt.show()
