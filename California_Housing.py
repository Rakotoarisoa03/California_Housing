import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Configuration graphique
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
print("Bibliothèques importées avec succès")
# Charger le jeu de données California Housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(f"Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\nVariable cible : MedHouseVal (valeur médiane du logement en centaines de milliers de $)")
print(f"\nDescription des variables :")
print("-" * 60)
descriptions = {
'MedInc': 'Revenu médian du quartier',
'HouseAge': 'Âge médian des logements',
'AveRooms': 'Nombre moyen de pièces par logement',
'AveBedrms': 'Nombre moyen de chambres par logement',
'Population': 'Population du quartier',
'AveOccup': 'Nombre moyen d\'occupants par logement',
'Latitude': 'Latitude géographique',
'Longitude': 'Longitude géographique',
'MedHouseVal': 'Prix médian du logement (cible)'
}
for col, desc in descriptions.items():
    print(f" {col:15s} → {desc}")

# Affichage 1ère lignes avec .head() 
print("\nLes premières lignes du dataset :")
print(df.head())

# Vérification des types de colonnes et détection des éventuelles valeurs manquantes 
print("\nVérification types colonnes et détection des valeurs manquantes :")
df.info()

# Statistiques descriptives 
print("\nStatistiques descriptives du dataset :")
print(df.describe())

# Vérification des valeurs manquantes
print("\nValeurs manquantes :")
print(df.isnull().sum())

# Vérification des doublons
nb_doublons = df.duplicated().sum()
print(f"\nNombre de lignes dupliquées : {nb_doublons}")

# Suppression des doublons s'il y en a
df = df.drop_duplicates()

# Détection visuelle des outliers avec des boxplots
colonnes_outliers = ['AveRooms', 'AveBedrms', 'AveOccup']

plt.figure(figsize=(12, 4))
for i, col in enumerate(colonnes_outliers, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Filtrage des valeurs aberrantes extrêmes
print("\nDimensions avant suppression des outliers :", df.shape)

df = df[
    (df['AveRooms'] < 50) &
    (df['AveBedrms'] < 20) &
    (df['AveOccup'] < 20)
]

print("Dimensions après suppression des outliers :", df.shape)


# Distribution de la variable cible (MedHouseVal)
plt.figure(figsize=(10, 5))
sns.histplot(df['MedHouseVal'], bins=50, kde=True)
plt.title("Distribution du prix médian des logements (MedHouseVal)")
plt.xlabel("Prix médian (en centaines de milliers de $)")
plt.ylabel("Fréquence")
plt.show()

# Boxplot de la variable cible
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['MedHouseVal'])
plt.title("Boxplot du prix médian des logements")
plt.show()

# Matrice de corrélation
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation des variables")
plt.show()

# Scatter plots : variables les plus corrélées à la cible
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['MedInc'], y=df['MedHouseVal'], alpha=0.5)
plt.title("Relation entre le revenu médian et le prix des logements")
plt.xlabel("Revenu médian (MedInc)")
plt.ylabel("Prix médian (MedHouseVal)")
plt.show()

# Carte géographique des prix des logements
plt.figure(figsize=(8, 6))
plt.scatter(
    df['Longitude'],
    df['Latitude'],
    c=df['MedHouseVal'],
    cmap='viridis',
    s=10
)
plt.colorbar(label="Prix médian des logements")
plt.title("Répartition géographique des prix des logements en Californie")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Création de nouvelles variables dérivées

# Ratio nombre de pièces par chambre
df['PiecesParChambre'] = df['AveRooms'] / df['AveBedrms']

# Estimation du nombre de logements par quartier
df['PopParLogement'] = df['Population'] / df['AveOccup']

# Vérification des nouvelles variables
print("\nAperçu des nouvelles variables créées :")
print(df[['PiecesParChambre', 'PopParLogement']].head())

# Séparation des variables explicatives et de la cible

# Variable cible
y = df['MedHouseVal']

# Variables explicatives (features)
X = df.drop(columns=['MedHouseVal'])

print("\nDimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

from sklearn.model_selection import train_test_split

# Séparation des données (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Vérification des dimensions
print("\nDimensions des ensembles :")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)


# Instanciation du modèle
model = LinearRegression()

# Entraînement sur les données d'entraînement
model.fit(X_train, y_train)

# Affichage de l'ordonnée à l'origine (β0)
print("Ordonnée à l'origine (β0) :", model.intercept_)

# Affichage des coefficients (βi)
coefficients = pd.DataFrame({
    'Variable': X_train.columns,
    'Coefficient': model.coef_
})
print("\nCoefficients des variables :")
print(coefficients)

# Visualisation des coefficients
plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Variable', data=coefficients)
plt.title("Impact des variables sur le prix médian des logements")
plt.xlabel("Coefficient (βi)")
plt.ylabel("Variable")
plt.show()

# Évaluation rapide du modèle sur l'ensemble test
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nÉvaluation du modèle sur X_test :")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Coefficient de détermination (R²) : {r2:.4f}")


# Prédire les prix avec le modèle entraîné
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Erreur quadratique moyenne (MSE)
mse = mean_squared_error(y_test, y_pred)

# Racine de l'erreur quadratique moyenne (RMSE)
rmse = np.sqrt(mse)

# Erreur absolue moyenne (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Coefficient de détermination (R²)
r2 = r2_score(y_test, y_pred)

# Affichage
print("Évaluation du modèle :")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")




plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--')  # ligne diagonale y = y_pred parfaite
plt.xlabel("Valeurs réelles (y_test)")
plt.ylabel("Valeurs prédites (y_pred)")
plt.title("Valeurs réelles vs Valeurs prédites")
plt.show()


# Calcul des résidus
residus = y_test - y_pred

# a) Histogramme des résidus
plt.figure(figsize=(8,4))
sns.histplot(residus, bins=50, kde=True)
plt.title("Distribution des résidus")
plt.xlabel("Résidu (réel - prédit)")
plt.ylabel("Fréquence")
plt.show()

# b) Résidus vs Prédictions
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residus, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')  # ligne horizontale à 0
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.title("Résidus vs Valeurs prédites")
plt.show()



coefficients = pd.DataFrame({
    'Variable': X_train.columns,
    'Coefficient': model.coef_
})

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Variable', data=coefficients,
            palette=['green' if c > 0 else 'red' for c in coefficients['Coefficient']])
plt.title("Impact des variables sur le prix médian des logements")
plt.xlabel("Coefficient")
plt.ylabel("Variable")
plt.show()




# Calcul des résidus et de l'erreur en pourcentage
residus = y_test - y_pred
erreur_pct = (residus / y_test) * 100

# Créer le DataFrame
resultats = X_test.copy()  # inclure toutes les features du test
resultats['Prix_reel'] = y_test
resultats['Prix_pred'] = y_pred
resultats['Residus'] = residus 
resultats['Erreur_pct'] = erreur_pct

# Aperçu
print(resultats.head())


# Exporter le DataFrame en fichier CSV
resultats.to_csv("resultats_regression.csv", index=False)
print("\nRésultats exportés avec succès dans 'resultats_regression.csv'")
