import pandas as pd #pandas bibliothèque python pour manipuler les données tabulaires (csv, excel)

# Charger le CSV
df = pd.read_csv('data/ford.csv') #pandas lit le csv, le transforme en dataframe et le stocke dans la variable df

# Afficher les 5 premières lignes
print("Aperçu des données :")
print(df.head()) #df.head() affiche les 5 premières lignes du dataframe par défaut | print(df.head(10)) affiche les 10 premières lignes

# Informations sur les colonnes et types
print("\nInfos du DataFrame :")
print(df.info())

# Statistiques descriptives des colonnes numériques
print("\nStatistiques descriptives :")
print(df.describe())

# Vérifier les valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())
