import pandas as pd
import os
from sklearn.preprocessing import StandardScaler #classe de scikitlearn pour standardiser les colonnes

# Définir le chemin vers le CSV
csv_path = os.path.join(os.path.dirname(__file__), '../data/ford.csv')
df = pd.read_csv(csv_path)

# Nettoyage des valeurs erron2es

#suppression des années impossibles
df = df[df['year'] <= 2025]

#suppression des mpg impossibles
df = df[df['mpg'] <= 100]

#suppression des prix trop inférieurs
df = df[df['price'] > 500]

#Encodage des colonnes catégorielles || encodage one-hot
categorical_cols = ['model', 'transmission', 'fuelType'] #liste des colonnes qui contiennent du texte
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True) #transforme les variables catégorielles en variables binaires

# Normalisation des colonnes numériques pour avoir toutes les données à la même échelle
numeric_cols = ['year', 'mileage', 'engineSize', 'tax', 'mpg'] #liste des colonnes numériques
scaler = StandardScaler() #création de l'objet scaler 
df[numeric_cols] = scaler.fit_transform(df[numeric_cols]) #calcul de la moyenne et de l'écart-type de chaque colonne | application de la formule pour avoir une moyenne = 0 et un écart-type = 1 pour chaque colonne

#Sauvegarde du dataset prétraité
preprocessed_path = os.path.join(os.path.dirname(__file__), '../data/ford_preprocessed.csv')
df.to_csv(preprocessed_path, index=False) #méthode de pandas pour sauvegarder un df en fichier csv

print(f"Nettoyage terminé ! Dataset sauvegardé dans {preprocessed_path}")

#Sauvegarde des artefacts nécessaires pour la prédiction

import joblib #bibliothèque pour sauvegarder
import json

# Créer le dossier models s'il n'existe pas
models_dir = os.path.join(os.path.dirname(__file__), '../models')
os.makedirs(models_dir, exist_ok=True)

# Sauvegarder le scaler
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
print(f"Scaler sauvegardé dans {os.path.join(models_dir, 'scaler.pkl')}")

# Sauvegarder la liste exacte des colonnes après préprocessing (ordre et noms)
columns_path = os.path.join(models_dir, 'columns.json')
with open(columns_path, 'w', encoding='utf-8') as f: #création du fichier en mode écriture 'w' | f -> fichier ouvert
    json.dump(list(df.columns), f, ensure_ascii=False, indent=2) #écriit l'objet donc la liste dans le fichier ouvert f au format json
print(f"Liste des colonnes sauvegardée dans {columns_path}")
