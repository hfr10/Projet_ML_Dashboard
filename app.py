# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import json
import numpy as np

app = Flask(__name__)

# Paths (adaptés si tu as une autre arborescence)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')         # optionnel
COLUMNS_PATH = os.path.join(BASE_DIR, 'models', 'columns.json')     # optionnel (liste des colonnes utilisées pour l'entraînement)

# Chargement du modèle
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}. Entraîne et sauvegarde d'abord le modèle.")
model = joblib.load(MODEL_PATH)

# Chargement optionnel du scaler et des colonnes (recommandé)
scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

feature_columns = None
if os.path.exists(COLUMNS_PATH):
    with open(COLUMNS_PATH, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)  # doit être une liste

# Colonnes catégorielles attendues (nominales dans ton dataset)
CATEGORICAL_COLS = ['model', 'transmission', 'fuelType']
NUMERIC_COLS = ['year', 'mileage', 'engineSize', 'tax', 'mpg']  # celles qu'on scalait

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

def preprocess_input(raw: dict):
    """
    raw : dictionnaire JSON envoyé par le client, par ex:
    {
      "model": "Focus",
      "year": 2018,
      "price": 14000,    # price peut être ignoré côté prédiction
      "transmission": "Manual",
      "mileage": 20000,
      "fuelType": "Petrol",
      "tax": 145,
      "mpg": 58.9,
      "engineSize": 1.2,
      "engineSize": 1.2

      
    }
    Retour : DataFrame avec colonnes alignées sur feature_columns si possible.
    """
    # Construire un DataFrame d'une seule ligne à partir du JSON
    df = pd.DataFrame([raw])

    # On retire la colonne price si fournie (cible)
    if 'price' in df.columns:
        df = df.drop(columns=['price'])

    # Encodage one-hot simple pour les colonnes catégorielles
    # (crée des colonnes comme 'fuelType_Petrol', 'transmission_Manual', 'model_Focus', ...)
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=False)

    # Si nous avons un scaler sauvegardé, on applique la même normalisation aux colonnes numériques
    if scaler is not None:
        # S'assurer que toutes les colonnes numériques existent (si absentes, remplir par 0)
        for col in NUMERIC_COLS:
            if col not in df.columns:
                df[col] = 0
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    else:
        # Sans scaler : vérifier que les colonnes existent, sinon les ajouter
        for col in NUMERIC_COLS:
            if col not in df.columns:
                df[col] = 0
        # NOTE: si le modèle a été entraîné sur des variables normalisées, l'absence du scaler dégradera la prédiction.
    
    # Si on dispose des colonnes exactes du modèle (feature_columns), on réindexe
    if feature_columns:
        # Ajoute les colonnes manquantes avec 0, et retire les colonnes en trop
        df = df.reindex(columns=feature_columns, fill_value=0)
    else:
        # Tentative prudente : ordonner les colonnes comme le modèle les a vues si possible
        try:
            cols_expected = getattr(model, 'feature_names_in_', None)
            if cols_expected is not None:
                df = df.reindex(columns=cols_expected, fill_value=0)
        except Exception:
            # si rien n'est disponible, on laisse les colonnes telles quelles
            pass

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Aucun JSON reçu'}), 400

        # Supporter soit un dict (une observation) soit une liste de dicts
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            return jsonify({'error': 'Le JSON doit être un objet ou une liste d\'objets'}), 400

        # Prétraiter chaque enregistrement et concaténer
        dfs = [preprocess_input(r) for r in records]
        X = pd.concat(dfs, ignore_index=True)

        # Prédiction
        preds = model.predict(X)

        return jsonify({'predictions': [float(p) for p in preds]})

    except Exception as e:
        # Log l'erreur côté serveur (ici on renvoie le message pour debug local)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
