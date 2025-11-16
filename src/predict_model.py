# src/predict_model.py
"""
Predict util for the trained model.

- Loads model, scaler and feature list from models/
- Provides functions to preprocess raw inputs and predict
- CLI allows quick local tests:
  python src/predict_model.py --json '{"year":2018,"mileage":20000,...}'
  python src/predict_model.py --json-file samples/input.json
  python src/predict_model.py --csv-file samples/inputs.csv --output samples/preds.csv
"""
import os
import json
import argparse
from typing import List, Optional, Tuple

import pandas as pd
import joblib
import numpy as np

# ----------------- Paths / defaults -----------------
BASE_DIR = os.path.dirname(__file__)  # src/
MODELS_DIR = os.path.join(BASE_DIR, '../models')
MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
COLUMNS_PATH = os.path.join(MODELS_DIR, 'columns.json')

# These must match what you used during preprocessing
NUMERIC_COLS = ['year', 'mileage', 'engineSize', 'tax', 'mpg']
CATEGORICAL_COLS = ['model', 'transmission', 'fuelType']  # will be one-hot encoded with drop_first=True

# ----------------- Load artifacts -----------------
def load_artifacts(model_path: str = MODEL_PATH,
                   scaler_path: str = SCALER_PATH,
                   columns_path: str = COLUMNS_PATH):
    """
    Charge et retourne (model, scaler_or_None, feature_columns_or_None).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train first.")

    model = joblib.load(model_path)

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    feature_columns = None
    if os.path.exists(columns_path):
        with open(columns_path, 'r', encoding='utf-8') as f:
            feature_columns = json.load(f)

    return model, scaler, feature_columns

# ----------------- Preprocessing -----------------
def preprocess_df(df: pd.DataFrame,
                  scaler: Optional[object],
                  feature_columns: Optional[List[str]],
                  numeric_cols: List[str] = NUMERIC_COLS,
                  categorical_cols: List[str] = CATEGORICAL_COLS,
                  model_obj: Optional[object] = None) -> pd.DataFrame:
    """
    Prétraite df pour le modèle :
    - Si df contient déjà toutes les feature_columns => on suppose qu'il est déjà prétraité et
      on renvoie df réindexé sur feature_columns.
    - Sinon :
        * on applique pd.get_dummies(..., drop_first=True) sur les colonnes catégorielles présentes
        * on ajoute les colonnes numériques manquantes (valeur 0)
        * si scaler fourni : on applique scaler.transform sur numeric_cols
        * on réindexe selon feature_columns (ou model.feature_names_in_ si disponible)
    """
    df_proc = df.copy()

    # remove target if present
    if 'price' in df_proc.columns:
        df_proc = df_proc.drop(columns=['price'])

    # If we have feature_columns and they are all present in df_proc => assume preprocessed
    if feature_columns is not None and set(feature_columns).issubset(set(df_proc.columns)):
        # Reindex to ensure ordering exactly matches model training
        df_proc = df_proc.reindex(columns=feature_columns, fill_value=0)
        return df_proc

    # Otherwise: raw input branch
    # 1) One-hot encode categorical columns present in the DF
    cols_to_encode = [c for c in categorical_cols if c in df_proc.columns]
    if cols_to_encode:
        df_proc = pd.get_dummies(df_proc, columns=cols_to_encode, drop_first=True)

    # 2) Ensure numeric columns exist
    for c in numeric_cols:
        if c not in df_proc.columns:
            df_proc[c] = 0

    # 3) Apply scaler.transform if we have a scaler
    if scaler is not None:
        # scaler.transform expects numeric columns in the same order
        df_proc[numeric_cols] = scaler.transform(df_proc[numeric_cols])
    else:
        # No scaler available -> warn (predictions may be wrong if model was trained on scaled data)
        # We don't modify numeric cols further; they are left as-is.
        pass

    # 4) Reindex to feature_columns (important)
    if feature_columns is not None:
        df_proc = df_proc.reindex(columns=feature_columns, fill_value=0)
    else:
        # try to use model.feature_names_in_ if provided
        try:
            cols_expected = getattr(model_obj, 'feature_names_in_', None)
            if cols_expected is not None:
                df_proc = df_proc.reindex(columns=cols_expected, fill_value=0)
        except Exception:
            pass

    return df_proc

# ----------------- Prediction helpers -----------------
def predict_df(df: pd.DataFrame, model, scaler: Optional[object] = None, feature_columns: Optional[List[str]] = None) -> Tuple[List[float], pd.DataFrame]:
    """
    Prétraite et prédit sur un DataFrame (1+ lignes). Retourne (preds_list, df_ready)
    """
    df_ready = preprocess_df(df, scaler=scaler, feature_columns=feature_columns, model_obj=model)
    preds = model.predict(df_ready)
    return preds.tolist(), df_ready

def predict_from_dicts(dicts: List[dict], model, scaler: Optional[object] = None, feature_columns: Optional[List[str]] = None) -> List[float]:
    """
    Accepte une liste de dicts (observations brutes) et renvoie la liste des prédictions.
    """
    df = pd.DataFrame(dicts)
    preds, _ = predict_df(df, model, scaler=scaler, feature_columns=feature_columns)
    return preds

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Predictor util for trained model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--json', help='JSON string for a single observation or list of observations')
    group.add_argument('--json-file', help='Path to a JSON file (object or list)')
    group.add_argument('--csv-file', help='Path to a CSV file containing input rows (raw or preprocessed)')
    parser.add_argument('--output', help='If provided and input is CSV, save predictions to this CSV path')
    args = parser.parse_args()

    model, scaler, feature_columns = load_artifacts()

    # Read input
    if args.json:
        try:
            data = json.loads(args.json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        if isinstance(data, dict):
            df_in = pd.DataFrame([data])
        else:
            df_in = pd.DataFrame(data)
    elif args.json_file:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            df_in = pd.DataFrame([data])
        else:
            df_in = pd.DataFrame(data)
    elif args.csv_file:
        df_in = pd.read_csv(args.csv_file)
    else:
        raise ValueError("No input provided")

    preds, df_ready = predict_df(df_in, model, scaler=scaler, feature_columns=feature_columns)

    # Print preview
    print("=== Data prétraitée (aperçu) ===")
    print(df_ready.head(3))
    print("\n=== Prédictions ===")
    results = df_in.reset_index(drop=True).copy()
    results['prediction'] = preds
    print(results[['prediction']].head(10))

    # Optionally save results if CSV input and output specified
    if args.output and args.csv_file:
        results.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")

if __name__ == '__main__':
    main()
