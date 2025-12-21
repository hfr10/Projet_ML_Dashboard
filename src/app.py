from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd
import os

from predict_model import preprocess_df, load_artifacts   # On réutilise ton code

app = Flask(__name__)

# --- Load model, scaler, columns at startup ---

from predict_model import MODEL_PATH, SCALER_PATH, COLUMNS_PATH


model, scaler, feature_columns = load_artifacts(
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    columns_path=COLUMNS_PATH
)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "JSON body missing"}), 400

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # preprocess
    df_ready = preprocess_df(df, scaler=scaler, feature_columns=feature_columns)

    # prediction
    pred = model.predict(df_ready)[0]

    return jsonify({"prediction": float(pred)})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is running"})


if __name__ == "__main__":
    app.run(debug=True)
