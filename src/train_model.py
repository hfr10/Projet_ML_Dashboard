import pandas as pd
from sklearn.model_selection import train_test_split #pour division des données entraînement/test
from sklearn.ensemble import RandomForestRegressor #modèle de machine learning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #pour mesure de performances du modèle
import joblib #pour sauvegarder et recharger le modèle entraîné
import os #pour gérer les chemins des fichiers

#Chargement du dataset prétraité
data_path = os.path.join(os.path.dirname(__file__), '../data/ford_preprocessed.csv')
df = pd.read_csv(data_path) #charge le csv en dataframe

#Séparation des features (X) et la cible (y)
X = df.drop(columns=['price']) #en entrée/features : toutes les colonnes sauf la cible/target
Y = df['price'] #la cible/target

#Division du dataset en train / test
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

#Créer et entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42) #paramétrage de la forêt
model.fit(X_train, Y_train) #liaison des features/variables à la cible/prix

#Évaluation du modèle
y_pred = model.predict(X_test) #prédiction des prix
mae = mean_absolute_error(y_test, y_pred) #erreur moyenne en valeur absolue | le plus petit possible
#rmse = mean_squared_error(y_test, y_pred, squared=False) #écart-type des erreurs | le plus petit possible
r2 = r2_score(y_test, y_pred) #qualité de prédiction | le plus proche possible de 1

print(f"Entraînement terminé !")
print(f"MAE  : {mae:.2f}")
#print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

#Sauvegarde du modèle entraîné
model_path = os.path.join(os.path.dirname(__file__), '../models/random_forest_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path) #enregistre le modèle dans un fichier .pkl (binaire)

print(f"Modèle sauvegardé dans {model_path}")
