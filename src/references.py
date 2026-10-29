
import time
import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 1. Charger les donnees
df = pd.read_csv("data/transactions.csv")

# 2. Separer les variables explicatives et la cible
X = df.drop(columns="fraude")
y = df["fraude"]

# 3. Diviser les donnees en entrainement et test
Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Definir les modeles a comparer
modeles = {
    "reference-logistique": LogisticRegression(max_iter=1000),
    "manuel-rf-500-16": RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        random_state=42
    ),
}

# 5. Configurer MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp4-automl")

# 6. Entrainer, evaluer et enregistrer les resultats
for nom, m in modeles.items():
    t0 = time.time()

    m.fit(Xtr, ytr)

    duree = round(time.time() - t0, 2)
    predictions = m.predict(Xte)
    f1 = round(
        f1_score(yte, predictions, zero_division=0),
        4
    )

    # Enregistrer les resultats dans MLflow
    with mlflow.start_run(run_name=nom):
        mlflow.log_params({
            "outil": "scikit-learn",
            "approche": nom
        })

        mlflow.log_metrics({
            "f1": f1,
            "duree_s": duree
        })

    # Afficher les resultats
    print(
        f"{nom:22s} F1 = {f1} "
        f"(entrainement : {duree} s)"
    )

print("Comparaison terminee. Consulte MLflow pour les resultats.")
