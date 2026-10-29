import sys
import time
from pathlib import Path
import pandas as pd
import mlflow
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60 # secondes de calcul
df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
Path('reports').mkdir(exist_ok=True)
journal = f'reports/flaml_{budget}s.log'
a = AutoML()
t0 = time.time()
a.fit(Xtr, ytr, task='classification', metric='f1', # on optimise le F1
 time_budget=budget, seed=42, verbose=0,
 log_file_name=journal, log_type='all') # trace de TOUS les essais
duree = round(time.time() - t0, 1)
f1 = round(f1_score(yte, a.predict(Xte)), 4) # jeu de test jamais vu
essayees = [k for k, v in a.best_config_per_estimator.items() if v is not None]
print(f'budget {budget} s | famille retenue : {a.best_estimator} | F1 test = {f1} | {duree} s')
print('familles essayees :', essayees)
print('meilleure config :', a.best_config)
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp4-automl')
with mlflow.start_run(run_name=f'flaml-{budget}s'):
 mlflow.log_params({'outil': 'flaml', 'budget_s': budget,
 'famille': a.best_estimator})
 mlflow.log_metrics({'f1': f1, 'f1_validation': round(1 - a.best_loss, 4),
 'duree_s': duree})
 mlflow.log_artifact(journal) # le journal suit le run 