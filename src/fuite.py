import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('data/transactions.csv')

# Colonne piégée : un numéro de dossier de litige n'est créé que lorsqu'un client
# conteste une fraude, plusieurs semaines APRÈS la transaction.
rng = np.random.default_rng(0)
df['num_dossier_litige'] = np.where(df['fraude'] == 1,
                                    rng.integers(10000, 99999, len(df)), 0)

X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

a = AutoML()
a.fit(Xtr, ytr, task='classification', metric='f1',
      time_budget=30, seed=42, verbose=0)
print('famille retenue :', a.best_estimator)

modele = a.model.estimator
if hasattr(modele, 'feature_importances_'):  # quelle variable pèse le plus ?
    imp = sorted(zip(X.columns, modele.feature_importances_), key=lambda t: -t[1])
    print('importance des variables :', [(c, round(float(v), 3)) for c, v in imp])

print('F1 en test, avec la colonne piégée :',
      round(f1_score(yte, a.predict(Xte)), 4))

Xprod = Xte.copy()
Xprod['num_dossier_litige'] = 0  # au moment de décider, aucun dossier n'existe encore
print('F1 "en production" (colonne vide) :',
      round(f1_score(yte, a.predict(Xprod), zero_division=0), 4))