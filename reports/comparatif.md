# Comparatif des approches — projet fraude (TP04)

> **Date de consultation des sources cloud :** 2026-10-22

## Tableau comparatif

| Approche | F1 test | Temps humain | Temps machine | Coût estimé (source, date) | Explicabilité | Export / réversibilité |
|---|---|---|---|---|---|---|
| Logistique (réf.) | 0.7016 | ~5 min | 0.11 s | 0 € (poste local) | Maximale (coeff. linéaires) | Total (joblib, ONNX, PMML) |
| Manuel RF-500-16 | 0.7893 | ~30 min | 4.8 s | 0 € (poste local) | Moyenne (feature importance) | Total (joblib, ONNX) |
| FLAML 60 s | 0.8582 | ~5 min | 60.2 s | 0 € (poste local) | Moyenne/Basse (boîte noire) | Total (pickle, ONNX) |
| Vertex AI AutoML | non mesuré | ~1 h (UI) | 1–8 h (budget max.) | ~21,25 $/nœud-heure ; endpoint ≈ 0,50 $/h (cloud.google.com/vertex-ai, 2026-10-22) | Bonne (Sampled Shapley, Vertex Explainable AI, dashboard Console) | Partielle — export TF SavedModel ; explainabilité désactivée hors cloud ; vendor lock-in fort |
| SageMaker Autopilot | non mesuré | ~1 h (Canvas) | 1–4 h (jobs parallèles) | Session Canvas 1,90 $/h + EC2 (ml.m5.xlarge ≈ 0,23 $/h × jobs) (aws.amazon.com/sagemaker/pricing, 2026-10-22) | Bonne (rapport SHAP + notebooks générés automatiquement, SageMaker Clarify) | Partielle — model.tar.gz récupérable sur S3 ; conversion ONNX manuelle ; vendor lock-in AWS |
| Azure Automated ML | non mesuré | ~1 h (Studio) | 1–4 h (cluster compute) | VM CPU DSv2 ≈ 0,20–1,20 $/h/nœud ; aucun surcoût logiciel AutoML (azure.microsoft.com/pricing/machine-learning, 2026-10-22) | Bonne (SHAP + InterpretML, rapport Studio, data slices) | Partielle — export MLflow ou ONNX ; portabilité via MLflow open-source ; vendor lock-in modéré |

## Notes de mesure (blocs A et B)

- **Logistique** : F1 = 0.7016, entraînement 0.11 s (local, `sklearn`, split 75/25, random_state=42)
- **RF manuel (500 arbres, max_depth=16)** : F1 = 0.7893, entraînement 4.8 s (local, `sklearn`)
- **FLAML 60 s** : F1 = 0.8582, wall-clock 60.2 s, meilleur estimateur XGBoost (val_loss ≈ 0.176)
- Tous les runs sont tracés dans MLflow (expérience `tp4-automl`, `mlflow.db` exclu du dépôt Git)

## Points de départ documentaires (Étape E1)

### Google Vertex AI AutoML Tabulaire
- **Compte / prérequis** : compte Google Cloud + facturation activée + quota nœuds dans la région ; crédits d'essai 300 $.
- **Tarification entraînement** : 21,252 $/nœud-heure (classification tabulaire) ; budget spécifié en heures ; facturation par tranches de 30 s. Endpoint déployé : ≈ 0,50 $/h (n1-standard-4).
- **Explicabilité** : Vertex Explainable AI — Sampled Shapley (global + local) ; dashboard Console ; uniquement disponible via endpoint cloud (désactivé sur export local).
- **Export / réversibilité** : TF SavedModel exportable pour modèles classification/régression ; architecture x86+AVX requise ; explainabilité perdue hors cloud → vendor lock-in fort.
- **Journalisation** : Vertex ML Metadata (essais, hyperparamètres, datasets, métriques), Model Registry intégré, Cloud Logging pour endpoints.

### Amazon SageMaker Autopilot (intégré Canvas)
- **Compte / prérequis** : compte AWS + IAM + S3 bucket ; Canvas App nécessite SageMaker Studio Domain.
- **Tarification entraînement** : session Canvas 1,90 $/h (workspace actif) + instances EC2 de training (ml.m5.xlarge ≈ 0,23 $/h × N jobs parallèles × durée). Endpoint temps-réel : tarif instance d'inférence.
- **Explicabilité** : rapport SHAP automatique ; candidate notebooks Jupyter générés ; SageMaker Clarify pour biais et feature importance ; disponible dans la console AWS.
- **Export / réversibilité** : artefact `model.tar.gz` téléchargeable depuis S3 ; contient pipeline sklearn/XGBoost ; conversion ONNX manuelle (onnxmltools) ; vendor lock-in AWS modéré.
- **Journalisation** : SageMaker Experiments (configurations, métriques), S3 pour artefacts, CloudWatch Logs pour jobs, Model Registry SageMaker.

### Azure Automated ML
- **Compte / prérequis** : abonnement Azure + workspace Azure ML + compute cluster ; essai gratuit avec crédits.
- **Tarification entraînement** : coût VM uniquement, aucun surcoût logiciel AutoML. CPU DSv2 : 0,20–1,20 $/h. GPU T4 (NC4as) : ≈ 0,53 $/h. Spot/low-priority : −60 à 90 %.
- **Explicabilité** : feature importance global dans Studio ; dashboard SHAP (InterpretML) ; rapport de métriques par coupe de données (data slices).
- **Export / réversibilité** : export MLflow model (pickle + conda) ou ONNX si supporté ; utilisable avec `mlflow.pyfunc.load_model()` hors Azure ; vendor lock-in modéré.
- **Journalisation** : Azure ML Experiments (runs, hyperparamètres, datasets), Model Registry AzureML, Azure Monitor pour endpoints, versionnage de dataset.

---

## Commentaire — Quand recommander chaque approche ?

1. **Régression logistique (référence)** : recommandée quand l'explicabilité est impérative (réglementation, audit métier) et que la donnée est tabulaire linéairement séparable ; coût nul, déploiement trivial, idéale comme baseline avant tout investissement ML.

2. **Random Forest manuel (TP2)** : pertinente quand on dispose d'un expert ML capable de tuner les hyperparamètres et qu'un contrôle total sur le modèle est requis (RGPD, export sans contrainte cloud) ; bon compromis entre performance (+7 pts F1 vs. logistique) et transparence.

3. **FLAML 60 s** : à privilégier pour les équipes avec un budget temps réduit qui veulent une performance supérieure sans expertise ML approfondie ; s'intègre dans un pipeline CI/CD local ou on-premise ; choix optimal quand les données sont sensibles et ne peuvent pas quitter l'infrastructure interne.

4. **Vertex AI / SageMaker Autopilot / Azure Automated ML** : recommandées pour les organisations déjà investies dans un écosystème cloud, traitant de grands volumes nécessitant des GPU, et acceptant le modèle pay-as-you-go ; idéales pour des POC rapides sans équipe Data Science dédiée ; à éviter si la maîtrise des données et la sortie du cloud sont des contraintes fortes (coût et vendor lock-in).

5. **Choisir entre les trois plateformes cloud** : préférer **Azure AutoML** si l'organisation utilise Microsoft 365/Azure AD (RBAC, SSO) et que la portabilité MLflow est prioritaire ; **Vertex AI** si BigQuery est la source de données principale ; **SageMaker** si l'infrastructure AWS est dominante et que l'intégration S3/Lambda/Redshift est requise pour le pipeline.
