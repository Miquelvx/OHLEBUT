"""
Entraînement du modèle WC2026 Predictor.

Deux modèles :
  1. Classificateur XGBoost  → probabilités H/D/A (résultat à 90min)
  2. Régresseur XGBoost      → home_goals et away_goals attendus

Validation : Walk-forward sur 4 fenêtres temporelles.

Sorties :
  - models/classifier.json   : modèle de classification
  - models/regressor_home.json
  - models/regressor_away.json
  - models/features.json     : liste des features utilisées
  - models/metrics.json      : métriques de validation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, log_loss,
    mean_absolute_error, mean_squared_error
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../collect"))
from init_db import get_connection

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models/")
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Features utilisées par le modèle
# ------------------------------------------------------------------
FEATURES = [
    # Classement FIFA
    "home_fifa_ranking",
    "away_fifa_ranking",
    "ranking_gap",

    # Forme récente home
    "home_form5_pts",
    "home_form5_scored",
    "home_form5_conceded",
    "home_form10_pts",
    "home_form10_scored",
    "home_form10_conceded",

    # Forme récente away
    "away_form5_pts",
    "away_form5_scored",
    "away_form5_conceded",
    "away_form10_pts",
    "away_form10_scored",
    "away_form10_conceded",

    # Head-to-head
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",

    # Contexte
    "neutral_venue",
    "competition_weight",
    "is_knockout",
]

TARGET_CLASS  = "result_90"      # H / D / A
TARGET_HOME   = "home_goals"
TARGET_AWAY   = "away_goals"

# Fenêtres Walk-forward
# Format : (date_fin_train, date_fin_test)
# Règles :
#   - Train minimum ~12 mois avant le début du test
#   - Fenêtres de test de ~4-6 mois pour rester granulaires
#   - Dernière fenêtre va jusqu'à aujourd'hui pour couvrir 2026
WALK_FORWARD_SPLITS = [
    ("2023-06-30", "2023-10-31"),   # train: 13 mois  | test: 4 mois
    ("2023-10-31", "2024-03-31"),   # train: 17 mois  | test: 5 mois
    ("2024-03-31", "2024-09-30"),   # train: 22 mois  | test: 6 mois
    ("2024-09-30", "2025-03-31"),   # train: 28 mois  | test: 6 mois
    ("2025-03-31", "2025-09-30"),   # train: 34 mois  | test: 6 mois
    ("2025-09-30", "2026-04-30"),   # train: 40 mois  | test: 7 mois (inclut 2026)
]


# ------------------------------------------------------------------
# Chargement des données
# ------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql_query("""
        SELECT *
        FROM match_features
        WHERE result_90 IS NOT NULL
        ORDER BY match_date ASC
    """, conn)
    conn.close()

    print(f"📊 {len(df)} matchs chargés")
    print(f"   Période : {df['match_date'].min()} → {df['match_date'].max()}")
    print(f"   Distribution result_90 : {df['result_90'].value_counts().to_dict()}")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Prépare X et y depuis le DataFrame.
    Force la conversion en float et impute les valeurs manquantes par la médiane.
    """
    X = df[FEATURES].copy()

    # Forcer toutes les colonnes en numérique (corrige les types object de SQLite)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Imputation par médiane (matchs sans historique h2h ou form)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    return X


# ------------------------------------------------------------------
# Paramètres XGBoost
# ------------------------------------------------------------------

CLF_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "mlogloss",
    "random_state":     42,
    "n_jobs":           -1,
}

REG_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "objective":        "reg:squarederror",
    "random_state":     42,
    "n_jobs":           -1,
}


# ------------------------------------------------------------------
# Walk-forward validation
# ------------------------------------------------------------------

def walk_forward_validation(df: pd.DataFrame) -> dict:
    """
    Évalue le modèle sur 4 fenêtres temporelles.
    Retourne les métriques agrégées.
    """
    print("\n" + "="*55)
    print("  Walk-Forward Validation")
    print("="*55)

    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    clf_metrics = []
    reg_metrics = []

    for i, (train_end, test_end) in enumerate(WALK_FORWARD_SPLITS):
        train_start = df["match_date"].min()

        train = df[df["match_date"] <= train_end].copy()
        test  = df[(df["match_date"] > train_end) &
                   (df["match_date"] <= test_end)].copy()

        if len(test) == 0:
            print(f"\n  Fenêtre {i+1} : pas de données de test, skip")
            continue

        print(f"\n  Fenêtre {i+1} : train → {train_end}  |  test {train_end[:7]} → {test_end[:7]}")
        print(f"    Train : {len(train)} matchs  |  Test : {len(test)} matchs")

        X_train = prepare_features(train)
        X_test  = prepare_features(test)

        # --- Classificateur ---
        y_train_clf = le.transform(train[TARGET_CLASS])
        y_test_clf  = le.transform(test[TARGET_CLASS])

        clf = XGBClassifier(**CLF_PARAMS)
        clf.fit(X_train, y_train_clf,
                eval_set=[(X_test, y_test_clf)],
                verbose=False)

        y_pred_clf   = clf.predict(X_test)
        y_proba_clf  = clf.predict_proba(X_test)
        acc          = accuracy_score(y_test_clf, y_pred_clf)
        ll           = log_loss(y_test_clf, y_proba_clf)

        print(f"    Classifier → Accuracy: {acc:.3f}  |  LogLoss: {ll:.3f}")
        clf_metrics.append({"accuracy": acc, "log_loss": ll,
                            "n_test": len(test)})

        # --- Régresseur buts ---
        for target, label in [(TARGET_HOME, "home"), (TARGET_AWAY, "away")]:
            reg = XGBRegressor(**REG_PARAMS)
            reg.fit(X_train, train[target], verbose=False)
            y_pred_reg = reg.predict(X_test)
            y_pred_reg = np.clip(np.round(y_pred_reg), 0, 10)
            mae  = mean_absolute_error(test[target], y_pred_reg)
            rmse = np.sqrt(mean_squared_error(test[target], y_pred_reg))
            print(f"    Regressor {label:4s} → MAE: {mae:.3f}  |  RMSE: {rmse:.3f}")

        reg_metrics.append({"mae_home": mae, "rmse_home": rmse,
                            "n_test": len(test)})

    # Métriques agrégées (moyenne pondérée par nombre de matchs)
    total_test = sum(m["n_test"] for m in clf_metrics)
    avg_acc = sum(m["accuracy"] * m["n_test"] for m in clf_metrics) / total_test
    avg_ll  = sum(m["log_loss"] * m["n_test"] for m in clf_metrics) / total_test

    print(f"\n{'─'*55}")
    print(f"  Métriques moyennes (pondérées) :")
    print(f"    Accuracy  : {avg_acc:.3f}  (baseline naïf ~0.46)")
    print(f"    Log Loss  : {avg_ll:.3f}  (baseline naïf ~1.10)")
    print(f"{'─'*55}")

    return {
        "accuracy":  round(avg_acc, 4),
        "log_loss":  round(avg_ll, 4),
        "n_splits":  len(clf_metrics),
        "total_test_matches": total_test,
    }


# ------------------------------------------------------------------
# Entraînement final sur toutes les données
# ------------------------------------------------------------------

def train_final_models(df: pd.DataFrame):
    """
    Entraîne les modèles finaux sur l'ensemble du dataset.
    Sauvegarde les modèles et les métriques de feature importance.
    """
    print("\n" + "="*55)
    print("  Entraînement final (toutes données)")
    print("="*55)

    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    X = prepare_features(df)
    y_clf  = le.transform(df[TARGET_CLASS])
    y_home = df[TARGET_HOME]
    y_away = df[TARGET_AWAY]

    # Classificateur
    print("\n  Entraînement classificateur H/D/A...")
    clf = XGBClassifier(**CLF_PARAMS)
    clf.fit(X, y_clf, verbose=False)
    clf.save_model(os.path.join(MODELS_DIR, "classifier.json"))
    print("  ✅ classifier.json sauvegardé")

    # Régresseur home_goals
    print("  Entraînement régresseur home_goals...")
    reg_home = XGBRegressor(**REG_PARAMS)
    reg_home.fit(X, y_home, verbose=False)
    reg_home.save_model(os.path.join(MODELS_DIR, "regressor_home.json"))
    print("  ✅ regressor_home.json sauvegardé")

    # Régresseur away_goals
    print("  Entraînement régresseur away_goals...")
    reg_away = XGBRegressor(**REG_PARAMS)
    reg_away.fit(X, y_away, verbose=False)
    reg_away.save_model(os.path.join(MODELS_DIR, "regressor_away.json"))
    print("  ✅ regressor_away.json sauvegardé")

    # Feature importance
    fi = pd.DataFrame({
        "feature":   FEATURES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Feature importance (classificateur) :")
    print(f"  {'Feature':<30} {'Importance':>10}")
    print(f"  {'─'*42}")
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<30} {row['importance']:>10.4f}  {bar}")

    # Sauvegarder les métadonnées
    metadata = {
        "features":      FEATURES,
        "classes":       list(le.classes_),
        "feature_importance": fi.to_dict(orient="records"),
    }
    with open(os.path.join(MODELS_DIR, "features.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return clf, reg_home, reg_away, le


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Entraînement — WC2026 Predictor")
    print("=" * 55)

    # Vérifier que XGBoost est installé
    try:
        import xgboost
        print(f"  XGBoost version : {xgboost.__version__}")
    except ImportError:
        print("❌ XGBoost non installé — lance : pip install xgboost")
        exit(1)

    df = load_data()

    # Walk-forward validation
    metrics = walk_forward_validation(df)

    # Entraînement final
    clf, reg_home, reg_away, le = train_final_models(df)

    # Sauvegarder les métriques
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n🎉 Entraînement terminé !")
    print(f"   Modèles sauvegardés dans : models/")
    print(f"   Accuracy walk-forward    : {metrics['accuracy']:.3f}")
    print(f"   Log Loss walk-forward    : {metrics['log_loss']:.3f}")