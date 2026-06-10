import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, log_loss,
    mean_absolute_error, mean_squared_error,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../collect"))
from init_db import get_connection

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models/")
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Features
# ------------------------------------------------------------------
FEATURES = [
    "home_fifa_ranking",
    "away_fifa_ranking",
    "ranking_gap",
    "home_form5_pts",
    "home_form5_scored",
    "home_form5_conceded",
    "home_form10_pts",
    "home_form10_scored",
    "home_form10_conceded",
    "away_form5_pts",
    "away_form5_scored",
    "away_form5_conceded",
    "away_form10_pts",
    "away_form10_scored",
    "away_form10_conceded",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
    "h2h_matches",
    "neutral_venue",
    "competition_weight",
    "is_knockout",
]

TARGET_CLASS = "result_90"
TARGET_HOME  = "home_goals"
TARGET_AWAY  = "away_goals"

WALK_FORWARD_SPLITS = [
    ("2023-06-30", "2023-10-31"),
    ("2023-10-31", "2024-03-31"),
    ("2024-03-31", "2024-09-30"),
    ("2024-09-30", "2025-03-31"),
    ("2025-03-31", "2025-09-30"),
    ("2025-09-30", "2026-06-10"),
]

# ------------------------------------------------------------------
# Hyperparamètres
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
# Chargement
# ------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT *
        FROM match_features
        WHERE result_90 IS NOT NULL
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY match_date ASC
    """, conn)
    conn.close()

    print(f"  {len(df)} matchs chargés")
    print(f"  Période : {df['match_date'].min()} → {df['match_date'].max()}")
    dist = df["result_90"].value_counts().to_dict()
    total = len(df)
    print(f"  Distribution : H={dist.get('H',0)} ({dist.get('H',0)/total*100:.0f}%)"
          f"  D={dist.get('D',0)} ({dist.get('D',0)/total*100:.0f}%)"
          f"  A={dist.get('A',0)} ({dist.get('A',0)/total*100:.0f}%)")
    return df


# ------------------------------------------------------------------
# FIX : imputation sans leakage — médiane calculée sur train uniquement
# ------------------------------------------------------------------

def impute(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train = X_train.copy()
    X_test  = X_test.copy()

    medians = {}
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")
        m = X_train[col].median()
        medians[col] = m
        X_train[col] = X_train[col].fillna(m)
        X_test[col]  = X_test[col].fillna(m)

    return X_train, X_test, medians


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    weights = []
    for _, row in df.iterrows():
        home_rank = float(row.get("home_fifa_ranking") or 50)
        away_rank = float(row.get("away_fifa_ranking") or 50)
        avg_rank  = (home_rank + away_rank) / 2

        # Plus avg_rank est petit (top équipes), plus le poids est élevé
        rank_w   = 1.0 + max(0.0, (100 - avg_rank) / 50.0)
        comp_w   = float(row.get("competition_weight") or 0.5)
        weights.append(rank_w * comp_w)

    w = np.array(weights, dtype=np.float64)
    # Normaliser pour garder la somme équivalente au nombre de matchs
    w = w / w.mean()
    return w


# ------------------------------------------------------------------
# Walk-forward validation
# ------------------------------------------------------------------

def walk_forward_validation(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 55)
    print("  Walk-Forward Validation")
    print("=" * 55)

    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    clf_metrics = []

    for i, (train_end, test_end) in enumerate(WALK_FORWARD_SPLITS):
        train = df[df["match_date"] <= train_end].copy()
        test  = df[(df["match_date"] >  train_end) &
                   (df["match_date"] <= test_end)].copy()

        if len(test) < 20:
            print(f"\n  Fenêtre {i+1} : {len(test)} matchs de test — skip")
            continue

        print(f"\n  Fenêtre {i+1} : train → {train_end}  |  test → {test_end}")
        print(f"    Train : {len(train)} matchs  |  Test : {len(test)} matchs")

        # FIX : imputation sans leakage
        X_train, X_test, _ = impute(train[FEATURES], test[FEATURES])

        y_train_clf = le.transform(train[TARGET_CLASS])
        y_test_clf  = le.transform(test[TARGET_CLASS])

        # Classificateur avec sample weights
        train_weights = compute_sample_weights(train)
        clf = XGBClassifier(**CLF_PARAMS)
        clf.fit(X_train, y_train_clf,
                sample_weight=train_weights,
                eval_set=[(X_test, y_test_clf)],
                verbose=False)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        acc = accuracy_score(y_test_clf, y_pred)
        ll  = log_loss(y_test_clf, y_proba)

        # Baseline naïf : prédire toujours la classe majoritaire du train
        majority_class = np.bincount(y_train_clf).argmax()
        naive_acc = (y_test_clf == majority_class).mean()
        naive_proba = np.bincount(y_train_clf) / len(y_train_clf)
        naive_ll = log_loss(y_test_clf,
                            np.tile(naive_proba, (len(y_test_clf), 1)))

        print(f"    Classifier → Acc: {acc:.3f} (baseline: {naive_acc:.3f})"
              f"  |  LogLoss: {ll:.3f} (baseline: {naive_ll:.3f})")

        # FIX : variables séparées pour home et away
        mae_home = mae_away = rmse_home = rmse_away = None
        for target, label in [(TARGET_HOME, "home"), (TARGET_AWAY, "away")]:
            reg = XGBRegressor(**REG_PARAMS)
            reg.fit(X_train, train[target],
                    sample_weight=train_weights,
                    verbose=False)
            y_pred_reg = np.clip(reg.predict(X_test), 0, 10)
            mae  = mean_absolute_error(test[target], y_pred_reg)
            rmse = np.sqrt(mean_squared_error(test[target], y_pred_reg))
            print(f"    Regressor {label:4s} → MAE: {mae:.3f}  |  RMSE: {rmse:.3f}")
            if label == "home":
                mae_home, rmse_home = mae, rmse
            else:
                mae_away, rmse_away = mae, rmse

        clf_metrics.append({
            "accuracy": acc,
            "log_loss": ll,
            "naive_accuracy": naive_acc,
            "naive_log_loss": naive_ll,
            "mae_home":  mae_home,
            "mae_away":  mae_away,
            "n_test":    len(test),
        })

    total_test = sum(m["n_test"] for m in clf_metrics)
    w_acc  = sum(m["accuracy"]  * m["n_test"] for m in clf_metrics) / total_test
    w_ll   = sum(m["log_loss"]  * m["n_test"] for m in clf_metrics) / total_test
    w_nacc = sum(m["naive_accuracy"] * m["n_test"] for m in clf_metrics) / total_test
    w_nll  = sum(m["naive_log_loss"] * m["n_test"] for m in clf_metrics) / total_test
    w_mah  = sum(m["mae_home"]  * m["n_test"] for m in clf_metrics) / total_test
    w_maa  = sum(m["mae_away"]  * m["n_test"] for m in clf_metrics) / total_test

    print(f"\n{'─'*55}")
    print(f"  Métriques moyennes pondérées :")
    print(f"    Accuracy  : {w_acc:.3f}  (baseline: {w_nacc:.3f})"
          f"  gain: +{(w_acc - w_nacc)*100:.1f}pp")
    print(f"    Log Loss  : {w_ll:.3f}  (baseline: {w_nll:.3f})"
          f"  gain: -{(w_nll - w_ll):.3f}")
    print(f"    MAE buts  : home {w_mah:.3f}  |  away {w_maa:.3f}")
    print(f"{'─'*55}")

    return {
        "accuracy":           round(w_acc,  4),
        "log_loss":           round(w_ll,   4),
        "baseline_accuracy":  round(w_nacc, 4),
        "baseline_log_loss":  round(w_nll,  4),
        "mae_home":           round(w_mah,  4),
        "mae_away":           round(w_maa,  4),
        "n_splits":           len(clf_metrics),
        "total_test_matches": total_test,
    }


# ------------------------------------------------------------------
# Entraînement final + calibration
# ------------------------------------------------------------------

def train_final_models(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  Entraînement final (toutes données)")
    print("=" * 55)

    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    # Split calibration : 85% entraînement, 15% calibration (les plus récents)
    cutoff_idx = int(len(df) * 0.85)
    df_sorted  = df.sort_values("match_date").reset_index(drop=True)
    train_df   = df_sorted.iloc[:cutoff_idx]
    cal_df     = df_sorted.iloc[cutoff_idx:]

    print(f"  Train    : {len(train_df)} matchs (→ {train_df['match_date'].max()})")
    print(f"  Calib.   : {len(cal_df)}  matchs (→ {cal_df['match_date'].max()})")

    X_train, X_cal, medians = impute(train_df[FEATURES], cal_df[FEATURES])
    y_train = le.transform(train_df[TARGET_CLASS])
    y_cal   = le.transform(cal_df[TARGET_CLASS])

    # Classificateur de base avec sample weights
    print("\n  Entraînement XGBClassifier de base (avec sample weights)...")
    train_weights = compute_sample_weights(train_df)
    clf_base = XGBClassifier(**CLF_PARAMS)
    clf_base.fit(X_train, y_train,
                 sample_weight=train_weights,
                 verbose=False)

    # Calibration isotonique sur le hold-out temporel
    print("  Calibration isotonique (cv='prefit')...")
    clf_calibrated = CalibratedClassifierCV(clf_base, cv="prefit", method="isotonic")
    clf_calibrated.fit(X_cal, y_cal)

    # Vérification : distribution des probas avant/après calibration
    proba_raw = clf_base.predict_proba(X_cal)
    proba_cal = clf_calibrated.predict_proba(X_cal)
    classes   = ["A", "D", "H"]
    print(f"\n  Probas moyennes (test de calibration) :")
    print(f"  {'Classe':<8} {'Brut':>8} {'Calibré':>10} {'Réel':>8}")
    real_freq = np.bincount(y_cal) / len(y_cal)
    for j, cls in enumerate(classes):
        print(f"  {cls:<8} {proba_raw[:,j].mean():>8.3f}"
              f" {proba_cal[:,j].mean():>10.3f}"
              f" {real_freq[j]:>8.3f}")

    # Sauvegarde du classificateur calibré (pickle — CalibratedClassifierCV
    # n'a pas de méthode .save_model() native)
    clf_path = os.path.join(MODELS_DIR, "classifier_calibrated.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump(clf_calibrated, f)
    print(f"\n  classifier_calibrated.pkl sauvegardé")

    # Sauvegarder aussi le modèle XGBoost natif (pour SHAP, inspection)
    clf_base.save_model(os.path.join(MODELS_DIR, "classifier.json"))

    # Régresseurs sur toutes les données avec sample weights
    all_weights = compute_sample_weights(df)
    X_all = df[FEATURES].copy()
    for col in X_all.columns:
        X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
        X_all[col] = X_all[col].fillna(X_all[col].median())

    print("\n  Entraînement XGBRegressor home_goals (avec sample weights)...")
    reg_home = XGBRegressor(**REG_PARAMS)
    reg_home.fit(X_all, df[TARGET_HOME],
                 sample_weight=all_weights,
                 verbose=False)
    reg_home.save_model(os.path.join(MODELS_DIR, "regressor_home.json"))

    print("  Entraînement XGBRegressor away_goals (avec sample weights)...")
    reg_away = XGBRegressor(**REG_PARAMS)
    reg_away.fit(X_all, df[TARGET_AWAY],
                 sample_weight=all_weights,
                 verbose=False)
    reg_away.save_model(os.path.join(MODELS_DIR, "regressor_away.json"))

    # Feature importance
    fi = pd.DataFrame({
        "feature":    FEATURES,
        "importance": clf_base.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 10 features (classificateur) :")
    print(f"  {'Feature':<32} Importance")
    print(f"  {'─'*45}")
    for _, row in fi.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row['feature']:<32} {row['importance']:.4f}  {bar}")

    # Sauvegarder les métadonnées
    metadata = {
        "features":           FEATURES,
        "classes":            list(le.classes_),
        "imputation_medians": {col: float(X_all[col].median()) for col in FEATURES},
        "feature_importance": fi.to_dict(orient="records"),
        "calibration_split":  cal_df["match_date"].min(),
    }
    with open(os.path.join(MODELS_DIR, "features.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return clf_calibrated, reg_home, reg_away, le, fi


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Entraînement — WC2026 Predictor")
    print("=" * 55)

    try:
        import xgboost
        print(f"  XGBoost : {xgboost.__version__}")
    except ImportError:
        print("  XGBoost non installé — pip install xgboost scikit-learn pandas")
        sys.exit(1)

    df = load_data()

    metrics = walk_forward_validation(df)

    clf, reg_home, reg_away, le, fi = train_final_models(df)

    # Métriques finales
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  Accuracy walk-forward  : {metrics['accuracy']:.3f}"
          f"  (baseline: {metrics['baseline_accuracy']:.3f})")
    print(f"  Log Loss walk-forward  : {metrics['log_loss']:.3f}"
          f"  (baseline: {metrics['baseline_log_loss']:.3f})")
    print(f"  MAE buts : home {metrics['mae_home']:.3f}"
          f"  |  away {metrics['mae_away']:.3f}")
    print("=" * 55)
    print("\n  Modeles sauvegardes dans models/")
    print("  Prochaine etape : python predict_wc2026.py")