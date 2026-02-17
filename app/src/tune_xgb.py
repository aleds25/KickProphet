import os
import json
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from app.src import config
except ImportError:
    import config

def load_data():
    """Carica il Training Set."""
    if not os.path.exists(config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"[-] Training set non trovato: {config.TRAIN_DATA_PATH}")

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    X = train_df.drop(config.TARGET_COL, axis=1)
    y = train_df[config.TARGET_COL]
    
    return X, y

def objective(trial, X, y):
    """Funzione obiettivo per Optuna (XGBoost)."""
    
    # Spazio di ricerca degli iperparametri XGBoost
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': 4,
        'tree_method': 'hist',  # Much faster and memory efficient
        'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params)
    
    # 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Usiamo il negative log loss come metrica di ottimizzazione
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)
    
    return scores.mean()

def main():
    print("=" * 60)
    print("  KickProphet — Ottimizzazione Iperparametri XGBoost (Optuna)")
    print("=" * 60)
    
    # 1. Caricamento dati
    try:
        X, y = load_data()
        print(f"[*] Dati caricati: {X.shape}")
    except FileNotFoundError as e:
        print(f"{e}")
        return

    # 2. Setup dello studio
    study = optuna.create_study(direction='maximize', study_name='xgb_optimization')
    
    # 3. Esecuzione ricerca
    n_trials = 20
    print(f"[>] Avvio ricerca con {n_trials} trials (Objective: Negative Log Loss)...")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    # 4. Risultati
    print("\n" + "=" * 60)
    print("  [+] RISULTATI OTTIMIZZAZIONE XGBOOST")
    print("=" * 60)
    print(f"  Miglior Negative Log Loss: {study.best_value:.4f} (Log Loss: {-study.best_value:.4f})")
    
    # 5. Salvataggio parametri
    config_dir = os.path.dirname(config.OPTUNA_XGB_PARAMS_PATH)
    os.makedirs(config_dir, exist_ok=True)
    
    with open(config.OPTUNA_XGB_PARAMS_PATH, 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"\n[+] Parametri salvati in: {config.OPTUNA_XGB_PARAMS_PATH}")

if __name__ == "__main__":
    main()
