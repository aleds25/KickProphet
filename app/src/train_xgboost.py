import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

def load_data():
    """Carica il dataset di training processato."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, '..', 'data', 'processed', 'KICKSTARTER_TRAIN.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ Errore: File non trovato in {train_path}. Esegui prima preprocessing.py")
    
    df = pd.read_csv(train_path)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def train_xgboost(X, y):
    """Addestra un modello XGBoost con Cross-Validation."""
    print(f"🚀 Inizio addestramento XGBoost... Data shape: {X.shape}")
    
    # Configurazione Modello
    # Parametri iniziali robusti per Gradient Boosting
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # Cross-Validation Strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metriche da calcolare
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    print("⏳ Esecuzione 5-Fold Cross-Validation per XGBoost...")
    results = cross_validate(xgb, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    # Stampa Risultati Medi
    print("\n--- 📊 RISULTATI XGBOOST CROSS-VALIDATION (Media ± Std) ---")
    cv_metrics = {}
    for metric in scoring:
        mean = np.mean(results[f'test_{metric}'])
        std = np.std(results[f'test_{metric}'])
        cv_metrics[metric] = mean
        print(f"✅ {metric.upper()}: {mean:.4f} ± {std:.4f}")
    
    # Salvataggio metriche
    base_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_path, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_df = pd.DataFrame([cv_metrics])
    metrics_df.to_csv(os.path.join(results_dir, 'metrics_xgboost.csv'), index=False)
    print(f"📄 Metriche salvate in app/results/metrics_xgboost.csv")

    # Addestramento finale
    print("\n⏳ Addestramento finale XGBoost sull'intero training set...")
    xgb.fit(X, y)
    
    return xgb

def plot_feature_importance(model, X):
    """Genera e salva il grafico delle feature più importanti per XGBoost."""
    importances = model.feature_importances_
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, hue='feature', palette='magma', legend=False)
    plt.title('Top 20 Feature Importance - XGBoost')
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, 'feature_importance_xgb.png'))
    print(f"📈 Grafico feature importance salvato in app/results/feature_importance_xgb.png")
    plt.close()

def main():
    try:
        # 1. Caricamento
        X, y = load_data()
        
        # 2. Training e CV
        model = train_xgboost(X, y)
        
        # 3. Analisi Importance
        plot_feature_importance(model, X)
        
        # 4. Salvataggio
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, '..', 'models', 'xgboost_model.joblib')
        joblib.dump(model, model_path)
        print(f"💾 Modello XGBoost salvato in {model_path}")
        
    except Exception as e:
        print(f"❌ Errore durante il training XGBoost: {e}")

if __name__ == "__main__":
    main()
