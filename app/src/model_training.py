import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score

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

def train_baseline(X, y):
    """Addestra un modello Random Forest con Cross-Validation."""
    print(f"🚀 Inizio addestramento Baseline (Random Forest)... Data shape: {X.shape}")
    
    # Configurazione Modello
    # n_jobs=-1 usa tutti i core disponibili
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced' # Gestisce eventuali piccoli sbilanciamenti
    )
    
    # Cross-Validation Strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metriche da calcolare
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    print("⏳ Esecuzione 5-Fold Cross-Validation...")
    results = cross_validate(rf, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    # Stampa Risultati Medi
    print("\n--- 📊 RISULTATI CROSS-VALIDATION (Media ± Std) ---")
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
    metrics_df.to_csv(os.path.join(results_dir, 'metrics_random_forest.csv'), index=False)
    print(f"📄 Metriche salvate in app/results/metrics_random_forest.csv")

    # Addestramento finale sull'intero training set per l'importanza delle feature
    print("\n⏳ Addestramento finale per analisi feature importance...")
    rf.fit(X, y)
    
    return rf

def plot_feature_importance(model, X):
    """Genera e salva il grafico delle feature più importanti."""
    importances = model.feature_importances_
    feature_names = X.columns
    
    # Prendi le top 20
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, hue='feature', palette='viridis', legend=False)
    plt.title('Top 20 Feature Importance - Random Forest Baseline')
    
    # Assicurati che la cartella results esista
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
    print(f"📈 Grafico feature importance salvato in app/results/feature_importance_rf.png")
    plt.close()

def main():
    try:
        # 1. Caricamento
        X, y = load_data()
        
        # 2. Training e CV
        model = train_baseline(X, y)
        
        # 3. Analisi Feature Importance
        plot_feature_importance(model, X)
        
        # 4. Salvataggio Modello
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, '..', 'models', 'baseline_rf.joblib')
        joblib.dump(model, model_path)
        print(f"💾 Modello salvato in {model_path}")
        
    except Exception as e:
        print(f"❌ Errore durante il training: {e}")

if __name__ == "__main__":
    main()
