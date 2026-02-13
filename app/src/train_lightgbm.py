import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, '..', 'data', 'processed', 'KICKSTARTER_TRAIN.csv')
    df = pd.read_csv(train_path)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def train_lightgbm(X, y):
    print(f"🚀 Inizio addestramento LightGBM... Data shape: {X.shape}")
    
    lgb = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    print("⏳ Esecuzione 5-Fold Cross-Validation per LightGBM...")
    results = cross_validate(lgb, X, y, cv=cv, scoring=scoring)
    
    print("\n--- 📊 RISULTATI LIGHTGBM CROSS-VALIDATION ---")
    cv_metrics = {}
    for metric in scoring:
        mean = np.mean(results[f'test_{metric}'])
        cv_metrics[metric] = mean
        print(f"✅ {metric.upper()}: {mean:.4f}")
    
    # Salvataggio metriche
    base_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_path, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_df = pd.DataFrame([cv_metrics])
    metrics_df.to_csv(os.path.join(results_dir, 'metrics_lightgbm.csv'), index=False)
    print(f"📄 Metriche salvate in app/results/metrics_lightgbm.csv")
    
    print("\n⏳ Addestramento finale LightGBM...")
    lgb.fit(X, y)
    return lgb

def main():
    X, y = load_data()
    model = train_lightgbm(X, y)
    
    # Feature Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance', ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, hue='feature', palette='plasma', legend=False)
    plt.title('Top 20 Feature Importance - LightGBM')
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_path, '..', 'results')
    plt.savefig(os.path.join(results_dir, 'feature_importance_lgb.png'))
    plt.close()
    
    joblib.dump(model, os.path.join(base_path, '..', 'models', 'lightgbm_model.joblib'))
    print("💾 Modello LightGBM salvato!")

if __name__ == "__main__":
    main()
