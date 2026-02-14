import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

try:
    from app.src import config
except ImportError:
    import config

# ──────────────────────────────────────────────
#  FUNZIONI
# ──────────────────────────────────────────────

def load_data():
    """Carica Training Set e Test Set processati."""
    if not os.path.exists(config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"File not found: {config.TRAIN_DATA_PATH}. Run preprocessing.py first.")

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    X_train = train_df.drop(config.TARGET_COL, axis=1)
    y_train = train_df[config.TARGET_COL]
    X_test = test_df.drop(config.TARGET_COL, axis=1)
    y_test = test_df[config.TARGET_COL]

    return X_train, y_train, X_test, y_test

def get_model_params():
    """Carica i parametri ottimali se esistono, altrimenti usa i default."""
    final_params = config.LGBM_DEFAULT_PARAMS.copy()
    
    if os.path.exists(config.OPTUNA_PARAMS_PATH):
        print(f"📖 Loaded optimized parameters from: {config.OPTUNA_PARAMS_PATH}")
        with open(config.OPTUNA_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
            final_params.update(best_params)
    else:
        print("💡 Using default parameters.")
        
    return final_params

def train(X_train, y_train, params):
    print(f"[*] Training LightGBM... Data shape: {X_train.shape}")
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    print("[+] Training complete.")
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    print("\n--- TEST SET METRICS ---")
    for name, value in metrics.items():
        print(f"  [+] {name.upper()}: {value:.4f}")

    print(f"\n--- CLASSIFICATION REPORT ---\n")
    print(classification_report(y_test, y_pred, target_names=['Failed (0)', 'Successful (1)']))
    return metrics, y_pred, y_proba

def save_plots(model, X_train, y_test, y_pred, y_proba):
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Failed', 'Successful']).plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title('Confusion Matrix — LightGBM', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix_final.png'), dpi=150)
    plt.close()

    # ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.4f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('ROC Curve', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'roc_curve_final.png'), dpi=150)
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values(by='importance', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', hue='feature', data=fi_df, palette='viridis', ax=ax, legend=False)
    ax.set_title('Feature Importance (Top 20)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'feature_importance_final.png'), dpi=150)
    plt.close()

# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    try:
        print("=" * 60)
        print("  KickProphet — Final Model Training")
        print("=" * 60)
        
        # 1. Load Data
        X_train, y_train, X_test, y_test = load_data()
        
        # 2. Train
        params = get_model_params()
        model = train(X_train, y_train, params)
        
        # 3. Evaluate
        metrics, y_pred, y_proba = evaluate(model, X_test, y_test)
        
        # 4. Save Model
        joblib.dump(model, config.MODEL_PATH)
        print(f"\n[+] Model saved: {config.MODEL_PATH}")
        
        # 5. Save Results
        pd.DataFrame([metrics]).to_csv(os.path.join(config.RESULTS_DIR, 'metrics_final.csv'), index=False)
        save_plots(model, X_train, y_test, y_pred, y_proba)
        
        # Save feature columns to ensure alignment during prediction
        joblib.dump(X_train.columns.tolist(), os.path.join(config.ARTIFACTS_DIR, 'model_features.joblib'))
        print(f"[+] Feature columns saved to artifacts.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
