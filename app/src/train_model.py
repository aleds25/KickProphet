"""
train_model.py — Script di produzione per il modello finale.

Addestra il modello vincitore (LightGBM) sull'intero Training Set,
valuta sul Test Set e salva modello, metriche e grafici.

Uso:
    python app/src/train_model.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)


# ──────────────────────────────────────────────
#  CONFIGURAZIONE
# ──────────────────────────────────────────────

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, '..', 'data', 'processed')
MODELS_DIR = os.path.join(BASE_PATH, '..', 'models')
RESULTS_DIR = os.path.join(BASE_PATH, '..', 'results')

MODEL_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}


# ──────────────────────────────────────────────
#  FUNZIONI
# ──────────────────────────────────────────────

def load_data():
    """Carica Training Set e Test Set."""
    train_path = os.path.join(DATA_DIR, 'KICKSTARTER_TRAIN.csv')
    test_path = os.path.join(DATA_DIR, 'KICKSTARTER_TEST.csv')

    for path, name in [(train_path, 'TRAIN'), (test_path, 'TEST')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ File {name} non trovato: {path}\n"
                f"   Esegui prima preprocessing.py"
            )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    return X_train, y_train, X_test, y_test


def train(X_train, y_train):
    """Addestra LightGBM sull'intero Training Set."""
    print(f"🚀 Addestramento LightGBM... Data shape: {X_train.shape}")

    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    print("✅ Addestramento completato.")
    return model


def evaluate(model, X_test, y_test):
    """Valuta il modello sul Test Set e restituisce metriche."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    print("\n--- 📊 METRICHE SUL TEST SET ---")
    for name, value in metrics.items():
        print(f"  ✅ {name.upper()}: {value:.4f}")

    print(f"\n--- 📋 CLASSIFICATION REPORT ---\n")
    print(classification_report(y_test, y_pred, target_names=['Failed (0)', 'Successful (1)']))

    return metrics, y_pred, y_proba


def save_confusion_matrix(y_test, y_pred):
    """Salva la Confusion Matrix come immagine."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Failed', 'Successful'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title('Confusion Matrix — LightGBM (Test Set)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix_final.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📈 Confusion Matrix salvata: {output_path}")


def save_roc_curve(y_test, y_proba):
    """Salva la ROC Curve come immagine."""
    fig, ax = plt.subplots(figsize=(7, 6))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    ax.plot(fpr, tpr, color='#4C72B0', linewidth=2.5, label=f'LightGBM (AUC = {auc_val:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#4C72B0')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — LightGBM (Test Set)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'roc_curve_final.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📈 ROC Curve salvata: {output_path}")


def save_feature_importance(model, feature_names):
    """Salva il grafico Feature Importance (Top 20)."""
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values(by='importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=fi_df, hue='feature', palette='viridis', legend=False, ax=ax)
    ax.set_title('Top 20 Feature Importance — LightGBM', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.set_ylabel('')
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'feature_importance_final.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📈 Feature Importance salvata: {output_path}")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    # Crea cartelle output
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # 1. Caricamento
        print("=" * 60)
        print("  KickProphet — Training Modello Finale (LightGBM)")
        print("=" * 60)
        X_train, y_train, X_test, y_test = load_data()
        print(f"📊 Training Set: {X_train.shape}")
        print(f"📊 Test Set:     {X_test.shape}")

        # 2. Training
        model = train(X_train, y_train)

        # 3. Valutazione
        metrics, y_pred, y_proba = evaluate(model, X_test, y_test)

        # 4. Salvataggio Modello
        model_path = os.path.join(MODELS_DIR, 'final_model.joblib')
        joblib.dump(model, model_path)
        print(f"\n💾 Modello salvato: {model_path}")

        # 5. Salvataggio Metriche
        metrics_path = os.path.join(RESULTS_DIR, 'metrics_final.csv')
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"📄 Metriche salvate: {metrics_path}")

        # 6. Salvataggio Grafici
        save_confusion_matrix(y_test, y_pred)
        save_roc_curve(y_test, y_proba)
        save_feature_importance(model, X_train.columns)

        print("\n" + "=" * 60)
        print("  ✅ Pipeline completata con successo!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Errore: {e}")
        raise


if __name__ == "__main__":
    main()
