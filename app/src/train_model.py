import pandas as pd
import numpy as np
import os
import json
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, brier_score_loss
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
    """
    Allena il modello base LightGBM e poi lo calibra.
    Ritorna sia il modello calibrato (per predizioni) sia il base (per SHAP e analisi).
    """
    print(f"[*] Training Base LightGBM... Data shape: {X_train.shape}")
    base_model = LGBMClassifier(**params)
    base_model.fit(X_train, y_train)
    
    print("[*] Calibrating Probabilities (Isotonic Regression, CV=5)...")
    # CalibratedClassifierCV con CV allena internamente 5 modelli su fold diversi
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    print("[+] Training & Calibration complete.")
    return calibrated_model, base_model

def evaluate(model, X_test, y_test):
    print(f"[*] Evaluating with Decision Threshold: {config.DECISION_THRESHOLD:.2f}")
    
    # 1. Probabilità calibrate
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Applicazione Threshold custom
    y_pred = (y_proba >= config.DECISION_THRESHOLD).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba)
    }

    print("\n--- TEST SET METRICS ---")
    for name, value in metrics.items():
        print(f"  [+] {name.upper()}: {value:.4f}")

    print(f"\n--- CLASSIFICATION REPORT (Threshold {config.DECISION_THRESHOLD}) ---\n")
    print(classification_report(y_test, y_pred, target_names=['Failed (0)', 'Successful (1)']))
    return metrics, y_pred, y_proba

def analyze_shap(base_model, X_train, X_test):
    """Calcola e salva i grafici SHAP usando il modello base."""
    print("[*] Generating SHAP Analysis...")
    try:
        # Usa TreeExplainer per LightGBM (molto veloce)
        explainer = shap.TreeExplainer(base_model)
        
        # Calcola shap values su un campione del test set per velocità
        sample_size = min(2000, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)
        
        # Suppress specific SHAP warning for LightGBM
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
            shap_values = explainer.shap_values(X_sample)
        
        # LightGBM binary output a volte è lista, a volte array. Gestiamo entrambi.
        if isinstance(shap_values, list):
            # print(f"    [Debug] SHAP output is list of len {len(shap_values)}, selecting class 1")
            shap_values = shap_values[1] # Classe 1 (Success)
            
        # Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'SHAP Summary Plot (Top Features)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'shap_summary_final.png'), dpi=150)
        plt.close()
        print(f"[+] SHAP plots saved to {config.RESULTS_DIR}")
        
    except Exception as e:
        print(f"⚠️ SHAP Analysis failed: {e}")

def save_plots(calib_model, base_model, X_train, y_test, y_pred, y_proba):
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Failed', 'Successful']).plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'Confusion Matrix (Threshold {config.DECISION_THRESHOLD})', fontsize=13)
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

    # Calibration Curve (Reliability Diagram)
    print("[*] Generating Calibration Curve...")
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibrated Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'calibration_curve.png'), dpi=150)
    plt.close()

    # Feature Importance (dal modello BASE, non calibrato)
    importances = base_model.feature_importances_
    fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values(by='importance', ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', hue='feature', data=fi_df, palette='viridis', ax=ax, legend=False)
    ax.set_title('Feature Importance (Top 20 - Base Model)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'feature_importance_final.png'), dpi=150)
    plt.close()

# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    try:
        print("=" * 60)
        print("  KickProphet — Final Model Training & Calibration")
        print("=" * 60)
        
        # 1. Load Data
        X_train, y_train, X_test, y_test = load_data()
        
        # 2. Train (Calibrated) & Base
        params = get_model_params()
        calibrated_model, base_model = train(X_train, y_train, params)
        
        # 3. Evaluate (with Threshold)
        metrics, y_pred, y_proba = evaluate(calibrated_model, X_test, y_test)
        
        # 4. Save Model (Saving the CALIBRATED one for inference)
        joblib.dump(calibrated_model, config.MODEL_PATH)
        print(f"\n[+] Calibrated Model saved: {config.MODEL_PATH}")
        
        # 5. Save Results & Plots
        pd.DataFrame([metrics]).to_csv(os.path.join(config.RESULTS_DIR, 'metrics_final.csv'), index=False)
        save_plots(calibrated_model, base_model, X_train, y_test, y_pred, y_proba)
        
        # 6. SHAP Analysis (on Base Model)
        analyze_shap(base_model, X_train, X_test)
        
        # Save feature columns
        joblib.dump(X_train.columns.tolist(), os.path.join(config.ARTIFACTS_DIR, 'model_features.joblib'))
        print(f"[+] Feature columns saved to artifacts.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
