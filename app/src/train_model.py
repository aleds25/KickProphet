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
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    ConfusionMatrixDisplay, brier_score_loss
)
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

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
    lgbm_params = config.LGBM_DEFAULT_PARAMS.copy()
    xgb_params = config.XGB_DEFAULT_PARAMS.copy()
    
    # Load LGBM Params
    if os.path.exists(config.OPTUNA_PARAMS_PATH):
        print(f"📖 Loaded optimized LGBM parameters from: {config.OPTUNA_PARAMS_PATH}")
        with open(config.OPTUNA_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
            lgbm_params.update(best_params)
    else:
        print("💡 Using default LGBM parameters.")

    # Load XGB Params
    if os.path.exists(config.OPTUNA_XGB_PARAMS_PATH):
        print(f"📖 Loaded optimized XGB parameters from: {config.OPTUNA_XGB_PARAMS_PATH}")
        with open(config.OPTUNA_XGB_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
            # Remove deprecated param just in case
            if 'use_label_encoder' in best_params:
                del best_params['use_label_encoder']
            xgb_params.update(best_params)
    else:
        print("💡 Using default XGB parameters.")
        
    return lgbm_params, xgb_params

def optimize_threshold(y_true, y_proba, beta=0.5):
    """
    Finds the decision threshold that maximizes F-beta Score.
    beta < 1 favors Precision (e.g. 0.5).
    beta > 1 favors Recall (e.g. 2.0).
    Default beta=0.5 prioritizes avoiding False Positives (wasting money).
    """
    print(f"[*] Optimizing Decision Threshold (Metric: F{beta}-Score)...")
    thresholds = np.arange(0.3, 0.8, 0.01)
    scores = []
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta)
        scores.append(score)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        
    best_idx = np.argmax(scores)
    best_thresh = thresholds[best_idx]
    best_score = scores[best_idx]
    
    print(f"    [+] Best Threshold: {best_thresh:.2f}")
    print(f"        -> F{beta}-Score: {best_score:.4f}")
    print(f"        -> Precision: {precisions[best_idx]:.4f}")
    print(f"        -> Recall:    {recalls[best_idx]:.4f}")
    return best_thresh

def train_ensemble(X_train, y_train, lgbm_params, xgb_params):
    """
    Trains an Ensemble of LightGBM and XGBoost.
    Returns the Calibrated VotingClassifier.
    """
    print(f"[*] Training Ensemble (LGBM + XGB)... Data shape: {X_train.shape}")
    
    # 1. Define Base Models
    lgbm_clf = LGBMClassifier(**lgbm_params)
    xgb_clf = XGBClassifier(**xgb_params)
    
    # 2. Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[('lgbm', lgbm_clf), ('xgb', xgb_clf)],
        voting='soft'
    )
    
    # 3. Fit
    voting_clf.fit(X_train, y_train)
    
    # 4. Calibrate
    print("[*] Calibrating Ensemble (Isotonic)...")
    calibrated_model = CalibratedClassifierCV(voting_clf, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    print("[+] Ensemble Training & Calibration complete.")
    return calibrated_model, voting_clf

def evaluate(model, X_test, y_test, threshold=0.5):
    print(f"[*] Evaluating with Decision Threshold: {threshold:.2f}")
    
    # 1. Probabilità calibrate
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Applicazione Threshold custom
    y_pred = (y_proba >= threshold).astype(int)

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

    print(f"\n--- CLASSIFICATION REPORT (Threshold {threshold:.2f}) ---\n")
    print(classification_report(y_test, y_pred, target_names=['Failed (0)', 'Successful (1)']))
    return metrics, y_pred, y_proba

def analyze_shap(voting_model, X_train, X_test):
    """Calcola e salva i grafici SHAP per ogni modello nell'Ensemble."""
    print("[*] Generating SHAP Analysis for Ensemble...")
    
    # Campione per velocità 
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    
    # Fix: Use zip to pair names from .estimators with fitted models from .estimators_
    for (name, _), model in zip(voting_model.estimators, voting_model.estimators_):
        print(f"    [*] Analyzing SHAP for: {name}...")
        try:
            # SHAP Explainer
            explainer = shap.TreeExplainer(model)
            
            # Suppress specific warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
                shap_values = explainer.shap_values(X_sample)

            # Gestione output SHAP (lista vs array)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Classe 1
            
            # Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f'SHAP Summary Plot - {name}', fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(config.RESULTS_DIR, f'shap_summary_{name}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"        [+] Saved: {save_path}")
            
        except Exception as e:
            print(f"        ⚠️ SHAP failed for {name}: {e}")

def save_plots(calib_model, base_model, X_train, y_test, y_pred, y_proba):
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Failed', 'Successful']).plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'Confusion Matrix', fontsize=13)
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

    # Feature Importance (LGBM)
    try:
        lgbm_model = base_model.estimators_[0]
        importances = lgbm_model.feature_importances_
        fi_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values(by='importance', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', hue='feature', data=fi_df, palette='viridis', ax=ax, legend=False)
        ax.set_title('Feature Importance (Top 20 - LGBM)', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'feature_importance_final.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"⚠️ Feature Importance failed for LGBM: {e}")
    
    

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
        
        # 2. Train Ensemble
        lgbm_params, xgb_params = get_model_params()
        calibrated_model, base_voting_model = train_ensemble(X_train, y_train, lgbm_params, xgb_params)
        
        # 3. Find Best Threshold (CROSS-VALIDATION)
        # We generate "clean" predictions on training data to optimize threshold without overfitting.
        print("[*] Generating CV predictions for threshold optimization...")
        y_cv_proba = cross_val_predict(
            calibrated_model, 
            X_train, 
            y_train, 
            cv=5, 
            method='predict_proba', 
            n_jobs=-1
        )[:, 1]
        
        best_threshold = optimize_threshold(y_train, y_cv_proba, beta=0.7) # Optimize for F0.7 (Milder Precision focus)     
        # Evaluate on Test Set
        metrics, y_pred, y_proba = evaluate(calibrated_model, X_test, y_test, threshold=best_threshold)
        
        # 4. Save Model
        joblib.dump(calibrated_model, config.MODEL_PATH)
        print(f"\n[+] Calibrated Ensemble Model saved: {config.MODEL_PATH}")
        
        # 5. Save Results & Plots
        pd.DataFrame([metrics]).to_csv(os.path.join(config.RESULTS_DIR, 'metrics_final.csv'), index=False)
        save_plots(calibrated_model, base_voting_model, X_train, y_test, y_pred, y_proba)
        
        # 6. SHAP Analysis (Ensemble)
        # analyze_shap(base_voting_model, X_train, X_test)
        
        # Save feature columns
        joblib.dump(X_train.columns.tolist(), os.path.join(config.ARTIFACTS_DIR, 'model_features.joblib'))
        print(f"[+] Feature columns saved to artifacts.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
