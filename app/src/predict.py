import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from app.src import config
from datetime import datetime
from app.src import config
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.src.preprocessing import clean_text, get_nlp_features, get_embeddings, cyclical_encoding

import requests

# Fallback Currency Rates (Last Updated: Feb 2026)
# Used only if API fails.
STATIC_FX_RATES = {
    'USD': 1.00, 'EUR': 1.08, 'GBP': 1.27, 'CAD': 0.74, 'AUD': 0.65, 'JPY': 0.0067,
    'CHF': 1.14, 'SEK': 0.096, 'DKK': 0.14, 'NOK': 0.095, 'HKD': 0.13, 'MXN': 0.058,
    'SGD': 0.74, 'NZD': 0.61
}

def get_latest_usd_rates():
    """Fetches real-time exchange rates to USD. Returns dictionary and source 'LIVE' or 'STATIC'."""
    try:
        # Using a free, reliable public API
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
        if response.status_code == 200:
            data = response.json()
            # The API returns rates relative to USD (e.g. USD -> EUR = 0.92)
            # We need the inverse (Value in USD) e.g. 1 EUR = 1.08 USD
            rates_to_usd = {k: 1/v for k, v in data['rates'].items() if v > 0}
            print(f"[*] Currency API Success. Rates Updated: {data.get('date')}")
            return rates_to_usd, "LIVE"
    except Exception as e:
        print(f"[!] Currency API Failed ({e}). Using Backup Static Rates.")
    
    return STATIC_FX_RATES, "STATIC"

def load_system():
    """Loads the trained model and all feature engineering artifacts."""
    print("[*] Loading System Artifacts...")
    artifacts = {}
    
    # Load Model
    artifacts['model'] = joblib.load(config.MODEL_PATH)
    
    # Load Feature Engineering Artifacts
    artifacts['pca'] = joblib.load(os.path.join(config.ARTIFACTS_DIR, 'pca_model.joblib'))
    artifacts['tfidf'] = joblib.load(os.path.join(config.ARTIFACTS_DIR, 'tfidf_vectorizer.joblib'))
    artifacts['encodings'] = joblib.load(os.path.join(config.ARTIFACTS_DIR, 'target_encodings.joblib'))
    
    # Load Feature List (to ensure correct column order)
    artifacts['features'] = joblib.load(os.path.join(config.ARTIFACTS_DIR, 'model_features.joblib'))
    
    # Load NLP Model for Coherence Check (Predict-time only)
    # We could pickle this, but loading fresh is safer for version matches
    artifacts['nlp_model'] = SentenceTransformer(config.NLP_MODEL_NAME)

    print("[+] System Loaded.")
    return artifacts

def prepare_single_sample(data, artifacts):
    """
    Transforms a single project dictionary into a model-ready DataFrame row.
    Expected keys: name, blurb, goal, duration_days, category, sub_category, country, currency
    """
    # 1. Create Initial DataFrame
    df = pd.DataFrame([data])
    
    # 2. Basic Features
    df['goal'] = pd.to_numeric(df['goal'])
    
    
    # Calculate goal_usd using live FX rate
    currency = data.get('currency', 'USD')
    
    # Determine Rate strategy
    # Optimally, we would cache this or pass it in, but for a script fetching once is fine.
    rates, source = get_latest_usd_rates()
    rate = rates.get(currency, 1.0)
    
    if currency != 'USD':
        print(f"    -> Converting {currency} to USD using {source} rate: {rate:.4f}")

    df['goal_usd'] = df['goal'] * rate
    df['goal_usd_log'] = np.log1p(df['goal_usd'])
    df['duration_days'] = pd.to_numeric(df['duration_days'], errors='coerce')
    goal_usd = df['goal_usd'].iloc[0]
    duration = df['duration_days'].iloc[0]
    df['goal_per_day'] = goal_usd / (duration if duration > 0 else 1)
    
    # Optional Features (defaults if not provided)
    df['has_video'] = int(data.get('has_video', False))
    df['prelaunch_activated'] = int(data.get('prelaunch_activated', False))
    
    prep_days = float(data.get('preparation_days', 0))
    df['preparation_days_log'] = np.log1p(prep_days)
    
    # Text Features
    df['full_text'] = df['name'] + " " + df['blurb']
    df['name_len'] = len(str(df['name'].iloc[0]))
    df['blurb_len'] = len(str(df['blurb'].iloc[0]))
    df['name_word_count'] = len(str(df['name'].iloc[0]).split())
    df['blurb_word_count'] = len(str(df['blurb'].iloc[0]).split())
    df['name_word_count'] = len(str(df['name'].iloc[0]).split())
    df['blurb_word_count'] = len(str(df['blurb'].iloc[0]).split())
    df['name_is_upper'] = 1 if str(df['name'].iloc[0]).isupper() else 0
    
    # NLP: Sentiment & Readability & Garbage
    pol, sub, read, avg_len, dig_ratio = get_nlp_features(df['full_text'].iloc[0])
    df['sentiment_polarity'] = pol
    df['sentiment_subjectivity'] = sub
    df['readability_score'] = read
    df['avg_word_len'] = avg_len
    df['digit_ratio'] = dig_ratio
    
    # NLP: Embeddings & Coherence
    # We need to generate embeddings using the simpler get_embeddings or the artifact model
    # To ensure consistency with Coherence, let's use the artifact model
    nlp_model = artifacts['nlp_model']
    text_embedding = nlp_model.encode(df['full_text'].fillna("").iloc[0], normalize_embeddings=True)
    
    # Coherence Score
    sub_cat = df['sub_category'].iloc[0]
    cat_embedding = nlp_model.encode(str(sub_cat), normalize_embeddings=True)
    
    # Cosine Similarity (1D arrays need reshape)
    coherence = cosine_similarity([text_embedding], [cat_embedding])[0][0]
    df['desc_cat_similarity'] = coherence

    # --- SEMANTIC GATING & CLAMPING FOR PREPARATION DAYS ---
    raw_prep_days = float(data.get('preparation_days', 0))
    clamped_days = min(raw_prep_days, 60)
    
    # Semantic Gating: If the description is irrelevant (low coherence), revoke the bonus.
    if coherence < 0.17:
        print(f"    -> [!] Detected Low Quality/Irrelevant Text (Coherence: {coherence:.2f}). Ignoring Preparation Days.")
        final_prep_days = 0
    else:
        final_prep_days = clamped_days
        
    # Re-calculate the log feature with the final value
    df['preparation_days_log'] = np.log1p(final_prep_days)
    # -------------------------------------------------------

    # PCA (Expects list of embeddings)
    embeddings_list = [text_embedding]
    pca_out = artifacts['pca'].transform(embeddings_list)
    pca_cols = [f'pca_embed_{i}' for i in range(config.PCA_COMPONENTS)]
    df_pca = pd.DataFrame(pca_out, columns=pca_cols, index=df.index)
    df = pd.concat([df, df_pca], axis=1)
    
    # NLP: TF-IDF
    df['text_clean'] = df['full_text'].apply(clean_text)
    tfidf_out = artifacts['tfidf'].transform(df['text_clean'])
    tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_out.shape[1])]
    df_tfidf = pd.DataFrame(tfidf_out.toarray(), columns=tfidf_cols, index=df.index)
    df = pd.concat([df, df_tfidf], axis=1)
    
    # Target Encoding
    global_mean = artifacts['encodings']['global_mean']
    sub_cat_map = artifacts['encodings']['sub_category']
    # Use map or fallback to global mean
    val = sub_cat_map.get(df['sub_category'].iloc[0], global_mean)
    df['sub_cat_encoded'] = val
    
    # Cyclical Time (Use provided launch_date or default to now)
    launch_date_str = data.get('launch_date')
    if launch_date_str:
        try:
            launch_dt = datetime.strptime(launch_date_str, '%Y-%m-%d')
        except ValueError:
            print(f"[!] Invalid date format {launch_date_str}, using current time.")
            launch_dt = datetime.now()
    else:
        launch_dt = datetime.now()

    df['launch_month_sin'] = np.sin(2 * np.pi * launch_dt.month / 12)
    df['launch_month_cos'] = np.cos(2 * np.pi * launch_dt.month / 12)
    df['launch_day_sin'] = np.sin(2 * np.pi * launch_dt.weekday() / 7)
    df['launch_day_cos'] = np.cos(2 * np.pi * launch_dt.weekday() / 7)
    df['is_weekend'] = 1 if launch_dt.weekday() >= 5 else 0
    
    # One-Hot Encoding Manual Setup
    # Map input 'category' to 'main_category' feature
    main_cat = data.get('category')
    if main_cat:
        # Check if the specific one-hot column exists in features (e.g. main_category_Technology)
        col_name = f"main_category_{main_cat}"
        # We set it to 1, but we need to make sure the column is added to df first or we can just add it
        # However, checking against artifacts['features'] later handles the 0s.
        # But we need to know valid feature names.
        # Let's just create the column if it's a valid one-hot column (we assume features list has it)
        # Actually, safer to iterate features or just set it if matches pattern.
        # But we don't have the feature list easily accessible to check *existence* efficiently inside this loop if we iterate.
        # Better strategy: pre-initialize all OHE cols to 0, then set active to 1.
        pass
    
    # Initialize all potential OHE columns found in artifacts to 0
    for col in artifacts['features']:
        if col not in df.columns:
            df[col] = 0

    # Set active OHE columns to 1
    if main_cat:
        col_name = f"main_category_{main_cat}"
        if col_name in df.columns:
            df[col_name] = 1
            
    country = data.get('country')
    if country:
        col_name = f"country_{country}"
        if col_name in df.columns:
            df[col_name] = 1
            
    # Reorder
    df_final = df[artifacts['features']]
    return df_final

def predict_success(data):
    """
    Main prediction function.
    data = {
        'name': 'Super Game',
        'blurb': 'An amazing game.',
        'goal': 10000,
        'duration_days': 30,
        'category': 'Games',
        'sub_category': 'Tabletop Games',
        'country': 'US',
        'currency': 'USD'
    }
    """
    artifacts = load_system()
    X = prepare_single_sample(data, artifacts)
    
    model = artifacts['model']
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    
    return {
        'success_probability': round(prob, 4),
        'prediction': 'SUCCESS' if pred == 1 else 'FAIL',
        'percent': f"{prob*100:.1f}%"
    }

if __name__ == "__main__":
    # Test Prediction
    sample = {
        'name': 'The Ultimate Smart Coffee Maker',
        'blurb': 'Revolutionary AI-powered coffee brewing system for the modern home. Brews perfect coffee every time.',
        'goal': 50000,
        'duration_days': 45,
        'category': 'Technology',
        'sub_category': 'Gadgets', # Must match training data keys mostly
        'country': 'US',
        'currency': 'USD'
    }
    
    print("\nPredicting for sample project...")
    result = predict_success(sample)
    print("\n" + "="*30)
    print(f"PREDICTION: {result['prediction']}")
    print(f"PROBABILITY: {result['percent']}")
    print("="*30 + "\n")
