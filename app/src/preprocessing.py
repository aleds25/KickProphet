import pandas as pd
import numpy as np
import ast
import json
import re
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import textstat
from sklearn.metrics.pairwise import cosine_similarity

try:
    from app.src import config
except ImportError:
    # Fallback for running script directly
    import config

# --- HELPER FUNCTIONS ---

def extract_categories(x):
    """Extracts main and sub categories from the 'category' JSON string."""
    if not isinstance(x, str) or not x.strip():
        return 'Unknown', 'Unknown'
    try:
        cat_dict = json.loads(x)
    except:
        try:
            cat_dict = ast.literal_eval(x)
        except:
            return 'Unknown', 'Unknown'
    
    if not isinstance(cat_dict, dict):
        return 'Unknown', 'Unknown'
        
    slug = cat_dict.get('slug', '')
    parts = slug.split('/')
    if len(parts) >= 2:
        return parts[0], parts[1]
    elif len(parts) == 1 and parts[0]:
        return parts[0], parts[0]
    return 'Unknown', 'Unknown'



def clean_text(text):
    """Cleans text for TF-IDF processing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def cyclical_encoding(df):
    """Applies cyclical encoding to launch date."""
    if 'launched_at' in df.columns:
        df['launch_month_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.month / 12)
        df['launch_month_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.month / 12)
        df['launch_day_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.dayofweek / 7)
        df['launch_day_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.dayofweek / 7)
    return df

def get_nlp_features(text):
    """Extracts sentiment and readability scores."""
    if not isinstance(text, str) or not text.strip():
        return 0, 0, 0 # Default values (neutral, objective, hard)
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Readability (higher is easier)
    try:
        readability = textstat.flesch_reading_ease(text)
    except:
        readability = 50 
        
    # Valid / Garbage Detection
    # 1. Avg Word Length
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    
    # 2. Digit Ratio
    digit_count = sum(c.isdigit() for c in text)
    digit_ratio = digit_count / len(text) if len(text) > 0 else 0
        
    return polarity, subjectivity, readability, avg_word_len, digit_ratio

def get_embeddings(texts, model_name=config.NLP_MODEL_NAME):
    """Generates dense vector embeddings."""
    print(f"[*] Loading NLP Model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"[*] Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

# --- CORE PROCESSING LOGIC ---

def load_and_clean_base_data(input_path):
    """Loads data and performs stateless cleaning (dates, basic features)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    
    # Target
    if 'state' in df.columns:
        df['target'] = df['state'].apply(lambda x: 1 if str(x).lower().strip() == 'successful' else 0)

    # Dates
    cols_date = ['launched_at', 'deadline', 'created_at']
    for c in cols_date:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], unit='s')

    df['duration_days'] = (df['deadline'] - df['launched_at']).dt.days
    df['duration_days'] = df['duration_days'].apply(lambda x: x if x > 0 else 1) # Prevent div by zero
    df = cyclical_encoding(df)
    
    if 'launched_at' in df.columns:
        df['launch_hour'] = df['launched_at'].dt.hour
        df['is_weekend'] = df['launched_at'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    # Prep Time
    if 'created_at' in df.columns:
        df['preparation_days'] = (df['launched_at'] - df['created_at']).dt.days
        df['preparation_days'] = df['preparation_days'].apply(lambda x: x if x >= 0 else 0)
        df['preparation_days_log'] = np.log1p(df['preparation_days'])

    # Text Features
    df['name'] = df['name'].fillna('')
    df['blurb'] = df['blurb'].fillna('')
    df['full_text'] = df['name'] + " " + df['blurb']
    df['name_len'] = df['name'].astype(str).apply(len)
    df['blurb_len'] = df['blurb'].astype(str).apply(len)
    df['name_word_count'] = df['name'].astype(str).apply(lambda x: len(x.split()))
    df['blurb_word_count'] = df['blurb'].astype(str).apply(lambda x: len(x.split()))
    df['has_question'] = df['name'].astype(str).str.contains('?', regex=False).astype(int)
    df['name_is_upper'] = df['name'].astype(str).apply(lambda x: 1 if x.isupper() else 0)

    # Categories
    if 'category' in df.columns:
        df['main_category'], df['sub_category'] = zip(*df['category'].apply(extract_categories))

    # Video
    if 'video' in df.columns:
        df['has_video'] = df['video'].apply(lambda x: 1 if isinstance(x, str) and '{' in x else 0)

    # Currency & Goal
    if 'goal' in df.columns:
        df['goal'] = pd.to_numeric(df['goal'], errors='coerce')
        # Use static_usd_rate if available, else goal
        if 'static_usd_rate' in df.columns:
            df['goal_usd'] = df['goal'] * df['static_usd_rate']
        else:
            df['goal_usd'] = df['goal']
        df['goal_usd_log'] = np.log1p(df['goal_usd'])
        df['goal_per_day'] = df['goal_usd'] / df['duration_days']

    return df

def apply_quality_control(df):
    """Filters out invalid rows. Only for training/evaluation."""
    rows_before = len(df)
    df = df.dropna(subset=['launched_at', 'goal_usd'])
    df = df[(df['duration_days'] > 0) & (df['duration_days'] <= 60)]
    df = df[df['goal_usd'] >= 10]
    if 'launched_at' in df.columns and 'created_at' in df.columns:
        df = df[df['launched_at'] >= df['created_at']]
    print(f"Quality Control: dropped {rows_before - len(df)} rows.")
    return df

def feature_engineering_fit_transform(df):
    """
    Applies all feature engineering steps and FITS transformations (PCA, TF-IDF).
    Saves artifacts.
    """
    print("[*] Starting Feature Engineering (Fit & Transform)...")
    
    # 1. Clean Text & Basic NLP
    df['text_clean'] = df['full_text'].apply(clean_text)
    
    print("[*] Extracting Sentiment, Readability, & Garbage Metrics...")
    nlp_stats = df['full_text'].astype(str).apply(lambda x: pd.Series(get_nlp_features(x)))
    df[['sentiment_polarity', 'sentiment_subjectivity', 'readability_score', 'avg_word_len', 'digit_ratio']] = nlp_stats
    
    # 2. Embeddings & Semantic Coherence
    print("[*] Generating Semantic Embeddings & Coherence Scores...")
    # Load model once to encode both text and categories
    model = SentenceTransformer(config.NLP_MODEL_NAME)
    
    # A. Text Embeddings
    text_embeddings = model.encode(df['full_text'].fillna("").tolist(), show_progress_bar=True, normalize_embeddings=True)
    
    # B. Category Embeddings (Coherence Check)
    # Get unique sub-categories to avoid re-encoding millions of times
    unique_cats = df['sub_category'].unique().astype(str)
    cat_embeddings_dict = {cat: model.encode(cat, normalize_embeddings=True) for cat in unique_cats}
    
    # Map embeddings to dataframe rows
    # Note: Doing this row-by-row can be slow, vectorizing is better.
    # We'll create a matrix of category embeddings aligned with df
    cat_embeds_matrix = np.array([cat_embeddings_dict[x] for x in df['sub_category'].astype(str)])
    
    # Calculate Cosine Similarity (Dot product of normalized vectors)
    # SentenceTransformer embeddings are normalized by default? Usually yes.
    # We verify normalization or just use cosine_similarity function
    
    # Efficient calculation: diagonal of (A . B^T)
    # But we want row-wise dot product: sum(A * B, axis=1)
    coherence_scores = np.sum(text_embeddings * cat_embeds_matrix, axis=1)
    df['desc_cat_similarity'] = coherence_scores
    
    embeddings = text_embeddings # For PCA later
    
    # 3. PCA on Embeddings
    print(f"[*] Reducing Dimensions with PCA (n={config.PCA_COMPONENTS})...")
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # Save PCA model
    joblib.dump(pca, os.path.join(config.ARTIFACTS_DIR, 'pca_model.joblib'))
    
    # Add PCA features to DF
    pca_cols = [f'pca_embed_{i}' for i in range(config.PCA_COMPONENTS)]
    df_pca = pd.DataFrame(embeddings_pca, columns=pca_cols, index=df.index)
    df = pd.concat([df, df_pca], axis=1)

    # 4. TF-IDF (Classic Keyword Spotting) - Reduced
    print("[*] Fitting TF-IDF...")
    tfidf = TfidfVectorizer(max_features=config.MAX_TEXT_FEATURES)
    tfidf_matrix = tfidf.fit_transform(df['text_clean'])
    
    # Save TF-IDF
    joblib.dump(tfidf, os.path.join(config.ARTIFACTS_DIR, 'tfidf_vectorizer.joblib'))
    
    # Add TF-IDF features
    tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)
    df = pd.concat([df, df_tfidf], axis=1)
    
    # 5. Target Encoding (Smoothing)
    print("[*] Target Encoding...")
    # ... (rest remains similar but carefully merged)
    global_mean = df[config.TARGET_COL].mean()
    
    # Sub-category
    agg = df.groupby('sub_category')[config.TARGET_COL].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth_weight = 10
    smooth_means = (counts * means + smooth_weight * global_mean) / (counts + smooth_weight)
    
    # Save Artifacts
    encoding_map = {
        'sub_category': smooth_means.to_dict(),
        'global_mean': global_mean
    }
    joblib.dump(encoding_map, os.path.join(config.ARTIFACTS_DIR, 'target_encodings.joblib'))
    
    df['sub_cat_encoded'] = df['sub_category'].map(smooth_means)
    
    # 6. Drop Text Columns
    df = df.drop(columns=config.TEXT_COLS_TO_DROP + ['sub_category'], errors='ignore')
    
    return df

def feature_engineering_transform_new(df, artifacts):
    """
    Applies transformations to NEW data using saved artifacts.
    No fitting, no target usage.
    """
    # 1. Goal Relative to Category
    sub_cat_medians = artifacts['sub_cat_medians']
    global_median = artifacts['global_goal_median']
    
    df['sub_cat_median'] = df['sub_category'].map(sub_cat_medians).fillna(global_median)
    df['goal_to_cat_diff'] = df['goal_usd_log'] - df['sub_cat_median']
    df.drop(columns=['sub_cat_median'], inplace=True)

    # 2. Target Encoding (Map)
    target_enc_map = artifacts['target_enc_map']
    global_mean = artifacts['target_enc_global_mean']
    
    # Check if sub_category exists (might be one-hot encoded later, but here we expect raw)
    if 'sub_category' in df.columns:
        df['sub_category'] = df['sub_category'].map(target_enc_map).fillna(global_mean)

    # 3. Imputation
    impute_medians = artifacts['impute_medians']
    for col, median_val in impute_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    
    for col in config.CAT_COLS_IMPUTE:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # 4. TF-IDF
    tfidf = artifacts['tfidf_vectorizer']
    if 'full_text' in df.columns:
        df['text_clean'] = df['full_text'].apply(clean_text)
        tfidf_data = tfidf.transform(df['text_clean'])
        tfidf_cols = [f'word_{w}' for w in tfidf.get_feature_names_out()]
        df_tfidf = pd.DataFrame(tfidf_data.toarray(), columns=tfidf_cols, index=df.index)
        df = pd.concat([df, df_tfidf], axis=1)

    return df

def feature_engineering_transform_new(df):
    """
    Applies transformations to NEW data using saved artifacts.
    """
    print("[*] Applying Feature Engineering to New Data...")
    
    # 1. Clean Text & Basic NLP
    df['text_clean'] = df['full_text'].apply(clean_text)
    
    print("[*] Extracting Sentiment, Readability & Garbage Metrics...")
    nlp_stats = df['full_text'].astype(str).apply(lambda x: pd.Series(get_nlp_features(x)))
    df[['sentiment_polarity', 'sentiment_subjectivity', 'readability_score', 'avg_word_len', 'digit_ratio']] = nlp_stats
    
    # 2. Embeddings & Coherence
    print("[*] Generating Semantic Embeddings & Coherence...")
    model = SentenceTransformer(config.NLP_MODEL_NAME) # Re-load model (inefficient but safe script-wise)
    text_embeddings = model.encode(df['full_text'].fillna("").tolist(), show_progress_bar=True, normalize_embeddings=True)
    
    # Coherence
    if 'sub_category' in df.columns:
        unique_cats = df['sub_category'].unique().astype(str)
        cat_embeddings_dict = {cat: model.encode(cat, normalize_embeddings=True) for cat in unique_cats}
        cat_embeds_matrix = np.array([cat_embeddings_dict.get(str(x), cat_embeddings_dict.get('Unknown')) for x in df['sub_category']])
        df['desc_cat_similarity'] = np.sum(text_embeddings * cat_embeds_matrix, axis=1)
    else:
        df['desc_cat_similarity'] = 0.5 # Default neutral if no category
        
    embeddings = text_embeddings
    
    # 3. PCA Transform
    print("[*] PCA Transform...")
    pca = joblib.load(os.path.join(config.ARTIFACTS_DIR, 'pca_model.joblib'))
    embeddings_pca = pca.transform(embeddings)
    
    pca_cols = [f'pca_embed_{i}' for i in range(config.PCA_COMPONENTS)]
    df_pca = pd.DataFrame(embeddings_pca, columns=pca_cols, index=df.index)
    df = pd.concat([df, df_pca], axis=1)
    
    # 4. TF-IDF Transform
    print("[*] TF-IDF Transform...")
    tfidf = joblib.load(os.path.join(config.ARTIFACTS_DIR, 'tfidf_vectorizer.joblib'))
    tfidf_matrix = tfidf.transform(df['text_clean'])
    
    tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)
    df = pd.concat([df, df_tfidf], axis=1)
    
    # 5. Target Encoding Map
    print("[*] Applying Target Encoding...")
    enc_map = joblib.load(os.path.join(config.ARTIFACTS_DIR, 'target_encodings.joblib'))
    df['sub_cat_encoded'] = df['sub_category'].map(enc_map['sub_category']).fillna(enc_map['global_mean'])

    # 6. Drop Text Columns
    df = df.drop(columns=config.TEXT_COLS_TO_DROP + ['sub_category'], errors='ignore')
    
    return df

def finalize_dataset(df, is_training=True):
    """Final cleanup: Dropping columns, One-Hot Encoding."""
    
    # Drop Text Cols
    df = df.drop(columns=[c for c in config.TEXT_COLS_TO_DROP if c in df.columns])

    # Drop Leakage Cols
    df = df.drop(columns=[c for c in config.LEAKAGE_COLS if c in df.columns], errors='ignore')

    # Drop redundant cols
    df = df.drop(columns=[c for c in config.REDUNDANT_COLS if c in df.columns], errors='ignore')

    # One-Hot Encoding
    # Note: For production, we should align with training columns. 
    # Here, pandas get_dummies can be tricky if categories are missing.
    # In a perfect world, we'd save the OneHotEncoder artifact too.
    # For now, we assume alignment happens in Training/Evaluation phase.
    df = pd.get_dummies(df, columns=[c for c in config.ONE_HOT_COLS if c in df.columns], drop_first=True)

    # Move Target to Last
    if config.TARGET_COL in df.columns:
        cols = [c for c in df.columns if c != config.TARGET_COL] + [config.TARGET_COL]
        df = df[cols]

    return df

# --- MAIN EXEUCTION ---

def main():
    print("⏳ Starting Preprocessing Pipeline...")
    
    # 1. Load 
    df = load_and_clean_base_data(config.RAW_DATA_PATH)
    
    # 2. Quality Control (Only for Training)
    df = apply_quality_control(df)

    # 3. Feature Engineering & Artifact Saving
    # Note: We process the ENTIRE dataset to fit artifacts, 
    # then we split. Or we split then fit on train.
    # Best practice: Split, then fit on Train, then Transform Test.
    
    print("Splitting Data...")
    X = df.drop(config.TARGET_COL, axis=1)
    y = df[config.TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Recombine for easier processing logic (pandas oriented)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print("Fitting transform on TRAIN...")
    train_df = feature_engineering_fit_transform(train_df)
    
    print("Transforming TEST...")
    test_df = feature_engineering_transform_new(test_df)

    # 4. Finalize
    train_df = finalize_dataset(train_df)
    test_df = finalize_dataset(test_df)
    
    # Align Columns (One-Hot might have created mismatches)
    train_cols = train_df.columns.tolist()
    # Ensure test has same columns as train
    for c in train_cols:
        if c not in test_df.columns:
            test_df[c] = 0
    # Drop extra columns in test
    test_df = test_df[train_cols]

    # 5. Save Splits
    print(f"Saving to {config.TRAIN_DATA_PATH}...")
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    
    print("✅ Preprocessing Complete.")

if __name__ == "__main__":
    main()
