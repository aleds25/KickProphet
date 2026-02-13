import pandas as pd
import numpy as np
import ast
import json
import re
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

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

def kfold_target_encoding(X_train, y_train, X_test, cat_col, n_folds=5, m=20):
    """
    Applica K-Fold Target Encoding con Smoothing per evitare Data Leakage.
    """
    print(f"⚙️  K-Fold Target Encoding su '{cat_col}' (Folds={n_folds}, m={m})...")
    
    # 1. Setup
    global_mean = y_train.mean()
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    col_name_enc = f"{cat_col}_target_enc"
    X_train[col_name_enc] = np.nan
    
    # 2. Loop sui Folds (TRAIN set)
    # Concateniamo temporaneamente X e y per gestire gli indici
    temp_train = pd.concat([X_train, y_train.rename('target')], axis=1)
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        # Dati su cui calcolare le medie (tutti tranne il fold corrente)
        X_fold_train = temp_train.iloc[train_idx]
        
        # Calcolo statistiche
        agg = X_fold_train.groupby(cat_col)['target'].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        
        # Smoothing
        smooth_map = (counts * means + m * global_mean) / (counts + m)
        
        # Mappiamo SOLO sulle righe del fold corrente (val_idx)
        # Usiamo map e fillna con global_mean per categorie mai viste
        X_train.loc[X_train.index[val_idx], col_name_enc] = \
            X_train.loc[X_train.index[val_idx], cat_col].map(smooth_map)
            
    # Riempiamo i buchi residui (categorie rare non presenti in alcuni fold di training)
    X_train[col_name_enc] = X_train[col_name_enc].fillna(global_mean)
    
    # 3. Applicazione sul TEST set
    # Usiamo TUTTO il training set senza fold (qui non c'è rischio leakage verso il test)
    agg_full = temp_train.groupby(cat_col)['target'].agg(['count', 'mean'])
    counts_full = agg_full['count']
    means_full = agg_full['mean']
    
    smooth_map_full = (counts_full * means_full + m * global_mean) / (counts_full + m)
    
    X_test[col_name_enc] = X_test[cat_col].map(smooth_map_full).fillna(global_mean)
    
    # 4. Sostituzione finale
    X_train[cat_col] = X_train[col_name_enc]
    X_test[cat_col] = X_test[col_name_enc]
    
    X_train.drop(columns=[col_name_enc], inplace=True)
    X_test.drop(columns=[col_name_enc], inplace=True)
    
    return X_train, X_test

# --- MAIN PIPELINE ---

def main():
    # 1. CARICAMENTO
    print("⏳ Caricamento dataset...")
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, '..', 'data', 'processed', 'KICKSTARTER_CLEAN_BASE.csv')
    
    # Check esistenza file
    if not os.path.exists(input_path):
        print(f"❌ Errore: File non trovato in {input_path}")
        return

    df = pd.read_csv(input_path, low_memory=False)
    print(f"📊 Dataset iniziale: {df.shape}")

    # 2. BASIC FEATURE ENGINEERING (Stateless)
    # A. TARGET
    if 'state' in df.columns:
        df['target'] = df['state'].apply(lambda x: 1 if str(x).lower().strip() == 'successful' else 0)

    # B. DATES
    cols_date = ['launched_at', 'deadline', 'created_at']
    for c in cols_date:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], unit='s')

    df['duration_days'] = (df['deadline'] - df['launched_at']).dt.days

    # Cyclical Encoding
    df['launch_month_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.month / 12)
    df['launch_month_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.month / 12)
    df['launch_day_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.dayofweek / 7)
    df['launch_day_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.dayofweek / 7)

    # Preparation Time
    if 'created_at' in df.columns:
        df['preparation_days'] = (df['launched_at'] - df['created_at']).dt.days
        df['preparation_days'] = df['preparation_days'].apply(lambda x: x if x >= 0 else 0)
        df['preparation_days_log'] = np.log1p(df['preparation_days'])

    # C. TEXT FEATURES (Simple)
    df['name'] = df['name'].fillna('')
    df['blurb'] = df['blurb'].fillna('')
    df['full_text'] = df['name'] + " " + df['blurb']
    df['name_len'] = df['name'].astype(str).apply(len)
    df['blurb_len'] = df['blurb'].astype(str).apply(len)
    df['has_question'] = df['name'].astype(str).str.contains('?', regex=False).astype(int)

    # D. CATEGORY HIERARCHY
    if 'category' in df.columns:
        df['main_category'], df['sub_category'] = zip(*df['category'].apply(extract_categories))

    # E. VIDEO
    if 'video' in df.columns:
        df['has_video'] = df['video'].apply(lambda x: 1 if isinstance(x, str) and '{' in x else 0)

    # 3. CURRENCY & GOAL
    if 'goal' in df.columns:
        df['goal'] = pd.to_numeric(df['goal'], errors='coerce')
        if 'static_usd_rate' in df.columns:
            df['goal_usd'] = df['goal'] * df['static_usd_rate']
        else:
            df['goal_usd'] = df['goal']
        df['goal_usd_log'] = np.log1p(df['goal_usd'])

    # 4. QUALITY CONTROL
    print(f"📊 Righe prima del QC: {len(df)}")
    df = df.dropna(subset=['launched_at', 'goal_usd']) # Drop if critical fields are NaN
    df = df[(df['duration_days'] > 0) & (df['duration_days'] <= 60)]
    df = df[df['goal_usd'] >= 10]
    if 'launched_at' in df.columns and 'created_at' in df.columns:
        df = df[df['launched_at'] >= df['created_at']]
    print(f"✅ Quality Control completato. Righe rimaste: {len(df)}")

    # 5. RIMOZIONE COLONNE LEAKAGE
    leakage_cols = [
        'pledged', 'backers_count', 'usd_pledged', 'converted_pledged_amount', 
        'state', 'state_changed_at', 'spotlight', 'percent_funded', 
        'id', 'creator', 'profile', 'photo', 'urls', 'source_url', 'slug', 'location', 
        'category', 'launched_at', 'deadline', 'created_at', 'goal', 'goal_usd', 
        'currency_symbol', 'static_usd_rate', 'currency_trailing_code', 'fx_rate', 
        'video', 'country_displayable_name', 'sort_date', 
        'disable_communication', 'is_disliked', 'is_in_post_campaign_pledging_phase', 
        'is_launched', 'is_liked', 'is_starrable', 'staff_pick', 'usd_type', 'usd_exchange_rate',
        'current_currency'
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    # 6. TRAIN / TEST SPLIT
    # FONDAMENTALE: Lo split deve avvenire PRIMA di qualsiasi target encoding
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✂️  Split completato. Train: {X_train.shape}, Test: {X_test.shape}")

    # 7. STATEFUL FEATURE ENGINEERING
    
    # 7a. GOAL RELATIVE TO CATEGORY (Vettorizzato)
    print("⏳ Calcolo Goal Relative to Category (Difference Log)...")
    sub_cat_medians = X_train.groupby('sub_category')['goal_usd_log'].median()
    global_median = X_train['goal_usd_log'].median()

    X_train['sub_cat_median'] = X_train['sub_category'].map(sub_cat_medians).fillna(global_median)
    X_test['sub_cat_median'] = X_test['sub_category'].map(sub_cat_medians).fillna(global_median)

    X_train['goal_to_cat_diff'] = X_train['goal_usd_log'] - X_train['sub_cat_median']
    X_test['goal_to_cat_diff'] = X_test['goal_usd_log'] - X_test['sub_cat_median']
    
    X_train.drop(columns=['sub_cat_median'], inplace=True)
    X_test.drop(columns=['sub_cat_median'], inplace=True)

    # 7b. K-FOLD TARGET ENCODING (Maniacale & Safe)
    # Sostituisce la vecchia logica di smoothing semplice
    X_train, X_test = kfold_target_encoding(
        X_train, y_train, X_test, 
        cat_col='sub_category', 
        n_folds=5, 
        m=20  # Aumento leggermente lo smoothing per sicurezza
    )
    print("✅ Target Encoding 'sub_category' completato (K-Fold Strategy).")

    # 8. IMPUTAZIONE
    cols_num = [
        'duration_days', 'preparation_days_log', 'name_len', 'blurb_len', 'goal_usd_log',
        'launch_month_sin', 'launch_month_cos', 'launch_day_sin', 'launch_day_cos', 
        'goal_to_cat_diff', 'has_video'
    ]
    for col in cols_num:
        if col in X_train.columns:
            mediana_train = X_train[col].median()
            X_train[col] = X_train[col].fillna(mediana_train)
            X_test[col] = X_test[col].fillna(mediana_train)

    cat_fill = ['country', 'main_category', 'currency']
    for col in cat_fill:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna('Unknown')
            X_test[col] = X_test[col].fillna('Unknown')

    # 9. TEXT CLEANING & TF-IDF
    print("⏳ Elaborazione Testo & TF-IDF...")
    X_train['text_clean'] = X_train['full_text'].apply(clean_text)
    X_test['text_clean'] = X_test['full_text'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    tfidf.fit(X_train['text_clean'])
    
    X_train_tfidf = tfidf.transform(X_train['text_clean'])
    X_test_tfidf = tfidf.transform(X_test['text_clean'])
    
    tfidf_cols = [f'word_{w}' for w in tfidf.get_feature_names_out()]
    df_train_tfidf = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_cols, index=X_train.index)
    df_test_tfidf = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_cols, index=X_test.index)
    
    X_train = pd.concat([X_train, df_train_tfidf], axis=1)
    X_test = pd.concat([X_test, df_test_tfidf], axis=1)

    cols_text = ['name', 'blurb', 'full_text', 'text_clean']
    X_train = X_train.drop(columns=[c for c in cols_text if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in cols_text if c in X_test.columns])

    # 10. ONE-HOT ENCODING
    print("⏳ One-Hot Encoding...")
    cols_cat = ['country', 'currency', 'main_category']
    cols_cat = [c for c in cols_cat if c in X_train.columns]

    X_train = pd.get_dummies(X_train, columns=cols_cat, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cols_cat, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # 11. SALVATAGGIO
    train_output = X_train.copy()
    train_output['target'] = y_train
    test_output = X_test.copy()
    test_output['target'] = y_test

    train_path = os.path.join(base_path, '..', 'data', 'processed', 'KICKSTARTER_TRAIN.csv')
    test_path = os.path.join(base_path, '..', 'data', 'processed', 'KICKSTARTER_TEST.csv')

    # Assicurati che la cartella esista
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)

    print(f"💾 Files aggiornati salvati in:\n   {train_path}\n   {test_path}")

if __name__ == "__main__":
    main()
