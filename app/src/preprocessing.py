import pandas as pd
import numpy as np
import ast
import json
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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
    text = text.lower() # Lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text) # Rimuovi punteggiatura e caratteri speciali
    text = re.sub(r'\s+', ' ', text).strip() # Rimuovi spazi extra
    return text

def main():
    # 1. CARICAMENTO
    print("⏳ Caricamento dataset...")

    #path dinamici
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, '..', 'data', 'processed', 'KICKSTARTER_CLEAN_BASE.csv')
    
    df = pd.read_csv(input_path, low_memory=False)
    print(f"📊 Dataset iniziale: {df.shape}")

    # 2. BASIC FEATURE ENGINEERING (Stateless)
    # A. TARGET (0/1)
    if 'state' in df.columns:
        df['target'] = df['state'].apply(lambda x: 1 if str(x).lower().strip() == 'successful' else 0)

    # B. DATE HANDLING & CYCLICAL TEMPORAL FEATURES
    cols_date = ['launched_at', 'deadline', 'created_at']
    for c in cols_date:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], unit='s')

    df['duration_days'] = (df['deadline'] - df['launched_at']).dt.days

    # Cyclical Encoding per Mese (periodo 12)
    df['launch_month_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.month / 12)
    df['launch_month_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.month / 12)

    # Cyclical Encoding per Giorno della Settimana (periodo 7)
    df['launch_day_sin'] = np.sin(2 * np.pi * df['launched_at'].dt.dayofweek / 7)
    df['launch_day_cos'] = np.cos(2 * np.pi * df['launched_at'].dt.dayofweek / 7)

    # Feature "Preparation Time"
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

    # D. CATEGORY HIERARCHY (Main + Sub)
    if 'category' in df.columns:
        df['main_category'], df['sub_category'] = zip(*df['category'].apply(extract_categories))

    # E. VIDEO (New Feature)
    if 'video' in df.columns:
        df['has_video'] = df['video'].apply(lambda x: 1 if isinstance(x, str) and '{' in x else 0)

    print("✅ Feature Engineering (Date, Text, Category, Video) completato.")

    # 3. CURRENCY NORMALIZATION
    if 'goal' in df.columns:
        df['goal'] = pd.to_numeric(df['goal'], errors='coerce')
        if 'static_usd_rate' in df.columns:
            df['goal_usd'] = df['goal'] * df['static_usd_rate']
        else:
            df['goal_usd'] = df['goal']
        df['goal_usd_log'] = np.log1p(df['goal_usd'])

    print("✅ Goal convertito in USD e log-trasformato.")

    # 3c. FEATURE CROSSING (Financial Ratios)
        

    print("✅ Feature Crossing (Log-Financial Ratios) completato.")

    # 3b. QUALITY CONTROL (Rimozione Errori Logici)
    print(f"📊 Righe prima del QC: {len(df)}")
    df = df[(df['duration_days'] > 0) & (df['duration_days'] <= 60)]
    df = df.dropna(subset=['goal_usd'])
    df = df[df['goal_usd'] >= 10]
    if 'launched_at' in df.columns and 'created_at' in df.columns:
        df = df[df['launched_at'] >= df['created_at']]
    print(f"✅ Quality Control completato. Righe rimaste: {len(df)}")

    # 4. RIMOZIONE COLONNE
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
    print(f"✅ Colonne rimosse. Rimaste: {len(df_clean.columns)}")

    # 5. TRAIN / TEST SPLIT
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✂️ Split completato. Train: {X_train.shape}, Test: {X_test.shape}")

    # 6. STATEFUL FEATURE ENGINEERING (Goal relative to category)
    print("⏳ Calcolo mediale del goal per categoria (Cascata)...")
    # Calcolo mediale su TRAIN per evitare leakage
    sub_medians = X_train.groupby('sub_category')['goal_usd_log'].median()
    main_medians = X_train.groupby('main_category')['goal_usd_log'].median()

    def get_relative_goal(row, medians_sub, medians_main):
        sub = row['sub_category']
        main = row['main_category']
        
        median = medians_sub.get(sub, np.nan)
        if pd.isna(median) or sub == 'Unknown':
            median = medians_main.get(main, np.nan)
        
        if pd.isna(median) or median == 0:
            return 1.0 # Fallback neutro se non c'è una mediana valida
        
        return row['goal_usd_log'] / median

    X_train['goal_to_cat_ratio'] = X_train.apply(lambda r: get_relative_goal(r, sub_medians, main_medians), axis=1)
    X_test['goal_to_cat_ratio'] = X_test.apply(lambda r: get_relative_goal(r, sub_medians, main_medians), axis=1)
    
    X_train['goal_to_cat_ratio_log'] = np.log1p(X_train['goal_to_cat_ratio'])
    X_test['goal_to_cat_ratio_log'] = np.log1p(X_test['goal_to_cat_ratio'])
    print("✅ Feature 'goal_to_cat_ratio_log' creata.")

    # 7. IMPUTAZIONE (Stateful)
    cols_num = [
        'duration_days', 'preparation_days_log', 'name_len', 'blurb_len', 'goal_usd_log',
        'launch_month_sin', 'launch_month_cos', 'launch_day_sin', 'launch_day_cos', 
        'goal_to_cat_ratio_log', 'has_video'
    ]
    for col in cols_num:
        if col in X_train.columns:
            mediana_train = X_train[col].median()
            X_train[col] = X_train[col].fillna(mediana_train)
            X_test[col] = X_test[col].fillna(mediana_train)

    cat_fill = ['country', 'main_category', 'sub_category', 'currency']
    for col in cat_fill:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna('Unknown')
            X_test[col] = X_test[col].fillna('Unknown')
    print("✅ Imputazione completata.")

    # 7. TEXT CLEANING
    print("⏳ Pulizia del testo in corso...")
    X_train['text_clean'] = X_train['full_text'].apply(clean_text)
    X_test['text_clean'] = X_test['full_text'].apply(clean_text)
    print("✅ Testo pulito.")

    # 8. TF-IDF
    print("⏳ TF-IDF Vectorization...")
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
    print(f"✅ TF-IDF completato.")

    # 9. ONE-HOT ENCODING
    print("⏳ One-Hot Encoding...")
    cols_cat = ['country', 'currency', 'main_category', 'sub_category']
    cols_cat = [c for c in cols_cat if c in X_train.columns]

    X_train = pd.get_dummies(X_train, columns=cols_cat, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cols_cat, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    print(f"✅ Encoding completato. Colonne finali: {X_train.shape[1]}")

    # 10. SALVATAGGIO
    train_output = X_train.copy()
    train_output['target'] = y_train
    test_output = X_test.copy()
    test_output['target'] = y_test

    train_path = os.path.join(base_path, '..', 'data', 'processed', 'KICKSTARTER_TRAIN.csv')
    test_path = os.path.join(base_path, '..', 'data', 'processed', 'KICKSTARTER_TEST.csv')

    train_output.to_csv(train_path, index=False)
    test_output.to_csv(test_path, index=False)

    print(f"💾 Files aggiornati salvati in:\n   {train_path}\n   {test_path}")

if __name__ == "__main__":
    main()
