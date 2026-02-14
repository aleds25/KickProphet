import os

# ──────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# Navigate up from app/src/ to root, then to app/data/processed
DATA_DIR = os.path.join(BASE_PATH, '..', 'data', 'processed')
MODELS_DIR = os.path.join(BASE_PATH, '..', 'models')
ARTIFACTS_DIR = os.path.join(MODELS_DIR, 'artifacts')
RESULTS_DIR = os.path.join(BASE_PATH, '..', 'results')

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
#  FILES
# ──────────────────────────────────────────────
RAW_DATA_PATH = os.path.join(DATA_DIR, 'KICKSTARTER_CLEAN_BASE.csv')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'KICKSTARTER_TRAIN.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'KICKSTARTER_TEST.csv')

MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.joblib')
OPTUNA_PARAMS_PATH = os.path.join(MODELS_DIR, 'best_params_lgbm.json')

# ──────────────────────────────────────────────
#  FEATURES & COLUMNS
# ──────────────────────────────────────────────
TARGET_COL = 'target'
TEXT_COLS_TO_DROP = ['name', 'blurb', 'full_text', 'text_clean']

# Columns to drop to prevent leakage
LEAKAGE_COLS = [
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

# Columns for Imputation
NUM_COLS_IMPUTE = [
    'duration_days', 'preparation_days_log', 'name_len', 'blurb_len', 'goal_usd_log',
    'launch_month_sin', 'launch_month_cos', 'launch_day_sin', 'launch_day_cos', 
    'goal_to_cat_diff', 'has_video'
]

CAT_COLS_IMPUTE = ['country', 'main_category', 'currency']

# Columns for One-Hot Encoding
ONE_HOT_COLS = ['country', 'currency', 'main_category']

# ──────────────────────────────────────────────
#  MODEL PARAMETERS (DEFAULTS)
# ──────────────────────────────────────────────
LGBM_DEFAULT_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
