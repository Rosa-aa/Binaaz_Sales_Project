"""
config.py
---------
Project-wide constants and settings for the Bina.az price prediction pipeline.
"""

DATA_PATH = "house_sale.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns dropped early because they duplicate another column (>95% identical values)
DUPLICATE_COLS = [
    'estate_id', 'currency_y', 'estate_rel_url_y', 'estate_rel_url',
    'total_price', 'id_y', 'estate_details_id_y',
    'id_x', 'rel_url', 'estate_rel_url_x', 'estate_details_id_x',
]

# Columns dropped because they are low-quality (mostly missing, or not informative)
LOW_QUALITY_COLS = ['Binanın növü', 'featured', 'vip', 'Torpaq sahəsi', 'hour_y']

# Raw datetime/day/hour columns not used for modeling
DATETIME_RAW_COLS = ['datetime_scrape_x', 'datetime_scrape_y', 'day_x', 'hour_x', 'day_y']

# Renaming map: original (English/mixed) column names -> Azerbaijani
COLUMN_RENAME_MAP = {
    'price':          'qiymət',
    'currency_x':     'valyuta',
    'unit_price':     'kvadrat_metr_qiyməti',
    'location':       'yer',
    'address':        'ünvan',
    'city':           'şəhər',
    'city_when':      'şəhər_vaxtı',
    'lat':            'enlik',
    'lng':            'uzunluq',
    'repair':         'təmir_statusu',
    'Təmir':          'təmir_statusu',
    'Kateqoriya':     'kateqoriya',
    'area':           'sahə',
    'rooms':          'otaq_sayı',
    'floor':          'mərtəbə',
    'total_floors':   'ümumi_mərtəbə',
    'İpoteka':        'ipoteka',
    'Çıxarış':        'çıxarış',
    'bill_of_sale':   'alqı_sənədi',
    'views':          'baxış_sayı',
    'img_url':        'şəkil_linki',
    'description':    'təsvir',
    'extra_info':     'əlavə_məlumat',
    'products_label': 'elan_etiketi',
    'owner_name':     'mülkiyyətçi_adı',
    'owner_title':    'mülkiyyətçi_tipi',
    'shop_name':      'agentlik_adı',
    'shop_title':     'agentlik_tipi',
}

# Columns dropped right after renaming (not needed for modeling)
POST_RENAME_DROP_COLS = ['şəhər_vaxtı', 'valyuta']

# Columns not usable as model features (free text, images, owner name, etc.)
NON_FEATURE_COLS = ['şəkil_linki', 'təsvir', 'əlavə_məlumat', 'mülkiyyətçi_adı', 'ünvan']

# Columns dropped right before modeling because they leak the target
# (derived directly from 'qiymət', or otherwise not safe to use)
LEAKAGE_COLS = [
    "kvadrat_metr_qiyməti",
    "qiymət_per_m2",
    "qiymət_qrupu",
    "attributes",
    "updated",
    "sirket_adı",
]

TARGET_COL = "qiymət"

# Encoding groups
LABEL_ENCODE_COLS = [
    'təmir_statusu', 'elan_etiketi', 'alqı_sənədi',
    'mülkiyyətçi_tipi', 'agentlik_tipi', 'kateqoriya',
    'çıxarış', 'ipoteka',
]
TARGET_ENCODE_COLS = ['yer', 'agentlik_adı']
ONE_HOT_COLS = ['şəhər']

# Outlier columns checked with the IQR method
OUTLIER_CHECK_COLS = ['qiymət', 'sahə', 'baxış_sayı', 'kvadrat_metr_qiyməti']
VIEW_COUNT_QUANTILE_CAP = 0.99

# Final tuned XGBoost hyperparameters
BEST_MODEL_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
)
