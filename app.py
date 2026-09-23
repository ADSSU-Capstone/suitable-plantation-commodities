import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NGP Plantation Commodity Prediction Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem; font-weight: bold; color: #1E6F3C;
        text-align: center; padding: 1rem 0;
        border-bottom: 3px solid #1E6F3C;
    }
    .sub-header {
        font-size: 1.05rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E6F3C 0%, #2E8B57 100%);
        padding: 1.3rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 1.9rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px 8px 0 0;
        padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E6F3C !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
DATA_FILE = 'CENRO Bunawan Data 2015-2025 (1).xlsx'

POSSIBLE_TARGET_COLS = ['COMMODITY_', 'COMMODITY', 'COMMODITY_TYPE']

PREDICTIVE_FEATURES = ['MUNI_CITY', 'BARANGAY', 'ZONE', 'TENURE',
                       'PART_TYP', 'YR_ESTAB', 'AREA_HA']
CATEGORICAL_FEATURES = ['MUNI_CITY', 'BARANGAY', 'ZONE', 'TENURE', 'PART_TYP']
NUMERIC_FEATURES = ['YR_ESTAB', 'AREA_HA']

HEADER_KEYWORDS = ['FID', 'MUNI_CITY', 'COMMODITY', 'BARANGAY', 'YR_ESTAB']


# ============================================================
# HELPER — Clean commodity label
# ============================================================
def _clean_commodity_label(val):
    """Collapse multi-label strings into a single primary class label."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == '' or s.lower() == 'nan':
        return None
    s_lower = s.lower()

    if 'indigenous' in s_lower and 'bamboo' in s_lower:
        return 'Indigenous & Bamboo'

    if 'bamboo' in s_lower or 'patong' in s_lower or 'giant bamboo' in s_lower:
        return 'Bamboo'
    if 'indigenous' in s_lower:
        return 'Indigenous'
    if ('agroforestry' in s_lower or 'rubber' in s_lower
            or 'coffee' in s_lower or 'cacao' in s_lower):
        return 'Agroforestry'
    if ('high value' in s_lower or 'mangosteen' in s_lower
            or 'durian' in s_lower or 'rambutan' in s_lower
            or 'lanzones' in s_lower or 'calamansi' in s_lower):
        return 'High Value'
    if ('timber' in s_lower or 'falcata' in s_lower or 'narra' in s_lower
            or 'lauan' in s_lower or 'mahogany' in s_lower
            or 'gmelina' in s_lower or 'molave' in s_lower):
        return 'Timber'
    if 'fruit' in s_lower or 'mango' in s_lower or 'lemonsito' in s_lower:
        return 'Other Fruit Trees'
    return 'Other'


# ============================================================
# DATA LOADING — header auto-detection, all strings
# ============================================================
@st.cache_data(show_spinner=True)
def load_data():
    search_paths = [
        '.',
        '/content/data',
        '/content',
        os.path.dirname(os.path.abspath(__file__)),
    ]
    filepath = None
    for path in search_paths:
        if not path or not os.path.isdir(path):
            continue
        candidate = os.path.join(path, DATA_FILE)
        if os.path.exists(candidate):
            filepath = candidate
            break
        for f in glob.glob(os.path.join(path, '*.xlsx')):
            bname = os.path.basename(f).lower()
            if ('cenro' in bname or 'bunawan' in bname
                    or 'reforestation' in bname):
                filepath = f
                break
        if filepath:
            break

    if filepath is None:
        return None, ("Data file not found. Please place the CENRO Bunawan "
                      "Excel file in the app directory.")

    try:
        xls = pd.ExcelFile(filepath)
        sheet_names = xls.sheet_names

        # Prefer 'Reforestation' sheet
        target_sheet = None
        for s in sheet_names:
            if 'reforestation' in str(s).lower():
                target_sheet = s
                break
        if target_sheet is None:
            target_sheet = sheet_names[0]

        # ---------- READ EVERYTHING AS STRING, NO HEADER ----------
        raw = pd.read_excel(
            xls, sheet_name=target_sheet, header=None, dtype=str
        )
        # Replace NaN with empty strings right away
        raw = raw.fillna('')

        # ---------- FIND HEADER ROW ----------
        header_row = 0
        HEADER_KEYWORDS_UPPER = [k.upper() for k in HEADER_KEYWORDS]

        for i in range(min(10, len(raw))):
            row_vals = raw.iloc[i].astype(str).str.upper().str.strip().tolist()
            matches = sum(1 for kw in HEADER_KEYWORDS_UPPER if kw in row_vals)
            if matches >= 2:
                header_row = i
                break

        # ---------- EXTRACT HEADER NAMES ----------
        header_series = raw.iloc[header_row]

        new_cols = []
        for idx, val in enumerate(header_series):
            try:
                name = str(val).strip().replace('\ufeff', '')
            except Exception:
                name = ''
            if name == '' or name.lower() in ('nan', 'none'):
                name = f'COL_{idx}'
            new_cols.append(name.upper())

        # ---------- BUILD CLEAN DATAFRAME ----------
        df = raw.iloc[header_row + 1:].reset_index(drop=True).copy()
        df.columns = new_cols

        # Drop placeholder columns
        df = df.loc[:, ~df.columns.str.startswith('COL_')]

        # Drop fully empty rows
        df = df.dropna(how='all')

        # Replace empty strings with NaN for later processing
        df = df.replace({'': np.nan})

        return df, None
    except Exception as e:
        import traceback
        return None, f"Error loading file: {e}\n\n{traceback.format_exc()}"


# ============================================================
# PREPROCESSING
# ============================================================
@st.cache_data(show_spinner=True)
def preprocess(df):
    df = df.copy()

    # ----------------------------------------------------------
    # 1. Column normalization
    # ----------------------------------------------------------
    df.columns = [str(c).strip().replace('\ufeff', '').upper()
                  for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith('UNNAMED')]
    df = df.loc[:, ~df.columns.str.startswith('COL_')]

    # ----------------------------------------------------------
    # 2. Auto-detect target column
    # ----------------------------------------------------------
    target_col = None
    for cand in POSSIBLE_TARGET_COLS:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is None:
        raise KeyError(
            f"No commodity/target column found. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows without target
    df = df[df[target_col].notna()]
    df = df[df[target_col].astype(str).str.strip() != '']

    # ----------------------------------------------------------
    # 3. Rename known feature columns
    # ----------------------------------------------------------
    rename_map = {}
    for col in df.columns:
        cu = col.upper().strip()
        if cu == 'MUNI_CITY':   rename_map[col] = 'MUNI_CITY'
        elif cu == 'BARANGAY':  rename_map[col] = 'BARANGAY'
        elif cu == 'ZONE':      rename_map[col] = 'ZONE'
        elif cu == 'TENURE':    rename_map[col] = 'TENURE'
        elif cu == 'PART_TYP':  rename_map[col] = 'PART_TYP'
        elif cu == 'YR_ESTAB':  rename_map[col] = 'YR_ESTAB'
        elif cu == 'AREA_HA':   rename_map[col] = 'AREA_HA'
        elif cu == 'SPECIES':   rename_map[col] = 'SPECIES'
        elif cu == 'FID':       rename_map[col] = 'FID'
        elif cu == 'UNIQ_ID':   rename_map[col] = 'UNIQ_ID'
        elif cu == 'NAME_PART': rename_map[col] = 'NAME_PART'
    df = df.rename(columns=rename_map)

    # Store target under a stable internal name
    df['COMMODITY_RAW'] = df[target_col]

    # ----------------------------------------------------------
    # 4. Deduplicate
    # ----------------------------------------------------------
    key_cols = [c for c in ['FID', 'UNIQ_ID', 'MUNI_CITY', 'BARANGAY',
                            'YR_ESTAB', 'COMMODITY_RAW']
                if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep='first')

    # Drop rows with >30% missing
    df = df.dropna(thresh=int(df.shape[1] * 0.7))

    # ----------------------------------------------------------
    # 5. Clean categorical fields
    # ----------------------------------------------------------
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = 'UNKNOWN'
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace({'NAN': np.nan, '': np.nan, 'NONE': np.nan})
        df[col] = df[col].fillna('UNKNOWN')

    # ----------------------------------------------------------
    # 6. Numeric fields (values arrive as strings)
    # ----------------------------------------------------------
    if 'YR_ESTAB' not in df.columns:
        df['YR_ESTAB'] = np.nan
    df['YR_ESTAB'] = pd.to_numeric(
        df['YR_ESTAB'].astype(str).str.strip(),
        errors='coerce'
    )
    if df['YR_ESTAB'].notna().sum() > 0:
        df['YR_ESTAB'] = df['YR_ESTAB'].fillna(df['YR_ESTAB'].median())

    if 'AREA_HA' not in df.columns:
        df['AREA_HA'] = np.nan
    df['AREA_HA'] = pd.to_numeric(
        df['AREA_HA'].astype(str).str.strip(),
        errors='coerce'
    )
    if df['AREA_HA'].notna().sum() > 0:
        df['AREA_HA'] = df['AREA_HA'].fillna(df['AREA_HA'].median())
        q1 = df['AREA_HA'].quantile(0.01)
        q99 = df['AREA_HA'].quantile(0.99)
        df['AREA_HA'] = df['AREA_HA'].clip(lower=q1, upper=q99)

    # ----------------------------------------------------------
    # 7. Target: collapse multi-labels into primary class
    # ----------------------------------------------------------
    df['COMMODITY_CLEAN'] = df['COMMODITY_RAW'].apply(_clean_commodity_label)
    df = df[df['COMMODITY_CLEAN'].notna()]

    # Drop classes with fewer than 2 samples
    class_counts = df['COMMODITY_CLEAN'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df['COMMODITY_CLEAN'].isin(valid_classes)]

    # Ensure all predictive features exist
    for col in PREDICTIVE_FEATURES:
        if col not in df.columns:
            df[col] = 'UNKNOWN'

    return df


# ============================================================
# LOAD & PREPROCESS
# ============================================================
df_raw, load_err = load_data()

st.markdown(
    '<div class="main-header">🌳 NGP Plantation Commodity Prediction Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Machine Learning-Based Prediction of Suitable '
    'Plantation Commodities for National Greening Program Sites<br>'
    '<i>CENRO Bunawan, Agusan del Sur (2015–2025)</i></div>',
    unsafe_allow_html=True
)

if df_raw is None:
    st.error(f"❌ {load_err}")
    st.info("Please upload the CENRO Bunawan dataset to the app directory "
            "and refresh.")
    st.stop()

# Debug expander — shows raw columns for troubleshooting
with st.expander("🐛 Debug: Raw Loaded Data", expanded=False):
    st.write(f"**Shape:** {df_raw.shape}")
    st.write("**Columns:**")
    st.write(list(df_raw.columns))
    st.write("**Dtypes:**")
    st.write(df_raw.dtypes.astype(str).to_frame().T)
    st.write("**First 3 rows:**")
    st.dataframe(df_raw.head(3), use_container_width=True)

try:
    df = preprocess(df_raw)
except KeyError as e:
    st.error(f"❌ Column detection error: {e}")
    st.write("**Available columns in the loaded file:**")
    st.write(list(df_raw.columns))
    st.stop()

if len(df) < 10:
    st.error("❌ Not enough data after preprocessing. "
             "Please check the dataset.")
    st.stop()


# ============================================================
# DATA SUMMARY
# ============================================================
with st.expander("🔍 Data Summary", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Municipalities", df['MUNI_CITY'].nunique())
    c3.metric("Barangays", df['BARANGAY'].nunique())
    if df['YR_ESTAB'].notna().any():
        c4.metric("Years Covered",
                  f"{int(df['YR_ESTAB'].min())}–{int(df['YR_ESTAB'].max())}")
    else:
        c4.metric("Years Covered", "N/A")

    st.write("**Commodity Class Distribution (Target):**")
    target_counts = df['COMMODITY_CLEAN'].value_counts()
    target_df = target_counts.rename('Count').to_frame()
    target_df['Percentage'] = (
        target_df['Count'] / target_df['Count'].sum() * 100
    ).round(2)
    st.dataframe(target_df, use_container_width=True)

    st.write("**Records per Municipality:**")
    st.dataframe(
        df['MUNI_CITY'].value_counts().rename('Records').to_frame(),
        use_container_width=True
    )

    st.write("**Records per Year Established:**")
    st.dataframe(
        df['YR_ESTAB'].value_counts().sort_index().rename('Records').to_frame(),
        use_container_width=True
    )


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Analysis Controls")

municipality_filter = st.sidebar.multiselect(
    "Filter by Municipality:",
    options=sorted(df['MUNI_CITY'].unique()),
    default=sorted(df['MUNI_CITY'].unique())
)

zone_filter = st.sidebar.multiselect(
    "Filter by Zone:",
    options=sorted(df['ZONE'].unique()),
    default=sorted(df['ZONE'].unique())
)

tenure_filter = st.sidebar.multiselect(
    "Filter by Tenure Type:",
    options=sorted(df['TENURE'].unique()),
    default=sorted(df['TENURE'].unique())
)

test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20, 5) / 100.0
cv_folds = st.sidebar.slider("Cross-Validation Folds", 3, 10, 10)
n_estimators = st.sidebar.slider("Random Forest: n_estimators",
                                 50, 500, 200, 50)
max_depth = st.sidebar.selectbox("Random Forest: max_depth",
                                 [None, 10, 20, 30], index=3)
min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 5, 1)
min_support = st.sidebar.slider("Association: min support",
                                0.01, 0.30, 0.05, 0.01)
min_confidence = st.sidebar.slider("Association: min confidence",
                                   0.1, 1.0, 0.4, 0.05)

df_f = df[
    (df['MUNI_CITY'].isin(municipality_filter)) &
    (df['ZONE'].isin(zone_filter)) &
    (df['TENURE'].isin(tenure_filter))
].copy()

if len(df_f) < 10:
    st.warning("⚠️ Too few records after filtering. "
               "Adjust the sidebar filters.")
    st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "📈 EDA",
    "🤖 Model Training & Evaluation",
    "🔍 Feature Importance",
    "📉 Cross-Validation",
    "🔗 Association Rules",
    "💡 Recommendations",
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("📊 Executive Overview")
    st.caption("Input-Process-Output (IPO) Framework — Input Stage")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><p>Total Plantation Sites</p>'
        f'<h2>{len(df_f):,}</h2></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="metric-card"><p>Municipalities</p>'
        f'<h2>{df_f["MUNI_CITY"].nunique()}</h2></div>',
        unsafe_allow_html=True
    )
    c3.markdown(
        f'<div class="metric-card"><p>Barangays</p>'
        f'<h2>{df_f["BARANGAY"].nunique()}</h2></div>',
        unsafe_allow_html=True
    )
    c4.markdown(
        f'<div class="metric-card"><p>Commodity Classes</p>'
        f'<h2>{df_f["COMMODITY_CLEAN"].nunique()}</h2></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("📌 Plantation Sites by Municipality")
    mun_counts = df_f['MUNI_CITY'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    mun_counts.plot(kind='bar', color='#1E6F3C', edgecolor='black', ax=ax)
    ax.set_ylabel("Number of Plantation Sites")
    ax.set_title("NGP Plantation Sites per Municipality")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("🌱 Commodity Type Distribution (Target Variable)")
    col1, col2 = st.columns(2)
    with col1:
        comm_counts = df_f['COMMODITY_CLEAN'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#1E6F3C', '#2E8B57', '#3CB371', '#66CDAA',
                  '#90EE90', '#95A5A6', '#B0C4DE']
        ax.pie(comm_counts.values, labels=comm_counts.index,
               autopct='%1.1f%%', colors=colors[:len(comm_counts)],
               startangle=90)
        ax.set_title("Commodity Type Distribution")
        st.pyplot(fig)
        plt.close()
    with col2:
        comm_df = comm_counts.rename('Count').to_frame()
        comm_df['Percentage'] = (
            comm_df['Count'] / comm_df['Count'].sum() * 100
        ).round(2)
        st.dataframe(comm_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Site Characteristics Snapshot")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Zone Distribution:**")
        st.dataframe(df_f['ZONE'].value_counts().rename('Count').to_frame(),
                     use_container_width=True)
        st.write("**Tenure Type Distribution:**")
        st.dataframe(df_f['TENURE'].value_counts().rename('Count').to_frame(),
                     use_container_width=True)
    with col2:
        st.write("**Partner Organization Type:**")
        st.dataframe(df_f['PART_TYP'].value_counts().rename('Count').to_frame(),
                     use_container_width=True)
        st.write("**Area (hectares) Statistics:**")
        st.dataframe(
            df_f['AREA_HA'].describe().round(2).rename('Value').to_frame(),
            use_container_width=True
        )


# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
    st.header("📈 Exploratory Data Analysis")
    st.caption("Phase 3 of the Data Science Framework — exploring site patterns")

    st.subheader("🕐 Temporal Patterns")
    col1, col2 = st.columns(2)
    with col1:
        by_year = df_f.groupby('YR_ESTAB').size()
        fig, ax = plt.subplots(figsize=(7, 4))
        by_year.plot(kind='bar', color='#1E6F3C', edgecolor='black', ax=ax)
        ax.set_ylabel("Number of Sites")
        ax.set_title("Plantation Sites Established per Year")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        by_zone = df_f.groupby(['YR_ESTAB', 'ZONE']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        by_zone.plot(kind='bar', stacked=True, ax=ax,
                     color=['#2E8B57', '#90EE90', '#3CB371'])
        ax.set_ylabel("Number of Sites")
        ax.set_title("Sites per Year by Zone")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🏘️ Top 15 Barangays by Number of Plantation Sites")
    top_brgy = (df_f.groupby(['MUNI_CITY', 'BARANGAY']).size()
                .sort_values(ascending=False).head(15))
    top_brgy.index = [f"{m} – {b}" for m, b in top_brgy.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    top_brgy.sort_values().plot(kind='barh', color='#1E6F3C',
                                edgecolor='black', ax=ax)
    ax.set_xlabel("Number of Plantation Sites")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📦 Area (ha) Distribution by Commodity Type")
    fig, ax = plt.subplots(figsize=(10, 5))
    commodities = list(df_f['COMMODITY_CLEAN'].unique())
    data_to_plot = [df_f[df_f['COMMODITY_CLEAN'] == c]['AREA_HA']
                    .dropna().values for c in commodities]
    try:
        bp = ax.boxplot(data_to_plot, labels=commodities, patch_artist=True)
        colors_box = ['#1E6F3C', '#2E8B57', '#3CB371', '#66CDAA',
                      '#90EE90', '#95A5A6', '#B0C4DE']
        for patch, color in zip(bp['boxes'], colors_box[:len(commodities)]):
            patch.set_facecolor(color)
        ax.set_ylabel("Area (hectares)")
        ax.set_title("Area Distribution per Commodity Type")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not generate boxplot: {e}")

    st.markdown("---")
    st.subheader("🔗 Cross-Tabulations")
    st.write("**Zone vs Commodity:**")
    st.dataframe(pd.crosstab(df_f['ZONE'], df_f['COMMODITY_CLEAN']),
                 use_container_width=True)
    st.write("**Tenure vs Commodity:**")
    st.dataframe(pd.crosstab(df_f['TENURE'], df_f['COMMODITY_CLEAN']),
                 use_container_width=True)
    st.write("**Partner Type vs Commodity:**")
    st.dataframe(pd.crosstab(df_f['PART_TYP'], df_f['COMMODITY_CLEAN']),
                 use_container_width=True)


# ------------------------------------------------------------
# TAB 3: MODEL TRAINING & EVALUATION
# ------------------------------------------------------------
with tab3:
    st.header("🤖 Machine Learning Model Training & Evaluation")
    st.caption("Phase 4 & 5 of the DSF — Modeling and Evaluation")

    st.markdown(f"""
    **Model Configuration:**
    - Algorithm: Random Forest Classifier
    - Features: {', '.join(PREDICTIVE_FEATURES)}
    - Target: Commodity Type
    - Train/Test Split: {int((1-test_size)*100)}% / {int(test_size*100)}%
    - Cross-Validation: {cv_folds}-fold stratified
    """)

    X = df_f[PREDICTIVE_FEATURES].copy()
    y = df_f['COMMODITY_CLEAN'].copy()

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        st.error("Need at least 2 commodity classes to train a classifier. "
                 "Adjust filters.")
        st.stop()

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size,
            random_state=42, stratify=y_encoded
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant',
                                   fill_value='UNKNOWN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',
                                 sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    with st.spinner("Training Random Forest model..."):
        rf_pipeline.fit(X_train, y_train)
        y_pred = rf_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro',
                                 zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro',
                             zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted',
                                    zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted',
                                zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted',
                           zero_division=0)

    st.markdown("### 📊 Model Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision (Macro)", f"{prec_macro:.4f}")
    c3.metric("Recall (Macro)", f"{rec_macro:.4f}")
    c4.metric("F1-Score (Macro)", f"{f1_macro:.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Precision (Weighted)", f"{prec_weighted:.4f}")
    c6.metric("Recall (Weighted)", f"{rec_weighted:.4f}")
    c7.metric("F1-Score (Weighted)", f"{f1_weighted:.4f}")
    c8.metric("Classes Predicted", len(set(y_pred)))

    st.markdown("---")
    st.markdown("### 📋 Per-Class Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=le_target.classes_,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.session_state['rf_pipeline'] = rf_pipeline
    st.session_state['le_target'] = le_target
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['y_pred'] = y_pred


# ------------------------------------------------------------
# TAB 4: FEATURE IMPORTANCE
# ------------------------------------------------------------
with tab4:
    st.header("🔍 Feature Importance Analysis")
    st.caption("Explainable AI (XAI) — key site characteristics driving "
               "commodity suitability")

    if 'rf_pipeline' not in st.session_state:
        st.warning("⚠️ Please train the model in the "
                   "'Model Training & Evaluation' tab first.")
    else:
        rf_pipeline = st.session_state['rf_pipeline']
        preprocessor = rf_pipeline.named_steps['preprocessor']
        classifier = rf_pipeline.named_steps['classifier']

        try:
            cat_encoder = (preprocessor
                           .named_transformers_['cat']
                           .named_steps['onehot'])
            cat_feature_names = cat_encoder.get_feature_names_out(
                CATEGORICAL_FEATURES).tolist()
        except Exception:
            cat_feature_names = [f"cat_{i}" for i in range(100)]

        all_feature_names = NUMERIC_FEATURES + cat_feature_names
        importances = classifier.feature_importances_
        n_features = min(len(all_feature_names), len(importances))

        importance_df = pd.DataFrame({
            'Feature': all_feature_names[:n_features],
            'Importance': importances[:n_features]
        }).sort_values('Importance', ascending=False)

        st.markdown("### 🌲 Gini Importance (Top Features)")
        top_n = st.slider("Number of top features to display", 5, 30, 15)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(top_n).sort_values('Importance')
        ax.barh(top_features['Feature'], top_features['Importance'],
                color='#1E6F3C', edgecolor='black')
        ax.set_xlabel("Gini Importance")
        ax.set_title(f"Top {top_n} Feature Importances — Random Forest")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Top Feature Importance Table:**")
        st.dataframe(importance_df.head(top_n).round(5),
                     use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Aggregated Importance by Original Feature")

        def map_to_original(feat_name):
            for orig in CATEGORICAL_FEATURES:
                if feat_name.startswith(orig + '_'):
                    return orig
            if feat_name in NUMERIC_FEATURES:
                return feat_name
            return 'Other'

        importance_df['Original_Feature'] = (
            importance_df['Feature'].apply(map_to_original)
        )
        agg_importance = (importance_df.groupby('Original_Feature')['Importance']
                          .sum().sort_values(ascending=False))

        fig, ax = plt.subplots(figsize=(8, 5))
        agg_importance.plot(kind='bar', color='#2E8B57',
                            edgecolor='black', ax=ax)
        ax.set_ylabel("Aggregated Importance")
        ax.set_title("Aggregated Feature Importance by Original Variable")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            agg_importance.round(5).rename('Aggregated Importance').to_frame(),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 🧠 Interpretation for Decision-Makers")
        top_orig = agg_importance.head(3)
        if len(top_orig) >= 3:
            st.info(f"""
            **Key Drivers of Commodity Suitability:**

            Based on the Random Forest model, the most influential site
            characteristics are:

            1. **{top_orig.index[0]}** (importance: {top_orig.iloc[0]:.4f})
            2. **{top_orig.index[1]}** (importance: {top_orig.iloc[1]:.4f})
            3. **{top_orig.index[2]}** (importance: {top_orig.iloc[2]:.4f})

            These features should be prioritized when:
            - Planning new NGP plantation sites
            - Assessing suitability of existing sites
            - Training field staff on site-species matching
            """)


# ------------------------------------------------------------
# TAB 5: CROSS-VALIDATION
# ------------------------------------------------------------
with tab5:
    st.header("📉 Cross-Validation & Model Robustness")
    st.caption(f"{cv_folds}-fold stratified cross-validation to assess "
               f"model stability")

    if 'rf_pipeline' not in st.session_state:
        st.warning("⚠️ Please train the model in the "
                   "'Model Training & Evaluation' tab first.")
    else:
        rf_pipeline = st.session_state['rf_pipeline']
        le_target = st.session_state['le_target']

        X = df_f[PREDICTIVE_FEATURES].copy()
        y = df_f['COMMODITY_CLEAN'].copy()
        y_encoded = le_target.transform(y)

        min_class_count = pd.Series(y_encoded).value_counts().min()
        actual_folds = min(cv_folds, min_class_count)

        if actual_folds < 2:
            st.warning("Not enough samples per class for cross-validation. "
                       "Need at least 2 per class.")
        else:
            st.markdown(f"**Performing {actual_folds}-fold Stratified "
                        f"Cross-Validation...**")

            with st.spinner(f"Running {actual_folds}-fold cross-validation..."):
                cv = StratifiedKFold(n_splits=actual_folds,
                                     shuffle=True, random_state=42)
                cv_scores = cross_val_score(rf_pipeline, X, y_encoded,
                                            cv=cv, scoring='accuracy',
                                            n_jobs=-1)

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean CV Accuracy", f"{cv_scores.mean():.4f}")
            c2.metric("Std Dev", f"{cv_scores.std():.4f}")
            c3.metric("Folds", actual_folds)

            st.markdown("### 📊 Per-Fold Accuracy")
            fold_df = pd.DataFrame({
                'Fold': [f"Fold {i+1}" for i in range(len(cv_scores))],
                'Accuracy': cv_scores
            })
            st.dataframe(fold_df.round(4), use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(fold_df['Fold'], fold_df['Accuracy'],
                   color='#1E6F3C', edgecolor='black')
            ax.axhline(y=cv_scores.mean(), color='red',
                       linestyle='--',
                       label=f'Mean = {cv_scores.mean():.4f}')
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{actual_folds}-Fold Cross-Validation Accuracy")
            ax.set_ylim(0, 1)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.markdown("### 🎯 Hyperparameter Optimization (Grid Search)")

            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
            }
            st.write("**Parameter Grid:**")
            st.json(param_grid)

            if st.button("🚀 Run Grid Search (may take 1-3 minutes)"):
                with st.spinner("Running grid search with cross-validation..."):
                    try:
                        grid_search = GridSearchCV(
                            rf_pipeline, param_grid,
                            cv=min(actual_folds, 5),
                            scoring='accuracy', n_jobs=-1
                        )
                        grid_search.fit(X, y_encoded)
                        st.success(
                            f"✅ Best Parameters: {grid_search.best_params_}"
                        )
                        st.metric("Best CV Score",
                                  f"{grid_search.best_score_:.4f}")

                        results_df = pd.DataFrame(grid_search.cv_results_)
                        top_results = results_df[
                            ['params', 'mean_test_score', 'std_test_score']
                        ].sort_values('mean_test_score',
                                      ascending=False).head(10)
                        st.write("**Top 10 Parameter Combinations:**")
                        st.dataframe(top_results.round(4),
                                     use_container_width=True)
                    except Exception as e:
                        st.error(f"Grid search failed: {e}")


# ------------------------------------------------------------
# TAB 6: ASSOCIATION RULES
# ------------------------------------------------------------
with tab6:
    st.header("🔗 Association Rule Mining")
    st.caption("Discovering co-occurring site characteristics using "
               "Apriori and FP-Growth")

    transactions_df = pd.DataFrame({
        'MUNI': df_f['MUNI_CITY'].astype(str),
        'ZONE': df_f['ZONE'].astype(str),
        'TENURE': df_f['TENURE'].astype(str),
        'PARTNER': df_f['PART_TYP'].astype(str),
        'COMMODITY': df_f['COMMODITY_CLEAN'].astype(str),
    })

    transactions = transactions_df.values.tolist()
    transactions = [
        [str(x) for x in row if x and x != 'nan' and x != 'UNKNOWN']
        for row in transactions
    ]
    transactions = [t for t in transactions if len(t) >= 2]

    if len(transactions) < 20:
        st.warning("Not enough transactions after filtering.")
    else:
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_ary, columns=te.columns_)

        st.markdown("---")
        st.subheader("📌 Apriori Algorithm")

        try:
            frequent_ap = apriori(basket, min_support=min_support,
                                  use_colnames=True)
        except Exception as e:
            frequent_ap = pd.DataFrame()
            st.error(f"Apriori error: {e}")

        if frequent_ap.empty:
            st.warning(f"No frequent itemsets found with min support = "
                       f"{min_support}.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Frequent Itemsets", len(frequent_ap))
            c2.metric("Min Support", f"{min_support:.3f}")
            c3.metric("Min Confidence", f"{min_confidence:.2f}")

            st.markdown("**Top 10 Frequent Itemsets (by support):**")
            top_itemsets = (frequent_ap
                            .sort_values('support', ascending=False)
                            .head(10).copy())
            top_itemsets['itemsets'] = top_itemsets['itemsets'].apply(
                lambda x: ', '.join(sorted(x)))
            st.dataframe(top_itemsets, use_container_width=True)

            try:
                rules_ap = association_rules(
                    frequent_ap, metric='confidence',
                    min_threshold=min_confidence
                )
                rules_ap = rules_ap.sort_values('lift', ascending=False)
            except Exception:
                rules_ap = pd.DataFrame()

            if not rules_ap.empty:
                st.markdown(f"**Top 10 Apriori Rules "
                            f"(out of {len(rules_ap)}):**")
                display_rules = rules_ap.head(10).copy()
                display_rules['antecedents'] = (
                    display_rules['antecedents']
                    .apply(lambda x: ', '.join(sorted(x)))
                )
                display_rules['consequents'] = (
                    display_rules['consequents']
                    .apply(lambda x: ', '.join(sorted(x)))
                )
                st.dataframe(
                    display_rules[['antecedents', 'consequents',
                                   'support', 'confidence', 'lift']].round(4),
                    use_container_width=True
                )

        st.markdown("---")
        st.subheader("📌 FP-Growth Algorithm")

        try:
            frequent_fp = fpgrowth(basket, min_support=min_support,
                                   use_colnames=True)
        except Exception as e:
            frequent_fp = pd.DataFrame()
            st.error(f"FP-Growth error: {e}")

        if not frequent_fp.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Frequent Itemsets", len(frequent_fp))
            c2.metric("Min Support", f"{min_support:.3f}")
            c3.metric("Min Confidence", f"{min_confidence:.2f}")

            st.markdown("**Top 10 Frequent Itemsets (FP-Growth):**")
            top_fp = (frequent_fp
                      .sort_values('support', ascending=False)
                      .head(10).copy())
            top_fp['itemsets'] = top_fp['itemsets'].apply(
                lambda x: ', '.join(sorted(x)))
            st.dataframe(top_fp, use_container_width=True)

            try:
                rules_fp = association_rules(
                    frequent_fp, metric='confidence',
                    min_threshold=min_confidence
                )
                rules_fp = rules_fp.sort_values('lift', ascending=False)
            except Exception:
                rules_fp = pd.DataFrame()

            if not rules_fp.empty:
                st.markdown(f"**Top 10 FP-Growth Rules "
                            f"(out of {len(rules_fp)}):**")
                display_fp = rules_fp.head(10).copy()
                display_fp['antecedents'] = (
                    display_fp['antecedents']
                    .apply(lambda x: ', '.join(sorted(x)))
                )
                display_fp['consequents'] = (
                    display_fp['consequents']
                    .apply(lambda x: ', '.join(sorted(x)))
                )
                st.dataframe(
                    display_fp[['antecedents', 'consequents',
                                'support', 'confidence', 'lift']].round(4),
                    use_container_width=True
                )


# ------------------------------------------------------------
# TAB 7: RECOMMENDATIONS
# ------------------------------------------------------------
with tab7:
    st.header("💡 Recommendations & Decision Support")
    st.caption("Phase 6 of the DSF — Deployment/Communication of Insights")

    year_min_val = (int(df_f['YR_ESTAB'].min())
                    if df_f['YR_ESTAB'].notna().any() else 'N/A')
    year_max_val = (int(df_f['YR_ESTAB'].max())
                    if df_f['YR_ESTAB'].notna().any() else 'N/A')

    st.markdown(f"""
    ### 📊 Analysis Summary

    Based on **{len(df_f):,}** NGP plantation records from
    **{df_f['MUNI_CITY'].nunique()}** municipalities
    (**{', '.join(sorted(df_f['MUNI_CITY'].unique()))}**) covering
    **{year_min_val}–{year_max_val}**, using Random Forest classification:

    #### 1. For DENR / CENRO Bunawan
    - **Use the predictive model** to screen new NGP sites before
      commodity selection.
    - **Prioritize site characteristics** with highest feature
      importance when evaluating proposals.
    - **Integrate the model** into existing NGP planning workflows.
    - **Monitor model performance** annually and retrain with new data.

    #### 2. For NGP Implementing Bodies
    - **Adopt data-driven commodity selection** instead of relying
      solely on expert opinion.
    - **Use zone designation** (Production vs Protection) as a strong
      guide:
      - Protection zones → Indigenous species and Bamboo
      - Production zones → Timber, Agroforestry, High Value
    - **Consider tenure type** (CADT, CBFM, Untenured, PA) as it
      reflects community management capacity.

    #### 3. For Local Government Units (LGUs)
    - **Use the model outputs** to prioritize barangays for
      reforestation funding.
    - **Align commodity choices** with local ecological conditions to
      maximize seedling survival.
    - **Leverage the dashboard** for transparent, evidence-based
      decision-making.

    #### 4. For Forest Managers & Practitioners
    - **Use feature importance insights** to explain why certain
      commodities are recommended for specific sites.
    - **Avoid species-site mismatching** by consulting model
      predictions before procurement.
    - **Combine model outputs with local knowledge** for best results.

    #### 5. For People's Organizations & Indigenous Communities
    - **Advocate for native species** in Protection zones — the model
      supports this ecologically sound choice.
    - **Use the evidence** to negotiate for culturally important
      species in ancestral domain areas (CADT).

    #### 6. For Future Researchers
    - **Integrate environmental data** (soil, climate, topography) to
      improve model accuracy.
    - **Add survival rate data** as an additional target variable.
    - **Apply explainable AI techniques** (SHAP, LIME) for deeper
      interpretability.
    - **Develop a full decision support system** with GIS integration.

    #### 7. Limitations
    - Model relies on **historical CENRO Bunawan data** — may not
      generalize to other regions without retraining.
    - **Missing environmental variables** (soil pH, rainfall,
      elevation) limits predictive power.
    - **Class imbalance** in commodity types affects per-class
      performance.
    - **No primary data collection** — results depend on secondary
      data quality.
    """)

    st.markdown("---")
    st.subheader("📥 Download Results")

    try:
        if 'rf_pipeline' in st.session_state:
            y_pred = st.session_state['y_pred']
            y_test = st.session_state['y_test']
            le_target = st.session_state['le_target']

            results_df = pd.DataFrame({
                'Actual': le_target.inverse_transform(y_test),
                'Predicted': le_target.inverse_transform(y_pred)
            })
            results_df['Correct'] = (results_df['Actual']
                                     == results_df['Predicted'])

            st.download_button(
                "📊 Download Test Set Predictions (CSV)",
                results_df.to_csv(index=False).encode('utf-8'),
                "ngp_predictions.csv",
                "text/csv"
            )

        summary = pd.DataFrame({
            'Metric': ['Total Records', 'Municipalities', 'Barangays',
                       'Commodity Classes'],
            'Value': [
                len(df_f),
                df_f['MUNI_CITY'].nunique(),
                df_f['BARANGAY'].nunique(),
                df_f['COMMODITY_CLEAN'].nunique()
            ]
        })
        st.download_button(
            "📊 Download Summary (CSV)",
            summary.to_csv(index=False).encode('utf-8'),
            "ngp_summary.csv",
            "text/csv"
        )
    except Exception as e:
        st.warning(f"Could not prepare download: {e}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "🎓 Capstone Dashboard • Machine Learning-Based Prediction of Suitable "
    "Plantation Commodities for National Greening Program Sites • "
    "Paje, C.L.M. & Salvado, M.M. • Agusan del Sur State University • 2026"
)
