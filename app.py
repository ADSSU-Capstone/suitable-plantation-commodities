import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

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

TARGET_COL = 'COMMODITY_'
PREDICTIVE_FEATURES = ['MUNI_CITY', 'BARANGAY', 'ZONE', 'TENURE', 'PART_TYP', 'YR_ESTAB', 'AREA_HA']

COMMODITY_CLASSES = ['Timber', 'Agroforestry', 'High Value', 'Bamboo', 'Indigenous', 'Other']


# ============================================================
# DATA LOADING
# ============================================================
def _normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace('\ufeff', '').upper() for c in df.columns]
    return df


def _clean_commodity_label(val):
    """Collapse multi-label strings into a single primary class label."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == '' or s.lower() == 'nan':
        return None
    s_lower = s.lower()

    # Priority-ordered classification
    if 'bamboo' in s_lower:
        return 'Bamboo'
    if 'indigenous' in s_lower:
        return 'Indigenous'
    if 'agroforestry' in s_lower:
        return 'Agroforestry'
    if 'high value' in s_lower:
        return 'High Value'
    if 'timber' in s_lower or 'falcata' in s_lower or 'narra' in s_lower or 'lauan' in s_lower:
        return 'Timber'
    if 'fruit' in s_lower:
        return 'Other'
    return 'Other'


@st.cache_data(show_spinner=True)
def load_data():
    search_paths = ['.', '/content/data', os.path.dirname(os.path.abspath(__file__))]
    filepath = None
    for path in search_paths:
        candidate = os.path.join(path, DATA_FILE)
        if os.path.exists(candidate):
            filepath = candidate
            break
        # Also search loosely
        if os.path.isdir(path):
            for f in glob.glob(os.path.join(path, '*.xlsx')):
                if 'cenro' in os.path.basename(f).lower():
                    filepath = f
                    break
        if filepath:
            break

    if filepath is None:
        return None, "Data file not found. Please place the CENRO Bunawan Excel file in the app directory."

    try:
        xls = pd.ExcelFile(filepath)
        sheet_names = xls.sheet_names
        # Use 'Reforestation' sheet if available, else first
        target_sheet = 'Reforestation' if 'Reforestation' in sheet_names else sheet_names[0]
        df = pd.read_excel(xls, sheet_name=target_sheet, header=0)
        df = _normalize_columns(df)
        return df, None
    except Exception as e:
        return None, f"Error loading file: {e}"


# ============================================================
# PREPROCESSING
# ============================================================
@st.cache_data(show_spinner=True)
def preprocess(df):
    df = df.copy()

    # Drop rows without target
    df = df[df[TARGET_COL].notna()]
    df = df[df[TARGET_COL].astype(str).str.strip() != '']

    # Standardize column names for features
    rename_map = {}
    for col in df.columns:
        cu = col.upper().strip()
        if cu == 'MUNI_CITY': rename_map[col] = 'MUNI_CITY'
        elif cu == 'BARANGAY': rename_map[col] = 'BARANGAY'
        elif cu == 'ZONE': rename_map[col] = 'ZONE'
        elif cu == 'TENURE': rename_map[col] = 'TENURE'
        elif cu == 'PART_TYP': rename_map[col] = 'PART_TYP'
        elif cu == 'YR_ESTAB': rename_map[col] = 'YR_ESTAB'
        elif cu == 'AREA_HA': rename_map[col] = 'AREA_HA'
        elif cu == TARGET_COL.upper(): rename_map[col] = TARGET_COL
    df = df.rename(columns=rename_map)

    # Remove duplicate rows on key columns
    key_cols = [c for c in ['FID', 'UNIQ_ID', 'MUNI_CITY', 'BARANGAY', 'YR_ESTAB', TARGET_COL] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep='first')

    # Drop rows with >30% missing
    df = df.dropna(thresh=int(df.shape[1] * 0.7))

    # Clean categorical fields
    for col in ['MUNI_CITY', 'BARANGAY', 'ZONE', 'TENURE', 'PART_TYP']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace({'NAN': np.nan, '': np.nan, 'NONE': np.nan})
            df[col] = df[col].fillna('UNKNOWN')

    # Numeric fields
    df['YR_ESTAB'] = pd.to_numeric(df['YR_ESTAB'], errors='coerce')
    df['AREA_HA'] = pd.to_numeric(df['AREA_HA'], errors='coerce')

    # Impute numeric with median
    if df['YR_ESTAB'].isna().sum() > 0:
        df['YR_ESTAB'] = df['YR_ESTAB'].fillna(df['YR_ESTAB'].median())
    if df['AREA_HA'].isna().sum() > 0:
        df['AREA_HA'] = df['AREA_HA'].fillna(df['AREA_HA'].median())

    # Outlier capping for AREA_HA (winsorize at 1st and 99th percentile)
    if df['AREA_HA'].notna().sum() > 0:
        q1 = df['AREA_HA'].quantile(0.01)
        q99 = df['AREA_HA'].quantile(0.99)
        df['AREA_HA'] = df['AREA_HA'].clip(lower=q1, upper=q99)

    # Target: collapse multi-labels into primary class
    df['COMMODITY_CLEAN'] = df[TARGET_COL].apply(_clean_commodity_label)
    df = df[df['COMMODITY_CLEAN'].notna()]

    # Drop classes with fewer than 2 samples
    class_counts = df['COMMODITY_CLEAN'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df['COMMODITY_CLEAN'].isin(valid_classes)]

    return df


# ============================================================
# LOAD & PREPROCESS
# ============================================================
df_raw, load_err = load_data()

st.markdown('<div class="main-header">🌳 NGP Plantation Commodity Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning-Based Prediction of Suitable Plantation Commodities '
            'for National Greening Program Sites<br>'
            '<i>CENRO Bunawan, Agusan del Sur (2015–2025)</i></div>',
            unsafe_allow_html=True)

if df_raw is None:
    st.error(f"❌ {load_err}")
    st.info("Please upload the CENRO Bunawan dataset to the app directory and refresh.")
    st.stop()

df = preprocess(df_raw)

if len(df) < 10:
    st.error("❌ Not enough data after preprocessing to build a model. Please check the dataset.")
    st.stop()


# ============================================================
# DATA SUMMARY EXPANDER
# ============================================================
with st.expander("🔍 Data Summary", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Municipalities", df['MUNI_CITY'].nunique())
    c3.metric("Barangays", df['BARANGAY'].nunique())
    if df['YR_ESTAB'].notna().any():
        c4.metric("Years Covered", f"{int(df['YR_ESTAB'].min())}–{int(df['YR_ESTAB'].max())}")
    else:
        c4.metric("Years Covered", "N/A")

    st.write("**Commodity Class Distribution (Target):**")
    target_counts = df['COMMODITY_CLEAN'].value_counts()
    target_df = target_counts.rename('Count').to_frame()
    target_df['Percentage'] = (target_df['Count'] / target_df['Count'].sum() * 100).round(2)
    st.dataframe(target_df, use_container_width=True)

    st.write("**Records per Municipality:**")
    st.dataframe(df['MUNI_CITY'].value_counts().rename('Records').to_frame(), use_container_width=True)

    st.write("**Records per Year Established:**")
    st.dataframe(df['YR_ESTAB'].value_counts().sort_index().rename('Records').to_frame(), use_container_width=True)


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
n_estimators = st.sidebar.slider("Random Forest: n_estimators", 50, 500, 200, 50)
max_depth = st.sidebar.selectbox("Random Forest: max_depth", [None, 10, 20, 30], index=3)
min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 5, 1)

df_f = df[
    (df['MUNI_CITY'].isin(municipality_filter)) &
    (df['ZONE'].isin(zone_filter)) &
    (df['TENURE'].isin(tenure_filter))
].copy()

if len(df_f) < 10:
    st.warning("⚠️ Too few records after filtering. Adjust the sidebar filters.")
    st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 EDA",
    "🤖 Model Training & Evaluation",
    "🔍 Feature Importance",
    "📉 Cross-Validation",
    "💡 Recommendations"
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("📊 Executive Overview")
    st.caption("Input-Process-Output (IPO) Framework — Input Stage")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Plantation Sites</p><h2>{len(df_f):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Municipalities</p><h2>{df_f["MUNI_CITY"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Barangays</p><h2>{df_f["BARANGAY"].nunique()}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Commodity Classes</p><h2>{df_f["COMMODITY_CLEAN"].nunique()}</h2></div>', unsafe_allow_html=True)

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
        colors = ['#1E6F3C', '#2E8B57', '#3CB371', '#66CDAA', '#90EE90', '#95A5A6']
        ax.pie(comm_counts.values, labels=comm_counts.index, autopct='%1.1f%%',
               colors=colors[:len(comm_counts)], startangle=90)
        ax.set_title("Commodity Type Distribution")
        st.pyplot(fig)
        plt.close()
    with col2:
        comm_df = comm_counts.rename('Count').to_frame()
        comm_df['Percentage'] = (comm_df['Count'] / comm_df['Count'].sum() * 100).round(2)
        st.dataframe(comm_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Site Characteristics Snapshot")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Zone Distribution:**")
        st.dataframe(df_f['ZONE'].value_counts().rename('Count').to_frame(), use_container_width=True)
        st.write("**Tenure Type Distribution:**")
        st.dataframe(df_f['TENURE'].value_counts().rename('Count').to_frame(), use_container_width=True)
    with col2:
        st.write("**Partner Organization Type:**")
        st.dataframe(df_f['PART_TYP'].value_counts().rename('Count').to_frame(), use_container_width=True)
        st.write("**Area (hectares) Statistics:**")
        st.dataframe(df_f['AREA_HA'].describe().round(2).rename('Value').to_frame(), use_container_width=True)


# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
    st.header("📈 Exploratory Data Analysis")
    st.caption("Phase 3 of the Data Science Framework — exploring patterns in site characteristics")

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
        by_zone.plot(kind='bar', stacked=True, ax=ax, color=['#2E8B57', '#90EE90'])
        ax.set_ylabel("Number of Sites")
        ax.set_title("Sites per Year by Zone")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🏘️ Top 15 Barangays by Number of Plantation Sites")
    top_brgy = df_f.groupby(['MUNI_CITY', 'BARANGAY']).size() \
                   .sort_values(ascending=False).head(15)
    top_brgy.index = [f"{m} – {b}" for m, b in top_brgy.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    top_brgy.sort_values().plot(kind='barh', color='#1E6F3C', edgecolor='black', ax=ax)
    ax.set_xlabel("Number of Plantation Sites")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📦 Area (ha) Distribution by Commodity Type")
    fig, ax = plt.subplots(figsize=(10, 5))
    commodities = df_f['COMMODITY_CLEAN'].unique()
    data_to_plot = [df_f[df_f['COMMODITY_CLEAN'] == c]['AREA_HA'].dropna().values for c in commodities]
    try:
        bp = ax.boxplot(data_to_plot, labels=commodities, patch_artist=True)
        colors_box = ['#1E6F3C', '#2E8B57', '#3CB371', '#66CDAA', '#90EE90', '#95A5A6']
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
    st.subheader("🔗 Cross-Tabulation: Zone vs Commodity")
    ct_zone = pd.crosstab(df_f['ZONE'], df_f['COMMODITY_CLEAN'])
    st.dataframe(ct_zone, use_container_width=True)

    st.subheader("🔗 Cross-Tabulation: Tenure vs Commodity")
    ct_tenure = pd.crosstab(df_f['TENURE'], df_f['COMMODITY_CLEAN'])
    st.dataframe(ct_tenure, use_container_width=True)

    st.subheader("🔗 Cross-Tabulation: Partner Type vs Commodity")
    ct_part = pd.crosstab(df_f['PART_TYP'], df_f['COMMODITY_CLEAN'])
    st.dataframe(ct_part, use_container_width=True)


# ------------------------------------------------------------
# TAB 3: MODEL TRAINING & EVALUATION
# ------------------------------------------------------------
with tab3:
    st.header("🤖 Machine Learning Model Training & Evaluation")
    st.caption("Phase 4 & 5 of the Data Science Framework — Modeling and Evaluation")

    st.markdown(f"""
    **Model Configuration:**
    - Algorithm: Random Forest Classifier (ensemble of decision trees)
    - Features: {', '.join(PREDICTIVE_FEATURES)}
    - Target: Commodity Type
    - Train/Test Split: {int((1-test_size)*100)}% / {int(test_size*100)}%
    - Cross-Validation: {cv_folds}-fold stratified
    """)

    # Prepare X and y
    X = df_f[PREDICTIVE_FEATURES].copy()
    y = df_f['COMMODITY_CLEAN'].copy()

    # Check class distribution
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        st.error("Need at least 2 commodity classes to train a classifier. Adjust filters.")
        st.stop()

    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    # Train/Test split with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
    except ValueError:
        # Fallback if too few samples per class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )

    # Preprocessing pipeline
    numeric_features = ['YR_ESTAB', 'AREA_HA']
    categorical_features = ['MUNI_CITY', 'BARANGAY', 'ZONE', 'TENURE', 'PART_TYP']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Random Forest pipeline
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

    # Train
    with st.spinner("Training Random Forest model..."):
        rf_pipeline.fit(X_train, y_train)
        y_pred = rf_pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

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
    report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le_target.classes_, yticklabels=le_target.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Normalized confusion matrix
    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=le_target.classes_, yticklabels=le_target.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Store in session state for later tabs
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
    st.caption("Explainable AI (XAI) — identifying key site characteristics driving commodity suitability")

    if 'rf_pipeline' not in st.session_state:
        st.warning("⚠️ Please train the model in the 'Model Training & Evaluation' tab first.")
    else:
        rf_pipeline = st.session_state['rf_pipeline']
        le_target = st.session_state['le_target']

        # Extract feature names after preprocessing
        preprocessor = rf_pipeline.named_steps['preprocessor']
        classifier = rf_pipeline.named_steps['classifier']

        # Get feature names
        try:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(
                ['MUNI_CITY', 'BARANGAY', 'ZONE', 'TENURE', 'PART_TYP']
            ).tolist()
        except Exception:
            cat_feature_names = [f"cat_{i}" for i in range(100)]

        num_feature_names = ['YR_ESTAB', 'AREA_HA']
        all_feature_names = num_feature_names + cat_feature_names

        # Gini importance
        importances = classifier.feature_importances_
        n_features = min(len(all_feature_names), len(importances))
        importance_df = pd.DataFrame({
            'Feature': all_feature_names[:n_features],
            'Importance': importances[:n_features]
        }).sort_values('Importance', ascending=False)

        st.markdown("### 🌲 Gini Importance (Top 20 Features)")
        top_n = st.slider("Number of top features to display", 5, 30, 15)

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(top_n).sort_values('Importance')
        ax.barh(top_features['Feature'], top_features['Importance'], color='#1E6F3C', edgecolor='black')
        ax.set_xlabel("Gini Importance")
        ax.set_title(f"Top {top_n} Feature Importances — Random Forest")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Top Feature Importance Table:**")
        st.dataframe(importance_df.head(top_n).round(5), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Aggregated Importance by Original Feature")
        st.caption("One-hot encoded features are aggregated back to their source column")

        def map_to_original(feat_name):
            for orig in ['MUNI_CITY', 'BARANGAY', 'ZONE', 'TENURE', 'PART_TYP']:
                if feat_name.startswith(orig + '_'):
                    return orig
            if feat_name in ['YR_ESTAB', 'AREA_HA']:
                return feat_name
            return 'Other'

        importance_df['Original_Feature'] = importance_df['Feature'].apply(map_to_original)
        agg_importance = importance_df.groupby('Original_Feature')['Importance'].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        agg_importance.plot(kind='bar', color='#2E8B57', edgecolor='black', ax=ax)
        ax.set_ylabel("Aggregated Importance")
        ax.set_title("Aggregated Feature Importance by Original Variable")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(agg_importance.round(5).rename('Aggregated Importance').to_frame(), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧠 Interpretation for Decision-Makers")
        top_orig = agg_importance.head(3)
        st.info(f"""
        **Key Drivers of Commodity Suitability:**
        
        Based on the Random Forest model, the most influential site characteristics are:
        
        1. **{top_orig.index[0]}** (importance: {top_orig.iloc[0]:.4f})
        2. **{top_orig.index[1]}** (importance: {top_orig.iloc[1]:.4f})
        3. **{top_orig.index[2]}** (importance: {top_orig.iloc[2]:.4f})
        
        These features should be prioritized when:
        - Planning new NGP plantation sites
        - Assessing suitability of existing sites for commodity diversification
        - Training field staff on site-species matching
        """)


# ------------------------------------------------------------
# TAB 5: CROSS-VALIDATION
# ------------------------------------------------------------
with tab5:
    st.header("📉 Cross-Validation & Model Robustness")
    st.caption("10-fold stratified cross-validation to assess model stability and prevent overfitting")

    if 'rf_pipeline' not in st.session_state:
        st.warning("⚠️ Please train the model in the 'Model Training & Evaluation' tab first.")
    else:
        rf_pipeline = st.session_state['rf_pipeline']
        le_target = st.session_state['le_target']

        X = df_f[PREDICTIVE_FEATURES].copy()
        y = df_f['COMMODITY_CLEAN'].copy()
        y_encoded = le_target.transform(y)

        # Determine minimum class count
        min_class_count = pd.Series(y_encoded).value_counts().min()
        actual_folds = min(cv_folds, min_class_count)

        if actual_folds < 2:
            st.warning("Not enough samples per class for cross-validation. Need at least 2 per class.")
        else:
            st.markdown(f"**Performing {actual_folds}-fold Stratified Cross-Validation...**")

            with st.spinner(f"Running {actual_folds}-fold cross-validation..."):
                cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(rf_pipeline, X, y_encoded, cv=cv,
                                            scoring='accuracy', n_jobs=-1)

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
            ax.bar(fold_df['Fold'], fold_df['Accuracy'], color='#1E6F3C', edgecolor='black')
            ax.axhline(y=cv_scores.mean(), color='red', linestyle='--',
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
            st.caption("Systematic search over specified parameter values with cross-validation")

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
                            rf_pipeline, param_grid, cv=min(actual_folds, 5),
                            scoring='accuracy', n_jobs=-1, verbose=0
                        )
                        grid_search.fit(X, y_encoded)

                        st.success(f"✅ Best Parameters: {grid_search.best_params_}")
                        st.metric("Best CV Score", f"{grid_search.best_score_:.4f}")

                        # Show top results
                        results_df = pd.DataFrame(grid_search.cv_results_)
                        top_results = results_df[['params', 'mean_test_score', 'std_test_score']] \
                            .sort_values('mean_test_score', ascending=False).head(10)
                        st.write("**Top 10 Parameter Combinations:**")
                        st.dataframe(top_results.round(4), use_container_width=True)

                        # Store best model
                        st.session_state['best_model'] = grid_search.best_estimator_
                    except Exception as e:
                        st.error(f"Grid search failed: {e}")


# ------------------------------------------------------------
# TAB 6: RECOMMENDATIONS
# ------------------------------------------------------------
with tab6:
    st.header("💡 Recommendations & Decision Support")
    st.caption("Phase 6 of the Data Science Framework — Deployment/Communication of Insights")

    st.markdown(f"""
    ### 📊 Analysis Summary

    Based on **{len(df_f):,}** NGP plantation records from
    **{df_f['MUNI_CITY'].nunique()}** municipalities
    (**{', '.join(sorted(df_f['MUNI_CITY'].unique()))}**) covering
    **{int(df_f['YR_ESTAB'].min())}–{int(df_f['YR_ESTAB'].max())}**,
    using Random Forest classification:

    #### 1. For DENR / CENRO Bunawan
    - **Use the predictive model** to screen new NGP sites before commodity selection.
    - **Prioritize site characteristics** with highest feature importance when
      evaluating proposals.
    - **Integrate the model** into existing NGP planning workflows via a simple
      web interface (this dashboard can be extended).
    - **Monitor model performance** annually and retrain with new plantation data.

    #### 2. For NGP Implementing Bodies
    - **Adopt data-driven commodity selection** instead of relying solely on
      expert opinion.
    - **Use zone designation** (Production vs Protection) as a strong guide:
      Protection zones → Indigenous species and Bamboo
      Production zones → Timber, Agroforestry, High Value
    - **Consider tenure type** (CADT, CBFM, Untenured, PA) as it reflects
      community management capacity.

    #### 3. For Local Government Units (LGUs)
    - **Use the model outputs** to prioritize barangays for reforestation funding.
    - **Align commodity choices** with local ecological conditions to maximize
      seedling survival.
    - **Leverage the dashboard** for transparent, evidence-based decision-making.

    #### 4. For Forest Managers & Practitioners
    - **Use feature importance insights** to explain why certain commodities
      are recommended for specific sites.
    - **Avoid species-site mismatching** by consulting model predictions before
      procurement.
    - **Combine model outputs with local knowledge** for best results.

    #### 5. For People's Organizations & Indigenous Communities
    - **Advocate for native species** in Protection zones — the model supports
      this ecologically sound choice.
    - **Use the evidence** to negotiate for culturally important species in
      ancestral domain areas (CADT).

    #### 6. For Future Researchers
    - **Integrate environmental data** (soil, climate, topography) to improve
      model accuracy.
    - **Add survival rate data** as an additional target variable for regression.
    - **Apply explainable AI techniques** (SHAP, LIME) for deeper interpretability.
    - **Develop a full decision support system** with GIS integration.

    #### 7. Limitations
    - Model relies on **historical CENRO Bunawan data** — may not generalize
      to other regions without retraining.
    - **Missing environmental variables** (soil pH, rainfall, elevation) limits
      predictive power.
    - **Class imbalance** in commodity types affects per-class performance.
    - **No primary data collection** — results depend on secondary data quality.
    """)

    st.markdown("---")
    st.subheader("📥 Download Results")

    # Prepare download data
    try:
        if 'rf_pipeline' in st.session_state:
            y_pred = st.session_state['y_pred']
            y_test = st.session_state['y_test']
            le_target = st.session_state['le_target']

            results_df = pd.DataFrame({
                'Actual': le_target.inverse_transform(y_test),
                'Predicted': le_target.inverse_transform(y_pred)
            })
            results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

            st.download_button(
                "📊 Download Test Set Predictions (CSV)",
                results_df.to_csv(index=False).encode('utf-8'),
                "ngp_predictions.csv", "text/csv"
            )

        # Summary metrics
        summary = pd.DataFrame({
            'Metric': ['Total Records', 'Municipalities', 'Barangays', 'Commodity Classes'],
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
            "ngp_summary.csv", "text/csv"
        )
    except Exception as e:
        st.warning(f"Could not prepare download: {e}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "🎓 Capstone Dashboard • Machine Learning-Based Prediction of Suitable Plantation Commodities "
    "for National Greening Program Sites • "
    "Paje, C.L.M. & Salvado, M.M. • Agusan del Sur State University • 2026"
)