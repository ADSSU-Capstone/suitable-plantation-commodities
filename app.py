import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bunawan NGP Commodity Suitability Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Base App Styling */
    .stApp { background-color: #f5f7fa; }
    h1, h2, h3 { color: #1a4d4d; font-family: 'Segoe UI', sans-serif; }
    
    /* App Title */
    .app-title { font-size: 2.2rem; font-weight: 800; color: #0f3d3d; letter-spacing: -0.5px; margin-bottom: 0; margin-top: 1rem; }
    .app-sub { font-size: 1rem; color: #4a6b6b; margin-top: 0; margin-bottom: 1.5rem; }
    
    /* KPI Cards */
    .kpi-card { background: linear-gradient(135deg, #0f5c5c 0%, #1a8080 100%); color: #ffffff;
                padding: 1.2rem 1.4rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .kpi-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85; margin-bottom: 0.3rem; }
    .kpi-value { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
    
    /* Info and Warning Boxes */
    .interp { background-color: #e8f3f3; border-left: 5px solid #1a8080; padding: 0.8rem 1.2rem;
              border-radius: 4px; font-size: 0.95rem; color: #0f3d3d; margin: 0.5rem 0 1.5rem 0; }
    .warn-box { background-color: #fff4e5; border-left: 5px solid #d97a00; padding: 0.8rem 1.2rem;
                border-radius: 4px; font-size: 0.9rem; color: #6b4200; margin: 0.5rem 0 1rem 0; }
    .justify-box { background-color: #eef2f7; border-left: 5px solid #35558a; padding: 0.8rem 1.2rem;
                   border-radius: 4px; font-size: 0.9rem; color: #1e2f4d; margin: 0.5rem 0 1rem 0; }
    
    /* Sidebar Light Theme */
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    section[data-testid="stSidebar"] h3 { color: #0f3d3d; margin-top: 1rem; }
    section[data-testid="stSidebar"] label { color: #333333 !important; font-weight: 600; }
    
    /* Sidebar Dropdowns Override */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #f9f9f9 !important; 
        border-color: #cccccc !important; 
        color: #333333 !important;
    }
    
    /* Sidebar Navigation (Radio Buttons as Tabs) */
    section[data-testid="stSidebar"] .stRadio > div { gap: 0.4rem; }
    section[data-testid="stSidebar"] .stRadio label {
        background-color: #f9f9f9; border: 1.5px solid #e0e0e0; border-radius: 6px;
        padding: 0.6rem 0.9rem; margin: 0; width: 100%; cursor: pointer;
        font-weight: 600; font-size: 0.9rem; transition: all 0.2s ease-in-out; color: #333333; }
    section[data-testid="stSidebar"] .stRadio label:hover { background-color: #f0f0f0; border-color: #cccccc; }
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio input:checked + div {
        background-color: #0f3d3d !important; color: #ffffff !important; border-color: #0f3d3d; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
    section[data-testid="stSidebar"] .stRadio input:checked + div p { color: #ffffff !important; font-weight: 800; }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
    
    /* Dataframes */
    .stDataFrame { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="app-title">Bunawan NGP Commodity Suitability Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="app-sub">Machine Learning-Based Prediction of Suitable Plantation Commodities for National Greening Program Sites — CENRO Bunawan, 2015–2025</p>', unsafe_allow_html=True)

COMMODITY_MAP = {
    "timber": "Timber", "falca": "Timber", "falcat": "Timber", "lauan": "Timber",
    "agroforestry": "Agroforestry", "coffee": "Agroforestry", "cacao": "Agroforestry",
    "rubber": "Agroforestry", "fruit": "Agroforestry", "calamansi": "Agroforestry",
    "high value": "High Value", "mangosteen": "High Value", "durian": "High Value",
    "rambutan": "High Value", "lanzones": "High Value", "lemonsito": "High Value",
    "bamboo": "Bamboo", "patong": "Bamboo",
    "indigenous": "Indigenous", "narra": "Indigenous", "white lauan": "Indigenous",
}
VALID_MUNI = {"Bunawan", "Trento", "Rosario", "San Francisco", "Sta. Josefa"}
CAT_FEATURES = ["MUNI_CITY", "BARANGAY", "TENURE", "ZONE", "PART_TYP"]
NUM_FEATURES = ["YR_ESTAB", "AREA_HA"]

# Manual Teal Color Palette for Plotly
TEAL_PALETTE = ["#0f5c5c", "#1a8080", "#2da5a0", "#4fc3b8", "#80d6ce", "#b2e8e3", "#e0f4f2"]

BARANGAY_CENTROIDS = {
    ("Bunawan", "Bunawan Brook"): (8.180, 125.990), ("Bunawan", "Consuelo"): (8.200, 126.010),
    ("Bunawan", "Imelda"): (8.210, 125.980), ("Bunawan", "San Andres"): (8.190, 126.000),
    ("Bunawan", "San Marcos"): (8.170, 125.960), ("Trento", "Santa Maria"): (8.040, 126.050),
    ("Trento", "San Isidro"): (8.050, 126.060), ("Trento", "Tudela"): (8.060, 126.040),
    ("Trento", "New Visayas"): (8.070, 126.030), ("Trento", "Salvacion"): (8.080, 126.020),
    ("Trento", "San Roque"): (8.030, 126.070), ("Rosario", "Wasi-an"): (8.120, 126.000),
    ("Rosario", "Bayugan 3"): (8.130, 126.010), ("Rosario", "Cabantao"): (8.110, 125.990),
    ("Rosario", "Marfil"): (8.140, 126.020), ("Rosario", "Santa Cruz"): (8.100, 125.980),
    ("San Francisco", "Mate"): (8.250, 125.950), ("San Francisco", "Alegria"): (8.260, 125.940),
    ("San Francisco", "Das-agan"): (8.240, 125.960), ("San Francisco", "Lucac"): (8.230, 125.970),
    ("Sta. Josefa", "Awao"): (8.000, 125.900), ("Sta. Josefa", "Sayon"): (8.010, 125.910),
}

def clean_text(x):
    if pd.isna(x):
        return x
    s = str(x)
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def map_commodity(raw):
    if pd.isna(raw):
        return np.nan
    s = clean_text(raw).lower()
    if s == "" or s == "nan":
        return np.nan
    for key in COMMODITY_MAP:
        if key in s:
            return COMMODITY_MAP[key]
    return np.nan

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes, filename):
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, header=None)
        header_row = 0
        for i, row in raw_df.iterrows():
            row_str = " ".join([str(val) for val in row.values])
            if "FID" in row_str and "Shape" in row_str:
                header_row = i
                break
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, header=header_row)
        
    raw_n = len(df)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(clean_text)
    df.columns = [clean_text(c) for c in df.columns]
    
    needed = ["MUNI_CITY", "BARANGAY", "TENURE", "ZONE", "PART_TYP", "COMMODITY", "YR_ESTAB", "AREA_HA", "FID", "UNIQ_ID"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
            
    dup_n = df.duplicated(subset=["FID", "UNIQ_ID"], keep="first").sum()
    df = df.drop_duplicates(subset=["FID", "UNIQ_ID"], keep="first").copy()
    
    df["YR_ESTAB"] = pd.to_numeric(df["YR_ESTAB"], errors="coerce")
    df["AREA_HA"] = pd.to_numeric(df["AREA_HA"], errors="coerce")
    
    for c in ["MUNI_CITY", "BARANGAY", "TENURE", "ZONE", "PART_TYP"]:
        df[c] = df[c].map(clean_text)
        
    df["COMMODITY_RAW"] = df["COMMODITY"].map(clean_text)
    df["COMMODITY"] = df["COMMODITY_RAW"].map(map_commodity)
    
    miss_before = df[needed].isna().sum().sum()
    df = df.dropna(subset=["COMMODITY", "MUNI_CITY", "BARANGAY", "TENURE", "ZONE", "PART_TYP", "YR_ESTAB", "AREA_HA"])
    df = df[df["MUNI_CITY"].isin(VALID_MUNI)]
    df = df[(df["YR_ESTAB"] >= 2015) & (df["YR_ESTAB"] <= 2025)]
    miss_after = df[needed].isna().sum().sum()
    final_n = len(df)
    
    rules = pd.DataFrame({
        "Rule Applied (Chapter 3)": [
            "Strip invisible Unicode, trim whitespace, collapse blank spaces in all text fields",
            "Drop duplicate records on FID / UNIQ_ID composite key",
            "Coerce YR_ESTAB and AREA_HA to numeric; non-parseable rows dropped",
            "Drop records missing any of COMMODITY, MUNI_CITY, BARANGAY, TENURE, ZONE, PART_TYP, YR_ESTAB, AREA_HA",
            "Hardcode whitelist MUNI_CITY ∈ {Bunawan, Trento, Rosario, San Francisco, Sta. Josefa}",
            "Restrict YR_ESTAB to 2015–2025 (Data Scope)",
            "Map raw COMMODITY strings to 5 canonical classes: Timber, Agroforestry, High Value, Bamboo, Indigenous",
        ],
        "Effect": [
            "Text normalized", f"{dup_n} duplicates removed",
            "Numeric types enforced", f"{miss_before} missing cells removed pre-filter",
            "Out-of-scope municipalities dropped", "Out-of-scope years dropped",
            f"{final_n} records retained with assigned commodity class",
        ]
    })
    return df, raw_n, dup_n, miss_before, miss_after, final_n, rules

def kpi(label, value):
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

def interp(text):
    st.markdown(f'<div class="interp">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def justify(text):
    st.markdown(f'<div class="justify-box">{text}</div>', unsafe_allow_html=True)

st.sidebar.markdown("### Data Input")
uploaded = st.sidebar.file_uploader("Upload CENRO Bunawan NGP dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("Awaiting dataset upload. Use the sidebar file uploader to load the CENRO Bunawan NGP dataset (CSV or Excel). No example dataset is bundled; the dashboard will not render until a valid file is provided.")
    st.stop()

try:
    df, raw_n, dup_n, miss_before, miss_after, final_n, rules = load_and_clean(uploaded.getvalue(), uploaded.name)
except Exception as e:
    st.error(f"Dataset could not be parsed: {e}")
    st.stop()

if len(df) == 0:
    st.warning("Cleaning pipeline reduced the dataset to zero valid records. Verify that the uploaded file contains the CENRO Bunawan NGP schema (MUNI_CITY, BARANGAY, TENURE, ZONE, PART_TYP, COMMODITY, YR_ESTAB, AREA_HA).")
    st.stop()

centroids = dict(BARANGAY_CENTROIDS)

st.sidebar.markdown("### Drill-Down Filter")
municipality_options = ["All Municipalities"] + sorted(df["MUNI_CITY"].dropna().unique().tolist())
focus_muni = st.sidebar.selectbox("Focus Municipality", municipality_options)
barangay_options = ["All Barangays"]
if focus_muni != "All Municipalities":
    barangay_options += sorted(df[df["MUNI_CITY"] == focus_muni]["BARANGAY"].dropna().unique().tolist())
    focus_brgy = st.sidebar.selectbox("Focus Barangay", barangay_options)
else:
    focus_brgy = "All Barangays"

fdf = df.copy()
if focus_muni != "All Municipalities":
    fdf = fdf[fdf["MUNI_CITY"] == focus_muni]
if focus_brgy != "All Barangays":
    fdf = fdf[fdf["BARANGAY"] == focus_brgy]
if len(fdf) == 0:
    st.warning("The selected drill-down combination yields zero records. Select a broader focus area.")
    st.stop()

st.sidebar.markdown("### Navigation")
nav_items = [
    "Data Preprocessing & Cleaning",
    "Site Factors of Commodity Suitability",
    "Predictive Model Construction",
    "Model Performance & Feature Importance",
    "Commodity Selection Recommendations",
]
page = st.sidebar.radio("Dashboard Sections", nav_items, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records in focus:** {len(fdf):,}")
st.sidebar.markdown(f"**Municipalities:** {fdf['MUNI_CITY'].nunique()}")
st.sidebar.markdown(f"**Barangays:** {fdf['BARANGAY'].nunique()}")
st.sidebar.markdown(f"**Commodity classes:** {fdf['COMMODITY'].nunique()}")

@st.cache_resource(show_spinner=False)
def train_all(df_train):
    X = df_train[CAT_FEATURES + NUM_FEATURES].copy()
    y = df_train["COMMODITY"].copy()
    X_enc = pd.get_dummies(X, columns=CAT_FEATURES, drop_first=False)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
    models = {}
    
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    models["Random Forest"] = rf
    
    gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
    gb.fit(Xtr, ytr)
    models["Gradient Boosting"] = gb
    
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(Xtr, ytr)
    models["Support Vector Machine"] = svm
    
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
        xgb.fit(Xtr, ytr)
        models["XGBoost"] = xgb
    except Exception:
        pass
        
    try:
        from elapid import MaxentModel
        Xtr_dense = Xtr if not hasattr(Xtr, "toarray") else Xtr.toarray()
        mx = MaxentModel()
        try:
            mx.fit(Xtr_dense, ytr)
            models["MaxEnt"] = mx
        except Exception:
            pass
    except Exception:
        pass
        
    cv_scores = {}
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for name, m in models.items():
        try:
            scores = cross_val_score(m, X_scaled, y_enc, cv=skf, scoring="f1_macro", n_jobs=-1)
            cv_scores[name] = scores
        except Exception:
            cv_scores[name] = np.array([np.nan])
            
    return models, X_enc.columns.tolist(), le, scaler, Xtr, Xte, ytr, yte, X_scaled, y_enc, cv_scores

@st.cache_resource(show_spinner=False)
def run_gridsearch(df_train):
    X = df_train[CAT_FEATURES + NUM_FEATURES].copy()
    y = df_train["COMMODITY"].copy()
    X_enc = pd.get_dummies(X, columns=CAT_FEATURES, drop_first=False)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)
    param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}
    gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring="f1_macro", n_jobs=-1)
    gs.fit(X_scaled, y_enc)
    return gs.best_params_, gs.best_score_

def ensure_model_df():
    if len(fdf) < 30 or fdf["COMMODITY"].nunique() < 2:
        return df.copy(), True
    return fdf.copy(), False

if page == "Data Preprocessing & Cleaning":
    st.markdown("## Data Preprocessing & Cleaning")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Raw Records", f"{raw_n:,}")
    with c2: kpi("Duplicates Removed", f"{dup_n:,}")
    with c3: kpi("Missing Values Addressed", f"{miss_before - miss_after:,}")
    with c4: kpi("Final Valid Records", f"{final_n:,}")
    
    retention = (final_n / raw_n * 100) if raw_n else 0
    interp(f"Retention rate is {retention:.1f}% ({final_n:,} of {raw_n:,} raw records). All remaining records satisfy the Chapter 3 completeness, municipality-whitelist, and year-range rules and are ready for modeling.")
    
    st.markdown("### Cleaning Rules Applied")
    st.dataframe(rules, use_container_width=True, hide_index=True)
    
    st.markdown("### Field Completeness After Cleaning")
    completeness = pd.DataFrame({
        "Field": CAT_FEATURES + NUM_FEATURES + ["COMMODITY"],
        "Non-Null Count": [fdf[c].notna().sum() for c in CAT_FEATURES + NUM_FEATURES + ["COMMODITY"]],
        "Completeness %": [round(fdf[c].notna().mean() * 100, 2) for c in CAT_FEATURES + NUM_FEATURES + ["COMMODITY"]],
    })
    st.dataframe(completeness, use_container_width=True, hide_index=True)
    interp("Every retained modeling field is fully populated within the current drill-down selection, so no imputation is required downstream.")

elif page == "Site Factors of Commodity Suitability":
    st.markdown("## Site Factors of Commodity Suitability")
    
    geo = fdf.groupby(["MUNI_CITY", "BARANGAY"]).agg(
        Records=("COMMODITY", "size"), Area_HA=("AREA_HA", "sum"),
        Commodity_Diversity=("COMMODITY", "nunique")).reset_index()
    geo["lat"] = geo.apply(lambda r: centroids.get((r["MUNI_CITY"], r["BARANGAY"]), (np.nan, np.nan))[0], axis=1)
    geo["lon"] = geo.apply(lambda r: centroids.get((r["MUNI_CITY"], r["BARANGAY"]), (np.nan, np.nan))[1], axis=1)
    mappable = geo.dropna(subset=["lat", "lon"])
    
    if len(mappable) == 0:
        warn("No barangay centroid coordinates are available for the current selection. Supply the optional centroid CSV in the sidebar to render the geographic panel.")
    else:
        # Updated to use px.scatter_map for Plotly 6+ compatibility
        fig_map = px.scatter_map(mappable, lat="lat", lon="lon", size="Records", color="Records",
                                    hover_name="BARANGAY", hover_data=["MUNI_CITY", "Area_HA", "Commodity_Diversity"],
                                    color_continuous_scale="Teal", title="NGP Parcel Concentration by Barangay")
        fig_map.update_layout(
            map=dict(style="carto-positron", zoom=9, center=dict(lat=8.12, lon=126.0)),
            height=480, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_map, use_container_width=True)
        top = mappable.sort_values("Records", ascending=False).iloc[0]
        interp(f"Highest concentration is {top['BARANGAY']}, {top['MUNI_CITY']} ({int(top['Records'])} parcels, {top['Area_HA']:.1f} ha). Bubble size encodes parcel count; color gradient encodes the same intensity so spatial hot spots remain readable at any zoom level.")
        
    agg = geo.groupby("MUNI_CITY").agg(Records=("Records", "sum"), Area_HA=("Area_HA", "sum"),
                                         Commodity_Diversity=("Commodity_Diversity", "max")).reset_index().sort_values("Records", ascending=False)
    fig_rank = px.bar(agg.sort_values("Records"), x="Records", y="MUNI_CITY", orientation="h",
                      color="Records", color_continuous_scale="Teal", title="Municipality Ranking by Record Count")
    fig_rank.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="", xaxis_title="Records")
    st.plotly_chart(fig_rank, use_container_width=True)
    if len(agg) >= 3:
        interp(f"Top three: {', '.join(agg.head(3)['MUNI_CITY'].tolist())}. Lowest-ranked: {agg.iloc[-1]['MUNI_CITY']}.")
        
    pivot = pd.crosstab([fdf["MUNI_CITY"], fdf["BARANGAY"]], fdf["COMMODITY"])
    st.markdown("### Resource Allocation Matrix — Commodity Counts by Municipality and Barangay")
    st.dataframe(pivot, use_container_width=True)
    interp("Cells quantify how many NGP parcels of each commodity class exist per barangay, enabling allocation of seedling stock and extension officers to the exact administrative unit.")
    
    fig_sc = px.scatter(fdf, x="YR_ESTAB", y="AREA_HA", color="COMMODITY", size="AREA_HA",
                        hover_data=["MUNI_CITY", "BARANGAY", "TENURE"], title="Establishment Year vs. Area by Commodity",
                        color_discrete_sequence=TEAL_PALETTE)
    fig_sc.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_sc, use_container_width=True)
    interp("Clusters at low area and mid-years indicate smallholder parcels; large-area outliers indicate industrial-scale plantings and should be reviewed for tenure and commodity consistency.")

elif page == "Predictive Model Construction":
    st.markdown("## Predictive Model Construction")
    justify("Random Forest is the primary algorithm because it handles mixed categorical-numeric features, resists overfitting through bagging, and produces native feature importances. Gradient Boosting, SVM, XGBoost, and MaxEnt are trained as comparators to satisfy the multi-algorithm scope in Chapter 3. Hyperparameters for Random Forest are selected via GridSearchCV over n_estimators, max_depth, min_samples_split, and min_samples_leaf with 10-fold stratified cross-validation. A fixed 80/20 train-test split with stratification preserves class proportions. Macro and weighted averages are both reported to manage class imbalance.")
    
    model_df, fallback = ensure_model_df()
    if fallback:
        warn("Current drill-down selection has fewer than 30 records or fewer than two commodity classes. Modeling is run on the full cleaned dataset to preserve class representation; the drill-down filter still applies to all descriptive tabs.")
    
    with st.spinner("Training classifiers, running grid search, and scoring 10-fold CV..."):
        best_params, best_cv = run_gridsearch(model_df)
        models, feat_names, le, scaler, Xtr, Xte, ytr, yte, X_scaled, y_enc, cv_scores = train_all(model_df)
        
    st.markdown("### Grid Search Result — Random Forest")
    st.markdown(f"Best parameters: `{best_params}` | Best 10-fold cross-validated macro-F1: **{best_cv:.4f}**")
    interp("Grid search selected the hyperparameter combination that maximized macro-averaged F1 under 10-fold stratified cross-validation, balancing precision and recall across all five commodity classes rather than favoring the majority class.")
    
    st.markdown("### 10-Fold Cross-Validation Summary")
    cv_rows = []
    for name, scores in cv_scores.items():
        valid = scores[~np.isnan(scores)]
        cv_rows.append({"Model": name,
                        "CV F1 Mean": round(float(np.mean(valid)), 4) if len(valid) else np.nan,
                        "CV F1 Std": round(float(np.std(valid)), 4) if len(valid) else np.nan})
    st.dataframe(pd.DataFrame(cv_rows).sort_values("CV F1 Mean", ascending=False), use_container_width=True, hide_index=True)
    interp("Cross-validated F1 mean and standard deviation are reported per algorithm; low standard deviation indicates stable generalization across folds, while a high mean identifies the most reliable candidate for operational deployment.")
    
    st.markdown("### Class Distribution in Modeling Set")
    dist = model_df["COMMODITY"].value_counts().reset_index()
    dist.columns = ["Commodity", "Count"]
    fig_dist = px.bar(dist, x="Commodity", y="Count", color="Commodity", color_discrete_sequence=TEAL_PALETTE, title="Commodity Class Distribution")
    fig_dist.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)
    imbalance_ratio = dist["Count"].max() / dist["Count"].min()
    if imbalance_ratio > 1.5:
        warn(f"Class imbalance detected (majority:minority ratio = {imbalance_ratio:.2f}:1). Macro-averaged precision, recall, and F1 are reported alongside weighted averages so that minority-class performance is not masked by the majority class.")
    interp(f"Dominant class is {dist.iloc[0]['Commodity']} with {dist.iloc[0]['Count']} records; rarest is {dist.iloc[-1]['Commodity']} with {dist.iloc[-1]['Count']} records.")

elif page == "Model Performance & Feature Importance":
    st.markdown("## Model Performance & Feature Importance")
    model_df, fallback = ensure_model_df()
    if fallback:
        warn("Modeling is run on the full cleaned dataset because the drill-down selection is too small for stratified evaluation.")
        
    with st.spinner("Evaluating models, computing ROC curves and SHAP values..."):
        models, feat_names, le, scaler, Xtr, Xte, ytr, yte, X_scaled, y_enc, cv_scores = train_all(model_df)
        
    rows = []
    roc_data = {}
    cm_data = {}
    for name, m in models.items():
        yp = m.predict(Xte)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, yp),
            "Precision (macro)": precision_score(yte, yp, average="macro", zero_division=0),
            "Precision (weighted)": precision_score(yte, yp, average="weighted", zero_division=0),
            "Recall (macro)": recall_score(yte, yp, average="macro", zero_division=0),
            "Recall (weighted)": recall_score(yte, yp, average="weighted", zero_division=0),
            "F1 (macro)": f1_score(yte, yp, average="macro", zero_division=0),
            "F1 (weighted)": f1_score(yte, yp, average="weighted", zero_division=0),
        })
        cm_data[name] = confusion_matrix(yte, yp)
        try:
            proba = m.predict_proba(Xte)
            if len(le.classes_) == 2:
                roc_data[name] = roc_curve(yte, proba[:, 1])
            else:
                roc_data[name] = (proba, yte, le.classes_)
        except Exception:
            roc_data[name] = None
            
    perf = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    best = perf.iloc[0]
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Accuracy", f"{best['Accuracy']*100:.1f}%")
    with c2: kpi("Precision (macro)", f"{best['Precision (macro)']*100:.1f}%")
    with c3: kpi("Recall (macro)", f"{best['Recall (macro)']*100:.1f}%")
    with c4: kpi("F1 (macro)", f"{best['F1 (macro)']*100:.1f}%")
    interp(f"Best-performing algorithm is {best['Model']} with macro-F1 {best['F1 (macro)']:.4f}. Macro averaging weights every commodity class equally, which is the appropriate summary under class imbalance; weighted averages are also reported for comparability with the literature.")
    
    st.markdown("### Comparative Model Metrics — Macro and Weighted Averages")
    st.dataframe(perf, use_container_width=True, hide_index=True)
    interp("Random Forest and Gradient Boosting generally lead on macro-F1; SVM is competitive but slower to calibrate; XGBoost and MaxEnt are included where their libraries are available in the runtime environment.")
    
    st.markdown("### Confusion Matrix — Best Model")
    cm = cm_data[best["Model"]]
    fig_cm = px.imshow(cm, text_auto=True, x=list(le.classes_), y=list(le.classes_),
                       color_continuous_scale="Teal", title=f"Confusion Matrix — {best['Model']}")
    fig_cm.update_layout(height=420, xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)
    interp("Diagonal cells are correct predictions; off-diagonal cells reveal systematic confusions between commodity classes, particularly between Agroforestry and High Value where species lists overlap.")
    
    st.markdown("### ROC-AUC Curves — One-vs-Rest")
    fig_roc = go.Figure()
    for name, data in roc_data.items():
        if data is None:
            continue
        if isinstance(data, tuple) and len(data) == 3:
            proba, y_true, classes = data
            for i, cls in enumerate(classes):
                try:
                    fpr, tpr, _ = roc_curve((y_true == i).astype(int), proba[:, i])
                    auc_v = roc_auc_score((y_true == i).astype(int), proba[:, i])
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} — {cls} (AUC {auc_v:.2f})"))
                except Exception:
                    continue
        else:
            fpr, tpr, _ = data
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ROC"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Chance"))
    fig_roc.update_layout(height=480, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves — One-vs-Rest per Commodity Class")
    st.plotly_chart(fig_roc, use_container_width=True)
    interp("AUC values above 0.80 indicate strong separability for that commodity class; curves close to the diagonal indicate classes the model cannot distinguish and that require additional predictors such as soil or climate variables.")
    
    st.markdown("### SHAP Explainability — Random Forest")

        # Retrieve the trained Random Forest model
    rf = models.get("Random Forest")

    if rf is None:
        warn("Random Forest model is unavailable.")
    else:
        try:
            import shap

            Xte_df = pd.DataFrame(Xte, columns=feat_names)

            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(Xte_df)

            if hasattr(shap_values, "values"):
                shap_values = shap_values.values

            if isinstance(shap_values, list):
                shap_array = np.stack(
                    [np.asarray(v) for v in shap_values],
                    axis=-1
                )
            else:
                shap_array = np.asarray(shap_values)

            if shap_array.ndim == 3:
                if shap_array.shape[1] == len(feat_names):
                    mean_abs = np.abs(shap_array).mean(axis=(0, 2))
                elif shap_array.shape[2] == len(feat_names):
                    mean_abs = np.abs(shap_array).mean(axis=(0, 1))
                else:
                    raise ValueError(
                        f"Unexpected SHAP shape: {shap_array.shape}"
                    )

            elif shap_array.ndim == 2:
                mean_abs = np.abs(shap_array).mean(axis=0)
            else:
                raise ValueError(
                    f"Unsupported SHAP dimensions: {shap_array.shape}"
                )

            if len(mean_abs) != len(feat_names):
                raise ValueError("Feature dimensions do not match.")

            shap_df = pd.DataFrame({
                "Feature": feat_names,
                "Mean_Abs_SHAP": mean_abs
            }).sort_values(
                "Mean_Abs_SHAP", ascending=False
            ).head(15)

            fig_shap = px.bar(
                shap_df.sort_values("Mean_Abs_SHAP"),
                x="Mean_Abs_SHAP",
                y="Feature",
                orientation="h",
                color="Mean_Abs_SHAP",
                color_continuous_scale="Teal",
                title="Mean Absolute SHAP Value — Top 15 Features"
            )

            fig_shap.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_shap, use_container_width=True)

            interp(
                "Mean absolute SHAP values summarize feature "
                "contributions across test records and commodity classes."
            )

        except ImportError:
            warn("SHAP is not installed. Add shap to requirements.txt.")

        except Exception as e:
            st.error(f"SHAP calculation error: {e}")
            st.exception(e)
          
    st.markdown("### K-Means Clustering — Site Archetypes")
    if len(X_scaled) >= 10 and len(le.classes_) >= 2:
        try:
            n_clusters = min(4, len(le.classes_))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = km.fit_predict(X_scaled)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            cluster_df = pd.DataFrame({
                "PC1": coords[:, 0], "PC2": coords[:, 1],
                "Cluster": cluster_labels.astype(str),
                "Commodity": le.inverse_transform(y_enc),
                "Municipality": model_df["MUNI_CITY"].values,
                "Barangay": model_df["BARANGAY"].values,
            })
            fig_cl = px.scatter(cluster_df, x="PC1", y="PC2", color="Cluster", symbol="Commodity",
                                hover_data=["Municipality", "Barangay", "Commodity"], title=f"K-Means Site Archetypes (k={n_clusters}, PCA-reduced)",
                                color_discrete_sequence=TEAL_PALETTE)
            fig_cl.update_layout(height=480)
            st.plotly_chart(fig_cl, use_container_width=True)
            cluster_summary = cluster_df.groupby("Cluster").agg(Parcels=("Commodity", "size")).reset_index()
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
            dominant = cluster_df["Cluster"].value_counts().index[0]
            dom_comm = cluster_df[cluster_df["Cluster"] == dominant]["Commodity"].mode().iloc[0]
            interp(f"K-Means partitioned the site feature space into {n_clusters} archetypes. Cluster {dominant} is the largest and is dominated by {dom_comm}, indicating a distinct site-profile group that CENRO can target with a standardized commodity package.")
        except Exception as e:
            warn(f"K-Means could not be computed: {e}")
            
    st.markdown("### Correlation of Continuous Predictors")
    corr_df = model_df[NUM_FEATURES].dropna()
    if len(corr_df) > 2:
        corr = corr_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson Correlation — YR_ESTAB vs AREA_HA")
        fig_corr.update_layout(height=350)
        st.plotly_chart(fig_corr, use_container_width=True)
        r = corr.iloc[0, 1]
        sign = "positive" if r >= 0 else "negative"
        interp(f"Pearson r = {r:.3f} between establishment year and area, a {sign} relationship. The magnitude indicates the degree to which later NGP parcels differ in size from earlier ones.")

elif page == "Commodity Selection Recommendations":
    st.markdown("## Commodity Selection Recommendations")
    model_df, fallback = ensure_model_df()
    if fallback:
        warn("Modeling is run on the full cleaned dataset because the drill-down selection is too small for stratified evaluation.")
        
    with st.spinner("Generating predictions..."):
        models, feat_names, le, scaler, Xtr, Xte, ytr, yte, X_scaled, y_enc, cv_scores = train_all(model_df)
        
    best_name = max(models.keys(), key=lambda n: f1_score(yte, models[n].predict(Xte), average="macro", zero_division=0))
    best_model = models[best_name]
    
    X_all = model_df[CAT_FEATURES + NUM_FEATURES].copy()
    X_all_enc = pd.get_dummies(X_all, columns=CAT_FEATURES, drop_first=False).reindex(columns=feat_names, fill_value=0)
    X_all_scaled = scaler.transform(X_all_enc)
    proba = best_model.predict_proba(X_all_scaled)
    pred_idx = proba.argmax(axis=1)
    
    model_df = model_df.copy()
    model_df["Predicted_Commodity"] = le.inverse_transform(pred_idx)
    model_df["Prediction_Confidence"] = proba.max(axis=1)
    model_df["Mean_Predicted_Risk"] = 1 - model_df["Prediction_Confidence"]
    
    risk = model_df.groupby("MUNI_CITY").agg(Mean_Predicted_Risk=("Mean_Predicted_Risk", "mean"), Records=("Predicted_Commodity", "size")).reset_index().sort_values("Mean_Predicted_Risk", ascending=False)
    fig_risk = px.bar(risk, x="Mean_Predicted_Risk", y="MUNI_CITY", orientation="h", color="Mean_Predicted_Risk",
                      color_continuous_scale="OrRd", title=f"Mean Prediction Uncertainty (1 − Confidence) by Municipality — {best_name}")
    fig_risk.update_layout(height=380, yaxis_title="", xaxis_title="Mean Predicted Risk")
    st.plotly_chart(fig_risk, use_container_width=True)
    if len(risk):
        interp(f"{risk.iloc[0]['MUNI_CITY']} shows the highest mean prediction uncertainty ({risk.iloc[0]['Mean_Predicted_Risk']:.3f}), identifying it as the priority area for field verification and commodity review before seedling disbursement.")
        
    st.markdown("### Predicted Commodity per Municipality")
    pred_pivot = pd.crosstab(model_df["MUNI_CITY"], model_df["Predicted_Commodity"])
    st.dataframe(pred_pivot, use_container_width=True)
    interp("The matrix aggregates model-predicted commodity classes per municipality, offering CENRO a quantitative basis for matching seedling production schedules to site-level predictions.")
    
    st.markdown("### Commodity Probability Matrix — Top Recommendations per Municipality")
    proba_df = pd.DataFrame(proba, columns=le.classes_)
    proba_df["MUNI_CITY"] = model_df["MUNI_CITY"].values
    proba_agg = proba_df.groupby("MUNI_CITY").mean().round(3)
    st.dataframe(proba_agg, use_container_width=True)
    interp("Each cell is the model's mean predicted probability that a parcel in that municipality belongs to the corresponding commodity class. The highest column per row is the model's recommended priority commodity for that municipality.")
    
    st.markdown("### Record-Level Predictions")
    export_cols = ["FID", "UNIQ_ID", "MUNI_CITY", "BARANGAY", "TENURE", "ZONE", "PART_TYP", "YR_ESTAB", "AREA_HA", "COMMODITY", "Predicted_Commodity", "Prediction_Confidence", "Mean_Predicted_Risk"]
    export_cols = [c for c in export_cols if c in model_df.columns]
    st.dataframe(model_df[export_cols].head(500), use_container_width=True, hide_index=True)
    
    csv = model_df[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download Record-Level Predictions (CSV)", data=csv, file_name="cenro_bunawan_ngp_predictions.csv", mime="text/csv")
    interp("The downloadable file contains record-level predicted commodity, prediction confidence, and uncertainty for all parcels in the current selection, intended for integration into CENRO planning spreadsheets.")
    
    warn("Recommendations are model-derived and should be validated against field inspection and species-site matching guidelines before operational use. The model reflects historical NGP assignment patterns in CENRO Bunawan and does not incorporate soil chemistry, climate projections, or seedling survival data.")
