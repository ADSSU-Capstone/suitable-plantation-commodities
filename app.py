import os
import glob
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    permutation_importance,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


DATA_FILE = "CENRO Bunawan Data 2015-2025 (1).xlsx"

POSSIBLE_TARGETS = ["COMMODITY_", "COMMODITY", "COMMODITY_TYPE"]

HEADER_KEYS = ["FID", "MUNI_CITY", "COMMODITY", "BARANGAY", "YR_ESTAB"]

CFG = {
    "categorical": ["MUNI_CITY", "BARANGAY", "ZONE", "TENURE", "PART_TYP"],
    "numeric": ["YR_ESTAB", "AREA_HA"],
    "id_cols": ["FID", "UNIQ_ID"],
    "rename": {
        "MUNI_CITY": "MUNI_CITY",
        "BARANGAY": "BARANGAY",
        "ZONE": "ZONE",
        "TENURE": "TENURE",
        "PART_TYP": "PART_TYP",
        "YR_ESTAB": "YR_ESTAB",
        "AREA_HA": "AREA_HA",
        "SPECIES": "SPECIES",
        "FID": "FID",
        "UNIQ_ID": "UNIQ_ID",
        "NAME_PART": "NAME_PART",
    },
}


def clean_text(s):
    if pd.isna(s):
        return s
    s = str(s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = re.sub(r"[^\x00-\x7F]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_commodity(v):
    if pd.isna(v):
        return None
    s = clean_text(v).lower()
    if s == "" or s == "nan":
        return None
    if "indigenous" in s and "bamboo" in s:
        return "Indigenous and Bamboo"
    if "bamboo" in s or "patong" in s or "giant bamboo" in s:
        return "Bamboo"
    if "indigenous" in s:
        return "Indigenous"
    if any(k in s for k in ["agroforestry", "rubber", "coffee", "cacao"]):
        return "Agroforestry"
    if any(k in s for k in [
        "high value", "mangosteen", "durian", "rambutan",
        "lanzones", "calamansi",
    ]):
        return "High Value"
    if any(k in s for k in [
        "timber", "falcata", "narra", "lauan",
        "mahogany", "gmelina", "molave",
    ]):
        return "Timber"
    if "fruit" in s or "mango" in s or "lemonsito" in s:
        return "Other Fruit Trees"
    return "Other"


def locate_file():
    roots = [
        ".",
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for r in roots:
        candidate = os.path.join(r, DATA_FILE)
        if os.path.exists(candidate):
            return candidate
        for f in glob.glob(os.path.join(r, "*.xlsx")):
            name = os.path.basename(f).lower()
            if "cenro" in name or "bunawan" in name:
                return f
    return None


def read_source():
    path = locate_file()
    if path is None:
        return None
    xls = pd.ExcelFile(path)
    sheet = next(
        (s for s in xls.sheet_names if "reforestation" in str(s).lower()),
        xls.sheet_names[0],
    )
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)
    raw = raw.fillna("")
    header_row = 0
    for i in range(min(10, len(raw))):
        vals = raw.iloc[i].astype(str).str.upper().str.strip().tolist()
        matches = sum(1 for k in HEADER_KEYS if k in vals)
        if matches >= 2:
            header_row = i
            break
    names = []
    for idx, val in enumerate(raw.iloc[header_row]):
        n = clean_text(val) if val != "" else ""
        if n == "" or n.lower() in ("nan", "none"):
            n = f"COL_{idx}"
        names.append(n.upper())
    df = raw.iloc[header_row + 1:].reset_index(drop=True).copy()
    df.columns = names
    df = df.loc[:, ~df.columns.str.startswith("COL_")]
    df = df.dropna(how="all")
    df = df.replace({"": np.nan})
    return df


def prepare(df):
    df = df.copy()
    df.columns = [clean_text(c).upper() for c in df.columns]

    target = next((c for c in POSSIBLE_TARGETS if c in df.columns), None)
    if target is None:
        return None, "target column not found"

    df = df[df[target].notna()].copy()

    rename_map = {}
    for c in df.columns:
        key = c.upper().strip()
        if key in CFG["rename"]:
            rename_map[c] = CFG["rename"][key]
    df = df.rename(columns=rename_map)
    df["COMMODITY_RAW"] = df[target]

    key_cols = [c for c in ["FID", "UNIQ_ID", "MUNI_CITY",
                             "BARANGAY", "YR_ESTAB", "COMMODITY_RAW"]
                if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")

    df = df.dropna(thresh=int(df.shape[1] * 0.7))

    for c in CFG["categorical"]:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        df[c] = df[c].astype(str).apply(clean_text).str.upper()
        df[c] = df[c].replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        df[c] = df[c].fillna("UNKNOWN")

    for c in CFG["numeric"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(
            df[c].astype(str).str.strip(), errors="coerce"
        )

    if "YR_ESTAB" in df.columns and df["YR_ESTAB"].notna().any():
        df["YR_ESTAB"] = df["YR_ESTAB"].fillna(df["YR_ESTAB"].median())

    if "AREA_HA" in df.columns and df["AREA_HA"].notna().any():
        df["AREA_HA"] = df["AREA_HA"].fillna(df["AREA_HA"].median())
        q1 = df["AREA_HA"].quantile(0.01)
        q99 = df["AREA_HA"].quantile(0.99)
        df["AREA_HA"] = df["AREA_HA"].clip(lower=q1, upper=q99)

    df["COMMODITY"] = df["COMMODITY_RAW"].apply(clean_commodity)
    df = df[df["COMMODITY"].notna()]

    counts = df["COMMODITY"].value_counts()
    valid = counts[counts >= 2].index
    df = df[df["COMMODITY"].isin(valid)].copy()

    for c in CFG["categorical"] + CFG["numeric"]:
        if c not in df.columns:
            df[c] = "UNKNOWN" if c in CFG["categorical"] else np.nan

    return df, None


def make_pipeline(model, numeric_cols, cat_cols):
    num_tf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ])
    cat_tf = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_tf, numeric_cols),
        ("cat", cat_tf, cat_cols),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", model),
    ])


def get_feature_names(pipe, numeric_cols, cat_cols):
    try:
        oh = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
        cat_names = oh.get_feature_names_out(cat_cols).tolist()
    except Exception:
        cat_names = [f"cat_{i}" for i in range(50)]
    return numeric_cols + cat_names


def to_display_labels(arr, le):
    return le.inverse_transform(arr)


def safe_boxplot(ax, data, labels):
    try:
        return ax.boxplot(data, tick_labels=labels, patch_artist=True)
    except TypeError:
        return ax.boxplot(data, labels=labels, patch_artist=True)


st.set_page_config(
    page_title="Commodity Suitability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; }
    h1, h2, h3 { color: #1E6F3C; }
    </style>
    """,
    unsafe_allow_html=True,
)

raw_df = read_source()

if raw_df is None:
    st.error("Data file not found.")
    st.stop()

df_prepared, err = prepare(raw_df)

if err or df_prepared is None or len(df_prepared) < 10:
    st.error("The dataset could not be prepared for analysis.")
    st.stop()

df_all = df_prepared

municipalities = sorted(df_all["MUNI_CITY"].dropna().unique().tolist())
zones = sorted(df_all["ZONE"].dropna().unique().tolist())
tenures = sorted(df_all["TENURE"].dropna().unique().tolist())

st.sidebar.markdown("### Filters")

muni_sel = st.sidebar.multiselect("Municipality", municipalities, default=municipalities)
zone_sel = st.sidebar.multiselect("Zone", zones, default=zones)
tenure_sel = st.sidebar.multiselect("Tenure", tenures, default=tenures)

df = df_all[
    df_all["MUNI_CITY"].isin(muni_sel)
    & df_all["ZONE"].isin(zone_sel)
    & df_all["TENURE"].isin(tenure_sel)
].copy()

if len(df) < 10:
    st.warning("Not enough records under the selected filters.")
    st.stop()

st.sidebar.markdown("### Model Settings")

test_frac = st.sidebar.slider("Test fraction", 0.10, 0.40, 0.20, 0.05)
cv_folds = st.sidebar.slider("CV folds", 3, 10, 10)
n_est = st.sidebar.slider("Random Forest trees", 50, 500, 200, 50)
max_d = st.sidebar.selectbox("Max depth", [None, 10, 20, 30], index=3)
min_split = st.sidebar.slider("Min samples split", 2, 10, 2)
min_leaf = st.sidebar.slider("Min samples leaf", 1, 5, 1)
min_sup = st.sidebar.slider("Association min support", 0.01, 0.30, 0.05, 0.01)
min_conf = st.sidebar.slider("Association min confidence", 0.10, 1.00, 0.40, 0.05)

X = df[CFG["categorical"] + CFG["numeric"]].copy()
y_raw = df["COMMODITY"].copy()

le = LabelEncoder()
y = le.fit_transform(y_raw)

st.markdown(
    f"### Plantation Records: {len(df):,} | Classes: {y_raw.nunique()} "
    f"| Municipalities: {df['MUNI_CITY'].nunique()} | "
    f"Barangays: {df['BARANGAY'].nunique()}"
)

t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
    "Overview",
    "Results by Category",
    "Results Across Time",
    "Variable Relationships",
    "Model Results",
    "Significant Factors",
    "Model Validation",
    "Association Patterns",
])

with t1:
    st.subheader("Commodity Distribution")

    dist = y_raw.value_counts()
    share = (dist / dist.sum() * 100).round(2)

    cA, cB, cC, cD = st.columns(4)
    cA.metric("Total records", f"{len(df):,}")
    cB.metric("Distinct commodity classes", y_raw.nunique())
    cC.metric("Largest class share", f"{share.iloc[0]}%")
    cD.metric(
        "Imbalance ratio (max/min)",
        f"{(dist.iloc[0] / dist.iloc[-1]):.2f}" if dist.iloc[-1] > 0 else "n/a",
    )

    colA, colB = st.columns([1.1, 1])
    with colA:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        dist.plot(kind="bar", color="#1E6F3C", ax=ax, edgecolor="black")
        ax.set_ylabel("Records")
        ax.set_title("Records by commodity class")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with colB:
        tbl = pd.DataFrame({
            "Records": dist,
            "Percent": share,
        })
        st.dataframe(tbl, use_container_width=True)

    st.subheader("Municipality Distribution")
    muni_counts = df["MUNI_CITY"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 3))
    muni_counts.plot(kind="bar", color="#2E8B57", ax=ax, edgecolor="black")
    ax.set_ylabel("Records")
    ax.set_title("Records by municipality")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        f"{dist.index[0]} is the dominant class at {share.iloc[0]}% "
        f"of {dist.sum():,} records."
    )

with t2:
    st.subheader("Commodity Composition by Site Category")

    def stack_chart(col, title):
        ct = pd.crosstab(df[col], df["COMMODITY"])
        if ct.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ct.plot(kind="bar", stacked=True, ax=ax,
                colormap="Greens_r", edgecolor="black", linewidth=0.4)
        ax.set_ylabel("Records")
        ax.set_title(title)
        plt.xticks(rotation=0)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    stack_chart("ZONE", "Commodity by zone")
    stack_chart("TENURE", "Commodity by tenure")
    stack_chart("PART_TYP", "Commodity by partner type")

    st.subheader("Crosstab: Zone vs Commodity")
    st.dataframe(
        pd.crosstab(df["ZONE"], df["COMMODITY"], margins=True, margins_name="Total"),
        use_container_width=True,
    )

    st.subheader("Crosstab: Tenure vs Commodity")
    st.dataframe(
        pd.crosstab(df["TENURE"], df["COMMODITY"], margins=True, margins_name="Total"),
        use_container_width=True,
    )

with t3:
    st.subheader("Records by Establishment Year")

    years = df["YR_ESTAB"].dropna().astype(int)
    if years.empty:
        st.info("No valid establishment year values.")
    else:
        fig, ax = plt.subplots(figsize=(9, 3.4))
        years.value_counts().sort_index().plot(kind="bar",
                                                color="#1E6F3C",
                                                ax=ax, edgecolor="black")
        ax.set_ylabel("Records")
        ax.set_title("Records per establishment year")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        yearly = pd.crosstab(df["YR_ESTAB"].astype(int), df["COMMODITY"])
        st.dataframe(yearly, use_container_width=True)

        recent = int(years.max())
        recent_share = (
            df[df["YR_ESTAB"].astype(int) == recent]["COMMODITY"]
            .value_counts(normalize=True) * 100
        ).round(2)
        if not recent_share.empty:
            st.caption(
                f"Records from {recent} show {recent_share.index[0]} "
                f"at {recent_share.iloc[0]}% of the year's total."
            )

with t4:
    st.subheader("Area Distribution by Commodity")

    commodities = sorted(df["COMMODITY"].unique().tolist())
    boxes = [df[df["COMMODITY"] == c]["AREA_HA"].dropna().values
             for c in commodities]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    bp = safe_boxplot(ax, boxes, commodities)
    palette = ["#1E6F3C", "#2E8B57", "#3CB371", "#66CDAA",
               "#90EE90", "#95A5A6", "#B0C4DE"]
    for patch, color in zip(bp["boxes"], palette[:len(commodities)]):
        patch.set_facecolor(color)
    ax.set_ylabel("Area (hectares)")
    ax.set_title("Area distribution per commodity")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Area Summary per Commodity")
    summary = (
        df.groupby("COMMODITY")["AREA_HA"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )
    st.dataframe(summary, use_container_width=True)

    st.subheader("Association Test: Zone vs Commodity")
    ct = pd.crosstab(df["ZONE"], df["COMMODITY"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        try:
            from scipy.stats import chi2_contingency
            chi2, p, dof, _ = chi2_contingency(ct)
            n = ct.values.sum()
            k = min(ct.shape) - 1
            v = np.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else np.nan
            st.write(
                f"chi-square = {chi2:.3f}, p = {p:.4f}, "
                f"Cramer's V = {v:.3f}"
            )
        except Exception:
            st.write("chi-square test not available.")

with t5:
    st.subheader("Random Forest Classifier Results")

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_frac, stratify=y, random_state=42
        )
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_frac, random_state=42
        )

    rf = make_pipeline(
        RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        CFG["numeric"],
        CFG["categorical"],
    )

    with st.spinner("Training model..."):
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xte)

    present = sorted(set(yte) | set(pred))
    present_names = [le.classes_[i] for i in present]

    acc = accuracy_score(yte, pred)
    p_macro = precision_score(yte, pred, average="macro",
                              labels=present, zero_division=0)
    r_macro = recall_score(yte, pred, average="macro",
                           labels=present, zero_division=0)
    f_macro = f1_score(yte, pred, average="macro",
                       labels=present, zero_division=0)
    f_weighted = f1_score(yte, pred, average="weighted",
                          labels=present, zero_division=0)

    try:
        proba = rf.predict_proba(Xte)
        auc_macro = roc_auc_score(
            yte, proba, multi_class="ovr", average="macro",
            labels=present,
        ) if len(present) > 2 else np.nan
    except Exception:
        auc_macro = np.nan

    cA, cB, cC, cD, cE = st.columns(5)
    cA.metric("Accuracy", f"{acc:.3f}")
    cB.metric("Precision (macro)", f"{p_macro:.3f}")
    cC.metric("Recall (macro)", f"{r_macro:.3f}")
    cD.metric("F1 (macro)", f"{f_macro:.3f}")
    cE.metric("AUC (macro OvR)",
              f"{auc_macro:.3f}" if not np.isnan(auc_macro) else "n/a")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(yte, pred, labels=present)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=present_names,
                yticklabels=present_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Per-Class Report")
    try:
        rep = classification_report(
            yte, pred, labels=present,
            target_names=present_names,
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(rep).transpose().round(3),
                     use_container_width=True)
    except Exception as e:
        st.write(f"report error: {e}")

    st.session_state["model"] = rf
    st.session_state["le"] = le
    st.session_state["y_test"] = yte
    st.session_state["y_pred"] = pred
    st.session_state["present"] = present
    st.session_state["present_names"] = present_names
    st.session_state["X_train"] = Xtr
    st.session_state["X_test"] = Xte
    st.session_state["y_train"] = ytr

with t6:
    st.subheader("Feature Importance")

    if "model" not in st.session_state:
        st.info("Train the model in the Model Results tab first.")
    else:
        rf = st.session_state["model"]
        feature_names = get_feature_names(rf, CFG["numeric"], CFG["categorical"])
        clf = rf.named_steps["clf"]
        gini = clf.feature_importances_
        n = min(len(feature_names), len(gini))
        gini_df = pd.DataFrame({
            "Feature": feature_names[:n],
            "Gini": gini[:n],
        }).sort_values("Gini", ascending=False)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Gini importance (top 15)**")
            top = gini_df.head(15).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6.4, 5))
            ax.barh(top["Feature"], top["Gini"],
                    color="#1E6F3C", edgecolor="black")
            ax.set_xlabel("Gini")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with colB:
            st.markdown("**Aggregated by original variable**")
            def agg(name):
                for c in CFG["categorical"]:
                    if name.startswith(c + "_"):
                        return c
                if name in CFG["numeric"]:
                    return name
                return "Other"
            gini_df["Source"] = gini_df["Feature"].apply(agg)
            aggimp = (
                gini_df.groupby("Source")["Gini"].sum()
                .sort_values(ascending=False)
            )
            fig, ax = plt.subplots(figsize=(6.4, 5))
            aggimp.plot(kind="bar", color="#2E8B57", ax=ax, edgecolor="black")
            ax.set_ylabel("Aggregated Gini")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("Permutation Importance")
        try:
            Xte = st.session_state["X_test"]
            yte = st.session_state["y_test"]
            perm = permutation_importance(
                rf, Xte, yte, n_repeats=5, random_state=42, n_jobs=-1,
            )
            perm_df = pd.DataFrame({
                "Feature": CFG["categorical"] + CFG["numeric"],
                "Mean decrease": perm.importances_mean,
                "Std": perm.importances_std,
            }).sort_values("Mean decrease", ascending=False)
            st.dataframe(perm_df.round(4), use_container_width=True)

            top_name = perm_df.iloc[0]["Feature"]
            st.caption(
                f"{top_name} produces the largest accuracy drop "
                f"({perm_df.iloc[0]['Mean decrease']:.4f}) when permuted."
            )
        except Exception as e:
            st.write(f"permutation importance not available: {e}")

with t7:
    st.subheader("Stratified K-Fold Cross-Validation")

    if "model" not in st.session_state:
        st.info("Train the model in the Model Results tab first.")
    else:
        rf = st.session_state["model"]
        min_class = pd.Series(y).value_counts().min()
        actual = min(cv_folds, int(min_class))
        if actual < 2:
            st.write("Not enough records per class for CV.")
        else:
            with st.spinner(f"Running {actual}-fold CV..."):
                skf = StratifiedKFold(n_splits=actual, shuffle=True,
                                      random_state=42)
                scores = cross_val_score(
                    rf, X, y, cv=skf, scoring="accuracy", n_jobs=-1,
                )
            cA, cB, cC = st.columns(3)
            cA.metric("Mean CV accuracy", f"{scores.mean():.3f}")
            cB.metric("Std", f"{scores.std():.3f}")
            cC.metric("Folds", actual)

            fig, ax = plt.subplots(figsize=(9, 3))
            ax.bar(range(1, len(scores) + 1), scores,
                   color="#1E6F3C", edgecolor="black")
            ax.axhline(scores.mean(), color="red", linestyle="--",
                       label=f"mean = {scores.mean():.3f}")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.dataframe(
                pd.DataFrame({
                    "Fold": range(1, len(scores) + 1),
                    "Accuracy": scores.round(4),
                }),
                use_container_width=True,
            )

        st.subheader("Hyperparameter Search (Grid Search)")

        if st.button("Run grid search"):
            rf_base = make_pipeline(
                RandomForestClassifier(
                    class_weight="balanced", random_state=42, n_jobs=-1,
                ),
                CFG["numeric"], CFG["categorical"],
            )
            grid = {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [10, 20, None],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            }
            with st.spinner("Searching hyperparameters..."):
                search = GridSearchCV(
                    rf_base, grid, cv=min(actual, 5),
                    scoring="accuracy", n_jobs=-1,
                )
                search.fit(X, y)
            st.write("best params:", search.best_params_)
            st.write(f"best CV accuracy: {search.best_score_:.3f}")

            results = pd.DataFrame(search.cv_results_)[
                ["params", "mean_test_score", "std_test_score"]
            ].sort_values("mean_test_score", ascending=False).head(10)
            st.dataframe(results.round(4), use_container_width=True)

with t8:
    st.subheader("Association Patterns in Site Attributes")

    tx_df = pd.DataFrame({
        "MUNI": df["MUNI_CITY"],
        "ZONE": df["ZONE"],
        "TENURE": df["TENURE"],
        "PARTNER": df["PART_TYP"],
        "COMMODITY": df["COMMODITY"],
    })

    tx = [
        [clean_text(x) for x in row if clean_text(x) and clean_text(x) != "UNKNOWN"]
        for row in tx_df.values.tolist()
    ]
    tx = [t for t in tx if len(t) >= 2]

    if len(tx) < 20:
        st.info("Not enough transactions for association analysis.")
    else:
        enc = TransactionEncoder()
        basket = pd.DataFrame(enc.fit(tx).transform(tx), columns=enc.columns_)

        st.markdown("**Apriori — top rules by lift**")
        try:
            frequent = apriori(basket, min_support=min_sup, use_colnames=True)
            if frequent.empty:
                st.write("No frequent itemsets at this support.")
            else:
                rules = association_rules(frequent, metric="confidence",
                                          min_threshold=min_conf)
                rules = rules.sort_values("lift", ascending=False)
                if rules.empty:
                    st.write("No rules at this confidence.")
                else:
                    show = rules.head(10).copy()
                    show["antecedents"] = show["antecedents"].apply(
                        lambda s: ", ".join(sorted(s))
                    )
                    show["consequents"] = show["consequents"].apply(
                        lambda s: ", ".join(sorted(s))
                    )
                    st.dataframe(
                        show[["antecedents", "consequents",
                              "support", "confidence", "lift"]].round(3),
                        use_container_width=True,
                    )
                    top = show.iloc[0]
                    st.caption(
                        f"Strongest pattern: {{{top['antecedents']}}} -> "
                        f"{{{top['consequents']}}} "
                        f"(lift {top['lift']:.2f}, confidence "
                        f"{top['confidence']:.2f})."
                    )
        except Exception as e:
            st.write(f"apriori error: {e}")

        st.markdown("**FP-Growth — top rules by lift**")
        try:
            frequent_fp = fpgrowth(basket, min_support=min_sup,
                                   use_colnames=True)
            if not frequent_fp.empty:
                rules_fp = association_rules(frequent_fp,
                                             metric="confidence",
                                             min_threshold=min_conf)
                rules_fp = rules_fp.sort_values("lift", ascending=False)
                if not rules_fp.empty:
                    show = rules_fp.head(10).copy()
                    show["antecedents"] = show["antecedents"].apply(
                        lambda s: ", ".join(sorted(s))
                    )
                    show["consequents"] = show["consequents"].apply(
                        lambda s: ", ".join(sorted(s))
                    )
                    st.dataframe(
                        show[["antecedents", "consequents",
                              "support", "confidence", "lift"]].round(3),
                        use_container_width=True,
                    )
        except Exception as e:
            st.write(f"fp-growth error: {e}")
