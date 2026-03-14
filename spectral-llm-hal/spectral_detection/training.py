"""
spectral_detection/training.py

Classification, ablation, per-domain classification, and SHAP utilities.

Ablation variants tested:
  1. Entropy only            — H_sem
  2. Geometry only           — D_cos, M_bar
  3. Entropy + Geometry      — H_sem, D_cos, M_bar
  4. All 5 geometric         — H_sem, D_cos, M_bar, K, sig2_S

'All 5 + refused' is intentionally excluded: refusals have already been merged
into the hallucination label during feature extraction, so using frac_refused
as a predictor would introduce label leakage.

Methods sourced / adapted from hallucination_utils.py (Debanjan Bhattacharya).
"""

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from spectral_detection.data.cleaning import GEO_FEATURES


# ── Classifier setup ──────────────────────────────────────────────────────────

def build_classifiers(random_seed=42):
    """Return the standard dict of three classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=random_seed),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=random_seed, verbosity=0),
    }


# ── Ablation sets ─────────────────────────────────────────────────────────────

ABLATION_SETS = {
    "Entropy only (H_sem)":         [0],
    "Geometry only (D_cos, M_bar)": [1, 2],
    "Entropy + Geometry":           [0, 1, 2],
    "All 5 geometric":              [0, 1, 2, 3, 4],
}
# Note: 'All 5 + refused' removed — refusals merged into label, would be leakage.


# ── Full ablation run ─────────────────────────────────────────────────────────

def run_ablation(feat_df, geo_features=None, label_col="label",
                 n_splits=5, random_seed=42, verbose=True):
    """
    Feature-subset × classifier ablation via stratified k-fold CV.

    Parameters
    ----------
    feat_df : pd.DataFrame
    geo_features : list, optional
    label_col : str
    n_splits : int
    random_seed : int
    verbose : bool

    Returns
    -------
    df_clf : pd.DataFrame   AUC_mean, AUC_std per (Variant, Classifier).
    X_geo_sc : np.ndarray   StandardScaled feature matrix (all 5 features).
    y_all : np.ndarray      Binary labels.

    Method adapted from hallucination_utils.py.
    """
    geo_features = geo_features or GEO_FEATURES
    X_geo = feat_df[geo_features].values
    y_all = feat_df[label_col].values

    scaler  = StandardScaler()
    X_geo_sc = scaler.fit_transform(X_geo)

    classifiers = build_classifiers(random_seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    clf_rows = []
    for variant, feat_idx in ABLATION_SETS.items():
        X_sub = X_geo_sc[:, feat_idx]
        for clf_name, clf in classifiers.items():
            scores = cross_val_score(clf, X_sub, y_all, cv=cv, scoring="roc_auc")
            clf_rows.append({
                "Variant": variant, "Classifier": clf_name,
                "AUC_mean": scores.mean(), "AUC_std": scores.std(),
            })
            if verbose:
                print(f"  {variant:35s} | {clf_name:22s} | "
                      f"AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    df_clf = pd.DataFrame(clf_rows)
    return df_clf, X_geo_sc, y_all


# ── Per-domain classification ─────────────────────────────────────────────────

def run_per_domain_clf(feat_df, analysis_domains, excluded_domains=None,
                        geo_features=None, label_col="label",
                        domain_col="domain", random_seed=42, verbose=True):
    """
    Run all three classifiers on each domain sub-group.
    Uses min(5, min_class_count) folds to handle small domains.

    Parameters
    ----------
    feat_df : pd.DataFrame
    analysis_domains : list     Domains with ≥ MIN_QUESTIONS threshold.
    excluded_domains : list
    geo_features : list, optional
    label_col, domain_col : str
    random_seed : int
    verbose : bool

    Returns
    -------
    df_dom_clf : pd.DataFrame

    Method from hallucination_utils.py.
    """
    geo_features = geo_features or GEO_FEATURES
    classifiers  = build_classifiers(random_seed)
    domain_rows  = []

    for dom in analysis_domains:
        df_d = feat_df[feat_df[domain_col] == dom]
        X_d  = df_d[geo_features].values
        y_d  = df_d[label_col].values
        if len(np.unique(y_d)) < 2:
            if verbose:
                print(f"  Skipping {dom} (single class)")
            continue
        X_d_sc = StandardScaler().fit_transform(X_d)
        for clf_name, clf in classifiers.items():
            n_folds = min(5, min(Counter(y_d).values()))
            if n_folds < 2:
                continue
            cv_d = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                   random_state=random_seed)
            scores = cross_val_score(clf, X_d_sc, y_d, cv=cv_d,
                                     scoring="roc_auc")
            domain_rows.append({
                "Domain": dom, "Classifier": clf_name,
                "AUC_mean": scores.mean(), "AUC_std": scores.std(),
                "n_questions": len(df_d),
            })

    if verbose and excluded_domains:
        print(f"Excluded from per-domain classification: {excluded_domains}")

    df_dom_clf = pd.DataFrame(domain_rows)
    return df_dom_clf