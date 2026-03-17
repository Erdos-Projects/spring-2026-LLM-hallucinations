"""
spectral_detection/training.py

Classification, ablation, per-domain evaluation, and final model evaluation.

Classifiers: LogisticRegression, ElasticNet, Random Forest, GradientBoosting, XGBoost.
Evaluation: Stratified 80/20 hold-out (default) or full-dataset 5-fold CV.
"""

import copy
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import xgboost as xgb

from spectral_detection.data.cleaning import FEATURES


# ── Classifier registry ───────────────────────────────────────────────────────

def build_classifiers(random_seed=42):
    """Standard classifier dict used throughout the pipeline."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_seed),
        "ElasticNet Logit": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                solver="saga", penalty="elasticnet", l1_ratio=0.5,
                max_iter=5000, random_state=random_seed)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=random_seed),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_seed),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=random_seed, verbosity=0),
    }


# ── Ablation feature sets (indices into FEATURES) ─────────────────────────────
# FEATURES = ["H_sem", "D_cos", "D_cos_var", "D_pair", "K", "sig2_S"]
#                0        1        2            3        4      5

ABLATION_SETS = {
    "Entropy only (H_sem)":                    [0],
    "Geometry only (D_cos, D_cos_var, D_pair)": [1, 2, 3],
    "Entropy + Geometry":                       [0, 1, 2, 3],
    "All 6 geometric":                          [0, 1, 2, 3, 4, 5],
}


# ── Train/test split ──────────────────────────────────────────────────────────

def make_train_test_split(feat_df, features=None, label_col="label",
                           test_size=0.2, random_seed=42):
    """Stratified 80/20 split with StandardScaler. Returns (X_tr, X_te, y_tr, y_te, scaler)."""
    features = features or FEATURES
    X, y = feat_df[features].values, feat_df[label_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    print(f"Train: {len(y_tr)} | Test: {len(y_te)} | "
          f"Hallu rate: train={y_tr.mean()*100:.1f}%, test={y_te.mean()*100:.1f}%")
    return X_tr, X_te, y_tr, y_te, scaler


# ── Model selection via CV ────────────────────────────────────────────────────

def run_cv_model_selection(feat_or_X, y=None, features=None,
                            n_splits=5, random_seed=42, verbose=True):
    """
    5-fold stratified CV to select the best classifier.
    Accepts either a DataFrame (full dataset) or pre-split X array + y.
    Returns (cv_results_df, best_classifier_name).
    """
    features = features or FEATURES

    if isinstance(feat_or_X, np.ndarray):
        X_sc, mode = feat_or_X, "holdout-train"
        if y is None:
            raise ValueError("y required when passing array")
    else:
        X_sc = StandardScaler().fit_transform(feat_or_X[features].values)
        y = feat_or_X["label"].values
        mode = "full-dataset"

    if verbose:
        print(f"CV model selection: {mode}, n={len(y)}, hallu={y.mean()*100:.1f}%")

    classifiers = build_classifiers(random_seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    rows = []
    for name, clf in classifiers.items():
        auc = cross_val_score(clf, X_sc, y, cv=cv, scoring="roc_auc")
        ap = cross_val_score(clf, X_sc, y, cv=cv, scoring="average_precision")
        rows.append({
            "Classifier": name,
            "CV_AUC_mean": auc.mean(), "CV_AUC_std": auc.std(),
            "CV_AP_mean": ap.mean(), "CV_AP_std": ap.std(),
        })
        if verbose:
            print(f"  {name:25s} | AUC={auc.mean():.4f}+/-{auc.std():.4f}  AP={ap.mean():.4f}")

    df_cv = pd.DataFrame(rows).sort_values("CV_AUC_mean", ascending=False)
    best = df_cv.iloc[0]["Classifier"]
    if verbose:
        print(f"\nBest: {best}")
    return df_cv, best


# ── Ablation ──────────────────────────────────────────────────────────────────

def run_ablation(feat_df, features=None, label_col="label",
                  n_splits=5, random_seed=42, use_holdout=False, verbose=True):
    """
    Feature-subset x classifier ablation.
    Returns (results_df, X_train, X_test, y_train, y_test).
    X_test/y_test are None when use_holdout=False.
    """
    features = features or FEATURES
    classifiers = build_classifiers(random_seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    if use_holdout:
        X_tr, X_te, y_tr, y_te, _ = make_train_test_split(
            feat_df, features=features, label_col=label_col, random_seed=random_seed)
        print(f"Ablation: HOLDOUT (train={len(y_tr)}, test={len(y_te)})")
    else:
        X_tr = StandardScaler().fit_transform(feat_df[features].values)
        y_tr = feat_df[label_col].values
        X_te, y_te = None, None
        print(f"Ablation: CV-FULL (n={len(y_tr)}, hallu={y_tr.mean()*100:.1f}%)")

    rows = []
    for variant, feat_idx in ABLATION_SETS.items():
        Xtr_sub = X_tr[:, feat_idx]
        for clf_name, clf in classifiers.items():
            cv_auc = cross_val_score(clf, Xtr_sub, y_tr, cv=cv, scoring="roc_auc")
            row = {
                "Variant": variant, "Classifier": clf_name,
                "CV_AUC_mean": cv_auc.mean(), "CV_AUC_std": cv_auc.std(),
                "AUC_mean": cv_auc.mean(), "AUC_std": cv_auc.std(),
            }
            if use_holdout:
                fitted = copy.deepcopy(clf)
                fitted.fit(Xtr_sub, y_tr)
                test_auc = roc_auc_score(y_te, fitted.predict_proba(X_te[:, feat_idx])[:, 1])
                row["Test_AUC"] = test_auc
            else:
                row["Test_AUC"] = cv_auc.mean()

            if verbose:
                metric = row.get("Test_AUC", cv_auc.mean())
                print(f"  {variant:42s} | {clf_name:25s} | AUC={metric:.4f}")
            rows.append(row)

    return pd.DataFrame(rows), X_tr, X_te, y_tr, y_te


# ── Final model evaluation ────────────────────────────────────────────────────

def _compute_threshold_table(y_true, y_prob, thresholds=(0.2, 0.3, 0.5, 0.7, 0.8)):
    """Precision/recall/F1/specificity at several thresholds."""
    rows = []
    for thr in thresholds:
        yp = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, yp),
            "precision": precision_score(y_true, yp, zero_division=0),
            "recall": recall_score(y_true, yp, zero_division=0),
            "f1": f1_score(y_true, yp, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return pd.DataFrame(rows)


def _compute_metrics(y_true, y_prob):
    """Core evaluation metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc":          roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "brier_score":      brier_score_loss(y_true, y_prob),
        "accuracy_at_0.5":  accuracy_score(y_true, y_pred),
        "precision_at_0.5": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_0.5":    recall_score(y_true, y_pred, zero_division=0),
        "f1_at_0.5":        f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_final_model(clf_name, X_train, X_test, y_train, y_test,
                          feat_names, random_seed=42):
    """Fit best model on train, evaluate on test. Returns results bundle dict."""
    model = build_classifiers(random_seed)[clf_name]
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)

    # Extract logistic coefficients if applicable
    coef_df = None
    if clf_name in ("Logistic Regression", "ElasticNet Logit"):
        try:
            lr = model.named_steps["model"] if hasattr(model, "named_steps") else model
            coef_df = pd.DataFrame({
                "feature": feat_names,
                "coefficient": lr.coef_.ravel(),
                "abs_coef": np.abs(lr.coef_.ravel()),
            }).sort_values("abs_coef", ascending=False)
        except Exception:
            pass

    return {
        "best_model_name": clf_name,
        "final_model": model,
        "test_metrics": _compute_metrics(y_test, y_prob),
        "threshold_df": _compute_threshold_table(y_test, y_prob),
        "roc_curve_df": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "pr_curve_df": pd.DataFrame({"precision": pr_prec, "recall": pr_rec}),
        "test_pred_df": pd.DataFrame({
            "y_true": np.asarray(y_test), "y_prob": y_prob,
            "y_pred_0.5": (y_prob >= 0.5).astype(int),
        }),
        "coef_df": coef_df,
    }


def evaluate_cv_predictions(feat_df, clf_name, features=None,
                              label_col="label", n_splits=5, random_seed=42):
    """Out-of-fold evaluation on full dataset. Returns same bundle structure."""
    features = features or FEATURES
    X = StandardScaler().fit_transform(feat_df[features].values)
    y = feat_df[label_col].values
    clf = build_classifiers(random_seed)[clf_name]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    print(f"CV out-of-fold: {clf_name} (n={len(y)}, {n_splits}-fold)")
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    fpr, tpr, _ = roc_curve(y, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y, y_prob)

    return {
        "best_model_name": clf_name,
        "eval_mode": "cv_oof",
        "test_metrics": _compute_metrics(y, y_prob),
        "threshold_df": _compute_threshold_table(y, y_prob),
        "roc_curve_df": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "pr_curve_df": pd.DataFrame({"precision": pr_prec, "recall": pr_rec}),
        "test_pred_df": pd.DataFrame({
            "y_true": y, "y_prob": y_prob, "y_pred_0.5": (y_prob >= 0.5).astype(int),
        }),
        "coef_df": None,
    }


def run_per_domain_clf(feat_df, analysis_domains, excluded_domains=None,
                        features=None, label_col="label",
                        domain_col="domain", random_seed=42, verbose=True):
    """CV classification per domain sub-group. Uses adaptive fold count."""
    features = features or FEATURES
    classifiers = build_classifiers(random_seed)
    rows = []

    for dom in analysis_domains:
        df_d = feat_df[feat_df[domain_col] == dom]
        X_d = StandardScaler().fit_transform(df_d[features].values)
        y_d = df_d[label_col].values

        if len(np.unique(y_d)) < 2:
            if verbose:
                print(f"  Skipping {dom} (single class)")
            continue

        for clf_name, clf in classifiers.items():
            n_folds = min(5, min(Counter(y_d).values()))
            if n_folds < 2:
                continue
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            scores = cross_val_score(clf, X_d, y_d, cv=cv, scoring="roc_auc")
            rows.append({
                "Domain": dom, "Classifier": clf_name,
                "AUC_mean": scores.mean(), "AUC_std": scores.std(),
                "n_questions": len(df_d),
            })

    if verbose and excluded_domains:
        print(f"Excluded: {excluded_domains}")
    return pd.DataFrame(rows)


# ── Cross-benchmark generalization ────────────────────────────────────────────

def run_cross_benchmark_test(feat_dfs, test_benchmark, features=None,
                              label_col="label", random_seed=42, verbose=True):
    """
    Train on all benchmarks except one, test on the held-out benchmark.

    This tests whether the geometric features generalize across datasets:
    a classifier that only saw HaluEval, MMLU, TriviaQA, TruthfulQA
    must now score DefAn questions it has never seen.

    Parameters
    ----------
    feat_dfs       : dict of {name: DataFrame} for each benchmark
    test_benchmark : str, key in feat_dfs to hold out for testing
    features       : list of feature columns (or feature index lists for ablation)
    label_col      : str
    random_seed    : int
    verbose        : bool

    Returns
    -------
    results : list of dicts with Classifier, Variant, Train_AUC, Test_AUC, etc.
    """
    features = features or FEATURES

    # Split into train (all others) and test (held-out benchmark)
    train_parts = []
    for name, df in feat_dfs.items():
        if name == test_benchmark:
            continue
        train_parts.append(df)
    df_train = pd.concat(train_parts, ignore_index=True)
    df_test = feat_dfs[test_benchmark]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features].values)
    X_test = scaler.transform(df_test[features].values)
    y_train = df_train[label_col].values
    y_test = df_test[label_col].values

    train_benchmarks = [n for n in feat_dfs if n != test_benchmark]

    if verbose:
        print(f"Cross-benchmark test: train on {train_benchmarks} "
              f"({len(y_train)} q), test on {test_benchmark} ({len(y_test)} q)")
        print(f"  Train hallu rate: {y_train.mean()*100:.1f}%")
        print(f"  Test  hallu rate: {y_test.mean()*100:.1f}%")

    classifiers = build_classifiers(random_seed)
    rows = []
    for clf_name, clf in classifiers.items():
        clf_copy = copy.deepcopy(clf)
        clf_copy.fit(X_train, y_train)

        y_prob_train = clf_copy.predict_proba(X_train)[:, 1]
        y_prob_test = clf_copy.predict_proba(X_test)[:, 1]
        train_auc = roc_auc_score(y_train, y_prob_train)
        test_auc = roc_auc_score(y_test, y_prob_test)

        y_pred = (y_prob_test >= 0.5).astype(int)
        rows.append({
            "Classifier": clf_name,
            "Train_AUC": round(train_auc, 4),
            "Test_AUC": round(test_auc, 4),
            "Test_AP": round(average_precision_score(y_test, y_prob_test), 4),
            "Test_F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Train_n": len(y_train),
            "Test_n": len(y_test),
        })
        if verbose:
            print(f"  {clf_name:25s}  Train AUC={train_auc:.4f}  "
                  f"Test AUC={test_auc:.4f}")

    return pd.DataFrame(rows)


def run_cross_benchmark_ablation(feat_dfs, test_benchmark, ablation_sets=None,
                                  label_col="label", random_seed=42, verbose=True):
    """
    Cross-benchmark ablation: for each feature subset, train on all other
    benchmarks and test on the held-out one.

    Returns DataFrame with Variant, Classifier, Train_AUC, Test_AUC.
    """
    ablation_sets = ablation_sets or ABLATION_SETS
    all_features = FEATURES

    train_parts = [df for name, df in feat_dfs.items() if name != test_benchmark]
    df_train = pd.concat(train_parts, ignore_index=True)
    df_test = feat_dfs[test_benchmark]

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(df_train[all_features].values)
    X_test_full = scaler.transform(df_test[all_features].values)
    y_train = df_train[label_col].values
    y_test = df_test[label_col].values

    if verbose:
        train_names = [n for n in feat_dfs if n != test_benchmark]
        print(f"Cross-benchmark ablation: train={train_names}, test={test_benchmark}")

    classifiers = build_classifiers(random_seed)
    rows = []
    for variant, feat_idx in ablation_sets.items():
        X_tr = X_train_full[:, feat_idx]
        X_te = X_test_full[:, feat_idx]
        for clf_name, clf in classifiers.items():
            clf_copy = copy.deepcopy(clf)
            clf_copy.fit(X_tr, y_train)
            y_prob = clf_copy.predict_proba(X_te)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob)
            rows.append({
                "Variant": variant,
                "Classifier": clf_name,
                "Test_AUC": round(test_auc, 4),
            })
            if verbose:
                print(f"  {variant:40s} | {clf_name:25s} | AUC={test_auc:.4f}")

    return pd.DataFrame(rows)
