"""
spectral_detection/training.py

Classification, ablation, per-domain classification, and evaluation utilities.

CLASSIFIERS
-----------
Five classifiers are available:
  1. LogisticRegression         — vanilla L2, baseline
  2. ElasticNet Logit           — SAGA solver, l1_ratio=0.5 (AJ's approach);
                                  simultaneous L1+L2 regularisation; implicit
                                  feature selection via sparse coefficients
  3. RandomForestClassifier     — 200 trees, max_depth=6
  4. GradientBoostingClassifier — sklearn GBM (added from AJ)
  5. XGBClassifier              — gradient boosting via xgboost

DEFAULT TRAINING PROTOCOL — CV ON FULL DATASET
-----------------------------------------------
5-fold stratified cross-validation over the entire feature matrix.
This is the primary evaluation path for this project because:
  - All 500 (per benchmark) or 2500 (combined) samples contribute to both
    training and validation, giving more stable AUC estimates.
  - With a fixed, small classifier set and no hyperparameter tuning, the
    selection bias from picking the best CV model is minor.
  - A single 80/20 split would leave only 100–500 test samples, giving
    wide confidence intervals on reported metrics.

OPTIONAL PATH — STRATIFIED HOLD-OUT (use_holdout=True)
-------------------------------------------------------
An 80/20 stratified split can be enabled via use_holdout=True in
run_ablation() and run_cv_model_selection().  This is worth using when:
  - You are tuning hyperparameters inside CV (nested CV / hold-out prevents
    hyperparameter leakage)
  - You want a deployment-realistic performance estimate on unseen data
  - You have enough samples that losing 20% to a held-out set does not
    destabilise the CV estimates on the remaining 80%
The full evaluation suite (PR curve, calibration, threshold diagnostics,
confusion matrices) requires the hold-out path and is exposed via
evaluate_final_model().

ABLATION VARIANTS
-----------------
  1. Entropy only            — H_sem
  2. Geometry only           — D_cos, D_cos_var, M_bar
  3. Entropy + Geometry      — H_sem, D_cos, D_cos_var, M_bar
  4. All 6 geometric         — all features including K and sig2_S

'frac_refused' is excluded — refusals are merged into the label,
so using it as a predictor would introduce label leakage.

Methods sourced / adapted from hallucination_utils.py (Debanjan Bhattacharya)
and AJ's work.
"""

import copy
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import xgboost as xgb

from spectral_detection.data.cleaning import GEO_FEATURES


# ── Classifier registry ────────────────────────────────────────────────────────

def build_classifiers(random_seed=42):
    """
    Return the standard dict of five classifiers used throughout the pipeline.

    Includes:
      - LogisticRegression     : vanilla L2 baseline
      - ElasticNet Logit       : SAGA solver, l1_ratio=0.5 (AJ's approach)
      - Random Forest
      - GradientBoosting       : sklearn GBM (AJ's choice)
      - XGBoost

    All tree models use default hyperparameters; tune if needed.
    """
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


# ── Ablation sets (indices into GEO_FEATURES list) ────────────────────────────
# GEO_FEATURES = ["H_sem", "D_cos", "D_cos_var", "D_pair", "M_bar", "K", "sig2_S"]
#                   0         1         2            3         4      5     6

ABLATION_SETS = {
    "Entropy only (H_sem)":                    [0],
    "Geometry only (D_cos, D_cos_var, D_pair, M_bar)": [1, 2, 3, 4],
    "Entropy + Geometry":                      [0, 1, 2, 3, 4],
    "All 7 geometric":                         [0, 1, 2, 3, 4, 5, 6],
}


# ── Train / test split ─────────────────────────────────────────────────────────

def make_train_test_split(feat_df, geo_features=None, label_col="label",
                           test_size=0.2, random_seed=42):
    """
    Stratified 80/20 train/test split.

    This is AJ's recommended protocol: CV runs only on the training portion;
    the test set is held out and used only for final reported metrics.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays/Series
    X_train_sc, X_test_sc             : StandardScaled versions
    """
    geo_features = geo_features or GEO_FEATURES
    X = feat_df[geo_features].values
    y = feat_df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed)

    scaler    = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"Train: {len(y_train)}  |  Test: {len(y_test)}")
    print(f"Train hallucination rate: {y_train.mean()*100:.1f}%")
    print(f"Test  hallucination rate: {y_test.mean()*100:.1f}%")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ── Model selection via CV ────────────────────────────────────────────────────

def run_cv_model_selection(feat_df_or_X, y=None, geo_features=None,
                            n_splits=5, random_seed=42, verbose=True):
    """
    5-fold stratified CV to select the best classifier.

    DEFAULT: run on the full dataset (feat_df_or_X is a DataFrame).
    HOLDOUT PATH: pass pre-split X_train_sc array + y_train from
                  make_train_test_split() to restrict CV to training data.

    Parameters
    ----------
    feat_df_or_X : pd.DataFrame  (full feature matrix, default path)
                   OR np.ndarray (pre-split X_train_sc, holdout path)
    y            : np.ndarray, optional — required when feat_df_or_X is an array
    geo_features : list, optional
    n_splits     : int
    random_seed  : int
    verbose      : bool

    Returns
    -------
    df_cv     : pd.DataFrame  CV_AUC_mean/std + CV_AP_mean/std per Classifier
    best_name : str           name of best classifier by mean CV AUC
    """
    geo_features = geo_features or GEO_FEATURES

    # resolve input
    if isinstance(feat_df_or_X, np.ndarray):
        X_sc = feat_df_or_X
        if y is None:
            raise ValueError("y must be provided when passing a pre-split array.")
        mode = "holdout-train"
    else:
        X = feat_df_or_X[geo_features].values
        y = feat_df_or_X["label"].values
        X_sc = StandardScaler().fit_transform(X)
        mode = "full-dataset"

    if verbose:
        print(f"CV model selection — mode: {mode}  |  n={len(y)}  |  "
              f"hallu rate={y.mean()*100:.1f}%")

    classifiers = build_classifiers(random_seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    rows = []
    for clf_name, clf in classifiers.items():
        auc_scores = cross_val_score(clf, X_sc, y, cv=cv, scoring="roc_auc")
        ap_scores  = cross_val_score(clf, X_sc, y, cv=cv,
                                     scoring="average_precision")
        rows.append({
            "Classifier":  clf_name,
            "CV_AUC_mean": auc_scores.mean(),
            "CV_AUC_std":  auc_scores.std(),
            "CV_AP_mean":  ap_scores.mean(),
            "CV_AP_std":   ap_scores.std(),
        })
        if verbose:
            print(f"  {clf_name:25s} | AUC={auc_scores.mean():.4f}"
                  f"±{auc_scores.std():.4f}  AP={ap_scores.mean():.4f}")

    df_cv = pd.DataFrame(rows).sort_values("CV_AUC_mean", ascending=False)
    best_name = df_cv.iloc[0]["Classifier"]
    if verbose:
        print(f"\nBest classifier: {best_name}")
    return df_cv, best_name


# ── Ablation ──────────────────────────────────────────────────────────────────

def run_ablation(feat_df, geo_features=None, label_col="label",
                 n_splits=5, random_seed=42, use_holdout=False,
                 verbose=True):
    """
    Feature-subset × classifier ablation study.

    DEFAULT (use_holdout=False)
    ---------------------------
    5-fold stratified CV over the full dataset.
    Primary metric: CV_AUC_mean.  Gives the most stable estimates at our
    sample sizes (~500 per benchmark, 2500 combined).

    Returns: df_clf, X_sc (full, scaled), y_all
             X_test_sc and y_test are None in this mode.

    OPTIONAL HOLD-OUT PATH (use_holdout=True)
    -----------------------------------------
    Stratified 80/20 split first.  CV runs on the training portion only.
    The model is then refitted on full training data and evaluated once on
    the held-out 20%.  Use when you want a deployment-realistic estimate or
    when comparing many configurations (reduces selection bias).

    Returns: df_clf, X_train_sc, X_test_sc, y_train, y_test

    Parameters
    ----------
    feat_df      : pd.DataFrame
    geo_features : list, optional
    label_col    : str
    n_splits     : int
    random_seed  : int
    use_holdout  : bool   False = CV on full dataset (default)
                          True  = 80/20 split + CV on train + test eval
    verbose      : bool

    Returns (use_holdout=False)
    ---------------------------
    df_clf   : pd.DataFrame  CV_AUC_mean/std + Test_AUC (=CV_AUC_mean) per row
    X_sc     : np.ndarray    StandardScaled full feature matrix
    None, y_all, None        placeholders matching (X_tr, X_te, y_tr, y_te)

    Returns (use_holdout=True)
    --------------------------
    df_clf, X_train_sc, X_test_sc, y_train, y_test

    Method adapted from hallucination_utils.py.
    """
    geo_features = geo_features or GEO_FEATURES
    classifiers  = build_classifiers(random_seed)
    cv           = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=random_seed)

    if use_holdout:
        X_train_sc, X_test_sc, y_train, y_test, _ = make_train_test_split(
            feat_df, geo_features=geo_features,
            label_col=label_col, random_seed=random_seed)
        print(f"Ablation — HOLDOUT MODE  (train={len(y_train)}, test={len(y_test)})")
    else:
        X = feat_df[geo_features].values
        y_all = feat_df[label_col].values
        X_sc = StandardScaler().fit_transform(X)
        X_train_sc = X_sc
        y_train    = y_all
        X_test_sc  = None
        y_test     = None
        print(f"Ablation — CV-FULL MODE  (n={len(y_all)}, "
              f"hallu={y_all.mean()*100:.1f}%)")

    rows = []
    for variant, feat_idx in ABLATION_SETS.items():
        Xtr_sub = X_train_sc[:, feat_idx]
        for clf_name, clf in classifiers.items():
            cv_scores = cross_val_score(clf, Xtr_sub, y_train,
                                        cv=cv, scoring="roc_auc")
            row = {
                "Variant":     variant,
                "Classifier":  clf_name,
                "CV_AUC_mean": cv_scores.mean(),
                "CV_AUC_std":  cv_scores.std(),
                "AUC_mean":    cv_scores.mean(),  # backward compat alias
                "AUC_std":     cv_scores.std(),
            }

            if use_holdout:
                Xte_sub  = X_test_sc[:, feat_idx]
                clf_fit  = copy.deepcopy(clf)
                clf_fit.fit(Xtr_sub, y_train)
                prob     = clf_fit.predict_proba(Xte_sub)[:, 1]
                test_auc = roc_auc_score(y_test, prob)
                row["Test_AUC"] = test_auc
                if verbose:
                    print(f"  {variant:42s} | {clf_name:25s} | "
                          f"CV={cv_scores.mean():.4f}  Test={test_auc:.4f}")
            else:
                row["Test_AUC"] = cv_scores.mean()  # same value, for table compat
                if verbose:
                    print(f"  {variant:42s} | {clf_name:25s} | "
                          f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

            rows.append(row)

    df_clf = pd.DataFrame(rows)

    if use_holdout:
        return df_clf, X_train_sc, X_test_sc, y_train, y_test
    else:
        return df_clf, X_train_sc, None, y_train, None


# ── Final model evaluation on held-out test set ───────────────────────────────

def evaluate_final_model(clf_name, X_train_sc, X_test_sc,
                          y_train, y_test, feat_names,
                          random_seed=42):
    """
    Fit the selected model on full training data and evaluate on held-out test.

    Returns a results_bundle dict with everything needed for plotting:
      - test_metrics       : dict of scalar metrics
      - threshold_df       : DataFrame (precision/recall/F1/spec at 5 thresholds)
      - roc_curve_df       : DataFrame (fpr, tpr)
      - pr_curve_df        : DataFrame (precision, recall)
      - test_pred_df       : DataFrame (y_true, y_prob, y_pred_0.5)
      - coef_df            : DataFrame of logistic coefficients (if applicable)
      - best_model_name    : str
      - final_model        : fitted estimator

    Mirrors AJ's results_bundle structure.
    """
    classifiers = build_classifiers(random_seed)
    final_model = classifiers[clf_name]
    final_model.fit(X_train_sc, y_train)

    y_prob     = final_model.predict_proba(X_test_sc)[:, 1]
    y_pred_05  = (y_prob >= 0.5).astype(int)

    test_metrics = {
        "roc_auc":          roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
        "brier_score":      brier_score_loss(y_test, y_prob),
        "accuracy_at_0.5":  accuracy_score(y_test, y_pred_05),
        "precision_at_0.5": precision_score(y_test, y_pred_05, zero_division=0),
        "recall_at_0.5":    recall_score(y_test, y_pred_05, zero_division=0),
        "f1_at_0.5":        f1_score(y_test, y_pred_05, zero_division=0),
    }

    # threshold diagnostics
    thresholds = [0.2, 0.3, 0.5, 0.7, 0.8]
    thr_rows   = []
    for thr in thresholds:
        yp = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        thr_rows.append({
            "threshold":   thr,
            "accuracy":    accuracy_score(y_test, yp),
            "precision":   precision_score(y_test, yp, zero_division=0),
            "recall":      recall_score(y_test, yp, zero_division=0),
            "f1":          f1_score(y_test, yp, zero_division=0),
            "specificity": spec,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    threshold_df = pd.DataFrame(thr_rows)

    # ROC / PR curve data
    fpr, tpr, roc_thr = roc_curve(y_test, y_prob)
    roc_curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})

    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
    pr_curve_df = pd.DataFrame({"precision": pr_prec, "recall": pr_rec})

    test_pred_df = pd.DataFrame({
        "y_true":     np.asarray(y_test),
        "y_prob":     y_prob,
        "y_pred_0.5": y_pred_05,
    })

    # logistic coefficients (if LR variant)
    coef_df = None
    if clf_name in ("Logistic Regression", "ElasticNet Logit"):
        try:
            if hasattr(final_model, "named_steps"):
                lr = final_model.named_steps["model"]
            else:
                lr = final_model
            coef_df = pd.DataFrame({
                "feature":       feat_names,
                "coefficient":   lr.coef_.ravel(),
                "abs_coef":      np.abs(lr.coef_.ravel()),
            }).sort_values("abs_coef", ascending=False)
        except Exception:
            pass

    return {
        "best_model_name": clf_name,
        "final_model":     final_model,
        "test_metrics":    test_metrics,
        "threshold_df":    threshold_df,
        "roc_curve_df":    roc_curve_df,
        "pr_curve_df":     pr_curve_df,
        "test_pred_df":    test_pred_df,
        "coef_df":         coef_df,
    }


# ── CV-path evaluation (always runs — no hold-out needed) ─────────────────────

def evaluate_cv_predictions(feat_df, clf_name, geo_features=None,
                              label_col="label", n_splits=5, random_seed=42):
    """
    Produce a results_bundle using out-of-fold predicted probabilities from
    stratified k-fold cross-validation on the FULL dataset.

    This is the default evaluation path (use_holdout=False).  It uses
    sklearn's cross_val_predict to collect one predicted probability per
    question from the fold in which that question was the validation set.
    The result is a complete set of N predictions — one per sample — that
    can be used for PR curves, calibration curves, and threshold diagnostics
    without requiring a held-out test split.

    Why this is valid
    -----------------
    cross_val_predict returns out-of-fold predictions: each sample's
    probability is predicted by a model that was trained on the other k-1
    folds and never saw that sample.  These predictions are therefore
    unbiased estimates of generalisation performance in the same way that
    standard CV AUC is — and they support the full suite of threshold and
    calibration diagnostics.

    Parameters
    ----------
    feat_df      : pd.DataFrame  full feature matrix
    clf_name     : str           classifier name (from build_classifiers())
    geo_features : list, optional
    label_col    : str
    n_splits     : int
    random_seed  : int

    Returns
    -------
    results_bundle : dict  same structure as evaluate_final_model(), with keys:
        test_metrics, threshold_df, roc_curve_df, pr_curve_df,
        test_pred_df, coef_df (None for non-LR models), best_model_name,
        eval_mode = 'cv_oof'
    """
    from sklearn.model_selection import cross_val_predict

    geo_features = geo_features or GEO_FEATURES

    X = feat_df[geo_features].values
    y = feat_df[label_col].values
    X_sc = StandardScaler().fit_transform(X)

    classifiers = build_classifiers(random_seed)
    clf = classifiers[clf_name]
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    print(f"CV out-of-fold predictions: {clf_name}  (n={len(y)}, {n_splits}-fold)")
    y_prob = cross_val_predict(clf, X_sc, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(int)

    test_metrics = {
        "roc_auc":           roc_auc_score(y, y_prob),
        "average_precision": average_precision_score(y, y_prob),
        "brier_score":       brier_score_loss(y, y_prob),
        "accuracy_at_0.5":   accuracy_score(y, y_pred_05),
        "precision_at_0.5":  precision_score(y, y_pred_05, zero_division=0),
        "recall_at_0.5":     recall_score(y, y_pred_05, zero_division=0),
        "f1_at_0.5":         f1_score(y, y_pred_05, zero_division=0),
    }

    thresholds = [0.2, 0.3, 0.5, 0.7, 0.8]
    thr_rows = []
    for thr in thresholds:
        yp = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        thr_rows.append({
            "threshold":   thr,
            "accuracy":    accuracy_score(y, yp),
            "precision":   precision_score(y, yp, zero_division=0),
            "recall":      recall_score(y, yp, zero_division=0),
            "f1":          f1_score(y, yp, zero_division=0),
            "specificity": spec,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    fpr, tpr, _ = roc_curve(y, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y, y_prob)

    return {
        "best_model_name": clf_name,
        "eval_mode":       "cv_oof",
        "test_metrics":    test_metrics,
        "threshold_df":    pd.DataFrame(thr_rows),
        "roc_curve_df":    pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "pr_curve_df":     pd.DataFrame({"precision": pr_prec, "recall": pr_rec}),
        "test_pred_df":    pd.DataFrame({"y_true": y, "y_prob": y_prob,
                                          "y_pred_0.5": y_pred_05}),
        "coef_df":         None,   # needs a single fitted model; not available in OOF path
    }

def run_per_domain_clf(feat_df, analysis_domains, excluded_domains=None,
                        geo_features=None, label_col="label",
                        domain_col="domain", random_seed=42, verbose=True):
    """
    Run all classifiers on each domain sub-group.
    Uses min(5, min_class_count) folds for small domains.
    CV only (no separate test split within each domain — too few samples).

    Method from hallucination_utils.py.
    """
    geo_features = geo_features or GEO_FEATURES
    classifiers  = build_classifiers(random_seed)
    rows         = []

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
            cv_d   = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=random_seed)
            scores = cross_val_score(clf, X_d_sc, y_d, cv=cv_d, scoring="roc_auc")
            rows.append({
                "Domain": dom, "Classifier": clf_name,
                "AUC_mean": scores.mean(), "AUC_std": scores.std(),
                "n_questions": len(df_d),
            })

    if verbose and excluded_domains:
        print(f"Excluded from per-domain classification: {excluded_domains}")

    return pd.DataFrame(rows)


