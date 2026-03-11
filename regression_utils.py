"""
regression_utils.py
Reusable functions for the regression-based hallucination detection pipeline.

Builds on hallucination_utils.py (data loading, EDA, embeddings, feature
extraction) and adds:
  - Model registries (regressors + classifiers, including AdaBoost)
  - Grid search with cross-validation
  - Train / Validation / Test splitting
  - Regression training + evaluation
  - Binary vs regression comparison
  - Cross-dataset generalization evaluation
  - Regression-specific plots (pred vs actual, residuals, calibration,
    learning curves, ablation, comparison charts)

Design: all pipeline functions accept data + config dicts, so switching
the source/target dataset is a one-line change.

Usage:
    from regression_utils import *
    from hallucination_utils import *
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_val_predict,
    train_test_split, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score,
    roc_auc_score, roc_curve, accuracy_score, f1_score,
)
from scipy.stats import pearsonr, spearmanr

import xgboost as xgb

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

GEO_FEATURES = ["H_sem", "D_cos", "M_bar", "K", "sig2_S"]


# ==================================================================
#  0. COMPATIBILITY HELPERS
# ==================================================================

def ensure_p_halluc(feat_df):
    """Ensure feat_df has a 'p_halluc' column.

    If 'p_halluc' is missing but 'hall_rate_strict' exists, creates it
    as an alias. If neither exists, raises KeyError. Modifies in place
    and returns the dataframe.
    """
    if "p_halluc" not in feat_df.columns:
        if "hall_rate_strict" in feat_df.columns:
            feat_df["p_halluc"] = feat_df["hall_rate_strict"]
        elif "incorrect_rate_def" in feat_df.columns:
            feat_df["p_halluc"] = feat_df["incorrect_rate_def"]
        else:
            raise KeyError(
                "feat_df has no 'p_halluc', 'hall_rate_strict', or "
                "'incorrect_rate_def' column. Cannot derive regression target."
            )
    return feat_df


# ==================================================================
#  1. MODEL REGISTRIES
# ==================================================================

def get_regressors(seed=42):
    """Return an OrderedDict of named regression models."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=seed),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, verbosity=0),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.05),
        "AdaBoost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=200, learning_rate=0.1, random_state=seed),
    }


def get_classifiers(seed=42):
    """Return an OrderedDict of named classification models."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=seed),
        "Random Forest (cls)": RandomForestClassifier(
            n_estimators=300, max_depth=7, random_state=seed),
        "XGBoost (cls)": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=seed, verbosity=0),
        "AdaBoost (cls)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=4),
            n_estimators=200, learning_rate=0.1, random_state=seed),
    }


def get_grid_params():
    """Return grid search parameter grids for models that support it."""
    return {
        "Ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "Lasso": {"alpha": [0.001, 0.005, 0.01, 0.05, 0.1]},
        "ElasticNet": {
            "alpha": [0.005, 0.01, 0.05, 0.1],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
        "Random Forest": {
            "n_estimators": [100, 300],
            "max_depth": [4, 6, 8],
        },
        "XGBoost": {
            "n_estimators": [100, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.08, 0.1],
        },
        "AdaBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.3],
        },
    }


# ==================================================================
#  2. GRID SEARCH
# ==================================================================

def run_grid_search(regressors, X_train, y_train, seed=42, cv_folds=5,
                    scoring="r2", verbose=1):
    """Run grid search for each regressor that has a param grid.

    Returns (best_regressors, grid_results_df) where best_regressors is
    the input dict with grid-searched models replaced by their best
    estimators.
    """
    grids = get_grid_params()
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    results = []
    best = dict(regressors)  # shallow copy

    for name, reg in regressors.items():
        if name not in grids:
            if verbose:
                print(f"  {name:20s}: no grid defined, skipping")
            continue
        if verbose:
            print(f"  {name:20s}: searching {grids[name]} ...")

        gs = GridSearchCV(
            reg, grids[name], cv=cv, scoring=scoring,
            n_jobs=-1, refit=True,
        )
        gs.fit(X_train, y_train)
        best[name] = gs.best_estimator_

        results.append({
            "Model": name,
            "Best_Params": str(gs.best_params_),
            "Best_CV_Score": gs.best_score_,
        })
        if verbose:
            print(f"    -> best: {gs.best_params_}  CV {scoring}={gs.best_score_:.4f}")

    return best, pd.DataFrame(results)


# ==================================================================
#  3. TRAIN / VAL / TEST SPLITTING
# ==================================================================

def split_train_val_test(feat_df, features, target_reg="p_halluc",
                         target_cls="label", test_frac=0.15,
                         val_frac=0.15, seed=42):
    """Three-way stratified split.

    Returns a dict with keys:
        X_train, X_val, X_test (raw, unscaled),
        y_train_reg, y_val_reg, y_test_reg,
        y_train_cls, y_val_cls, y_test_cls,
        idx_train, idx_val, idx_test,
        feat_train, feat_val, feat_test,
        scaler (fitted on train).
    """
    ensure_p_halluc(feat_df)
    X = feat_df[features].values
    y_reg = feat_df[target_reg].values
    y_cls = feat_df[target_cls].values

    X_tv, X_test, yr_tv, yr_te, yc_tv, yc_te, idx_tv, idx_te = \
        train_test_split(X, y_reg, y_cls, np.arange(len(feat_df)),
                         test_size=test_frac, random_state=seed,
                         stratify=y_cls)

    val_rel = val_frac / (1 - test_frac)
    X_tr, X_va, yr_tr, yr_va, yc_tr, yc_va, idx_tr, idx_va = \
        train_test_split(X_tv, yr_tv, yc_tv, idx_tv,
                         test_size=val_rel, random_state=seed,
                         stratify=yc_tv)

    scaler = StandardScaler().fit(X_tr)

    return {
        "X_train": X_tr, "X_val": X_va, "X_test": X_test,
        "X_train_sc": scaler.transform(X_tr),
        "X_val_sc": scaler.transform(X_va),
        "X_test_sc": scaler.transform(X_test),
        "y_train_reg": yr_tr, "y_val_reg": yr_va, "y_test_reg": yr_te,
        "y_train_cls": yc_tr, "y_val_cls": yc_va, "y_test_cls": yc_te,
        "idx_train": idx_tr, "idx_val": idx_va, "idx_test": idx_te,
        "feat_train": feat_df.iloc[idx_tr].reset_index(drop=True),
        "feat_val": feat_df.iloc[idx_va].reset_index(drop=True),
        "feat_test": feat_df.iloc[idx_te].reset_index(drop=True),
        "scaler": scaler,
    }


def print_split_summary(sp):
    """Print train/val/test sizes and target statistics."""
    for name, X, yr, yc in [
        ("Train", sp["X_train"], sp["y_train_reg"], sp["y_train_cls"]),
        ("Val",   sp["X_val"],   sp["y_val_reg"],   sp["y_val_cls"]),
        ("Test",  sp["X_test"],  sp["y_test_reg"],  sp["y_test_cls"]),
    ]:
        print(f"  {name:5s}: n={len(X):4d}  "
              f"p_halluc mean={yr.mean():.4f} std={yr.std():.4f}  "
              f"binary rate={yc.mean():.3f}")


# ==================================================================
#  4. REGRESSION TRAINING + EVALUATION
# ==================================================================

def train_evaluate_regressors(regressors, sp, cv_folds=5, seed=42):
    """Train regressors with CV on train, evaluate on train/val/test.

    Returns (results_df, fitted_regressors_dict).
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    Xtr, Xva, Xte = sp["X_train_sc"], sp["X_val_sc"], sp["X_test_sc"]
    ytr, yva, yte = sp["y_train_reg"], sp["y_val_reg"], sp["y_test_reg"]
    rows = []
    fitted = {}

    for name, reg in regressors.items():
        cv_preds = cross_val_predict(reg, Xtr, ytr, cv=cv)
        cv_r2 = r2_score(ytr, cv_preds)
        cv_mse = mean_squared_error(ytr, cv_preds)
        cv_mae = mean_absolute_error(ytr, cv_preds)

        reg.fit(Xtr, ytr)
        fitted[name] = reg

        tr_p = reg.predict(Xtr)
        tr_r2 = r2_score(ytr, tr_p)

        va_p = reg.predict(Xva)
        va_r2 = r2_score(yva, va_p)
        va_mse = mean_squared_error(yva, va_p)
        va_mae = mean_absolute_error(yva, va_p)

        te_p = reg.predict(Xte)
        te_r2 = r2_score(yte, te_p)
        te_mse = mean_squared_error(yte, te_p)
        te_mae = mean_absolute_error(yte, te_p)
        r_te, p_te = pearsonr(yte, te_p)

        rows.append({
            "Model": name,
            "CV_R2": cv_r2, "CV_MSE": cv_mse, "CV_MAE": cv_mae,
            "Train_R2": tr_r2,
            "Val_R2": va_r2, "Val_MSE": va_mse, "Val_MAE": va_mae,
            "Test_R2": te_r2, "Test_MSE": te_mse, "Test_MAE": te_mae,
            "Test_Pearson_r": r_te,
            "Overfit_Gap_Val": tr_r2 - va_r2,
            "Overfit_Gap_Test": tr_r2 - te_r2,
        })
        print(f"  {name:20s} | CV R²={cv_r2:.4f}  Val R²={va_r2:.4f}  "
              f"Test R²={te_r2:.4f}  Gap={tr_r2 - va_r2:+.4f}")

    return pd.DataFrame(rows), fitted


# ==================================================================
#  5. BINARY VS REGRESSION COMPARISON
# ==================================================================

def compare_regression_vs_classification(regressors, classifiers, sp):
    """Threshold regression at 0.5 for binary metrics; compare with classifiers.

    Returns a DataFrame with AUC, Accuracy, F1 for each model on Val and Test.
    """
    Xtr, Xva, Xte = sp["X_train_sc"], sp["X_val_sc"], sp["X_test_sc"]
    ytr_r, ytr_c = sp["y_train_reg"], sp["y_train_cls"]
    rows = []

    for name, clf in classifiers.items():
        clf.fit(Xtr, ytr_c)
        for sn, Xs, ys in [("Val", Xva, sp["y_val_cls"]),
                            ("Test", Xte, sp["y_test_cls"])]:
            proba = clf.predict_proba(Xs)[:, 1]
            pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(ys, proba) if len(np.unique(ys)) > 1 else np.nan
            rows.append({"Model": name, "Approach": "Classification",
                         "Split": sn, "AUC": auc,
                         "Accuracy": accuracy_score(ys, pred),
                         "F1": f1_score(ys, pred, zero_division=0)})

    for name, reg in regressors.items():
        reg.fit(Xtr, ytr_r)
        for sn, Xs, ys in [("Val", Xva, sp["y_val_cls"]),
                            ("Test", Xte, sp["y_test_cls"])]:
            preds = np.clip(reg.predict(Xs), 0, 1)
            pred = (preds >= 0.5).astype(int)
            auc = roc_auc_score(ys, preds) if len(np.unique(ys)) > 1 else np.nan
            rows.append({"Model": f"{name} (reg->cls)",
                         "Approach": "Regression->Binary", "Split": sn,
                         "AUC": auc,
                         "Accuracy": accuracy_score(ys, pred),
                         "F1": f1_score(ys, pred, zero_division=0)})

    return pd.DataFrame(rows)


# ==================================================================
#  6. CROSS-DATASET EVALUATION
# ==================================================================

def evaluate_cross_dataset(regressors, classifiers, sp_source,
                           target_feat_df, target_name,
                           features=None):
    """Train on source, evaluate on target dataset.

    Returns a DataFrame with AUC, Accuracy, F1 (and R² if target has p_halluc).
    """
    features = features or GEO_FEATURES
    # Try to create p_halluc if missing (safe — cross-dataset target may not have it)
    try:
        ensure_p_halluc(target_feat_df)
    except KeyError:
        pass
    scaler = StandardScaler().fit(sp_source["X_train"])
    Xtr = scaler.transform(sp_source["X_train"])
    ytr_r = sp_source["y_train_reg"]
    ytr_c = sp_source["y_train_cls"]

    X_tgt = scaler.transform(target_feat_df[features].values)
    y_tgt_cls = target_feat_df["label"].values
    has_cont = "p_halluc" in target_feat_df.columns
    y_tgt_cont = target_feat_df["p_halluc"].values if has_cont else None

    rows = []
    for name, reg in regressors.items():
        reg.fit(Xtr, ytr_r)
        preds = np.clip(reg.predict(X_tgt), 0, 1)
        pred_bin = (preds >= 0.5).astype(int)
        auc = roc_auc_score(y_tgt_cls, preds) if len(np.unique(y_tgt_cls)) > 1 else np.nan
        r2 = r2_score(y_tgt_cont, preds) if has_cont else np.nan
        rows.append({
            "Model": name, "Approach": "Regression",
            f"{target_name}_AUC": auc,
            f"{target_name}_Acc": accuracy_score(y_tgt_cls, pred_bin),
            f"{target_name}_F1": f1_score(y_tgt_cls, pred_bin, zero_division=0),
            f"{target_name}_R2": r2,
        })

    for name, clf in classifiers.items():
        clf.fit(Xtr, ytr_c)
        proba = clf.predict_proba(X_tgt)[:, 1]
        pred_bin = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_tgt_cls, proba) if len(np.unique(y_tgt_cls)) > 1 else np.nan
        rows.append({
            "Model": name, "Approach": "Classification",
            f"{target_name}_AUC": auc,
            f"{target_name}_Acc": accuracy_score(y_tgt_cls, pred_bin),
            f"{target_name}_F1": f1_score(y_tgt_cls, pred_bin, zero_division=0),
            f"{target_name}_R2": np.nan,
        })

    return pd.DataFrame(rows)


# ==================================================================
#  7. ABLATION STUDY
# ==================================================================

def run_ablation(regressors_subset, sp, feat_train_df,
                 feat_val_df, feat_test_df,
                 geo_features=None, extended_features=None):
    """Run ablation over feature subsets x models.

    Returns a DataFrame with Train/Val/Test R² per variant x model.
    """
    geo_features = geo_features or GEO_FEATURES
    extended_features = extended_features or geo_features + ["frac_refused", "score_mean", "len_mean"]

    sc_geo = StandardScaler().fit(feat_train_df[geo_features].values)
    Xtr_g = sc_geo.transform(feat_train_df[geo_features].values)
    Xva_g = sc_geo.transform(feat_val_df[geo_features].values)
    Xte_g = sc_geo.transform(feat_test_df[geo_features].values)

    sc_ext = StandardScaler().fit(feat_train_df[extended_features].values)
    Xtr_e = sc_ext.transform(feat_train_df[extended_features].values)
    Xva_e = sc_ext.transform(feat_val_df[extended_features].values)
    Xte_e = sc_ext.transform(feat_test_df[extended_features].values)

    ytr = sp["y_train_reg"]
    yva = sp["y_val_reg"]
    yte = sp["y_test_reg"]

    ablation_sets = {
        "Entropy only (H_sem)":      ("geo", [0]),
        "Geometry (D_cos, M_bar)":   ("geo", [1, 2]),
        "Entropy + Geometry":        ("geo", [0, 1, 2]),
        "All 5 geometric":           ("geo", list(range(len(geo_features)))),
        "All 5 + extended":          ("ext", list(range(len(extended_features)))),
    }

    rows = []
    for variant, (fset, idxs) in ablation_sets.items():
        if fset == "geo":
            Xtr, Xva, Xte = Xtr_g[:, idxs], Xva_g[:, idxs], Xte_g[:, idxs]
        else:
            Xtr, Xva, Xte = Xtr_e[:, idxs], Xva_e[:, idxs], Xte_e[:, idxs]
        for name, reg in regressors_subset.items():
            reg.fit(Xtr, ytr)
            rows.append({
                "Variant": variant, "Model": name,
                "Train_R2": r2_score(ytr, reg.predict(Xtr)),
                "Val_R2": r2_score(yva, reg.predict(Xva)),
                "Test_R2": r2_score(yte, reg.predict(Xte)),
                "Test_MSE": mean_squared_error(yte, reg.predict(Xte)),
            })
    return pd.DataFrame(rows)


# ==================================================================
#  8. BOOTSTRAP CONFIDENCE INTERVALS
# ==================================================================

def bootstrap_regression_ci(y_true, y_pred, n_boot=2000, seed=42):
    """Bootstrap 95% CI for R², MSE, MAE on a held-out set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = {"R2": [], "MSE": [], "MAE": []}
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if np.std(yt) < 1e-10:
            continue
        boot["R2"].append(r2_score(yt, yp))
        boot["MSE"].append(mean_squared_error(yt, yp))
        boot["MAE"].append(mean_absolute_error(yt, yp))
    result = {}
    for metric, vals in boot.items():
        arr = np.array(vals)
        result[metric] = {
            "mean": arr.mean(),
            "ci_lo": np.percentile(arr, 2.5),
            "ci_hi": np.percentile(arr, 97.5),
            "values": arr,
        }
    return result


# ==================================================================
#  9. REGRESSION-SPECIFIC PLOTS
# ==================================================================

def plot_regression_results_bar(df_reg, title_prefix=""):
    """Bar chart: CV R² vs Val R² vs Test R², MSE, overfitting gaps."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    x = np.arange(len(df_reg)); w = 0.25
    axes[0].bar(x - w, df_reg["CV_R2"], w, label="CV R²", color="steelblue", alpha=0.8)
    axes[0].bar(x,     df_reg["Val_R2"], w, label="Val R²", color="orange", alpha=0.8)
    axes[0].bar(x + w, df_reg["Test_R2"], w, label="Test R²", color="tomato", alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(df_reg["Model"], rotation=30, ha="right")
    axes[0].set_ylabel("R²"); axes[0].set_title(f"{title_prefix}CV vs Val vs Test R²", fontweight="bold")
    axes[0].legend(); axes[0].axhline(0, color="gray", linestyle="--", alpha=0.3)

    w2 = 0.35
    axes[1].bar(x - w2/2, df_reg["Val_MSE"], w2, label="Val MSE", color="orange", alpha=0.8)
    axes[1].bar(x + w2/2, df_reg["Test_MSE"], w2, label="Test MSE", color="tomato", alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(df_reg["Model"], rotation=30, ha="right")
    axes[1].set_ylabel("MSE"); axes[1].set_title(f"{title_prefix}Val vs Test MSE", fontweight="bold")
    axes[1].legend()

    axes[2].bar(x - w2/2, df_reg["Overfit_Gap_Val"], w2, label="Train-Val", color="orange", alpha=0.8)
    axes[2].bar(x + w2/2, df_reg["Overfit_Gap_Test"], w2, label="Train-Test", color="tomato", alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(df_reg["Model"], rotation=30, ha="right")
    axes[2].set_ylabel("R² gap"); axes[2].set_title(f"{title_prefix}Overfitting Gaps", fontweight="bold")
    axes[2].axhline(0, color="gray", linestyle="--"); axes[2].legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_pred_vs_actual(regressors, sp, title_prefix=""):
    """Scatter: predicted vs actual p_halluc for val + test."""
    n = len(regressors)
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for i, (name, reg) in enumerate(regressors.items()):
        ax = axes[i]
        reg.fit(sp["X_train_sc"], sp["y_train_reg"])
        for Xs, ys, c, lab in [
            (sp["X_val_sc"], sp["y_val_reg"], "orange", "Val"),
            (sp["X_test_sc"], sp["y_test_reg"], "tomato", "Test"),
        ]:
            p = np.clip(reg.predict(Xs), 0, 1)
            ax.scatter(ys, p, alpha=0.35, s=18, color=c, edgecolors="none", label=lab)
        ax.plot([0, 1], [0, 1], "k--", lw=1.5)
        va_r2 = r2_score(sp["y_val_reg"], np.clip(reg.predict(sp["X_val_sc"]), 0, 1))
        te_r2 = r2_score(sp["y_test_reg"], np.clip(reg.predict(sp["X_test_sc"]), 0, 1))
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nVal R²={va_r2:.3f}, Test R²={te_r2:.3f}", fontweight="bold", fontsize=10)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{title_prefix}Predicted vs Actual", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_residuals(regressors, sp, title_prefix=""):
    """Residual plots for val + test."""
    n = len(regressors)
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for i, (name, reg) in enumerate(regressors.items()):
        ax = axes[i]
        reg.fit(sp["X_train_sc"], sp["y_train_reg"])
        for Xs, ys, c, lab in [
            (sp["X_val_sc"], sp["y_val_reg"], "orange", "Val"),
            (sp["X_test_sc"], sp["y_test_reg"], "tomato", "Test"),
        ]:
            p = np.clip(reg.predict(Xs), 0, 1)
            ax.scatter(p, ys - p, alpha=0.35, s=18, color=c, edgecolors="none", label=lab)
        ax.axhline(0, color="black", linestyle="--", lw=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
        ax.set_title(name, fontweight="bold", fontsize=10); ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{title_prefix}Residual Plots", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_comparison_bar(df_compare, title_prefix=""):
    """Side-by-side AUC bars for classification vs regression->binary."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, split, ttl in [(axes[0], "Val", "Validation AUC"),
                            (axes[1], "Test", "Test AUC")]:
        sub = df_compare[df_compare["Split"] == split]
        colors = ["steelblue" if a == "Classification" else "coral"
                  for a in sub["Approach"]]
        ax.barh(sub["Model"], sub["AUC"], color=colors)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("AUC-ROC"); ax.set_title(f"{title_prefix}{ttl}", fontweight="bold")
    axes[0].legend(handles=[
        Patch(facecolor="steelblue", label="Classification"),
        Patch(facecolor="coral", label="Regression->Binary"),
    ], fontsize=9)
    plt.tight_layout()
    return fig


def plot_comparison_roc(regressors, classifiers, sp, title_prefix=""):
    """ROC curves on test set for both approaches."""
    fig, ax = plt.subplots(figsize=(9, 7))
    Xtr, Xte = sp["X_train_sc"], sp["X_test_sc"]
    yte = sp["y_test_cls"]

    for name, clf in classifiers.items():
        clf.fit(Xtr, sp["y_train_cls"])
        proba = clf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, proba):.3f})", lw=2)

    for name, reg in regressors.items():
        reg.fit(Xtr, sp["y_train_reg"])
        preds = np.clip(reg.predict(Xte), 0, 1)
        fpr, tpr, _ = roc_curve(yte, preds)
        ax.plot(fpr, tpr, "--", label=f"{name} reg (AUC={roc_auc_score(yte, preds):.3f})", lw=1.5)

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{title_prefix}ROC: Classification vs Regression (Test)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


def plot_calibration(y_true, y_pred, model_name="", n_bins=10, title_prefix=""):
    """Calibration plot: binned predicted vs actual."""
    edges = np.linspace(0, 1, n_bins + 1)
    pred_m, act_m, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_pred >= lo) & (y_pred < hi + 1e-9)
        if m.sum() > 0:
            pred_m.append(y_pred[m].mean())
            act_m.append(y_true[m].mean())
            counts.append(m.sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
    axes[0].scatter(pred_m, act_m, s=80, zorder=5, color="steelblue")
    for pm, am, c in zip(pred_m, act_m, counts):
        axes[0].annotate(f"n={c}", (pm, am), fontsize=7, ha="center", va="bottom")
    axes[0].set_xlabel("Mean predicted (binned)"); axes[0].set_ylabel("Mean actual")
    axes[0].set_title(f"{title_prefix}Calibration ({model_name})", fontweight="bold")
    axes[0].legend(); axes[0].set_xlim(-0.05, 1.05); axes[0].set_ylim(-0.05, 1.05)

    axes[1].bar(range(len(counts)), counts, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Bin"); axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction distribution", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_learning_curves(regressors_subset, sp, title_prefix=""):
    """Learning curves: train on increasing fractions, plot R² on all splits."""
    fracs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n = len(regressors_subset)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, reg) in zip(axes, regressors_subset.items()):
        tr_s, va_s, te_s = [], [], []
        for frac in fracs:
            k = max(5, int(frac * len(sp["X_train_sc"])))
            Xsub = sp["X_train_sc"][:k]
            ysub = sp["y_train_reg"][:k]
            reg.fit(Xsub, ysub)
            tr_s.append(r2_score(ysub, reg.predict(Xsub)))
            va_s.append(r2_score(sp["y_val_reg"], reg.predict(sp["X_val_sc"])))
            te_s.append(r2_score(sp["y_test_reg"], reg.predict(sp["X_test_sc"])))
        ax.plot(fracs * 100, tr_s, "o-", label="Train", color="steelblue")
        ax.plot(fracs * 100, va_s, "d:", label="Val", color="orange")
        ax.plot(fracs * 100, te_s, "s--", label="Test", color="tomato")
        ax.set_xlabel("Training data (%)"); ax.set_ylabel("R²")
        ax.set_title(f"{title_prefix}{name}", fontweight="bold"); ax.legend()

    plt.tight_layout()
    return fig


def plot_cross_dataset_summary(df_xd, target_name, title_prefix=""):
    """Bar chart + ROC placeholder for cross-dataset results."""
    auc_col = f"{target_name}_AUC"
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(df_xd) + 1)))
    colors = ["steelblue" if a == "Regression" else "coral" for a in df_xd["Approach"]]
    ax.barh(df_xd["Model"], df_xd[auc_col], color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("AUC-ROC")
    ax.set_title(f"{title_prefix}Cross-Dataset AUC on {target_name}", fontweight="bold")
    ax.legend(handles=[
        Patch(facecolor="steelblue", label="Regression"),
        Patch(facecolor="coral", label="Classification"),
    ], fontsize=9)
    plt.tight_layout()
    return fig


def plot_target_eda(feat_df, domain_col="domain", title_prefix=""):
    """EDA plots for the regression target p_halluc."""
    ensure_p_halluc(feat_df)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.hist(feat_df["p_halluc"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", lw=1.5, label="Binary threshold")
    ax.set_xlabel("p_halluc"); ax.set_ylabel("Count")
    ax.set_title("Distribution of p_halluc", fontweight="bold"); ax.legend()

    ax = axes[0, 1]
    if domain_col and domain_col in feat_df.columns:
        order = feat_df.groupby(domain_col)["p_halluc"].median().sort_values().index
        sns.boxplot(data=feat_df, y=domain_col, x="p_halluc", order=order, ax=ax,
                    fliersize=2, color="lightblue")
        ax.axvline(0.5, color="red", linestyle="--", lw=1, alpha=0.5)
        ax.set_title("p_halluc by domain", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No domain column", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("p_halluc by domain (N/A)", fontweight="bold")

    ax = axes[0, 2]
    near = feat_df[(feat_df["p_halluc"] > 0.3) & (feat_df["p_halluc"] < 0.7)]
    ax.hist(feat_df["p_halluc"], bins=40, color="steelblue", edgecolor="white", alpha=0.4, label="All")
    ax.hist(near["p_halluc"], bins=40, color="orange", edgecolor="white", alpha=0.7,
            label=f"Near boundary\nn={len(near)} ({100*len(near)/len(feat_df):.1f}%)")
    ax.axvline(0.5, color="red", linestyle="--", lw=1.5)
    ax.set_title("Info lost by binarization", fontweight="bold"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    if "score_mean" in feat_df.columns:
        ax.scatter(feat_df["score_mean"], feat_df["p_halluc"], alpha=0.3, s=15, color="steelblue")
        r_v, p_v = pearsonr(feat_df["score_mean"], feat_df["p_halluc"])
        ax.set_xlabel("Judge score_mean"); ax.set_ylabel("p_halluc")
        ax.set_title(f"p_halluc vs score (r={r_v:.3f})", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No score_mean column", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title("p_halluc vs score (N/A)", fontweight="bold")

    ax = axes[1, 1]
    sp = np.sort(feat_df["p_halluc"].values)
    ax.plot(sp, np.arange(1, len(sp) + 1) / len(sp), color="steelblue", lw=2)
    ax.axvline(0.5, color="red", linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("p_halluc"); ax.set_ylabel("CDF")
    ax.set_title("CDF of p_halluc", fontweight="bold")

    axes[1, 2].axis("off")
    plt.suptitle(f"{title_prefix}Regression Target Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_features_vs_target(feat_df, geo_features=None, title_prefix=""):
    """Scatter: each geometric feature vs p_halluc with correlation."""
    ensure_p_halluc(feat_df)
    geo_features = geo_features or GEO_FEATURES
    nice = {"H_sem": "Semantic Entropy", "D_cos": "Cosine Dispersion",
            "M_bar": "Mahalanobis Distance", "K": "Cluster Count",
            "sig2_S": "Similarity Variance"}
    n = len(geo_features)
    nrows = (n + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 5 * nrows))
    axes = axes.flatten()
    for i, feat in enumerate(geo_features):
        ax = axes[i]
        ax.scatter(feat_df[feat], feat_df["p_halluc"], c=feat_df["p_halluc"],
                   cmap="RdYlBu_r", s=12, alpha=0.4, edgecolors="none")
        r_p, _ = pearsonr(feat_df[feat], feat_df["p_halluc"])
        r_s, _ = spearmanr(feat_df[feat], feat_df["p_halluc"])
        ax.set_xlabel(nice.get(feat, feat)); ax.set_ylabel("p_halluc")
        ax.set_title(f"{nice.get(feat, feat)}\nr={r_p:.3f}, rho={r_s:.3f}", fontweight="bold", fontsize=10)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{title_prefix}Features vs Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig
