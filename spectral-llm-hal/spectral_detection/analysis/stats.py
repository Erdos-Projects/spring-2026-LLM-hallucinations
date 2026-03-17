"""
spectral_detection/analysis/stats.py

Statistical testing:
  1. Two-sample KS test per geometric feature (Bonferroni-corrected)
  2. Permutation test on mean entropy difference
  3. Bootstrap confidence interval on AUC-ROC

All tests run at benchmark level (~500 q) and combined level (2500 q).
Per-domain splits are too small for reliable inference and are omitted.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from spectral_detection.data.cleaning import FEATURES


# ── KS Tests ──────────────────────────────────────────────────────────────────

def run_ks_tests(feat_df, features=None, label_col="label",
                  alpha=0.05, verbose=True):
    """
    Two-sample KS test for each feature: hallucinated (y=1) vs correct (y=0).
    Applies Bonferroni correction across all tested features.

    Returns DataFrame with columns: Feature, KS_stat, p_value, sig, Significant.
    """
    features = features or FEATURES
    n_tests = len(features)
    alpha_bonf = alpha / n_tests

    if verbose:
        print(f"KS tests (Bonferroni alpha={alpha_bonf:.4f}, {n_tests} tests):")

    rows = []
    for feat in features:
        g0 = feat_df.loc[feat_df[label_col] == 0, feat].values
        g1 = feat_df.loc[feat_df[label_col] == 1, feat].values
        if len(g0) < 2 or len(g1) < 2:
            continue

        stat, p = ks_2samp(g0, g1)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        rows.append({
            "Feature": feat, "KS_stat": round(stat, 4),
            "p_value": p, "sig": sig,
            "Significant": p < alpha_bonf,
        })
        if verbose:
            mark = "+" if p < alpha_bonf else " "
            print(f"  {mark}  {feat:12s}  D={stat:.4f}  p={p:.2e}  {sig}")

    return pd.DataFrame(rows)


# Backward-compatible alias
run_global_ks_tests = run_ks_tests


# ── Permutation Test ──────────────────────────────────────────────────────────

def run_permutation_test(feat_df, n_permutations=10_000,
                          label_col="label", entropy_col="H_sem",
                          random_seed=42, verbose=True):
    """
    One-sided permutation test: mean(feature | y=1) > mean(feature | y=0).
    Returns (delta_observed, null_distribution, p_value).
    """
    vals_correct = feat_df.loc[feat_df[label_col] == 0, entropy_col].values
    vals_hallu   = feat_df.loc[feat_df[label_col] == 1, entropy_col].values
    delta_obs    = vals_hallu.mean() - vals_correct.mean()

    all_vals   = feat_df[entropy_col].values
    all_labels = feat_df[label_col].values
    rng = np.random.default_rng(random_seed)

    null_deltas = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(all_labels)
        null_deltas[i] = all_vals[shuffled == 1].mean() - all_vals[shuffled == 0].mean()

    # +1/+1 correction (Phipson & Smyth 2010): avoids p=0 and accounts for
    # the observed statistic being one valid permutation
    p_value = ((null_deltas >= delta_obs).sum() + 1) / (n_permutations + 1)

    if verbose:
        print(f"Permutation test ({n_permutations:,} iterations):")
        print(f"  Observed delta = {delta_obs:.4f}")
        print(f"  p-value        = {p_value:.6f}")

    return delta_obs, null_deltas, p_value


# ── Bootstrap AUC ─────────────────────────────────────────────────────────────

def run_bootstrap_auc(feat_df, features=None, geo_features=None, label_col="label",
                       n_bootstrap=2000, random_seed=42, verbose=True,
                       n_estimators=100):
    """
    Bootstrap 95% CI on AUC-ROC for a Random Forest on the given features.
    Returns (auc_samples, ci_lower, ci_upper).
    """
    features = features or geo_features or FEATURES

    # Note: StandardScaler is technically unnecessary for RF (tree-based models
    # are scale-invariant), but kept for consistency if swapped to other models.
    X = StandardScaler().fit_transform(feat_df[features].values)
    y = feat_df[label_col].values
    rng = np.random.default_rng(random_seed)
    n = len(y)

    auc_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) < 10 or len(np.unique(y[oob])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=6, random_state=0)
        rf.fit(X[idx], y[idx])
        auc_samples.append(roc_auc_score(y[oob], rf.predict_proba(X[oob])[:, 1]))

    auc_arr = np.array(auc_samples)
    ci_lo, ci_hi = np.percentile(auc_arr, [2.5, 97.5])

    if verbose:
        print(f"Bootstrap AUC (RF, {len(features)} features, B={n_bootstrap}):")
        print(f"  AUC = {auc_arr.mean():.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    return auc_arr, ci_lo, ci_hi
