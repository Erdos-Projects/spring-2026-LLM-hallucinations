"""
spectral_detection/analysis/stats.py

Statistical testing utilities:
  - Global KS test (two-sample, per geometric feature)
  - Permutation test on mean entropy difference
  - Bootstrap confidence interval on AUC-ROC

⚠️  PER-DOMAIN versions of KS and Bootstrap are intentionally omitted.
    Domain labels in these datasets are LLM-generated, noisy, and uniformly
    sampled from benchmark sub-categories rather than curated domain strata.
    Splitting already-small per-benchmark samples (≈500 q) by domain produces
    sub-groups too small for reliable inference.  All statistical tests are
    therefore run at the benchmark level (≈500 q) and at the combined level.

Statistical test guide (for notebook markdown)
------------------------------------------------
1. Two-Sample Kolmogorov-Smirnov (KS) Test
   Null hypothesis H₀: the distribution of feature x_k is the same for
   hallucinated (y=1) and correct (y=0) questions.
   Test statistic D_n = sup_x |F̂_{y=1}(x) − F̂_{y=0}(x)| — the maximum
   gap between the two empirical CDFs.
   Interpretation: a large D_n (≈0.3–0.5) with p < α_Bonf means the feature
   separates the two classes non-parametrically; higher D_n → more separable.
   We apply Bonferroni correction: α_adj = 0.05 / 5 = 0.01.

2. Permutation Test on Mean Entropy Difference
   Null hypothesis H₀: the mean semantic entropy is the same for hallucinated
   and correct questions (i.e. entropy carries no signal).
   Procedure: compute observed Δ = mean(H|y=1) − mean(H|y=0), then repeat
   10 000 times with shuffled labels to build a null distribution.
   p-value = fraction of permutations yielding Δ ≥ Δ_obs.
   Interpretation: a small p-value (< 0.001) confirms that hallucinated
   questions have significantly higher entropy than correct ones — validating
   the core claim of Farquhar et al. (2024).

3. Bootstrap Confidence Interval on AUC-ROC
   Null hypothesis H₀ (implicit): AUC = 0.5, i.e. the classifier discriminates
   at chance level.
   Procedure: draw B = 2 000 bootstrap samples (with replacement), train RF on
   each in-bag set, score the out-of-bag set, collect AUC values, and report
   the 2.5th–97.5th percentile interval.
   Interpretation: if the 95% CI is entirely above 0.5, the classifier reliably
   outperforms chance.  A tight CI (width < 0.1) indicates stable performance
   regardless of which questions happen to land in the test fold.
   Bootstrap is run on full benchmarks and the combined dataset only — not on
   per-domain sub-groups, which are too small for stable OOB sets.

Methods sourced / adapted from hallucination_utils.py (Debanjan Bhattacharya).
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from spectral_detection.data.cleaning import GEO_FEATURES


# ── KS Tests ──────────────────────────────────────────────────────────────────

def run_global_ks_tests(feat_df, geo_features=None, label_col="label",
                         alpha=0.05, verbose=True):
    """
    Two-sample KS test for each geometric feature (hallucinated vs correct).
    Applies Bonferroni correction across the 5 features.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Question-level feature dataframe.
    geo_features : list, optional
        Feature columns to test (default: all 5 geometric features).
    label_col : str
    alpha : float
        Family-wise error rate before Bonferroni correction.
    verbose : bool

    Returns
    -------
    pd.DataFrame  with columns Feature, KS_stat, p_value, sig, Significant.

    Method adapted from hallucination_utils.py.
    """
    geo_features = geo_features or GEO_FEATURES
    n_tests = len(geo_features)
    alpha_bonf = alpha / n_tests

    if verbose:
        print(f"Global KS tests  (Bonferroni α = {alpha_bonf:.4f}, n_tests = {n_tests}):")

    rows = []
    for feat in geo_features:
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
            star = "✓" if p < alpha_bonf else " "
            print(f"  {star}  {feat:12s}  D={stat:.4f}  p={p:.2e}  {sig}")
    return pd.DataFrame(rows)


# ── Permutation Test ───────────────────────────────────────────────────────────

def run_permutation_test(feat_df, n_permutations=10000,
                          label_col="label", entropy_col="H_sem",
                          random_seed=42, verbose=True):
    """
    Permutation test: does mean semantic entropy differ between classes?

    H₀: mean(H | y=1) = mean(H | y=0).
    H₁: mean(H | y=1) > mean(H | y=0)  [one-sided].

    Parameters
    ----------
    feat_df : pd.DataFrame
    n_permutations : int
    label_col, entropy_col : str
    random_seed : int
    verbose : bool

    Returns
    -------
    delta_obs : float       Observed difference (bits).
    perm_deltas : np.ndarray  Null distribution (length n_permutations).
    perm_pval : float       One-sided p-value.

    Method from hallucination_utils.py.
    """
    ent_correct = feat_df.loc[feat_df[label_col] == 0, entropy_col].values
    ent_hallu   = feat_df.loc[feat_df[label_col] == 1, entropy_col].values
    delta_obs   = ent_hallu.mean() - ent_correct.mean()

    all_ent    = feat_df[entropy_col].values
    all_labels = feat_df[label_col].values
    rng = np.random.default_rng(random_seed)

    perm_deltas = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(all_labels)
        perm_deltas[i] = (all_ent[shuffled == 1].mean()
                          - all_ent[shuffled == 0].mean())

    perm_pval = (perm_deltas >= delta_obs).sum() / n_permutations

    if verbose:
        print(f"Permutation test ({n_permutations} iterations):")
        print(f"  Observed Δ = {delta_obs:.4f} bits")
        print(f"  p-value    = {perm_pval:.6f}")

    return delta_obs, perm_deltas, perm_pval


# ── Bootstrap AUC ─────────────────────────────────────────────────────────────

def run_bootstrap_auc(feat_df, geo_features=None, label_col="label",
                       n_bootstrap=2000, random_seed=42, verbose=True,
                       n_estimators=100):
    """
    Bootstrap 95% confidence interval on AUC-ROC for a Random Forest
    trained on the geometric features.

    Only run on full benchmarks or the combined dataset — not per domain.

    Parameters
    ----------
    feat_df : pd.DataFrame
    geo_features : list, optional
    label_col : str
    n_bootstrap : int
    random_seed : int
    verbose : bool
    n_estimators : int   Trees per bootstrap RF (kept low for speed).

    Returns
    -------
    auc_boot : np.ndarray    All valid OOB AUC values.
    ci_lo, ci_hi : float    2.5th and 97.5th percentile.

    Method adapted from hallucination_utils.py.
    """
    geo_features = geo_features or GEO_FEATURES

    X = StandardScaler().fit_transform(feat_df[geo_features].values)
    y = feat_df[label_col].values
    rng = np.random.default_rng(random_seed)
    n = len(y)

    auc_list = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) < 10 or len(np.unique(y[oob])) < 2:
            continue
        if len(np.unique(y[idx])) < 2:
            continue
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=6, random_state=0)
        rf.fit(X[idx], y[idx])
        proba = rf.predict_proba(X[oob])[:, 1]
        auc_list.append(roc_auc_score(y[oob], proba))

    auc_boot = np.array(auc_list)
    ci_lo, ci_hi = np.percentile(auc_boot, [2.5, 97.5])

    if verbose:
        print(f"Bootstrap AUC (RF, {len(geo_features)} geometric features, "
              f"B={n_bootstrap}):")
        print(f"  AUC = {auc_boot.mean():.4f}  "
              f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    return auc_boot, ci_lo, ci_hi