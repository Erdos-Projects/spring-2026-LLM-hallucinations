"""
spectral_detection/visualization.py

All plotting functions for the hallucination-detection pipeline.

PLOTS INCLUDED:
  - Response-level stacked bar by domain
  - Label proportion heatmap by domain
  - Question label profiles by domain (stacked bar of fractions)
  - Hallucination rate by domain (bar + violin)
  - Geometric feature distributions (global, split by label) [KDE + histogram]
  - Correlation matrix
  - Entropy vs Dispersion scatter
  - Feature pairplot
  - Permutation test histogram
  - Bootstrap AUC bar chart
  - ROC curves (per-feature + combined)
  - Ablation grouped bar chart
  - Per-domain classification heatmap
  - SHAP beeswarm (per benchmark + combined — NOT per domain)

PLOTS EXCLUDED (by project specification):
  - Per-domain feature KDE grids
  - Feature heatmap by domain
  - KS statistic heatmap by domain
  - Normalised gain importance per domain
  - SHAP per domain
  - UMAP projections

Methods sourced / adapted from hallucination_utils.py (Debanjan Bhattacharya).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from spectral_detection.data.cleaning import (
    GEO_FEATURES, FEAT_NICE_NAMES, LABEL_ORDER, LABEL_COLORS,
)

sns.set_style("whitegrid")


# ── Response-level EDA ─────────────────────────────────────────────────────────

def plot_response_label_breakdown(df, label_order=None, label_colors=None,
                                   domain_col="domain",
                                   correctness_col="correctness",
                                   domain_question_counts=None,
                                   show_type_panel=True,
                                   figsize=(16, 6),
                                   title_prefix="Response Label Breakdown"):
    """
    Stacked horizontal bar charts of response-level proportions by domain
    (and optionally by answer type, for DefAn only).
    Method from hallucination_utils.py.
    """
    label_order = label_order or LABEL_ORDER
    colors = label_colors or LABEL_COLORS

    ct_domain = pd.crosstab(df[domain_col], df[correctness_col])
    ct_domain = ct_domain.reindex(
        columns=[c for c in label_order if c in ct_domain.columns])
    ct_pct = ct_domain.div(ct_domain.sum(axis=1), axis=0).fillna(0)

    has_type = (show_type_panel and "type" in df.columns
                and df["type"].nunique() > 1)
    n_panels = 2 if has_type else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    sort_col = "incorrect" if "incorrect" in ct_pct.columns else ct_pct.columns[0]
    ct_plot = ct_pct.sort_values(sort_col, ascending=True)
    y_labels = (
        [f"{d}  (n={domain_question_counts.get(d, '?')})" for d in ct_plot.index]
        if domain_question_counts is not None else ct_plot.index.tolist()
    )

    bottom = np.zeros(len(ct_plot))
    for lab in label_order:
        if lab not in ct_plot.columns:
            continue
        vals = ct_plot[lab].values
        axes[0].barh(y_labels, vals, left=bottom,
                     color=colors.get(lab, "gray"), label=lab)
        bottom += vals
    axes[0].set_xlabel("Proportion of Responses")
    axes[0].set_title(f"{title_prefix} by Domain", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_xlim(0, 1)

    if has_type:
        ct_type = pd.crosstab(df["type"], df[correctness_col])
        ct_type = ct_type.reindex(
            columns=[c for c in label_order if c in ct_type.columns])
        ct_t_pct = ct_type.div(ct_type.sum(axis=1), axis=0).fillna(0)
        ct_t_pct = ct_t_pct.sort_values(sort_col, ascending=True)
        bottom = np.zeros(len(ct_t_pct))
        for lab in label_order:
            if lab not in ct_t_pct.columns:
                continue
            vals = ct_t_pct[lab].values
            axes[1].barh(ct_t_pct.index, vals, left=bottom,
                         color=colors.get(lab, "gray"), label=lab)
            bottom += vals
        axes[1].set_xlabel("Proportion of Responses")
        axes[1].set_title(f"{title_prefix} by Answer Type", fontweight="bold")
        axes[1].legend(loc="lower right", fontsize=9)
        axes[1].set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_label_proportion_heatmap(df, label_order=None, domain_col="domain",
                                   correctness_col="correctness",
                                   figsize=(9, 6),
                                   title="Response Label Proportions by Domain"):
    """Annotated heatmap of response-level label proportions per domain.
    Method from hallucination_utils.py."""
    label_order = label_order or LABEL_ORDER
    ct = pd.crosstab(df[domain_col], df[correctness_col])
    ct = ct.reindex(columns=[c for c in label_order if c in ct.columns])
    ct_pct = ct.div(ct.sum(axis=1), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ct_pct, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


# ── Question-level EDA ─────────────────────────────────────────────────────────

def plot_question_label_profiles(feat_df, frac_cols, frac_nice_names,
                                  frac_colors, domain_col="domain",
                                  figsize=(12, 6),
                                  title="Average Response Label Profile by Domain"):
    """Stacked horizontal bar of mean label fractions per domain.
    Method from hallucination_utils.py."""
    domain_fracs = (
        feat_df.groupby(domain_col)[frac_cols].mean()
        .sort_values(frac_cols[1], ascending=False)
    )
    survived = feat_df[domain_col].value_counts()
    df_plot = domain_fracs.reset_index()
    y_labels = [f"{d}  (n={survived.get(d, '?')})" for d in df_plot[domain_col]]

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(df_plot))
    for col, nice, color in zip(frac_cols, frac_nice_names, frac_colors):
        vals = df_plot[col].values
        ax.barh(y_labels, vals, left=bottom, color=color, label=nice)
        bottom += vals
    ax.set_xlabel("Mean Fraction of Responses")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return fig


def plot_hallucination_rate_by_domain(feat_df, domain_stats,
                                       strict_rate_col,
                                       domain_col="domain",
                                       figsize=(16, 6)):
    """
    Bar + violin of hallucination rate per domain.
    Method from hallucination_utils.py.
    """
    ds = domain_stats.reset_index()
    y_labels = [
        f"{row[domain_col]}  (n={int(row['n_questions'])})"
        for _, row in ds.iterrows()
    ]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(ds)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].barh(y_labels, ds["hall_rate_mean"] * 100,
                 xerr=ds["hall_rate_std"] * 100, color=colors, capsize=4)
    axes[0].set_xlabel("Mean Hallucination Rate (%) — refusals counted as hallucinations")
    axes[0].set_title("Hallucination Rate by Domain", fontweight="bold")
    axes[0].axvline(50, color="gray", linestyle="--", alpha=0.5)

    sns.violinplot(data=feat_df, y=domain_col, x=strict_rate_col,
                   order=ds[domain_col].tolist(), orient="h",
                   palette="RdYlGn_r", ax=axes[1], inner="box")
    axes[1].set_xlabel("Per-Question Hallucination Rate")
    axes[1].set_title("Rate Distribution (violin)", fontweight="bold")
    axes[1].axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


# ── Geometric feature distributions ───────────────────────────────────────────

def plot_feature_distributions(feat_df, geo_features=None, feat_nice=None,
                                label_col="label", label_names=None,
                                figsize=(16, 10),
                                suptitle="Geometric Feature Distributions"):
    """
    KDE + histogram for each geometric feature, split by binary label.
    Method from hallucination_utils.py.
    """
    geo_features = geo_features or GEO_FEATURES
    feat_nice    = feat_nice or FEAT_NICE_NAMES
    label_names  = label_names or {0: "Correct", 1: "Hallucinated (incl. refused)"}
    _lc = {0: "steelblue", 1: "tomato"}

    nrows = (len(geo_features) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=figsize)
    axes = axes.flatten()
    for i, feat in enumerate(geo_features):
        ax = axes[i]
        for lv in sorted(label_names):
            vals = feat_df.loc[feat_df[label_col] == lv, feat]
            c = _lc.get(lv, "gray")
            ax.hist(vals, bins=40, alpha=0.35, density=True, color=c,
                    label=label_names[lv])
            if len(vals) > 2:
                try:
                    vals.plot.kde(ax=ax, color=c, linewidth=2)
                except Exception:
                    pass
        ax.set_title(feat_nice.get(feat, feat), fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    for j in range(len(geo_features), len(axes)):
        axes[j].axis("off")
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(feat_df, cols=None, figsize=(10, 8),
                             title="Correlation Matrix"):
    """Lower-triangle Pearson correlation heatmap.
    Method from hallucination_utils.py."""
    cols = cols or GEO_FEATURES
    corr = feat_df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, mask=mask, ax=ax, linewidths=0.5)
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_entropy_vs_dispersion(feat_df, label_col="label",
                                figsize=(9, 7),
                                title="Entropy vs Dispersion"):
    """Scatter of H_sem vs D_cos coloured by binary label.
    Method from hallucination_utils.py."""
    fig, ax = plt.subplots(figsize=figsize)
    c = feat_df[label_col].map({0: "steelblue", 1: "tomato"})
    ax.scatter(feat_df["H_sem"], feat_df["D_cos"], c=c,
               alpha=0.3, s=12, edgecolors="none")
    ax.set_xlabel("Semantic Entropy (bits)")
    ax.set_ylabel("Cosine Dispersion")
    ax.set_title(title, fontweight="bold")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=8, label="Hallucinated (incl. refused)"),
    ])
    plt.tight_layout()
    return fig


def plot_feature_pairplot(feat_df, geo_features=None, label_col="label",
                           label_names=None, max_points=1500,
                           random_seed=42, suptitle="Feature Pairplot"):
    """Seaborn pairplot with KDE diagonals.  Downsamples for speed.
    Method from hallucination_utils.py."""
    geo_features = geo_features or GEO_FEATURES
    label_names  = label_names or {0: "Correct", 1: "Hallucinated"}

    pdf = feat_df[geo_features + [label_col]].copy()
    pdf[label_col] = pdf[label_col].map(label_names)
    if len(pdf) > max_points:
        pdf = (
            pdf.groupby(label_col, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max_points // 2),
                                       random_state=random_seed))
            .reset_index(drop=True)
        )
    palette = dict(zip(label_names.values(), ["steelblue", "tomato"]))
    g = sns.pairplot(pdf, hue=label_col, palette=palette, diag_kind="kde",
                     plot_kws={"alpha": 0.2, "s": 8, "edgecolor": "none"},
                     diag_kws={"fill": False, "common_norm": False}, height=2.2)
    g.fig.suptitle(suptitle, y=1.02, fontweight="bold")
    plt.tight_layout()
    return g.fig


# ── Statistical test plots ─────────────────────────────────────────────────────

def plot_permutation_test(perm_deltas, delta_obs, n_permutations, perm_pval,
                           figsize=(9, 5),
                           title="Permutation Test: Semantic Entropy"):
    """Histogram of null distribution with observed delta.
    Method from hallucination_utils.py."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(perm_deltas, bins=60, color="gray", alpha=0.7, density=True,
            label="Null distribution")
    ax.axvline(delta_obs, color="red", linewidth=2.5, linestyle="--",
               label=f"Observed = {delta_obs:.4f}")
    ax.set_xlabel("Mean entropy difference (hallucinated − correct)")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}  (p = {perm_pval:.6f})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_bootstrap_auc(auc_boot, ci_lo, ci_hi, dataset_name="",
                        figsize=(8, 4)):
    """Histogram of bootstrap AUC distribution with CI shading."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(auc_boot, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(auc_boot.mean(), color="navy", linewidth=2,
               label=f"Mean = {auc_boot.mean():.4f}")
    ax.axvline(ci_lo, color="red", linewidth=1.5, linestyle="--",
               label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    ax.axvline(ci_hi, color="red", linewidth=1.5, linestyle="--")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance (0.5)")
    ax.set_xlabel("Bootstrap AUC-ROC")
    ax.set_ylabel("Count")
    ttl = f"{dataset_name}: Bootstrap AUC Distribution" if dataset_name else "Bootstrap AUC Distribution"
    ax.set_title(ttl, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── Classification & ROC ──────────────────────────────────────────────────────

def plot_ablation_bar(df_clf, dataset_name="", figsize_aspect=2.5):
    """
    Grouped bar chart: AUC across feature subsets × classifiers.
    Method from hallucination_utils.py.
    """
    g = sns.catplot(
        data=df_clf, kind="bar",
        x="Variant", y="AUC_mean", hue="Classifier",
        palette="Set2", height=5.5, aspect=figsize_aspect,
        capsize=0.05, errwidth=1.5,
    )
    g.ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    g.ax.set_ylim(0.4, 1.0)
    ttl = f"{dataset_name}: Ablation — Feature Subsets × Classifier" if dataset_name else "Ablation — Feature Subsets × Classifier"
    g.ax.set_title(ttl, fontweight="bold")
    g.ax.set_ylabel("AUC-ROC (5-fold CV)")
    g.ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return g.fig


def plot_roc_curves(X_geo_sc, y_all, geo_features=None,
                    dataset_name="", random_seed=42, figsize=(9, 7)):
    """
    ROC curves for each individual feature plus the combined RF model.
    Method from hallucination_utils.py.

    How to interpret ROC curves
    ----------------------------
    - The x-axis is the False Positive Rate (FPR = FP / (FP+TN)) and the
      y-axis is the True Positive Rate (recall = TP / (TP+FN)).
    - A perfect classifier sits at the top-left corner (FPR=0, TPR=1).
    - The diagonal dashed line is chance performance (AUC=0.5).
    - AUC (Area Under Curve): 1.0 = perfect, 0.5 = random.
    - Curves that are higher and to the left are better.
    - Individual-feature curves show how discriminative each feature is alone.
    - The combined RF curve shows how much the features gain by working together.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, roc_curve

    geo_features = geo_features or GEO_FEATURES
    feat_colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, (feat, color) in enumerate(zip(geo_features, feat_colors)):
        vals = X_geo_sc[:, i]
        fpr, tpr, _ = roc_curve(y_all, vals)
        auc_s = roc_auc_score(y_all, vals)
        ax.plot(fpr, tpr, color=color, linewidth=1.5, alpha=0.8,
                label=f"{feat} (AUC={auc_s:.3f})")

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_seed)
    rf.fit(X_geo_sc, y_all)
    y_scores = rf.predict_proba(X_geo_sc)[:, 1]
    fpr_a, tpr_a, _ = roc_curve(y_all, y_scores)
    auc_a = roc_auc_score(y_all, y_scores)
    ax.plot(fpr_a, tpr_a, color="black", linewidth=2.5,
            label=f"All 5 — RF (AUC={auc_a:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ttl = f"{dataset_name}: ROC — Individual Features vs Combined" if dataset_name else "ROC — Individual Features vs Combined"
    ax.set_title(ttl, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    return fig


def plot_ablation_roc_curves(X_geo_sc, y_all, classifiers,
                              ablation_sets, dataset_name="",
                              random_seed=42, figsize=(11, 8)):
    """
    ROC curves for each ablation variant × classifier combination.
    One sub-panel per classifier, all ablation variants overlaid.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold
    import itertools

    n_clf = len(classifiers)
    fig, axes = plt.subplots(1, n_clf, figsize=figsize, sharey=True)
    if n_clf == 1:
        axes = [axes]

    variant_colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    for ax, (clf_name, clf) in zip(axes, classifiers.items()):
        for (variant, (_, feat_idx)), color in zip(ablation_sets.items(), variant_colors):
            X_sub = X_geo_sc[:, feat_idx]
            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 100)
            for train_idx, test_idx in cv.split(X_sub, y_all):
                if len(np.unique(y_all[train_idx])) < 2:
                    continue
                import copy
                clf_c = copy.deepcopy(clf)
                clf_c.fit(X_sub[train_idx], y_all[train_idx])
                fpr, tpr, _ = roc_curve(y_all[test_idx],
                                         clf_c.predict_proba(X_sub[test_idx])[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs.append(roc_auc_score(y_all[test_idx],
                                           clf_c.predict_proba(X_sub[test_idx])[:, 1]))
            if not tprs:
                continue
            mean_tpr = np.mean(tprs, axis=0)
            ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2,
                    label=f"{variant}  (AUC={np.mean(aucs):.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(clf_name, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")

    ttl = f"{dataset_name}: Ablation ROC Curves" if dataset_name else "Ablation ROC Curves"
    plt.suptitle(ttl, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_per_domain_clf_heatmap(df_dom_clf, dataset_name="", figsize=None):
    """Heatmap of per-domain AUC scores by classifier."""
    pivot = df_dom_clf.pivot_table(index="Domain", columns="Classifier",
                                   values="AUC_mean").round(3)
    figsize = figsize or (10, max(4, len(pivot) * 0.55 + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.4, vmax=1.0, ax=ax, linewidths=0.5,
                linecolor="white", cbar_kws={"label": "AUC-ROC"})
    ttl = f"{dataset_name}: Per-Domain AUC-ROC" if dataset_name else "Per-Domain AUC-ROC"
    ax.set_title(ttl, fontweight="bold")
    plt.tight_layout()
    return fig


# ── SHAP ──────────────────────────────────────────────────────────────────────

def plot_shap_beeswarm(X_sc, y, geo_features=None, title="SHAP Beeswarm",
                        figsize=(10, 5)):
    """
    Fit an XGBoost classifier and plot a SHAP beeswarm summary.
    For benchmark-level and combined-dataset SHAP only (not per domain).
    """
    import shap
    import xgboost as xgb

    geo_features = geo_features or GEO_FEATURES

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_model.fit(X_sc, y)

    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(X_sc)
    if isinstance(sv, list):
        sv = sv[1]

    fig = plt.figure(figsize=figsize)
    shap.summary_plot(sv, X_sc, feature_names=geo_features,
                      show=False, plot_size=None)
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    return fig