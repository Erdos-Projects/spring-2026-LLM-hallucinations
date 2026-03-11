"""
hallucination_utils.py
Shared data-loading, preprocessing, and EDA plotting functions for the
hallucination detection pipeline.

Supports all five datasets:
    DefAn, HaluEval, MMLU, TriviaQA, TruthfulQA

All five share these universal columns:
    id, question, reference_answer, model_answer, correctness,
    correctness_score, domain, dataset, adversarial, type

All five use the same three correctness labels:
    correct, incorrect, refused

DefAn is the only dataset without a pre-existing prompt_id column
(it is derived from the id field). The other four also carry:
    prompt_id, prompt_tokens, generated_tokens, sample_num,
    temperature, timestamp

Domain labels are assigned per-response by the LLM judge and are
frequently inconsistent across the 20 responses for a single question
(14-36% of questions depending on dataset). Datasets other than DefAn
have severe domain sprawl (30-90 unique strings). This module provides:
    - domain_mode: majority-vote domain per question
    - optional consolidation into canonical top-level categories

Usage:
    from hallucination_utils import (
        load_dataset, print_loading_summary,
        plot_response_label_breakdown, plot_feature_distributions, ...
    )
"""

import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ==================================================================
# CONSTANTS
# ==================================================================

# Labels present in all five datasets
LABEL_ORDER = ["correct", "incorrect", "refused"]

LABEL_COLORS = {
    "correct":   "#2196F3",
    "incorrect": "#E53935",
    "refused":   "#FFA726",
}

# The 5 embedding-geometry features
GEO_FEATURES = ["H_sem", "D_cos", "M_bar", "K", "sig2_S"]

FEAT_NICE_NAMES = {
    "H_sem":  "Semantic Entropy",
    "D_cos":  "Cosine Dispersion",
    "M_bar":  "Mahalanobis Distance",
    "K":      "Cluster Count",
    "sig2_S": "Similarity Variance",
}

# Canonical top-level domains for optional consolidation.
# Anything that doesn't match maps to "Other".
_CANONICAL_PATTERNS = [
    ("Humanities",            r"(?i)\bhumanit"),
    ("STEM",                  r"(?i)\bstem\b"),
    ("Social Sciences",       r"(?i)\bsocial\s*sci"),
    ("Medicine & Health",     r"(?i)\bmedic|\bhealth|\bnurse|\bpharmac"),
    ("Law & Business",        r"(?i)\blaw|\bbusiness|\bfinance|\becon|\bmarket|\btax|\baccoun"),
    ("Sports",                r"(?i)\bsport"),
    ("History",               r"(?i)\bhistor"),
    ("Science",               r"(?i)\bscien|\bbiol|\bphysic|\bchemist|\bnatur"),
    ("Mathematics",           r"(?i)\bmath|\bstatist|\blogic"),
    ("Entertainment",         r"(?i)\bentertain|\bfilm|\bmusic|\bmedia|\bcomic|\bpop\s*cult"),
    ("Geography",             r"(?i)\bgeograph|\btourism|\bclimate"),
    ("Literature",            r"(?i)\bliterat|\bfiction"),
    ("Religion & Philosophy", r"(?i)\brelig|\bphilosoph|\btheolog|\bspiritual"),
    ("Language",              r"(?i)\blinguist|\betymol|\blanguage|\bword\s*game"),
    ("Food & Nutrition",      r"(?i)\bfood|\bnutrit|\bculinar|\bcook|\bbeverage|\bconfect"),
    ("Psychology",            r"(?i)\bpsych"),
]


# ==================================================================
#  1. DATA LOADING & PREPROCESSING
# ==================================================================


def load_dataset(path, required_cols=None):
    """Load a JSONL dataset and ensure prompt_id and answer_len exist.

    Parameters
    ----------
    path : str
        Path to the .jsonl file.
    required_cols : list, optional
        Columns that must be present.  Raises ValueError if any missing.

    Returns
    -------
    pd.DataFrame
        Response-level dataframe, one row per model answer.
    """
    df = pd.read_json(path, lines=True)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # ensure prompt_id (DefAn needs derivation from id)
    if "prompt_id" not in df.columns:
        df["prompt_id"] = df["id"].astype(str).str.rsplit("_", n=1).str[0]

    # answer length
    if "answer_len" not in df.columns:
        df["answer_len"] = df["model_answer"].astype(str).str.len()

    return df


def load_all_datasets(data_dir):
    """Load every *.jsonl in data_dir.

    Returns a dict keyed by the dataset column value
    (e.g. 'defan', 'mmlu', ...).
    """
    datasets = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        df = load_dataset(os.path.join(data_dir, fname))
        name = df["dataset"].iloc[0]
        datasets[name] = df
    return datasets


# -- domain helpers ------------------------------------------------


def _mode(series):
    """Statistical mode; first value on ties."""
    m = series.mode()
    return m.iloc[0] if len(m) else series.iloc[0]


def consolidate_domain(raw_domain):
    """Map a raw (potentially noisy) domain string to a canonical category.

    Returns "Other" if no pattern matches.
    """
    for canonical, pattern in _CANONICAL_PATTERNS:
        if re.search(pattern, raw_domain):
            return canonical
    return "Other"


def add_canonical_domain(df, col="domain"):
    """Add a domain_canonical column via consolidate_domain."""
    df = df.copy()
    df["domain_canonical"] = df[col].apply(consolidate_domain)
    return df


# -- question-level metadata ---------------------------------------


def compute_question_metadata(df, domain_col="domain"):
    """Aggregate per-question metadata from a response-level dataframe.

    Returns a DataFrame indexed by prompt_id with columns:
        question, reference_answer, dataset, adversarial,
        domain_mode, n_unique_domains, domain_inconsistent,
        and type if it has more than one unique value.
    """
    agg = {
        "question":         ("question", "first"),
        "reference_answer": ("reference_answer", "first"),
        "dataset":          ("dataset", "first"),
        "adversarial":      ("adversarial", "first"),
        "domain_mode":      (domain_col, _mode),
        "n_unique_domains": (domain_col, "nunique"),
    }
    # type is only meaningful for DefAn; other datasets have '' everywhere
    if "type" in df.columns and df["type"].nunique() > 1:
        agg["type"] = ("type", "first")

    q_meta = df.groupby("prompt_id").agg(**agg)
    q_meta["domain_inconsistent"] = q_meta["n_unique_domains"] > 1
    return q_meta


def questions_per_domain(df, domain_col="domain"):
    """Unique question counts per domain, descending."""
    return (
        df.groupby(domain_col)["prompt_id"]
        .nunique()
        .sort_values(ascending=False)
    )


def split_analysis_domains(feat_df, min_questions, domain_col="domain"):
    """Return (analysis_domains, excluded_domains) based on a
    minimum question-count threshold."""
    counts = feat_df[domain_col].value_counts()
    analysis = sorted(counts[counts >= min_questions].index.tolist())
    excluded = sorted(set(counts.index) - set(analysis))
    return analysis, excluded


# ==================================================================
#  2. PRINTING / DIAGNOSTIC HELPERS
# ==================================================================


def print_loading_summary(df, correctness_col="correctness"):
    """Quick summary: shape, question count, samples/question, correctness,
    domain and type listings."""
    ds_name = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
    spq = df.groupby("prompt_id").size()
    n_domains = df["domain"].nunique()
    print(f"Dataset:          {ds_name}")
    print(f"Total rows:       {len(df)}")
    print(f"Unique questions: {df['prompt_id'].nunique()}")
    print(
        f"Samples/question: min={spq.min()}, max={spq.max()}, "
        f"median={spq.median()}"
    )
    print(f"\nCorrectness distribution:")
    vc = df[correctness_col].value_counts()
    for lab in LABEL_ORDER:
        if lab in vc.index:
            pct = vc[lab] / len(df) * 100
            print(f"  {lab:12s}: {vc[lab]:6d}  ({pct:.1f}%)")
    preview = sorted(df["domain"].unique())
    trunc = f"  (showing 10 of {n_domains})" if n_domains > 10 else ""
    print(f"\nDomains ({n_domains} unique){trunc}:")
    for d in preview[:10]:
        print(f"  {d}")
    if n_domains > 10:
        print(f"  ...")
    if "type" in df.columns and df["type"].nunique() > 1:
        print(f"\nAnswer types: {sorted(df['type'].unique())}")


def print_domain_consistency(q_meta):
    """Show domain inconsistency statistics."""
    n_inc = q_meta["domain_inconsistent"].sum()
    n_tot = len(q_meta)
    print(f"Domain inconsistency: {n_inc}/{n_tot} questions "
          f"({n_inc / n_tot * 100:.1f}%)")
    print(f"Max unique domains per question: {q_meta['n_unique_domains'].max()}")


def print_filtering_diagnostic(
    feat_df, raw_domain_counts, skipped, min_questions,
    domain_col="domain", skipped_details=None,
):
    """Print what was dropped, surviving counts, and threshold flags.

    Returns (analysis_domains, excluded_domains).
    """
    if skipped > 0 and skipped_details:
        df_skip = pd.DataFrame(skipped_details)
        print(f"Skipped {skipped} questions:")
        grp = domain_col if domain_col in df_skip.columns else "domain"
        print(df_skip.groupby(grp).size().sort_values(ascending=False).to_string())
        print()
    else:
        print("No questions were skipped.\n")

    survived = feat_df[domain_col].value_counts().sort_values(ascending=False)
    print("Surviving questions per domain:")
    for dom, n in survived.items():
        raw_n = raw_domain_counts.get(dom, 0)
        dropped = raw_n - n
        flag = "  *** BELOW THRESHOLD" if n < min_questions else ""
        print(f"  {dom:40s}: {n:4d} / {raw_n:4d}  (dropped {dropped}){flag}")

    analysis, excluded = split_analysis_domains(feat_df, min_questions, domain_col)
    print(f"\nDomains for ML (>= {min_questions} questions): {len(analysis)}")
    if excluded:
        print(f"Excluded: {excluded}")
    return analysis, excluded


# ==================================================================
#  3. RESPONSE-LEVEL EDA PLOTS
# ==================================================================


def plot_response_label_breakdown(
    df, label_order=None, label_colors=None,
    domain_col="domain", correctness_col="correctness",
    domain_question_counts=None, show_type_panel=True,
    figsize=(16, 6), title_prefix="Response Label Breakdown",
):
    """Stacked horizontal bar charts of response-level proportions by domain
    (and optionally by answer type).

    The type panel is automatically suppressed when type has only one
    unique value (all datasets except DefAn).
    """
    label_order = label_order or LABEL_ORDER
    colors = label_colors or LABEL_COLORS

    ct_domain = pd.crosstab(df[domain_col], df[correctness_col])
    ct_domain = ct_domain.reindex(columns=[c for c in label_order if c in ct_domain.columns])
    ct_pct = ct_domain.div(ct_domain.sum(axis=1), axis=0).fillna(0)

    has_type = show_type_panel and "type" in df.columns and df["type"].nunique() > 1
    n_panels = 2 if has_type else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # -- domain panel --
    sort_col = "incorrect" if "incorrect" in ct_pct.columns else ct_pct.columns[0]
    ct_plot = ct_pct.sort_values(sort_col, ascending=True)
    if domain_question_counts is not None:
        y_labels = [f"{d}  (n={domain_question_counts.get(d, '?')})" for d in ct_plot.index]
    else:
        y_labels = ct_plot.index.tolist()

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

    # -- type panel (DefAn only) --
    if has_type:
        ct_type = pd.crosstab(df["type"], df[correctness_col])
        ct_type = ct_type.reindex(columns=[c for c in label_order if c in ct_type.columns])
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


def plot_label_proportion_heatmap(
    df, label_order=None, domain_col="domain",
    correctness_col="correctness", figsize=(9, 6),
    title="Response Label Proportions by Domain",
):
    """Annotated heatmap of response-level label proportions per domain."""
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


# ==================================================================
#  4. QUESTION-LEVEL EDA PLOTS
# ==================================================================


def plot_question_label_profiles(
    feat_df, frac_cols, frac_nice_names, frac_colors,
    domain_col="domain", figsize=(12, 6),
    title="Average Response Label Profile by Domain",
):
    """Stacked horizontal bar of mean label fractions per domain."""
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


def plot_naive_vs_strict_rate(
    feat_df, naive_col, strict_col, size_col,
    domain_col="domain", figsize=(8, 6),
    title="Naive vs Strict Rate", xlabel="Naive Rate",
    ylabel="Strict Rate", size_label="refusal fraction",
):
    """Scatter of naive vs strict rate per domain, bubble-sized by size_col."""
    rate = feat_df.groupby(domain_col).agg(
        naive_rate=(naive_col, "mean"),
        strict_rate=(strict_col, "mean"),
        size_val=(size_col, "mean"),
        n_questions=("label", "count"),
    ).sort_values("naive_rate", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    for dom in rate.index:
        row = rate.loc[dom]
        ax.scatter(row["naive_rate"], row["strict_rate"],
                   s=50 + 400 * row["size_val"], alpha=0.7)
        ax.annotate(f"{dom} (n={int(row['n_questions'])})",
                    (row["naive_rate"], row["strict_rate"]),
                    fontsize=7, ha="left", va="bottom")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (bubble size = {size_label})", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_rate_distributions(
    feat_df, cols, colors, titles, figsize=None,
):
    """Histograms of per-question rate columns with mean lines."""
    n = len(cols)
    figsize = figsize or (7 * n, 5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, col, color, ttl in zip(axes, cols, colors, titles):
        ax.hist(feat_df[col], bins=30, color=color, alpha=0.7, edgecolor="white")
        mu = feat_df[col].mean()
        ax.axvline(mu, color="black", linestyle="--", label=f"Mean = {mu:.3f}")
        ax.set_xlabel(col)
        ax.set_ylabel("Number of Questions")
        ax.set_title(ttl, fontweight="bold")
        ax.legend()
    plt.tight_layout()
    return fig


def plot_domain_consistency(q_meta, figsize=(6, 4)):
    """Bar chart of how many unique domains per question."""
    fig, ax = plt.subplots(figsize=figsize)
    q_meta["n_unique_domains"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Number of unique domains per question", fontweight="bold")
    ax.set_xlabel("n_unique_domains")
    ax.set_ylabel("n questions")
    plt.tight_layout()
    return fig


# ==================================================================
#  5. GEOMETRIC FEATURE PLOTS
# ==================================================================


def plot_feature_distributions(
    feat_df, geo_features=None, feat_nice=None,
    label_col="label", label_names=None,
    figsize=(16, 10), suptitle="Geometric Feature Distributions",
):
    """KDE + histogram for each geometric feature, split by binary label."""
    geo_features = geo_features or GEO_FEATURES
    feat_nice = feat_nice or FEAT_NICE_NAMES
    label_names = label_names or {0: "Correct", 1: "Hallucinated"}
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


def plot_per_domain_feature_kdes(
    feat_df, domains_ordered, geo_features=None,
    feat_nice=None, domain_col="domain", label_col="label",
    analysis_domains=None, figsize_per_cell=(3.5, 3),
    suptitle="Feature Distributions by Domain",
):
    """Grid of per-domain KDEs: rows = features, columns = domains."""
    geo_features = geo_features or GEO_FEATURES
    feat_nice = feat_nice or FEAT_NICE_NAMES
    nd, nf = len(domains_ordered), len(geo_features)

    fig, axes = plt.subplots(
        nf, nd,
        figsize=(figsize_per_cell[0] * nd, figsize_per_cell[1] * nf),
        sharey=False,
    )
    if nd == 1:
        axes = axes.reshape(-1, 1)

    for row, feat in enumerate(geo_features):
        nice = feat_nice.get(feat, feat)
        for col, dom in enumerate(domains_ordered):
            ax = axes[row][col]
            sub = feat_df[feat_df[domain_col] == dom]
            for lv, c, nm in [(0, "steelblue", "Correct"),
                               (1, "tomato", "Hallucinated")]:
                vals = sub[sub[label_col] == lv][feat]
                if len(vals) > 2:
                    try:
                        vals.plot.kde(ax=ax, color=c, label=nm, linewidth=2)
                    except Exception:
                        pass
                    ax.axvline(vals.mean(), color=c, linestyle="--", alpha=0.4)
            if row == 0:
                ttl = f"{dom} (n={len(sub)})"
                if analysis_domains is not None and dom not in analysis_domains:
                    ttl += " *"
                ax.set_title(ttl, fontweight="bold", fontsize=8)
            ax.set_ylabel(nice if col == 0 else "", fontsize=8)
            if row == 0 and col == nd - 1:
                ax.legend(fontsize=7)

    sfx = "  (* = below ML threshold)" if analysis_domains else ""
    plt.suptitle(f"{suptitle}{sfx}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    feat_df, cols=None, figsize=(10, 8),
    title="Correlation Matrix",
):
    """Lower-triangle Pearson correlation heatmap."""
    cols = cols or GEO_FEATURES
    corr = feat_df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                mask=mask, ax=ax, linewidths=0.5)
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_entropy_vs_dispersion(
    feat_df, label_col="label", figsize=(9, 7),
    title="Entropy vs Dispersion",
):
    """Scatter of H_sem vs D_cos coloured by binary label."""
    fig, ax = plt.subplots(figsize=figsize)
    c = feat_df[label_col].map({0: "steelblue", 1: "tomato"})
    ax.scatter(feat_df["H_sem"], feat_df["D_cos"], c=c, alpha=0.3, s=12,
               edgecolors="none")
    ax.set_xlabel("Semantic Entropy (bits)")
    ax.set_ylabel("Cosine Dispersion")
    ax.set_title(title, fontweight="bold")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=8, label="Hallucinated"),
    ])
    plt.tight_layout()
    return fig


def plot_feature_pairplot(
    feat_df, geo_features=None, label_col="label",
    label_names=None, max_points=1500, random_seed=42,
    suptitle="Feature Pairplot",
):
    """Seaborn pairplot with KDE diagonals.  Downsamples for speed."""
    geo_features = geo_features or GEO_FEATURES
    label_names = label_names or {0: "Correct", 1: "Hallucinated"}

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
                     diag_kws={"fill": False, "common_norm": False},
                     height=2.2)
    g.fig.suptitle(suptitle, y=1.02, fontweight="bold")
    plt.tight_layout()
    return g.fig


def plot_domain_feature_heatmaps(
    feat_df, geo_features=None, domain_col="domain",
    label_col="label", figsize=(16, 5),
    suptitle="Feature Heatmap by Domain",
):
    """Side-by-side z-scored heatmaps (correct vs hallucinated)."""
    geo_features = geo_features or GEO_FEATURES
    names = {0: "Correct", 1: "Hallucinated"}
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (lv, nm) in zip(axes, names.items()):
        raw = (feat_df[feat_df[label_col] == lv]
               .groupby(domain_col)[geo_features].mean())
        z = (raw - raw.mean()) / (raw.std() + 1e-9)
        sns.heatmap(z, annot=raw.round(3), fmt=".3f", cmap="coolwarm",
                    center=0, ax=ax, cbar_kws={"label": "Z-score"})
        ax.set_title(f"{nm} -- Mean Features", fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
    plt.suptitle(f"{suptitle} (Z-scored background; raw values annotated)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ==================================================================
#  6. DOMAIN SUMMARY & HALLUCINATION RATE PLOTS
# ==================================================================


def build_domain_stats(
    feat_df, strict_rate_col, domain_col="domain",
    label_col="label", extra_mean_cols=None,
    analysis_domains=None,
):
    """Summary table of hallucination rate, entropy, question count
    per domain."""
    agg = {
        "n_questions":     (label_col, "count"),
        "n_hallucinated":  (label_col, "sum"),
        "hall_rate_mean":  (strict_rate_col, "mean"),
        "hall_rate_std":   (strict_rate_col, "std"),
        "mean_entropy":    ("H_sem", "mean"),
    }
    if extra_mean_cols:
        for c in extra_mean_cols:
            agg[f"mean_{c}"] = (c, "mean")

    ds = (feat_df.groupby(domain_col).agg(**agg)
          .sort_values("hall_rate_mean", ascending=False))
    ds["pct_hallucinated"] = (ds["n_hallucinated"] / ds["n_questions"] * 100).round(1)
    if analysis_domains is not None:
        ds["in_analysis"] = [d in analysis_domains for d in ds.index]
    return ds


def plot_hallucination_rate_by_domain(
    feat_df, domain_stats, strict_rate_col,
    domain_col="domain", figsize=(16, 6),
):
    """Bar + violin of hallucination rate per domain."""
    ds = domain_stats.reset_index()
    y_labels = [
        f"{row[domain_col]}  (n={int(row['n_questions'])})"
        for _, row in ds.iterrows()
    ]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(ds)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].barh(y_labels, ds["hall_rate_mean"] * 100,
                 xerr=ds["hall_rate_std"] * 100, color=colors, capsize=4)
    axes[0].set_xlabel("Mean Hallucination Rate (%)")
    axes[0].set_title("Hallucination Rate by Domain (Strict)", fontweight="bold")
    axes[0].axvline(50, color="gray", linestyle="--", alpha=0.5)

    sns.violinplot(data=feat_df, y=domain_col, x=strict_rate_col,
                   order=ds[domain_col].tolist(), orient="h",
                   palette="RdYlGn_r", ax=axes[1], inner="box")
    axes[1].set_xlabel("Per-Question Strict Hallucination Rate")
    axes[1].set_title("Rate Distribution", fontweight="bold")
    axes[1].axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


# ==================================================================
#  7. STATISTICAL TEST PLOTS
# ==================================================================


def plot_ks_heatmap(
    df_ks, geo_features=None, survived_counts=None,
    figsize=None, title="KS Statistic: Domain x Feature",
):
    """Heatmap of per-domain KS statistics with significance stars."""
    geo_features = geo_features or GEO_FEATURES
    ks_piv = (df_ks.pivot(index="Domain", columns="Feature", values="KS_stat")
              .fillna(0).reindex(columns=geo_features))
    sig_piv = (df_ks.pivot(index="Domain", columns="Feature", values="Significant")
               .fillna(False).reindex(columns=geo_features))

    if survived_counts is not None:
        y_labels = [f"{d}  (n={survived_counts.get(d, 0)})" for d in ks_piv.index]
    else:
        y_labels = ks_piv.index.tolist()

    nd = len(ks_piv)
    figsize = figsize or (10, max(4, nd * 0.6 + 1))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ks_piv, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=1, yticklabels=y_labels, ax=ax,
                cbar_kws={"label": "KS statistic"})
    for i, dom in enumerate(ks_piv.index):
        for j, feat in enumerate(ks_piv.columns):
            if bool(sig_piv.loc[dom, feat]):
                ax.text(j + 0.5, i + 0.78, "\u2605",
                        ha="center", va="center", color="black", fontsize=13)
    ax.set_title(f"{title}  (\u2605 = Bonferroni sig)", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_permutation_test(
    perm_deltas, delta_obs, n_permutations, perm_pval,
    figsize=(9, 5), title="Permutation Test: Semantic Entropy",
):
    """Histogram of null distribution with observed delta."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(perm_deltas, bins=60, color="gray", alpha=0.7, density=True,
            label="Null distribution")
    ax.axvline(delta_obs, color="red", linewidth=2.5, linestyle="--",
               label=f"Observed = {delta_obs:.4f}")
    ax.set_xlabel("Mean entropy difference (hallucinated - correct)")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}  (p = {perm_pval:.6f})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig


# ==================================================================
#  8. UMAP PLOT
# ==================================================================


def plot_umap(
    proj, labels, label_colors=None, label_order=None,
    figsize=(10, 8), title="UMAP of Response Embeddings",
):
    """Scatter of 2-D UMAP projection coloured by correctness label."""
    lc = label_colors or LABEL_COLORS
    c = [lc.get(l, "gray") for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(proj[:, 0], proj[:, 1], c=c, s=2, alpha=0.3)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    present = sorted(set(labels))
    order = label_order or LABEL_ORDER
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=lc.get(l, "gray"),
               markersize=8, label=l.capitalize())
        for l in order if l in present
    ])
    plt.tight_layout()
    return fig


# ==================================================================
#  9. EMBEDDING
# ==================================================================


def embed_responses(
    df,
    model_name="all-MiniLM-L6-v2",
    text_col="model_answer",
    cache_path=None,
    batch_size=256,
    normalize=True,
):
    """Embed all responses using a SentenceTransformer model.

    If *cache_path* is provided and the file exists, embeddings are loaded
    from disk.  Otherwise they are computed and optionally saved.

    Parameters
    ----------
    df : pd.DataFrame
        Response-level dataframe.
    model_name : str
        SentenceTransformer model identifier.
    text_col : str
        Column containing the text to embed.
    cache_path : str, optional
        Path to a .npy file for caching embeddings.
    batch_size : int
    normalize : bool
        L2-normalize embeddings (recommended for cosine-based features).

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (len(df), embedding_dim).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        embs = np.load(cache_path)
        assert embs.shape[0] == len(df), (
            f"Cache size mismatch: {embs.shape[0]} vs {len(df)} rows"
        )
        return embs

    from sentence_transformers import SentenceTransformer

    print(f"Computing embeddings with {model_name}...")
    embedder = SentenceTransformer(model_name)
    texts = df[text_col].astype(str).tolist()
    embs = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    if cache_path:
        np.save(cache_path, embs)
        print(f"Saved embeddings to {cache_path}")

    print(f"Embedding matrix: {embs.shape}")
    return embs


# ==================================================================
#  10. GEOMETRIC FEATURE FUNCTIONS
# ==================================================================


def semantic_entropy(embs, threshold=0.85):
    """Compute semantic entropy and cluster count from response embeddings.

    Agglomerative clustering groups responses whose cosine similarity
    exceeds *threshold*.  Entropy is computed over the resulting cluster
    size distribution.

    Parameters
    ----------
    embs : np.ndarray
        Response embeddings for a single question, shape (n, d).
    threshold : float
        Cosine similarity threshold for merging clusters.

    Returns
    -------
    H : float
        Shannon entropy in bits.
    K : int
        Number of clusters.
    """
    n = len(embs)
    dist_matrix = np.clip(1.0 - cosine_similarity(embs), 0, 2)
    np.fill_diagonal(dist_matrix, 0)

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="precomputed",
            linkage="average", distance_threshold=1 - threshold,
        )
    except TypeError:
        # older sklearn versions use affinity instead of metric
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity="precomputed",
            linkage="average", distance_threshold=1 - threshold,
        )

    labels = clustering.fit_predict(dist_matrix)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    H = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return H, int(len(counts))


def cosine_dispersion(embs):
    """Mean cosine distance from each response embedding to the centroid.

    Parameters
    ----------
    embs : np.ndarray
        Shape (n, d).

    Returns
    -------
    float
    """
    centroid = embs.mean(axis=0, keepdims=True)
    return float(np.mean(1.0 - cosine_similarity(embs, centroid).flatten()))


def mahalanobis_distance(embs, mu, cov_inv):
    """Mean Mahalanobis distance of response embeddings from a reference.

    Parameters
    ----------
    embs : np.ndarray
        Shape (n, d).
    mu : np.ndarray
        Reference mean, shape (d,).
    cov_inv : np.ndarray
        Inverse covariance (precision) matrix, shape (d, d).

    Returns
    -------
    float
    """
    diffs = embs - mu
    mahal_sq = np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)
    mahal_sq = np.clip(mahal_sq, 0, None)
    return float(np.mean(np.sqrt(mahal_sq)))


def similarity_variance(embs):
    """Variance of pairwise cosine similarities among response embeddings.

    Parameters
    ----------
    embs : np.ndarray
        Shape (n, d).

    Returns
    -------
    float
    """
    sim = cosine_similarity(embs)
    upper = sim[np.triu_indices(len(embs), k=1)]
    return float(np.var(upper))


# ==================================================================
#  11. REFERENCE DISTRIBUTION FITTING
# ==================================================================


def fit_reference_distribution(all_embeddings, df, correctness_col="correctness"):
    """Fit the Mahalanobis reference distribution on correct-labeled responses.

    Uses Ledoit-Wolf shrinkage for a well-conditioned covariance estimate.

    Parameters
    ----------
    all_embeddings : np.ndarray
        Full embedding matrix, shape (len(df), d).
    df : pd.DataFrame
        Response-level dataframe aligned with all_embeddings.
    correctness_col : str
        Column containing correctness labels.

    Returns
    -------
    mu_ref : np.ndarray
        Reference mean, shape (d,).
    cov_inv : np.ndarray
        Precision matrix, shape (d, d).
    """
    correct_mask = (df[correctness_col] == "correct").values
    correct_embs = all_embeddings[correct_mask]
    print(f"Correct responses for reference: {correct_embs.shape[0]}")

    if correct_embs.shape[0] < 10:
        print("WARNING: very few correct responses. Using all embeddings as fallback.")
        correct_embs = all_embeddings

    mu_ref = correct_embs.mean(axis=0)
    lw = LedoitWolf()
    lw.fit(correct_embs)
    cov_inv = lw.precision_
    print(f"Reference fitted. mu shape: {mu_ref.shape}, precision shape: {cov_inv.shape}")
    return mu_ref, cov_inv


# ==================================================================
#  12. QUESTION-LEVEL FEATURE EXTRACTION
# ==================================================================


def extract_question_features(
    df,
    all_embeddings,
    mu_ref,
    cov_inv,
    sim_threshold=0.85,
    domain_col="domain",
    correctness_col="correctness",
):
    """Extract geometric features and build the question-level feature dataframe.

    For each question (grouped by prompt_id):
      - Computes the 5 geometric features from all response embeddings
      - Counts correct / incorrect / refused responses
      - Derives a strict hallucination rate (correct vs incorrect only)
      - Assigns a binary label via majority vote on the strict rate

    Questions where all responses are refused (n_definitive == 0) are skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Response-level dataframe with prompt_id and correctness_col.
    all_embeddings : np.ndarray
        Embedding matrix aligned with df, shape (len(df), d).
    mu_ref : np.ndarray
        Reference mean from fit_reference_distribution.
    cov_inv : np.ndarray
        Precision matrix from fit_reference_distribution.
    sim_threshold : float
        Cosine similarity threshold for semantic entropy clustering.
    domain_col : str
        Column for domain (raw or canonical).
    correctness_col : str
        Column containing correctness labels.

    Returns
    -------
    feat_df : pd.DataFrame
        One row per question with geometric features, label fractions,
        and the binary label column.
    skipped : int
        Number of questions skipped (all refused).
    skipped_details : list[dict]
        Details of each skipped question.
    """
    # pre-compute question metadata for domain_mode
    q_meta = compute_question_metadata(df, domain_col=domain_col)

    prompt_ids = df["prompt_id"].unique()
    records = []
    skipped = 0
    skipped_details = []

    for i, pid in enumerate(prompt_ids):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(prompt_ids)} ...")

        mask = df["prompt_id"] == pid
        sub = df[mask]
        idx = np.where(mask.values)[0]
        embs = all_embeddings[idx]
        n = len(sub)

        # domain from question metadata
        dom = q_meta.loc[pid, "domain_mode"]
        dom_incon = bool(q_meta.loc[pid, "domain_inconsistent"])

        # count each label
        counts = sub[correctness_col].value_counts()
        n_correct   = int(counts.get("correct", 0))
        n_incorrect = int(counts.get("incorrect", 0))
        n_refused   = int(counts.get("refused", 0))

        # binary label: majority vote on correct vs incorrect only
        n_definitive = n_correct + n_incorrect
        if n_definitive == 0:
            skipped += 1
            skipped_details.append({
                "prompt_id": pid,
                domain_col: dom,
                "n_refused": n_refused,
                "n_samples": n,
            })
            continue

        hall_rate_strict = n_incorrect / n_definitive
        label = 1 if hall_rate_strict > 0.5 else 0

        # geometric features (all embeddings)
        H, K = semantic_entropy(embs, threshold=sim_threshold)
        D = cosine_dispersion(embs)
        M = mahalanobis_distance(embs, mu_ref, cov_inv)
        sig2 = similarity_variance(embs)

        # auxiliary stats
        score_mean = float(sub["correctness_score"].mean())
        score_std  = float(sub["correctness_score"].std(ddof=0))
        len_mean   = float(sub["answer_len"].mean())
        len_std    = float(sub["answer_len"].std(ddof=0))

        row0 = sub.iloc[0]
        rec = {
            "prompt_id": pid,
            "question": row0["question"],
            domain_col: dom,
            "domain_inconsistent": dom_incon,
            "adversarial": bool(row0["adversarial"]),
            "n_samples": n,
            # raw counts
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_refused": n_refused,
            "n_definitive": n_definitive,
            # fractions (out of all n responses)
            "frac_correct": n_correct / n,
            "frac_incorrect": n_incorrect / n,
            "frac_refused": n_refused / n,
            # strict rate (correct vs incorrect only)
            "hall_rate_strict": hall_rate_strict,
            # regression target (same value, named for regression pipeline)
            "p_halluc": hall_rate_strict,
            # naive rate (everything non-correct = bad)
            "hall_rate_naive": 1.0 - (n_correct / n),
            # auxiliary
            "score_mean": score_mean,
            "score_std": score_std,
            "len_mean": len_mean,
            "len_std": len_std,
            # geometric features
            "H_sem": H,
            "D_cos": D,
            "M_bar": M,
            "K": K,
            "sig2_S": sig2,
            # binary target
            "label": label,
        }

        # include type if meaningful
        if "type" in sub.columns and df["type"].nunique() > 1:
            rec["type"] = row0["type"]

        records.append(rec)

    feat_df = pd.DataFrame(records)

    print(f"\nQuestions processed: {len(feat_df)}")
    print(f"Questions skipped (all refused): {skipped}")
    print(f"Label distribution:")
    print(f"  Correct (0): {(feat_df['label'] == 0).sum()}")
    print(f"  Hallucinated (1): {(feat_df['label'] == 1).sum()}")
    print(f"  Rate: {feat_df['label'].mean() * 100:.1f}%")

    return feat_df, skipped, skipped_details
