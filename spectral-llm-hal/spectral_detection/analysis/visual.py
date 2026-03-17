"""
visualisation script for Laplacian-eigenvalue hallucination detection.

Implements:
    V1 - Eigenvalue Distribution by Label (KDE)
    V3 - Layer-wise AUROC Profile
    V4 - ROC Curve Comparison Across Datasets
    V5 - Dataset Cross-Generalisation Heatmap

Skipped intentionally:
    V2
    Step 3 (bootstrap statistical validation)
    Step 4 (summary / figure pipeline)

Assumed input:
    A pandas DataFrame with:
      - one row per sample
      - a binary label column
      - a dataset-name column
      - columns for raw top-10 eigenvalue features for all layers/heads
      - optionally a baseline AttentionScore column

Feature naming convention assumed:
    eig_l{layer}_h{head}_k{rank}
where:
    layer in [0, ..., n_layers-1]
    head  in [0, ..., n_heads-1]
    rank  in [1, ..., top_k]

Example:
    eig_l0_h0_k1, eig_l0_h0_k2, ..., eig_l31_h31_k10

"""

from __future__ import annotations

import os
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import os
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import json
from scipy.stats import mannwhitneyu


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load dataframe from parquet/csv/pkl.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(path)

    raise ValueError(f"Unsupported file format: {suffix}")

def build_feature_column_names(
    n_layers: int,
    n_heads: int,
    top_k: int,
) -> List[str]:
    """
    Build assumed raw eigenvalue feature names.

    Expected naming:
        eig_l{layer}_h{head}_k{rank}
    """
    cols = []
    for l in range(n_layers):
        for h in range(n_heads):
            for k in range(1, top_k + 1):
                cols.append(f"eig_l{l}_h{h}_k{k}")
    return cols


def get_existing_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def get_rank_columns(df: pd.DataFrame, rank: int, n_layers: int, n_heads: int) -> List[str]:
    """
    Columns corresponding to eigenvalue rank k across all layers and heads.
    """
    cols = [f"eig_l{l}_h{h}_k{rank}" for l in range(n_layers) for h in range(n_heads)]
    return get_existing_columns(df, cols)


def get_layer_columns(df: pd.DataFrame, layer: int, n_heads: int, top_k: int) -> List[str]:
    """
    Columns corresponding to one layer across all heads and all top-k ranks.
    """
    cols = [f"eig_l{layer}_h{h}_k{k}" for h in range(n_heads) for k in range(1, top_k + 1)]
    return get_existing_columns(df, cols)


def get_top5_columns_from_top10(colnames: Sequence[str]) -> List[str]:
    """
    Select only k=1,...,5 from columns named like eig_l{layer}_h{head}_k{rank}
    """
    keep = []
    for c in colnames:
        if c.endswith(("_k1", "_k2", "_k3", "_k4", "_k5")):
            keep.append(c)
    return keep

def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    """
    Cohen's d for two groups.
    """
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]

    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return np.nan

    s0 = np.var(x0, ddof=1)
    s1 = np.var(x1, ddof=1)
    sp = np.sqrt(((n0 - 1) * s0 + (n1 - 1) * s1) / (n0 + n1 - 2))
    if sp == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x0)) / sp


def make_probe_pipeline(pca_dim: int) -> Pipeline:
    """
    Standard pipeline:
        impute -> scale -> PCA -> logistic regression
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_dim, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(
            penalty="l2",
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs"
        )),
    ])


def fit_predict_scores(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    model: Pipeline,
) -> np.ndarray:
    """
    Fit model and return predicted probabilities on test set.
    """
    m = clone(model)
    m.fit(X_train, y_train)
    return m.predict_proba(X_test)[:, 1]


def bootstrap_roc_band(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 200,
    random_state: int = 137,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean ROC and 95% bootstrap CI band.
    """
    rng = np.random.default_rng(random_state)
    fpr_grid = np.linspace(0, 1, 200)
    tprs = []

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_b = y_true[idx]
        s_b = y_score[idx]

        # Need both classes present
        if len(np.unique(y_b)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_b, s_b)
        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)

    if len(tprs) == 0:
        mean_tpr = np.full_like(fpr_grid, np.nan)
        low_tpr = np.full_like(fpr_grid, np.nan)
        high_tpr = np.full_like(fpr_grid, np.nan)
        return fpr_grid, low_tpr, high_tpr

    tprs = np.asarray(tprs)
    low_tpr = np.nanpercentile(tprs, 2.5, axis=0)
    high_tpr = np.nanpercentile(tprs, 97.5, axis=0)
    return fpr_grid, low_tpr, high_tpr

# ============================================================
# V1 — Eigenvalue Distribution by Label (KDE)
# ============================================================

def plot_v1_kde_by_rank(
    df: pd.DataFrame,
    label_col: str,
    n_layers: int,
    n_heads: int,
    top_k: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    For each eigenvalue rank k, pool all layer/head values across datasets,
    then plot KDE for correct vs hallucinated.
    Also compute Cohen's d for each rank.
    """
    label_names = {0: "correct", 1: "hallucinated"}
    stats_rows = []

    fig, axes = plt.subplots(top_k, 1, figsize=(10, 2.4 * top_k), sharex=False)
    if top_k == 1:
        axes = [axes]

    for rank in range(1, top_k + 1):
        cols = get_rank_columns(df, rank, n_layers, n_heads)
        if len(cols) == 0:
            raise ValueError(f"No columns found for rank {rank}. Check naming convention.")

        tmp = df[[label_col] + cols].copy()
        long_df = tmp.melt(
            id_vars=[label_col],
            value_vars=cols,
            var_name="feature",
            value_name="eigval"
        )
        long_df = long_df.dropna(subset=["eigval"])

        x0 = long_df.loc[long_df[label_col] == 0, "eigval"].values
        x1 = long_df.loc[long_df[label_col] == 1, "eigval"].values
        d = cohens_d(x0, x1)
        stats_rows.append({"rank": rank, "cohens_d": d})

        ax = axes[rank - 1]
        sns.kdeplot(
            data=long_df[long_df[label_col] == 0],
            x="eigval",
            fill=True,
            common_norm=False,
            alpha=0.35,
            linewidth=1.2,
            label=label_names[0] if rank == 1 else None,
            ax=ax,
        )
        sns.kdeplot(
            data=long_df[long_df[label_col] == 1],
            x="eigval",
            fill=True,
            common_norm=False,
            alpha=0.35,
            linewidth=1.2,
            label=label_names[1] if rank == 1 else None,
            ax=ax,
        )

        ax.set_title(f"Rank k={rank}   |   Cohen's d = {d:.3f}")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Laplacian eigenvalue")
    axes[0].legend()
    fig.suptitle("V1 — Eigenvalue Distribution by Label (pooled over datasets, layers, heads)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_path.with_suffix(".csv"), index=False)
    return stats_df


# ============================================================
# V2 — Mann–Whitney p-Value Heatmap (Layers × Heads)
# ============================================================

def get_layer_head_topk_columns(
    df: pd.DataFrame,
    layer: int,
    head: int,
    top_k: int,
) -> List[str]:
    """
    Return the columns for one (layer, head) block:
        eig_l{layer}_h{head}_k1, ..., eig_l{layer}_h{head}_k{top_k}
    """
    cols = [f"eig_l{layer}_h{head}_k{k}" for k in range(1, top_k + 1)]
    return get_existing_columns(df, cols)


def compute_v2_pvalue_matrix(
    df: pd.DataFrame,
    label_col: str,
    n_layers: int,
    n_heads: int,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (layer, head):
      1. Compute per-sample mean of the top-k eigenvalues in that block
      2. Compare hallucinated vs correct using two-sided Mann–Whitney U
      3. Store p-value
      4. Store significance indicator p < 0.05

    Returns:
      pval_df: shape (n_layers, n_heads)
      sig_df:  shape (n_layers, n_heads), boolean
    """
    pvals = np.full((n_layers, n_heads), np.nan, dtype=float)

    for layer in range(n_layers):
        for head in range(n_heads):
            cols = get_layer_head_topk_columns(df, layer, head, top_k)
            if len(cols) == 0:
                continue

            tmp = df[[label_col] + cols].copy()
            tmp["mean_topk"] = tmp[cols].mean(axis=1)
            tmp = tmp.dropna(subset=[label_col, "mean_topk"])

            x_hall = tmp.loc[tmp[label_col] == 1, "mean_topk"].values
            x_corr = tmp.loc[tmp[label_col] == 0, "mean_topk"].values

            if len(x_hall) == 0 or len(x_corr) == 0:
                continue

            try:
                # two-sided MWU test, exactly as specified in the document
                _, p = mannwhitneyu(x_hall, x_corr, alternative="two-sided")
            except ValueError:
                p = np.nan

            pvals[layer, head] = p

    pval_df = pd.DataFrame(
        pvals,
        index=[f"layer_{l}" for l in range(n_layers)],
        columns=[f"head_{h}" for h in range(n_heads)],
    )
    sig_df = pval_df < 0.05
    return pval_df, sig_df


def plot_v2_heatmap(
    df: pd.DataFrame,
    label_col: str,
    dataset_col: str | None,
    n_layers: int,
    n_heads: int,
    top_k: int,
    output_path: Path,
    title_suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Build the V2 heatmap:
      - value = p-value for each (layer, head)
      - contour at p = 0.05
      - summary metric = % heads with p < 0.05
    """
    pval_df, sig_df = compute_v2_pvalue_matrix(
        df=df,
        label_col=label_col,
        n_layers=n_layers,
        n_heads=n_heads,
        top_k=top_k,
    )

    prop_sig = float(np.nanmean(sig_df.values.astype(float)))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Use -log10(p) for display so smaller p-values stand out more clearly
    with np.errstate(divide="ignore"):
        display_mat = -np.log10(pval_df.astype(float).values)

    sns.heatmap(
        display_mat,
        ax=ax,
        cmap="viridis",
        cbar=True,
        xticklabels=[f"h{h}" for h in range(n_heads)],
        yticklabels=[f"l{l}" for l in range(n_layers)],
    )

    # Overlay black contour at p = 0.05
    p_numeric = pval_df.astype(float).values
    X, Y = np.meshgrid(np.arange(n_heads) + 0.5, np.arange(n_layers) + 0.5)
    ax.contour(
        X,
        Y,
        p_numeric,
        levels=[0.05],
        colors="black",
        linewidths=1.5,
    )

    ax.set_title(
        f"V2 — Mann–Whitney p-value heatmap{title_suffix}\n"
        f"% heads with p < 0.05 = {100 * prop_sig:.1f}%"
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    pval_df.to_csv(output_path.with_suffix(".csv"))
    sig_df.to_csv(output_path.with_name(output_path.stem + "_significant_mask.csv"))

    return pval_df, sig_df, prop_sig


def run_v2_all_datasets(
    df: pd.DataFrame,
    label_col: str,
    dataset_col: str,
    n_layers: int,
    n_heads: int,
    top_k: int,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Run V2:
      - once on pooled data
      - once separately for each dataset

    Saves one heatmap per run and returns a summary table with the
    proportion of significant heads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # pooled
    pooled_pval_df, pooled_sig_df, pooled_prop = plot_v2_heatmap(
        df=df,
        label_col=label_col,
        dataset_col=dataset_col,
        n_layers=n_layers,
        n_heads=n_heads,
        top_k=top_k,
        output_path=output_dir / "V2_heatmap_pooled.png",
        title_suffix=" (Pooled)",
    )
    rows.append({
        "dataset": "Pooled",
        "num_significant_heads": int(np.nansum(pooled_sig_df.values)),
        "total_heads": int(np.isfinite(pooled_pval_df.values).sum()),
        "prop_significant_heads": pooled_prop,
        "percent_significant_heads": 100 * pooled_prop,
    })

    # per dataset
    for ds in sorted(df[dataset_col].dropna().unique()):
        sub = df[df[dataset_col] == ds].copy()
        pval_df, sig_df, prop_sig = plot_v2_heatmap(
            df=sub,
            label_col=label_col,
            dataset_col=dataset_col,
            n_layers=n_layers,
            n_heads=n_heads,
            top_k=top_k,
            output_path=output_dir / f"V2_heatmap_{str(ds).replace('/', '_').replace(' ', '_')}.png",
            title_suffix=f" ({ds})",
        )

        rows.append({
            "dataset": ds,
            "num_significant_heads": int(np.nansum(sig_df.values)),
            "total_heads": int(np.isfinite(pval_df.values).sum()),
            "prop_significant_heads": prop_sig,
            "percent_significant_heads": 100 * prop_sig,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "V2_summary_significant_heads.csv", index=False)
    return summary_df


# ============================================================
# V3 — Layer-wise AUROC Profile
# ============================================================

def plot_v3_layerwise_auroc(
    df: pd.DataFrame,
    label_col: str,
    dataset_col: str,
    n_layers: int,
    n_heads: int,
    top_k: int,
    full_pca_dim: int,
    layer_pca_dim: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    For each dataset:
      - train/test split
      - full all-layer probe -> one AUROC (horizontal reference line)
      - per-layer probe      -> AUROC per layer
    """
    results = []

    fig, ax = plt.subplots(figsize=(11, 6))

    datasets = sorted(df[dataset_col].dropna().unique())
    full_cols = build_feature_column_names(n_layers, n_heads, top_k)
    full_cols = get_existing_columns(df, full_cols)
    if len(full_cols) == 0:
        raise ValueError("No raw eigenvalue columns found for full model.")

    for ds in datasets:
        sub = df[df[dataset_col] == ds].copy()
        sub = sub.dropna(subset=[label_col])

        X_full = sub[full_cols]
        y = sub[label_col].values

        if len(np.unique(y)) < 2:
            print(f"[V3] Skipping {ds}: only one class present.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        all_layer_model = make_probe_pipeline(min(full_pca_dim, X_train.shape[1], len(X_train)))
        y_score_full = fit_predict_scores(X_train, y_train, X_test, all_layer_model)
        auc_full = roc_auc_score(y_test, y_score_full)

        layer_aucs = []
        layer_idxs = []

        for layer in range(n_layers):
            layer_cols = get_layer_columns(sub, layer, n_heads, top_k)
            if len(layer_cols) == 0:
                continue

            X_layer = sub[layer_cols]
            Xl_train, Xl_test, yl_train, yl_test = train_test_split(
                X_layer, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )

            layer_dim = min(layer_pca_dim, Xl_train.shape[1], len(Xl_train))
            if layer_dim < 1:
                continue

            layer_model = make_probe_pipeline(layer_dim)
            y_score_layer = fit_predict_scores(Xl_train, yl_train, Xl_test, layer_model)
            auc_layer = roc_auc_score(yl_test, y_score_layer)

            results.append({
                "dataset": ds,
                "layer": layer,
                "layer_auc": auc_layer,
                "all_layer_auc": auc_full,
            })
            layer_idxs.append(layer)
            layer_aucs.append(auc_layer)

        if len(layer_idxs) > 0:
            ax.plot(layer_idxs, layer_aucs, marker="o", linewidth=1.8, label=f"{ds} per-layer")
            ax.axhline(auc_full, linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_title("V3 — Layer-wise AUROC Profile")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Test AUROC")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path.with_suffix(".csv"), index=False)
    return res_df

# ============================================================
# V4 — ROC Curve Comparison Across Datasets
# ============================================================

def plot_v4_roc_comparison(
    df: pd.DataFrame,
    label_col: str,
    dataset_col: str,
    baseline_col: str | None,
    n_layers: int,
    n_heads: int,
    top_k: int,
    full_pca_dim: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    For each dataset:
      - LapEigvals top-10 PCA(256)
      - AttentionScore baseline
      - LapEigvals top-5 ablation
    Plot ROC curves in a 2x3 grid (5 datasets + 1 legend panel).
    """
    datasets = sorted(df[dataset_col].dropna().unique())
    full_cols = build_feature_column_names(n_layers, n_heads, top_k)
    full_cols = get_existing_columns(df, full_cols)
    top5_cols = get_top5_columns_from_top10(full_cols)

    if len(full_cols) == 0:
        raise ValueError("No raw eigenvalue columns found for V4.")
    if len(top5_cols) == 0:
        raise ValueError("No top-5 subset columns found for V4.")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    summary_rows = []

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df[dataset_col] == ds].copy()
        y = sub[label_col].values

        if len(np.unique(y)) < 2:
            ax.set_title(f"{ds} (skipped: one class)")
            ax.axis("off")
            continue

        # Common train/test split indices
        idx = np.arange(len(sub))
        idx_train, idx_test, y_train, y_test = train_test_split(
            idx, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # Method 1: LapEigvals top-10
        X = sub[full_cols]
        X_train = X.iloc[idx_train]
        X_test = X.iloc[idx_test]
        model_full = make_probe_pipeline(min(full_pca_dim, X_train.shape[1], len(X_train)))
        s_full = fit_predict_scores(X_train, y_train, X_test, model_full)
        auc_full = roc_auc_score(y_test, s_full)
        fpr, tpr, _ = roc_curve(y_test, s_full)
        ax.plot(fpr, tpr, linewidth=2, label=f"LapEigvals top-10 (AUC={auc_full:.3f})")
        fpr_grid, low, high = bootstrap_roc_band(y_test, s_full, n_bootstrap=N_BOOTSTRAP_ROC, random_state=RANDOM_STATE)
        ax.fill_between(fpr_grid, low, high, alpha=0.18)

        # Method 2: AttentionScore baseline
        if baseline_col is not None and baseline_col in sub.columns:
            s_base = sub.iloc[idx_test][baseline_col].values.astype(float)
            auc_base = roc_auc_score(y_test, s_base)
            fpr_b, tpr_b, _ = roc_curve(y_test, s_base)
            ax.plot(fpr_b, tpr_b, linewidth=2, label=f"AttentionScore (AUC={auc_base:.3f})")
            fpr_grid_b, low_b, high_b = bootstrap_roc_band(
                y_test, s_base, n_bootstrap=N_BOOTSTRAP_ROC, random_state=RANDOM_STATE + 1
            )
            ax.fill_between(fpr_grid_b, low_b, high_b, alpha=0.18)
        else:
            auc_base = np.nan

        # Method 3: LapEigvals top-5 ablation
        X5 = sub[top5_cols]
        X5_train = X5.iloc[idx_train]
        X5_test = X5.iloc[idx_test]
        model_top5 = make_probe_pipeline(min(full_pca_dim, X5_train.shape[1], len(X5_train)))
        s_top5 = fit_predict_scores(X5_train, y_train, X5_test, model_top5)
        auc_top5 = roc_auc_score(y_test, s_top5)
        fpr5, tpr5, _ = roc_curve(y_test, s_top5)
        ax.plot(fpr5, tpr5, linewidth=2, label=f"LapEigvals top-5 (AUC={auc_top5:.3f})")
        fpr_grid_5, low_5, high_5 = bootstrap_roc_band(
            y_test, s_top5, n_bootstrap=N_BOOTSTRAP_ROC, random_state=RANDOM_STATE + 2
        )
        ax.fill_between(fpr_grid_5, low_5, high_5, alpha=0.18)

        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(ds)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.25)

        summary_rows.append({
            "dataset": ds,
            "auc_lapeig_top10": auc_full,
            "auc_attentionscore": auc_base,
            "auc_lapeig_top5": auc_top5,
        })

    # Legend panel
    if len(datasets) < 6:
        axes[3].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[3].legend(handles, labels, loc="center", frameon=False, fontsize=11)

    fig.suptitle("V4 — ROC Curve Comparison Across Datasets", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path.with_suffix(".csv"), index=False)
    return summary_df

# ============================================================
# V5 — Dataset Cross-Generalisation Heatmap (LapEigvals only)
# ============================================================

def cross_dataset_auc_matrix(
    df: pd.DataFrame,
    dataset_col: str,
    label_col: str,
    feature_cols: Sequence[str],
    model_builder,
) -> pd.DataFrame:
    """
    Train on each dataset separately, test on every dataset.
    Return AUROC matrix.
    """
    datasets = sorted(df[dataset_col].dropna().unique())
    mat = pd.DataFrame(index=datasets, columns=datasets, dtype=float)

    for train_ds in datasets:
        train_df = df[df[dataset_col] == train_ds].copy()
        y_train = train_df[label_col].values

        if len(np.unique(y_train)) < 2:
            print(f"[V5] Skipping train dataset {train_ds}: one class only.")
            continue

        X_train = train_df[list(feature_cols)]
        model = model_builder(X_train)
        model.fit(X_train, y_train)

        for test_ds in datasets:
            test_df = df[df[dataset_col] == test_ds].copy()
            y_test = test_df[label_col].values

            if len(np.unique(y_test)) < 2:
                mat.loc[train_ds, test_ds] = np.nan
                continue

            X_test = test_df[list(feature_cols)]
            s_test = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, s_test)
            mat.loc[train_ds, test_ds] = auc

    return mat


def plot_v5_cross_generalisation(
    df: pd.DataFrame,
    dataset_col: str,
    label_col: str,
    n_layers: int,
    n_heads: int,
    top_k: int,
    full_pca_dim: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    Single AUROC heatmap:
      - LapEigvals only
    """
    full_cols = build_feature_column_names(n_layers, n_heads, top_k)
    full_cols = get_existing_columns(df, full_cols)
    if len(full_cols) == 0:
        raise ValueError("No raw eigenvalue columns found for V5.")

    def lapeig_model_builder(X_train):
        dim = min(full_pca_dim, X_train.shape[1], len(X_train))
        return make_probe_pipeline(dim)

    lapeig_mat = cross_dataset_auc_matrix(
        df=df,
        dataset_col=dataset_col,
        label_col=label_col,
        feature_cols=full_cols,
        model_builder=lapeig_model_builder,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        lapeig_mat.astype(float),
        annot=True,
        fmt=".3f",
        cmap="viridis",
        ax=ax,
        cbar=True
    )
    ax.set_title("V5 — Cross-Generalisation (LapEigvals)")
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Train dataset")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    lapeig_mat.to_csv(output_path.with_suffix(".csv"))
    return lapeig_mat

# ============================================================
# Statistical Validation via Bootstrap
# ============================================================

def get_lambda_max_columns(df: pd.DataFrame, n_layers: int, n_heads: int) -> List[str]:
    """
    Return the columns corresponding to the largest eigenvalue (k=1)
    for every layer/head pair.
    """
    cols = [f"eig_l{l}_h{h}_k1" for l in range(n_layers) for h in range(n_heads)]
    cols = get_existing_columns(df, cols)

    if len(cols) == 0:
        raise ValueError(
            "No k=1 eigenvalue columns found. "
            "Expected columns like eig_l{layer}_h{head}_k1."
        )
    return cols


def add_lambda_max_mean_column(
    df: pd.DataFrame,
    n_layers: int,
    n_heads: int,
    out_col: str = "lambda_max_mean",
) -> pd.DataFrame:
    """
    For each sample, compute the average of the largest eigenvalue (k=1)
    over all heads and layers.

    This corresponds to the document's:
      'largest Laplacian eigenvalue λ_max (averaged across all heads and layers)'
    """
    df = df.copy()
    lambda_max_cols = get_lambda_max_columns(df, n_layers, n_heads)
    df[out_col] = df[lambda_max_cols].mean(axis=1)
    return df


def observed_delta(values: np.ndarray, labels: np.ndarray) -> float:
    """
    Observed mean difference:
        delta = mean(values | y=1) - mean(values | y=0)
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)

    vals_1 = values[labels == 1]
    vals_0 = values[labels == 0]

    if len(vals_1) == 0 or len(vals_0) == 0:
        return np.nan

    return float(np.mean(vals_1) - np.mean(vals_0))


def paired_bootstrap_test_lambda_max(
    values: np.ndarray,
    labels: np.ndarray,
    B: int = 10_000,
    random_state: int = 137,
) -> Dict[str, float | np.ndarray]:
    """
    Paired bootstrap test on lambda_max_mean.

    Algorithm from the document:
      1. delta_obs = mean(lambda_max | y=1) - mean(lambda_max | y=0)
      2. For b = 1..B:
            sample rows with replacement
            delta_b = same statistic on bootstrap sample
      3. One-sided p-value = proportion of bootstrap deltas <= 0
      4. 95% CI = percentile interval [2.5%, 97.5%]

    Returns:
      {
        "delta_obs": ...,
        "ci_low": ...,
        "ci_high": ...,
        "p_value": ...,
        "boot_deltas": np.ndarray
      }
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)

    valid = np.isfinite(values) & np.isfinite(labels)
    values = values[valid]
    labels = labels[valid]

    if len(values) == 0:
        raise ValueError("No valid rows for bootstrap.")

    if len(np.unique(labels)) < 2:
        raise ValueError("Bootstrap requires both classes to be present.")

    n = len(values)
    rng = np.random.default_rng(random_state)

    delta_obs = observed_delta(values, labels)

    boot_deltas = np.empty(B, dtype=float)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        vb = values[idx]
        yb = labels[idx]

        # If a bootstrap resample accidentally contains one class only,
        # redraw once until both classes are present.
        # This is rare when n is large, but keeps the statistic well-defined.
        tries = 0
        while len(np.unique(yb)) < 2 and tries < 20:
            idx = rng.integers(0, n, size=n)
            vb = values[idx]
            yb = labels[idx]
            tries += 1

        if len(np.unique(yb)) < 2:
            boot_deltas[b] = np.nan
        else:
            boot_deltas[b] = observed_delta(vb, yb)

    boot_deltas = boot_deltas[np.isfinite(boot_deltas)]

    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))

    # one-sided test for H1: delta > 0
    p_value = float(np.mean(boot_deltas <= 0))

    return {
        "delta_obs": float(delta_obs),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "boot_deltas": boot_deltas,
    }


def summarise_bootstrap_results(
    df: pd.DataFrame,
    label_col: str,
    dataset_col: str,
    lambda_col: str = "lambda_max_mean",
    B: int = 10_000,
    random_state: int = 137,
) -> pd.DataFrame:
    """
    Run the bootstrap:
      - once on the pooled data
      - separately for each dataset

    Returns a summary table in the format requested by the document.
    """
    rows = []

    # pooled
    pooled = df.dropna(subset=[label_col, lambda_col]).copy()
    if len(np.unique(pooled[label_col])) >= 2:
        res = paired_bootstrap_test_lambda_max(
            pooled[lambda_col].values,
            pooled[label_col].values,
            B=B,
            random_state=random_state,
        )
        rows.append({
            "dataset": "Pooled",
            "delta_obs": res["delta_obs"],
            "ci_low_95": res["ci_low"],
            "ci_high_95": res["ci_high"],
            "p_value_one_sided": res["p_value"],
            "n_samples": len(pooled),
            "n_hallucinated": int((pooled[label_col] == 1).sum()),
            "n_correct": int((pooled[label_col] == 0).sum()),
        })

    # per dataset
    for i, ds in enumerate(sorted(df[dataset_col].dropna().unique())):
        sub = df[df[dataset_col] == ds].dropna(subset=[label_col, lambda_col]).copy()
        if len(sub) == 0:
            continue
        if len(np.unique(sub[label_col])) < 2:
            rows.append({
                "dataset": ds,
                "delta_obs": np.nan,
                "ci_low_95": np.nan,
                "ci_high_95": np.nan,
                "p_value_one_sided": np.nan,
                "n_samples": len(sub),
                "n_hallucinated": int((sub[label_col] == 1).sum()),
                "n_correct": int((sub[label_col] == 0).sum()),
            })
            continue

        res = paired_bootstrap_test_lambda_max(
            sub[lambda_col].values,
            sub[label_col].values,
            B=B,
            random_state=random_state + i + 1,
        )

        rows.append({
            "dataset": ds,
            "delta_obs": res["delta_obs"],
            "ci_low_95": res["ci_low"],
            "ci_high_95": res["ci_high"],
            "p_value_one_sided": res["p_value"],
            "n_samples": len(sub),
            "n_hallucinated": int((sub[label_col] == 1).sum()),
            "n_correct": int((sub[label_col] == 0).sum()),
        })

    return pd.DataFrame(rows)


def plot_bootstrap_distribution(
    boot_deltas: np.ndarray,
    delta_obs: float,
    p_value: float,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot histogram of bootstrap deltas with:
      - vertical line at 0
      - vertical line at observed delta
      - shaded 95% percentile CI
      - text box showing delta_obs, 95% CI, and one-sided p-value
    """
    ci_low = np.percentile(boot_deltas, 2.5)
    ci_high = np.percentile(boot_deltas, 97.5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(boot_deltas, bins=60, alpha=0.8, density=True)
    ax.axvline(0.0, linestyle="--", linewidth=1.8, label="0")
    ax.axvline(delta_obs, linestyle="-", linewidth=2.2, label=f"delta_obs = {delta_obs:.4f}")
    ax.axvspan(ci_low, ci_high, alpha=0.18, label=f"95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    stats_text = (
        f"$\\delta_{{obs}}$ = {delta_obs:.4f}\n"
        f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]\n"
        f"one-sided p = {p_value:.4g}"
    )

    ax.text(
        0.98, 0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", alpha=0.9)
    )

    ax.set_title(title)
    ax.set_xlabel("Bootstrap delta")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_bootstrap_validation(
    df: pd.DataFrame,
    label_col: str,
    dataset_col: str,
    n_layers: int,
    n_heads: int,
    output_dir: Path,
    B: int = 10_000,
    random_state: int = 137,
) -> pd.DataFrame:
    """
    Full bootstrap workflow:
      1. add lambda_max_mean per sample
      2. pooled bootstrap
      3. per-dataset bootstrap
      4. save summary table
      5. save pooled + per-dataset bootstrap histograms
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df2 = add_lambda_max_mean_column(df, n_layers=n_layers, n_heads=n_heads, out_col="lambda_max_mean")

    # summary table
    summary_df = summarise_bootstrap_results(
        df=df2,
        label_col=label_col,
        dataset_col=dataset_col,
        lambda_col="lambda_max_mean",
        B=B,
        random_state=random_state,
    )
    summary_df.to_csv(output_dir / "bootstrap_summary_table.csv", index=False)

    # pooled plot
    pooled = df2.dropna(subset=[label_col, "lambda_max_mean"]).copy()
    pooled_res = paired_bootstrap_test_lambda_max(
        pooled["lambda_max_mean"].values,
        pooled[label_col].values,
        B=B,
        random_state=random_state,
    )
    plot_bootstrap_distribution(
        boot_deltas=pooled_res["boot_deltas"],
        delta_obs=pooled_res["delta_obs"],
        p_value=pooled_res["p_value"],
        title="Bootstrap Test on mean lambda_max (Pooled)",
        output_path=output_dir / "bootstrap_pooled_hist.png",
    )   

    # per-dataset plots
    for i, ds in enumerate(sorted(df2[dataset_col].dropna().unique())):
        sub = df2[df2[dataset_col] == ds].dropna(subset=[label_col, "lambda_max_mean"]).copy()
        if len(sub) == 0 or len(np.unique(sub[label_col])) < 2:
            continue

        res = paired_bootstrap_test_lambda_max(
            sub["lambda_max_mean"].values,
            sub[label_col].values,
            B=B,
            random_state=random_state + i + 1,
        )

        safe_name = str(ds).replace("/", "_").replace(" ", "_")
        plot_bootstrap_distribution(
            boot_deltas=res["boot_deltas"],
            delta_obs=res["delta_obs"],
            p_value=res["p_value"],
            title=f"Bootstrap Test on mean lambda_max ({ds})",
            output_path=output_dir / f"bootstrap_{safe_name}_hist.png",
            )

    return summary_df

# ============================================================
# V5 — Dataset Cross-Generalisation Heatmap
# ============================================================

# def cross_dataset_auc_matrix(
#     df: pd.DataFrame,
#     dataset_col: str,
#     label_col: str,
#     feature_cols: Sequence[str],
#     model_builder,
# ) -> pd.DataFrame:
#     """
#     Train on each dataset separately, test on every dataset.
#     Return AUROC matrix.
#     """
#     datasets = sorted(df[dataset_col].dropna().unique())
#     mat = pd.DataFrame(index=datasets, columns=datasets, dtype=float)

#     for train_ds in datasets:
#         train_df = df[df[dataset_col] == train_ds].copy()
#         y_train = train_df[label_col].values

#         if len(np.unique(y_train)) < 2:
#             print(f"[V5] Skipping train dataset {train_ds}: one class only.")
#             continue

#         X_train = train_df[list(feature_cols)]
#         model = model_builder(X_train)

#         model.fit(X_train, y_train)

#         for test_ds in datasets:
#             test_df = df[df[dataset_col] == test_ds].copy()
#             y_test = test_df[label_col].values

#             if len(np.unique(y_test)) < 2:
#                 mat.loc[train_ds, test_ds] = np.nan
#                 continue

#             X_test = test_df[list(feature_cols)]
#             s_test = model.predict_proba(X_test)[:, 1]
#             auc = roc_auc_score(y_test, s_test)
#             mat.loc[train_ds, test_ds] = auc

#     return mat


# def plot_v5_cross_generalisation(
#     df: pd.DataFrame,
#     dataset_col: str,
#     label_col: str,
#     baseline_col: str | None,
#     n_layers: int,
#     n_heads: int,
#     top_k: int,
#     full_pca_dim: int,
#     output_path: Path,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Side-by-side AUROC matrices:
#       - LapEigvals
#       - AttentionScore
#     """
#     full_cols = build_feature_column_names(n_layers, n_heads, top_k)
#     full_cols = get_existing_columns(df, full_cols)
#     if len(full_cols) == 0:
#         raise ValueError("No raw eigenvalue columns found for V5.")

#     def lapeig_model_builder(X_train):
#         dim = min(full_pca_dim, X_train.shape[1], len(X_train))
#         return make_probe_pipeline(dim)

#     lapeig_mat = cross_dataset_auc_matrix(
#         df=df,
#         dataset_col=dataset_col,
#         label_col=label_col,
#         feature_cols=full_cols,
#         model_builder=lapeig_model_builder,
#     )

#     if baseline_col is not None and baseline_col in df.columns:
#         datasets = sorted(df[dataset_col].dropna().unique())
#         base_mat = pd.DataFrame(index=datasets, columns=datasets, dtype=float)

#         # "Train on each dataset separately" is not meaningful for a fixed unsupervised score.
#         # We therefore evaluate the same baseline score on each test dataset and copy the value
#         # across each training row so the matrix is comparable in layout.
#         for test_ds in datasets:
#             test_df = df[df[dataset_col] == test_ds].copy()
#             y_test = test_df[label_col].values
#             if len(np.unique(y_test)) < 2:
#                 auc = np.nan
#             else:
#                 s = test_df[baseline_col].values.astype(float)
#                 auc = roc_auc_score(y_test, s)
#             for train_ds in datasets:
#                 base_mat.loc[train_ds, test_ds] = auc
#     else:
#         base_mat = pd.DataFrame(index=lapeig_mat.index, columns=lapeig_mat.columns, dtype=float)

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#     sns.heatmap(
#         lapeig_mat.astype(float),
#         annot=True,
#         fmt=".3f",
#         cmap="viridis",
#         ax=axes[0],
#         cbar=True
#     )
#     axes[0].set_title("V5 — Cross-Generalisation (LapEigvals)")
#     axes[0].set_xlabel("Test dataset")
#     axes[0].set_ylabel("Train dataset")

#     sns.heatmap(
#         base_mat.astype(float),
#         annot=True,
#         fmt=".3f",
#         cmap="viridis",
#         ax=axes[1],
#         cbar=True
#     )
#     axes[1].set_title("V5 — Cross-Generalisation (AttentionScore)")
#     axes[1].set_xlabel("Test dataset")
#     axes[1].set_ylabel("Train dataset")

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     lapeig_mat.to_csv(output_path.with_name(output_path.stem + "_lapeig.csv"))
#     base_mat.to_csv(output_path.with_name(output_path.stem + "_attentionscore.csv"))

#     return lapeig_mat, base_mat