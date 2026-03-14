import pandas as pd

# Valid categories 
VALID_DOMAINS = {
    "STEM",
    "Humanities",
    "Social Sciences",
    "Medicine & Health",
    "Law, Business, and Miscellaneous"
}

def compute_majority_valid_domain(domain_series):
    valid_subset = domain_series[domain_series.isin(VALID_DOMAINS)]
    
    if not valid_subset.empty:
        return valid_subset.mode().iloc[0]
    else:
        return pd.NA




"""
spectral_detection/analysis/eda.py

EDA printing / diagnostic helpers.

These functions produce text output (summaries, tables) rather than plots.
For visualisations see spectral_detection/visualization.py.

Methods sourced from hallucination_utils.py (Debanjan Bhattacharya).
"""

import numpy as np
import pandas as pd

from spectral_detection.data.cleaning import LABEL_ORDER


# ── Printing helpers ───────────────────────────────────────────────────────────

def print_loading_summary(df, correctness_col="correctness"):
    """
    Quick summary: shape, question count, samples/question, correctness,
    domain and type listings.
    Method from hallucination_utils.py.
    """
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
    """
    Show domain inconsistency statistics.
    Method from hallucination_utils.py.
    """
    n_inc = q_meta["domain_inconsistent"].sum()
    n_tot = len(q_meta)
    print(f"Domain inconsistency: {n_inc}/{n_tot} questions "
          f"({n_inc / n_tot * 100:.1f}%)")
    print(f"Max unique domains per question: {q_meta['n_unique_domains'].max()}")


def print_filtering_diagnostic(feat_df, raw_domain_counts, skipped,
                                min_questions, domain_col="domain",
                                skipped_details=None):
    """
    Print what was dropped, surviving counts, and threshold flags.
    Returns (analysis_domains, excluded_domains).
    Method from hallucination_utils.py.
    """
    from spectral_detection.data.cleaning import split_analysis_domains

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


def print_results_summary(df, feat_df, skipped, analysis_domains,
                           excluded_domains, delta_obs, perm_pval,
                           auc_boot_mean, ci_lo, ci_hi,
                           df_clf, min_questions,
                           label_col="label", domain_col="domain"):
    """
    End-of-notebook results summary.
    """
    ds_name = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
    print("=" * 70)
    print(f"{ds_name.upper()} HALLUCINATION DETECTION — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Dataset:                 {ds_name}")
    print(f"  Total questions (raw):   {df['prompt_id'].nunique()}")
    print(f"  Questions analyzed:      {len(feat_df)}")
    print(f"  Questions skipped:       {skipped}  (all responses refused)")
    print(f"  Samples per question:    20")
    print(f"  Domains in ML (>= {min_questions}):   {len(analysis_domains)}")
    if excluded_domains:
        print(f"  Excluded domains:        {excluded_domains}")
    print()

    vc = df["correctness"].value_counts()
    print("  Response-level breakdown:")
    for lab in LABEL_ORDER:
        if lab in vc.index:
            pct = vc[lab] / len(df) * 100
            print(f"    {lab:12s}: {vc[lab]:6d}  ({pct:.1f}%)")
    print()
    print("  NOTE: 'refused' responses are merged with 'incorrect' (hallucinated) "
          "for all question-level binary labels.")
    print()

    n0 = (feat_df[label_col] == 0).sum()
    n1 = (feat_df[label_col] == 1).sum()
    print(f"  Binary label (refused → hallucinated):")
    print(f"    Correct: {n0},  Hallucinated (incl. refused): {n1}")
    print(f"    Hallucination rate: {feat_df[label_col].mean() * 100:.1f}%")
    print()
    print(f"  Permutation test (entropy difference):")
    print(f"    Δ = {delta_obs:.4f} bits,  p = {perm_pval:.6f}")
    print()
    print(f"  Bootstrap AUC (RF, 5 geometric features):")
    print(f"    AUC = {auc_boot_mean:.4f},  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print()
    if df_clf is not None:
        best = df_clf.loc[df_clf["AUC_mean"].idxmax()]
        print(f"  Best classifier config:")
        print(f"    {best['Classifier']} / {best['Variant']}  AUC={best['AUC_mean']:.4f}")
    print("=" * 70)