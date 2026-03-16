"""
spectral_detection/data/cleaning.py

Data loading, preprocessing, and domain-consolidation utilities.

Supports all five datasets: DefAn, HaluEval, MMLU, TriviaQA, TruthfulQA.

Universal columns shared by all datasets:
    id, question, reference_answer, model_answer, correctness,
    correctness_score, domain, dataset, adversarial, type

All datasets use three correctness labels: correct, incorrect, refused.

NOTE — Domain validity warning
-------------------------------
Domains are assigned per-response by the LLM judge and are frequently
inconsistent across the 20 responses for a single question (14–36% of
questions, depending on dataset). Datasets other than DefAn have severe
domain sprawl (30–90 unique strings). The domain_mode column below uses
majority-vote per question, but this is still a noisy proxy. See
analysis/eda.py for diagnostics.  All domain-level results should be
interpreted cautiously — they are exploratory rather than confirmatory.

Methods sourced from hallucination_utils.py (Debanjan Bhattacharya).
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Label constants ────────────────────────────────────────────────────────────

LABEL_ORDER   = ["correct", "incorrect", "refused"]
LABEL_COLORS  = {"correct": "#2196F3", "incorrect": "#E53935", "refused": "#FFA726"}
GEO_FEATURES  = ["H_sem", "D_cos", "D_cos_var", "D_pair",
                  "M_bar", "K", "sig2_S"]
FEAT_NICE_NAMES = {
    "H_sem":    "Semantic Entropy",
    "D_cos":    "Cosine Dispersion (mean centroid)",
    "D_cos_var":"Cosine Dispersion (variance centroid)",
    "D_pair":   "Mean Pairwise Cosine Distance",
    "M_bar":    "Mahalanobis Distance (mean)",
    "K":        "Cluster Count",
    "sig2_S":   "Similarity Variance",
}

# ── Canonical domain patterns ──────────────────────────────────────────────────
# Method from hallucination_utils.py

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


# ── Loading ────────────────────────────────────────────────────────────────────

def load_dataset(path, required_cols=None):
    """
    Load a JSONL dataset; ensure prompt_id and answer_len columns exist.
    Method from hallucination_utils.py.

    Parameters
    ----------
    path : str
        Path to .jsonl file.
    required_cols : list, optional
        Columns that must be present; raises ValueError if any are missing.

    Returns
    -------
    pd.DataFrame  — response-level dataframe, one row per model answer.
    """
    df = pd.read_json(path, lines=True)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # DefAn has no prompt_id; derive from id
    if "prompt_id" not in df.columns:
        df["prompt_id"] = df["id"].astype(str).str.rsplit("_", n=1).str[0]

    if "answer_len" not in df.columns:
        df["answer_len"] = df["model_answer"].astype(str).str.len()

    return df


def load_all_datasets(data_dir):
    """
    Load every *.jsonl in data_dir.
    Returns dict keyed by dataset name (e.g. 'defan', 'mmlu', ...).
    Method from hallucination_utils.py.
    """
    datasets = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        df = load_dataset(os.path.join(data_dir, fname))
        name = df["dataset"].iloc[0]
        datasets[name] = df
    return datasets


# ── Domain helpers ─────────────────────────────────────────────────────────────

def _mode(series):
    """Statistical mode; first value on ties. Method from hallucination_utils.py."""
    m = series.mode()
    return m.iloc[0] if len(m) else series.iloc[0]


def consolidate_domain(raw_domain):
    """
    Map a raw (potentially noisy) domain string to a canonical category.
    Returns "Other" if no pattern matches.
    Method from hallucination_utils.py.
    """
    for canonical, pattern in _CANONICAL_PATTERNS:
        if re.search(pattern, raw_domain):
            return canonical
    return "Other"


def add_canonical_domain(df, col="domain"):
    """
    Add a domain_canonical column via consolidate_domain.
    Method from hallucination_utils.py.
    """
    df = df.copy()
    df["domain_canonical"] = df[col].apply(consolidate_domain)
    return df


# ── Question-level metadata ────────────────────────────────────────────────────

def compute_question_metadata(df, domain_col="domain"):
    """
    Aggregate per-question metadata from a response-level dataframe.
    Method from hallucination_utils.py.

    Returns DataFrame indexed by prompt_id with columns:
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
    if "type" in df.columns and df["type"].nunique() > 1:
        agg["type"] = ("type", "first")

    q_meta = df.groupby("prompt_id").agg(**agg)
    q_meta["domain_inconsistent"] = q_meta["n_unique_domains"] > 1
    return q_meta


def questions_per_domain(df, domain_col="domain"):
    """
    Unique question counts per domain, descending.
    Method from hallucination_utils.py.
    """
    return (
        df.groupby(domain_col)["prompt_id"]
        .nunique()
        .sort_values(ascending=False)
    )


def split_analysis_domains(feat_df, min_questions, domain_col="domain"):
    """
    Return (analysis_domains, excluded_domains) based on minimum question count.
    Method from hallucination_utils.py.
    """
    counts = feat_df[domain_col].value_counts()
    analysis = sorted(counts[counts >= min_questions].index.tolist())
    excluded  = sorted(set(counts.index) - set(analysis))
    return analysis, excluded
