"""
spectral_detection/data/cleaning.py

Data loading, preprocessing, and domain consolidation for five QA benchmarks:
DefAn, HaluEval, MMLU, TriviaQA, TruthfulQA.

All datasets share: id, question, reference_answer, model_answer, correctness,
correctness_score, domain, dataset, adversarial, type.

Correctness labels: correct | incorrect | refused.

NOTE: Domain labels are LLM-generated and noisy (14-36% inconsistency across
responses for a single question). All domain-level results are exploratory.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

LABEL_ORDER  = ["correct", "incorrect", "refused"]
LABEL_COLORS = {"correct": "#2196F3", "incorrect": "#E53935", "refused": "#FFA726"}

FEATURES = [
    "H_sem", "D_cos", "D_cos_var", "D_pair", "K", "sig2_S",
]

FEATURE_LABELS = {
    "H_sem":     "Semantic Entropy",
    "D_cos":     "Cosine Dispersion (mean centroid)",
    "D_cos_var": "Cosine Dispersion (variance centroid)",
    "D_pair":    "Mean Pairwise Cosine Distance",
    "K":         "Cluster Count",
    "sig2_S":    "Similarity Variance",
}

# Backward-compatible aliases
GEO_FEATURES   = FEATURES
FEAT_NICE_NAMES = FEATURE_LABELS

# ── Canonical domain patterns ──────────────────────────────────────────────────

_DOMAIN_PATTERNS = [
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
    """Load a JSONL dataset; ensure prompt_id and answer_len exist."""
    df = pd.read_json(path, lines=True)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    if "prompt_id" not in df.columns:
        df["prompt_id"] = df["id"].astype(str).str.rsplit("_", n=1).str[0]
    if "answer_len" not in df.columns:
        df["answer_len"] = df["model_answer"].astype(str).str.len()
    return df


def load_all_datasets(data_dir):
    """Load every *.jsonl in data_dir. Returns dict keyed by dataset name."""
    datasets = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        df = load_dataset(os.path.join(data_dir, fname))
        datasets[df["dataset"].iloc[0]] = df
    return datasets


# ── Domain helpers ─────────────────────────────────────────────────────────────

def _mode(series):
    """Statistical mode; first value on ties."""
    m = series.mode()
    return m.iloc[0] if len(m) else series.iloc[0]


def consolidate_domain(raw_domain):
    """Map a noisy domain string to a canonical category. Falls back to 'Other'."""
    for canonical, pattern in _DOMAIN_PATTERNS:
        if re.search(pattern, raw_domain):
            return canonical
    return "Other"


def add_canonical_domain(df, col="domain"):
    """Add domain_canonical column via regex consolidation."""
    df = df.copy()
    df["domain_canonical"] = df[col].apply(consolidate_domain)
    return df


# ── Question-level metadata ───────────────────────────────────────────────────

def compute_question_metadata(df, domain_col="domain"):
    """Aggregate per-question metadata (domain mode, inconsistency flag, etc.)."""
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
    """Unique question counts per domain, descending."""
    return (
        df.groupby(domain_col)["prompt_id"]
        .nunique()
        .sort_values(ascending=False)
    )


def split_analysis_domains(feat_df, min_questions, domain_col="domain"):
    """Split domains into analysis (>= min_questions) and excluded sets."""
    counts = feat_df[domain_col].value_counts()
    analysis = sorted(counts[counts >= min_questions].index.tolist())
    excluded = sorted(set(counts.index) - set(analysis))
    return analysis, excluded
