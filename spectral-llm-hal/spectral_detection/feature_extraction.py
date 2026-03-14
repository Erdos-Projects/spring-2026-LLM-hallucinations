"""
spectral_detection/feature_extraction.py

Embedding and geometric-feature extraction for the hallucination-detection
pipeline (Stage 3 of the workflow).

Five features are computed per question from the N=20 response embeddings:
  H_sem   — Semantic entropy     [Farquhar et al., 2024]
  D_cos   — Cosine dispersion    [Ricco et al., 2025]
  M_bar   — Mahalanobis distance [Lee et al., 2018]
  K       — Cluster count
  sig2_S  — Similarity variance

⚠️  REFUSAL MERGE POLICY (key change vs. Debanjan's notebooks)
    Refused responses are treated as hallucinations for question-level binary
    labelling.  Rationale: a refusal means the model failed to produce a
    factually grounded answer, which is functionally equivalent to a
    hallucination from a downstream-reliability perspective.
    Specifically:
        label = 1  if  (n_incorrect + n_refused) / n_total > 0.5  else  0
    The raw counts (n_correct, n_incorrect, n_refused) are still stored in the
    feature DataFrame for traceability.

Extra functions (pairwise_similarities, semantic_entropy variant) are from a
separate notebook contributed by the project owner — merged here.

Methods sourced from hallucination_utils.py (Debanjan Bhattacharya).
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from spectral_detection.data.cleaning import (
    GEO_FEATURES, compute_question_metadata,
)


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_responses(df, model_name="all-MiniLM-L6-v2",
                    text_col="model_answer", cache_path=None,
                    batch_size=256, normalize_embs=True):
    """
    Embed all responses using a SentenceTransformer model.
    Loads from cache_path if the file exists; otherwise computes and saves.

    Parameters
    ----------
    df : pd.DataFrame   Response-level dataframe.
    model_name : str    SentenceTransformer identifier.
    text_col : str      Column containing text to embed.
    cache_path : str    Optional .npy cache file path.
    batch_size : int
    normalize_embs : bool  L2-normalise embeddings (recommended for cosine).

    Returns
    -------
    np.ndarray  shape (len(df), embedding_dim)

    Method from hallucination_utils.py.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        embs = np.load(cache_path)
        assert embs.shape[0] == len(df), (
            f"Cache size mismatch: {embs.shape[0]} vs {len(df)} rows"
        )
        return embs

    from sentence_transformers import SentenceTransformer

    print(f"Computing embeddings with {model_name} ...")
    embedder = SentenceTransformer(model_name)
    texts = df[text_col].astype(str).tolist()
    embs = embedder.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=normalize_embs,
    )
    if cache_path:
        np.save(cache_path, embs)
        print(f"Saved embeddings to {cache_path}")
    print(f"Embedding matrix: {embs.shape}")
    return embs


# ── Geometric feature functions ────────────────────────────────────────────────

def semantic_entropy(embs, threshold=0.85):
    """
    Compute semantic entropy H and cluster count K from response embeddings.

    Agglomerative clustering groups responses whose cosine similarity exceeds
    threshold; entropy is computed over the resulting cluster-size distribution.

    Parameters
    ----------
    embs : np.ndarray   shape (n, d) — all N embeddings for one question.
    threshold : float   Cosine-similarity merge threshold.

    Returns
    -------
    H : float   Shannon entropy in bits.
    K : int     Number of semantic clusters.

    Method from hallucination_utils.py.
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
        # older sklearn: affinity instead of metric
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
    """
    Mean cosine distance from each embedding to the centroid.
    Method from hallucination_utils.py.

    Parameters
    ----------
    embs : np.ndarray   shape (n, d).

    Returns
    -------
    float
    """
    centroid = embs.mean(axis=0, keepdims=True)
    return float(np.mean(1.0 - cosine_similarity(embs, centroid).flatten()))


def mahalanobis_distance(embs, mu, cov_inv):
    """
    Mean Mahalanobis distance of response embeddings from a reference.
    Method from hallucination_utils.py.

    Parameters
    ----------
    embs : np.ndarray   shape (n, d).
    mu : np.ndarray     Reference mean, shape (d,).
    cov_inv : np.ndarray  Precision matrix, shape (d, d).

    Returns
    -------
    float
    """
    diffs = embs - mu
    mahal_sq = np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)
    mahal_sq = np.clip(mahal_sq, 0, None)
    return float(np.mean(np.sqrt(mahal_sq)))


def similarity_variance(embs):
    """
    Variance of pairwise cosine similarities among response embeddings.
    Method from hallucination_utils.py.

    Parameters
    ----------
    embs : np.ndarray   shape (n, d).

    Returns
    -------
    float
    """
    sim = cosine_similarity(embs)
    upper = sim[np.triu_indices(len(embs), k=1)]
    return float(np.var(upper))


# ── Additional geometric functions (from project-owner's separate notebook) ──

SIMILARITY_THRESHOLD = 0.85   # default shared with semantic_entropy


def semantic_entropy_v2(E, tau=SIMILARITY_THRESHOLD):
    """
    Compute Hsem via agglomerative clustering on cosine distance.
    Returns (Hsem, cluster_labels, K).

    Alternative implementation contributed by project owner.
    Uses raw cosine metric rather than precomputed distance matrix.
    """
    N = len(E)
    distance_threshold = 1.0 - tau

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    cluster_labels = clustering.fit_predict(E)
    K = cluster_labels.max() + 1
    proportions = np.bincount(cluster_labels) / N
    Hsem = -sum(p * np.log2(p) for p in proportions if p > 0)
    return float(Hsem), cluster_labels, int(K)


def cosine_dispersion_v2(E):
    """
    Dcos — mean cosine distance from each embedding to the unit-normalised centroid.
    Contributed by project owner.
    """
    centroid = E.mean(axis=0, keepdims=True)        # (1, dim)
    centroid = normalize(centroid)                  # unit-normalise
    sims = cosine_similarity(E, centroid).flatten()
    return float(np.mean(1.0 - sims))


def pairwise_similarities(E):
    """
    Upper-triangle of the N×N cosine similarity matrix.
    Contributed by project owner.

    Parameters
    ----------
    E : np.ndarray   shape (N, d).

    Returns
    -------
    np.ndarray  of shape (N*(N-1)//2,) — all pairwise cosine similarities.
    """
    S = cosine_similarity(E)
    idx = np.triu_indices(len(E), k=1)
    return S[idx]


# ── Reference distribution ────────────────────────────────────────────────────

def fit_reference_distribution(all_embeddings, df, correctness_col="correctness"):
    """
    Fit the Mahalanobis reference distribution on correct-labelled responses.
    Uses Ledoit-Wolf shrinkage for a well-conditioned covariance estimate.

    Parameters
    ----------
    all_embeddings : np.ndarray   Full embedding matrix, shape (len(df), d).
    df : pd.DataFrame             Response-level dataframe aligned with embeddings.
    correctness_col : str

    Returns
    -------
    mu_ref : np.ndarray   Reference mean, shape (d,).
    cov_inv : np.ndarray  Precision matrix, shape (d, d).

    Method from hallucination_utils.py.
    """
    correct_mask = (df[correctness_col] == "correct").values
    correct_embs = all_embeddings[correct_mask]
    print(f"Correct responses for reference: {correct_embs.shape[0]}")

    if correct_embs.shape[0] < 10:
        print("WARNING: very few correct responses — using all embeddings as fallback.")
        correct_embs = all_embeddings

    mu_ref = correct_embs.mean(axis=0)
    lw = LedoitWolf()
    lw.fit(correct_embs)
    cov_inv = lw.precision_
    print(f"Reference fitted.  mu: {mu_ref.shape},  precision: {cov_inv.shape}")
    return mu_ref, cov_inv


# ── Question-level feature extraction ─────────────────────────────────────────

def extract_question_features(df, all_embeddings, mu_ref, cov_inv,
                               sim_threshold=0.85, domain_col="domain",
                               correctness_col="correctness"):
    """
    Extract geometric features and build the question-level feature DataFrame.

    For each question (grouped by prompt_id):
      - Computes the 5 geometric features from all response embeddings.
      - Counts correct / incorrect / refused responses.
      - Assigns a binary label where REFUSED is merged with INCORRECT.

    ⚠️  REFUSAL MERGE: label = 1  iff  (n_incorrect + n_refused) / n_total > 0.5
        Questions where ALL responses are refused are still skipped (n_total = 0
        after this merge they would still give an uninformative label).

    Parameters
    ----------
    df : pd.DataFrame        Response-level data.
    all_embeddings : np.ndarray  Aligned with df.
    mu_ref, cov_inv          From fit_reference_distribution.
    sim_threshold : float    Cosine similarity threshold for clustering.
    domain_col : str         Column for domain (raw or canonical).
    correctness_col : str

    Returns
    -------
    feat_df : pd.DataFrame   One row per question.
    skipped : int            Questions skipped (all refused).
    skipped_details : list[dict]

    Method from hallucination_utils.py, modified to merge refusals.
    """
    q_meta = compute_question_metadata(df, domain_col=domain_col)
    prompt_ids = df["prompt_id"].unique()

    records = []
    skipped = 0
    skipped_details = []

    for i, pid in enumerate(prompt_ids):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(prompt_ids)} ...")

        mask = df["prompt_id"] == pid
        sub  = df[mask]
        idx  = np.where(mask.values)[0]
        embs = all_embeddings[idx]
        n    = len(sub)

        dom       = q_meta.loc[pid, "domain_mode"]
        dom_incon = bool(q_meta.loc[pid, "domain_inconsistent"])

        counts     = sub[correctness_col].value_counts()
        n_correct  = int(counts.get("correct", 0))
        n_incorrect = int(counts.get("incorrect", 0))
        n_refused  = int(counts.get("refused", 0))

        # ── REFUSAL MERGE: refused counts as hallucinated ──────────────────
        # Skip only if we have literally zero definitive signal
        n_non_refused = n_correct + n_incorrect
        if n_non_refused == 0 and n_refused == n:
            # All refused — no signal at all
            skipped += 1
            skipped_details.append({
                "prompt_id": pid,
                domain_col: dom,
                "n_refused": n_refused,
                "n_samples": n,
            })
            continue

        # Hallucination rate: (incorrect + refused) / total
        n_hallu_merged = n_incorrect + n_refused
        hall_rate_merged = n_hallu_merged / n

        # Strict rate (correct vs incorrect only, for diagnostic comparison)
        n_definitive = n_correct + n_incorrect
        hall_rate_strict = (n_incorrect / n_definitive) if n_definitive > 0 else 1.0

        # Binary label with refusal merge
        label = 1 if hall_rate_merged > 0.5 else 0

        # ── Geometric features ─────────────────────────────────────────────
        H, K  = semantic_entropy(embs, threshold=sim_threshold)
        D     = cosine_dispersion(embs)
        M     = mahalanobis_distance(embs, mu_ref, cov_inv)
        sig2  = similarity_variance(embs)

        # ── Auxiliary stats ────────────────────────────────────────────────
        score_mean = float(sub["correctness_score"].mean())
        score_std  = float(sub["correctness_score"].std(ddof=0))
        len_mean   = float(sub["answer_len"].mean())
        len_std    = float(sub["answer_len"].std(ddof=0))

        row0 = sub.iloc[0]
        rec = {
            "prompt_id":          pid,
            "question":           row0["question"],
            domain_col:           dom,
            "domain_inconsistent": dom_incon,
            "adversarial":        bool(row0["adversarial"]),
            "dataset":            row0["dataset"],
            "n_samples":          n,
            # raw counts
            "n_correct":          n_correct,
            "n_incorrect":        n_incorrect,
            "n_refused":          n_refused,
            "n_definitive":       n_definitive,
            # fractions (all n responses)
            "frac_correct":       n_correct / n,
            "frac_incorrect":     n_incorrect / n,
            "frac_refused":       n_refused / n,
            # strict rate (correct vs incorrect only) — diagnostic
            "hall_rate_strict":   hall_rate_strict,
            # merged rate — primary label target
            "hall_rate_merged":   hall_rate_merged,
            "p_halluc":           hall_rate_merged,
            "hall_rate_naive":    1.0 - (n_correct / n),
            # auxiliary
            "score_mean":         score_mean,
            "score_std":          score_std,
            "len_mean":           len_mean,
            "len_std":            len_std,
            # geometric features
            "H_sem":  H,
            "D_cos":  D,
            "M_bar":  M,
            "K":      K,
            "sig2_S": sig2,
            # binary target (refusal merged)
            "label":  label,
        }

        if "type" in sub.columns and df["type"].nunique() > 1:
            rec["type"] = row0["type"]

        records.append(rec)

    feat_df = pd.DataFrame(records)

    print(f"\nQuestions processed : {len(feat_df)}")
    print(f"Questions skipped   : {skipped}  (all refused, no signal)")
    print(f"Label distribution  : Correct={( feat_df['label']==0).sum()}  "
          f"Hallucinated (incl. refused)={(feat_df['label']==1).sum()}")
    print(f"Hallucination rate  : {feat_df['label'].mean()*100:.1f}%")

    return feat_df, skipped, skipped_details


# ── Build question-metadata for domain analysis ────────────────────────────────
# (re-export for convenience in notebooks)

def build_domain_stats(feat_df, strict_rate_col, domain_col="domain",
                        label_col="label", extra_mean_cols=None,
                        analysis_domains=None):
    """
    Summary table of hallucination rate, entropy, and question count per domain.
    Method from hallucination_utils.py.
    """
    import pandas as pd  # local re-import for safety in functional style

    agg = {
        "n_questions":    (label_col, "count"),
        "n_hallucinated": (label_col, "sum"),
        "hall_rate_mean": (strict_rate_col, "mean"),
        "hall_rate_std":  (strict_rate_col, "std"),
        "mean_entropy":   ("H_sem", "mean"),
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