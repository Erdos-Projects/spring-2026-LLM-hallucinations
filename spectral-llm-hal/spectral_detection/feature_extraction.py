"""
spectral_detection/feature_extraction.py

Embedding and geometric-feature extraction — Stage 3 of the pipeline.

Six features are computed per question from the N=20 response embeddings:

  Symbol      Description                                  Reference
  ─────────   ────────────────────────────────────────     ─────────────────────────
  H_sem       Semantic entropy over response clusters      Farquhar et al. (2024)
  D_cos       Mean centroid cosine distance                Ricco et al. (2025)
  D_cos_var   Variance of centroid cosine distances        this project
  M_bar       Mean Mahalanobis distance from reference     Lee et al. (2018)
  K           Number of semantic clusters
  sig2_S      Variance of pairwise cosine similarities

LaTeX formulas (for notebook rendering):

  H_sem    = -sum_{k=1}^{K}  p_k log2(p_k),      p_k = |C_k| / N

  D_cos    = (1/N) sum_j (1 - cos(e_j, ebar/||ebar||)),   ebar = mean_j e_j

  D_cos_var = Var_j [ 1 - cos(e_j, ebar/||ebar||) ]

  M_bar    = (1/N) sum_j sqrt( (e_j - mu_ref)^T Sigma_ref^{-1} (e_j - mu_ref) )

  K        = number of agglomerative clusters (cosine threshold tau = 0.85)

  sig2_S   = Var { S_{jk} : j < k },   S_{jk} = cos(e_j, e_k)


REFUSAL MERGE POLICY
    Refused responses are treated as hallucinations for binary labelling.
    label = 1  iff  (n_incorrect + n_refused) / n_total > 0.5

VECTORISED COSINE COMPUTATION
    When all questions have the same number of responses, all cosine matrices
    are computed in one batched matrix-multiply:
        S = E_batch @ E_batch.transpose(0,2,1)   shape (n_q, N, N)
    Valid because embed_responses L2-normalises by default.
    Agglomerative clustering still requires a per-question loop (sklearn).

Methods sourced from hallucination_utils.py (Debanjan B.);
vectorised cosine path and D_cos_var adapted from AJ Vargas.
"""

import os
import numpy as np
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
    Embed all responses with a SentenceTransformer.
    Loads from cache_path if the file exists; otherwise computes and saves.

    Parameters
    ----------
    df             : pd.DataFrame
    model_name     : str
    text_col       : str
    cache_path     : str, optional   .npy cache file
    batch_size     : int
    normalize_embs : bool  L2-normalise (required for vectorised cosine path)

    Returns
    -------
    np.ndarray  shape (len(df), embedding_dim)

    Method from hallucination_utils.py.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        embs = np.load(cache_path)
        assert embs.shape[0] == len(df), (
            f"Cache size mismatch: {embs.shape[0]} vs {len(df)} rows")
        return embs

    from sentence_transformers import SentenceTransformer
    print(f"Computing embeddings with {model_name} ...")
    embedder = SentenceTransformer(model_name)
    embs = embedder.encode(
        df[text_col].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embs,
    )
    if cache_path:
        np.save(cache_path, embs)
        print(f"Saved embeddings to {cache_path}")
    print(f"Embedding matrix: {embs.shape}")
    return embs


# ── Scalar per-question feature functions ─────────────────────────────────────

def semantic_entropy(embs, threshold=0.85):
    """
    H_sem = -sum_k p_k log2(p_k)  via agglomerative clustering (tau = threshold).

    Returns (H, K).
    Method from hallucination_utils.py.
    """
    n = len(embs)
    dist_matrix = np.clip(1.0 - cosine_similarity(embs), 0, 2)
    np.fill_diagonal(dist_matrix, 0)
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="precomputed",
            linkage="average", distance_threshold=1 - threshold)
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity="precomputed",
            linkage="average", distance_threshold=1 - threshold)
    labels = clustering.fit_predict(dist_matrix)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    H = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return H, int(len(counts))


def cosine_dispersion(embs):
    """
    D_cos = (1/N) sum_j (1 - cos(e_j, ebar/||ebar||))
    Method from hallucination_utils.py.
    """
    centroid = embs.mean(axis=0)
    centroid_unit = centroid / (np.linalg.norm(centroid) + 1e-12)
    dists = 1.0 - (embs @ centroid_unit)
    return float(dists.mean())


def cosine_dispersion_variance(embs):
    """
    D_cos_var = Var_j [ 1 - cos(e_j, ebar/||ebar||) ]

    Captures whether the response cloud is symmetric around the centroid or
    has a long tail of outlying responses (asymmetric scatter).
    Adapted from AJ.
    """
    centroid = embs.mean(axis=0)
    centroid_unit = centroid / (np.linalg.norm(centroid) + 1e-12)
    dists = 1.0 - (embs @ centroid_unit)
    return float(dists.var())


def mahalanobis_distances_all(embs, mu, cov_inv):
    """
    Compute Mahalanobis distance statistics from a single pass.

        d_j = sqrt( (e_j - μ)ᵀ Σ⁻¹ (e_j - μ) )

        M_bar = (1/N) Σ_j d_j    — mean distance [Lee et al. 2018; in pipeline]

    Only M_bar is stored in GEO_FEATURES. M_var and M_max are computed here
    for completeness but discarded by extract_question_features.

    Returns
    -------
    M_bar : float
    """
    diffs    = embs - mu
    mahal_sq = np.einsum("ni,ij,nj->n", diffs, cov_inv, diffs)
    dists    = np.sqrt(np.clip(mahal_sq, 0, None))
    return float(dists.mean()), float(dists.var()), float(dists.max())


def mahalanobis_distance(embs, mu, cov_inv):
    """
    M_bar only — kept for backward compatibility with existing code.
    Prefer mahalanobis_distances_all() for new code.
    Method from hallucination_utils.py.
    """
    M_bar, _, _ = mahalanobis_distances_all(embs, mu, cov_inv)
    return M_bar


def mean_pairwise_cosine_distance(embs):
    """
    D_pair = mean cosine distance across all N(N-1)/2 response pairs.

        D_pair = (1 / C(N,2)) Σ_{j<k} (1 - S_{jk})

    where S_{jk} = cos(e_j, e_k).

    This differs from D_cos (centroid distance) in that it measures the
    average separation between responses directly, without reference to
    a centroid.  A tight cluster with a centroid offset from origin gives
    low D_pair but potentially non-zero D_cos; scattered responses give
    high D_pair regardless of centroid location.

    Adapted from AJ's pairwise_geometry_features().
    """
    sim   = cosine_similarity(embs)
    upper = sim[np.triu_indices(len(embs), k=1)]
    return float((1.0 - upper).mean())


def similarity_variance(embs):
    """
    sig2_S = Var { cos(e_j, e_k) : j < k }
    Method from hallucination_utils.py.
    """
    sim = cosine_similarity(embs)
    upper = sim[np.triu_indices(len(embs), k=1)]
    return float(np.var(upper))


# Vectorised batch geometry (AJ's approach) 

def _batch_cosine_matrices(E_batch):
    """
    Compute all N×N cosine similarity matrices in one batched matmul.

    Parameters
    ----------
    E_batch : np.ndarray  shape (n_q, n_resp, dim) — must be L2-normalised.

    Returns
    -------
    S : np.ndarray  shape (n_q, n_resp, n_resp)

    Because embeddings are unit-norm, dot-product == cosine similarity.
    S = E @ E^T  (AJ's vectorised approach).
    """
    return E_batch @ E_batch.transpose(0, 2, 1)


def _batch_pairwise_geometry(E_batch, S):
    """
    Vectorised D_cos, D_cos_var, sig2_S, D_pair for all questions at once.

    Parameters
    ----------
    E_batch : np.ndarray  shape (n_q, n_resp, dim)
    S       : np.ndarray  shape (n_q, n_resp, n_resp)

    Returns
    -------
    D_cos, D_cos_var, sig2_S, D_pair : each np.ndarray of shape (n_q,)
    """
    n_q, n_resp, _ = E_batch.shape

    # centroid-based distances
    centroid = E_batch.mean(axis=1)                               # (n_q, dim)
    norm_c   = np.linalg.norm(centroid, axis=1, keepdims=True)
    c_unit   = centroid / (norm_c + 1e-12)                       # (n_q, dim)
    cos_c    = np.einsum("nrd,nd->nr", E_batch, c_unit)           # (n_q, n_resp)
    d_c      = 1.0 - cos_c                                       # (n_q, n_resp)
    D_cos     = d_c.mean(axis=1)
    D_cos_var = d_c.var(axis=1)

    # pairwise upper-triangle
    iu     = np.triu_indices(n_resp, k=1)
    pairs  = S[:, iu[0], iu[1]]                                  # (n_q, n_pairs)
    sig2_S = pairs.var(axis=1)
    D_pair = (1.0 - pairs).mean(axis=1)                          # mean pairwise distance

    return D_cos, D_cos_var, sig2_S, D_pair


# ── Reference distribution ─────────────────────────────────────────────────────

def fit_reference_distribution(all_embeddings, df, correctness_col="correctness"):
    """
    Fit M_bar reference (mu, Sigma^{-1}) on correct-labelled responses.
    Uses Ledoit-Wolf shrinkage for a well-conditioned precision matrix.

    Method from hallucination_utils.py.
    """
    correct_mask = (df[correctness_col] == "correct").values
    correct_embs = all_embeddings[correct_mask]
    print(f"Correct responses for reference fit: {correct_embs.shape[0]}")
    if correct_embs.shape[0] < 10:
        print("WARNING: very few correct responses — using all embeddings as fallback.")
        correct_embs = all_embeddings
    mu_ref  = correct_embs.mean(axis=0)
    lw      = LedoitWolf()
    lw.fit(correct_embs)
    cov_inv = lw.precision_
    print(f"Reference fitted.  mu: {mu_ref.shape},  Sigma^-1: {cov_inv.shape}")
    return mu_ref, cov_inv


# ── Question-level feature extraction ─────────────────────────────────────────

def extract_question_features(df, all_embeddings, mu_ref, cov_inv,
                               sim_threshold=0.85, domain_col="domain",
                               correctness_col="correctness"):
    """
    Extract 6 geometric features per question and return a feature DataFrame.

    REFUSAL MERGE: label = 1  iff  (n_incorrect + n_refused) / n_total > 0.5
    Questions where ALL responses are refused are skipped.

    VECTORISED PATH: if all questions have an equal number of responses,
    D_cos, D_cos_var, sig2_S are computed via batched matmul (AJ's approach).
    H_sem and K still use a per-question clustering loop.

    Parameters
    ----------
    df               : pd.DataFrame  response-level data
    all_embeddings   : np.ndarray    aligned with df, shape (len(df), d)
    mu_ref, cov_inv  : from fit_reference_distribution
    sim_threshold    : float
    domain_col       : str
    correctness_col  : str

    Returns
    -------
    feat_df          : pd.DataFrame  one row per question, 6 geometric features
    skipped          : int
    skipped_details  : list[dict]

    Method from hallucination_utils.py; extended with D_cos_var and vectorised
    cosine computation (AJ).
    """
    import pandas as pd

    q_meta     = compute_question_metadata(df, domain_col=domain_col)
    prompt_ids = df["prompt_id"].unique()

    # ── vectorised path check ──────────────────────────────────────────────────
    _sizes   = df.groupby("prompt_id").size()
    _uniform = (_sizes == _sizes.iloc[0]).all()

    if _uniform:
        _n_resp = int(_sizes.iloc[0])
        print(f"Vectorised cosine path: {len(prompt_ids)} q x {_n_resp} resp")
        pid_to_qi = {pid: i for i, pid in enumerate(prompt_ids)}
        d = all_embeddings.shape[1]
        E_batch = np.zeros((len(prompt_ids), _n_resp, d),
                           dtype=all_embeddings.dtype)
        _rc = np.zeros(len(prompt_ids), dtype=int)
        for row_i in range(len(df)):
            pid = df["prompt_id"].iloc[row_i]
            qi  = pid_to_qi[pid]
            ri  = _rc[qi]
            E_batch[qi, ri] = all_embeddings[row_i]
            _rc[qi] += 1
        S_batch                                  = _batch_cosine_matrices(E_batch)
        D_cos_v, D_cos_var_v, sig2v, D_pair_v   = _batch_pairwise_geometry(E_batch, S_batch)
        use_vec = True
    else:
        print("Unequal response counts — using per-question scalar loop.")
        pid_to_qi = {}
        use_vec   = False

    # ── main extraction loop ───────────────────────────────────────────────────
    records         = []
    skipped         = 0
    skipped_details = []

    for i, pid in enumerate(prompt_ids):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(prompt_ids)} ...")

        mask = df["prompt_id"] == pid
        sub  = df[mask]
        idx  = np.where(mask.values)[0]
        embs = all_embeddings[idx]
        n    = len(sub)

        dom       = q_meta.loc[pid, "domain_mode"]
        dom_incon = bool(q_meta.loc[pid, "domain_inconsistent"])

        counts     = sub[correctness_col].value_counts()
        n_correct  = int(counts.get("correct",   0))
        n_incorrect = int(counts.get("incorrect", 0))
        n_refused  = int(counts.get("refused",   0))

        if n_correct == 0 and n_incorrect == 0:
            skipped += 1
            skipped_details.append({
                "prompt_id": pid, domain_col: dom,
                "n_refused": n_refused, "n_samples": n,
            })
            continue

        # label: refusals merged as hallucinated
        n_hallu_merged   = n_incorrect + n_refused
        hall_rate_merged = n_hallu_merged / n
        n_definitive     = n_correct + n_incorrect
        hall_rate_strict = (n_incorrect / n_definitive) if n_definitive > 0 else 1.0
        label = 1 if hall_rate_merged > 0.5 else 0

        # ── 9 geometric features ───────────────────────────────────────────────
        H, K = semantic_entropy(embs, threshold=sim_threshold)  # always loop

        if use_vec:
            qi   = pid_to_qi[pid]
            D    = float(D_cos_v[qi])
            Dvar = float(D_cos_var_v[qi])
            sig2 = float(sig2v[qi])
            Dpair = float(D_pair_v[qi])
        else:
            D    = cosine_dispersion(embs)
            Dvar = cosine_dispersion_variance(embs)
            sig2 = similarity_variance(embs)
            Dpair = mean_pairwise_cosine_distance(embs)

        M_bar, _, _ = mahalanobis_distances_all(embs, mu_ref, cov_inv)

        # ── auxiliary ─────────────────────────────────────────────────────────
        score_mean = float(sub["correctness_score"].mean())
        score_std  = float(sub["correctness_score"].std(ddof=0))
        len_mean   = float(sub["answer_len"].mean())
        len_std    = float(sub["answer_len"].std(ddof=0))

        row0 = sub.iloc[0]
        rec = {
            "prompt_id":           pid,
            "question":            row0["question"],
            domain_col:            dom,
            "domain_inconsistent": dom_incon,
            "adversarial":         bool(row0["adversarial"]),
            "dataset":             row0["dataset"],
            "n_samples":           n,
            "n_correct":           n_correct,
            "n_incorrect":         n_incorrect,
            "n_refused":           n_refused,
            "n_definitive":        n_definitive,
            "frac_correct":        n_correct / n,
            "frac_incorrect":      n_incorrect / n,
            "frac_refused":        n_refused / n,
            "hall_rate_strict":    hall_rate_strict,
            "hall_rate_merged":    hall_rate_merged,
            "p_halluc":            hall_rate_merged,
            "hall_rate_naive":     1.0 - (n_correct / n),
            "score_mean":          score_mean,
            "score_std":           score_std,
            "len_mean":            len_mean,
            "len_std":             len_std,
            # ── 7 geometric features ──────────────────────────────────────────────
            "H_sem":    H,
            "D_cos":    D,
            "D_cos_var": Dvar,
            "D_pair":   Dpair,
            "M_bar":    M_bar,
            "K":        K,
            "sig2_S":   sig2,
            "label":    label,
        }
        if "type" in sub.columns and df["type"].nunique() > 1:
            rec["type"] = row0["type"]
        records.append(rec)

    feat_df = pd.DataFrame(records)
    print(f"\nProcessed : {len(feat_df)}  |  Skipped (all-refused): {skipped}")
    print(f"Label     : Correct={( feat_df['label']==0).sum()}  "
          f"Hallucinated={(feat_df['label']==1).sum()}  "
          f"Rate={feat_df['label'].mean()*100:.1f}%")
    return feat_df, skipped, skipped_details


# ── Domain stats ───────────────────────────────────────────────────────────────

def build_domain_stats(feat_df, strict_rate_col, domain_col="domain",
                        label_col="label", extra_mean_cols=None,
                        analysis_domains=None):
    """
    Summary table: hallucination rate, entropy, question count per domain.
    Method from hallucination_utils.py.
    """
    import pandas as pd

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
    ds["pct_hallucinated"] = (
        ds["n_hallucinated"] / ds["n_questions"] * 100).round(1)
    if analysis_domains is not None:
        ds["in_analysis"] = [d in analysis_domains for d in ds.index]
    return ds


# ── Extra functions (project owner's notebook) ────────────────────────────────

SIMILARITY_THRESHOLD = 0.85


def semantic_entropy_v2(E, tau=SIMILARITY_THRESHOLD):
    """Alternative H_sem using raw cosine metric. Contributed by project owner."""
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="cosine",
        linkage="average", distance_threshold=1.0 - tau)
    cluster_labels = clustering.fit_predict(E)
    K = cluster_labels.max() + 1
    proportions = np.bincount(cluster_labels) / len(E)
    Hsem = -sum(p * np.log2(p) for p in proportions if p > 0)
    return float(Hsem), cluster_labels, int(K)


def cosine_dispersion_v2(E):
    """D_cos with sklearn normalize centroid. Contributed by project owner."""
    centroid = E.mean(axis=0, keepdims=True)
    centroid = normalize(centroid)
    sims = cosine_similarity(E, centroid).flatten()
    return float(np.mean(1.0 - sims))


def pairwise_similarities(E):
    """Upper-triangle of the N×N cosine similarity matrix. Project owner."""
    S = cosine_similarity(E)
    idx = np.triu_indices(len(E), k=1)
    return S[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  SPECTRAL GRAPH FEATURES  (from AJ's work — refactored into our architecture)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Given N response embeddings for one question, we build a weighted graph
#  whose adjacency matrix is the N×N cosine similarity matrix W (diagonal=0,
#  negative values clipped to 0).  The graph Laplacian is:
#
#       L = D − W,   D_{ii} = Σ_j W_{ij}   (degree matrix)
#
#  Eigendecomposition of L gives 0 = λ₁ ≤ λ₂ ≤ λ₃ ≤ … ≤ λ_N.
#  The second eigenvalue λ₂ (Fiedler value) is the algebraic connectivity of
#  the graph.  λ₂ = 0 iff the graph is disconnected; high λ₂ means the
#  responses form a tightly connected, agreeing cluster.
#
#  Refactored from AJ's spectral_graph_features() and cluster_structure_features().
# ══════════════════════════════════════════════════════════════════════════════

# Names and display labels for the two new feature groups
SPECTRAL_FEATURES = ["lam2", "lam3", "SGR", "spectral_entropy",
                     "ipr_fiedler", "HFER"]

SPECTRAL_NICE_NAMES = {
    "lam2":             "Fiedler Value (λ₂)",
    "lam3":             "Third Eigenvalue (λ₃)",
    "SGR":              "Spectral Gap Ratio (λ₂/λ₃)",
    "spectral_entropy": "Spectral Entropy",
    "ipr_fiedler":      "IPR Fiedler Vector",
    "HFER":             "HFER (λ₂ × Fiedler Entropy)",
}

EXTENDED_CLUSTER_FEATURES = ["largest_cluster_frac", "second_largest_cluster_frac",
                              "singleton_cluster_frac"]

EXTENDED_CLUSTER_NICE_NAMES = {
    "largest_cluster_frac":        "Largest Cluster Fraction",
    "second_largest_cluster_frac": "Second-Largest Cluster Fraction",
    "singleton_cluster_frac":      "Singleton Cluster Fraction",
}

# All new features together
ALL_NEW_FEATURES = SPECTRAL_FEATURES + EXTENDED_CLUSTER_FEATURES
# Total: 6 spectral + 3 extended cluster = 9 new features
ALL_NEW_NICE_NAMES = {**SPECTRAL_NICE_NAMES, **EXTENDED_CLUSTER_NICE_NAMES}


def spectral_graph_features_single(S_i):
    """
    Compute spectral graph features for a single N×N cosine similarity matrix.

    Graph construction
    ------------------
    W_{jk} = max(S_{jk}, 0)  with W_{jj} = 0
    L = D − W,  D_{ii} = Σ_j W_{ij}

    Features
    --------
    lam2             : λ₂ — Fiedler value (algebraic connectivity)
                       λ₂ ≈ 0  → graph nearly disconnected (responses diverge)
                       λ₂ large → tightly connected (responses agree)

    lam3             : λ₃ — third-smallest eigenvalue

    SGR              : λ₂ / (λ₃ + ε)  — Spectral Gap Ratio
                       High SGR → clean bipartition of responses

    spectral_entropy : H = −Σ q_i log₂ q_i,  q_i = λ_i / Σ λ_i  (λ_i > 0)
                       High H → eigenvalues spread evenly → complex structure

    ipr_fiedler      : Σ_j v₂[j]⁴  — Inverse Participation Ratio of Fiedler vector v₂
                       High IPR → partition is localised on a few responses
                       Low IPR  → partition is diffuse (all responses equally split)

    HFER             : λ₂ × H_fiedler
                       where H_fiedler = −Σ p_j log₂ p_j,
                       p_j = |v₂[j]| / Σ |v₂[j]|
                       Combines algebraic connectivity with Fiedler geometry

    Reference: AJ (this project), building on spectral graph theory.
    """
    W = np.clip(S_i, 0, None).copy()
    np.fill_diagonal(W, 0.0)

    d = W.sum(axis=1)
    L = np.diag(d) - W

    evals, evecs = np.linalg.eigh(L)          # ascending order, symmetric

    lam2 = float(evals[1])
    lam3 = float(evals[2])
    sgr  = lam2 / (lam3 + 1e-8)

    # spectral entropy over positive eigenvalues
    q = evals / (evals.sum() + 1e-12)
    q = q[q > 0]
    h_spec = float(-(q * np.log2(q)).sum())

    # Fiedler vector properties
    v2      = evecs[:, 1]
    ipr     = float(np.sum(v2 ** 4))
    p       = np.abs(v2)
    p       = p / (p.sum() + 1e-12)
    p       = p[p > 0]
    h_fied  = float(-(p * np.log2(p)).sum())
    hfer    = lam2 * h_fied

    return {
        "lam2":             lam2,
        "lam3":             lam3,
        "SGR":              float(sgr),
        "spectral_entropy": h_spec,
        "ipr_fiedler":      ipr,
        "HFER":             hfer,
    }


def extended_cluster_features_single(cluster_labels, n_responses):
    """
    Compute extended cluster structure features for one question.

    **cluster_gap** — REMOVED. `lam2` and `lam3` already capture the
    dominance structure of the response partition via the graph Laplacian;
    `cluster_gap` is redundant with those and with `largest_cluster_frac`.

    Features
    --------
    largest_cluster_frac        : p₁ — fraction of responses in the dominant cluster
                                  High p₁ → model confidently converges on one answer

    second_largest_cluster_frac : p₂ — fraction in the runner-up cluster
                                  High p₂ + low p₁ → two competing answers

    singleton_cluster_frac      : fraction of responses that form their own cluster
                                  (clusters of size 1)
                                  High value → many isolated outlier responses → incoherence

    Reference: AJ (this project), extending Farquhar et al. (2024).
    """
    counts       = np.bincount(cluster_labels)
    probs        = counts / counts.sum()
    probs_sorted = np.sort(probs)[::-1]

    p1 = float(probs_sorted[0])
    p2 = float(probs_sorted[1]) if len(probs_sorted) > 1 else 0.0

    return {
        "largest_cluster_frac":        p1,
        "second_largest_cluster_frac": p2,
        "singleton_cluster_frac":      float((counts == 1).sum() / n_responses),
    }


def extract_spectral_features(df, all_embeddings, sim_threshold=0.85,
                               domain_col="domain", verbose=True):
    """
    Compute spectral graph + extended cluster features for every question.

    This function is designed to be called on the COMBINED dataset after
    embed_responses() has already been run.  It returns a DataFrame that
    can be merged with the existing feat_df from extract_question_features().

    The cosine similarity matrices S are computed via the vectorised batch
    matmul path (AJ's approach) when all questions have equal response counts.

    Parameters
    ----------
    df             : pd.DataFrame  response-level data (same as used for GEO_FEATURES)
    all_embeddings : np.ndarray    L2-normalised embeddings, shape (len(df), d)
    sim_threshold  : float         clustering threshold (same τ as extract_question_features)
    domain_col     : str
    verbose        : bool

    Returns
    -------
    pd.DataFrame  indexed by prompt_id, columns = SPECTRAL_FEATURES +
                  EXTENDED_CLUSTER_FEATURES + ['dataset']
    """
    import pandas as pd

    prompt_ids = df["prompt_id"].unique()
    _sizes     = df.groupby("prompt_id").size()
    _uniform   = (_sizes == _sizes.iloc[0]).all()

    # Build E_batch for vectorised cosine computation
    if _uniform:
        _n_resp  = int(_sizes.iloc[0])
        pid_to_qi = {pid: i for i, pid in enumerate(prompt_ids)}
        d        = all_embeddings.shape[1]
        E_batch  = np.zeros((len(prompt_ids), _n_resp, d),
                            dtype=all_embeddings.dtype)
        _rc      = np.zeros(len(prompt_ids), dtype=int)
        for row_i in range(len(df)):
            pid = df["prompt_id"].iloc[row_i]
            qi  = pid_to_qi[pid]
            E_batch[qi, _rc[qi]] = all_embeddings[row_i]
            _rc[qi] += 1
        S_batch  = _batch_cosine_matrices(E_batch)   # (n_q, N, N)
        if verbose:
            print(f"Vectorised cosine path: {len(prompt_ids)} q × {_n_resp} resp")
    else:
        E_batch   = None
        S_batch   = None
        pid_to_qi = {}
        if verbose:
            print("Scalar path (unequal response counts)")

    records = []
    for i, pid in enumerate(prompt_ids):
        if verbose and (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(prompt_ids)} ...")

        mask = df["prompt_id"] == pid
        sub  = df[mask]
        idx  = np.where(mask.values)[0]
        embs = all_embeddings[idx]
        n    = len(sub)

        # ── cosine similarity matrix ───────────────────────────────────────
        if _uniform:
            qi  = pid_to_qi[pid]
            S_i = S_batch[qi]                      # (N, N) already computed
        else:
            from sklearn.metrics.pairwise import cosine_similarity as _cs
            S_i = _cs(embs)

        # ── spectral graph features ────────────────────────────────────────
        spec = spectral_graph_features_single(S_i)

        # ── extended cluster features ──────────────────────────────────────
        dist_matrix = np.clip(1.0 - S_i, 0, 2)
        np.fill_diagonal(dist_matrix, 0)
        try:
            clust = AgglomerativeClustering(
                n_clusters=None, metric="precomputed",
                linkage="average", distance_threshold=1 - sim_threshold)
        except TypeError:
            clust = AgglomerativeClustering(
                n_clusters=None, affinity="precomputed",
                linkage="average", distance_threshold=1 - sim_threshold)
        labels = clust.fit_predict(dist_matrix)
        ext    = extended_cluster_features_single(labels, n)

        rec = {"prompt_id": pid, "dataset": sub.iloc[0]["dataset"]}
        rec.update(spec)
        rec.update(ext)
        records.append(rec)

    result = pd.DataFrame(records).set_index("prompt_id")
    if verbose:
        print(f"\nSpectral features computed for {len(result)} questions.")
    return result
