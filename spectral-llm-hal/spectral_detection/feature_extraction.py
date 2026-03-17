"""
spectral_detection/feature_extraction.py

Embedding and geometric feature extraction.

Six features per question from N=20 response embeddings:
  H_sem     : Semantic entropy (cluster-based)       [Farquhar et al. 2024]
  D_cos     : Mean centroid cosine distance           [Ricco et al. 2025]
  D_cos_var : Variance of centroid distances
  D_pair    : Mean pairwise cosine distance
  K         : Cluster count
  sig2_S    : Pairwise similarity variance

All features are unsupervised: they measure the geometry of response
embeddings without using correctness labels.

Refusal policy: refused responses are merged with incorrect for binary labelling.
  label = 1  iff  (n_incorrect + n_refused) / n_total > 0.5
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from spectral_detection.data.cleaning import FEATURES, compute_question_metadata


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_responses(df, model_name="all-MiniLM-L6-v2",
                    text_col="model_answer", cache_path=None,
                    batch_size=256, normalize_embs=True):
    """Embed responses with SentenceTransformer. Caches to .npy if cache_path given."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        embs = np.load(cache_path)
        assert embs.shape[0] == len(df), f"Cache mismatch: {embs.shape[0]} vs {len(df)}"
        return embs

    from sentence_transformers import SentenceTransformer
    print(f"Computing embeddings with {model_name} ...")
    embedder = SentenceTransformer(model_name)
    embs = embedder.encode(
        df[text_col].astype(str).tolist(),
        batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=normalize_embs,
    )
    if cache_path:
        np.save(cache_path, embs)
        print(f"Saved embeddings to {cache_path}")
    print(f"Embedding matrix: {embs.shape}")
    return embs


# ── Per-question feature functions ─────────────────────────────────────────────

def semantic_entropy(embs, threshold=0.85):
    """H_sem = -sum p_k log2(p_k) via agglomerative clustering. Returns (H, K)."""
    dist = np.clip(1.0 - cosine_similarity(embs), 0, 2)
    np.fill_diagonal(dist, 0)
    try:
        clust = AgglomerativeClustering(
            n_clusters=None, metric="precomputed",
            linkage="average", distance_threshold=1 - threshold)
    except TypeError:
        clust = AgglomerativeClustering(
            n_clusters=None, affinity="precomputed",
            linkage="average", distance_threshold=1 - threshold)
    labels = clust.fit_predict(dist)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(embs)
    H = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return H, int(len(counts))


def cosine_dispersion(embs):
    """D_cos = mean(1 - cos(e_j, centroid))."""
    centroid = embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    return float((1.0 - embs @ centroid).mean())


def cosine_dispersion_variance(embs):
    """D_cos_var = var(1 - cos(e_j, centroid))."""
    centroid = embs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    return float((1.0 - embs @ centroid).var())


def mean_pairwise_cosine_distance(embs):
    """D_pair = mean(1 - S_jk) over all pairs j < k."""
    sim = cosine_similarity(embs)
    upper = sim[np.triu_indices(len(embs), k=1)]
    return float((1.0 - upper).mean())


def similarity_variance(embs):
    """sig2_S = var(cos(e_j, e_k)) over all pairs j < k."""
    sim = cosine_similarity(embs)
    upper = sim[np.triu_indices(len(embs), k=1)]
    return float(np.var(upper))


# ── Vectorised batch computation ───────────────────────────────────────────────

def _batch_cosine_matrices(E_batch):
    """Batched N x N cosine similarity: S = E @ E^T (requires L2-normalised E)."""
    return E_batch @ E_batch.transpose(0, 2, 1)


def _batch_pairwise_geometry(E_batch, S):
    """Vectorised D_cos, D_cos_var, sig2_S, D_pair for all questions at once."""
    n_q, n_resp, _ = E_batch.shape

    # Centroid-based distances
    centroid = E_batch.mean(axis=1)
    c_unit = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-12)
    d_c = 1.0 - np.einsum("nrd,nd->nr", E_batch, c_unit)
    D_cos = d_c.mean(axis=1)
    D_cos_var = d_c.var(axis=1)

    # Pairwise upper-triangle
    iu = np.triu_indices(n_resp, k=1)
    pairs = S[:, iu[0], iu[1]]
    sig2_S = pairs.var(axis=1)
    D_pair = (1.0 - pairs).mean(axis=1)

    return D_cos, D_cos_var, sig2_S, D_pair


# ── Question-level feature extraction ─────────────────────────────────────────

def extract_question_features(df, all_embeddings,
                               sim_threshold=0.85, domain_col="domain",
                               correctness_col="correctness"):
    """
    Extract 6 geometric features per question. Returns (feat_df, n_skipped, skip_details).

    All features are unsupervised: computed from response embedding geometry only,
    without using correctness labels.

    Vectorised path used when all questions have equal response counts.
    Questions where ALL responses are refused are skipped.
    """
    q_meta = compute_question_metadata(df, domain_col=domain_col)
    prompt_ids = df["prompt_id"].unique()

    # Check if vectorised path is possible
    sizes = df.groupby("prompt_id").size()
    uniform = (sizes == sizes.iloc[0]).all()

    if uniform:
        n_resp = int(sizes.iloc[0])
        print(f"Vectorised path: {len(prompt_ids)} questions x {n_resp} responses")
        pid_to_idx = {pid: i for i, pid in enumerate(prompt_ids)}
        d = all_embeddings.shape[1]
        E_batch = np.zeros((len(prompt_ids), n_resp, d), dtype=all_embeddings.dtype)
        counters = np.zeros(len(prompt_ids), dtype=int)
        for row_i in range(len(df)):
            qi = pid_to_idx[df["prompt_id"].iloc[row_i]]
            E_batch[qi, counters[qi]] = all_embeddings[row_i]
            counters[qi] += 1
        S_batch = _batch_cosine_matrices(E_batch)
        D_cos_v, D_cos_var_v, sig2_v, D_pair_v = _batch_pairwise_geometry(E_batch, S_batch)
    else:
        print("Unequal response counts: using scalar loop.")
        pid_to_idx = {}

    records, skipped, skip_details = [], 0, []

    for i, pid in enumerate(prompt_ids):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(prompt_ids)} ...")

        mask = df["prompt_id"] == pid
        sub = df[mask]
        row_indices = np.where(mask.values)[0]
        embs = all_embeddings[row_indices]
        n = len(sub)

        dom = q_meta.loc[pid, "domain_mode"]
        dom_inconsistent = bool(q_meta.loc[pid, "domain_inconsistent"])

        counts = sub[correctness_col].value_counts()
        n_correct = int(counts.get("correct", 0))
        n_incorrect = int(counts.get("incorrect", 0))
        n_refused = int(counts.get("refused", 0))

        # Skip all-refused questions
        if n_correct == 0 and n_incorrect == 0:
            skipped += 1
            skip_details.append({"prompt_id": pid, domain_col: dom,
                                  "n_refused": n_refused, "n_samples": n})
            continue

        # Binary label: refusals count as hallucinated
        hallu_rate = (n_incorrect + n_refused) / n
        label = 1 if hallu_rate > 0.5 else 0

        # Compute features
        H, K = semantic_entropy(embs, threshold=sim_threshold)

        if uniform:
            qi = pid_to_idx[pid]
            D, Dvar, sig2, Dpair = (
                float(D_cos_v[qi]), float(D_cos_var_v[qi]),
                float(sig2_v[qi]), float(D_pair_v[qi]),
            )
        else:
            D = cosine_dispersion(embs)
            Dvar = cosine_dispersion_variance(embs)
            sig2 = similarity_variance(embs)
            Dpair = mean_pairwise_cosine_distance(embs)

        n_definitive = n_correct + n_incorrect
        row0 = sub.iloc[0]
        records.append({
            "prompt_id": pid,
            "question": row0["question"],
            domain_col: dom,
            "domain_inconsistent": dom_inconsistent,
            "adversarial": bool(row0["adversarial"]),
            "dataset": row0["dataset"],
            "n_samples": n,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_refused": n_refused,
            "n_definitive": n_definitive,
            "frac_correct": n_correct / n,
            "frac_incorrect": n_incorrect / n,
            "frac_refused": n_refused / n,
            "hall_rate_strict": (n_incorrect / n_definitive) if n_definitive > 0 else 1.0,
            "hall_rate_merged": hallu_rate,
            "p_halluc": hallu_rate,
            "score_mean": float(sub["correctness_score"].mean()),
            "score_std": float(sub["correctness_score"].std(ddof=0)),
            "len_mean": float(sub["answer_len"].mean()),
            "len_std": float(sub["answer_len"].std(ddof=0)),
            "H_sem": H, "D_cos": D, "D_cos_var": Dvar, "D_pair": Dpair,
            "K": K, "sig2_S": sig2,
            "label": label,
            **({"type": row0["type"]} if "type" in sub.columns and df["type"].nunique() > 1 else {}),
        })

    feat_df = pd.DataFrame(records)
    print(f"\nProcessed: {len(feat_df)}  |  Skipped (all-refused): {skipped}")
    print(f"Label: Correct={(feat_df['label']==0).sum()}, "
          f"Hallucinated={(feat_df['label']==1).sum()}, "
          f"Rate={feat_df['label'].mean()*100:.1f}%")
    return feat_df, skipped, skip_details


# ── Domain stats ───────────────────────────────────────────────────────────────

def build_domain_stats(feat_df, strict_rate_col="hall_rate_merged",
                        domain_col="domain", label_col="label"):
    """Summary table: hallucination rate, entropy, question count per domain."""
    ds = feat_df.groupby(domain_col).agg(
        n_questions=(label_col, "count"),
        n_hallucinated=(label_col, "sum"),
        hall_rate_mean=(strict_rate_col, "mean"),
        hall_rate_std=(strict_rate_col, "std"),
        mean_entropy=("H_sem", "mean"),
    ).sort_values("hall_rate_mean", ascending=False)
    ds["pct_hallucinated"] = (ds["n_hallucinated"] / ds["n_questions"] * 100).round(1)
    return ds


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL GRAPH FEATURES
# ══════════════════════════════════════════════════════════════════════════════
#
# Graph: W_jk = max(S_jk, 0), W_jj = 0.  Laplacian L = D - W.
# Eigendecomposition: 0 = lam1 <= lam2 <= ... <= lam_N.
# lam2 (Fiedler value) = algebraic connectivity.

SPECTRAL_FEATURES = [
    "lam2", "lam3", "SGR", "spectral_entropy", "ipr_fiedler", "HFER",
]
SPECTRAL_NICE_NAMES = {
    "lam2":             "Fiedler Value (lam2)",
    "lam3":             "Third Eigenvalue (lam3)",
    "SGR":              "Spectral Gap Ratio (lam2/lam3)",
    "spectral_entropy": "Spectral Entropy",
    "ipr_fiedler":      "IPR Fiedler Vector",
    "HFER":             "HFER (lam2 x Fiedler Entropy)",
}

EXTENDED_CLUSTER_FEATURES = [
    "largest_cluster_frac", "second_largest_cluster_frac", "singleton_cluster_frac",
]
EXTENDED_CLUSTER_NICE_NAMES = {
    "largest_cluster_frac":        "Largest Cluster Fraction",
    "second_largest_cluster_frac": "Second-Largest Cluster Fraction",
    "singleton_cluster_frac":      "Singleton Cluster Fraction",
}

ALL_NEW_FEATURES   = SPECTRAL_FEATURES + EXTENDED_CLUSTER_FEATURES
ALL_NEW_NICE_NAMES = {**SPECTRAL_NICE_NAMES, **EXTENDED_CLUSTER_NICE_NAMES}


def spectral_graph_features_single(S_i):
    """
    Spectral features from an N x N cosine similarity matrix:
      lam2, lam3, SGR, spectral_entropy, ipr_fiedler, HFER.
    """
    W = np.clip(S_i, 0, None).copy()
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(axis=1)) - W
    evals, evecs = np.linalg.eigh(L)

    lam2, lam3 = float(evals[1]), float(evals[2])
    sgr = lam2 / (lam3 + 1e-8)

    # Spectral entropy over positive eigenvalues
    q = evals / (evals.sum() + 1e-12)
    q = q[q > 0]
    h_spec = float(-(q * np.log2(q)).sum())

    # Fiedler vector properties
    v2 = evecs[:, 1]
    ipr = float(np.sum(v2 ** 4))
    p = np.abs(v2)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    h_fiedler = float(-(p * np.log2(p)).sum())

    return {
        "lam2": lam2, "lam3": lam3, "SGR": float(sgr),
        "spectral_entropy": h_spec, "ipr_fiedler": ipr,
        "HFER": lam2 * h_fiedler,
    }


def extended_cluster_features_single(cluster_labels, n_responses):
    """Cluster structure: largest/second-largest fraction, singleton fraction."""
    counts = np.bincount(cluster_labels)
    probs = np.sort(counts / counts.sum())[::-1]
    return {
        "largest_cluster_frac":        float(probs[0]),
        "second_largest_cluster_frac": float(probs[1]) if len(probs) > 1 else 0.0,
        "singleton_cluster_frac":      float((counts == 1).sum() / n_responses),
    }


def extract_spectral_features(df, all_embeddings, sim_threshold=0.85,
                               domain_col="domain", verbose=True):
    """
    Compute spectral + extended cluster features for every question.
    Returns DataFrame indexed by prompt_id.
    """
    prompt_ids = df["prompt_id"].unique()
    sizes = df.groupby("prompt_id").size()
    uniform = (sizes == sizes.iloc[0]).all()

    # Vectorised cosine if possible
    if uniform:
        n_resp = int(sizes.iloc[0])
        pid_to_idx = {pid: i for i, pid in enumerate(prompt_ids)}
        d = all_embeddings.shape[1]
        E_batch = np.zeros((len(prompt_ids), n_resp, d), dtype=all_embeddings.dtype)
        counters = np.zeros(len(prompt_ids), dtype=int)
        for row_i in range(len(df)):
            qi = pid_to_idx[df["prompt_id"].iloc[row_i]]
            E_batch[qi, counters[qi]] = all_embeddings[row_i]
            counters[qi] += 1
        S_batch = _batch_cosine_matrices(E_batch)
        if verbose:
            print(f"Vectorised path: {len(prompt_ids)} q x {n_resp} resp")
    else:
        S_batch, pid_to_idx = None, {}
        if verbose:
            print("Scalar path (unequal response counts)")

    records = []
    for i, pid in enumerate(prompt_ids):
        if verbose and (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(prompt_ids)} ...")

        mask = df["prompt_id"] == pid
        embs = all_embeddings[np.where(mask.values)[0]]
        n = len(embs)

        S_i = S_batch[pid_to_idx[pid]] if uniform else cosine_similarity(embs)

        spec = spectral_graph_features_single(S_i)

        # Cluster features
        dist = np.clip(1.0 - S_i, 0, 2)
        np.fill_diagonal(dist, 0)
        try:
            clust = AgglomerativeClustering(
                n_clusters=None, metric="precomputed",
                linkage="average", distance_threshold=1 - sim_threshold)
        except TypeError:
            clust = AgglomerativeClustering(
                n_clusters=None, affinity="precomputed",
                linkage="average", distance_threshold=1 - sim_threshold)
        labels = clust.fit_predict(dist)
        ext = extended_cluster_features_single(labels, n)

        rec = {"prompt_id": pid, "dataset": df[mask].iloc[0]["dataset"]}
        rec.update(spec)
        rec.update(ext)
        records.append(rec)

    result = pd.DataFrame(records).set_index("prompt_id")
    if verbose:
        print(f"\nSpectral features: {len(result)} questions.")
    return result
