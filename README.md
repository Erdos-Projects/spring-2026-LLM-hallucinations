# spring-2026-LLM-hallucinations

# Spectral Detection — LLM Hallucination via Semantic Entropy, Embedding Geometry, and Attention Heads Spectrum.

## Project Overview

Part I:

Original work, for detecting LLM hallucinations using semantic entropy (Farquhar et al., 2024) and embedding
geometry (Ricco et al., 2025; Lee et al., 2018).

All responses generated with **Llama-3.2-3B** (Colab H100, ~2 hours), labelled by **GPT-4.1-nano** (CPU runtime, ~20h in total).

Part II:

Detects LLM hallucinations using the top-k eigenvalues of the Laplacian of the Attention Heads (Binkoswki et. al 2025), and analyzes the layer wise AUC.

All responses generated with **Llama-3.2-3B** (Colab H100, ~40–60 min), labelled by **GPT-4.1-nano** (CPU runtime, ~1.5–3 h), with nuclear sampling.  Llama allows extraction of the attention heads for the last token of the decoder's step.



## Folder Structure

```text
spring-2026-llm_hallucinations-project/
│
├── data/                        # Data
│   ├── raw/                     # Raw .jsonl files (from Benchmarks)
│   ├── processed/               # Processed (judged) files and binary files of eigenvalues of Attention heads
│   └── .gitignore               # Avoid committing some temporary datasets.
│
├── spectral_detection/          # Main Python package
│   ├── config.py                # Configuration (macros, paths, constants)
│   │
│   ├── data/
│   │   ├── loaders.py           # Dataset loaders
│   │   ├── features.py          # Calculation of Top Laplacian eigenvalues from attention heads
│   │   └── cleaning.py          # Data cleaning
│   │
│   ├── analysis/
│   │   ├── stats.py             # Statistics routines (pre or post-processing)
│   │   └── eda.py               # EDA
│   │
│   ├── data_generation.py       # model inference for creation of data => LLM and LLM-as-judge
│   ├── feature_extraction.py    # embed_responses, geometric features
│   ├── visualization.py         # All plot functions
│   └── training.py              # ML training routines, ablation
│
├── notebooks/ 
│   ├── 01_data_generation.ipynb # First notebook for data generation in colab
│   ├── 02_entropy_eda.ipynb
│   ├── 03_entropy_feature_extraction.ipynb
│   ├── 04_entropy_statistics.ipynb
│   ├── 05_entropy_training.ipynb
│   ├── 05_entropy_visualization.ipynb
│   ├── 05_entropy_visualization_partII.ipynb
│   ├── 06_spectral_entropy_advanced.ipynb
|   |
│   └── AttentionHeads
│   │   ├── .py
│   │   └── .py
│
├── requirements.txt
└── README.md
```

---

### Datasets

| Dataset | Questions | Notes |
|---------|-----------|-------|
| DefAn | 500 | Definitional QA; unique `type` field; no canonical domain needed |
| HaluEval | 500 | Curated hallucination pairs; 30 raw domains |
| MMLU | 500 | 57-subject benchmark; 65 raw domains → canonical |
| TriviaQA | 500 | Trivia reading-comprehension; 64 raw domains |
| TruthfulQA | 500 | Adversarial misconception questions; 90 raw domains |

---

### Six Geometric Features (Baseline Pipeline)
 
| Symbol | Name | Description | Reference |
|--------|------|-------------|-----------|
| H_sem | Semantic Entropy | Shannon entropy over semantic clusters (τ=0.85) | Farquhar et al. (2024) |
| D_cos | Cosine Dispersion (mean centroid) | Mean distance from each embedding to the centroid | Ricco et al. (2025) |
| D_cos_var | Cosine Dispersion (variance centroid) | Variance of per-response centroid distances; captures asymmetric scatter | - |
| D_pair | Mean Pairwise Cosine Distance | Mean of (1 − S_jk) across all response pairs; complements D_cos | - |
| K | Cluster Count | Number of agglomerative clusters | — |
| sig2_S | Similarity Variance | Variance of pairwise cosine similarities | — |

---

### Spectral Graph Features (`spectral_graph.ipynb`)

An extended feature set built on the graph Laplacian **L = D − W**, where W is
the N×N cosine similarity matrix (diagonal zeroed, negatives clipped).
Eigendecomposition gives 0 = λ₁ ≤ λ₂ ≤ λ₃ ≤ … ≤ λ_N.

### Group A — Laplacian spectrum (6 features)

| Symbol | Description |
|--------|-------------|
| lam2 | Fiedler value λ₂ — algebraic connectivity; low = responses nearly disconnect |
| lam3 | Third eigenvalue λ₃ |
| SGR | Spectral Gap Ratio λ₂/(λ₃+ε) — clean bipartition signal |
| spectral_entropy | Shannon entropy over normalised eigenvalue distribution |
| ipr_fiedler | Inverse Participation Ratio of Fiedler vector v₂ |
| HFER | λ₂ × Fiedler entropy — combines connectivity with partition geometry |

### Group B — Extended cluster structure (4 features)

| Symbol | Description |
|--------|-------------|
| largest_cluster_frac | Fraction of responses in the dominant cluster (p₁) |
| second_largest_cluster_frac | Fraction in the runner-up cluster (p₂) |
| singleton_cluster_frac | Fraction of singleton clusters — strong incoherence signal |

---

### Domain validity
Domains are LLM-assigned tags (noisy and inconsistent).
All inferential statistics run at **benchmark level** (≈500 q) and **combined level** (2500 q).
Domain results are exploratory only, and will not be used for training.

### Ablation variants (no leakage)
1. Entropy only (H_sem)
2. Geometry only (D_cos, D_cos_var, M_bar)
3. Entropy + Geometry
4. All 7 geometric

## References

- Binkowski J et. al. 2025. Hallucination detection in llms using spectral features of attention maps. *Conference on Empirical Methods in Natural Language Processing*.
- Farquhar et al. (2024). Detecting hallucinations in LLMs using semantic entropy. *Nature*.
- Ricco et al. (2025). Hallucination detection: A probabilistic framework using embedding distance. *arXiv*.
- Lee et al. (2018). A simple unified framework for detecting OOD samples. *NeurIPS*.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*.

---

## Run Order Instructions

```bash
# 1. EDA:  notebooks/02_eda.ipynb

# 2. Embed + extract features:  notebooks/03_feature_extraction.ipynb

# 3. Statistical tests:  notebooks/04_statistics.ipynb

# 4. Training, ROC, SHAP:  notebooks/05_training.ipynb

# 5.  Visualization part I and II:  notebooks/05_visualization, notebooks/05_visualization_partII
```




