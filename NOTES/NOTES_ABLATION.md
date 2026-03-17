### Complete picture. 

Two separate ablation studies, run in two different notebooks.

---

### `training.ipynb` — Baseline ablation (9 GEO_FEATURES)

Runs on each individual benchmark + combined dataset, 5 classifiers × 4 variants:

| Variant | Features | Scientific question |
|---------|----------|-------------------|
| Entropy only | H_sem | Does semantic entropy alone detect hallucinations? (Farquhar et al. claim) |
| Geometry only | D_cos, D_cos_var, D_pair, M_bar | Can pure embedding geometry beat entropy |
| Entropy + Geometry | H_sem + the 6 geometry features | Does combining entropy with geometry improve over either alone? |
| All 7 geometric | All 7 baseline features | Does adding K and σ²_S on top of the above add signal |

---

### `spectral_graph.ipynb` — Extended ablation (18 features, combined dataset only)

Runs on the combined 2500-question dataset, 5 classifiers × 4 variants:

| Variant | Features | Scientific question |
|---------|----------|-------------------|
| Spectral graph only | lam2, lam3, SGR, spectral_entropy, ipr_fiedler, HFER | Testing how graph Laplacian spectrum alone detects hallucinations |
| Full 16 features | All 7 baseline + all 9 new | Does adding spectral features improve our existing pipeline |
| Most significant | Built at runtime from KS-significant features + top correlated baseline features | Feature selection: does a lean set of the strongest features match or beat the full set |

---

### Classifiers

| Classifier | Type |
|-----------|------|
| Logistic Regression | L2-regularised, baseline |
| ElasticNet Logit | SAGA solver, L1+L2 (l1_ratio=0.5), implicit feature selection |
| Random Forest | 200 trees, max_depth=6 |
| Gradient Boosting | sklearn GBM |
| XGBoost | 200 trees, max_depth=4 |