# spring-2026-LLM-hallucinations

Team project: spring-2026-LLM-hallucinations

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
│   │
│   ├── metrics/
│   │   └── semantic.py          # Calculation of metrics from samples
│   │
│   └── training.py              # ML training routines
│
├── notebooks/ 
│   ├── 01_data_generation.ipynb # First notebook for data generation in colab
│   ├── 02_eda.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 03_statistics.ipynb
│   ├── 04_training.ipynb
│   ├── 05_visualization.ipynb
│
├── requirements.txt
└── README.md


# Hallucination Detection via Spectral Features of Attention Maps

This project studies whether **Laplacian eigenvalues of attention maps** can be used to detect hallucinated responses from an LLM. Starting from token-level attention matrices, we extract graph-based features from every attention head and layer, compress them with PCA, and train lightweight classifiers to distinguish **correct** from **hallucinated** answers. The workflow follows a progression from raw signal analysis, to predictive modeling, to statistical validation.

## Feature Engineering

For each layer $l$ and head $h$, the attention map $A^{(l,h)}$ is viewed as a directed graph, and its graph Laplacian is defined by
$$
L^{(l,h)} = D^{(l,h)} - A^{(l,h)}.
$$
Because the Laplacian is lower triangular in this setup, its eigenvalues lie on the diagonal and are bounded in $[-1,1]$. From each layer-head block, we retain the **top 10 eigenvalues**, producing a large raw feature vector per example. With $32$ layers and all heads included, this yields roughly **10,320 raw spectral features per question**.

These raw features are high-dimensional and highly correlated, so we apply **Principal Component Analysis (PCA)** as the central feature-engineering step. PCA serves two purposes: it reduces dimensionality and it denoises the feature space by concentrating most of the useful variance into a smaller number of orthogonal components. In the main pipeline, the full spectral vector is reduced to **384 PCA dimensions**, while some visualization and layer-wise analyses use smaller PCA projections such as 64 dimensions. This makes downstream training faster and more stable without discarding the dominant structure in the eigenvalue features.

## Training Pipeline

The primary supervised pipeline is:

1. standardize the raw spectral features,  
2. apply PCA,  
3. train a binary classifier on the reduced representation.

The baseline probe in the project document is **logistic regression on PCA-projected eigenvalue features**, which serves as a strong and interpretable reference model. In later experiments on the combined **TriviaQA + MMLU** dataset, we also compare several alternative classifiers, including **Linear SVM + calibration**, **Random Forest**, **AdaBoost**, and **SGDClassifier**. This lets us test whether the signal is primarily linear or whether it benefits from more complex nonlinear decision boundaries.

Empirically, the best-performing models on the combined dataset are the **linear methods**, with logistic regression achieving the strongest AUROC and linear SVM performing very similarly. This suggests that PCA is not just a compression step: it also helps organize the spectral features into a representation where a relatively simple linear boundary already captures most of the useful label signal.

## Visualizations

The project uses five main visualizations plus a bootstrap test:

- **V1: Eigenvalue KDE by rank.**  
  Compares the distribution of each eigenvalue rank for correct versus hallucinated examples. This checks whether the raw signal is already visible before any classifier is trained. Cohen’s $d$ is used as a simple effect-size summary.

- **V2: Layer × head p-value heatmap.**  
  For each layer-head block, a two-sided Mann–Whitney test compares the mean top-10 eigenvalue between the two labels. The percentage of heads with $p<0.05$ acts as a localization metric for where the signal is strongest in the network.

- **V3: Layer-wise AUROC profile.**  
  Trains a separate probe on each individual layer and compares it with the all-layer model. This reveals whether the signal is concentrated in a few layers or distributed across the network. The expected pattern is that deeper layers perform better, but the all-layer model still wins overall.

- **V4: ROC comparison across datasets.**  
  Compares the Laplacian-eigenvalue pipeline against baselines and ablations across datasets. This is the main diagnostic for predictive performance and robustness of the feature set.

- **V5: Cross-dataset generalization heatmap.**  
  Trains on one dataset and tests on another, producing a matrix of AUROC values. Strong off-diagonal performance indicates that the spectral signal transfers across benchmarks instead of depending on dataset-specific quirks. The project document notes that **MMLU** and **TriviaQA** are expected to transfer well to **NQ-Open**, while **TruthfulQA** may behave as an outlier because of its adversarial phrasing.

- **Bootstrap statistical validation.**  
  Independently of the classifier, the project tests whether hallucinated samples have a higher mean leading Laplacian eigenvalue $\lambda_{\max}$. The bootstrap computes the observed difference
  $$
  \delta_{\mathrm{obs}}=\overline{\lambda_{\max}}(y=1)-\overline{\lambda_{\max}}(y=0),
  $$
  and estimates a one-sided p-value and 95% confidence interval by resampling with replacement. This provides formal inferential support for the raw spectral hypothesis without assuming Gaussianity.

## Final Results and Main Takeaways

A key empirical result is that the **combined TriviaQA + MMLU dataset** performs better than either dataset alone and yields the **best AUROC observed so far**. This suggests that the two datasets contribute complementary information and that the spectral signature of hallucination becomes clearer when the training set is larger and more diverse.

Across classifier comparisons on the merged dataset, **Logistic Regression + PCA** achieves the strongest AUROC, while **Linear SVM + Calibration** performs similarly and slightly edges it in test accuracy. Tree-based ensemble models such as Random Forest and AdaBoost perform worse, which points to an important conclusion: after PCA, the hallucination signal appears to be **well captured by a low-dimensional, mostly linear representation**. In other words, PCA is doing much of the heavy lifting by transforming the raw topological features into a cleaner feature space where simple linear probes are already effective.

Overall, the project supports the following story: **attention-map Laplacian eigenvalues contain a measurable and transferable signature of hallucination**, PCA turns this large and noisy spectral representation into a compact and trainable feature space, and simple linear models on top of PCA are strong enough to achieve competitive hallucination detection performance. The bootstrap results strengthen this interpretation by showing that the leading eigenvalue itself carries statistically meaningful class information.