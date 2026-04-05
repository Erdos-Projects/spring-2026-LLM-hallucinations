## Spring-2026-LLM-hallucinations


# LLM Hallucination Detector

<img src="/images/robot_detect.png" alt="Description" style="display: block; margin: 0 auto;"/>

### LLM Hallucination via Attention Heads Spectrum, Semantic Entropy, and Embedding Geometry

# Project members:

[Helmut Wahanik](https://github.com/hwahanik),
[Santanil Jana](https://github.com/sjana01),
[Debanjan Sarkar](https://github.com/debanjan-cosmo),
[AJ Vargas](https://github.com/ajv7674-lgtm),
[Guoqin Liu](https://github.com/hellolgq)

## Mini-Abstract

Hallucinations commonly occur when a Large Language Model (LLM) faces unexpected or rare prompts. While rapid advances in LLMs followed the introduction of the self-attention mechanism (in the seminal paper “Attention Is All You Need”), the underlying mechanics of neural network inference remain opaque. Because these models often operate as a “black box”, active research is dedicated to shedding light on exactly how an LLM processes information. 

*This leads to our first question:*

`Question 1:`  Can we observe some natural correlation between hallucinations produced by an LLM and the numerical data that propagates through it? 

For answering it, our project runs a **Data Science Neurosurgery on the Transformer’s Attention heads**, and trains the top-k eigenvalues of the Attention-Map Laplacian against the label “hallucination or not”.  AI creativity and power comes from random draws.  LLMs process the information by inferring the next “token” of a response through probability sampling, at a certain “temperature” $T$.  Low temp $T\sim 0$ $\rightarrow$ always picks tokens with highest probability, making the choice “deterministic.  High temp implies a more “generative” and “creative” process.

*As a consequence we make the next question:*

`Question 2:`  Can we predict hallucinations by analyzing exclusively the geometric and spectral properties of probability distributions built during the Generative process of an LLM? 

Indeed, to answer this we note that there exists a natural mapping between prompts and the space of distribution of responses.  In the second part of our project we convert a variable-length set of text responses into a fixed-size **numeric feature vector** that captures whether the model is hallucinating.  For each question $q_{i}$ in our dataset, we sample $N = 20$ responses. Each response is a string of text. We embed each response into a vector $e_{ij} \in \mathbb{R}^{384}$ (using a sentence transformer). At this point, we have $N=20$ vectors per question, but we do not use these 20 vectors as 20 separate rows in our dataset. Instead, we aggregate them into a single feature vector that summarises the distribution of responses for that question.  

Our work introduces a feature vector that summarizes the distribution of responses, and is composed of `semantic` and `geometric` metrics as those introduced by Farquhar et. al. 2024, Ricco et al., 2025, Zhao et al., 2025, Lee et al., 2018, such as the following:

1. Semantic entropy $H_{sem}$: How many distinct meanings appear in the 20 responses? 

2. Cosine dispersion $D_{cos}$ and pair-wise mean $D_{pair}$: How spread out are the 20 embeddings around their centre?  How heavily do they interact with each other?  

3. Cluster count $K$: How many semantic clusters do the 20 responses form?

4. Similarity variance $\sigma^{2}_{S}$: How uneven is the agreement between response pairs?

We go a step further and introduce additional features corresponding to the `graph connectivity` of the distribution of responses, indicating the degree of connectivity of the distribution of responses.  See also `09_geo_spectral_graph.ipynb`

### Embeddings and t-SNE

Semantic meaning captures the contextual intent of language independent of its literal syntax. Text embeddings quantify this meaning by mapping natural language into continuous, high-dimensional vector spaces.  In our work, we use `all-MiniLM-L6-v2`, an optimized Sentence Transformer available in Huggingface, that maps the questions to the space $\mathbb{R}^{384}$.  The relationships between phrases is measured by the cosine similarity ($\frac{s_1 \cdot s_2} {| s_1 | | s_2 |} $), which is also one of the basic constituent of our geometric training features.  

Hallucinations clusters unevenly spread meanings in $\mathbb{R}^{384}$ and therefore have elongated Convex Hulls!   We train a t-SNE, which acts through the minimization of the KL divergence "replicating" high-dimensional neighborhoods into lower dimensional spaces, as seen below.  

<figure>
<img src="/images/tSNEprojection.png" alt="Description" style="display: block; margin: 0 auto;">
</figure>

*In the figure, we map the 10,000 responses of the MMLU benchmark (20 responses for all 500 chosen questions) and highlight 3 response clouds graded by different response "certainty".  Observe the dramatic Convex Hull for a Hallucination!*

### Data Collection Pipeline:

Generative training data was collected and computed using diverse benchmarks (shown in the table below).  Each question was reformatted to fit a Python dictionary of the form:  

```bash
[“question”, “reference_answer”, “choices”], 
```

and passed to our LLM of choice, Llama-3.2-3B (quantized to 4bits).  Llama is forced to answer at a temperature of 1.0 (“to be more creative”).  The LLM’s response is collected in a standardized `JSON` file.  For answering `Question 1`, we simultaneously extract the Attention Heads matrices (stored temporarily in VRAM while running in Colab on a A100 GPU) to calculate the top-k eigenvalues which we encode in binary files.  For `Question 2` we generate $N=20$ responses for each prompt.  The responses are evaluated by a “more powerful” LLM-Judge.  Our judge is a “lightweight” but strong LLM, such as GPT-Mini-4o or GPT-4.1-nano;  we use standardized prompting (please refer to `config.py`) in order to “instruct” the judge to also answer under pre-defined formats.  Multi-sampled responses are cleaned and filtered, and embedded into a the high dimensional space $\mathbb{R}^{384}$ using pre-trained Sentence Transfomers, and aggregated as features as described in `notebooks/06_geo_feature.ipynb`.  The data collection workflow is shown below:

<img src="/images/data_collection.png" alt="Description" width="1000" />

## Data Computing:

For eigenvalue extraction, responses were generated with **Llama-3.2-3B**, running in Colab in a GPU A100, and labelled by **GPT-4.1-nano**, using nuclear sampling at $\tau=0.85$.  Llama allows extraction of the attention head maps during the last token's decoding.

---
## `Part I: Hallucination Detection via Spectral Features of Attention Maps`

In Part I of our project we study whether **Laplacian eigenvalues of attention maps** can be used to detect hallucinated responses from an LLM. Starting from token-level attention matrices, we extract graph-based features from every attention head and layer, compress them with PCA, and train lightweight classifiers to distinguish **correct** from **hallucinated** answers. The workflow follows a progression from raw signal analysis, to predictive modeling, to statistical validation.

### Feature Engineering

For each layer $l$ and head $h$, the attention map $A^{(l,h)}$ is viewed as a directed graph, and its graph Laplacian is defined by
$L^{(l,h)} = D^{(l,h)} - A^{(l,h)}$.
Because the Laplacian is lower triangular given that we extract the Attentions at the decoder level, the eigenvalues lie on the diagonal and are bounded in $[-1,1]$. From each layer-head block, we retain the **top 10 eigenvalues**, producing a large raw feature vector per example. With $32$ layers and all heads included, this yields roughly **6720 raw spectral features per question**!

These raw features are high-dimensional and highly correlated, so we apply **Principal Component Analysis (PCA)** as the central feature-engineering step. PCA serves two purposes: it reduces dimensionality and it denoises the feature space by concentrating most of the useful variance into a smaller number of orthogonal components. In the main pipeline, the full spectral vector is reduced to **384 PCA dimensions**, while some visualization and layer-wise analyses use smaller PCA projections such as 64 dimensions. This makes downstream training faster and more stable without discarding the dominant structure in the eigenvalue features.

<img src="/images/scree_plot.png" alt="Description" width="800" />

As seen in the Scree plot, *most of the variance is retained by the first few principal components*.

### Training Pipeline

The primary supervised pipeline is:

1. Standardize the raw spectral features,  
2. Apply PCA,  
3. Train a binary classifier on the reduced representation.

The baseline probe in the project document is **logistic regression on PCA-projected eigenvalue features**, which serves as a strong and interpretable reference model. In later experiments on the combined **TriviaQA + MMLU** dataset, we also compare several alternative classifiers, including **Linear SVM + calibration**, **Random Forest**, **AdaBoost**, and **SGDClassifier**. This lets us test whether the signal is primarily linear or whether it benefits from more complex nonlinear decision boundaries.

The test AUROC score are as follows.

<p float="left">
  <img src="/images/auroc_log_reg.png" alt="Description 1" width="600" />
  <img src="/images/auroc_dataset.png" alt="Description 2" width="600" />
</p>

The combined *TriviaQA* and *MMLU* dataset yields the strongest overall performance. Empirically, the best results are achieved by **linear models**, with **Logistic Regression** attaining the highest AUROC and **Linear SVM** performing nearly as well. This indicates that PCA is doing more than simple dimensionality reduction—it is organizing the spectral features into a space where the label signal is largely captured by a linear decision boundary. The comparatively weaker performance of tree-based ensemble methods further supports this, suggesting that after PCA, the data becomes largely linearly separable and does not require more complex nonlinear modeling.


### Visualizations

The project uses five main visualizations plus a bootstrap test:

- **V1: Eigenvalue KDE by rank.**  
  Compares the distribution of each eigenvalue rank for correct versus hallucinated examples. This checks whether the raw signal is already visible before any classifier is trained. Cohen’s $d$ is used as a simple effect-size summary.

  <img src="/images/eig_dist.png" alt="Description" width="800" />

  The strongest separation appears at low ranks, showing that leading eigenvalues already carry label signal. Hallucinated samples show broader and heavier-tailed distributions, consistent with more diffuse attention flow.

- **V2: Layer × head p-value heatmap.**  
  For each layer-head block, a two-sided Mann–Whitney test compares the mean top-10 eigenvalue between the two labels. The percentage of heads with $p<0.05$ acts as a localization metric for where the signal is strongest in the network. Hallucinated samples show broader and heavier-tailed distributions, consistent with more diffuse attention flow.

  <img src="/images/pvalue_heatmap.png" alt="Description" width="1000" />

  The pooled dataset has the highest share of heads with $p<0.05$. This means spectral differences between correct and hallucinated responses are most consistently detectable when datasets are combined.
  
- **V3: Layer-wise AUROC profile.**  
  Trains a separate probe on each individual layer and compares it with the all-layer model. This reveals whether the signal is concentrated in a few layers or distributed across the network. The expected pattern is that deeper layers perform better, but the all-layer model still wins overall.

  <img src="/images/layerwise_auroc.png" alt="Description" width="1000" />

  This plot answers two questions. First, it shows whether some layers are more informative than others. Second, by comparing the per-layer curve with the all-layer dashed line, it tests whether hallucination signal is localized to a single layer or distributed across many layers. If the dashed all-layer line remains above every individual layer peak, that supports the idea that the model benefits from combining information across the whole network.

- **V4: ROC comparison across datasets.**  
  Compares the Laplacian-eigenvalue pipeline against baselines and ablations across datasets. This is the main diagnostic for predictive performance and robustness of the feature set.

- **V5: Cross-dataset generalization heatmap.**  
  Trains on one dataset and tests on another, producing a matrix of AUROC values. Strong off-diagonal performance indicates that the spectral signal transfers across benchmarks instead of depending on dataset-specific quirks. The project document notes that **MMLU** and **TriviaQA** are expected to transfer well to **NQ-Open**, while **TruthfulQA** may behave as an outlier because of its adversarial phrasing.

- **Bootstrap statistical validation.**  
  Independently of the classifier, the project tests whether hallucinated samples have a higher mean leading Laplacian eigenvalue $\lambda_{1}$. The bootstrap computes the observed difference
  $\delta_{\mathrm{obs}}=\overline{\lambda_{1}}(y=1)-\overline{\lambda_{1}}(y=0)$,
  and estimates a one-sided $p$-value and 95% confidence interval by resampling with replacement. This provides formal inferential support for the raw spectral hypothesis without assuming Gaussianity.

  <img src="/images/bootstrap_halueval_hist.png" alt="Description" width="1000" />

### Final Results and Main Takeaways

A key empirical result is that the **combined TriviaQA + MMLU dataset** performs better than either dataset alone and yields the **best AUROC observed so far**. This suggests that the two datasets contribute complementary information and that the spectral signature of hallucination becomes clearer when the training set is larger and more diverse.

Across classifier comparisons on the merged dataset, **Logistic Regression + PCA** achieves the strongest AUROC, while **Linear SVM + Calibration** performs similarly and slightly edges it in test accuracy. Tree-based ensemble models such as Random Forest and AdaBoost perform worse, which points to an important conclusion: after PCA, the hallucination signal appears to be **well captured by a low-dimensional, mostly linear representation**. In other words, PCA is doing much of the heavy lifting by transforming the raw topological features into a cleaner feature space where simple linear probes are already effective.

Overall, the project supports the following story: **Attention-map Laplacian eigenvalues contain a measurable and transferable signature of hallucination**, PCA turns this large and noisy spectral representation into a compact and trainable feature space, and simple linear models on top of PCA are strong enough to achieve competitive hallucination detection performance. The bootstrap results strengthen this interpretation by showing that the leading eigenvalue itself carries statistically meaningful class information.  To productionize this tool, it is crucial
to obtain API access to the internal architecture of the Transformer during run-time.  

Due to constraints of using external vendors during Agentic RAG deployment, we acknowledged the need of a second tool for the `LLM Detector`, and the Semantic and Geometric workflow satisfies
these needs as shown next.

## `Part II: Assessing question-level hallucination risk from response cloud features`

In Part II of our project we study whether patterns in the distribution of an LLM's repeatedly sampled reponses to a given question can reliably determine whether the LLM is likely to give a hallucinated answer for that question. Our basic premise is that when an LLM is uncertain about a question, it will tend to exhibit higher semantic spread / fragmentation / instability across repeatedly sampled responses, which should lead to higher hallucination risk. We thus train our models to predict hallucination probability from features chosen precisely to capture semantic spread / fragmentation / instability within response clouds. (See below the "Folder Structure" and "Datasets" portion of this readme for the feature breakdown).

### Some remarks on class distribution across datasets

We found HaluEval was the ‘easiest’ dataset for the our LLM, as its prompts contained context; TriviaQA had the most balanced distribution of $\%$ correctness ($p_{hat}$) – “medium difficulty”; the rest were more heavily skewed to $\%$ incorrectness – “hard difficulty” – overall balanced class distribution across all 5 datasets.

For threshold correctness ($y$), MMLU was the most difficult: hardest for the LLM to get $>50\%$ in MMLU, but usually it got at least a few right in each response cluster.

Overall the distribution of correctness is bimodal across all datasets: most often the model LLM fails entirely, or is entirely correct in each response cloud.

<p align="center">
<img src="/images/boxplot.png" alt="Description" width="500" style="display: block; margin: 0 auto;"/>
</p>  

*Box plot graph of p_hat, the estimated probability of a hallucination, calculated from N=20 responses.*

<figure>
<img src="/images/kde_all.png" alt="Description" style="display: block; margin: 0 auto;"/>
</figure>

*KDE plots fitting distributions of Semantic and Geometric Features.  Many features display bi-modal behaviour, implying good separation between classes.*

We also evaluated the distribution of domains across the different datasets, but domain-information has been chosen as exclusively exploratory, given that such category shows spurious results when retrieved via judge-LLM responses.

<figure>
<img src="/images/triviarate.png" alt="Description" style="display: block; margin: 0 auto;"/>
</figure> 

*Distribution of domains vs hallucinations.  EDA only.*


<figure>
<img src="/images/tSNEmulti.png" alt="Description" style="display: block; margin: 0 auto;"/>
</figure> 

*Experiment running high-temp Llama-3.2-3B for 100-samples, upper row t-SNE 2D map showing that hallucinations agglomerative clustering number is evidently larger (right hand side), resulting in more clusters with less components.  The KDE density corroborates this (second row).  In the third row we can observe how the D_cos displays a clear fat tail, i.e. a clear higher frequency of weak links between responses.*


### Baseline training story and results

We carried out our training on the baseline feature set (/spectral-llm-hal/notebooks/05_training.ipynb) in essentially two regimes: 5 individual training runs on each 500 question/10,000 sample datasets individually, and one training run on the 2500 question/50,000 sample dataset aggregated from the 5 individuals. In each run we held out 20% of the data for a final test once the best performing model on training data was selected. Additionally, during each training run we carried out an ablation study to determine the best configuration of features for model performance.

Due to the relatively small dataset sizes and class imbalance within across datasets, we stratified the target variable in each train/test split, and we evaluated model performance on training data using 5-fold cross validation. Since our goal for this project was not perfect performance optimization, we opted not to include hyperparameter tuning to inform model selection. The four models we tested were baseline Logistic Regression, ElasticNet logit (saga solver + l1_ratio = 0.5), Random Forest, Gradient Boosting, and XGBoost.

We saw some variation in best feature configuration across the individual datasets, **with geometry features carrying the most predictive signal in the harder datasets**. As far as performance on the test sets, we found that all models struggled with the "difficult" datasets in isolation (though still performing well above chance, receiving AUC scores ranging from ~0.69-0.82 on the difficult sets). However, performance improved significantly when the models were trained on the aggregated dataset, receiving AUC score ~0.91 on the test set.

In the end we found that the **ElasticNet Logit trained on all 6 features** gave the best overall perfoamnce, showing that our baseline feature selection carries clean hallucination signals captured by simple logistic-based probes (with combined $l_{1}$ and $l_{2}$ penalties offcourse), aligning well with our initial premise.

We also tested feature importance and found the majority of hallucination detection signal was carried by the features $D_{cos}$ (semantic spread) and $H_{sem}$ (fragmentation measure), with $\sigma^{2}_{S}$ (instability measure) carrying relatively smaller but complementary signal.

<figure>
<img src="/images/combined_ablation.png" alt="Description" style="display: block; margin: 0 auto;"/>
</figure>

*Combined ablation results across 6 different models.*

<figure>
<img src="/images/combined_beeswarm.png" alt="Description" style="display: block; margin: 0 auto;"/>
</figure>

*Feature importance winners: H_sem (semantic entropy) and D_cos (cosine dispersion).  D_pair, (pairwise interaction) is highly correlated to the cosine dispersion and can be dropped having only a small performance drop.*


### Secondary feature set examination

In addition to our main 6-feature baseline discussed above, we wondered whether in place of dispersion and entropy we could find a different set of features that carried a similar predictive power. For this we looked to **response-graph spectral features** (see Group A below) and **extended cluster structure features** (Group B). We tested these alternate features (`08_geo_spectral_graph.ipynb`) and found that taken together, Group A + Group B features would indeed carry comparable predictive power, however when taken together with the baseline features, they did not contribute meaningful additional signal.

Being able to work with this secondary feature set may potentially be helpful in generalizations of our pipeline, where different LLMs, datasets, or higher sample sizes (etc) may be preferable.

### Statistical tests and EDA findings

We also performed statistical tests such as `KS Test`, to validate that distributions were distinct enough, `Permutation Test`  to determine whether the model has learned a genuine dependency between the features and the target labels (null hypothesis $H_{0}$: mean entropy is the same for hallucinated and correct questions), and `Bootstrap` to prove that the classifiers used for training our model reliably outperform chance.  See also `07_geo_statistics.ipynb` and figures below.

<div style="display: flex; justify-content: space-between;">
  <figure style="width: 50%;">
    <img src="/images/combined_permutation.png" alt="Alt text 1" style="width: 100%;">
  </figure>
  <figure style="width: 50%;">
    <img src="/images/combined_bootstrap.png" alt="Alt text 2" style="width: 100%;">
  </figure>
</div>

*(upper) Permutation test: Hallucinated questions have nearly 2 bits higher entropy. No permutation out of 10,000 matched this gap.  (lower) Bootstrap: the classifier reliably outperforms chance.*

### Impact on Large Scale AI Deployment

1. Our hallucination detector mitigates risk by catching severe AI errors very early. 
2. It can integrate seamlessly into any Agentic API, monitoring high-temperature sampling in real-time to trigger fact-checking, only when necessary. 
3. Because it is was optimized specifically for anomaly detection, it delivers the reliability of standard judge models at half the compute cost.
4. Ultimately, our novel system will help protect any company’s brand reputation, the safety and quality of it’s AI systems, while allowing it’s business to scale AI features profitably.

## Folder Structure

```text
spring-2026-llm_hallucinations-project/
│
├── data/                        # Data
│   ├── raw/                     # Some raw .jsonl files (from Benchmarks)
│   ├── processed/               # Processed (judged) files and other temp graphic files
│   ├── temp/                    # Computing temp files
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
│   │   ├── visual.py            # Visualizations for attention heads analysis
│   │   ├── pipeline.py          # Pipeline for attention heads analysis
│   │   ├── stats.py             # Statistics routines (pre or post-processing)
│   │   └── eda.py               # EDA for geometric properties
│   │
│   ├── data_generation.py       # model inference for creation of data => LLM and LLM-as-judge
│   ├── feature_extraction.py    # embed_responses, geometric features
│   ├── visualization.py         # All plot functions for geometric, semantic, and spectral features
│   └── training.py              # ML training routines, ablation
│
├── notebooks/ 
│   ├── 01_data_generation.ipynb                    # Data Computing
│   ├── 02_attention_datagen.ipynb
│   ├── 03_attention_feature_training.ipynb         # Part I
│   ├── 04_attention_visualization.ipynb        
│   ├── 05_geo_eda.ipynb                            # Part II
│   ├── 06_geo_feature_extraction.ipynb
│   ├── 07_geo_statistics.ipynb
│   ├── 08_geo_training.ipynb
│   ├── 09_geo_spectral_graph.ipynb
│   ├── 10_geo_visualization.ipynb
│   ├── 11_geo_visualization_partII.ipynb
│   ├── 12_geo_cluster_level_data_generation.ipynb  # Paraphrasing research
│   ├── 13_geo_cluster_judge.ipynb
│   └── 14_geo_cluster_vs_sample_analysis.ipynb
│
├── requirements.txt
└── README.md
```



### Datasets (Benchmarks in Hugging Face repo)

| Dataset | Questions | Notes |
|---------|-----------|-------|
| DefAn | 500 | Definitional QA; unique `type` field; no canonical domain needed |
| HaluEval | 500 | Curated hallucination pairs; 30 raw domains |
| MMLU | 500 | 57-subject benchmark; 65 raw domains → canonical |
| TriviaQA | 500 | Trivia reading-comprehension; 64 raw domains |
| TruthfulQA | 500 | Adversarial misconception questions; 90 raw domains |


### Six Geometric Features (Baseline Pipeline)
 
| Symbol | Name | Description | Reference |
|--------|------|-------------|-----------|
| H_sem | Semantic Entropy | Shannon entropy over semantic clusters | Farquhar et al. (2024) |
| D_cos | Cosine Dispersion (mean centroid) | Mean distance from each embedding to the centroid | Ricco et al. (2025) |
| D_cos_var | Cosine Dispersion (variance centroid) | Variance of per-response centroid distances; captures asymmetric scatter | - |
| D_pair | Mean Pairwise Cosine Distance | Mean of $(1 − S_{jk})$ across all response pairs; complements D_cos | - |
| K | Cluster Count | Number of agglomerative clusters | — |
| sig2_S | Similarity Variance | Variance of pairwise cosine similarities | — |


### Spectral Graph Features (`08_geo_spectral_graph.ipynb`)

An extended feature set built on the graph Laplacian **$L = D − W$**, where $W$ is
the $N\times N$ cosine similarity matrix (diagonal zeroed, negatives clipped).
Eigendecomposition gives $0 = \lambda_{1} \leq \lambda_{2} \leq \lambda_{3} \leq \ldots \leq \lambda_{N}$, and corresponding eigenvectors.

### Group A — Laplacian spectrum (6 features)

| Symbol | Description |
|--------|-------------|
| lam2 | Fiedler value $\lambda_{2}$ — algebraic connectivity; low = responses nearly disconnect |
| lam3 | Third eigenvalue $\lambda_{3}$ |
| SGR | Spectral Gap Ratio $\frac{\lambda_{2}}{(\lambda_{3}+\epsilon)}$: clean bipartition signal |
| spectral_entropy | Shannon entropy over normalised eigenvalue distribution |
| ipr_fiedler | Inverse Participation Ratio of Fiedler vector $v_2$ |
| HFER | $\lambda_{2} \times$ Fiedler entropy — combines connectivity with partition geometry |

### Group B — Extended cluster structure (4 features)

| Symbol | Description |
|--------|-------------|
| largest_cluster_frac | Fraction of responses in the dominant cluster ($p_{1}$) |
| second_largest_cluster_frac | Fraction in the runner-up cluster ($p_{2}$) |
| singleton_cluster_frac | Fraction of singleton clusters — strong incoherence signal |

## References

- Binkowski J et. al. 2025. Hallucination detection in llms using spectral features of attention maps. *Conference on Empirical Methods in Natural Language Processing*.
- Farquhar et al. (2024). Detecting hallucinations in LLMs using semantic entropy. *Nature*.
- Ricco et al. (2025). Hallucination detection: A probabilistic framework using embedding distance. *arXiv*.
- Lee et al. (2018). A simple unified framework for detecting OOD samples. *NeurIPS*.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Vaswani, A., et. al. (2017). Attention is all you need. Advances in neural information processing systems, 30.

---

## Run Order Instructions

```bash

# Attention heads detection

# Generating data for all the project:  01_data_generation.ipynb
# Data for attention matrices           02_attention_datagen.ipynb
# EDA and Training attention            03_attention_feature_training.ipynb
# Visualization attention               04_attention_visualization.ipynb

# Semantic and Geometric detection

# EDA Geometric and Semantic            05_geo_eda.ipynb
# Embed + extract features              06_geo_feature_extraction.ipynb
# Stats geometric                       07_geo_statistics.ipynb
# Training geometric                    08_geo_training.ipynb
# Spectral                              09_geo_spectral_graph.ipynb
# t-SNE visualizations                  10_geo_visualization.ipynb
# Advanced visualizations               11_geo_visualization_partII.ipynb

# Research on DeFan paraphrasing        12_geo_cluster_level_data_generation.ipynb  # Paraphrasing research
#                                       13_geo_cluster_judge.ipynb
#                                       14_geo_cluster_vs_sample_analysis.ipynb
```