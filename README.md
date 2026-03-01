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