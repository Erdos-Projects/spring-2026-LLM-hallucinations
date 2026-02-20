# spring-2026-LLM-hallucinations

Team project: spring-2026-LLM-hallucinations

## Folder Structure

```text
spring-2026-llm_hallucinations-project/
│
├── data/                        # Data (e.g. loaded from Hugging Face)
│   ├── raw/                     # Raw .jsonl files (from the Benchmarks)
│   ├── processed/               # Cleaned files and .npz features (e.g. Attention heads)
│   └── .gitignore               # Avoid committing temporary datasets.
│
├── spectral_detection/          # Main Python package
│   ├── config.py                # Configuration (macros, paths, constants)
│   │
│   ├── data/
│   │   ├── loaders.py           # Dataset loaders
│   │   └── cleaning.py          # Data cleaning
│   │
│   ├── analysis/
│   │   ├── stats.py             # Statistics routines (pre and post-processing)
│   │   └── eda.py               # EDA
│   │
│   ├── data_generation.py       # model inference for creation of data => LLM and LLM-as-judge
│   │
│   ├── spectral_comp/
│   │   ├── lapEval.py           # Calculation of Top Laplacian eigenvalues
│   │   └── noelEval.py          # Calculation of Noel's Spectral Geometry measures
│   │
│   └── training.py              # ML training
│
├── notebooks/ 
│   ├── 01_data_generation.ipynb # First notebook for EDA
│   ├── 02_eda.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 03_statistics.ipynb
│   ├── 04_training.ipynb
│   ├── 05_visualization.ipynb
│
├── requirements.txt
└── README.md