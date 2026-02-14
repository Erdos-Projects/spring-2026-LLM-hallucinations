# spring-2026-LLM-hallucinations
Team project: spring-2026-LLM-hallucinations

# Folder Structure 

spectral-llm_hallucinations-project/
│
├── data/                        # Data (e.g. loaded from Hugging Face)
│   ├── raw/                     # Raw .jsonl files (from the Benchmarks)
│   ├── processed/               # Cleaned files and .npz features (Attention heads)
│   └── .gitignore               # Avoid committing temporary datasets.
│
├── spectral_detection/          # Main Python package
│   ├── __init__.py
│   ├── config.py                # Configuration (macros, paths, constants)
│   │
│   ├── data/                    
│   │   ├── __init__.py
│   │   ├── loaders.py           # Dataset loaders
│   │   └── cleaning.py          # Data cleaning
│   │
│   ├── analysis/                
│   │   ├── __init__.py
│   │   ├── stats.py             # Statistics routines (pre and post-processing)
│   │   └── eda.py               # EDA
│   │
│   ├── data_generation.py       # model serving => LLM and LLM-as-judge
│   │
│   ├── spectral_comp/                
│   │   ├── __init__.py
│   │   ├── lapEval.py           # Calculation of Top Laplacian eigenvalues
│   │   └── noelEval.py          # Calculation of Noel's Spectral Geometry measures 
│   │
│   └── training.py              # ML training
│
├── notebooks/                   
|   ├── 01_data_generation.ipynb      
│   ├── 02_eda.ipynb             
│   ├── 03_feature_extraction.ipynb 
│   ├── 03_statistics.ipynb      
│   ├── 04_training.ipynb        
│   ├── 05_visualization.ipynb      
│   
├── requirements.txt
└── README.md