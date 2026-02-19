# spectral_detection/data/loaders.py
from datasets import load_dataset
import re # regex

def extract_boxed_solution(solution_str):
    """
    Helper function for the MATH dataset: extracts content inside {boxed}
    """
    match = re.search(r"\\boxed\{(.+?)\}", solution_str)
    return match.group(1) if match else solution_str

def load_truthfulqa():
    """
    Loads TruthfulQA.
    """
    print("Loading TruthfulQA from Hugging Face...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    # Convert to our standard list-of-dictionaries format
    data = []
    for row in ds:
        data.append({
            "question": row["question"],
            "reference_answer": row["best_answer"],
            "dataset": "truthfulqa",
            "subject": row["category"], # classify by subject
            "aliases": [] # This dataset does not include aliases such as 
                          # (e.g., "USA", "United States", "U.S.")
        })
    return data

def load_triviaqa(n=2000):
    """
    Loads TriviaQA (rc.nocontext).
    We shuffle and pick a random 'n' samples, given that original size is much larger
    """
    print(f"Loading TriviaQA (sampling {n})...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    
    data = []
    for row in ds:
        data.append({
            "question": row["question"],
            "reference_answer": row["answer"]["value"],
            # TriviaQA provides aliases (e.g., "USA", "United States", "U.S.")
            # This helps the Judge be more accurate.
            "aliases": row["answer"]["aliases"], 
            "dataset": "triviaqa",
            "subject": "general_knowledge"
        })
    return data

def load_mmlu(n=2000):
    """
    MMLU stores answers as integers (0,1,2,3) and must be converted to A, B, C, D.
    """
    print(f"Loading MMLU (sampling {n})...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    
    # Mapping for the multiple choice answer
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    data = []
    for row in ds:
        data.append({
            "question": row["question"],
            "choices": row["choices"], # We need the choices to build the prompt
            "reference_answer": idx_to_letter[row["answer"]],
            "dataset": "mmlu",
            "subject": row["subject"],
            "aliases": []
        })
    return data

def load_math(n=2000):
    """
    NOTE: The original 'hendrycks/competition_math' is often locked/DMCA'd.
    We use 'DigitalLearningGmbH/MATH-lighteval' as a reliable mirror.
    # Check also https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval
    """
    print(f"Loading MATH (sampling {n})...")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    
    data = []
    for row in ds:
        data.append({
            "question": row["problem"],
            "reference_answer": extract_boxed_solution(row["solution"]),
            "dataset": "math",
            "subject": row["type"],
            "level": row["level"], # Level 1 through 5, where Level 5 is the hardest
            "aliases": []
        })
    return data