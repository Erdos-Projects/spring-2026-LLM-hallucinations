# spectral_detection/data/loaders.py
from datasets import load_dataset
import re
import json

def extract_boxed_solution(solution_str):
    match = re.search(r"\\boxed\{(.+?)\}", str(solution_str))
    return match.group(1) if match else solution_str

def extract_gsm8k_solution(solution_str):
    """Extracts the final scalar answer separated by the standard #### delimiter."""
    if "####" in str(solution_str):
        return solution_str.split("####")[-1].strip()
    return solution_str

def load_gsm8k(sample_size=1319):
    """
    Retrieves the GSM8K benchmark. 
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    return [{
        "question": r["question"],
        "reference_answer": extract_gsm8k_solution(r["answer"]),
        "dataset": "gsm8k",
        "topic_label": "math" 
    } for r in ds]

def load_nq_open(sample_size=3610):
    """
    Retrieves the validation set of the NQ-Open (Natural Questions Open) dataset.
    """
    ds = load_dataset("nq_open", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    
    return [{
        "question": r["question"],
        "reference_answer": r["answer"], 
        "dataset": "nq_open",
        "topic_label": ""
    } for r in ds]

def load_truthfulqa(sample_size=817):
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    return [{
        "question": r["question"], 
        "reference_answer": r["best_answer"], 
        "dataset": "truthfulqa",
        "topic_label": r.get("category", "")
    } for r in ds]

def load_triviaqa(sample_size=2000):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    return [{
        "question": r["question"], 
        "reference_answer": r["answer"]["value"], 
        "dataset": "triviaqa",
        "topic_label": ""
    } for r in ds]

def load_defan(ds):
    return [{
        "question" : r["question"],
        "reference_answer" : r["answer"],
        "type" : r["type"],
        "dataset" : "defan",
        "topic_label" : r.get("domain", "")
    } for r in ds]

def load_mmlu(sample_size=2000):
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    return [{
        "question": r["question"], 
        "choices": r["choices"], 
        "reference_answer": idx_to_letter[r["answer"]], 
        "dataset": "mmlu",
        "topic_label": r.get("subject", "")
    } for r in ds]

def load_math(sample_size=2000):
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    return [{
        "question": r["problem"], 
        "reference_answer": extract_boxed_solution(r["solution"]), 
        "dataset": "math",
        "topic_label": r.get("type", "")
    } for r in ds]

def load_halueval(sample_size=2000):
    """
    Retrieve random sample of the HaluEval (QA) benchmark.
    """
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    
    return [{
        "question": f"Context: {r['knowledge']}\n\nQuestion: {r['question']}", 
        "reference_answer": r["right_answer"], 
        "dataset": "halueval",
        "topic_label": ""
    } for r in ds]

def load_json_file(target_jsonl_path):
    """Parses a .jsonl file, queries the JSON API, and rewrites the matrix."""
    with open(target_jsonl_path, "r") as file_stream:
        records = [json.loads(line) for line in file_stream]
    return records