# spectral_detection/data/loaders.py
from datasets import load_dataset
import re
import json

def extract_boxed_solution(solution_str):
    match = re.search(r"\\boxed\{(.+?)\}", str(solution_str))
    return match.group(1) if match else solution_str

def load_truthfulqa():
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    return [{"question": r["question"], 
             "reference_answer": r["best_answer"], 
             "dataset": "truthfulqa"} for r in ds]

def load_triviaqa(sample_size=2000):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    return [{"question": r["question"], 
             "reference_answer": r["answer"]["value"], 
             "dataset": "triviaqa"} for r in ds]

def load_defan(ds):
    return [{"question" : r["questions"],
             "reference_answer" : r["answer"],
             "type" : r["type"],
             "domain" : r["domain"],
             "dataset" : "defan"} for r in ds]

def load_mmlu(sample_size=2000):
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    return [{
        "question": r["question"], 
        "choices": r["choices"], 
        "reference_answer": idx_to_letter[r["answer"]], 
        "dataset": "mmlu"
    } for r in ds]

def load_math(sample_size=2000):
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    return [{
        "question": r["problem"], 
        "reference_answer": extract_boxed_solution(r["solution"]), 
        "dataset": "math"
    } for r in ds]

def load_json_file(target_jsonl_path):
    """Parses a .jsonl file, queries the JSON API, and rewrites the matrix."""
    with open(target_jsonl_path, "r") as file_stream:
        records = [json.loads(line) for line in file_stream]
    return records