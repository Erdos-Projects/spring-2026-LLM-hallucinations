# spectral_detection/config.py
from pathlib import Path

# --- File System Paths ---
BASE_DIR = Path("/content/spectral-llm_hallucinations-project")
CHECKPOINT_DIR = Path("/content/drive/MyDrive/spectral_pipeline") 
DATA_PROCESSED = BASE_DIR / "data/processed"

for path in [BASE_DIR, CHECKPOINT_DIR, DATA_PROCESSED]:
    path.mkdir(parents=True, exist_ok=True)

# --- Hyperparameters & Identifiers ---
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
JUDGE_MODEL_ID = "gpt-4o-mini"


# --- Hyperparameters ---
# The scalar values governing the softmax temperature during token sampling.
# 0.1: Very focused, deterministic (good for baseline).
# 1.0: High creativity 
TEMPERATURES = [0.1, 1.0]

# --- Generative Prompts ---
PROMPTS = {
    "factual": """Answer the following question as briefly as possible.
Here are several examples:
Question: What is the capital of France?
Answer: Paris
Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare
Question: {question}
Answer:""",

    "mmlu": """The following is a multiple choice question.
Explain your reasoning for each option, then state your final answer as a single letter (A, B, C, or D).
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}
Reasoning and answer:""",

    "math": """Solve the following math problem step by step.
Put your final answer inside \\boxed{{}}.
Problem: {question}
Solution:"""
}

# --- Evaluation Prompt ---
# The prompt constrains the judge to select between 4 categories of correctness and 10 domains of knowledge. 
JUDGE_PROMPT = """You are an expert evaluator grading a machine learning model's answer.

Question: {question}
Reference Answer: {reference}
Model Answer: {model_answer}

Perform two classification tasks:

1. CORRECTNESS: Classify the model answer into exactly ONE of the following categories:
- "correct": The answer is factually accurate and semantically matches the reference.
- "hallucination": The answer contains fabricated facts, invents entities, or provides incorrect factual knowledge.
- "illogical": The answer fails at mathematical reasoning, contradicts its own premises, or makes a logical/calculation error, even if it doesn't invent facts.
- "refused": The model explicitly declines to answer (e.g., "I cannot answer that").

2. DOMAIN: Classify the subject matter into exactly one of:
- "Mathematics & Logic"
- "Physical Sciences"
- "Life Sciences"
- "Computer Science & Engineering"
- "History & Geography"
- "Politics & Law"
- "Economics & Business"
- "Philosophy & Religion"
- "Arts & Literature"
- "General Knowledge"

Respond ONLY with a valid JSON object in this exact format:
{{"correctness": "classification", "domain": "classification"}}"""