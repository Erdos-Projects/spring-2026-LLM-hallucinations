# spectral_detection/config.py
import os
from pathlib import Path

# --- Project Directories ---
# The root folder in Colab
BASE_DIR = Path("/content/spectral-llm_hallucinations-project")

# CHECKPOINT_DIR: Backup inside Google Drive. 
CHECKPOINT_DIR = Path("/content/drive/MyDrive/spectral_llm_pipeline") 

DATA_PROCESSED = BASE_DIR / "data/processed"

# Create these directories if they don't exist yet
for p in [BASE_DIR, CHECKPOINT_DIR, DATA_PROCESSED]:
    p.mkdir(parents=True, exist_ok=True)

# --- LLM Models ---
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
JUDGE_MODEL_ID = "gpt-4o-mini"

# --- Generation Parameters ---
# We generate two answers for every question:
# 0.1: Very focused, deterministic (good for baseline).
# 1.0: High creativity (Binkowski found this produces more balanced hallucination rates).
TEMPERATURES = [0.1, 1.0]

PROMPTS = {
    # REVISED LISTING 3 PROMPT (for TruthfulQA & TriviaQA)
    # This uses "few-shot" examples to teach the model to be brief.
    "factual": """Answer the following question as briefly as possible.
Here are several examples:

Question: What is the capital of France?
Answer: Paris

Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Question: What is the boiling point of water in Celsius?
Answer: 100 degrees

Question: {question}
Answer:""",

    # MMLU PROMPT
    # Forces the model to explain "Why" before giving the letter answer (Chain-of-Thought, CoT).
    # This provides more "reasoning tokens" for the spectral analysis to look at.
    "mmlu": """The following is a multiple choice question.
Explain your reasoning for each option, then state your final answer as a single letter (A, B, C, or D).

{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Reasoning and answer:""",

    # MATH PROMPT
    # Standard format for math problems, asking for a boxed final number.
    "math": """Solve the following math problem step by step.
Put your final answer inside \\boxed{{}}.

Problem: {question}
Solution:"""
}

# Force GPT-4o-mini to grade the answers.
JUDGE_PROMPT = """You are evaluating whether a model's answer is correct.

Question: {question}
Reference Answer: {reference}
Acceptable Aliases: {aliases}
Model Answer: {model_answer}

Classify the model answer as one of:
- "correct": semantically matches the reference or is truthful.
- "incorrect": factually wrong, hallucinated, or fabricated.
- "refused": the model explicitly declines to answer (e.g., "I don't know").

Respond with ONLY the classification word."""