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
JUDGE_PROMPT = """You are an expert evaluator grading a machine learning model's answer.

Question: {question}
Reference Answer: {reference}
Model Answer: {model_answer}

Perform the following classification tasks:

1. CORRECTNESS: Classify the model answer into exactly ONE of the following categories:
- "correct": The answer is factually accurate and semantically matches the reference.
- "incorrect": The answer is not correct, according to the definition above.  
- "refused": The model explicitly declines to answer (e.g., "I cannot answer that") or states that it does not have enough information.

For example:

Question: Who is the young guitarist who played with buddy guy?
Reference Answer: [Quinn Sullivan, Eric Gales]
Model Answer: Ronnie Earl
Correctness: "incorrect"

Question: What is the name of the actor who plays Iron Man in the Marvel movies?
Reference Answer: [Robert Downey Jr.]
Model Answer: Robert Downey Jr. played the role of Tony Stark/Iron Man in the Marvel Cinematic Universe films.
Correctness: "correct"

Question: what is the capital of France?
Reference Answer: [Paris]
Model Answer: I don't have enough information to answer this question.
Correctness: "refused"

Question: who was the first person to walk on the moon?
Reference Answer: [Neil Armstrong]
Model Answer: I apologize, but I cannot provide an answer without verifying the historical facts.
Correctness: "refused"


2. DOMAIN (Categorical): Classify the subject matter into exactly ONE of the following:
- "STEM"
- "Humanities"
- "Social Sciences"
- "Medicine & Health"
- "Law, Business, and Miscellaneous"

3. ADVERSARIAL (Boolean):
- true: The prompt is a "trick" question, tests a common misconception, or is designed to induce a hallucination.
- false: A straightforward, standard, well-intentioned inquiry.

4. CORRECTNESS_SCORE (1-5 Scale):
- 1: Completely Incorrect.
- 2: Mostly Incorrect.
- 3: Partially Correct.
- 4: Mostly Correct.
- 5: Perfectly Correct.

Respond ONLY with a valid JSON object in this exact format:
{{
  "correctness": "string",
  "domain": "string",
  "adversarial": boolean,
  "correctness_score": integer
}}"""