# spectral_detection/data_generation.py
import json
import time
import os
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI
from . import config
from tqdm.contrib.logging import logging_redirect_tqdm
from spectral_detection.data import features
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

class Pipeline:
    def __init__(self):
        """Allocates the primary language model utilizing NF4 quantization."""
        print(f"Initializing model mapping for {config.LLAMA_MODEL_ID}...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLAMA_MODEL_ID,
            quantization_config=quant_config,
            device_map="auto", # Automatically puts model on GPU
            attn_implementation="eager"
        )
        self.model.eval()
        self.model.config.output_attentions = True
        self.model.config.return_dict = True

    def _format_prompt(self, record):
        ds = record["dataset"]
        if ds == "mmlu":
            return config.PROMPTS["mmlu"].format(
                question=record["question"],
                choice_a=record["choices"][0],
                choice_b=record["choices"][1],
                choice_c=record["choices"][2],
                choice_d=record["choices"][3]
            )
        elif ds == "math":
            return config.PROMPTS["math"].format(question=record["question"])

        # Any other case would be factual
        return config.PROMPTS["factual"].format(question=record["question"])

    def generate_dataset(self, answers_per_prompt, data_list, dataset_name, temperature, 
                     overwrite=False, extract_laplacian=False, num_top_eigenvalues=10):                      
        """Computes response sequences and saves states to a target .jsonl file.
        If extract_laplacian=True, all eigenvalue arrays are accumulated in memory 
        and saved as a single PyTorch binary dictionary at the end.
        """        
        output_file = config.CHECKPOINT_DIR / f"{dataset_name}_t{temperature}_n{answers_per_prompt}.jsonl"
        eigen_file = config.CHECKPOINT_DIR / f"{dataset_name}_t{temperature}_n{answers_per_prompt}_eigen.pt"

        if overwrite:
            if output_file.exists():
                output_file.unlink()
                print(f"Existing file {output_file.name} deleted.")
            if eigen_file.exists():
                eigen_file.unlink()
                print(f"Existing features file {eigen_file.name} deleted.")
        
        # --- RESUME LOGIC ---
        finished_ids = set()
        if output_file.exists():
            with open(output_file, "r") as f:
                finished_ids = {json.loads(line)["id"] for line in f if line.strip()}
                
        laplacian_features_dict = {}
        if extract_laplacian and eigen_file.exists():
            laplacian_features_dict = torch.load(eigen_file, weights_only=True)

        with open(output_file, "a") as file_stream:
            for i, record in enumerate(tqdm(data_list, desc=f"Generating {dataset_name}")):
                unique_identifier = f"{dataset_name}_{i:05d}_t{temperature}"
                
                # Check if the first stochastic sample is already processed
                if f"{unique_identifier}_ans00" in finished_ids:
                    continue
                
                prompt_text = self._format_prompt(record)
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                input_length = inputs.input_ids.shape[1]
                
                # Determine max length (MATH requires longer chains of thought)
                max_tokens = 512 if dataset_name == "math" else 128

                # Calculate the conditional probabilities and sample tokens
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0), 
                        pad_token_id=self.tokenizer.eos_token_id,
                        top_p=0.95,  # Establishes the cumulative probability threshold used by Farquhar et al. 
                        num_return_sequences=answers_per_prompt # <=== Multiple answers per prompt
                    )
                
                for ans_id, output_sequence in enumerate(outputs):
                    generated_tokens = output_sequence[input_length:]
                    answer_id = f"{unique_identifier}_ans{ans_id:02d}"

                    # --- SPECTRAL EXTRACTION ---
                    if extract_laplacian:
                        features.extract_laplacian(self.model, laplacian_features_dict, num_top_eigenvalues, output_sequence, answer_id)
                
                    raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    clean_answer = raw_answer.split("\n")[0].split("Question:")[0].strip()
                
                    result = {
                        "id" : answer_id, 
                        "prompt_id" : unique_identifier,               
                        "sample_num" : f"sample{ans_id:02d}",
                        "dataset" : dataset_name,
                        "question" : record["question"],
                        "reference_answer" : record["reference_answer"],
                        "model_answer" : clean_answer,
                        "temperature" : temperature,
                        "prompt_tokens" : input_length,
                        "generated_tokens" : len(generated_tokens),
                        "timestamp" : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "type" : record.get("type", "") if dataset_name == "defan" else "",
                        "domain" : record.get("domain", "") if dataset_name == "defan" else ""
                    }
                    
                    file_stream.write(json.dumps(result) + "\n")
            
                if i % 5 == 0: 
                    file_stream.flush()
        
        # --- FINAL SAVE ---
        # After the loop finishes, save the entire dictionary to a single binary file
        if extract_laplacian:
            print(f"Saving {len(laplacian_features_dict)} spectral tensors to {eigen_file.name}...")
            torch.save(laplacian_features_dict, eigen_file)
            
        
    # SPECTRAL / "EIGENVALUE" FEATURE EXTRACTION
    @torch.no_grad()
    def extract_topk_laplacian_eigs(self, output_sequence_1d: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Compute top-k Laplacian-like "eigenvalues" from attentions for a *given* token sequence.
        output_sequence_1d: shape [T] (full prompt+generated sequence, as returned by generate()).
        Returns: float16 CPU tensor of shape [L*H*k].
        """
        input_ids = output_sequence_1d.unsqueeze(0).to(self.model.device)  # [1, T]

        out = self.model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        attns = out.attentions
        if attns is None:
            raise RuntimeError("Attentions are still None even with eager attention.")

        # [L, 1, H, T, T] -> [L, H, T, T]
        A = torch.stack(attns, dim=0).squeeze(1)
        _, _, T_len, _ = A.shape

        col_sums = A.sum(dim=-2)  # [L, H, T]
        denom = (T_len - torch.arange(T_len, device=A.device)).clamp_min(1)  # [T] (causal assumption)
        d_ii = col_sums / denom
        a_ii = torch.diagonal(A, dim1=-2, dim2=-1)  # [L, H, T]
        eigenvalues = d_ii - a_ii  # [L, H, T]

        k = min(int(k), T_len)
        sorted_eigvals, _ = torch.sort(eigenvalues, dim=-1, descending=True)
        top_k = sorted_eigvals[..., :k]  # [L, H, k]

        feat = top_k.flatten().detach().cpu().to(torch.float16)

        del out, A, col_sums, denom, d_ii, a_ii, eigenvalues, sorted_eigvals, top_k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return feat

    def generate_dataset_with_judge_and_eigs(
        self,
        answers_per_prompt,
        data_list,
        dataset_name,
        temperature,
        judge_api_key: str,
        save_pt_path,
        k_eigenvalues: int = 10,
        overwrite_pt: bool = False,
        overwrite_jsonl: bool = False):
        """
        One-pass pipeline:
        - generate answers
        - compute eigenfeatures for the exact generated token sequence
        - judge the decoded answer via OpenAI
        - save compact .pt: data[id] = {label, domain, eig_top10}

        Optionally also writes/maintains the JSONL (same format as your current generate_dataset()).
        """
        save_pt_path = Path(save_pt_path)

        # Optional JSONL output for traceability
        output_file = config.CHECKPOINT_DIR / f"{dataset_name}_t{temperature}_n{answers_per_prompt}_eigen.jsonl"
        if overwrite_jsonl and output_file.exists():
            output_file.unlink()

        # Load existing PT if resuming
        if save_pt_path.exists() and not overwrite_pt:
            payload = torch.load(save_pt_path, map_location="cpu")
            pt_data = payload.get("data", {})
        else:
            pt_data = {}

        # Prepare judge client
        judge = LLMJudge(api_key=judge_api_key)

        # Track finished ids so we can resume safely
        finished_ids = set(pt_data.keys())

        with open(output_file, "a") as file_stream:
            for i, record in enumerate(tqdm(data_list, desc=f"Gen+Judge+Eigs {dataset_name}")):
                unique_identifier = f"{dataset_name}_{i:05d}_t{temperature}"

                prompt_text = self._format_prompt(record)
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                input_length = inputs.input_ids.shape[1]

                max_tokens = 512 if dataset_name == "math" else 128

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0),
                        pad_token_id=self.tokenizer.eos_token_id,
                        top_p=0.95,
                        num_return_sequences=answers_per_prompt,
                    )

                for ans_id, output_sequence in enumerate(outputs):
                    record_id = f"{unique_identifier}_ans{ans_id:02d}"
                    if record_id in finished_ids:
                        continue

                    generated_tokens = output_sequence[input_length:]
                    raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    clean_answer = raw_answer.split("\n")[0].split("Question:")[0].strip()

                    # ---- EIGENFEATURES from the exact generated token sequence ----
                    eig_feat = self.extract_topk_laplacian_eigs(
                        output_sequence_1d=output_sequence,
                        k=k_eigenvalues,
                    )

                    # ---- JUDGE ----
                    # Make inputs safe for format()
                    safe_q = str(record["question"]).replace("{", "{{").replace("}", "}}")
                    safe_ref = str(record["reference_answer"]).replace("{", "{{").replace("}", "}}")
                    safe_ans = str(clean_answer).replace("{", "{{").replace("}", "}}")

                    prompt = config.JUDGE_PROMPT.format(
                        question=safe_q,
                        reference=safe_ref,
                        model_answer=safe_ans,
                    )

                    try:
                        response = judge.client.chat.completions.create(
                            model=config.JUDGE_MODEL_ID,
                            response_format={"type": "json_object"},
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                        )
                        eval_data = json.loads(response.choices[0].message.content)                   
                        correctness = str(eval_data.get("correctness", "unknown")).lower()
                        domain = str(eval_data.get("domain", "unknown"))
                        adversarial = eval_data.get("adversarial", False)
                        correctness_score = eval_data.get("correctness_score", -1)
                        
                    except Exception:
                        correctness = "error"
                        domain = "error"
                        adversarial = False
                        correctness_score = -1

                    # ---- Save compact PT entry ----
                    pt_data[record_id] = {
                        "label": correctness,           # judge label
                        "domain": domain,               # judge domain
                        "adversarial": adversarial,     # boolean
                        "score": correctness_score,     # integer
                        "eig_top10": eig_feat,          # float16 [L*H*10]
                    }
                    finished_ids.add(record_id)

                    # ---- Optional JSONL record for debugging/traceability ----
                    result = {
                        "id": record_id,
                        "prompt_id": unique_identifier,
                        "sample_num": f"sample{ans_id:02d}",
                        "dataset": dataset_name,
                        "question": record["question"],
                        "reference_answer": record["reference_answer"],
                        "model_answer": clean_answer,
                        "correctness": correctness,
                        "domain": domain,
                        "adversarial": adversarial,
                        "correctness_score": correctness_score,
                        "temperature": temperature,
                        "prompt_tokens": input_length,
                        "generated_tokens": len(generated_tokens),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "type": record.get("type", "") if dataset_name == "defan" else "",
                    }
                    file_stream.write(json.dumps(result) + "\n")
                    # Periodic checkpoint for safety
                    if len(pt_data) % 50 == 0:
                        payload = {
                            "meta": {
                                "k_eigenvalues": int(k_eigenvalues),
                                "feature_name": "eig_top10",
                                "label_field": "label",
                                "judge_model": str(config.JUDGE_MODEL_ID),
                                "gen_model": str(config.LLAMA_MODEL_ID),
                                "dataset": dataset_name,
                                "temperature": float(temperature),
                                "answers_per_prompt": int(answers_per_prompt),
                            },
                            "data": pt_data,
                        }
                        save_pt_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(payload, save_pt_path)

                if i % 5 == 0:
                    file_stream.flush()

        # Final save
        payload = {
            "meta": {
                "k_eigenvalues": int(k_eigenvalues),
                "feature_name": "eig_top10",
                "label_field": "label",
                "judge_model": str(config.JUDGE_MODEL_ID),
                "gen_model": str(config.LLAMA_MODEL_ID),
                "dataset": dataset_name,
                "temperature": float(temperature),
                "answers_per_prompt": int(answers_per_prompt),
            },
            "data": pt_data,
        }
        save_pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_pt_path)
        print(f"Saved PT with {len(pt_data)} answers to: {save_pt_path.resolve()}")
        return save_pt_path

class LLMJudge:
    def __init__(self, api_key):
        """
        Initializes the OpenAI client for the LLM-as-a-Judge step.
        """
        self.client = OpenAI(api_key=api_key)

    def _atomic_save(self, records, target_path):
        """
        Atomic save
        """
        temp_path = str(target_path) + ".tmp"
        with open(temp_path, "w") as file_stream:
            for record in records: 
                file_stream.write(json.dumps(record) + "\n")
        
        os.replace(temp_path, target_path)

    def evaluate_file(self, target_jsonl_path, save_interval=500):
        """Parses the existing .jsonl file, queries the JSON API, and rewrites the matrix."""
        with open(target_jsonl_path, "r") as file_stream:
            records = [json.loads(line) for line in file_stream]
        
        unsaved_changes = False
        
        with logging_redirect_tqdm():
            for i, record in enumerate(tqdm(records, desc="Evaluating JSONL", mininterval=60.0)):
                if "correctness_score" not in record or record["correctness_score"] == "error":
                    try:
                        # Prevent .format() from crashing on LATEX curly braces
                        safe_q = str(record["question"]).replace("{", "{{").replace("}", "}}")
                        safe_ref = str(record["reference_answer"]).replace("{", "{{").replace("}", "}}")
                        safe_ans = str(record["model_answer"]).replace("{", "{{").replace("}", "}}")

                        prompt = config.JUDGE_PROMPT.format(
                            question=safe_q, 
                            reference=safe_ref,
                            model_answer=safe_ans
                        )
                        
                        # Call GPT-4o-mini
                        response = self.client.chat.completions.create(
                            model=config.JUDGE_MODEL_ID,
                            response_format={"type": "json_object"},
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0
                        )
                        
                        eval_data = json.loads(response.choices[0].message.content)
                        record["correctness"] = str(eval_data.get("correctness", "unknown")).lower()
                        record["domain"] = str(eval_data.get("domain", "unknown"))
                        record["adversarial"] = eval_data.get("adversarial", False)
                        record["correctness_score"] = eval_data.get("correctness_score", -1)
                        
                    except Exception:
                        record["correctness"] = "error"
                        record["domain"] = "error"
                        record["adversarial"] = False
                        record["correctness_score"] = -1
                        
                    unsaved_changes = True
                
                # Periodic saves to disk
                if unsaved_changes and (i + 1) % save_interval == 0:
                    self._atomic_save(records, target_jsonl_path)
                    unsaved_changes = False
                    
            # Final sweep to commit any residual records at loop termination
            if unsaved_changes:
                self._atomic_save(records, target_jsonl_path)

class AsyncLLMJudge:
    def __init__(self, api_key: str):
        """
        Strictly asynchronous client for non-blocking I/O operations.
        """
        self.async_client = AsyncOpenAI(api_key=api_key)

    def _atomic_save(self, records: list, target_path: str):
        """
        Executes kernel-level atomic file replacement to prevent data corruption.
        """
        temp_path = str(target_path) + ".tmp"
        with open(temp_path, "w") as file_stream:
            for record in records: 
                file_stream.write(json.dumps(record) + "\n")
        
        os.replace(temp_path, target_path)

    async def _evaluate_single(self, record: dict, semaphore: asyncio.Semaphore) -> bool:
        """
        An isolated coroutine mapping a single record to an API response.
        Returns True if the record's state was modified, False otherwise.
        """
        # Skip previously verified states
        if "correctness_score" in record and record["correctness_score"] not in ["error", -1]:
            return False

        async with semaphore:
            try:
                # Sanitize LaTeX curly braces for the string formatter
                safe_q = str(record["question"]).replace("{", "{{").replace("}", "}}")
                safe_ref = str(record["reference_answer"]).replace("{", "{{").replace("}", "}}")
                safe_ans = str(record["model_answer"]).replace("{", "{{").replace("}", "}}")

                prompt = config.JUDGE_PROMPT.format(
                    question=safe_q, 
                    reference=safe_ref,
                    model_answer=safe_ans
                )
                
                # Non-blocking network call
                response = await self.async_client.chat.completions.create(
                    model=config.JUDGE_MODEL_ID,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                
                eval_data = json.loads(response.choices[0].message.content)
                record["correctness"] = str(eval_data.get("correctness", "unknown")).lower()
                record["domain"] = str(eval_data.get("domain", "unknown"))
                record["adversarial"] = eval_data.get("adversarial", False)
                record["correctness_score"] = eval_data.get("correctness_score", -1)
                
            except Exception:
                record["correctness"] = "error"
                record["domain"] = "error"
                record["adversarial"] = False
                record["correctness_score"] = -1
                
            return True

    async def evaluate_file_async(self, target_jsonl_path: str, save_interval: int = 500, max_concurrent: int = 50):
        """
        Partitions the discrete set of records into distinct batches. Dispatches each batch 
        concurrently, awaiting resolution before executing an atomic disk write.
        """
        with open(target_jsonl_path, "r") as file_stream:
            records = [json.loads(line) for line in file_stream]
        
        # Bounding the concurrency space to prevent HTTP 429 errors
        semaphore = asyncio.Semaphore(max_concurrent)
        
        for i in range(0, len(records), save_interval):
            batch = records[i:i + save_interval]
            
            # Construct the set of unexecuted coroutines
            tasks = [self._evaluate_single(record, semaphore) for record in batch]
            
            # Execute concurrently and await completion of the batch vector
            results = await async_tqdm.gather(*tasks, desc=f"Evaluating Batch {(i//save_interval)+1}")
            
            # If any element evaluates to True, trigger the atomic disk write
            if any(results):
                self._atomic_save(records, target_jsonl_path)

        print(f"\nAsynchronous evaluation complete for {target_jsonl_path}")