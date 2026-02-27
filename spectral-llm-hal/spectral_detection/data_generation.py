# spectral_detection/data_generation.py
import json
import time
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI
from . import config
from tqdm.contrib.logging import logging_redirect_tqdm

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
            device_map="auto" # Automatically puts model on GPU
        )
        self.model.eval()

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

    def generate_dataset(self, answers_per_prompt, data_list, dataset_name, temperature, overwrite=False):
        """
        Computes response sequences and saves states to a target .jsonl file.
        If overwrite=True, any existing file for this configuration is destroyed 
        before execution begins.
        """        
        output_file = config.CHECKPOINT_DIR / f"{dataset_name}_t{temperature}_n{answers_per_prompt}.jsonl"

        # --- OVERWRITE LOGIC ---
        if overwrite and output_file.exists():
            output_file.unlink()
            print(f"Existing file {output_file.name} deleted. Starting fresh.")
        
        finished_ids = set()
        if output_file.exists():
            with open(output_file, "r") as f:
                finished_ids = {json.loads(line)["id"] for line in f if line.strip()}
                
        with open(output_file, "a") as file_stream:
            for i, record in enumerate(tqdm(data_list, desc=f"Generating {dataset_name}")):
                unique_identifier = f"{dataset_name}_{i:05d}_t{temperature}"
                
                if unique_identifier in finished_ids:
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
                    # Slice the tensor to remove the input prompt tokens
                    generated_tokens = output_sequence[input_length:]

                    # Raw string
                    raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    # Splits string at the first newline (Llama produces additional non-sense lines)
                    clean_answer = raw_answer.split("\n")[0].split("Question:")[0].strip()
                
                    result = {
                        "id" : f"{unique_identifier}_ans{ans_id:02d}", # <=== Fully unique ID
                        "prompt_id" : unique_identifier,               # <=== Groups the 20 samples
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
                    
                    # Write to disk
                    file_stream.write(json.dumps(result) + "\n")
            
                if i % 5 == 0: 
                    file_stream.flush()
        
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

        # [L, 1, H, T, T] -> [L, H, T, T]
        A = torch.stack(out.attentions, dim=0).squeeze(1)
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
        overwrite_jsonl: bool = False,
    ):
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
        output_file = config.CHECKPOINT_DIR / f"{dataset_name}_t{temperature}_n{answers_per_prompt}.jsonl"
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
                    except Exception:
                        correctness = "error"
                        domain = "error"

                    # ---- Save compact PT entry ----
                    pt_data[record_id] = {
                        "label": correctness,     # judge label
                        "domain": domain,         # judge domain (optional)
                        "eig_top10": eig_feat,    # float16 [L*H*10]
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

    def evaluate_file(self, target_jsonl_path):
        """Parses the existing .jsonl file, queries the JSON API, and rewrites the matrix."""
        with open(target_jsonl_path, "r") as file_stream:
            records = [json.loads(line) for line in file_stream]
        
        updated_records = []
        
        with logging_redirect_tqdm():
            for record in tqdm(records, desc="Evaluating JSONL", mininterval=60.0):
                if "correctness" not in record:
                    try:
                        # Improve inputs to prevent .format() from crashing on curly braces from LATEX outputs
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
                        record["correctness"] = eval_data.get("correctness", "unknown").lower()
                        record["domain"] = eval_data.get("domain", "unknown")
                            
                    except Exception:
                        record["correctness"] = "error"
                        record["domain"] = "error"
                
                updated_records.append(record)
                
            # Save back evaluation from judge
            with open(target_jsonl_path, "w") as file_stream:
                for record in updated_records: 
                    file_stream.write(json.dumps(record) + "\n")
                
    def test_judge(self, record):
        """
        Diagnostic method to isolate and print exact API or formatting errors.
        Pass a single dictionary record to this method.
        """
        import traceback
        print("\n--- Diagnostic: Testing Judge ---")
        
        try:
            print("[1] Attempting to format prompt...")
            safe_q = str(record["question"]).replace("{", "{{").replace("}", "}}")
            safe_ref = str(record["reference_answer"]).replace("{", "{{").replace("}", "}}")
            safe_ans = str(record["model_answer"]).replace("{", "{{").replace("}", "}}")

            prompt = config.JUDGE_PROMPT.format(
                question=safe_q, 
                reference=safe_ref,
                model_answer=safe_ans
            )
            
            print("Success. Prompt preview (first 100 chars):")
            print(prompt[:100] + "...\n")
            
            print("[2] Calling OpenAI API...")
            response = self.client.chat.completions.create(
                model=config.JUDGE_MODEL_ID,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw_content = response.choices[0].message.content
            print("Success. Raw API Response:")
            print(raw_content + "\n")
            
            print("[3] Attempting JSON parsing...")
            eval_data = json.loads(raw_content)
            print("Success. Parsed Data:")
            print(eval_data)
            
        except Exception as e:
            print("\n!!! ERROR ENCOUNTERED !!!")
            traceback.print_exc() # This prints the exact line and cause of the crash