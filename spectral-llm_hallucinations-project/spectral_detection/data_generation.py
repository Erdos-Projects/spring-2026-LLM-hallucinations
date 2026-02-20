# spectral_detection/data_generation.py
import json
import time
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI
from . import config

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
        return config.PROMPTS["factual"].format(question=record["question"])

    def generate_dataset(self, data_list, dataset_name, temperature, overwrite=False):
        """
        Computes response sequences and saves states to a target .jsonl file.
        If overwrite=True, any existing file for this configuration is destroyed 
        before execution begins.
        """        
        output_file = config.CHECKPOINT_DIR / f"{dataset_name}_t{temperature}.jsonl"

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
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_tokens = outputs[0][input_length:]
                
                result = {
                    "id": unique_identifier,
                    "dataset": dataset_name,
                    "question": record["question"],
                    "reference_answer": record["reference_answer"],
                    "model_answer": self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip(),
                    "temperature": temperature,
                    "prompt_tokens": input_length,
                    "generated_tokens": len(generated_tokens),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                
                # Write immediately to disk
                file_stream.write(json.dumps(result) + "\n")
                if i % 5 == 0: 
                    file_stream.flush()

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
        for record in tqdm(records, desc="Evaluating JSONL"):
            if "correctness" not in record:
                try:
                    # Escape LaTeX curly braces so .format() doesn't crash
                    safe_q = str(record["question"]).replace("{", "{{").replace("}", "}}")
                    safe_ref = str(record["reference_answer"]).replace("{", "{{").replace("}", "}}")
                    safe_ans = str(record["model_answer"]).replace("{", "{{").replace("}", "}}")

                    prompt = config.JUDGE_PROMPT.format(
                        question=record["question"], 
                        reference=record["reference_answer"],
                        model_answer=record["model_answer"]
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
            prompt = config.JUDGE_PROMPT.format(
                question=record["question"], 
                reference=record["reference_answer"],
                model_answer=record["model_answer"]
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