# spectral_detection/data_generation.py
import torch
import json
import time
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI
from . import config

class Pipeline:
    def __init__(self):
        """
        Initializes the model in 4-bit mode.
        
        Why 4-bit? 
        Colab's free T4 GPU has 16GB of VRAM. A standard 3B model in float16 
        takes ~6GB.  This will fit in memory in Colab, but Attention maps will occupy 
        significant memory.  4-bit quantization reduces the model size significantly
        """
        print(f"Initializing Pipeline with {config.LLAMA_MODEL_ID}...")
        
        # Configure quantization (compression)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_ID)
        # Llama 3 doesn't have a default pad token, so we set it to EOS (End of Sequence)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLAMA_MODEL_ID,
            quantization_config=bnb_config, # quantization configuration
            device_map="auto" # Automatically puts model on GPU
        )
        self.model.eval() # Set to evaluation mode (turns off dropout)

    def _format_prompt(self, record):
        """
        Selects the correct prompt template (Listing 3, MMLU, or MATH)
        based on the dataset name.
        """
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
        else:
            # For TruthfulQA and TriviaQA, use the revised Listing 3 prompt
            return config.PROMPTS["factual"].format(question=record["question"])

    def run_generation(self, dataset_list, dataset_name, temperature):
        """
        Loops through the dataset, generates answers, and saves to Drive.
        """
        # Define the file path in Google Drive
        filename = f"{dataset_name}_t{temperature}.jsonl"
        output_path = config.CHECKPOINT_DIR / filename
        
        # --- RESUME LOGIC ---
        # If the file already exists, we read it to see which IDs are done.
        # This prevents re-doing work if Colab crashes.
        finished_ids = set()
        if output_path.exists():
            with open(output_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        finished_ids.add(data["id"])
                    except:
                        continue # Skip broken lines
            print(f"Resuming {dataset_name} (Temp {temperature}): Found {len(finished_ids)} completed.")

        print(f"Generating for {dataset_name} at Temp {temperature}...")
        
        # Open file in 'append' mode ('a') so we add new lines to the end
        with open(output_path, "a") as f:
            
            for i, record in enumerate(tqdm(dataset_list)):
                
                # Create a unique ID for this specific run
                # Format: dataset_index_temperature (e.g., truthfulqa_00001_t1.0)
                unique_id = f"{dataset_name}_{i:05d}_t{temperature}"
                
                # Skip if already done
                if unique_id in finished_ids:
                    continue
                
                # Prepare input
                prompt_text = self._format_prompt(record)
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
                input_length = inputs.input_ids.shape[1]
                
                # Determine max length (MATH requires longer chains of thought)
                max_new = 512 if dataset_name == "math" else 128

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        do_sample=(temperature > 0), # True if temp > 0
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode the output (removing the prompt itself)
                generated_tokens = outputs[0][input_length:]
                response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # Build the final record to save
                result = {
                    "id": unique_id,
                    "dataset": dataset_name,
                    "question": record["question"],
                    "reference_answer": record["reference_answer"],
                    "model_answer": response_text,
                    "aliases": record.get("aliases", []),
                    "subject": record.get("subject"),
                    "temperature": temperature,
                    "prompt_tokens": input_length,
                    "generated_tokens": len(generated_tokens),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                
                # Write immediately to disk
                f.write(json.dumps(result) + "\n")
                
                # .flush() forces the computer to write to the drive right now,
                # rather than waiting for a buffer to fill. Safer for crashes.
                if i % 5 == 0:
                    f.flush()

class LLMJudge:
    def __init__(self, api_key):
        """
        Initializes the OpenAI client for the LLM-as-a-Judge step.
        """
        self.client = OpenAI(api_key=api_key)

    def evaluate_file(self, jsonl_path):
        """
        Reads a generated file, adds correctness labels, and saves it back.
        """
        print(f"Judging file: {jsonl_path}")
        
        # Read the file into memory
        records = []
        with open(jsonl_path, "r") as f:
            for line in f:
                records.append(json.loads(line))
        
        updated_records = []
        for record in tqdm(records):
            
            # If we haven't judged this one yet
            if "correctness" not in record:
                try:
                    # Fill in the Judge Prompt
                    prompt = config.JUDGE_PROMPT.format(
                        question=record["question"],
                        reference=record["reference_answer"],
                        aliases=record["aliases"],
                        model_answer=record["model_answer"]
                    )
                    
                    # Call GPT-4o-mini
                    response = self.client.chat.completions.create(
                        model=config.JUDGE_MODEL_ID,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0 # Deterministic grading
                    )
                    
                    # Parse the label
                    raw_label = response.choices[0].message.content.strip().lower()
                    if "incorrect" in raw_label:
                        record["correctness"] = "incorrect"
                    elif "correct" in raw_label:
                        record["correctness"] = "correct"
                    elif "refused" in raw_label:
                        record["correctness"] = "refused"
                    else:
                        record["correctness"] = "unknown"
                        
                except Exception as e:
                    print(f"Error judging record {record['id']}: {e}")
                    record["correctness"] = "error"
            
            updated_records.append(record)
            
        # Overwrite the file with the labeled data
        with open(jsonl_path, "w") as f:
            for record in updated_records:
                f.write(json.dumps(record) + "\n")