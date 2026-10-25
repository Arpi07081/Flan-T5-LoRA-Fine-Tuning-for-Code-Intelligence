import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup paths
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

MBPP_TEST = PROCESSED_DATA_DIR / "mbpp_test.jsonl"
SPIDER_TEST = PROCESSED_DATA_DIR / "spider_test.jsonl"

BASE_MODEL_NAME = "google/flan-t5-base"
FINETUNED_MODEL_DIR = RESULTS_DIR / "finetuned" / "finetuned_lora" / "merged"

BASE_OUT_DIR = RESULTS_DIR / "base_model"
FINETUNED_OUT_DIR = RESULTS_DIR / "finetuned"

# Ensure output directories exist
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
FINETUNED_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration (supports CUDA, Apple Silicon MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def generate_predictions(model, tokenizer, test_data, batch_size=8, max_new_tokens=256):
    model.eval()
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating"):
        batch = test_data[i:i + batch_size]
        inputs_text = [item["input"] for item in batch]
        
        # Tokenize
        inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                # use_cache=True is okay for pure generation without gradient checkpointing
            )
            
        # Decode
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j, item in enumerate(batch):
            result_item = item.copy()
            result_item["prediction"] = decoded_outputs[j].strip()
            results.append(result_item)
            
    return results

def run_evaluation_for_model(model_name_or_path, is_base_model=True):
    print(f"\n{'='*50}\nLoading model: {model_name_or_path}\n{'='*50}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        model.to(device)
    except Exception as e:
        print(f"Failed to load {model_name_or_path}. Check if the path exists and is correct.")
        print(f"Error: {e}")
        return

    out_dir = BASE_OUT_DIR if is_base_model else FINETUNED_OUT_DIR

    # Process MBPP
    out_path_mbpp = out_dir / "mbpp_preds.json"
    if out_path_mbpp.exists():
        print(f"\nSkipping MBPP - Predictions already exist at {out_path_mbpp}")
    elif MBPP_TEST.exists():
        print(f"\nProcessing MBPP (Test set: {MBPP_TEST.name})...")
        mbpp_data = load_jsonl(MBPP_TEST)
        mbpp_results = generate_predictions(model, tokenizer, mbpp_data)
        save_json(mbpp_results, out_path_mbpp)
        print(f"Saved MBPP predictions to {out_path_mbpp}")
    else:
        print(f"MBPP test file not found at {MBPP_TEST}")

    # Process SPIDER
    out_path_spider = out_dir / "spider_preds.json"
    if out_path_spider.exists():
        print(f"\nSkipping SPIDER - Predictions already exist at {out_path_spider}")
    elif SPIDER_TEST.exists():
        print(f"\nProcessing SPIDER (Test set: {SPIDER_TEST.name})...")
        spider_data = load_jsonl(SPIDER_TEST)
        spider_results = generate_predictions(model, tokenizer, spider_data)
        save_json(spider_results, out_path_spider)
        print(f"Saved SPIDER predictions to {out_path_spider}")
    else:
        print(f"SPIDER test file not found at {SPIDER_TEST}")

if __name__ == "__main__":
    print("Starting generation pipeline...")
    
    # 1. Evaluate Base Model
    run_evaluation_for_model(BASE_MODEL_NAME, is_base_model=True)
    
    # 2. Evaluate Finetuned Model
    if FINETUNED_MODEL_DIR.exists():
        run_evaluation_for_model(str(FINETUNED_MODEL_DIR), is_base_model=False)
    else:
        print(f"\nFinetuned model directory not found at {FINETUNED_MODEL_DIR}")
        print("Please ensure the model was merged and downloaded successfully.")
