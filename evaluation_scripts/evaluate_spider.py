import json
import os
import sqlparse
import re
import evaluate as hf_evaluate
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

rouge_metric = hf_evaluate.load('rouge')

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def normalize_sql_strict(sql_str):
    if not sql_str: return ""
    formatted = sqlparse.format(
        sql_str, reindent=False, keyword_case='lower', identifier_case='lower', strip_comments=True
    )
    return " ".join(formatted.split()).replace(";", "").strip()

def normalize_sql_relaxed(sql_str):
    if not sql_str: return ""
    # Strip everything down to lowercase alphanumeric text structure
    norm = normalize_sql_strict(sql_str).lower()
    # Remove quotes, ticks, and extra punctuation to give credit for structural similarity
    norm = re.sub(r'[\'\"`\;]', '', norm)
    # Convert obvious synonym operators (model sometimes learns <> vs !=)
    norm = norm.replace("!=", "<>")
    return " ".join(norm.split())

def extract_sql_query(text):
    if "```sql" in text:
        try: return text.split("```sql")[1].split("```")[0].strip()
        except: pass
    if "```" in text:
        try: return text.split("```")[1].split("```")[0].strip()
        except: pass
    return text.strip()

def compute_nlp_metrics(predictions, references):
    if not predictions or not references:
        return {"rougeL": 0.0, "bleu": 0.0}
    res = {"rougeL": 0.0, "bleu": 0.0}
    try:
        rouge_res = rouge_metric.compute(predictions=predictions, references=references)
        res["rougeL"] = rouge_res.get("rougeL", 0.0)
    except:
        pass
        
    try:
        # NLTK expects tokenized sequences
        tokenized_refs = [[[word for word in ref.split()]] for ref in references]
        tokenized_preds = [[word for word in pred.split()] for pred in predictions]
        
        # Smoothing method 4 prevents the 0.0 crash if there is no 4-gram overlap
        smoothie = SmoothingFunction().method4
        bleu_score = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smoothie)
        res["bleu"] = bleu_score
    except Exception as e:
        print(f"  [Warning] NLTK BLEU compute failed: {e}")
        
    return res

def evaluate_predictions(predictions):
    results = []
    
    em_strict_count = 0
    em_relaxed_count = 0
    
    pred_texts = []
    ref_texts = []
    
    total = len(predictions)
    
    for item in predictions:
        pred_raw = item.get("prediction", "")
        extracted_sql = extract_sql_query(pred_raw)
        
        target_raw = item.get("output", "")
        
        norm_pred_strict = normalize_sql_strict(extracted_sql)
        norm_target_strict = normalize_sql_strict(target_raw)
        
        norm_pred_relaxed = normalize_sql_relaxed(extracted_sql)
        norm_target_relaxed = normalize_sql_relaxed(target_raw)
        
        pred_texts.append(norm_pred_relaxed if norm_pred_relaxed else " ")
        ref_texts.append(norm_target_relaxed if norm_target_relaxed else " ")
        
        is_strict = (norm_pred_strict == norm_target_strict)
        is_relaxed = (norm_pred_relaxed == norm_target_relaxed)
        
        if is_strict: em_strict_count += 1
        if is_relaxed: em_relaxed_count += 1
            
        item_copy = item.copy()
        item_copy["norm_prediction"] = norm_pred_relaxed
        item_copy["norm_ground_truth"] = norm_target_relaxed
        item_copy["is_exact_match_strict"] = is_strict
        item_copy["is_exact_match_relaxed"] = is_relaxed
        results.append(item_copy)
        
    em_strict_percentage = (em_strict_count / total) * 100 if total > 0 else 0
    em_relaxed_percentage = (em_relaxed_count / total) * 100 if total > 0 else 0
    
    nlp_scores = compute_nlp_metrics(pred_texts, ref_texts)
    
    metrics = {
        "total": total,
        "exact_matches_strict": em_strict_count,
        "exact_matches_relaxed": em_relaxed_count,
        "em_strict_percentage": em_strict_percentage,
        "em_relaxed_percentage": em_relaxed_percentage,
        "rougeL": nlp_scores["rougeL"] * 100,
        "bleu": nlp_scores["bleu"] * 100
    }
    
    return metrics, results

def process_model_eval(model_dir):
    preds_file = model_dir / "spider_preds.json"
    if not preds_file.exists():
        print(f"Skipping {model_dir.name} - No spider_preds.json found.")
        return
        
    print(f"\nEvaluating SPIDER for {model_dir.name}...")
    predictions = load_json(preds_file)
    
    metrics, evaluated_results = evaluate_predictions(predictions)
    
    save_json(evaluated_results, model_dir / "spider_eval_detailed.json")
    save_json(metrics, model_dir / "spider_metrics.json")
    
    print(f"Strict EM: {metrics['em_strict_percentage']:.2f}% | Relaxed EM: {metrics['em_relaxed_percentage']:.2f}%")
    print(f"ROUGE-L: {metrics['rougeL']:.2f} | BLEU: {metrics['bleu']:.2f}")

if __name__ == "__main__":
    print("Starting SPIDER Evaluation (With Soft Metrics)...")
    process_model_eval(RESULTS_DIR / "base_model")
    process_model_eval(RESULTS_DIR / "finetuned")
