import json
import os
import multiprocessing
import traceback
import evaluate as hf_evaluate
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

TIMEOUT_SECONDS = 2.0

rouge_metric = hf_evaluate.load('rouge')

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def extract_python_code(text):
    if "```python" in text:
        try:
            return text.split("```python")[1].split("```")[0].strip()
        except:
            pass
    if "```" in text:
        try:
            return text.split("```")[1].split("```")[0].strip()
        except:
            pass
    return text.strip()

def extract_ground_truth(text):
    if "def " in text:
        return text.strip()
    return text.strip()

def run_test(code_str, test_cases, return_dict):
    full_script = code_str + "\n\n" + "\n".join(test_cases)
    namespace = {}
    
    # First, test if it just compiles without syntax errors
    try:
        compile(code_str, '<string>', 'exec')
        return_dict['compiles'] = True
    except SyntaxError as e:
        return_dict['compiles'] = False
        return_dict['status'] = 'fail_syntax'
        return_dict['error'] = f"SyntaxError: {e}"
        return
        
    try:
        exec(full_script, namespace)
        return_dict['status'] = 'pass'
    except AssertionError:
        return_dict['status'] = 'fail_assert'
        return_dict['error'] = 'AssertionError'
    except Exception as e:
        return_dict['status'] = 'fail_exception'
        return_dict['error'] = f"{type(e).__name__}: {e}"

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
    
    passed_count = 0
    failed_syntax_count = 0
    failed_assert_count = 0
    failed_other_count = 0
    timeout_count = 0
    compiled_count = 0
    
    pred_texts = []
    ref_texts = []
    
    total = len(predictions)
    
    for item in predictions:
        pred_text = item.get("prediction", "")
        extracted_code = extract_python_code(pred_text)
        test_cases = item.get("test_list", [])
        
        target = item.get("output", "")
        extracted_target = extract_ground_truth(target)
        
        pred_texts.append(extracted_code if extracted_code else " ")
        ref_texts.append(extracted_target if extracted_target else " ")
        
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        return_dict['status'] = 'timeout'
        return_dict['compiles'] = False
        
        p = multiprocessing.Process(target=run_test, args=(extracted_code, test_cases, return_dict))
        p.start()
        p.join(TIMEOUT_SECONDS)
        
        if p.is_alive():
            p.terminate()
            p.join()
            status = 'timeout'
            error_msg = 'Execution Timed Out'
            compiles = False 
        else:
            status = return_dict.get('status', 'fail_unknown')
            error_msg = return_dict.get('error', '')
            compiles = return_dict.get('compiles', False)
            
        if compiles:
            compiled_count += 1
            
        if status == 'pass':
            passed_count += 1
        elif status == 'timeout':
            timeout_count += 1
        elif status == 'fail_syntax':
            failed_syntax_count += 1
        elif status == 'fail_assert':
            failed_assert_count += 1
        else:
            failed_other_count += 1
            
        item_copy = item.copy()
        item_copy["eval_status"] = status
        item_copy["eval_error"] = error_msg
        item_copy["compiles"] = compiles
        results.append(item_copy)
        
    pass_at_1 = (passed_count / total) * 100 if total > 0 else 0
    compilation_rate = (compiled_count / total) * 100 if total > 0 else 0
    
    nlp_scores = compute_nlp_metrics(pred_texts, ref_texts)
    
    metrics = {
        "total": total,
        "passed_strict": passed_count,
        "failed_syntax": failed_syntax_count,
        "failed_assertion": failed_assert_count,
        "failed_exception": failed_other_count,
        "timeout": timeout_count,
        "compiled_count": compiled_count,
        "pass_at_1_percentage": pass_at_1,
        "compilation_rate_percentage": compilation_rate,
        "rougeL": nlp_scores["rougeL"] * 100,
        "bleu": nlp_scores["bleu"] * 100
    }
    
    return metrics, results

def process_model_eval(model_dir):
    preds_file = model_dir / "mbpp_preds.json"
    if not preds_file.exists():
        print(f"Skipping {model_dir.name} - No mbpp_preds.json found.")
        return
        
    print(f"\nEvaluating MBPP for {model_dir.name}...")
    predictions = load_json(preds_file)
    
    metrics, evaluated_results = evaluate_predictions(predictions)
    
    save_json(evaluated_results, model_dir / "mbpp_eval_detailed.json")
    save_json(metrics, model_dir / "mbpp_metrics.json")
    
    print(f"Pass@1: {metrics['pass_at_1_percentage']:.2f}% | Compilation Rate: {metrics['compilation_rate_percentage']:.2f}%")
    print(f"ROUGE-L: {metrics['rougeL']:.2f} | BLEU: {metrics['bleu']:.2f}")
    print(f"Details -> Strict Pass: {metrics['passed_strict']} | Syntax Fails: {metrics['failed_syntax']} | Assert Fails: {metrics['failed_assertion']}")

if __name__ == "__main__":
    print("Starting MBPP Evaluation (With Soft Metrics)...")
    process_model_eval(RESULTS_DIR / "base_model")
    process_model_eval(RESULTS_DIR / "finetuned")
