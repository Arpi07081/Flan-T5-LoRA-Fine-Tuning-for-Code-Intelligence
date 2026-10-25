# Flan-T5 LoRA Fine-Tuning for Code Intelligence

Fine-tuning [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) (250M parameters) on two coding tasks simultaneously using **LoRA** (Low-Rank Adaptation) — designed to run on a free Google Colab T4 GPU.

---

## Tasks

| Task | Dataset | Prompt Format | Training Examples |
|------|---------|---------------|-------------------|
| Python code generation | [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) | `Generate Python: {description}\nTest cases:\n{asserts}` | 464 (train + validation) |
| Text-to-SQL | [SPIDER](https://yale-lily.github.io/spider) | `Translate to SQL: {question}\nDatabase schema:\n{schema}` | 7,000 |

---

## Project Structure

```
Project/
├── Notebooks/
│   ├── 01_data_processing.ipynb   # Download & preprocess MBPP + SPIDER → JSONL
│   └── 02_finetuning_notebook.ipynb # Load data, apply LoRA, train, save model
├── data/
│   ├── raw/
│   │   ├── mbpp/                  # Raw MBPP files (auto-downloaded)
│   │   └── spider/                # SPIDER JSON files (manual upload required)
│   └── processed/
│       ├── mbpp_train.jsonl       # 464 formatted training examples
│       ├── mbpp_test.jsonl        # 500 held-out test examples
│       ├── spider_train.jsonl     # 7,000 formatted training examples
│       └── spider_test.jsonl      # 1,659 held-out test examples
├── Finetuned model/
│   └── finetuned/
│       └── finetuned_lora/
│           ├── adapter/           # LoRA adapter weights (~14 MB)
│           ├── merged/            # Merged model (full weights, ready for inference)
│           ├── checkpoints/       # Per-epoch training checkpoints
│           └── lora_loss_curve.png
└── Docs/
    ├── Finetuning Code Explanation.md  # Deep-dive explanation of the training code
    └── ML Course Project.pdf           # Project report
```

---

## How to Run

### 1. Data Preprocessing — `01_data_processing.ipynb`

Run this notebook **once** before training. It:

1. Downloads the full MBPP dataset from HuggingFace
2. Loads the SPIDER dataset from Google Drive (see requirements below)
3. Formats both into a unified `{"input": ..., "output": ...}` JSONL format
4. Saves all four splits to `data/processed/`

**SPIDER requirement:** Upload the following files to `FlanT5_Finetune/data/raw/spider/` on your Google Drive before running:
- `train_spider.json`
- `train_others.json`
- `tables.json`

### 2. Fine-Tuning — `02_finetuning_notebook.ipynb`

Trains `flan-t5-base` with LoRA adapters on the merged MBPP + SPIDER training sets. Key steps:

1. Load and shuffle preprocessed JSONL files (90/10 train/eval split)
2. Load `flan-t5-base` in fp32 (numerical stability)
3. Apply LoRA to query and value attention matrices
4. Train with `Seq2SeqTrainer` for 5 epochs, logging to Weights & Biases
5. Merge LoRA adapter back into the base model and save

---

## Key Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `google/flan-t5-base` | 250M params, encoder-decoder |
| LoRA rank (`r`) | 16 | ~3.5M trainable params (1.4%) |
| LoRA alpha | 32 | Scaling = alpha/r = 2.0 |
| LoRA targets | `q`, `v` | Query + Value attention matrices |
| Learning rate | 3e-4 | Standard LoRA LR |
| Epochs | 5 | |
| Batch size | 8 | Effective batch = 32 (grad accum × 4) |
| Precision | fp32 | fp16 disabled (caused NaN losses) |
| Max input tokens | 512 | T5 pre-training limit |
| Max output tokens | 256 | Covers all Python/SQL targets |

---

## Evaluation Metrics

- **During training:** ROUGE-1 and ROUGE-L (word-overlap proxy)
- **Final MBPP evaluation:** pass@1 (execute generated code against test cases)
- **Final SPIDER evaluation:** Exact Set Match (normalized SQL comparison)

---

## Dependencies

```bash
pip install datasets transformers sentencepiece evaluate rouge-score \
            accelerate peft bitsandbytes tqdm wandb matplotlib
```

All notebooks are designed for **Google Colab** and save outputs to Google Drive to persist across sessions.

---

## Results

Training loss and ROUGE metrics are tracked via [Weights & Biases](https://wandb.ai). The loss curve is saved to `Finetuned model/finetuned/finetuned_lora/lora_loss_curve.png`.

---

## References

- [Flan-T5 Paper](https://arxiv.org/abs/2210.11416)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [MBPP Dataset](https://huggingface.co/datasets/google-research-datasets/mbpp)
- [SPIDER Benchmark](https://yale-lily.github.io/spider)
- [PEFT Library](https://github.com/huggingface/peft)
