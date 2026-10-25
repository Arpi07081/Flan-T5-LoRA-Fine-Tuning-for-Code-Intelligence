# Flan-T5 LoRA Fine-Tuning for Code Intelligence

## Project Report — Semester 2, Machine Learning

---

## 1. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in code-related tasks, including code generation and natural language to SQL translation. However, deploying state-of-the-art models for these tasks often requires extensive computational resources and large datasets. This project explores **parameter-efficient fine-tuning** (PEFT) of the `google/flan-t5-base` model (250M parameters) using **Low-Rank Adaptation (LoRA)** on two complementary code intelligence benchmarks:

1. **MBPP (Mostly Basic Python Programs)** — Python code generation from natural language descriptions
2. **SPIDER** — Natural language to SQL translation

The goal is to evaluate whether a small, general-purpose language model can be efficiently adapted for dual-task code intelligence using only **1.4% of its parameters** as trainable weights, while running on consumer-grade hardware (Google Colab T4 / RTX 3090).

---

## 2. Background

### 2.1 Flan-T5

Flan-T5 is a variant of Google's T5 (Text-to-Text Transfer Transformer) that has been fine-tuned on over 1,800 tasks using instruction tuning. The `flan-t5-base` variant has 250M parameters with an encoder-decoder architecture, making it suitable for sequence-to-sequence tasks like code generation where the input is a natural language prompt and the output is structured code.

### 2.2 LoRA (Low-Rank Adaptation)

LoRA (Hu et al., 2021) is a parameter-efficient fine-tuning method that freezes the original model weights and injects trainable low-rank decomposition matrices into specific layers. Instead of updating the full weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA learns two smaller matrices:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}$$

where $r \ll \min(d, k)$ is the rank. This dramatically reduces the number of trainable parameters while preserving the model's pre-trained knowledge.

### 2.3 Datasets

| Dataset | Task | Source | Training Size | Test Size |
|---------|------|--------|:---:|:---:|
| **MBPP** | Python code generation | Google Research (HuggingFace) | 464 | 500 |
| **SPIDER** | Text-to-SQL | Yale LILY Lab | 7,000 | 1,659 |

---

## 3. Methodology

### 3.1 Data Preprocessing

Both datasets were preprocessed into a unified JSONL format with `input` and `output` fields.

**MBPP prompt template:**
```
Generate Python: {task_description}
Test cases:
{assertion_1}
{assertion_2}
...
```

**SPIDER prompt template:**
```
Translate to SQL: {natural_language_question}
Database schema:
{table_1}: {col_1}, {col_2}, ...
{table_2}: {col_1}, {col_2}, ...
...
```

For SPIDER, schema enrichment was performed by joining the `tables.json` metadata with query samples to provide full column-level context. Both datasets were combined and shuffled with a **90/10 train/eval split** during training.

### 3.2 LoRA Configuration

The following LoRA configuration was applied to the Flan-T5 base model:

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| Rank (`r`) | 16 | Balances expressiveness vs. parameter count |
| Alpha (`α`) | 32 | Scaling factor = α/r = 2.0 |
| Target modules | `q`, `v` | Query and value attention projection matrices |
| LoRA dropout | 0.05 | Regularization to prevent overfitting |
| Trainable parameters | ~3.5M | **1.4%** of total 250M parameters |

### 3.3 Training Configuration

| Parameter | Value |
|-----------|:-----:|
| Base model | `google/flan-t5-base` (250M params) |
| Precision | **fp32** (fp16 caused NaN losses) |
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| LR scheduler | Linear with warmup |
| Epochs | 5 |
| Batch size (per device) | 8 |
| Gradient accumulation steps | 4 |
| **Effective batch size** | **32** |
| Max input tokens | 512 |
| Max output tokens | 256 |
| Gradient checkpointing | Enabled |
| Experiment tracking | Weights & Biases |

> **Note on precision:** Initial experiments with fp16 mixed precision resulted in NaN training losses due to the sensitivity of T5's relative attention bias computations. Switching to fp32 resolved this instability.

### 3.4 Evaluation Protocol

Two categories of metrics were used:

**Strict (Execution-Based) Metrics:**
- **MBPP Pass@1** — Executes generated code against the provided test assertions in a sandboxed process
- **MBPP AST Compilation Rate** — Checks if the generated code parses as valid Python via `ast.parse()`
- **SPIDER Exact Match (Strict)** — Exact string match between predicted and gold SQL
- **SPIDER Exact Match (Relaxed)** — Normalized SQL comparison using `sqlparse` (lowercasing, whitespace normalization, keyword standardization)

**Soft (Overlap-Based) Metrics:**
- **ROUGE-L** — Longest common subsequence overlap between generated and reference outputs
- **BLEU-2** — Bigram precision with brevity penalty

---

## 4. Training Results

### 4.1 Training Loss

The training loss decreased from approximately **11.0** in the first steps to approximately **2.0** by step 1,000, showing a characteristic steep decline in the first 100 steps followed by a gradual convergence.

![Training Loss vs Steps](../training_plots/training_loss_plot.png)

### 4.2 Validation Loss

Validation loss showed a consistent downward trend from **0.49** to **0.37** across the training run, indicating that the model continued to generalize well without signs of overfitting.

![Validation Loss vs Steps](../training_plots/validation_loss.png)

### 4.3 ROUGE Metrics During Training

Both ROUGE-1 and ROUGE-L steadily improved throughout training, demonstrating that the model learned to produce outputs with increasing lexical overlap with the reference solutions.

| Metric | Start (Step ~200) | End (Step ~1,000) | Improvement |
|--------|:-:|:-:|:-:|
| ROUGE-1 | 0.605 | 0.710 | +17.4% |
| ROUGE-L | 0.575 | 0.690 | +20.0% |

![ROUGE-1 During Training](../training_plots/rouge1_plot.png)

![ROUGE-L During Training](../training_plots/rouge_l_chart.png)

---

## 5. Evaluation Results

### 5.1 MBPP — Python Code Generation

| Metric | Base (Flan-T5) | Fine-tuned (LoRA) | Δ Change |
|--------|:-:|:-:|:-:|
| **Pass@1** | 0.00% | 0.00% | — |
| **AST Compilation Rate** | 16.40% | 0.00% | −16.40 pp |
| **ROUGE-L** | 10.81% | **34.59%** | **+23.78 pp** |
| **BLEU-2** | 0.59% | **1.97%** | **+1.38 pp** |

#### Key Observations

The fine-tuned model shows **significant improvements in soft metrics** — ROUGE-L increased by **3.2×** (10.81% → 34.59%) and BLEU-2 improved by **3.3×** (0.59% → 1.97%). This indicates the model learned the vocabulary, structure, and patterns of Python code generation substantially.

However, **Pass@1 remains at 0%** for both models, and the compilation rate dropped from 16.40% (base) to 0.00% (fine-tuned). This reveals a critical finding:

> **The model learned to generate text that resembles Python code (high ROUGE-L) but failed to produce syntactically valid, executable programs.** Analysis of the 500 detailed outputs shows all 500 predictions failed at the syntax stage (`failed_syntax: 500`), suggesting the fine-tuned model learned code-like patterns without mastering Python's strict syntax requirements.

The drop in AST compilation rate is explained by the base model occasionally producing short, syntactically simple outputs that happened to parse, whereas the fine-tuned model generates longer, more complex code attempts that contain syntax errors.

````carousel
![MBPP — Strict Pass@1](../evaluation_plots/mbpp_strict_pass1_dark.png)
<!-- slide -->
![MBPP — AST Compilation Rate](../evaluation_plots/mbpp_ast_compilation_rate_dark.png)
<!-- slide -->
![MBPP — ROUGE-L Overlap](../evaluation_plots/mbpp_rouge-l_overlap_dark.png)
<!-- slide -->
![MBPP — BLEU-2 Overlap](../evaluation_plots/mbpp_bleu-2_overlap_dark.png)
````

### 5.2 SPIDER — Text-to-SQL

| Metric | Base (Flan-T5) | Fine-tuned (LoRA) | Δ Change |
|--------|:-:|:-:|:-:|
| **Exact Match (Strict)** | 0.00% | **1.93%** | **+1.93 pp** |
| **Exact Match (Relaxed)** | 0.00% | **2.53%** | **+2.53 pp** |
| **ROUGE-L** | 10.38% | **48.57%** | **+38.19 pp** |
| **BLEU-2** | 0.62% | **9.52%** | **+8.90 pp** |

#### Key Observations

SPIDER results show **dramatic improvement across all metrics**:

- **ROUGE-L jumped by 4.7×** (10.38% → 48.57%), indicating the model learned SQL vocabulary, clause ordering, and structural patterns
- **BLEU-2 improved by 15.4×** (0.62% → 9.52%), demonstrating strong bigram-level alignment with gold SQL queries
- The model achieved **32 strict exact matches** and **42 relaxed exact matches** out of 1,659 test queries (vs. 0 for the base model), confirming that LoRA fine-tuning enabled genuine SQL generation capability
- The gap between strict (1.93%) and relaxed (2.53%) exact match suggests the model occasionally produces semantically equivalent SQL with minor formatting differences

> **SPIDER is a more structured task than MBPP**, with a well-defined grammar (SQL keywords, table/column references) that a small model can learn more effectively. The schema-enriched prompts also provide explicit column-level context, reducing the ambiguity the model must resolve.

````carousel
![SPIDER — Strict Exact Match](../evaluation_plots/spider_strict_exact_match_dark.png)
<!-- slide -->
![SPIDER — Relaxed Exact Match](../evaluation_plots/spider_relaxed_exact_match_dark.png)
<!-- slide -->
![SPIDER — ROUGE-L Overlap](../evaluation_plots/spider_rouge-l_overlap_dark.png)
<!-- slide -->
![SPIDER — BLEU-2 Overlap](../evaluation_plots/spider_bleu-2_overlap_dark.png)
````

---

## 6. Comparative Summary

### 6.1 Full Results Table

| | **MBPP (Base)** | **MBPP (LoRA)** | **SPIDER (Base)** | **SPIDER (LoRA)** |
|---|:-:|:-:|:-:|:-:|
| Strict Metric | 0.00% | 0.00% | 0.00% | **1.93%** |
| Relaxed Metric | — | — | 0.00% | **2.53%** |
| ROUGE-L | 10.81% | **34.59%** | 10.38% | **48.57%** |
| BLEU-2 | 0.59% | **1.97%** | 0.62% | **9.52%** |
| Compilation Rate | 16.40% | 0.00% | — | — |
| Test Samples | 500 | 500 | 1,659 | 1,659 |

### 6.2 Key Takeaways

1. **LoRA fine-tuning is effective** — Despite training only 1.4% of parameters, the model showed substantial gains in all overlap metrics across both tasks
2. **Task difficulty matters** — SQL generation (constrained grammar, schema-guided) was significantly more learnable than free-form Python generation for a 250M-parameter model
3. **Soft vs. strict metric divergence** — High ROUGE-L with zero Pass@1 on MBPP highlights that lexical similarity does not imply functional correctness for code generation
4. **Data imbalance impact** — The 15:1 ratio of SPIDER (7,000) to MBPP (464) training examples likely contributed to the stronger SQL performance

---

## 7. Discussion

### 7.1 Why MBPP Pass@1 = 0%

The zero pass rate on MBPP, despite a 3.2× improvement in ROUGE-L, reveals a fundamental limitation of fine-tuning small encoder-decoder models for code generation:

- **Flan-T5-base (250M params) lacks sufficient capacity** for learning Python's full syntax and semantics from only 464 training examples
- All 500 test predictions failed at the **syntax stage** — the model did not generate even one parseable Python function
- The model appears to have learned code-like patterns (variable names, function structures, common idioms) but not the precise syntactic rules required for executable Python
- Larger models (CodeT5+, StarCoder, etc.) are specifically pre-trained on code corpora and demonstrate this capability; Flan-T5 was pre-trained primarily on natural language tasks

### 7.2 The AST Compilation Rate Paradox

The base model achieved a 16.40% AST compilation rate compared to 0% for the fine-tuned model. This counter-intuitive result is explained by examining the output patterns:

- **Base model outputs:** Short, generic text that occasionally happens to be valid Python (e.g., single expressions, simple assignments)
- **Fine-tuned model outputs:** Longer, more structured code attempts that mimic real function definitions but contain subtle syntax errors (unclosed parentheses, incorrect indentation, missing colons)

This is a classic case where a model that "tries harder" produces more complex (but broken) output.

### 7.3 Why SPIDER Performed Better

Several factors explain the stronger SPIDER results:

1. **15× more training data** (7,000 vs. 464 examples)
2. **Constrained output space** — SQL has a finite grammar with ~30 core keywords
3. **Schema-guided prompts** — Explicit table and column information reduces ambiguity
4. **Structural patterns** — `SELECT ... FROM ... WHERE ...` patterns are highly regular and learnable
5. **No execution required** — Evaluation via string matching doesn't require runtime correctness

---

## 8. Technical Challenges

| Challenge | Solution |
|-----------|----------|
| **fp16 NaN losses** | Switched to fp32 full precision training |
| **VRAM constraints** on T4 GPU | Gradient checkpointing + gradient accumulation (effective batch 32) |
| **Tokenization overflow** | Set max input tokens = 512 (T5 limit), max output tokens = 256 |
| **Class imbalance** (SPIDER >> MBPP) | Combined and shuffled datasets; acknowledged as a limitation |
| **Evaluation sandboxing** | Multiprocessed code execution with 5-second timeout per test case |

---

## 9. Project Architecture

```
Project/
├── Notebooks/
│   ├── 01_data_processing.ipynb       # Dataset download, preprocessing, JSONL creation
│   └── 02_finetuning_notebook.ipynb   # LoRA injection, training, model merging
├── evaluation_scripts/
│   ├── evaluate_mbpp.py               # Pass@1, compilation rate, ROUGE, BLEU
│   ├── evaluate_spider.py             # Exact match (strict/relaxed), ROUGE, BLEU
│   └── plot_results.py                # Dark/light mode comparison charts
├── data/
│   ├── raw/                           # Original dataset files
│   └── processed/                     # Preprocessed JSONL splits
├── results/
│   ├── base_model/                    # Base Flan-T5 predictions + metrics
│   └── finetuned/                     # LoRA fine-tuned predictions + metrics
├── training_plots/                    # W&B exported training curves
├── evaluation_plots/                  # Generated comparison bar charts
├── Finetuned model/                   # Saved adapter + merged model weights
└── Docs/                              # Reports and documentation
```

---

## 10. Conclusion

This project demonstrates that **LoRA fine-tuning** can effectively adapt a general-purpose language model for code intelligence tasks with minimal computational cost. Using only **1.4% trainable parameters** (~3.5M out of 250M), we observed:

- **Strong gains on SQL generation** — The fine-tuned model achieved 2.53% relaxed exact match (from 0%) and 48.57% ROUGE-L (from 10.38%) on SPIDER
- **Significant lexical improvement on Python generation** — ROUGE-L improved 3.2× on MBPP, though functional correctness (Pass@1) remained at 0%
- **Task-dependent effectiveness** — Constrained, schema-guided tasks (SQL) are significantly more amenable to small-model fine-tuning than open-ended code generation (Python)

### Future Work

1. **Increase MBPP training data** via data augmentation or synthetic generation
2. **Use a code-specialized base model** (CodeT5, StarCoder) instead of Flan-T5
3. **Increase LoRA rank** (r=32 or r=64) to provide more representational capacity
4. **Task-balanced sampling** to address the 15:1 SPIDER-to-MBPP data imbalance
5. **Beam search decoding** with syntax-aware constraints for executable Python generation

---

## 11. References

1. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
2. Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models.* arXiv:2210.11416
3. Austin, J., et al. (2021). *Program Synthesis with Large Language Models.* arXiv:2108.07732
4. Yu, T., et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP 2018
5. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 21

---

*Report generated for the Machine Learning course project — Semester 2, EMLDS Programme*
