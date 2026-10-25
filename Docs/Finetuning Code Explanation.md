# UNIZAR ML Project — Deep Dive Explanation

> **Goal**: Finetune `google/flan-t5-base` (a 250-million-parameter language model) on two coding tasks simultaneously:
> - **MBPP** — generate Python functions from natural language descriptions
> - **SPIDER** — translate natural language questions into SQL queries
>
> The technique used is **LoRA** (Low-Rank Adaptation), a parameter-efficient finetuning method that updates only a tiny fraction of the model's weights, keeping VRAM usage low enough to run on a free Google Colab T4 GPU.

---

## Table of Contents

1. [Big Picture — What Is This Script Doing?](#1-big-picture)
2. [CELL 1 — Install Dependencies](#2-cell-1--install-dependencies)
3. [CELL 2 — Configuration & Hyperparameters (Deep Dive)](#3-cell-2--configuration--hyperparameters)
4. [CELL 3 — Helper Functions](#4-cell-3--helper-functions)
5. [CELL 4 — Load & Merge Data](#5-cell-4--load--merge-data)
6. [CELL 5 — Load Model & Apply LoRA](#6-cell-5--load-model--apply-lora)
7. [CELL 6 — Sanity Check](#7-cell-6--sanity-check)
8. [CELL 7 — Training](#8-cell-7--training)
9. [CELL 8 — Merge, Plot, Save](#9-cell-8--merge-plot-save)
10. [Key Concepts Glossary](#10-key-concepts-glossary)
11. [Common Pitfalls](#11-common-pitfalls)

---

## 1. Big Picture

### What is Flan-T5?

**T5** (Text-to-Text Transfer Transformer) reformulates every NLP task as a text-in → text-out problem. Instead of separate model heads for classification, generation, etc., you simply give it a text prompt and it generates a text answer.

**Flan-T5** is T5 that has been *instruction-tuned* — Google ran it through hundreds of tasks with natural-language instructions (like "Translate to French:", "Summarise:", etc.) so it is already very good at following task-prefix prompts. This makes it ideal for our project.

**Architecture — Encoder-Decoder (seq2seq)**:

```
Input prompt  ──▶  [ENCODER]  ──▶  hidden states  ──▶  [DECODER]  ──▶  Output tokens
"Generate Python: ..."                                              "def add(a, b): ..."
```

The encoder reads the whole input at once. The decoder generates one token at a time, attending to its own previous outputs *and* the encoder's hidden states (this is called **cross-attention**).

### What is LoRA?

Training all 250M parameters of Flan-T5 requires enormous GPU memory. LoRA solves this by mathematics:

Every weight matrix `W` in the model (e.g., the query projection in attention) is **frozen**. Instead, we inject two small trainable matrices `A` and `B` alongside it:

```
W_new = W_frozen  +  (B × A) × (alpha / r)
         (frozen)      (trained)  (scaling)
```

- `A` is initialised randomly, shape `[r, d]`
- `B` is initialised to **zeros**, so at the start `B×A = 0` — the model behaves identically to the base model
- During training, only `A` and `B` are updated
- At the end, `B×A` can be merged back into `W` with zero inference overhead

**Why is this powerful?** The rank `r=16` means instead of updating a `[768×768]` = 589,824-parameter matrix, you only update `[768×16] + [16×768]` = 24,576 parameters — a **24× reduction** per layer.

### The Training Pipeline at a Glance

```
preprocess_data.py output (JSONL files)
        │
        ▼
CELL 4: Load MBPP + SPIDER → shuffle → 90/10 split
        │
        ▼
CELL 5: Load flan-t5-base (fp32) → inject LoRA adapters
        │
        ▼
CELL 7: Seq2SeqTrainer → 5 epochs → save adapter (~14 MB)
        │
        ▼
CELL 8: merge_and_unload() → full model → ready for evaluation
```

---

## 2. CELL 1 — Install Dependencies

```python
!pip install -q datasets transformers sentencepiece evaluate rouge-score \
             accelerate peft bitsandbytes tqdm wandb matplotlib
```

| Package | Purpose |
|---|---|
| `datasets` | HuggingFace library to stream/load NLP datasets |
| `transformers` | Core library — contains Flan-T5, tokenizer, Trainer |
| `sentencepiece` | Tokenizer backend used by T5 (subword tokenisation) |
| `evaluate` | HuggingFace metrics library (ROUGE, BLEU, etc.) |
| `rouge-score` | Backend for the ROUGE metric |
| `accelerate` | Distributed training backend used by Trainer internally |
| `peft` | **Parameter-Efficient Fine-Tuning** — provides `LoraConfig`, `get_peft_model` |
| `bitsandbytes` | 4/8-bit quantisation (included for compatibility even if not used here) |
| `wandb` | **Weights & Biases** — experiment tracking dashboard |
| `matplotlib` | Plotting the loss curve |

> **Note**: The `!` at the start is a Colab/Jupyter shell escape. In a regular `.py` file you'd run `pip install` separately in the terminal.

---

## 3. CELL 2 — Configuration & Hyperparameters

This is the most important cell to understand deeply. Every number here was chosen deliberately.

### 3.1 Paths and Infrastructure

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/FlanT5_Finetune")
```

All outputs go to Google Drive so they survive the Colab session ending. Colab's `/content/` directory is wiped after disconnect.

```python
d.mkdir(parents=True, exist_ok=True)
```

`parents=True` creates any missing intermediate directories. `exist_ok=True` doesn't raise an error if the directory already exists. This is defensive, idempotent setup.

---

### 3.2 Sequence Length Limits

```python
MAX_INPUT_LENGTH  = 512
MAX_TARGET_LENGTH = 256
```

**Why 512 for input?**
- T5 was pre-trained with a maximum of 512 tokens — going above doesn't help since positional embeddings beyond 512 were never trained
- The SPIDER schema strings can be long but almost always fit in 512 tokens
- Verified in `preprocess_data.py` CELL 6: >95% of examples fit within 512 tokens

**Why 256 for output?**
- Python functions in MBPP are typically 50–150 tokens
- SQL queries in SPIDER are typically 30–100 tokens
- 256 is a safe upper bound with minimal wasted computation
- Longer outputs dramatically increase VRAM during autoregressive generation at eval time

---

### 3.3 Training Hyperparameters — Deep Dive

```python
NUM_EPOCHS    = 5
BATCH_SIZE    = 8
GRAD_ACCUM    = 4
LEARNING_RATE = 3e-4
WARMUP_RATIO  = 0.05
```

#### `NUM_EPOCHS = 5`

The number of times the entire training dataset is passed through the model.

- **Too few**: model underfits — hasn't seen enough examples
- **Too many**: model overfits — memorises training data
- **5 epochs** is standard for instruction-tuned models being PEFT finetuned. The pre-trained weights act as a strong regulariser, so overfitting is less of a concern than training from scratch.

#### `BATCH_SIZE = 8`

The number of training examples processed in one forward+backward pass.

- **Larger batches** → faster training, but use more VRAM
- **Smaller batches** → noisier gradient estimates, sometimes better generalisation
- `BATCH_SIZE = 8` with 512-token inputs fits comfortably on a T4 GPU (16 GB VRAM) with LoRA

#### `GRAD_ACCUM = 4` → Effective batch size = 32

**Gradient accumulation** is a memory trick. Instead of computing the gradient on 32 examples at once, we:

1. Forward pass 8 examples → compute gradient → **don't update weights yet**
2. Forward pass another 8 → **add** gradient to the accumulation buffer
3. Repeat for 8 more, then 8 more
4. After 4 batches (32 examples total), **now** update the weights

Mathematically equivalent to a batch of 32, using VRAM of only batch size 8.

```
Without GRAD_ACCUM: update weights every 8 examples
With GRAD_ACCUM=4:  update weights every 32 examples
```

**Why does larger effective batch size matter?** With tiny batches, gradient estimates are noisy. A batch of 32 gives a more accurate direction for the gradient step, leading to smoother convergence.

> ⚠️ **Smoke test tip**: With 64 examples and `GRAD_ACCUM=4` you get only 2 optimizer steps per epoch. Set `GRAD_ACCUM=1` for subset tests to see meaningful training dynamics.

#### `LEARNING_RATE = 3e-4`

Controls the size of each weight update step:

```
new_weight = old_weight − learning_rate × gradient
```

- **Too large**: gradients overshoot — loss explodes or oscillates
- **Too small**: training progresses too slowly to converge in 5 epochs
- `3e-4` is the **standard LoRA learning rate** used in the original QLoRA paper and widely validated for seq2seq tasks. It is much higher than full finetuning LRs (`1e-5` to `5e-5`) because only the tiny LoRA matrices are being updated — large steps won't destabilise the 250M frozen parameters.

#### `WARMUP_RATIO = 0.05`

At the start of training, LoRA weights are random and gradients can be large and noisy. **Warmup** ramps the learning rate from ~0 up to `3e-4` over the first 5% of training steps, then linearly decays it:

```
LR
 ▲
 |        ╱╲
 |       ╱  ╲
 |      ╱    ╲___________
 └──────────────────────── steps
       warmup (5%)
```

This prevents the very first weight updates from being too large and destabilising the adapter initialisation.

---

### 3.4 LoRA Hyperparameters — Deep Dive

```python
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q", "v"]
```

#### `LORA_R = 16` (Rank)

The bottleneck dimension of the LoRA matrices A and B.

| Rank | Use case | Trainable params (per attention layer) |
|---|---|---|
| r = 4 | Lightweight style adaptation | ~6k |
| r = 8 | Simple classification tasks | ~12k |
| **r = 16** | **Code generation, reasoning** | **~24k** |
| r = 64 | Complex multi-task adaptation | ~98k |

With `r=16` targeting q and v across all T5-base layers: **~3.5M trainable params out of 250M (1.4%)**.

#### `LORA_ALPHA = 32` — Scaling Factor

The weight update is scaled by `alpha / r`:

```
W_new = W_frozen + (B × A) × (alpha / r)
                            = (32 / 16) = 2.0
```

`alpha = 2 × r` is the most common default. This convention decouples the adapter's "strength" from the learning rate, making it easier to search over `r` while keeping `alpha/r = 2` constant.

#### `LORA_DROPOUT = 0.05`

Randomly zeros 5% of adapter activations during training — a regularisation technique preventing the small adapter matrices from overfitting to the training data. 0.05 is conservative; Flan-T5's pre-training already provides strong regularisation.

Dropout is **disabled** at inference time (standard PyTorch behaviour with `model.eval()`).

#### `LORA_TARGETS = ["q", "v"]`

Specifies which weight matrices inside the model to inject LoRA into:

| Name | Role | Inject? | Reason |
|---|---|---|---|
| `q` | Query — controls *what* the model looks for | ✅ | Key for task adaptation |
| `k` | Key — controls *what is searchable* | ❌ | Marginal gain |
| `v` | Value — controls *what information flows through* | ✅ | Key for output quality |
| `o` | Output projection | ❌ | Marginal gain, more params |

Research (original LoRA paper) shows q+v captures the most useful adaptation signal for instruction-following tasks.

> ⚠️ **Critical**: Flan-T5 calls these `"q"` and `"v"` — NOT `"q_proj"` / `"v_proj"` (which is LLaMA naming). Getting this wrong causes PEFT to attach LoRA to **zero layers** with no error message. Always verify with `model.print_trainable_parameters()`.

---

## 4. CELL 3 — Helper Functions

### 4.1 `read_jsonl(path)`

```python
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```

Reads a **JSONL** (JSON Lines) file — each line is a complete JSON object. Simple, streaming-friendly, robust to corrupt files (one bad line only loses one example).

---

### 4.2 `normalize_sql(sql)`

```python
def normalize_sql(sql):
    sql = sql.replace("<=", " LESS_EQUAL ")
    sql = sql.replace(">=", " GREATER_EQUAL ")
    sql = sql.replace("<",  " LESS ")
    sql = sql.replace(">",  " GREATER ")
    return sql
```

The T5 SentencePiece tokenizer tokenizes `<` and `>` inconsistently (sometimes as their own token, sometimes merged into other tokens). Replacing them with word tokens makes vocabulary usage more standard and avoids potential gradient instability. Kept as a defensive measure from the QLoRA debugging phase.

---

### 4.3 `Seq2SeqCodeDataset` — The PyTorch Dataset

```python
class Seq2SeqCodeDataset(Dataset):
    def __init__(self, examples, tokenizer, max_input_length=512, max_target_length=256):
        ...
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ...
```

**PyTorch `Dataset` protocol**: Any class implementing `__len__` and `__getitem__` can be used as a data source. The Trainer calls `__getitem__(i)` to fetch individual examples.

#### `__getitem__` — Step by Step

**Tokenise the input (encoder side)**:
```python
enc = self.tokenizer(
    ex["input"],
    max_length=512,
    padding="max_length",   # pad shorter sequences to exactly 512 tokens
    truncation=True,        # cut longer sequences at 512 (silent)
    return_tensors="pt",    # return PyTorch tensors
)
```

All examples in a batch must be the same length for GPU matrix operations. `padding="max_length"` adds special `[PAD]` tokens to short sequences. `truncation=True` silently cuts long sequences — if >512 tokens, the later tokens are lost.

**Tokenise the targets (decoder side)**:
```python
dec = self.tokenizer(
    text_target=ex["output"],   # "text_target=" activates decoder mode
    max_length=256,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
```

Using `text_target=` (not just the text as a positional argument) activates the T5 tokenizer's **decoder mode**, which adds the correct `decoder_start_token_id` that T5 expects before every target sequence.

**The -100 label mask** — The Most Important Detail:
```python
label_ids = dec["input_ids"].squeeze().clone()
label_ids[label_ids == self.tokenizer.pad_token_id] = -100
```

The cross-entropy loss computes: `-log P(correct_token)` at every position. We do **not** want to penalise the model for padding positions — they carry no information.

PyTorch's `CrossEntropyLoss` has a built-in convention: **positions with label value `-100` are completely ignored**. After this line, the labels tensor looks like:

```
[3862, 9, 7, 5, 2, -100, -100, ..., -100]
 ← actual code/SQL tokens →  ← padding, ignored in loss →
```

**Return value** — what the Trainer receives for each example:
```python
{
  "input_ids":      tensor(512,)  # tokenised prompt
  "attention_mask": tensor(512,)  # 1=real token, 0=padding
  "labels":         tensor(256,)  # tokenised target, padding → -100
}
```

---

### 4.4 `make_compute_metrics` — ROUGE Evaluation

```python
def make_compute_metrics(tok):
    def compute_metrics(eval_preds):
        ...
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"rouge1": ..., "rougeL": ...}
    return compute_metrics
```

**Why a closure (function returning a function)?** The Trainer API requires `compute_metrics` to accept only `(eval_preds)`. But we also need the tokenizer inside. By closing over `tok`, the inner function has access to the tokenizer without it being passed as an argument.

**ROUGE explained**:
- **ROUGE-1**: What fraction of words in the reference appear in the prediction?
- **ROUGE-L**: What is the longest common subsequence (as a fraction of reference length)?

```
Reference : "SELECT name FROM employees WHERE salary > 50000"
Prediction: "SELECT name FROM employees WHERE salary GREATER 50000"
ROUGE-1   : 8/9 words match ≈ 0.89
```

> **Why ROUGE is only a proxy**: A SQL query with the right SELECT but wrong WHERE still gets decent ROUGE, but gives wrong answers. Final evaluation uses pass@1 (execute and check) for MBPP and Exact Set Match (normalise then compare) for SPIDER.

**`np.clip(predictions, 0, tok.vocab_size - 1)`**: When generating with `predict_with_generate`, occasional out-of-range token IDs can appear at vocab boundaries. Clipping prevents index-out-of-range errors during `batch_decode`.

---

## 5. CELL 4 — Load & Merge Data

```python
all_examples = mbpp_examples + spider_examples
rng = random.Random(SEED)
rng.shuffle(all_examples)
```

**Why shuffle?** The raw files are ordered — all MBPP first, then all SPIDER. Without shuffling, the last 10% eval split would contain only SPIDER examples.

**Why `random.Random(SEED)` not `random.shuffle()`?** Creating an isolated RNG with a fixed seed guarantees the exact same shuffle every run, regardless of what other random operations ran before. The global `random` module's state can be affected by other code.

```python
split_idx      = int(len(all_examples) * 0.90)
train_examples = all_examples[:split_idx]
eval_examples  = all_examples[split_idx:]
```

This **monitoring split** is separate from the held-out test sets (`mbpp_test.jsonl`, `spider_test.jsonl`). It's used only for in-training loss and ROUGE monitoring, not for the final reported results.

---

## 6. CELL 5 — Load Model & Apply LoRA

### Load the Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

`AutoTokenizer` inspects the model config and loads the correct class. For Flan-T5 this is a `T5Tokenizer` backed by **SentencePiece** — a subword algorithm:

```
"unbelievable" → ["▁un", "believ", "able"]
"SELECT"       → ["▁SELECT"]  (frequent → single token)
"fizzlebop"    → ["▁fi", "zzle", "bop"]  (rare → split)
```

This handles completely unknown words gracefully.

### Load the Base Model

```python
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # full 32-bit precision
    device_map="auto",
)
model.config.use_cache = False
```

**`torch_dtype=torch.float32`**: Full precision — stable for backpropagation. Uses ~1 GB VRAM. The code comment says "fp16" but the actual code correctly uses fp32 (a deliberate fix from the QLoRA debugging phase where fp16 caused NaN losses).

**`device_map="auto"`**: HuggingFace `accelerate` automatically places each model layer on GPU if available, falling back to CPU. No manual `.to(device)` calls needed.

**`use_cache = False`**: T5 caches Key-Value tensors between decoder timesteps for fast inference. During training with gradient checkpointing, the backward pass recomputes activations from earlier points — the cached tensors conflict with this, causing NaN gradients in decoder cross-attention layers. Disabling the cache prevents this.

### Apply LoRA

```python
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=LORA_R,               # rank = 16
    lora_alpha=LORA_ALPHA,  # scaling = 32
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",            # don't train biases
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: 3,538,944 || all params: 251,116,800 || trainable%: 1.41
```

`get_peft_model()` modifies the model in-place — it finds every module named `"q"` and `"v"` in every attention layer and wraps it in a `LoraLayer`:

```
Before: model.encoder.block[0].layer[0].SelfAttention.q  (frozen W)
After:  LoraLayer(
            W (frozen, no gradient),
            lora_A: Linear(64, 16, bias=False),   ← trainable
            lora_B: Linear(16, 768, bias=False),  ← trainable (init to 0)
        )
```

**`bias="none"`**: Biases are not added to the LoRA layers. They add negligible expressiveness for the parameter count. Keeping `bias="none"` makes the adapter file smaller and loading faster.

---

## 7. CELL 6 — Sanity Check

Always verify the data pipeline before spending GPU time training:

```python
sample = train_dataset[0]

# 1. Input should show the complete formatted prompt
print(tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
# → "Generate Python: Write a function that adds two numbers..."

# 2. Labels: real tokens first, then -100 for padding
print(sample["labels"][:20].tolist())
# → [3862, 9, 7, 1, 52, 2, -100, -100, ...]

# 3. Decoded target: should be the clean Python/SQL code
valid = sample["labels"][sample["labels"] != -100]
print(tokenizer.decode(valid, skip_special_tokens=True))
# → "def add(a, b):\n    return a + b"
```

This catches common bugs: input/target swapped, missing -100 masking, wrong tokenizer loaded.

---

## 8. CELL 7 — Training

### W&B Initialisation

```python
wandb_run = wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={...})
```

Calling `wandb.init()` before the Trainer starts records all hyperparameters and gives you the dashboard URL immediately. Every metric logged by the Trainer (loss, ROUGE, learning rate) is automatically sent to W&B via `report_to="wandb"`.

---

### `Seq2SeqTrainingArguments` — Every Flag Explained

#### Batch size & gradient accumulation
```python
per_device_train_batch_size=8,
per_device_eval_batch_size=8,
gradient_accumulation_steps=4,   # effective batch = 32
```

#### Learning rate schedule
```python
learning_rate=3e-4,
warmup_ratio=0.05,              # 5% warmup
lr_scheduler_type="linear",     # linear decay after warmup
max_grad_norm=1.0,              # clip gradient norm
```

**`max_grad_norm=1.0`**: If the gradient vector's total magnitude (L2 norm) exceeds 1.0, scale it down proportionally. Prevents a single bad batch from causing a catastrophically large weight update. Standard for all Transformer training.

#### Precision
```python
fp16=False,    # disabled — backward pass overflow occurred with fp16
bf16=False,    # T4 doesn't support bfloat16
```

The script runs in **full fp32** for numerical stability. fp16 uses half the memory but can overflow (producing NaN) when logit values or gradients get very large during seq2seq generation.

#### Memory optimisation
```python
gradient_checkpointing=True,
gradient_checkpointing_kwargs={"use_reentrant": False},
```

**Gradient checkpointing** is a memory vs. speed trade-off:

| | Memory usage | Training speed |
|---|---|---|
| Without | Higher (~35% more) | Faster |
| With | Lower | ~20% slower |

It works by only storing "checkpoint" activations at certain layers. During backpropagation, it recomputes the missing intermediate activations on demand instead of reading them from memory.

**`use_reentrant=False`**: PyTorch has two gradient checkpointing implementations. The default `reentrant=True` has known incompatibilities with some seq2seq autograd graphs. `use_reentrant=False` is the newer, safer implementation — always use this for T5-family models.

#### Generation at evaluation time
```python
predict_with_generate=True,
generation_max_length=256,
```

**`predict_with_generate=True`** — Critical for ROUGE metrics to work. Without it, the Trainer computes only the teacher-forced cross-entropy loss at eval time (input to model: correct previous token; output: predict next token). With it, the model actually **generates** output autoregressively (generates token by token, feeding its own predictions back in). The generated text is what `compute_metrics` (ROUGE) operates on.

#### Checkpointing strategy
```python
eval_strategy="epoch",          # evaluate after every epoch
save_strategy="epoch",          # save checkpoint after every epoch
save_total_limit=3,             # keep only 3 most recent checkpoints
load_best_model_at_end=False,   # unreliable with PEFT — use W&B instead
```

**`load_best_model_at_end=False`**: In standard finetuning this would reload the checkpoint with the lowest eval loss at the end of training. For PEFT models, dynamically reloading from a saved checkpoint is unreliable. Instead: monitor the W&B dashboard, identify the epoch with lowest eval_loss, and manually load that adapter from `CHECKPOINT_DIR`.

#### Logging
```python
logging_steps=1,    # log after every single optimizer step
report_to="wandb",
```

`logging_steps=1` ensures training loss is always visible, even in tiny subset runs where an epoch has only 2 optimizer steps.

---

### The Trainer

```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=Subset(train_dataset, range(64)),   # smoke test — 64 examples
    eval_dataset=Subset(eval_dataset, range(64)),     # smoke test
    processing_class=tokenizer,
    compute_metrics=make_compute_metrics(tokenizer),
)
```

> ⚠️ **For the full training run**: Replace `Subset(train_dataset, range(64))` with just `train_dataset` and `eval_dataset`.

**`processing_class=tokenizer`**: The modern replacement for the deprecated `tokenizer=` argument. Tells the Trainer how to handle data collation and decoding internally.

---

### What Happens During `trainer.train()`

For every batch in every epoch:

```
1. FORWARD PASS
   input_ids + attention_mask → Encoder → context vectors
   labels (teacher-forced) → Decoder → logits over vocabulary
   logits vs labels → cross-entropy loss (ignoring -100 positions)

2. BACKWARD PASS (Backpropagation)
   ∂loss/∂lora_A and ∂loss/∂lora_B computed for every LoRA layer
   Frozen weights: requires_grad=False → no gradient computed

3. GRADIENT ACCUMULATION
   Add batch gradient to accumulation buffer
   If buffer has 4 batches' worth: proceed to step 4
   Otherwise: load next batch and go back to step 1

4. OPTIMIZER STEP (AdamW)
   Adaptive learning rate per LoRA parameter
   Update A and B matrices
   Clear accumulation buffer

5. LR SCHEDULER STEP
   Advance warmup/decay schedule
```

At the end of each epoch:
- Eval batch generated with `predict_with_generate=True`
- ROUGE computed over generated vs reference text
- All metrics logged to W&B

---

### Save the Adapter

```python
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
```

For a PEFT model, `save_pretrained()` saves **only the LoRA adapter weights** (~14 MB), not the full 250M-parameter model. To reload later:

```python
from peft import PeftModel
base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
```

The **raw log history** printout verifies the training loss is finite and non-zero — the most basic sanity check after training:
```python
for entry in trainer.state.log_history:
    if "loss" in entry and "eval_loss" not in entry:
        print(f"  step {entry['step']:>4} | loss = {entry['loss']}")
```

---

## 9. CELL 8 — Merge, Plot, Save

### Merge the Adapter into the Base Model

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained(str(MERGED_DIR))
```

`merge_and_unload()` computes for every LoRA layer:
```
W_merged = W_frozen + (B × A) × (alpha / r)
```
and returns a standard `AutoModelForSeq2SeqLM` with no PEFT dependency. The evaluation scripts can then just:
```python
model = AutoModelForSeq2SeqLM.from_pretrained(MERGED_DIR)
```

### Loss Curve Parsing

```python
train_steps  = [e["step"] for e in log_history if "loss" in e and "eval_loss" not in e]
eval_steps   = [e["step"] for e in log_history if "eval_loss" in e]
```

The condition `"eval_loss" not in e` separates training log entries from evaluation entries (eval log entries sometimes also contain a `"loss"` key).

**A healthy loss curve**:
```
Loss
 │\
 │ \
 │  \────
 │       \────
 └──────────────── epochs
  Train   ↑Eval should track similarly or be slightly higher
```

If **eval loss starts rising while train loss falls** → overfitting. Use the adapter from the epoch just before eval loss started rising.

---

## 10. Key Concepts Glossary

| Term | Plain English |
|---|---|
| **Token** | A subword unit — the basic unit the model reads and writes. T5 uses ~32,000 tokens. |
| **Tokenizer** | Converts text ↔ integer token IDs. Must exactly match the model. |
| **Encoder** | Reads the entire input at once, produces contextual hidden representations |
| **Decoder** | Generates output one token at a time using encoder context and its own past outputs |
| **Cross-attention** | How the decoder "reads" encoder output at each generation step |
| **Teacher forcing** | During training, the decoder is given the *correct* previous token, not its own prediction |
| **Cross-entropy loss** | Measures how wrong the model's predicted probability distribution is vs. the correct answer |
| **Gradient** | A vector pointing in the direction that most increases the loss |
| **Backpropagation** | Algorithm to compute gradients efficiently using the chain rule |
| **AdamW** | Optimizer — adaptive learning rates per parameter + weight decay regularisation |
| **Weight decay** | Regularisation that penalises large weights to prevent overfitting |
| **LoRA** | Freezes base weights; adds tiny trainable A and B delta matrices alongside them |
| **VRAM** | GPU memory — the bottleneck for training large language models |
| **fp32** | 32-bit float — full precision, 4 bytes per number, stable |
| **fp16** | 16-bit float — half precision, 2 bytes, can overflow to NaN |
| **Gradient checkpointing** | Saves VRAM by recomputing activations during backprop instead of storing them |
| **KV-cache** | Caches key-value tensors for fast autoregressive inference. Must be disabled during training with gradient checkpointing. |
| **ROUGE** | word-overlap metric between generated text and reference |
| **JSONL** | one JSON object per line — simple, streaming-friendly format |
| **Checkpoint** | Saved snapshot of model weights at a training step |
| **Adapter** | The small LoRA weights (~14 MB) that can be loaded on top of the frozen base model |
| **Effective batch size** | `BATCH_SIZE × GRAD_ACCUM` — what the gradient mathematically "sees" |
| **Closure** | A function that "closes over" and captures variables from its enclosing scope |

---

## 11. Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| `logging_steps` too large for subset | Training Loss = `0.000000` in epoch table | Set `logging_steps=1` |
| `GRAD_ACCUM=4` with ≤64 examples | Only 2 optimizer steps/epoch, no visible learning | Set `GRAD_ACCUM=1` for smoke tests |
| Wrong `target_modules` names | PEFT applies to 0 layers, no error raised | Check `model.print_trainable_parameters()` — must be >0 |
| `use_cache=True` + gradient checkpointing | NaN in decoder cross-attention gradients | Set `model.config.use_cache = False` before applying LoRA |
| fp16 + seq2seq generation | NaN eval_loss | Use `fp16_full_eval=False` or switch to fp32 |
| Missing -100 label masking | Loss computed over padding tokens → wrong gradients | Verify `label_ids[label_ids == pad_id] = -100` |
| `Subset(...)` left in for full run | Model only trains on 64 examples | Replace with `train_dataset` / `eval_dataset` |
| Input > 512 tokens | Schema columns silently truncated | Check stats from `preprocess_data.py` CELL 6 |
| Forgetting `use_cache=False` | Silent gradient instability during training | Always set before `trainer.train()` |
