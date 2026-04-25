# Tutorial — Fine-tuning Gemma-4 26B A4B on Alpaca with Unsloth + LoRA

This is the conceptual companion to the README. The README is "do this and
it works"; this file is "and here is *why* each step looks the way it does".
If you've never done supervised fine-tuning (SFT) of an LLM before, read it
in order — every section feeds into the next.

## Contents

1. What we're doing, in one paragraph
2. Why Unsloth + LoRA + 4-bit quantisation
3. The Alpaca Cleaned dataset
4. The Gemma-4 chat template (and why it matters)
5. From raw rows to "what the model sees" (run `inspect_format.py`)
6. The fine-tuning loop in `finetune.py`
7. Reading the training curves (`plot_training.py`)
8. Inference with the LoRA adapter (`inference.py`)
9. Evaluating against ground truth (`evaluate.py`)
10. Common errors / Kelvin2 ROCm gotchas
11. Going further

---

## 1. What we're doing, in one paragraph

We start from `unsloth/gemma-4-26b-a4b-it`, a 26-billion-parameter
instruction-tuned mixture-of-experts model. We freeze almost all of it,
attach a small set of trainable **LoRA adapters** to the attention
projections, and update only those (~tens of MB of weights) on a few
hundred examples from the Alpaca Cleaned instruction-following dataset.
The result is a model that "speaks the same language" but with behaviour
shaped by the new dataset. We never touch the base weights — at inference
time we re-load the base model and apply the adapter on top.

This is the standard recipe for "personalising" a large open-weights
chat model on a single GPU.

## 2. Why Unsloth + LoRA + 4-bit quantisation

Three orthogonal tricks stack here:

- **4-bit quantisation (bitsandbytes / NF4)** — store the frozen base
  weights in 4 bits each instead of 16. ~4× memory reduction. Fine to
  forward-/backward-pass through, with a small accuracy hit that LoRA
  more than compensates for in practice.
- **LoRA (Low-Rank Adapters)** — instead of updating the full weight
  matrix `W` (shape `d_in × d_out`), train two small matrices
  `A` (`d_in × r`) and `B` (`r × d_out`) such that the *update* is
  `ΔW = B @ A`. With `r=16` the trainable parameter count drops by
  ~1000× compared to a full fine-tune.
- **Unsloth** — a thin shim over Hugging Face Transformers + PEFT + TRL
  that swaps in fused/optimised kernels, manages 4-bit loading, and adds
  a `FastModel.from_pretrained(...) / FastModel.get_peft_model(...)`
  ergonomic API. It also patches gradient checkpointing for memory.

Net effect: a 26B-parameter model trains comfortably on a single MI300X
(192 GB), and the saved adapter is tiny enough to ship next to your code.

## 3. The Alpaca Cleaned dataset

`yahma/alpaca-cleaned` is a hand-cleaned version of the 2023 Stanford
Alpaca dataset. Each row has three string fields:

- `instruction` — what the user is asking, e.g. *"Translate this to French"*.
- `input` — optional extra context the instruction refers to,
  e.g. the English sentence to translate. Often empty.
- `output` — the gold response.

It is *the* canonical "teach an LLM to follow instructions" dataset and
is tiny enough (~50 k rows) that we can run a meaningful tutorial on it.

`prepare_dataset.py` downloads it and writes:

```text
datasets/alpaca-cleaned/alpaca_cleaned.jsonl       # raw rows, for inspection
datasets/alpaca-cleaned/alpaca_gemma4_sft.jsonl    # prompt/completion pairs
datasets/alpaca-cleaned/preview.md                 # human-readable preview
```

## 4. The Gemma-4 chat template (and why it matters)

Gemma-4 is a *chat* model, so it expects every conversation turn to be
wrapped in special tokens that mark the role and end of the turn:

```text
<|turn>user
your message goes here<turn|>
<|turn>model
the model's response goes here<turn|>
```

If you fine-tune on plain `instruction + output` text (no template) the
model unlearns its turn boundaries and starts producing nonsense at
inference time. The single most common cause of "my fine-tune broke the
chat behaviour" is feeding it un-templated text.

`prepare_dataset.py` formats every row into:

```json
{
  "prompt":     "<|turn>user\n{instruction}\n\nInput:\n{input}<turn|>\n<|turn>model\n",
  "completion": "{output}<turn|>\n"
}
```

The split into `prompt` and `completion` matters: TRL's `SFTTrainer`
masks the prompt tokens so the loss is computed *only* on the completion
(see next section).

## 5. From raw rows to "what the model sees"

Run:

```bash
python inspect_format.py --num-examples 2
```

For each example you'll see five sections:

1. **Raw Alpaca row** — the `instruction` / `input` / `output` strings.
2. **SFT pair** — the `prompt` and `completion` strings the trainer concatenates.
3. **Token counts** — how many tokens belong to the prompt (loss-masked) vs
   the completion (loss target).
4. **Decoded full input** — what the model literally receives as input
   ids: `prompt + completion`, after tokenisation and decoding.
5. **Loss target** — the substring the model is graded on producing,
   one token at a time, with cross-entropy loss.

Add `--show-token-table` to see every token annotated with `*` (in loss)
or blank (masked).

The mental model:

```text
input_ids:  [   prompt tokens   ][ completion tokens ]
labels:     [ -100, -100, ..., -100 ][ completion tokens ]
                  ^                            ^
        ignored by the loss           cross-entropy targets
```

This is what `completion_only_loss=True` in `SFTConfig` does. If you
omit it, the loss also includes the prompt tokens and the model spends
most of its capacity learning to predict its *own input*, which it
already does perfectly — your effective signal-to-noise ratio drops.

## 6. The fine-tuning loop in `finetune.py`

Highlights of the code:

```python
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-26b-a4b-it",
    max_seq_length=2048,
    load_in_4bit=True,                # NF4 quantisation
)

model = FastModel.get_peft_model(
    model,
    r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention only
    use_gradient_checkpointing="unsloth",                     # save memory
)

training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,    # effective batch size = 8
    max_steps=30,                     # tutorial-tiny — increase for real runs
    learning_rate=2e-4,
    bf16=True,
    completion_only_loss=True,        # see section 5
    report_to="tensorboard",          # so plot_training.py has data
)

trainer = SFTTrainer(model=model, processing_class=tokenizer,
                     train_dataset=dataset, args=training_args)
trainer.train()
model.save_pretrained("outputs/.../final_adapter")
```

Key knobs to play with:

| Flag                      | Default | Why you'd change it                         |
| ------------------------- | ------- | ------------------------------------------- |
| `--max-steps`             | 30      | Bigger run; meaningful learning starts ~100 |
| `--train-samples`         | 512     | `0` to use the entire ~50k Alpaca rows      |
| `--lora-r` / `--lora-alpha` | 16/16  | Higher r = more capacity, more memory       |
| `--learning-rate`         | 2e-4    | LoRA tolerates higher LRs than full FT      |
| `--grad-accum`            | 8       | Larger effective batch without more memory  |

## 7. Reading the training curves

`plot_training.py` reads `trainer_state.json` (always written by the HF
Trainer) and the TensorBoard event files in `outputs/.../runs/<timestamp>/`,
and renders three PNGs into `outputs/.../plots/`:

- `loss.png` — the cross-entropy on completion tokens, one point per
  `logging_steps` step. Should trend downward; a plateau is fine for
  small runs.
- `lr.png` — the actual learning-rate schedule (warmup → linear decay
  by default). Useful to confirm warmup actually happened.
- `grad_norm.png` — gradient norm before clipping. Sudden spikes mean
  unstable training; sustained flat lines near zero mean nothing is
  learning.

If you have TensorBoard installed and want the live UI:

```bash
tensorboard --logdir outputs/gemma4-26b-a4b-it-alpaca-lora/runs
```

## 8. Inference with the LoRA adapter

`inference.py` reproduces the *exact* prompt format from
`prepare_dataset.py` so the model sees the same chat template at inference
time as at training time:

```python
prompt = f"<|turn>user\n{user_message}<turn|>\n<|turn>model\n"
inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
generated = model.generate(**inputs, max_new_tokens=200,
                           temperature=0.7, top_p=0.9, do_sample=True)
response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).split("<turn|>")[0].strip()
```

Two important details:

- `model.load_adapter(adapter_dir)` is what attaches the LoRA weights
  on top of the 4-bit base model. The base model lookup is the slow part
  (downloading + 4-bit quantisation); attaching the adapter is instant.
- `tokenizer(text=prompt, ...)` passes the prompt as a *keyword* argument
  because Gemma-4's processor is multimodal and its first positional
  argument is `images`. Forgetting `text=` produces a confusing error.

Pass `--no-adapter` to compare against the un-fine-tuned base.

## 9. Evaluating against ground truth

`evaluate.py` holds out the LAST `--num-samples` rows of the dataset (so
they are not used for training under the default `--train-samples 512`
cap), generates responses for each, and writes:

- `outputs/evaluation_report.md` — a markdown side-by-side of
  `instruction → ground truth → fine-tuned response → (optionally) base
  response`.
- `outputs/evaluation_metrics.json` — exact-match plus character- and
  word-level recall against ground truth.

The metrics are deliberately simple. For a real eval you should reach for
ROUGE-L, BERTScore, or LLM-as-a-judge; the goal here is to give you a
single command that produces *some* number per run so you can tell good
runs from bad ones.

`--include-base` re-loads the base model and runs the same prompts so you
can quantify what the fine-tune actually changed.

## 10. Common errors / Kelvin2 ROCm gotchas

These are baked into `finetune.py` / `inference.py` as monkey-patches.
Keep them if you copy the scripts elsewhere.

1. **`AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'`**
   torch 2.9.1's inductor reads `binary.metadata.cluster_dims`, which
   ROCm Triton 3.6.0 does not expose. We disable compilation:
   `os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"` *before* importing Unsloth.

2. **`RuntimeError: grouped gemm is not supported on ROCM`**
   `torch._grouped_mm` exists on the ROCm wheel but isn't implemented.
   We monkey-patch `transformers.integrations.moe._can_use_grouped_mm`
   to return `False`, which routes Gemma-4's MoE expert call through the
   pure-Python fallback.

3. **`TypeError: cannot pickle 'ConfigModuleInstance' object`**
   `datasets >= 4.0` spawns a worker pool whenever `num_proc >= 1`, and
   the Unsloth-patched tokenizer captured by TRL's `tokenize_fn` carries
   a `torch._dynamo` `ConfigModuleInstance` that is not picklable. We set
   `dataset_num_proc=None` in `SFTConfig` to keep tokenisation in-process.

4. **`Unsloth cannot find any torch accelerator`**
   You are on a login node. SSH/`srun` into a GPU compute node first.

5. **`Dataset file not found`**
   Run `python prepare_dataset.py` first.

## 11. Going further

- **More data, more steps.** Try `--train-samples 0 --max-steps 500`.
  Watch `loss.png` flatten before deciding to stop.
- **Different target modules.** Add the MLP projections
  (`up_proj`, `gate_proj`, `down_proj`) to `target_modules` to give the
  model more places to learn. Doubles the trainable params.
- **Different dataset.** Anything with `instruction / input / output`
  fields drops in by changing the `--dataset` flag of `prepare_dataset.py`.
  For free-form chat datasets you'll need to adapt `to_gemma4_pair()`.
- **Merging the adapter.** Once you're happy, you can call
  `model.merge_and_unload()` to fold the LoRA delta back into the base
  weights and ship a single non-LoRA model. Note that the merge has to
  happen at full precision — load the base model in 16-bit for that.
- **Real eval.** Replace the toy metrics in `evaluate.py` with
  `evaluate.load("rouge")` from Hugging Face, or wire up an LLM judge.
