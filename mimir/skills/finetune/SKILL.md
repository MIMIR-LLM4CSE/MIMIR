---
name: finetune
description: Configure, launch, and iteratively improve a LoRA fine-tuning run using the finetune server tools.
disable-model-invocation: false
---

You are running a LoRA fine-tuning optimization loop.

The goal is to produce a fine-tuned model that satisfies the user's constraints
(memory budget, loss target, throughput requirement, etc.) by iterating on the
runner configuration.

## Setup (first time only)

1. Check the current configuration: `ft_config_get()`
2. Set required fields:
   ```
   ft_config_set(
       model_id="...",
       train_data="/abs/path/to/train.jsonl",
       val_data="/abs/path/to/val.jsonl",   # optional
       lora_r=8,
       epochs=3,
       batch_size=8,
       lr=2e-4,
   )
   ```
3. Inspect training data to verify format: `ft_data_inspect(train_path=..., val_path=...)`

## Training loop

Repeat until all user constraints are satisfied:

1. **Launch**: `ft_run(confirm=True)` (local) or `ft_run_slurm(partition=..., confirm=True)` (HPC).
2. **Monitor**: `ft_status()` — wait until state is 'done' or 'crashed'.
   - If 'crashed': `ft_log_read(tail_lines=100)` to diagnose.
3. **Inspect results**: `ft_metrics_parse()`
   - Key metrics: `peak_vram_mb`, `train_loss`, `eval_loss`, `throughput_sps`, `train_runtime_s`.
4. **If constraints not met**, choose the appropriate fix:

   | Problem | Action |
   |---|---|
   | OOM / too much VRAM | Reduce `batch_size`, increase `lora_r` only if needed, or `ft_config_set(precision="fp16")` |
   | Loss too high | Increase `epochs`, lower `lr`, or read `_ft_runner.py` and add gradient checkpointing |
   | Too slow | Reduce `max_length`, increase `batch_size` if VRAM allows |
   | Need architecture change | Read `_ft_runner.py`, modify optimizer/scheduler/dtype, re-run |

5. **Compare runs**: `ft_runs_diff()` to confirm improvement before further changes.

## Modifying the runner

For changes beyond config knobs (optimizer, tokenizer, data preprocessing, architecture),
read `_ft_runner.py` with `read_file(...)` first, then use `replace_in_file(...)` to apply
a targeted change. After a session restart the runner is automatically reset from its canonical
copy — use `ft_runner_promote(confirm=True)` to persist a runner change across sessions.

## Rules

- Always call `ft_config_get()` before the first run to confirm settings.
- Make one focused change per run cycle.
- Read `_ft_runner.py` before any runner modification. Never edit blind.
- Do NOT use `bash_run` to run training — always use `ft_run` or `ft_run_slurm`.
- Only one run may be active at a time — call `ft_stop(confirm=True)` before launching a new one.

## When to stop

Stop and summarize when:
- All user constraints are satisfied, or
- After 5+ iterations with no progress — report what was tried and what the remaining blockers are.
