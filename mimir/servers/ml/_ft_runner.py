"""Standalone LoRA fine-tuning runner.

Invoked as a subprocess by server_finetune.py; not an MCP server.

Usage:
    python _ft_runner.py --run-dir <path>

Reads  <run-dir>/config.json, runs the full LoRA training loop,
writes <run-dir>/metrics.json on completion.
All output goes to stdout (captured as run.log by the server).

Structured log lines emitted for agent observation
---------------------------------------------------
Every line of interest starts with "[ft_runner]" and uses a
``key=value`` format so they are parseable by ft_metrics_parse():

  [ft_runner] epoch=<n> peak_vram_mb=<n> cpu_ram_mb=<n>
  [ft_runner] summary peak_vram_mb=<n> cpu_ram_mb=<n> \
      throughput_sps=<n> train_loss=<n> eval_loss=<n> \
      trainable_params=<n> total_params=<n> train_runtime_s=<n>

The agent can therefore:
  1. Run the training.
  2. Read structured summary metrics via ft_metrics_parse().
  3. Read this file via server_search / server_files.
  4. Modify this file (e.g. add gradient_checkpointing, change dtype,
     reduce batch size, switch optimizer) via server_files.
  5. Re-run and compare -- exactly the autoresearch loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _vram_mb() -> float:
    """Peak allocated GPU VRAM in MB, or 0 if CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    except Exception:
        pass
    return 0.0


def _cpu_ram_mb() -> float:
    """Resident set size of the current process in MB, or 0 if psutil is missing."""
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, 1)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA LLM fine-tuner")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory (must contain config.json; receives metrics.json).",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    config_path = os.path.join(run_dir, "config.json")

    if not os.path.isfile(config_path):
        print(f"[ft_runner] ERROR: config not found: {config_path}", flush=True)
        sys.exit(1)

    with open(config_path) as fh:
        cfg = json.load(fh)

    model_id       = cfg.get("model_id", "distilgpt2")
    train_data     = cfg["train_data"]
    val_data       = cfg.get("val_data", "")
    lora_r         = int(cfg.get("lora_r", 8))
    lora_alpha     = int(cfg.get("lora_alpha", 16))
    target_modules = cfg.get("target_modules", ["c_attn", "c_proj"])
    lora_dropout   = float(cfg.get("lora_dropout", 0.05))
    batch_size     = int(cfg.get("batch_size", 8))
    lr             = float(cfg.get("lr", 2e-4))
    epochs         = int(cfg.get("epochs", 3))
    max_length     = int(cfg.get("max_length", 256))
    output_dir     = cfg.get("output_dir") or os.path.join(run_dir, "model")
    # precision: "auto" | "fp32" | "fp16" | "bf16" | "int8"
    # "auto" -> fp16 when CUDA is available, fp32 otherwise
    precision      = cfg.get("precision", "auto").lower()

    print(f"[ft_runner] model_id={model_id}", flush=True)
    print(f"[ft_runner] train_data={train_data}", flush=True)
    print(f"[ft_runner] val_data={val_data or '(none)'}", flush=True)
    print(f"[ft_runner] lora_r={lora_r}  lora_alpha={lora_alpha}  target_modules={target_modules}", flush=True)
    print(f"[ft_runner] batch_size={batch_size}  lr={lr}  epochs={epochs}  max_length={max_length}", flush=True)
    print(f"[ft_runner] precision={precision}", flush=True)

    # ── int8 requires bitsandbytes — fail fast with a clear message ─────────
    if precision == "int8":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print(
                "[ft_runner] ERROR: precision=int8 requires bitsandbytes. "
                "Install with: pip install bitsandbytes",
                flush=True,
            )
            sys.exit(1)

    # ── deferred heavy imports (fast --help / syntax checks don't need them) ──
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

    # ── per-epoch memory callback ──────────────────────────────────────────
    class MemoryMetricsCallback(TrainerCallback):
        """Emits structured memory metrics after every epoch."""
        def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> None:
            epoch = int(state.epoch or 0)
            torch.cuda.reset_peak_memory_stats()  # reset so next epoch starts fresh
            print(
                f"[ft_runner] epoch={epoch}"
                f" peak_vram_mb={_vram_mb()}"
                f" cpu_ram_mb={_cpu_ram_mb()}",
                flush=True,
            )

    # ── dataset ───────────────────────────────────────────────────────────────
    data_files: dict[str, str] = {"train": train_data}
    if val_data:
        data_files["validation"] = val_data

    print("[ft_runner] Loading dataset...", flush=True)
    dataset = load_dataset("text", data_files=data_files)

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print("[ft_runner] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def _tokenize(examples: dict) -> dict:
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])

    # ── model + LoRA ──────────────────────────────────────────────────────────
    print("[ft_runner] Loading model...", flush=True)
    load_kwargs: dict = {}
    if precision == "int8":
        load_kwargs["load_in_8bit"] = True
    elif precision == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    # fp32 / auto: no dtype override
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # capture trainable / total param counts from the printed line
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ft_runner] trainable_params={trainable_params} total_params={total_params}", flush=True)

    # ── training args ────────────────────────────────────────────────────────
    eval_strategy = "epoch" if val_data else "no"
    _cuda = torch.cuda.is_available()
    use_fp16 = _cuda and precision in ("auto", "fp16")
    use_bf16 = _cuda and precision == "bf16"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy=eval_strategy,
        logging_steps=50,
        save_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
        callbacks=[MemoryMetricsCallback()],
    )

    # reset peak stats before training so the summary reflects training only
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()

    print("[ft_runner] Starting training...", flush=True)
    train_result = trainer.train()
    train_runtime_s = round(time.monotonic() - t0, 1)

    # ── collect metrics ────────────────────────────────────────────────────────
    metrics: dict = {
        k: round(v, 6) if isinstance(v, float) else v
        for k, v in train_result.metrics.items()
    }

    if val_data and "validation" in tokenized:
        eval_metrics = trainer.evaluate()
        metrics.update(
            {k: round(v, 6) if isinstance(v, float) else v for k, v in eval_metrics.items()}
        )

    # ── derived performance metrics ───────────────────────────────────────────
    total_samples = len(tokenized["train"]) * epochs
    throughput_sps = round(total_samples / train_runtime_s, 2) if train_runtime_s > 0 else 0.0
    peak_vram = _vram_mb()
    cpu_ram   = _cpu_ram_mb()

    metrics["peak_vram_mb"]     = peak_vram
    metrics["cpu_ram_mb"]       = cpu_ram
    metrics["throughput_sps"]   = throughput_sps
    metrics["train_runtime_s"]  = train_runtime_s
    metrics["trainable_params"] = trainable_params
    metrics["total_params"]     = total_params
    metrics["precision"]        = precision

    print(f"[ft_runner] Final metrics: {metrics}", flush=True)

    # ── structured summary line (parseable by ft_metrics_parse) ──────────────
    summary_parts = [
        f"peak_vram_mb={peak_vram}",
        f"cpu_ram_mb={cpu_ram}",
        f"throughput_sps={throughput_sps}",
        f"train_runtime_s={train_runtime_s}",
        f"trainable_params={trainable_params}",
        f"total_params={total_params}",
        f"precision={precision}",
    ]
    if "train_loss" in metrics:
        summary_parts.append(f"train_loss={metrics['train_loss']}")
    if "eval_loss" in metrics:
        summary_parts.append(f"eval_loss={metrics['eval_loss']}")
    print("[ft_runner] summary " + " ".join(summary_parts), flush=True)

    # ── write metrics.json ────────────────────────────────────────────────────
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[ft_runner] Metrics written to {metrics_path}", flush=True)

    # ── save model ────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[ft_runner] Model saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
