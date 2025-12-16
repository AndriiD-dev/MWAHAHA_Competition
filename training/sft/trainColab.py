import os
os.chdir("/content/MWAHAHA_Competition")
print("Current working dir:", os.getcwd())

import math
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from trl import SFTTrainer, SFTConfig

hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    dataset_path: str = "data/merged_sft_jokes.jsonl"
    output_dir: str = "qwen_lora_jokes"

    # sequence and training
    max_seq_length: int = 512          # shorter -> less memory, fine for jokes
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # evaluation
    eval_ratio: float = 0.1
    eval_strategy: str = "steps"
    eval_steps: int = 500

    # misc
    logging_steps: int = 10
    save_strategy: str = "steps"
    seed: int = 42


cfg = FinetuneConfig()


# ---------------------------------------------------------------------------
# Model + tokenizer (standard fp16 LoRA, no 4-bit)
# ---------------------------------------------------------------------------

def load_qwen_text(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        print("Using CUDA with fp16")
        torch_dtype = torch.float16
        device_map = "auto"
    else:
        print("CUDA not available, falling back to CPU fp32")
        torch_dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model, tokenizer


model, tokenizer = load_qwen_text(cfg.model_id)


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

peft_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "gate_proj", "down_proj",
    ],
)


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a multilingual stand-up comedian. "
    "You write short, original jokes in English. "
    "You ALWAYS obey the user’s constraints exactly (word inclusion, topic, language). "
    "You prefer concise setups and strong punchlines."
)


def build_text(example: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    example["text"] = text
    return example


def prepare_split(split_ds):
    split_ds = split_ds.map(build_text)
    cols = [c for c in split_ds.column_names if c != "text"]
    if cols:
        split_ds = split_ds.remove_columns(cols)
    return split_ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global model, tokenizer

    print("Loading dataset from:", cfg.dataset_path)
    raw_dataset = load_dataset("json", data_files={"train": cfg.dataset_path}, split="train")

    split = raw_dataset.train_test_split(cfg.eval_ratio, seed=cfg.seed)
    train_dataset = prepare_split(split["train"])
    eval_dataset = prepare_split(split["test"])

    print("Train size:", len(train_dataset))
    print("Eval size:", len(eval_dataset))

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,

        save_strategy=cfg.save_strategy,
        save_steps=cfg.eval_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,

        lr_scheduler_type="cosine",
        optim="adamw_torch",          # <--- standard AdamW, no 8-bit
        seed=cfg.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=False,                   # <--- disable bf16 AMP
        fp16=torch.cuda.is_available(),  # <--- enable fp16 AMP if CUDA

        report_to=[],
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=True,                 # pack short samples to reduce padding
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")
    print("Train metrics:", train_result.metrics)

    print("Running evaluation...")
    eval_metrics = trainer.evaluate()
    print("Eval metrics:", eval_metrics)

    if "eval_loss" in eval_metrics:
        try:
            print("Perplexity:", math.exp(eval_metrics["eval_loss"]))
        except OverflowError:
            print("Perplexity overflow")

    print("Saving adapter and tokenizer...")
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    full_out = os.path.abspath(cfg.output_dir)
    print("Training complete. Saved to:", full_out)

    zip_path = "/content/qwen_lora_jokes.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("Zipping model to:", zip_path)
    import subprocess
    subprocess.run(["zip", "-r", zip_path, full_out], check=True)


main()
