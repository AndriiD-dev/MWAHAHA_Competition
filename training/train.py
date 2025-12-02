# train_qwen_lora.py
"""
LoRA fine-tuning script for a text-only Qwen Instruct model on a jokes dataset.

The dataset is expected to be a JSONL file with one example per line:

    {"prompt": "...", "response": "..."}
    {"prompt": "...", "response": "..."}

`prompt`  – the input or context (for example, a setup or situation).
`response` – the desired model output (for example, the punchline).
"""

import math
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig


@dataclass
class FinetuneConfig:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    dataset_path: str = "data/merged_sft_jokes.jsonl"
    output_dir: str = "qwen_lora_jokes"

    # sequence and training
    max_seq_length: int = 1024
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
    eval_ratio: float = 0.05
    evaluation_strategy: str = "steps"
    eval_steps: int = 500

    # misc
    logging_steps: int = 10
    save_strategy: str = "steps"
    seed: int = 42


cfg = FinetuneConfig()


def load_qwen_text(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    model.config.use_cache = False
    return model, tokenizer


model, tokenizer = load_qwen_text(cfg.model_id)


peft_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ],
)


SYSTEM_PROMPT = (
    "You are a multilingual stand-up comedian. "
    "You write short, original jokes in English, Spanish, and Chinese. "
    "You ALWAYS obey the user’s constraints exactly (word inclusion, topic, language). "
    "You prefer concise setups and strong punchlines."
)


def build_text(example: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    example["text"] = text
    return example


def prepare_split(split_ds):
    split_ds = split_ds.map(build_text)
    cols_to_remove = [c for c in split_ds.column_names if c != "text"]
    if cols_to_remove:
        split_ds = split_ds.remove_columns(cols_to_remove)
    return split_ds


# load full dataset, then split into train / eval
raw_dataset = load_dataset(
    "json",
    data_files={"train": cfg.dataset_path},
    split="train",
)

split = raw_dataset.train_test_split(
    test_size=cfg.eval_ratio,
    seed=cfg.seed,
)

train_dataset = prepare_split(split["train"])
eval_dataset = prepare_split(split["test"])


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
    save_strategy="steps",
    save_steps=cfg.eval_steps,
    bf16=torch.cuda.is_available(),
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    seed=cfg.seed,
    eval_strategy=cfg.evaluation_strategy,
    eval_steps=cfg.eval_steps,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    max_seq_length=cfg.max_seq_length,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)


def main():
    train_result = trainer.train()

    eval_metrics = trainer.evaluate()
    print("Eval metrics:", eval_metrics)

    if "eval_loss" in eval_metrics:
        try:
            ppl = math.exp(eval_metrics["eval_loss"])
            print(f"Perplexity: {ppl:.2f}")
        except OverflowError:
            print("Perplexity overflow (loss too large).")

    # Save adapter and tokenizer
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Training complete. LoRA adapter saved to", cfg.output_dir)


if __name__ == "__main__":
    main()
