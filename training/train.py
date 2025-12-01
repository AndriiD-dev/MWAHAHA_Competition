# train_qwen_lora.py
"""
LoRA fine-tuning script for a text-only Qwen Instruct model on a jokes dataset.

The dataset is expected to be a JSONL file with one example per line:

    {"prompt": "...", "response": "..."}
    {"prompt": "...", "response": "..."}

`prompt`  – the input or context (for example, a setup or situation).
`response` – the desired model output (for example, the punchline).
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer



@dataclass
class FinetuneConfig:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"

    dataset_path: str = "data/jokes_lora_with_conditions.jsonl"
    output_dir: str = "qwen2_5_3b_instruct_jokes_lora"

    max_seq_length: int = 1024
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    logging_steps: int = 10
    save_strategy: str = "epoch"
    seed: int = 42


cfg = FinetuneConfig()

def load_qwen_text(model_id: str):
    """
    Load a text-only Qwen Instruct model and its tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

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

"""
Expected dataset format (JSONL):

{"prompt": "...", "response": "..."}
{"prompt": "...", "response": "..."}
...

We will ignore any extra fields.
"""

dataset = load_dataset(
    "json",
    data_files={"train": cfg.dataset_path},
    split="train",
)



def formatting_func(example: Dict[str, Any]) -> str:
    """
    Turn one row {prompt, response} into a single chat-formatted string
    using Qwen's chat template.
    """

    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_train_epochs,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    learning_rate=cfg.learning_rate,
    warmup_ratio=cfg.warmup_ratio,
    weight_decay=cfg.weight_decay,
    logging_steps=cfg.logging_steps,
    save_strategy=cfg.save_strategy,
    bf16=torch.cuda.is_available(),
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    report_to=[],
    seed=cfg.seed,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,
    train_dataset=dataset,
    dataset_text_field=None,
    formatting_func=formatting_func,
    max_seq_length=cfg.max_seq_length,
    args=training_args,
)


if __name__ == "__main__":
    trainer.train()
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("Training complete. LoRA adapter saved to", cfg.output_dir)
