import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Any

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

def format_with_chat_template(examples, tokenizer, chat_field="messages", eos_token_id=None):
    texts = []
    for msgs in examples[chat_field]:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        if eos_token_id is not None and not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

def main():
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    train_path = os.environ.get("TRAIN_JSONL", "data/train_chat.jsonl")
    output_dir = os.environ.get("OUTPUT_DIR", "checkpoints/rag-qlora")
    num_epochs = float(os.environ.get("NUM_EPOCHS", "2"))
    lr = float(os.environ.get("LEARNING_RATE", "1e-4"))
    per_device_bs = int(os.environ.get("PER_DEVICE_BATCH", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "16"))
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "4096"))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=None,  
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",  
    )

    dataset = load_dataset("json", data_files={"train": train_path})
    dataset = dataset.map(
        lambda batch: format_with_chat_template(batch, tokenizer, chat_field="messages", eos_token_id=tokenizer.eos_token_id),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,  # important for DDP + adapters
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
        packing=True,
    )

    trainer.train()

    # Save only the LoRA adapter
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
