import os
import json
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import *

DATASET_UTILIZATION = 0.033  # amount of entire dataset used

# ================================ Load training data ================================ #

logging.info(f"Loading source and target files from {DATA_DIR}...")


def load_and_prepare_data():
    with open(f"{DATA_DIR}/train_source.txt", "r", encoding="utf-8") as file:
        source = file.readlines()

    with open(f"{DATA_DIR}/train_target.txt", "r", encoding="utf-8") as file:
        target = file.readlines()

    data = [{"source": s.strip(), "target": t.strip()} for s, t in zip(source, target)]
    data = data[: int(len(data) * DATASET_UTILIZATION)]

    # Split into train/validation
    cutoff = int(0.9 * len(data))
    train_data = data[:cutoff]
    val_data = data[cutoff:]

    return {"train": train_data, "validation": val_data}


data = load_and_prepare_data()
logging.info(
    f"Loaded {len(data['train'])} training and {len(data['validation'])} validation examples."
)

# ============================== Create training dataset ============================= #

logging.info("Creating HuggingFace Dataset...")


def create_dataset(data):
    train_dataset = Dataset.from_list(data["train"])
    val_dataset = Dataset.from_list(data["validation"])
    return DatasetDict({"train": train_dataset, "validation": val_dataset})


dataset = create_dataset(data)

# ============================= Load model and tokenizer ============================= #

logging.info(f"Loading model ({MODEL_NAME}) and tokenizer...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    device_map="auto",
    quantization_config=NF4_CONFIG,
    cache_dir=CACHE_DIR,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, cache_dir=CACHE_DIR, add_bos_token=True, add_eos_token=False
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==================== Load parameter-efficient fine-tuning model ==================== #

logging.info("Applying LoRA configuration...")

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.01,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()

# ============================ Prepare training arguments ============================ #

logging.info("Setting up SFTTrainer with training arguments...")

training_args = SFTConfig(
    # ------------------------ Output & Logging ----------------------- #
    output_dir=OUTPUT_DIR,
    run_name=f"{MODEL_NAME} {SOURCE_LANG}-{TARGET_LANG}",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    report_to=[],
    logging_strategy="steps",
    logging_steps=10,
    disable_tqdm=False,
    # ------------------------ Data Processing ------------------------ #
    packing=True,
    max_seq_length=512,
    dataset_text_field="text",
    label_names=["labels"],
    # ----------------------- Core Training Loop ---------------------- #
    num_train_epochs=3,
    max_steps=-1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch sizes = 32
    dataloader_pin_memory=True,
    # -------------------- Evaluation & Checkpoints ------------------- #
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    # ------------------------- Optimization -------------------------- #
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.05,
    weight_decay=0.001,
    max_grad_norm=0.5,
    # --------------------------- Precision --------------------------- #
    bf16=True,
)


def formatting_func(example):
    return f"{SOURCE_LANG}:{example['source']}\n{TARGET_LANG}:{example['target']}"


trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=peft_config,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=formatting_func,
)

# ==================================== Train model =================================== #

logging.info("Beginning training...")
trainer.train()
logging.info("Training completed.")

# ================================ Save training logs ================================ #

logging.info("Saving training logs to disk...")

logs = trainer.state.log_history
logs_path = os.path.join(OUTPUT_DIR, "logs.json")

with open(logs_path, "w") as log:
    log.write(json.dumps(logs, indent=2))

logging.info("Training complete.")
