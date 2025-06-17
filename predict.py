import os
import json
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import *

# =============================== Load evaluation data =============================== #


def load_source_sentences():
    with open(f"{DATA_DIR}/test_source.txt", "r", encoding="utf-8") as file:
        source = [json.loads(line) for line in file]

    source_sents = [s.strip() for s in source]
    source_sents = source_sents[: int(len(source_sents) * DATASET_UTILIZATION)]

    return source_sents


source_sentences = load_source_sentences()
logging.info(f"Loaded {len(source_sentences)} evaluation examples.")

# ============================= Load model and tokenizer ============================= #

peft_model_path = os.path.join(OUTPUT_DIR, "checkpoint-105")  # change checkpoint path

logging.info(f"Loading base model: {MODEL_NAME}...")
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="cuda", cache_dir=CACHE_DIR
)

logging.info(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    add_bos_token=True,
    add_eos_token=False,  # always False for inference
)

logging.info(f"Loading PEFT model from {peft_model_path}")
model = PeftModel.from_pretrained(model_base, peft_model_path)
model.eval()

# =============================== Formatting Functions =============================== #


def create_eval_prompt(source_sent):
    return f"{SOURCE_LANG}:{source_sent}\n{TARGET_LANG}:"


def extract_prediction(full_output, prompt):
    return full_output.replace(prompt, "").strip()


# ============================== Batch Evaluation Logic ============================== #
def batch_generate_predictions(source_sents, batch_size=16):
    predictions = [None] * len(source_sents)

    # Attach indices to store final predictions in original order
    indexed_samples = [(i, sent) for i, sent in enumerate(source_sents)]

    # Sort by length for efficient padding
    indexed_sorted_samples = sorted(indexed_samples, key=lambda x: len(x[1]))

    # Process batched data in sorted order
    batches_generator = range(0, len(indexed_sorted_samples), batch_size)
    for i in tqdm(batches_generator, desc="Generating predictions"):

        current_batch = indexed_sorted_samples[i : i + batch_size]

        # Extract original indices and source sentences
        original_indices = [sample[0] for sample in current_batch]
        batch_sents = [sample[1] for sample in current_batch]

        eval_prompts = [create_eval_prompt(sent) for sent in batch_sents]

        inputs = tokenizer(
            eval_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        ).to("cuda")

        # Calculate max new tokens dynamically
        input_lengths = inputs["attention_mask"].sum(dim=1)
        max_new_tokens = int(max(input_lengths)) * 2

        outputs = model.generate(
            **inputs,
            disable_compile=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            top_p=None,
            top_k=None,
        )

        # Decode and extract predictions
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_preds = [extract_prediction(d, p) for d, p in zip(decoded, eval_prompts)]

        # Store predictions and references in the original order
        for index, pred in zip(original_indices, batch_preds):
            predictions[index] = pred

        # Free GPU memory
        del inputs, outputs
        torch.cuda.empty_cache()

    return predictions


# =============================== Generate Predictions =============================== #
logging.info("Generating predictions...")
predictions = batch_generate_predictions(source_sentences, batch_size=32)

# Save predictions
predictions_path = os.path.join(DATA_DIR, "predictions.txt")
with open(predictions_path, "w", encoding="utf-8") as file:
    for pred in predictions:
        file.write(json.dumps(pred) + "\n")
