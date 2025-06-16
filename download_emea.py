import os
from datasets import load_dataset

from config import *

# =============================== Download EMEA Dataset ============================== #
# https://huggingface.co/datasets/qanastek/EMEA-V3

SAVE_DIR = "emea_data/"
TEST_RATIO = 0.1
lang_pairs = ["en-mt", "en-et", "en-lt", "en-sk"]


def load_dataset_split(lang_pair):
    dataset = load_dataset(
        path="qanastek/EMEA-V3",
        name=lang_pair,
        split="train",
        download_mode="force_redownload",
        trust_remote_code=True,
    )
    return dataset


def save_sentences(save_path, sentences):
    with open(save_path, "w", encoding="utf-8") as file:
        file.write("\n".join(sentences))


for lang_pair in lang_pairs:
    # Load EMEA data from Hugging Face
    dataset = load_dataset_split(lang_pair)
    sentences = dataset["translation"]

    source_lang = lang_pair.split("-")[0]
    target_lang = lang_pair.split("-")[1]
    source_sents, target_sents = [], []

    for sent in sentences:
        source_sents.append(sent[source_lang])
        target_sents.append(sent[target_lang])

    save_path = f"{SAVE_DIR}/{lang_pair}"
    os.makedirs(save_path)

    # Split training and testing data
    cutoff = int(TEST_RATIO * len(source_sents))

    source_test = source_sents[:cutoff]
    target_test = target_sents[:cutoff]

    train_source_sents = source_sents[cutoff:]
    train_target_sents = target_sents[cutoff:]

    # Save sentences
    save_sentences(f"{save_path}/train_source.txt", train_source_sents)
    save_sentences(f"{save_path}/train_target.txt", train_target_sents)
    save_sentences(f"{save_path}/test_source.txt", source_test)
    save_sentences(f"{save_path}/test_target.txt", target_test)
