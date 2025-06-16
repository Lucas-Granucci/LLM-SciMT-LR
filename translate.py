import json
import asyncio
from tqdm import tqdm
from googletrans import Translator

from config import *

# =============================== Load evaluation data =============================== #

with open("emea_data/en-et/test_source.txt", "r", encoding="utf-8") as file:
    sentences = [json.loads(line) for line in file]

sentences = sentences[: int(len(sentences) * DATASET_UTILIZATION)]

# ========================= Get Google Translate Predictions ========================= #


async def translate_and_write_batches(sentences, batch_size, output_file):
    async with Translator() as translator:

        loop = asyncio.get_event_loop()

        for i in tqdm(range(0, len(sentences), batch_size)):
            sent_batch = sentences[i : i + batch_size]

            translations = await translator.translate(sent_batch, dest="et")
            translated_texts = [translation.text for translation in translations]

            await loop.run_in_executor(
                None, write_to_file, output_file, translated_texts
            )


def write_to_file(path, lines):
    with open(path, "a", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    asyncio.run(
        translate_and_write_batches(
            sentences, batch_size=10, output_file="emea_data/en-et/google.txt"
        )
    )
