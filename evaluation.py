import os
import sacrebleu
from config import *

# =============================== Load evaluation data =============================== #

predictions_path = os.path.join(DATA_DIR, "predictions.txt")
references_path = os.path.join(DATA_DIR, "test_target.txt")

with open(predictions_path, encoding="utf-8") as file:
    predictions = file.readlines()

with open(references_path, encoding="utf-8") as file:
    references = file.readlines()

cutoff = min(len(references), len(predictions))
references = references[:cutoff]
predictions = predictions[:cutoff]

# ================================== Run Evaluation ================================== #
logging.info("Calculating evaluation metrics...")

bleu = sacrebleu.corpus_bleu(predictions, [references]).score
chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2).score
ter = sacrebleu.metrics.TER().corpus_score(predictions, [references]).score

logging.info(f"BLEU: {bleu:.3f}, chrF++: {chrf:.3f}, TER: {ter:.3f}")
logging.info("Evaluation complete.")
