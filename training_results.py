import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from config import *

# ================================ Load training logs ================================ #

logs_path = os.path.join(OUTPUT_DIR, "logs.json")
with open(logs_path, "r") as file:
    logs = json.load(file)

records = []

for log in logs:
    curr_step = log["step"]
    if "eval_loss" in log:  # Evaluation logs
        records.append(
            {
                "step": curr_step,
                "eval_loss": log["eval_loss"],
                "eval_mean_token_accuracy": log["eval_mean_token_accuracy"],
                "type": "eval",
            }
        )

    elif "loss" in log:  # Training logs
        records.append(
            {
                "step": curr_step,
                "train_loss": log["loss"],
                "train_mean_token_accuracy": log["mean_token_accuracy"],
                "type": "train",
            }
        )
log_df = pd.DataFrame(records)
log_df = log_df.sort_values("step").reset_index(drop=True)

eval_logs = log_df[log_df["type"] == "eval"]
train_logs = log_df[log_df["type"] == "train"]

# =============================== Plot training results ============================== #

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot Loss
axs[0].plot(
    train_logs["step"], train_logs["train_loss"], label="Train Loss", marker="o"
)
axs[0].plot(eval_logs["step"], eval_logs["eval_loss"], label="Eval Loss", marker="x")
axs[0].set_title("Loss Comparison")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

# Plot Mean Token Accuracy
axs[1].plot(
    train_logs["step"],
    train_logs["train_mean_token_accuracy"],
    label="Train Accuracy",
    marker="o",
)
axs[1].plot(
    eval_logs["step"],
    eval_logs["eval_mean_token_accuracy"],
    label="Eval Accuracy",
    marker="x",
)
axs[1].set_title("Mean Token Accuracy Comparison")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
