"""
NOTE: THIS FILE WAS AI GENERATED. DOES NOT REPRESENT ORIGINAL WORK OF THE AUTHOR.
Plot script for multitask weather results.

Loads:
    targets_file = "targets_prepare_weather_multiloc_full.json"
    test_results_file = "test_results_prepare_weather_multiloc_full.json"
    train_results_file = "training_results_prepare_weather_multiloc_full.json"

What this script produces:
1. Multi-bar chart: test loss per model per task (with one subplot per task).
2. Per-task loss curves: train/val loss curves for every model across trials.
3. Statistical summaries: mean/std per model per task, printed to stdout.

To swap datasets, change the filenames at the top.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# CONFIG (swap filenames here)
# -------------------------
targets_file = "targets_prepare_weather_multiloc_full.json"
test_results_file = "test_results_prepare_weather_multiloc_full.json"
train_results_file = "training_results_prepare_weather_multiloc_full.json"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


targets = load_json(targets_file)
test_results = load_json(test_results_file)
train_results = load_json(train_results_file)

# targets: list[tasks][task variables]
num_tasks = len(targets)
task_names = [
    f"Task {i+1}" for i in range(num_tasks)
]  # optionally replace with city names

# ---------------------------------------------------------
# PROCESS TEST RESULTS
# test_results: list[trial][model_name] -> list per task
# ---------------------------------------------------------
model_names = list(test_results[0].keys())

# Convert to ndarray: shape (num_models, num_tasks, num_trials)
test_losses = {
    model: np.array([trial[model] for trial in test_results]).T
    # shape now: (num_tasks, num_trials)
    for model in model_names
}

# ---------------------------------------------------------
# PLOT: MULTI-BAR TEST LOSSES PER TASK
# ---------------------------------------------------------
fig, axes = plt.subplots(num_tasks, 1, figsize=(12, 3 * num_tasks), sharex=True)
if num_tasks == 1:
    axes = [axes]

x = np.arange(len(model_names))
bar_width = 0.8 / len(test_results)  # each trial gets its own thin bar group

for t in range(num_tasks):
    ax = axes[t]
    for trial_idx in range(len(test_results)):
        # Extract one number per model for this task/trial
        y = [test_results[trial_idx][model][t] for model in model_names]
        offset = (trial_idx - len(test_results) / 2) * bar_width
        ax.bar(x + offset, y, width=bar_width, label=f"Trial {trial_idx+1}")

    ax.set_title(f"Test Loss – {task_names[t]}")
    ax.set_ylabel("Loss")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

axes[-1].legend(loc="upper right")
fig.tight_layout()
plt.show()


# ----
# PLOT ERROR BAR TEST LOSSES
# ---
def plot_test_results_single_trial(test_results, task_names=None):
    """
    Plots test results per model per task.
    Assumes all trials are identical, so only the first trial is used.
    """

    model_names = list(test_results[0].keys())
    num_models = len(model_names)
    num_tasks = len(test_results[0][model_names[0]])

    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]

    # Use the first trial only
    trial = test_results[0]

    fig, axes = plt.subplots(num_tasks, 1, figsize=(12, 4 * num_tasks), sharex=True)
    if num_tasks == 1:
        axes = [axes]

    x = np.arange(num_models)

    for t_idx, task in enumerate(task_names):
        ax = axes[t_idx]

        values = [trial[model][t_idx] for model in model_names]
        ax.bar(x, values, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylabel("Test Loss")
        ax.set_title(f"Test Loss – {task}")

    plt.tight_layout()
    plt.show()


plot_test_results_single_trial(test_results, task_names)

# ---------------------------------------------------------
# STATISTICAL SUMMARY: mean/std per model per task
# ---------------------------------------------------------
print("\n=== STATISTICAL SUMMARY (Test Loss) ===\n")
for model in model_names:
    print(f"\nModel: {model}")
    arr = test_losses[model]  # shape (num_tasks, num_trials)
    for t in range(num_tasks):
        mean = arr[t].mean()
        std = arr[t].std()
        print(f"  {task_names[t]:15s}  mean={mean:.4f}  std={std:.4f}")

# ---------------------------------------------------------
# TRAINING LOSS CURVES (per-task)
# train_results: list of trials; each has:
#     model -> either dict (for baselines) or list of epochs
# ---------------------------------------------------------


def extract_model_epochs(model_entry):
    """
    model_entry is either:
    - dict with train_loss list (baselines)
    - list of dicts, each element contains train_loss for a specific epoch (deep models)
    Returns:
        epochs, train_losses (task-wise avg), val_losses (task-wise avg)
    """
    if isinstance(model_entry, dict):
        # Baselines: epoch='all'
        epochs = [0]
        train = [np.mean(model_entry["train_loss"])]
        val = [np.mean(model_entry["val_loss"])]
        return epochs, train, val

    # Deep models: list of epochs
    epochs = [e["epoch"] for e in model_entry]
    train = [np.mean(e["train_loss"]) for e in model_entry]
    val = [np.mean(e["val_loss"]) for e in model_entry]
    return epochs, train, val


# ---------------------------------------------------------
# PLOT: PER-TASK TRAIN/VAL CURVES ACROSS TRIALS
# ---------------------------------------------------------
for t in range(num_tasks):
    plt.figure(figsize=(14, 8))
    for model in model_names:
        # Aggregate across trials
        all_epochs = []
        all_train = []
        all_val = []

        for trial in train_results:
            entry = trial[model]
            epochs, train_loss, val_loss = extract_model_epochs(entry)

            all_epochs.append(epochs)
            all_train.append(train_loss)
            all_val.append(val_loss)

        # Convert to mean curves (pad to equal length)
        max_len = max(len(e) for e in all_epochs)

        def pad(seq_list):
            return np.array(
                [seq + [seq[-1]] * (max_len - len(seq)) for seq in seq_list]
            )

        mean_train = pad(all_train).mean(axis=0)
        std_train = pad(all_train).std(axis=0)
        mean_val = pad(all_val).mean(axis=0)
        std_val = pad(all_val).std(axis=0)

        plt.plot(mean_train, label=f"{model} Train")
        plt.fill_between(
            range(max_len), mean_train - std_train, mean_train + std_train, alpha=0.2
        )

        plt.plot(mean_val, label=f"{model} Val", linestyle="--")
        plt.fill_between(
            range(max_len), mean_val - std_val, mean_val + std_val, alpha=0.2
        )

    plt.title(f"Per-Task Loss Curves (Averaged over Trials) – {task_names[t]}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
