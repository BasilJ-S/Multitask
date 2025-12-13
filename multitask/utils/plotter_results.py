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

from multitask.models.baselines import BASELINE_NAMES

# -------------------------
# CONFIG (swap filenames here)
# -------------------------
file_suffix = "prepare_weather_multiloc_full"
plot_name = f"Multi Location Weather Prediction"
task_names = [
    "Kingston",
    "Ottawa",
    "Montreal",
]


targets_file = f"results/targets_{file_suffix}.json"
test_results_file = f"results/test_results_{file_suffix}.json"
train_results_file = f"results/training_results_{file_suffix}.json"


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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_test_results_with_errorbars(
    test_results, task_names=None, file_suffix="results"
):
    """
    test_results: list of dicts
        Each dict maps model -> list per task
    task_names: optional list of task names
    file_suffix: str for saving figure
    """

    model_names = list(test_results[0].keys())
    num_models = len(model_names)
    num_tasks = len(test_results[0][model_names[0]])

    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]

    # Collect data: shape (num_models, num_tasks, num_trials)
    data = {
        model: np.array([trial[model] for trial in test_results]).T
        for model in model_names
    }  # (num_tasks, num_trials)

    # Define colors per model
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(num_models)]

    fig, axes = plt.subplots(num_tasks, 1, figsize=(12, 4 * num_tasks), sharex=True)
    if num_tasks == 1:
        axes = [axes]

    x = np.arange(num_models)

    for t_idx, task in enumerate(task_names):
        ax = axes[t_idx]

        for i, model in enumerate(model_names):
            mean_val = data[model][t_idx].mean()
            std_val = data[model][t_idx].std()

            ax.bar(
                x[i],
                mean_val,
                yerr=std_val,
                capsize=5,
                color=colors[i],
                alpha=0.7,
                label=model,
            )

        ax.set_xticks(x)
        ax.set_ylabel("Test Loss")
        ax.set_title(f"Test Loss – {task}")
        ax.legend()

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(Path("plots") / f"test_loss_per_task_{file_suffix}.png")
    plt.show()


plot_test_results_with_errorbars(test_results, task_names)

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
# ---------------------------------------------------------
# PLOT: PER-TASK TRAIN/VAL CURVES ACROSS TRIALS
# ---------------------------------------------------------
for t_idx in range(num_tasks):
    plt.figure(figsize=(14, 8))
    task_name = task_names[t_idx]

    for model in model_names:

        if model in BASELINE_NAMES:
            # Baseline: plot horizontal line at mean val_loss across trials
            val_vals = [
                np.mean(trial[model]["val_loss"][t_idx]) for trial in train_results
            ]
            mean_val = np.mean(val_vals)
            std_val = np.std(val_vals)

            train_vals = [
                np.mean(trial[model]["train_loss"][t_idx]) for trial in train_results
            ]
            mean_train = np.mean(train_vals)
            std_train = np.std(train_vals)

            plt.axhline(mean_val, linestyle="--", label=f"{model} Val (mean ± std)")
            plt.fill_between(
                x=[0, 10],  # just span whole x-axis
                y1=mean_val - std_val,
                y2=mean_val + std_val,
                alpha=0.2,
            )

            plt.axhline(mean_train, label=f"{model} Train (mean ± std)")
            plt.fill_between(
                x=[0, 10],  # just span whole x-axis
                y1=mean_train - std_train,
                y2=mean_train + std_train,
                alpha=0.2,
            )
            continue  # skip the per-epoch plotting

        # Deep models: list of dicts per epoch
        all_epochs = []
        all_train = []
        all_val = []

        for trial in train_results:
            entry = trial[model]
            # entry is list of epoch dicts
            epochs = [e["epoch"] for e in entry]
            train_loss = [np.mean(e["train_loss"][t_idx]) for e in entry]
            val_loss = [np.mean(e["val_loss"][t_idx]) for e in entry]
            all_epochs.append(epochs)
            all_train.append(train_loss)
            all_val.append(val_loss)

        # pad to max length
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
        plt.plot(mean_val, linestyle="--", label=f"{model} Val")
        plt.fill_between(
            range(max_len), mean_val - std_val, mean_val + std_val, alpha=0.2
        )

    plt.title(f"Per-Task Loss Curves (Averaged over Trials) – {task_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path("plots") / f"loss_curve_task_{t_idx+1}_{file_suffix}.png")
    plt.show()
