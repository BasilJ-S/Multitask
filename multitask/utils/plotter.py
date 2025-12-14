from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from multitask.models.baselines import BASELINE_NAMES

PLOT_COLOURS = ["b", "g", "r", "c", "m", "y", "k"]
PLOT_LINESTYLES = [
    "-",
    "-.",
    "--",
    ":",
    "-",
    "-.",
    "--",
    ":",
]
BASELINE_COLOURS = ["gray", "black", "purple", "orange", "brown"]


def plot_task_loss_same_plot(
    results: dict[str, list[dict[str, Any]]],
    targets: list[list[str]],
    axs: Any,
    index: int,
    linestyles: list[str] = PLOT_LINESTYLES,
    colors: list[str] = PLOT_COLOURS,
) -> None:
    for i, (model_name, result) in enumerate(results.items()):
        epochs = [r["epoch"] for r in result]
        train_losses = [r["train_loss"] for r in result]
        val_losses = [r["val_loss"] for r in result]
        for j, task in enumerate(targets):
            task_train_losses = [tl[j] for tl in train_losses]
            task_val_losses = [vl[j] for vl in val_losses]

            axs[index].plot(
                epochs,
                task_train_losses,
                label=f"{model_name} Train Loss Task {j}",
                color=colors[i],
                linestyle=linestyles[2 * j],
            )
            axs[index].plot(
                epochs,
                task_val_losses,
                label=f"{model_name} Val Loss Task {j}",
                linestyle=linestyles[2 * j + 1],
                color=colors[i],
            )
    axs[index].set_title(f"Loss per Task")
    axs[index].set_xlabel("Epoch")
    axs[index].set_ylabel("Loss")
    axs[index].legend()


def plot_task_loss_separately(
    results: dict[str, list[dict[str, Any]]],
    targets: list[list[str]],
    colors: list[str] = PLOT_COLOURS,
) -> None:
    num_rows = (len(targets) + 1) // 2
    fig, axs = plt.subplots(
        num_rows,
        2,
        figsize=(12, num_rows * 4),
    )
    axs = axs.flatten() if num_rows > 1 else axs

    for i, baseline_name in enumerate(BASELINE_NAMES):
        baseline_color = BASELINE_COLOURS[i % len(BASELINE_COLOURS)]
        baseline_results = results.pop(baseline_name, None)

        if baseline_results is not None:
            # add horizontal lines for baseline
            for j, task in enumerate(targets):
                task_train_loss = baseline_results["train_loss"][j]
                task_val_loss = baseline_results["val_loss"][j]
                if j == 0:
                    axs[j].axhline(
                        y=task_train_loss,
                        label=f"{baseline_name} Baseline",
                        color=baseline_color,
                        linestyle="-",
                    )
                else:
                    axs[j].axhline(
                        y=task_train_loss,
                        color=baseline_color,
                        linestyle="-",
                    )
                axs[j].axhline(
                    y=task_val_loss,
                    color=baseline_color,
                    linestyle="--",
                )

    for i, (model_name, result) in enumerate(results.items()):

        epochs = [r["epoch"] for r in result]
        train_losses = [r["train_loss"] for r in result]
        val_losses = [r["val_loss"] for r in result]
        for j, task in enumerate(targets):

            task_train_losses = [tl[j] for tl in train_losses]
            task_val_losses = [vl[j] for vl in val_losses]
            if j == 0:
                axs[j].plot(
                    epochs,
                    task_train_losses,
                    label=f"{model_name}",
                    color=colors[i],
                    linestyle="-",
                )
            else:
                axs[j].plot(
                    epochs,
                    task_train_losses,
                    color=colors[i],
                    linestyle="-",
                )
            axs[j].plot(
                epochs,
                task_val_losses,
                linestyle="--",
                color=colors[i],
            )
            axs[j].set_title(f"Loss for {task}")
            axs[j].set_xlabel("Epoch")
            axs[j].set_ylabel("Loss")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    fig.legend(loc="lower center")

    fig.suptitle("Loss per Task")
    fig.show()


def plot_loss(
    results: dict[str, list[dict[str, Any]]],
    axs: Any,
    index: int,
    colors: list[str] = PLOT_COLOURS,
) -> None:
    for i, (model_name, result) in enumerate(results.items()):
        epochs = [r["epoch"] for r in result]
        train_losses = [sum(r["train_loss"]) for r in result]
        val_losses = [sum(r["val_loss"]) for r in result]

        axs[index].plot(
            epochs,
            train_losses,
            label=f"{model_name} Train Loss",
            color=colors[i],
        )
        axs[index].plot(
            epochs,
            val_losses,
            label=f"{model_name} Val Loss",
            linestyle="--",
            color=colors[i],
        )

    axs[index].set_title(f"Total Loss")
    axs[index].set_xlabel("Epoch")
    axs[index].set_ylabel("Loss")
    axs[index].legend()


def plot_test_loss(test_results: dict, targets):
    num_rows = (len(targets) + 1) // 2
    fig, axs = plt.subplots(
        num_rows,
        2,
        figsize=(12, num_rows * 4),
    )
    axs = axs.flatten() if num_rows > 1 else axs

    model_names = list(test_results.keys())
    x = range(len(model_names))
    bar_width = 0.8 / len(model_names)

    for j, task in enumerate(targets):
        for i, name in enumerate(model_names):
            baseline_results = test_results.get(name, None)

            if baseline_results is not None:
                task_test_loss = baseline_results[j]
                baseline_color = BASELINE_COLOURS[i % len(BASELINE_COLOURS)]

                axs[j].bar(
                    i * bar_width,
                    task_test_loss,
                    bar_width,
                    label=name if j == 0 else "",
                    color=baseline_color,
                )

        axs[j].set_title(f"Loss for {task}")
        axs[j].set_xlabel("Model")
        axs[j].set_ylabel("Loss")
        axs[j].set_xticks([i * bar_width + bar_width / 2 for i in x])

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    fig.legend(loc="lower center")
    fig.suptitle("Test Loss per Task")
    fig.show()


def plot_prediction_errors_violin(
    test_results: dict,
    task_names: list[str] = None,
    file_suffix: str = "results",
    show: bool = False,
) -> None:
    """
    Create violin plots of prediction errors per task.

    Args:
        test_results: dict mapping model_name -> {task_0: {mse, mae, predictions, targets}, ...}
        task_names: optional list of task names for display
        file_suffix: string for saving figure
        show: whether to display the plot
    """
    from pathlib import Path

    model_names = list(test_results.keys())

    # Determine number of tasks from first model
    first_model_results = test_results[model_names[0]]
    num_tasks = len([k for k in first_model_results.keys() if k.startswith("task_")])

    if task_names is None:
        task_names = [f"Task {i}" for i in range(num_tasks)]

    # Create figure with subplots (one per task)
    fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 5))
    if num_tasks == 1:
        axes = [axes]

    # Collect errors per model per task
    for task_idx in range(num_tasks):
        ax = axes[task_idx]
        errors_data = []
        model_labels = []

        for model_name in model_names:
            if f"task_{task_idx}" in test_results[model_name]:
                task_data = test_results[model_name][f"task_{task_idx}"]
                preds = np.array(task_data["predictions"])
                targets = np.array(task_data["targets"])
                errors = np.abs(preds - targets)

                errors_data.append(errors)
                model_labels.append(
                    model_name.replace("NaiveMultiTaskTimeseriesWrapper(", "").replace(
                        ")", ""
                    )
                )

        # Create violin plot
        parts = ax.violinplot(
            errors_data,
            positions=range(len(errors_data)),
            showmeans=True,
            showmedians=True,
        )
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=45, ha="right")
        ax.set_ylabel("Absolute Error")
        ax.set_title(f"Prediction Errors – {task_names[task_idx]}")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(Path("plots") / f"prediction_errors_violin_{file_suffix}.png", dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_prediction_metrics_comparison(
    test_results: dict,
    task_names: list[str] = None,
    file_suffix: str = "results",
    show: bool = False,
    metrics: list[str] = None,
) -> None:
    """
    Create bar plots comparing metrics (MAE, MSE, RMSE) across models per task.

    Args:
        test_results: dict mapping model_name -> {task_0: {mse, mae, rmse, ...}, ...}
        task_names: optional list of task names for display
        file_suffix: string for saving figure
        show: whether to display the plot
        metrics: list of metrics to plot (default: ["mae", "mse", "rmse"])
    """
    from pathlib import Path

    if metrics is None:
        metrics = ["mae", "mse", "rmse"]

    model_names = list(test_results.keys())
    first_model_results = test_results[model_names[0]]
    num_tasks = len([k for k in first_model_results.keys() if k.startswith("task_")])

    if task_names is None:
        task_names = [f"Task {i}" for i in range(num_tasks)]

    # Create figure with subplots (one per metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    model_colors = {m: cmap(i) for i, m in enumerate(model_names)}

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        x = np.arange(num_tasks)
        bar_width = 0.8 / len(model_names)

        for model_idx, model_name in enumerate(model_names):
            metric_values = []

            for task_idx in range(num_tasks):
                if f"task_{task_idx}" in test_results[model_name]:
                    task_data = test_results[model_name][f"task_{task_idx}"]
                    if metric in task_data:
                        metric_values.append(task_data[metric])
                    else:
                        metric_values.append(0)
                else:
                    metric_values.append(0)

            offset = (model_idx - len(model_names) / 2) * bar_width + bar_width / 2
            model_label = model_name.replace(
                "NaiveMultiTaskTimeseriesWrapper(", ""
            ).replace(")", "")
            ax.bar(
                x + offset,
                metric_values,
                bar_width,
                label=model_label,
                color=model_colors[model_name],
                alpha=0.8,
            )

        ax.set_xlabel("Task")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} Comparison Across Models")
        ax.set_xticks(x)
        ax.set_xticklabels(task_names)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(Path("plots") / f"metrics_comparison_{file_suffix}.png", dpi=150)
    if show:
        plt.show()
    plt.close()
