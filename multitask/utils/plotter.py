from typing import Any

import matplotlib.pyplot as plt

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
