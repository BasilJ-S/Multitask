from typing import Any

import matplotlib.pyplot as plt

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
BASELINE_COLOURS = ["gray", "black"]


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
    num_rows = len(targets) // 2
    fig, axs = plt.subplots(
        num_rows,
        2,
        figsize=(12, num_rows * 4),
    )
    axs = axs.flatten() if num_rows > 1 else axs

    for baseline_name, baseline_results, baseline_color in [
        ("Global Mean", results.pop("global_mean", None), "gray"),
        ("Linear", results.pop("linear", None), "black"),
        ("XGBoost", results.pop("xgboost", None), "purple"),
    ]:
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
