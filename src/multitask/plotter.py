import matplotlib.pyplot as plt

PLOT_COLOURS = ["b", "g", "r", "c", "m", "y", "k"]
PLOT_LINESTYLES = [
    "-",
    "-.",
    "--",
    ":",
]


def plot_loss_per_task(
    results, targets, axs, index, linestyles=PLOT_LINESTYLES, colors=PLOT_COLOURS
):
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
                linestyle=linestyles[2 * j - 1],
                color=colors[i],
            )
    axs[index].set_title(f"Loss per Task")
    axs[index].set_xlabel("Epoch")
    axs[index].set_ylabel("Loss")
    axs[index].legend()


def plot_loss(results, axs, index, colors=PLOT_COLOURS):
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
