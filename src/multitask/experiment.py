import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset as hfDataset
from datasets import load_dataset
from models import (
    MultiTaskMLP,
    MultiTaskMLP_Residual,
    MultiTaskMLP_TTSoftshare,
    MultiTaskNaiveMLP,
)
from multi_datasets import PreScaledHFDataset
from plotter import plot_loss, plot_loss_per_task
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import trange


def run_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    task_weights: list[float],
    optimizer: torch.optim.Optimizer | None = None,
    device=torch.device("cpu"),
    reverse_task_scalers: list[StandardScaler] | None = None,
):
    """
    Run one epoch of training or evaluation.
    If optimizer is provided, runs in training mode, otherwise in eval mode.
    Returns average loss over the epoch.
    """
    is_training = optimizer is not None
    if reverse_task_scalers is not None:
        assert len(reverse_task_scalers) == len(
            task_weights
        ), "Number of reverse scalers must match number of tasks."
        print(
            "Running epoch with reverse scaling for interpretability on original scale."
        )

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = [0.0 for _ in task_weights]
    total_samples = 0

    for batch in dataloader:
        X = batch["X"]  # (batch_size, num_features)
        X = X.to(device)  # (batch_size, num_features)
        y = [
            batch["y"][f"task_{i}"] for i in range(len(batch["y"]))
        ]  # list of (batch_size, )

        y = [y_i.to(device) for y_i in y]
        if is_training:
            optimizer.zero_grad()
            predictions = model(X.float())
        else:
            with torch.no_grad():
                predictions = model(X.float())

        if reverse_task_scalers is not None:
            # Reverse scaling on predictions
            predictions = [
                torch.tensor(
                    reverse_task_scalers[i].inverse_transform(
                        predictions[i].cpu().numpy()
                    ),
                    dtype=torch.float32,
                ).to(device)
                for i in range(len(predictions))
            ]
            y = [
                torch.tensor(
                    reverse_task_scalers[i].inverse_transform(y[i].cpu().numpy()),
                    dtype=torch.float32,
                ).to(device)
                for i in range(len(y))
            ]

        # compute per-task MSE (using the loss defined above) and sum them
        l_per_task = torch.stack(
            [
                loss_fn(pred, y_i.float().to(device)) * task_weight
                for pred, y_i, task_weight in zip(predictions, y, task_weights)
            ]
        )

        l = l_per_task.sum()

        if is_training:
            l.backward()
            optimizer.step()

        total_loss = [
            tl + l_i.item() * y[0].size(0) for tl, l_i in zip(total_loss, l_per_task)
        ]
        total_samples += y[0].size(0)

    avg_loss = [tl / total_samples for tl in total_loss]
    return avg_loss


ds = load_dataset("gvlassis/california_housing").with_format("torch")


hf_train: hfDataset = ds["train"]  # type: ignore
hf_validation: hfDataset = ds["validation"]  # type: ignore


targets = [["MedHouseVal", "AveRooms"], ["Longitude"]]
task_weights = [1.0, 0.5]  # Weight for each task in the loss computation
all_targets = [t for sublist in targets for t in sublist]
features = [col for col in hf_train.column_names if col not in all_targets]

for i, target_group in enumerate(targets):
    print(f"Targets for Task {i} with weight: {task_weights[i]}: {target_group}")


# Fit scalers on training data
target_scalers = [StandardScaler() for _ in targets]
feature_scaler = StandardScaler()
train_features = np.stack([hf_train[col] for col in features], axis=1)
train_targets = [
    np.stack([hf_train[col] for col in target_group], axis=1)
    for target_group in targets
]
feature_scaler.fit(train_features)
print(
    "Feature means:",
    feature_scaler.mean_,
    "stds:",
    feature_scaler.scale_,
)
for scaler, target_set in zip(target_scalers, train_targets):
    scaler.fit(target_set)
    print(
        f" Target means for {target_set}:",
        scaler.mean_,
        "stds:",
        scaler.scale_,
    )

# Create pre-scaled datasets
train_dataset = PreScaledHFDataset(
    hf_train, features, targets, scaler_X=feature_scaler, scaler_y=target_scalers
)
validation_dataset = PreScaledHFDataset(
    hf_validation, features, targets, scaler_X=feature_scaler, scaler_y=target_scalers
)

loss = torch.nn.MSELoss(reduction="mean")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=128)

input_size = len(features)
output_sizes = [len(t) for t in targets]

# use MultiTaskMLP for two targets (MedHouseVal and AveRooms)
modelList = [
    MultiTaskMLP_Residual(
        input_size=input_size,
        task_hidden_sizes=[16, 16],
        shared_hidden_sizes=[16, 16],
        output_sizes=output_sizes,
    ).to(device),
    MultiTaskMLP_TTSoftshare(
        input_size=input_size,
        hidden_sizes=[16, 16, 16],
        output_sizes=output_sizes,
        tt_rank=4,
    ).to(device),
    MultiTaskNaiveMLP(
        input_size=input_size, hidden_sizes=[16, 16, 16], output_sizes=output_sizes
    ).to(device),
    MultiTaskMLP(
        input_size=input_size,
        shared_hidden_sizes=[18, 18, 18],
        task_hidden_sizes=[],
        output_sizes=output_sizes,
    ).to(device),
]
results = {}
for model in modelList:
    results[model.__class__.__name__] = []
    print(
        f"Training model: {model.__class__.__name__} with {sum(p.numel() for p in model.parameters())} parameters"
    )
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    prev_train_loss = float("inf")
    prev_val_loss = float("inf")
    t = trange(10, unit="epoch")
    for _epoch in t:
        train_loss = run_epoch(
            train_dataloader,
            model,
            loss,
            task_weights=task_weights,
            optimizer=optimizer,
            device=device,
        )

        val_loss = run_epoch(
            validation_dataloader,
            model,
            loss,
            task_weights=task_weights,
            optimizer=None,
            device=device,
        )

        results[model.__class__.__name__].append(
            {"epoch": _epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        prev_val_loss = val_loss
        prev_train_loss = train_loss
        message = f"Model {model.__class__.__name__} | Epoch {_epoch} | Train: {train_loss} | Val: {val_loss}"
        t.set_description(message)
        t.write(message)

    # Validate on unscaled data for interpretability
    val_loss_unscaled = run_epoch(
        validation_dataloader,
        model,
        loss,
        task_weights=task_weights,
        optimizer=None,
        device=device,
        reverse_task_scalers=target_scalers,
    )
    print(
        f"Final unscaled validation MSE for model {model.__class__.__name__}: {val_loss_unscaled}"
    )


# Plot results

# Colour per model

fig, axs = plt.subplots(1, 2, figsize=(28, 12))


plot_loss_per_task(results, targets, axs, 0)
plot_loss(results, axs, 1)

# Add labels describing each task for whole figure
fig.suptitle(
    "Multi-Task MLP Model Comparison on California Housing Dataset", fontsize=16
)
task_text = [f"Task {i}: Predict {targets[i]}" for i in range(len(targets))]
fig.text(
    0.5,
    0.04,
    ", ".join(task_text),
    ha="center",
    fontsize=10,
)

plt.savefig("multitask_mlp_comparison.png")
plt.show()
