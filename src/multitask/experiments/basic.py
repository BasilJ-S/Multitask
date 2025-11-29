import numpy as np
import tensorly as tl
import torch
from datasets import Dataset as hfDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.tests import test_data
from sklearn.utils import validation
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import trange


class SingleTaskMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        layers = [torch.nn.Linear(input_size, hidden_sizes[0]), torch.nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x


class MultiTaskNaiveMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_sizes):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [
                SingleTaskMLP(input_size, hidden_sizes, output_size)
                for output_size in output_sizes
            ]
        )

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return outputs


class MultiTaskMLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        shared_hidden_sizes,
        task_hidden_sizes,
        output_sizes,
    ):
        super().__init__()
        self.input_size = input_size
        self.shared_hidden_sizes = shared_hidden_sizes
        self.task_hidden_sizes = task_hidden_sizes
        self.output_sizes = output_sizes

        shared_layers = [
            torch.nn.Linear(input_size, shared_hidden_sizes[0]),
            torch.nn.ReLU(),
        ]
        for i in range(1, len(shared_hidden_sizes)):
            shared_layers.append(
                torch.nn.Linear(shared_hidden_sizes[i - 1], shared_hidden_sizes[i])
            )
            shared_layers.append(torch.nn.ReLU())
        self.shared_layers = torch.nn.Sequential(*(shared_layers))

        self.task_layers = torch.nn.ModuleList()

        for output_size in output_sizes:
            task_layer_list = []
            if len(task_hidden_sizes) == 0:
                # No task-specific hidden layers
                task_layer_list.append(
                    torch.nn.Linear(shared_hidden_sizes[-1], output_size)
                )
            else:
                # Task-specific hidden layers
                task_layer_list.append(
                    torch.nn.Linear(shared_hidden_sizes[-1], task_hidden_sizes[0])
                )
                task_layer_list.append(torch.nn.ReLU())

                for i in range(1, len(task_hidden_sizes)):
                    task_layer_list.append(
                        torch.nn.Linear(task_hidden_sizes[i - 1], task_hidden_sizes[i])
                    )
                    task_layer_list.append(torch.nn.ReLU())
                task_layer_list.append(
                    torch.nn.Linear(task_hidden_sizes[-1], output_size)
                )
            self.task_layers.append(torch.nn.Sequential(*task_layer_list))

    def forward(self, x):
        # (batch_size, 1, input_size)
        x = self.shared_layers(x)  # (batch_size, 1, shared_hidden_sizes[-1])

        outputs = [task_layers(x) for task_layers in self.task_layers]
        # list of (batch_size, 1, output_size)

        return outputs


class SharedTTLinearLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_tasks: int,
        tt_rank: int,
    ):
        """
        Shared TT layer for multi-task learning.
        shape: tuple, shape of the weight matrix to be represented in TT format
        tt_rank: int, TT rank, assumed to be the same for all cores for simplicity
        num_tasks: int, number of tasks
        """
        super().__init__()
        tl.set_backend("pytorch")

        self.num_tasks = num_tasks
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank

        # Scalinig factor for initialization to keep variance small

        # --- Shared TT first core (core 0) ---
        self.first_shared_core = torch.nn.Parameter(
            torch.randn(num_tasks, tt_rank) * 0.1
        )

        # --- Shared TT cores (core 2)
        self.middle_shared_core = torch.nn.Parameter(
            torch.randn(tt_rank, in_features, tt_rank) * 0.1
        )

        # --- Task-specific last core (core 3) ---
        self.task_core = torch.nn.Parameter(torch.randn(tt_rank, out_features) * 0.1)

    def _get_contracted_cores(self):
        # cores = [U1, U2, ..., UN]
        W = self.first_shared_core  # (num_tasks, tt_rank)
        W = torch.einsum(
            "ij,jkl->ikl", W, self.middle_shared_core
        )  # (num_tasks, in_features, tt_rank)
        return torch.einsum(
            "ijk,kl->ijl", W, self.task_core
        )  # (num_tasks, in_features, out_features)

    def forward(
        self, x: list[torch.Tensor]
    ) -> list[
        torch.Tensor
    ]:  # list of (batch_size, input_size) -> list of (batch_size, output_size)
        x_concat = torch.stack(x, dim=1)  # (batch_size, num_tasks, input_size)
        task_weights = (
            self._get_contracted_cores()
        )  # (num_tasks, in_features, out_features)

        y = torch.einsum(
            "bni,nio->bno", x_concat, task_weights
        )  # (batch_size, num_tasks, output_size)
        y = [
            y[:, i, :] for i in range(self.num_tasks)
        ]  # list of (batch_size, output_size)
        return y


class MultiTaskMLP_Residual(torch.nn.Module):
    """
    Multi-task MLP with residual shared layers.
    """

    def __init__(
        self,
        input_size,
        task_hidden_sizes: list[int],
        shared_hidden_sizes: list[int],
        output_sizes,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = task_hidden_sizes
        self.output_sizes = output_sizes
        self.num_tasks = len(output_sizes)

        individual_output_size = sum(output_sizes)
        self.task_networks = torch.nn.ModuleList(
            [
                SingleTaskMLP(input_size, task_hidden_sizes, output_size)
                for output_size in output_sizes
            ]
        )
        self.shared_layers = SingleTaskMLP(
            individual_output_size, shared_hidden_sizes, individual_output_size
        )

    def forward(self, x) -> list[torch.Tensor]:
        x = [
            task_network(x) for task_network in self.task_networks
        ]  # num tasks x (batch_size, 1, output_size)

        x = torch.cat(x, dim=-1)  # (batch_size, 1, sum(output_sizes))
        shared = self.shared_layers(x)  # (batch_size, 1, sum(output_sizes))
        x = x + shared  # Residual connection
        out = [
            x[:, sum(self.output_sizes[:i]) : sum(self.output_sizes[: i + 1])]
            for i in range(self.num_tasks)
        ]
        return out


class MultiTaskMLP_TTSoftshare(torch.nn.Module):
    """
    Multi-task MLP with TT-soft sharing layers.
    Only one task specific layer to project to output size of each task.
    All other layers are shared using TT-soft sharing.
    """

    def __init__(
        self,
        input_size,
        hidden_sizes: list[int],
        output_sizes,
        tt_rank=4,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_sizes
        self.output_sizes = output_sizes
        self.num_tasks = len(output_sizes)

        self.in_layer = SharedTTLinearLayer(
            in_features=input_size,
            out_features=hidden_sizes[0],
            tt_rank=tt_rank,
            num_tasks=self.num_tasks,
        )

        shared_hidden_layers = []

        for layer in range(1, len(hidden_sizes)):
            shared_hidden_layers.append(
                SharedTTLinearLayer(
                    in_features=hidden_sizes[layer - 1],
                    out_features=hidden_sizes[layer],
                    tt_rank=tt_rank,
                    num_tasks=self.num_tasks,
                )
            )
        self.shared_hidden_layers = torch.nn.ModuleList(shared_hidden_layers)

        self.task_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_sizes[-1], output_size)
                for output_size in output_sizes
            ]
        )
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = [x for _ in range(self.num_tasks)]
        x = self.in_layer(x)
        x = [self.activation(x_i) for x_i in x]

        for layer in self.shared_hidden_layers:
            x = layer(x)
            x = [self.activation(x_i) for x_i in x]

        out = [layer(x_i) for x_i, layer in zip(x, self.task_layers)]
        return out


class PreScaledHFDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Hugging Face datasets that applies pre-scaling to features and targets.
    """

    def __init__(
        self,
        hf_dataset: hfDataset,
        feature_cols: list[str],
        target_cols: list[list[str]],
        scaler_X: StandardScaler | None = None,
        scaler_y: list[StandardScaler] | None = None,
    ):
        import numpy as np

        self.feature_cols = feature_cols
        self.target_cols = target_cols

        # Convert to NumPy arrays
        X = np.stack([hf_dataset[col] for col in feature_cols], axis=1)

        # Per task
        y_overall = []
        for i, task_targets in enumerate(target_cols):
            y = np.stack([hf_dataset[col] for col in task_targets], axis=1)
            if scaler_y is not None:
                y = scaler_y[i].transform(y)
            y_overall.append(y)

        # Apply scaling if provided
        if scaler_X is not None:
            X = scaler_X.transform(X)

        # Store as tensors directly
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = [torch.tensor(y_task, dtype=torch.float32) for y_task in y_overall]

    def __len__(self):
        return len(self.y[0])

    def __getitem__(self, idx):
        y = {f"task_{i}": y_task[idx] for i, y_task in enumerate(self.y)}
        batch = {
            "X": self.X[idx],
            "y": y,
        }
        return batch


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

    total_loss = 0.0
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
        l = torch.stack(
            [
                loss_fn(pred, y_i.float().to(device)) * task_weight
                for pred, y_i, task_weight in zip(predictions, y, task_weights)
            ]
        ).mean()

        if is_training:
            l.backward()
            optimizer.step()

        total_loss += l.item() * y[0].size(0)
        total_samples += y[0].size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


ds = load_dataset("gvlassis/california_housing").with_format("torch")


hf_train: hfDataset = ds["train"]  # type: ignore
hf_validation: hfDataset = ds["validation"]  # type: ignore


targets = [["MedHouseVal", "AveRooms"], ["Longitude"]]
task_weights = [1.0, 0.5]  # Weight for each task in the loss computation
all_targets = [t for sublist in targets for t in sublist]
features = [col for col in hf_train.column_names if col not in all_targets]
print("Features:", features, "Targets:", targets)

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
    t = trange(2, unit="epoch")
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
        message = f"Model {model.__class__.__name__} | Epoch {_epoch} | Train: {train_loss:.3f} | Val: {val_loss:.3f}"
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
        f"Final unscaled validation MSE for model {model.__class__.__name__}: {val_loss_unscaled:.3f}"
    )


# Plot results

import matplotlib.pyplot as plt

# Colour per model
colors = ["b", "g", "r", "c", "m", "y", "k"]

for i, (model_name, result) in enumerate(results.items()):
    epochs = [r["epoch"] for r in result]
    train_losses = [r["train_loss"] for r in result]
    val_losses = [r["val_loss"] for r in result]

    plt.plot(epochs, train_losses, label=f"{model_name} Train Loss", color=colors[i])
    plt.plot(
        epochs,
        val_losses,
        label=f"{model_name} Val Loss",
        linestyle="--",
        color=colors[i],
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("multitask_mlp_comparison.png")
plt.show()
