import tensorly as tl
import torch
from datasets import load_dataset
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import trange


class SingleTaskMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiTaskNaiveMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [
                SingleTaskMLP(input_size, hidden_size, output_size)
                for output_size in output_sizes
            ]
        )

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return outputs


class MultiTaskMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_sizes = output_sizes

        self.shared_layer1 = torch.nn.Linear(input_size, hidden_size)
        self.shared_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.shared_layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.ReLU()

        self.task_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, output_size) for output_size in output_sizes]
        )

    def forward(self, x):
        x = self.activation(self.shared_layer1(x))
        x = self.activation(self.shared_layer2(x))
        x = self.activation(self.shared_layer3(x))
        outputs = [task_layer(x) for task_layer in self.task_layers]
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
        self, x: torch.Tensor
    ):  # (batch_size, num_tasks, input_size) -> (batch_size, num_tasks, output_size)
        task_weights = (
            self._get_contracted_cores()
        )  # (num_tasks, in_features, out_features)
        y = torch.einsum(
            "bni,nio->bno", x, task_weights
        )  # (batch_size, num_tasks, output_size)
        return y


class MultiTaskMLP_TTSoftshare(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes, tt_rank=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_sizes = output_sizes
        self.num_tasks = len(
            output_sizes
        )  # So this is fucked, its supposed to have have anotehr task dimension, which is why I'm getting just 2d TT cores which is BAD.

        self.shared_layer1 = SharedTTLinearLayer(
            in_features=input_size,
            out_features=hidden_size,
            tt_rank=tt_rank,
            num_tasks=self.num_tasks,
        )
        self.shared_layer2 = SharedTTLinearLayer(
            in_features=hidden_size,
            out_features=hidden_size,
            tt_rank=tt_rank,
            num_tasks=self.num_tasks,
        )
        self.shared_layer_3 = SharedTTLinearLayer(
            in_features=hidden_size,
            out_features=hidden_size,
            tt_rank=tt_rank,
            num_tasks=self.num_tasks,
        )
        self.activation = torch.nn.ReLU()

        self.task_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, output_size) for output_size in output_sizes]
        )

    def forward(self, x):
        h1 = self.activation(
            self.shared_layer1(
                x,
            )
        )
        h2 = self.activation(
            self.shared_layer2(
                h1,
            )
        )
        h3 = self.activation(
            self.shared_layer_3(
                h2,
            )
        )  # batch x num_tasks x hidden_size
        out = [layer(h3[:, i, :]) for i, layer in enumerate(self.task_layers)]
        return out


ds = load_dataset("gvlassis/california_housing").with_format("torch")

loss = torch.nn.MSELoss(reduction="mean")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(ds["train"], batch_size=128, shuffle=True)
validation_dataloader = DataLoader(ds["validation"], batch_size=128)

# use MultiTaskMLP for two targets (MedHouseVal and AveRooms)
modelList = [
    MultiTaskMLP_TTSoftshare(
        input_size=7, hidden_size=16, output_sizes=[1, 1], tt_rank=6
    ).to(device),
    MultiTaskNaiveMLP(input_size=7, hidden_size=16, output_sizes=[1, 1]).to(device),
    MultiTaskMLP(input_size=7, hidden_size=18, output_sizes=[1, 1]).to(device),
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
    targets = ["MedHouseVal", "AveRooms"]
    t = trange(20, unit="epoch")
    for _epoch in t:
        train_sum = 0.0
        train_n = 0
        model.train()

        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            y = [batch[k] for k in targets]

            X = (
                torch.stack([batch[k] for k in batch if k not in targets], dim=1)
                .unsqueeze(1)
                .to(device)
            )  # (batch_size, 1, input_size)

            predictions = model(X.float())

            # compute per-task MSE (using the loss defined above) and sum them
            l = torch.stack(
                [
                    loss(pred.squeeze(), y_i.float().to(device))
                    for pred, y_i in zip(predictions, y)
                ]
            ).sum()

            l.backward()
            optimizer.step()
            train_sum += l.item() * y[0].size(0)
            train_n += y[0].size(0)
        train_loss = train_sum / train_n

        model.eval()

        val_sum = 0.0
        val_n = 0
        model.eval()
        with torch.no_grad():
            for batch in validation_dataloader:
                y = [batch[k] for k in targets]

                X = (
                    torch.stack([batch[k] for k in batch if k not in targets], dim=1)
                    .unsqueeze(1)
                    .to(device)
                )

                predictions = model(X.float())

                # compute per-task MSE (using the loss defined above) and sum them
                l = torch.stack(
                    [
                        loss(pred.squeeze(), y_i.float().to(device))
                        for pred, y_i in zip(predictions, y)
                    ]
                )
                l = l.sum()

                val_sum += l.item() * y[0].size(0)
                val_n += y[0].size(0)

        val_loss = val_sum / val_n

        results[model.__class__.__name__].append(
            {"epoch": _epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        prev_val_loss = val_loss
        prev_train_loss = train_loss
        message = f"Model {model.__class__.__name__} | Epoch {_epoch} | Train: {train_loss:.3f} | Val: {val_loss:.3f}"
        t.set_description(message)
        t.write(message)


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
