import torch


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
