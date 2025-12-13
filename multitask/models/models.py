import torch


class SingleTaskMLP(torch.nn.Module):
    """A basic single-task MLP model."""

    def __init__(
        self, input_size: int, hidden_sizes: list[int], output_size: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        layers = [
            torch.nn.Linear(self.input_size, self.hidden_sizes[0]),
            torch.nn.ReLU(),
        ]
        for i in range(1, len(self.hidden_sizes)):
            layers.append(
                torch.nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i])
            )
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.hidden_sizes[-1], self.output_size))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x


class MultiTaskNaiveMLP(torch.nn.Module):
    """Naive MLP with no parameter sharing between tasks."""

    def __init__(
        self, input_size: int, hidden_sizes: list[int], output_sizes: list[int]
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes

        self.models = torch.nn.ModuleList(
            [
                SingleTaskMLP(self.input_size, self.hidden_sizes, output_size)
                for output_size in self.output_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = [model(x) for model in self.models]
        return outputs


class NaiveMultiTaskTimeseriesWrapper(torch.nn.Module):
    """
    Wrapper around a non-timeseries model to handle timeseries input by flattening context length and forecast length.
    This assumes the underlying model follows the format of the other models in this file.
    This has no special handling of the time dimension; it simply flattens it into the input and output dimensions.
    """

    def __init__(
        self,
        model_class: type,
        input_size: int,
        output_sizes: list[int],
        context_length=1,
        forecast_horizon=1,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.model_class = model_class
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.kwargs = kwargs if kwargs is not None else {}

        self.model = self.model_class(
            input_size=self.input_size * self.context_length,
            output_sizes=[
                output_size * self.forecast_horizon for output_size in self.output_sizes
            ],
            **self.kwargs,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.view(x.size(0), -1)  # Flatten the input except for the batch dimension
        outputs = self.model(x)
        outputs = [
            outputs.view(x.size(0), self.forecast_horizon, output_size)
            for outputs, output_size in zip(outputs, self.output_sizes)
        ]  # reshape outputs to list of (batch_size, forecast_horizon, output_size)
        return outputs


class MultiTaskHardShareMLP(torch.nn.Module):
    """Hard parameter sharing multi-task MLP. Has shared initial layers and task-specific output layers."""

    def __init__(
        self,
        input_size: int,
        shared_hidden_sizes: list[int],
        task_hidden_sizes: list[int],
        output_sizes: list[int],
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.shared_hidden_sizes = shared_hidden_sizes
        self.task_hidden_sizes = task_hidden_sizes
        self.output_sizes = output_sizes

        shared_layers = [
            torch.nn.Linear(self.input_size, self.shared_hidden_sizes[0]),
            torch.nn.ReLU(),
        ]
        for i in range(1, len(self.shared_hidden_sizes)):
            shared_layers.append(
                torch.nn.Linear(
                    self.shared_hidden_sizes[i - 1], self.shared_hidden_sizes[i]
                )
            )
            shared_layers.append(torch.nn.ReLU())
        self.shared_layers = torch.nn.Sequential(*(shared_layers))

        self.task_layers = torch.nn.ModuleList()

        for output_size in self.output_sizes:
            task_layer_list = []
            if len(self.task_hidden_sizes) == 0:
                # No task-specific hidden layers
                task_layer_list.append(
                    torch.nn.Linear(self.shared_hidden_sizes[-1], output_size)
                )
            else:
                # Task-specific hidden layers
                task_layer_list.append(
                    torch.nn.Linear(
                        self.shared_hidden_sizes[-1], self.task_hidden_sizes[0]
                    )
                )
                task_layer_list.append(torch.nn.ReLU())

                for i in range(1, len(self.task_hidden_sizes)):
                    task_layer_list.append(
                        torch.nn.Linear(
                            self.task_hidden_sizes[i - 1], self.task_hidden_sizes[i]
                        )
                    )
                    task_layer_list.append(torch.nn.ReLU())
                task_layer_list.append(
                    torch.nn.Linear(self.task_hidden_sizes[-1], output_size)
                )
            self.task_layers.append(torch.nn.Sequential(*task_layer_list))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # (batch_size, input_size)
        x = self.shared_layers(x)  # (batch_size, shared_hidden_sizes[-1])

        outputs = [task_layers(x) for task_layers in self.task_layers]
        # list of (batch_size, output_size)
        return outputs


class MultiTaskTTSoftShareLinear(torch.nn.Module):
    """
    Shared TT layer for multi-task learning. Weights are represented in tensor train format
    with shared cores and task specific cores.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tasks: int,
        tt_rank: int,
    ) -> None:
        super().__init__()

        self.num_tasks = num_tasks
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank

        # Scaling factor for initialization to keep variance small
        self.variance_scale = 0.1

        self.task_core = torch.nn.Parameter(
            torch.randn(self.num_tasks, self.tt_rank) * self.variance_scale
        )

        self.in_core = torch.nn.Parameter(
            torch.randn(self.tt_rank, self.in_features, self.tt_rank)
            * self.variance_scale
        )

        self.out_core = torch.nn.Parameter(
            torch.randn(self.tt_rank, self.out_features) * self.variance_scale
        )

    def _get_contracted_cores(self):
        # cores = [U1, U2, ..., UN]
        W = self.task_core  # (num_tasks, tt_rank)
        W = torch.einsum(
            "ij,jkl->ikl", W, self.in_core
        )  # (num_tasks, in_features, tt_rank)
        return torch.einsum(
            "ijk,kl->ijl", W, self.out_core
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


class MultiTaskTTSoftShareMLP(torch.nn.Module):
    """
    Multi-task MLP with TT-soft sharing layers.
    Only one task specific layer to project to output size of each task.
    All other layers are shared using TT-soft sharing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_sizes: list[int],
        tt_rank=4,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes
        self.num_tasks = len(output_sizes)

        self.in_layer = MultiTaskTTSoftShareLinear(
            in_features=self.input_size,
            out_features=self.hidden_sizes[0],
            tt_rank=tt_rank,
            num_tasks=self.num_tasks,
        )

        shared_hidden_layers = []

        for layer in range(1, len(self.hidden_sizes)):
            shared_hidden_layers.append(
                MultiTaskTTSoftShareLinear(
                    in_features=self.hidden_sizes[layer - 1],
                    out_features=self.hidden_sizes[layer],
                    tt_rank=tt_rank,
                    num_tasks=self.num_tasks,
                )
            )
        self.shared_hidden_layers = torch.nn.ModuleList(shared_hidden_layers)

        self.task_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_sizes[-1], output_size)
                for output_size in self.output_sizes
            ]
        )
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x_list = [x for _ in range(self.num_tasks)]
        x_list = self.in_layer(x_list)
        x_list = [self.activation(x_i) for x_i in x_list]

        for layer in self.shared_hidden_layers:
            x_list = layer(x_list)
            x_list = [self.activation(x_i) for x_i in x_list]
        out = [layer(x_i) for x_i, layer in zip(x_list, self.task_layers)]
        return out


class MultiTaskTuckerSoftShareLinear(torch.nn.Module):
    """
    Shared Tucker layer for multi-task learning. Weights are represented in Tucker decomposition.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tasks: int,
        tucker_rank: int,
    ) -> None:
        super().__init__()

        self.num_tasks = num_tasks
        self.in_features = in_features
        self.out_features = out_features
        self.tucker_rank = tucker_rank

        # Scaling factor for initialization to keep variance small
        self.variance_scale = 0.1

        self.tensor_core = torch.nn.Parameter(
            torch.randn(self.tucker_rank, self.tucker_rank, self.tucker_rank)
            * self.variance_scale
        )

        self.task_core = torch.nn.Parameter(
            torch.randn(self.num_tasks, self.tucker_rank) * self.variance_scale
        )

        self.in_core = torch.nn.Parameter(
            torch.randn(self.in_features, self.tucker_rank) * self.variance_scale
        )

        self.out_core = torch.nn.Parameter(
            torch.randn(self.out_features, self.tucker_rank) * self.variance_scale
        )

    def _get_contracted_cores(self):
        W = torch.einsum(
            "ij,jkl->ikl", self.task_core, self.tensor_core
        )  # (num_tasks, tucker_rank, tucker_rank)
        W = torch.einsum(
            "ij,ljk->lik", self.in_core, W
        )  # (num_tasks, in_features, tucker_rank)
        return torch.einsum(
            "ij,klj->kli", self.out_core, W
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


class MultiTaskTuckerSoftShareMLP(torch.nn.Module):
    """
    Multi-task MLP with TT-soft sharing layers.
    Only one task specific layer to project to output size of each task.
    All other layers are shared using TT-soft sharing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_sizes: list[int],
        tucker_rank=4,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes
        self.num_tasks = len(output_sizes)

        self.in_layer = MultiTaskTuckerSoftShareLinear(
            in_features=self.input_size,
            out_features=self.hidden_sizes[0],
            tucker_rank=tucker_rank,
            num_tasks=self.num_tasks,
        )

        shared_hidden_layers = []

        for layer in range(1, len(self.hidden_sizes)):
            shared_hidden_layers.append(
                MultiTaskTuckerSoftShareLinear(
                    in_features=self.hidden_sizes[layer - 1],
                    out_features=self.hidden_sizes[layer],
                    tucker_rank=tucker_rank,
                    num_tasks=self.num_tasks,
                )
            )
        self.shared_hidden_layers = torch.nn.ModuleList(shared_hidden_layers)

        self.task_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_sizes[-1], output_size)
                for output_size in self.output_sizes
            ]
        )
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x_list = [x for _ in range(self.num_tasks)]
        x_list = self.in_layer(x_list)
        x_list = [self.activation(x_i) for x_i in x_list]

        for layer in self.shared_hidden_layers:
            x_list = layer(x_list)
            x_list = [self.activation(x_i) for x_i in x_list]
        out = [layer(x_i) for x_i, layer in zip(x_list, self.task_layers)]
        return out


class MultiTaskResidualNetwork(torch.nn.Module):
    """
    Multi-task network with residual shared layers and task-specific MLPs.
    """

    def __init__(
        self,
        input_size: int,
        task_hidden_sizes: list[int],
        shared_hidden_sizes: list[int],
        output_sizes: list[int],
    ):
        super().__init__()
        self.input_size = input_size
        self.task_hidden_sizes = task_hidden_sizes
        self.shared_hidden_sizes = shared_hidden_sizes
        self.output_sizes = output_sizes
        self.num_tasks = len(output_sizes)

        individual_output_size = sum(output_sizes)
        self.task_networks = torch.nn.ModuleList(
            [
                SingleTaskMLP(self.input_size, self.task_hidden_sizes, output_size)
                for output_size in self.output_sizes
            ]
        )
        self.shared_layers = SingleTaskMLP(
            individual_output_size, self.shared_hidden_sizes, individual_output_size
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x_list = [
            task_network(x) for task_network in self.task_networks
        ]  # num tasks x (batch_size, output_size)

        x = torch.cat(x_list, dim=-1)  # (batch_size, sum(output_sizes))
        shared = self.shared_layers(x)  # (batch_size, sum(output_sizes))
        x = x + shared  # Residual connection
        out = [
            x[:, sum(self.output_sizes[:i]) : sum(self.output_sizes[: i + 1])]
            for i in range(self.num_tasks)
        ]  # list of (batch_size, output_size)
        return out
