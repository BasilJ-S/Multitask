import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as hfDataset
from datasets import load_dataset
from logger import logger
from models import (
    MultiTaskHardShareMLP,
    MultiTaskNaiveMLP,
    MultiTaskResidualNetwork,
    MultiTaskTTSoftShareMLP,
    NaiveMultiTaskTimeseriesWrapper,
)
from multi_datasets import (
    PREDICTION_NODES,
    PreScaledHFDataset,
    PreScaledTimeseriesDataset,
)
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
        logger.info(
            "Running epoch with reverse scaling for interpretability on original scale."
        )

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = [0.0 for _ in task_weights]
    total_elements = [0 for _ in task_weights]

    for batch in dataloader:
        X = batch[
            "X"
        ]  # (batch_size, num_features) or (batch_size, seq_len, num_features)
        X = X.to(device)
        y = [
            batch["y"][f"task_{i}"] for i in range(len(batch["y"]))
        ]  # list of (batch_size, target_size) or (batch_size, forecast_length, target_size)

        y = [y_i.to(device) for y_i in y]
        if is_training:
            optimizer.zero_grad()
            predictions = model(X.float())
        else:
            with torch.no_grad():
                predictions = model(X.float())

        if reverse_task_scalers is not None:
            # Reverse scaling on predictions
            for task in range(len(predictions)):
                logger.debug(
                    f"Predictions for task {task} before inverse scaling: {predictions[task].shape}"
                )
                task_predictions_unscaled = [
                    torch.tensor(
                        reverse_task_scalers[task].inverse_transform(
                            predictions[task][i, :, :].cpu().numpy()
                        ),
                        dtype=torch.float32,
                    ).to(device)
                    for i in range(predictions[task].shape[0])
                ]
                task_predictions_unscaled = torch.stack(task_predictions_unscaled)
                logger.debug(
                    f"Predictions for task {task} after inverse scaling: {task_predictions_unscaled.shape}"
                )
                predictions[task] = task_predictions_unscaled
                logger.debug(f"Targets before inverse scaling: {y[task].shape}")
                task_y_unscaled = [
                    torch.tensor(
                        reverse_task_scalers[task].inverse_transform(
                            y[task][i, :, :].cpu().numpy()
                        ),
                        dtype=torch.float32,
                    ).to(device)
                    for i in range(y[task].shape[0])
                ]
                task_y_unscaled = torch.stack(task_y_unscaled)
                logger.debug(f"Targets after inverse scaling: {task_y_unscaled.shape}")
                y[task] = task_y_unscaled

        # compute per-task MSE (using the loss defined above) and sum them
        l_per_task = [
            loss_fn(pred, y_i.float().to(device)) for pred, y_i in zip(predictions, y)
        ]

        batch_elements = [l_i.numel() for l_i in l_per_task]

        # sum loss per task
        for task_idx, l_i in enumerate(l_per_task):
            total_loss[task_idx] += l_i.sum().item()
            total_elements[task_idx] += batch_elements[task_idx]

        if is_training:
            # backprop on the sum of mean losses
            batch_loss = torch.sum(torch.stack([l_i.mean() for l_i in l_per_task]))
            batch_loss.backward()
            optimizer.step()

    mean_task_losses = [tl / te for tl, te in zip(total_loss, total_elements)]
    return mean_task_losses


def prepare_housing_dataset(device=torch.device("cpu")):
    ds = load_dataset("gvlassis/california_housing").with_format("torch")

    hf_train: hfDataset = ds["train"]  # type: ignore
    hf_validation: hfDataset = ds["validation"]  # type: ignore

    targets = [["MedHouseVal", "AveRooms"], ["Longitude"]]
    task_weights = [1.0, 0.5]  # Weight for each task in the loss computation
    all_targets = [t for sublist in targets for t in sublist]
    features = [col for col in hf_train.column_names if col not in all_targets]

    for i, target_group in enumerate(targets):
        logger.info(
            f"Targets for Task {i} with weight: {task_weights[i]}: {target_group}"
        )

    # ---- CREATE PRE-SCALED DATASETS ----
    train_dataset = PreScaledHFDataset(
        hf_train, features, targets, scaler_X=None, scaler_y=None, create_scalers=True
    )
    feature_scaler = train_dataset.scaler_X
    target_scalers = train_dataset.scaler_y
    validation_dataset = PreScaledHFDataset(
        hf_validation,
        features,
        targets,
        scaler_X=feature_scaler,
        scaler_y=target_scalers,
    )
    input_size = len(features)
    output_sizes = [len(t) for t in targets]

    # use MultiTaskMLP for two targets (MedHouseVal and AveRooms)
    modelList = [
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskResidualNetwork,
            input_size=input_size,
            task_hidden_sizes=[16, 16],
            shared_hidden_sizes=[16, 16],
            output_sizes=output_sizes,
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskTTSoftShareMLP,
            input_size=input_size,
            hidden_sizes=[16, 16, 16],
            output_sizes=output_sizes,
            tt_rank=4,
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskNaiveMLP,
            input_size=input_size,
            output_sizes=output_sizes,
            hidden_sizes=[16, 16, 16],
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskHardShareMLP,
            input_size=input_size,
            shared_hidden_sizes=[18, 18, 18],
            task_hidden_sizes=[],
            output_sizes=output_sizes,
        ).to(device),
    ]
    return (
        train_dataset,
        validation_dataset,
        features,
        targets,
        task_weights,
        target_scalers,
        modelList,
    )


def prepare_ercot_dataset(device=torch.device("cpu")):

    train = pd.read_csv(
        "multitask/data/gridstatus_train_set.csv",
        index_col="interval_end_utc",
    )
    train_inference_times = pd.read_csv(
        "multitask/data/gridstatus_train_inference_times.csv"
    )
    validation = pd.read_csv(
        "multitask/data/gridstatus_validation_set.csv",
        index_col="interval_end_utc",
    )
    validation_inference_times = pd.read_csv(
        "multitask/data/gridstatus_validation_inference_times.csv"
    )

    target_cols = [[f"spp_{prediction_node}"] for prediction_node in PREDICTION_NODES]
    all_targets = [t for sublist in target_cols for t in sublist]
    features = [col for col in train.columns if col not in all_targets]
    task_weights = [1.0 for _ in target_cols]  # Weight for each task equally

    context_length = 24 * 2  # 2 days
    forecast_horizon = 24  # 1 day

    for i, target_group in enumerate(target_cols):
        logger.info(
            f"Targets for Task {i} with weight: {task_weights[i]}: {target_group}"
        )

    # ---- CREATE PRE-SCALED DATASETS ----
    train_dataset = PreScaledTimeseriesDataset(
        timeseries=train,
        inference_and_prediction_intervals=train_inference_times,
        feature_cols=features,
        target_cols=target_cols,
        create_scalers=True,
        context_window_hours=context_length,
        prediction_horizon_hours=forecast_horizon,
    )
    feature_scaler = train_dataset.scaler_X
    target_scalers = train_dataset.scaler_y
    validation_dataset = PreScaledTimeseriesDataset(
        timeseries=validation,
        inference_and_prediction_intervals=validation_inference_times,
        feature_cols=features,
        target_cols=target_cols,
        scaler_X=feature_scaler,
        scaler_y=target_scalers,
        context_window_hours=context_length,
        prediction_horizon_hours=forecast_horizon,
    )
    input_size = len(features)
    output_sizes = [len(t) for t in target_cols]

    # use MultiTaskMLP for two targets (MedHouseVal and AveRooms)
    modelList = [
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskResidualNetwork,
            input_size=input_size,
            task_hidden_sizes=[16, 16],
            shared_hidden_sizes=[16, 16],
            output_sizes=output_sizes,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskTTSoftShareMLP,
            input_size=input_size,
            hidden_sizes=[16, 16, 16],
            output_sizes=output_sizes,
            tt_rank=4,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskNaiveMLP,
            input_size=input_size,
            output_sizes=output_sizes,
            hidden_sizes=[16, 16, 16],
            context_length=context_length,
            forecast_horizon=forecast_horizon,
        ).to(device),
        NaiveMultiTaskTimeseriesWrapper(
            model_class=MultiTaskHardShareMLP,
            input_size=input_size,
            shared_hidden_sizes=[18, 18, 18],
            task_hidden_sizes=[],
            output_sizes=output_sizes,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
        ).to(device),
    ]
    return (
        train_dataset,
        validation_dataset,
        features,
        target_cols,
        task_weights,
        target_scalers,
        modelList,
    )


def get_model_name(model: torch.nn.Module) -> str:
    if isinstance(model, NaiveMultiTaskTimeseriesWrapper):
        return f"NaiveMultiTaskTimeseriesWrapper({model.model.__class__.__name__})"
    return model.__class__.__name__


if __name__ == "__main__":
    # ---- LOAD DATASET ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for preparer in [
        prepare_ercot_dataset,
        prepare_housing_dataset,
    ]:
        logger.info(f"Preparing dataset using {preparer.__name__}")
        (
            train_dataset,
            validation_dataset,
            features,
            targets,
            task_weights,
            target_scalers,
            modelList,
        ) = preparer(device=device)
        patience = 5

        loss = torch.nn.MSELoss(reduction="none")  # sum to compute per-task sums

        train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=128)

        results = {}
        for model in modelList:
            epochs_no_improve = 0
            best_val_loss = float("inf")
            results[get_model_name(model)] = []
            logger.info(
                f"Training model: {get_model_name(model)} with {sum(p.numel() for p in model.parameters())} parameters"
            )
            optimizer = optim.Adam(params=model.parameters(), lr=0.001)
            t = trange(25, unit="epoch")
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

                results[get_model_name(model)].append(
                    {"epoch": _epoch, "train_loss": train_loss, "val_loss": val_loss}
                )
                mean_val_loss = np.mean(val_loss)

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    logger.info(
                        f"No improvement in validation loss for {epochs_no_improve} epochs."
                    )
                if epochs_no_improve >= patience:
                    logger.info(
                        f"Early stopping triggered after {_epoch} epochs for model {get_model_name(model)}."
                    )
                    logger.info(
                        f"Best validation loss: {best_val_loss} achieved. Epochs without improvement: {epochs_no_improve}."
                    )
                    break

                message = f"Model {get_model_name(model)} | Epoch {_epoch} | Train: {train_loss} | Val: {val_loss}"
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
        logger.info(
            f"Final unscaled validation MSE for model {get_model_name(model)}: {val_loss_unscaled}"
        )

        # ---- PLOT RESULTS ----
        fig, axs = plt.subplots(1, 2, figsize=(28, 12))

        plot_loss_per_task(results, targets, axs, 0)
        plot_loss(results, axs, 1)

        # Add labels describing each task for whole figure
        fig.suptitle(
            f"Multi-Task MLP Model Comparison on {preparer.__name__}", fontsize=16
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
