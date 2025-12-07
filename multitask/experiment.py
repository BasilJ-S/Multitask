import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
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
from plotter import plot_loss, plot_task_loss_same_plot, plot_task_loss_separately
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
        l_per_task = get_loss_per_task(predictions, y, loss_fn)

        # sum loss per task
        for task_idx, l_i in enumerate(l_per_task):
            total_loss[task_idx] += l_i.detach().item() * y[task_idx].shape[0]
            total_elements[task_idx] += y[task_idx].shape[0]

        if is_training:
            # backprop on the sum of mean losses
            batch_loss = torch.sum(torch.stack([l_i for l_i in l_per_task]))
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


def run_baselines(
    train_dataset: PreScaledHFDataset | PreScaledTimeseriesDataset,
    validation_dataset: PreScaledHFDataset | PreScaledTimeseriesDataset,
    loss_fn: torch.nn.Module = torch.nn.MSELoss(reduction="none"),
):
    """
    Run simple baselines: Mean Predictor and Last Value Predictor.
    Returns a dictionary with baseline names as keys and their validation losses as values.
    """

    full_dataset_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
    full_batch = next(iter(full_dataset_dataloader))
    X_train_full = full_batch["X"].cpu()
    y_train_full = [
        full_batch["y"][f"task_{i}"].cpu() for i in range(len(full_batch["y"]))
    ]
    logger.info(f"X_train_full shape: {X_train_full.shape}")
    logger.info(f"y_train_full shapes: {[y.shape for y in y_train_full]}")

    full_validation_dataloader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset)
    )
    full_validation_batch = next(iter(full_validation_dataloader))
    X_validation_full = full_validation_batch["X"].cpu()
    y_validation_full = [
        full_validation_batch["y"][f"task_{i}"].cpu()
        for i in range(len(full_validation_batch["y"]))
    ]

    logger.info(f"X_validation_full shape: {X_validation_full.shape}")
    logger.info(f"y_validation_full shapes: {[y.shape for y in y_validation_full]}")

    baseline_results = {}

    # Global Mean Predictor
    logger.info("Running Global Mean Predictor baseline...")
    val_pred: list[torch.Tensor] = []
    train_pred: list[torch.Tensor] = []
    for task_idx in range(len(y_train_full)):
        task_y_train = y_train_full[task_idx]
        task_y_train_mean = torch.mean(task_y_train, dim=0)  # mean over batch
        task_y_train_mean = torch.mean(
            task_y_train_mean, dim=0
        )  # mean over time/targets if needed

        logger.info(
            f"Task {task_idx} mean value over training set: {task_y_train_mean}"
        )

        task_y_mean_train_size = task_y_train_mean.repeat(
            task_y_train.shape[0], task_y_train.shape[1], 1
        )

        train_pred.append(task_y_mean_train_size)

        task_y_mean_validation_size = task_y_train_mean.repeat(
            y_validation_full[task_idx].shape[0],
            y_validation_full[task_idx].shape[1],
            1,
        )
        logger.info(
            f"Mean Predictor predictions for task {task_idx} shape: {task_y_mean_validation_size.shape}"
        )
        val_pred.append(task_y_mean_validation_size)

    train_losses = get_loss_per_task(train_pred, y_train_full, loss_fn)
    val_losses = get_loss_per_task(val_pred, y_validation_full, loss_fn)

    logger.info(f"Global Mean Predictor train losses per task: {train_losses}")
    logger.info(f"Global Mean Predictor val losses per task: {val_losses}")

    baseline_results["global_mean"] = {
        "epoch": "all",
        "train_loss": train_losses,
        "val_loss": val_losses,
    }

    ## Linear Predictor
    val_pred = []
    train_pred = []
    logger.info("Running Linear Predictor baseline...")
    for task_idx in range(len(y_train_full)):
        y_train_task = y_train_full[task_idx]  # (B, T_f, D_out)
        X_train_task = X_train_full  # (B, T_c, D_in)
        B, T_c, D_in = X_train_task.shape
        _, T_f, D_out = y_train_task.shape

        B_val, _, _ = X_validation_full.shape

        # Solve linear model for each forecast step separately
        w_list = []
        train_pred_task = []
        val_pred_task = []

        for t in range(T_f):
            y_t = y_train_task[:, t, :]  # (B, D_out)
            X_flat = X_train_task.reshape(B, -1)  # flatten context: (B, T_c*D_in)
            X_b = torch.cat([X_flat, torch.ones(B, 1)], dim=1)  # (B, T_c*D_in+1)

            # Least squares
            w = torch.linalg.lstsq(X_b, y_t).solution  # (T_c*D_in+1, D_out)
            w_list.append(w)

            # Predictions
            train_pred_task.append((X_b @ w).unsqueeze(1))  # (B,1,D_out)
            X_val_flat = X_validation_full.reshape(B_val, -1)
            X_val_b = torch.cat([X_val_flat, torch.ones(B_val, 1)], dim=1)
            val_pred_task.append((X_val_b @ w).unsqueeze(1))

        # Concatenate across forecast horizon
        train_pred.append(torch.cat(train_pred_task, dim=1))  # (B, T_f, D_out)
        val_pred.append(torch.cat(val_pred_task, dim=1))

    train_losses = get_loss_per_task(train_pred, y_train_full, loss_fn)
    val_losses = get_loss_per_task(val_pred, y_validation_full, loss_fn)

    logger.info(f"Linear Predictor train losses per task: {train_losses}")
    logger.info(f"Linear Predictor val losses per task: {val_losses}")

    baseline_results["linear"] = {
        "epoch": "all",
        "train_loss": train_losses,
        "val_loss": val_losses,
    }

    # XGBoost Predictor
    logger.info("Running XGBoost baseline...")
    val_pred = []
    train_pred = []
    for task_idx in range(len(y_train_full)):
        logger.info(f"Training XGBoost for task {task_idx}...")
        task_y_train = y_train_full[task_idx].reshape(
            y_train_full[task_idx].shape[0], -1
        )  # flatten (B, T_f*D_out)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=10,
            max_depth=3,
            subsample=0.5,
            colsample_bytree=0.5,
            gamma=1,
            eta=0.1,
        )
        model.fit(
            X_train_full.reshape(X_train_full.shape[0], -1).numpy(),
            task_y_train.numpy(),
        )

        train_pred_task = model.predict(
            X_train_full.reshape(X_train_full.shape[0], -1).numpy()
        )
        val_pred_task = model.predict(
            X_validation_full.reshape(X_validation_full.shape[0], -1).numpy()
        )

        train_pred.append(
            torch.tensor(train_pred_task, dtype=torch.float32).reshape_as(
                y_train_full[task_idx]
            )
        )
        val_pred.append(
            torch.tensor(val_pred_task, dtype=torch.float32).reshape_as(
                y_validation_full[task_idx]
            )
        )

    train_losses = get_loss_per_task(train_pred, y_train_full, loss_fn)
    val_losses = get_loss_per_task(val_pred, y_validation_full, loss_fn)
    logger.info(f"XGBoost train losses per task: {train_losses}")
    logger.info(f"XGBoost val losses per task: {val_losses}")

    baseline_results["xgboost"] = {
        "epoch": "all",
        "train_loss": train_losses,
        "val_loss": val_losses,
    }

    return baseline_results


def get_loss_per_task(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    loss_fn: torch.nn.Module,
) -> list[torch.Tensor]:
    """
    Compute per-task loss given predictions and targets.
    Returns a list of losses for each task.
    """
    l_per_task = [
        loss_fn(pred, y_i.float().to(device)) for pred, y_i in zip(predictions, targets)
    ]
    l_mean_per_task = [torch.mean(l_i) for l_i in l_per_task]
    return l_mean_per_task


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

        results = run_baselines(train_dataset, validation_dataset)
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
        plot_task_loss_separately(results, targets)
        fig, axs = plt.subplots(1, 2, figsize=(28, 12))

        plot_task_loss_same_plot(results, targets, axs, 0)
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
