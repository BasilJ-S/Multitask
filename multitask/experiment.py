import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

from multitask.data_provider.data_providers import (
    prepare_dataset,
    prepare_ercot_full,
    prepare_housing_dataset,
    prepare_weather_full,
    prepare_weather_multiloc_full,
)
from multitask.data_provider.multi_datasets import (
    PreScaledHFDataset,
    PreScaledTimeseriesDataset,
)
from multitask.models.baselines import ALL_BASELINES
from multitask.models.model_trials import MODEL_LIST
from multitask.models.models import NaiveMultiTaskTimeseriesWrapper
from multitask.utils.logger import logger
from multitask.utils.plotter import (
    plot_loss,
    plot_task_loss_same_plot,
    plot_task_loss_separately,
    plot_test_loss,
)


def extract_batch(batch, device):
    X = batch["X"].to(device)
    y = [batch["y"][f"task_{i}"].to(device) for i in range(len(batch["y"]))]
    return X, y


def forward_pass(model, X, is_training, optimizer):
    if is_training:
        optimizer.zero_grad()
        return model(X.float())
    with torch.no_grad():
        return model(X.float())


def validate_scalers(scalers, task_weights):
    if len(scalers) != len(task_weights):
        raise ValueError("Mismatch: reverse_task_scalers vs. task count.")
    logger.info("Inverse-scaling enabled for interpretability.")


def inverse_scale_all_tasks(predictions, targets, scalers, device):
    new_preds, new_targets = [], []
    for task_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
        scaler = scalers[task_idx]

        pred_np = pred.cpu().numpy().reshape(-1, pred.shape[-1])
        tgt_np = tgt.cpu().numpy().reshape(-1, tgt.shape[-1])

        pred_unscaled = scaler.inverse_transform(pred_np)
        tgt_unscaled = scaler.inverse_transform(tgt_np)

        pred_unscaled = torch.tensor(pred_unscaled, device=device).reshape(pred.shape)
        tgt_unscaled = torch.tensor(tgt_unscaled, device=device).reshape(tgt.shape)

        new_preds.append(pred_unscaled)
        new_targets.append(tgt_unscaled)

    return new_preds, new_targets


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
    If optimizer is provided → training mode, otherwise eval mode.
    Returns: mean loss per task.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    if reverse_task_scalers is not None:
        validate_scalers(reverse_task_scalers, task_weights)

    total_loss = np.zeros(len(task_weights))
    total_count = np.zeros(len(task_weights))

    for batch in dataloader:
        X, y = extract_batch(batch, device)

        predictions = forward_pass(model, X, is_training, optimizer)

        if reverse_task_scalers is not None:
            predictions, y = inverse_scale_all_tasks(
                predictions, y, reverse_task_scalers, device
            )

        # ---- compute per-task loss ----
        per_task_loss = get_loss_per_task(predictions, y, loss_fn)

        # accumulate weighted totals
        for t, loss_val in enumerate(per_task_loss):
            bs = y[t].shape[0]
            total_loss[t] += loss_val.item() * bs
            total_count[t] += bs

        # backprop
        if is_training:
            total_batch_loss = torch.stack(per_task_loss).sum()
            total_batch_loss.backward()
            optimizer.step()

    return (total_loss / total_count).tolist()


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
    X_train_full, y_train_full = extract_batch(full_batch, device=torch.device("cpu"))

    full_validation_dataloader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset)
    )
    full_validation_batch = next(iter(full_validation_dataloader))
    X_validation_full, y_validation_full = extract_batch(
        full_validation_batch, device=torch.device("cpu")
    )

    logger.info(f"X_train_full shape: {X_train_full.shape}")
    logger.info(f"y_train_full shapes: {[y.shape for y in y_train_full]}")
    logger.info(f"X_validation_full shape: {X_validation_full.shape}")
    logger.info(f"y_validation_full shapes: {[y.shape for y in y_validation_full]}")

    baseline_results = {}

    # Global Mean Predictor
    for baseline_model in ALL_BASELINES:
        logger.info(f"Running baseline: {baseline_model.name}...")

        train_pred, val_pred = baseline_model.fit_and_predict(
            X_train_full, y_train_full, X_validation_full, y_validation_full
        )

        train_losses = get_loss_per_task(train_pred, y_train_full, loss_fn)
        val_losses = get_loss_per_task(val_pred, y_validation_full, loss_fn)

        train_losses = [l.item() for l in train_losses]
        val_losses = [l.item() for l in val_losses]

        logger.info(f"{baseline_model.name} train losses per task: {train_losses}")
        logger.info(f"{baseline_model.name} val losses per task: {val_losses}")

        baseline_results[baseline_model.name] = {
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
    l_per_task = [loss_fn(pred, y_i.float()) for pred, y_i in zip(predictions, targets)]
    l_mean_per_task = [torch.mean(l_i) for l_i in l_per_task]
    return l_mean_per_task


def get_model_name(model: torch.nn.Module) -> str:
    if isinstance(model, NaiveMultiTaskTimeseriesWrapper):
        return f"NaiveMultiTaskTimeseriesWrapper({model.model.__class__.__name__})"
    return model.__class__.__name__


def train_and_evaluate(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    loss: torch.nn.Module,
    task_weights: list[float],
    target_scalers: list[StandardScaler],
    device=torch.device("cpu"),
    lr: float = 0.001,
    wd: float = 0.0,
) -> list[dict]:
    patience = 5
    epochs_no_improve = 0
    best_val_loss = float("inf")
    results = []
    logger.info(
        f"Training model: {get_model_name(model)} with {sum(p.numel() for p in model.parameters())} parameters"
    )
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
    t = trange(150, unit="epoch")
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

        results.append(
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
        if _epoch % 10 == 0:
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
    results[-1]["val_loss_unscaled"] = val_loss_unscaled
    return results


def objective_builder(
    model_objective,
    model_class,
    input_size: int,
    output_sizes: list[int],
    context_length: int,
    forecast_horizon: int,
    device=torch.device("cpu"),
    seed=42,
    save_dir: str | None = None,
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    def objective(trial):
        nonlocal best_loss
        current_seed = seed + trial.number
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        model_params = model_objective(trial)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        model = NaiveMultiTaskTimeseriesWrapper(
            model_class,
            input_size=input_size,
            output_sizes=output_sizes,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            **model_params,
        ).to(device)
        # Here you would add code to train the model and evaluate it
        # For simplicity, let's assume we return a dummy validation loss
        results = train_and_evaluate(
            model,
            train_dataloader,
            validation_dataloader,
            loss,
            task_weights,
            target_scalers,
            device=device,
            lr=lr,
            wd=wd,
        )
        val_loss = results[-1]["val_loss"]
        trial.set_user_attr("per_task_val_loss", val_loss)
        val_loss = np.mean(val_loss)
        train_loss = results[-1]["train_loss"]
        val_loss_unscaled = results[-1]["val_loss_unscaled"]
        trial.set_user_attr("train_loss", np.mean(train_loss))
        trial.set_user_attr("val_loss_unscaled", np.mean(val_loss_unscaled))

        if val_loss < best_loss:
            best_loss = val_loss
            if save_dir is not None:
                model_path = os.path.join(save_dir, "best_model.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "trial_params": trial.params,
                        "val_loss": val_loss,
                        "trial_number": trial.number,
                    },
                    model_path,
                )
                logger.info(
                    f"New best model saved! Trial {trial.number}, Loss: {val_loss:.4f}"
                )

        return val_loss

    return objective


def get_save_dir(preparer_name: str, model_name: str) -> str:
    return f"multitask/model_store/{preparer_name}/{model_name}"


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Multi-Task Learning Experiment")
    args.add_argument(
        "--study",
        action="store_true",
        help="Run study on train and eval sets",
    )
    args.add_argument("--eval", action="store_true", help="Evaluate models on test set")
    args = args.parse_args()
    if not args.study and not args.eval:
        logger.error("Please specify --study and/or --eval to run the experiment.")
        exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.study:
        # ---- LOAD DATASET ----
        for preparer in [
            prepare_weather_multiloc_full,
            prepare_ercot_full,
            prepare_weather_full,
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
                input_size,
                output_sizes,
                context_length,
                forecast_horizon,
            ) = prepare_dataset(preparer, is_train=True)

            # Define study

            model_list = MODEL_LIST

            loss = torch.nn.MSELoss(reduction="none")  # sum to compute per-task sums

            train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            validation_dataloader = DataLoader(validation_dataset, batch_size=128)

            results = {}  # run_baselines(train_dataset, validation_dataset)
            for model_cls, model_objective, _ in model_list:
                study = optuna.create_study(
                    direction="minimize",
                    study_name=f"{preparer.__name__}_{model_cls.__name__}_study",
                    storage="sqlite:///multitask_model_comparison.db",
                    load_if_exists=True,
                )
                save_dir = get_save_dir(preparer.__name__, model_cls.__name__)

                objective = objective_builder(
                    model_objective,
                    model_cls,
                    input_size,
                    output_sizes,
                    context_length,
                    forecast_horizon,
                    device=device,
                    save_dir=save_dir,
                )
                study.optimize(objective, n_trials=20)

    if args.eval:

        results = {}
        for preparer in [
            prepare_weather_multiloc_full,
            prepare_ercot_full,
            prepare_weather_full,
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
                input_size,
                output_sizes,
                context_length,
                forecast_horizon,
            ) = prepare_dataset(preparer, is_train=False)

            # Define study

            model_list = MODEL_LIST

            loss = torch.nn.MSELoss(reduction="none")  # sum to compute per-task sums

            train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            validation_dataloader = DataLoader(validation_dataset, batch_size=128)

            results = run_baselines(train_dataset, validation_dataset)
            test_results = results.copy()
            for k, v in results.items():
                test_results[k] = v["val_loss"]
            for model_cls, model_objective, param_converter in model_list:

                model_state_dict = torch.load(
                    os.path.join(
                        get_save_dir(preparer.__name__, model_cls.__name__),
                        "best_model.pt",
                    ),
                    map_location=device,
                    weights_only=False,
                )

                # Use the parameters that were saved with the model to ensure architecture matches
                best_params = model_state_dict.get("trial_params", {})
                logger.info(
                    f"Best params for model {model_cls.__name__} on dataset {preparer.__name__}: {best_params}"
                )
                model_params = {
                    k: v
                    for k, v in best_params.items()
                    if k not in ["lr", "weight_decay", "seed"]
                }
                model_params = param_converter(**model_params)

                model = NaiveMultiTaskTimeseriesWrapper(
                    model_cls,
                    input_size=input_size,
                    output_sizes=output_sizes,
                    context_length=context_length,
                    forecast_horizon=forecast_horizon,
                    **model_params,
                ).to(device)
                model.load_state_dict(model_state_dict["model_state_dict"])

                lr = best_params["lr"]
                wd = best_params["weight_decay"]
                test_loss_unscaled = run_epoch(
                    validation_dataloader,
                    model,
                    loss,
                    task_weights=task_weights,
                    optimizer=None,
                    device=device,
                    reverse_task_scalers=target_scalers,
                )
                test_results[get_model_name(model)] = test_loss_unscaled
                clean_model = NaiveMultiTaskTimeseriesWrapper(
                    model_cls,
                    input_size=input_size,
                    output_sizes=output_sizes,
                    context_length=context_length,
                    forecast_horizon=forecast_horizon,
                    **model_params,
                ).to(device)
                results[get_model_name(model)] = train_and_evaluate(
                    model,
                    train_dataloader,
                    validation_dataloader,
                    loss,
                    task_weights,
                    target_scalers,
                    device=device,
                    lr=lr,
                    wd=wd,
                )
            with open(f"test_results_{preparer.__name__}.json", "w") as file:
                json.dump(
                    test_results, file, indent=4
                )  # 'indent' makes the file human-readable
            with open(f"training_results_{preparer.__name__}.json", "w") as file:
                json.dump(
                    results, file, indent=4
                )  # 'indent' makes the file human-readable
            with open(f"targets_{preparer.__name__}.json", "w") as file:
                json.dump(targets, file, indent=4)
            logger.info(f"Test results: {test_results}")
            logger.info(f"Training results: {results}")
            logger.info(f"Targets: {targets}")

            # ---- PLOT RESULTS ----
            plot_task_loss_separately(results, targets)
            fig, axs = plt.subplots(1, 2, figsize=(28, 12))

            plot_task_loss_same_plot(results, targets, axs, 0)
            plot_loss(results, axs, 1)
            plot_test_loss(test_results, targets)

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
