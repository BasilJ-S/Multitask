import torch
from datasets import Dataset as hfDataset
from sklearn.preprocessing import StandardScaler


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
        X = np.stack([hf_dataset[col] for col in self.feature_cols], axis=1)

        # Per task
        y_overall = []
        for i, task_targets in enumerate(self.target_cols):
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

    def __len__(self) -> int:
        return len(self.y[0])

    def __getitem__(
        self, idx: int
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        y = {f"task_{i}": y_task[idx] for i, y_task in enumerate(self.y)}
        batch = {
            "X": self.X[idx],
            "y": y,
        }
        return batch
