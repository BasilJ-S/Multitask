# Summary of Changes: Enhanced Metrics and Visualization

## Overview

Updated the test evaluation pipeline to compute and visualize multiple statistical metrics (MAE, MSE, RMSE) with violin plots for prediction error analysis.

## Changes Made

### 1. **[experiment.py](multitask/experiment.py)** - Enhanced `run_epoch()` Function

#### What Changed:

- Modified the `run_epoch()` function to optionally collect predictions and targets
- Added computation of detailed metrics when `i_want_ground_truth_for_interpretability=True`
- Function now returns either:
  - **Training mode**: Simple loss list (backward compatible)
  - **Eval mode without ground truth**: Simple loss list (backward compatible)
  - **Eval mode with ground truth**: Detailed metrics dict with MAE, MSE, RMSE, predictions, and targets per task

#### Key Changes:

```python
# Old: Only collected loss
return (total_loss / total_count).tolist()

# New: Returns detailed metrics when flag is set
if i_want_ground_truth_for_interpretability:
    # Computes MAE, MSE, RMSE per task
    # Returns: {task_0: {mse, mae, rmse, predictions, targets}, ...}
    return metrics_per_task
else:
    return (total_loss / total_count).tolist()
```

#### Test Run Update:

- Updated the final test run call to pass `i_want_ground_truth_for_interpretability=True`
- This automatically enables detailed metrics collection during test evaluation

**Files Modified:**

- `multitask/experiment.py` (lines 75-151 and 654-664)

---

### 2. **[plotter.py](multitask/utils/plotter.py)** - New Visualization Functions

#### Added Functions:

##### `plot_prediction_errors_violin()`

- Creates violin plots showing the distribution of absolute prediction errors
- One subplot per task
- Shows median, mean, and full distribution of errors
- Usage: Understand which tasks have more variable prediction errors

##### `plot_prediction_metrics_comparison()`

- Creates bar plots comparing MAE, MSE, RMSE across models
- One subplot per metric
- Color-coded by model for easy comparison
- Displays metrics for all tasks side-by-side
- Usage: Compare model performance across different metrics and tasks

**Files Modified:**

- `multitask/utils/plotter.py` (new imports + 2 new functions at end)

---

### 3. **[plotter_results.py](multitask/utils/plotter_results.py)** - Integrated New Visualizations

#### What Changed:

- Added imports for the new plotting functions
- Added automatic detection of detailed metrics format
- Aggregates metrics across trials for visualization
- Generates new plots if detailed metrics are available

#### New Output:

When you run the results plotting script, it now automatically:

1. ✅ Detects if test results contain detailed metrics
2. ✅ Creates violin plots of prediction errors
3. ✅ Creates bar plots comparing MAE, MSE, RMSE
4. ✅ Prints detailed metrics summary to console

**Files Modified:**

- `multitask/utils/plotter_results.py` (imports + section at end)

---

## Usage

### Running Tests

Your test evaluation now automatically captures detailed metrics:

```python
test_loss = run_epoch(
    test_dataloader,
    model,
    loss,
    task_weights=task_weights,
    optimizer=None,
    device=device,
    reverse_task_scalers=None,
    i_want_ground_truth_for_interpretability=True,  # ← Enables detailed metrics
)
```

### Viewing Results

Run the plotter as before:

```bash
poetry run python ./multitask/utils/plotter_results.py
```

The script will now automatically:

- Generate violin plots: `plots/prediction_errors_violin_*.png`
- Generate metric comparison plots: `plots/metrics_comparison_*.png`
- Print detailed MAE/MSE/RMSE statistics to console

---

## Metrics Computed

For each task and each model:

- **MAE** (Mean Absolute Error): Average absolute difference between predictions and targets
- **MSE** (Mean Squared Error): Average squared difference (penalizes larger errors more)
- **RMSE** (Root Mean Squared Error): Square root of MSE (same units as original data)

---

## Backward Compatibility

✅ **Fully backward compatible:**

- Training mode unchanged (doesn't collect ground truth)
- Default behavior unchanged (loss-only returns)
- Can still use old `run_epoch()` calls without the new flag

---

## Next Steps

You can now:

1. Run your experiments as normal - metrics are collected automatically
2. View detailed metrics visualizations in the `plots/` folder
3. Compare model performance across MAE, MSE, RMSE metrics
4. Analyze prediction error distributions with violin plots
