# Quick Reference: Enhanced Metrics System

## What Was Changed

Your `run_epoch()` function in [experiment.py](multitask/experiment.py) has been enhanced to compute **MAE, MSE, RMSE** metrics when evaluating on test data.

## Key Update

The **final test run** (line 654-664) now automatically collects predictions and targets:

```python
test_loss = run_epoch(
    test_dataloader,
    model,
    loss,
    task_weights=task_weights,
    optimizer=None,
    device=device,
    reverse_task_scalers=None,
    i_want_ground_truth_for_interpretability=True,  # ← NEW FLAG
)
```

## What You Get

Your test results JSON files now contain:

```json
{
  "model_name": {
    "task_0": {
      "mse": 0.1234,
      "mae": 0.0567,
      "rmse": 0.3513,
      "predictions": [1.2, 3.4, 5.6, ...],
      "targets": [1.1, 3.2, 5.8, ...]
    },
    "task_1": { ... }
  }
}
```

## New Visualizations

When you run:

```bash
poetry run python ./multitask/utils/plotter_results.py
```

You'll automatically get:

1. **Violin Plots** (`prediction_errors_violin_*.png`)

   - Shows distribution of prediction errors per task
   - One plot per task, one violin per model

2. **Metrics Comparison** (`metrics_comparison_*.png`)

   - Bar charts comparing MAE, MSE, RMSE across models
   - One subplot per metric

3. **Console Output**
   - Detailed statistics for each model and task

## Backward Compatibility

✅ No changes needed to training or validation code  
✅ Old code still works as before  
✅ Metrics are only computed when `i_want_ground_truth_for_interpretability=True`

## Example: Custom Metric Computation

If you want to use the detailed metrics in your own code:

```python
# In experiment.py or your analysis script
from multitask.experiment import run_epoch

# Run evaluation with detailed metrics
detailed_results = run_epoch(
    test_dataloader,
    model,
    loss_fn,
    task_weights=task_weights,
    optimizer=None,
    i_want_ground_truth_for_interpretability=True,
)

# Access metrics per task
for task_idx, metrics in detailed_results.items():
    print(f"{task_idx}:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")

    # Use predictions and targets for custom analysis
    preds = metrics['predictions']
    targets = metrics['targets']
```

## Files Modified

- [multitask/experiment.py](multitask/experiment.py) - Enhanced `run_epoch()`
- [multitask/utils/plotter.py](multitask/utils/plotter.py) - New plotting functions
- [multitask/utils/plotter_results.py](multitask/utils/plotter_results.py) - Integrated visualizations

---

**That's it!** Run your experiments normally and the metrics will be collected and visualized automatically. 🎉
