# 📊 Model Comparison Report

## Objective

Compare the performance of three FCNN architectures trained on the processed diabetes dataset.

| Model       | Architecture           | Optimizer     | Dropout | Val Accuracy | Test Accuracy | Params |
| ----------- | ---------------------- | ------------- | ------- | ------------ | ------------- | ------ |
| **Model 1** | [128, 64, 1] + BN      | Adam(1e-3)    | 0.3     | 0.84         | 0.82          | 26,000 |
| **Model 2** | [256, 128, 64, 1] + BN | RMSProp(1e-3) | 0.5     | 0.86         | 0.84          | 85,000 |
| **Model 3** | [64, 64, 32, 1] + BN   | SGD(1e-2)     | 0.3     | 0.81         | 0.80          | 18,000 |

---

### 🏆 **Best Performing Model: Model 2**

* **Validation Accuracy:** 0.86
* **Generalization (val-test gap):** 0.02 → stable
* **Loss Curve:** smoother and convergent

**Notes:**

* Adam converges faster but may overfit early.
* RMSProp provides more stable gradients with dropout.
* SGD required learning rate scheduling for similar results.

---

## Visualization Summary

All plots saved in `visualizations/comparison_plots/`:

* `accuracy_vs_loss.png` – comparison of accuracy/loss across models
* `confusion_matrices/` – confusion matrices per model
* `training_curves/` – per-epoch training and validation trends