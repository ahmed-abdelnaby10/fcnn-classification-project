# 🔍 Hyperparameters Analysis

## Overview

This analysis explores how tuning key hyperparameters affects model performance.

### 1️⃣ Learning Rate (LR)

| LR   | Accuracy | Loss | Comments              |
| ---- | -------- | ---- | --------------------- |
| 1e-4 | 0.80     | 0.45 | too slow convergence  |
| 1e-3 | 0.85     | 0.33 | optimal balance       |
| 1e-2 | 0.78     | 0.51 | overshooting observed |

**→ Optimal:** `1e-3` with Adam optimizer

---

### 2️⃣ Dropout Rate

| Dropout | Val Accuracy | Comments              |
| ------- | ------------ | --------------------- |
| 0.2     | 0.83         | light regularization  |
| 0.3     | 0.85         | stable generalization |
| 0.5     | 0.82         | slight underfitting   |

**→ Optimal:** `0.3`

---

### 3️⃣ Batch Size

| Batch Size | Val Accuracy | Notes              |
| ---------- | ------------ | ------------------ |
| 16         | 0.84         | noisy but fast     |
| 32         | 0.86         | balanced stability |
| 64         | 0.84         | slower convergence |

**→ Optimal:** `32`

---

### 4️⃣ Activation Functions

| Activation | Val Accuracy | Comments                     |
| ---------- | ------------ | ---------------------------- |
| ReLU       | 0.86         | fast, no vanishing gradients |
| LeakyReLU  | 0.85         | stable but slower            |
| Sigmoid    | 0.78         | saturation issues            |

**→ Optimal:** `ReLU`

---

### ✅ Final Recommended Configuration

| Parameter  | Value      |
| ---------- | ---------- |
| Optimizer  | Adam(1e-3) |
| Dropout    | 0.3        |
| Batch Size | 32         |
| Activation | ReLU       |
| Epochs     | 100        |

**Final Test Accuracy:** 0.84

---

> All results, plots, and logs are reproducible with the provided notebooks:
>
> * `01_EDA_and_preprocessing.ipynb`
> * `02_model_training.ipynb`
> * `03_hyperparameter_tuning.ipynb`

**Author:** Ahmed Mohamed Abdelnaby — FCNN Diabetes Project