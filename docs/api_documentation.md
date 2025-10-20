# 📘 API Documentation

## Overview

This document describes the API endpoints used in the FCNN Classification Project for diabetes prediction.

---

### 🧩 **1. `/predict`**

**Method:** `POST`

**Description:**
Accepts processed or raw patient data and returns a binary prediction for diabetes (1 = diabetic, 0 = non-diabetic).

**Request Body Example (JSON):**

```json
{
  "Pregnancies": 3,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 25,
  "Insulin": 100,
  "BMI": 28.0,
  "DiabetesPedigreeFunction": 0.45,
  "Age": 33
}
```

**Response Example:**

```json
{
  "prediction": 1,
  "probability": 0.87
}
```

**Error Responses:**

* `400`: Missing or invalid input.
* `500`: Internal server error.

---

### ⚙️ **2. `/train`**

**Method:** `POST`

**Description:**
Triggers model training using preprocessed datasets (`train.csv`, `val.csv`). Saves the trained model to `models/` directory.

**Response Example:**

```json
{
  "status": "training_completed",
  "model_path": "models/fcnn_v1.h5",
  "val_accuracy": 0.84
}
```

---

### 🧠 **3. `/evaluate`**

**Method:** `GET`

**Description:**
Loads the best saved model and evaluates it on `test.csv`, returning accuracy, loss, and confusion matrix.

**Response Example:**

```json
{
  "accuracy": 0.85,
  "loss": 0.32,
  "confusion_matrix": [[80, 10], [7, 50]]
}
```

---

### 🧾 **4. `/health`**

**Method:** `GET`

**Description:**
Checks the API status and model availability.

**Response Example:**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Deployment Notes

* Framework: **FastAPI**
* Model: **Fully Connected Neural Network (FCNN)**
* Input Validation: via **Pydantic** models
* Default port: **8000**
