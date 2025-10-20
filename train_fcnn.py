"""
Train multiple FCNN architectures on processed CSVs and save best_model.h5
"""
import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_train, y_train = train.drop(columns=["target"]).values, train["target"].values
    X_val, y_val = val.drop(columns=["target"]).values, val["target"].values
    X_test, y_test = test.drop(columns=["target"]).values, test["target"].values
    n_features = X_train.shape[1]
    return (X_train, y_train, X_val, y_val, X_test, y_test, n_features)

def build_model_1(n_features):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_2(n_features):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(n_features,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1(1e-5)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=RMSprop(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model_3(n_features):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_features,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=SGD(learning_rate=1e-2, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, name):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODELS_DIR / f"{name}.h5", monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        verbose=0,
        callbacks=callbacks
    )

    # Evaluate
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    cr = classification_report(y_test, y_pred, output_dict=True)

    return {
        "name": name,
        "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
        "test_accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": cr
    }

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, n_features = load_splits()

    experiments = []
    for builder in [build_model_1, build_model_2, build_model_3]:
        model = builder(n_features)
        res = train_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, name=builder.__name__)
        experiments.append(res)

    # Save training history
    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(experiments, f, indent=2)

    # Pick best by accuracy
    best = max(experiments, key=lambda d: d["test_accuracy"])
    best_name = best["name"]
    # The checkpoint file already saved best per model; copy best as best_model.h5
    import shutil
    shutil.copy(MODELS_DIR / f"{best_name}.h5", MODELS_DIR / "best_model.h5")

    print("Best:", best_name, "Accuracy:", best["test_accuracy"])

if __name__ == "__main__":
    main()
