"""
=============================================================================
STEP 2 — Train Bidirectional LSTM on real MediaPipe landmark data
=============================================================================
Run AFTER step1_prepare_data.py has created X.npy and y.npy.

    python step2_train.py

Outputs (saved to ./models/):
    bilstm_sign_lang.keras   ← trained model
    best_bilstm.keras        ← best checkpoint during training
    feat_min.npy             ← normalization stats (needed for inference)
    feat_rng.npy
    classes.npy              ← label list
    training_history.png     ← loss & accuracy curves
    confusion_matrix.png     ← per-class performance
=============================================================================
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import (Bidirectional, LSTM, Dense,
                                          Dropout, BatchNormalization, Input)
from tensorflow.keras.callbacks  import (EarlyStopping, ReduceLROnPlateau,
                                          ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils      import to_categorical

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# CONFIG
# =============================================================================
DATA_CONFIG = {
    "x_path": "Sign_to_text/X.npy",
    "y_path": "Sign_to_text/y.npy"
}
MODEL_CONFIG = {
    "lstm_units_1"  : 128,
    "lstm_units_2"  : 64,
    "dense_units"   : 64,
    "dropout_rate"  : 0.4,
    "learning_rate" : 1e-3,
    "batch_size"    : 32,
    "epochs"        : 100,
    "val_split"     : 0.15,
    "test_split"    : 0.15,
}

SAVE_DIR = "./models"
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# 1.  LOAD DATA
# =============================================================================
def load_data(cfg):
    print("[INFO] Loading X.npy and y.npy …")
    X     = np.load(cfg["x_path"]).astype(np.float32)
    y_raw = np.load(cfg["y_path"], allow_pickle=True)
    print(f"[INFO] X shape : {X.shape}")
    print(f"[INFO] y shape : {y_raw.shape}")
    print(f"[INFO] Classes : {np.unique(y_raw)}")

    # Sanity check
    assert X.ndim == 3,          "X must be 3-D: (samples, timesteps, features)"
    assert len(X) == len(y_raw), "X and y must have the same number of samples"
    assert X.shape[2] == 42,     "Expected 42 features (21 landmarks × x,y)"
    return X, y_raw


# =============================================================================
# 2.  PREPROCESS
# =============================================================================
def preprocess(X, y_raw, cfg):
    le    = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    n_cls = len(le.classes_)
    print(f"\n[INFO] {n_cls} classes: {list(le.classes_)}")

    # ── Splits ───────────────────────────────────────────────────────────────
    test_val = cfg["val_split"] + cfg["test_split"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y_int, test_size=test_val, random_state=SEED)

    rel_test = cfg["test_split"] / test_val
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=SEED)

    # ── Normalize (fit on train only) ────────────────────────────────────────
    N, T, F   = X_tr.shape
    flat_tr   = X_tr.reshape(-1, F)
    feat_min  = flat_tr.min(axis=0)
    feat_max  = flat_tr.max(axis=0)
    feat_rng  = np.where((feat_max - feat_min) == 0, 1.0, feat_max - feat_min)

    def norm(arr):
        N2, T2, F2 = arr.shape
        return ((arr.reshape(-1, F2) - feat_min) / feat_rng).reshape(N2, T2, F2)

    X_tr  = norm(X_tr)
    X_val = norm(X_val)
    X_te  = norm(X_te)
    
    # REMOVE BAD CLASSES
    bad_classes = ["careful", "hungry", "write"]

    mask = ~np.isin(y_raw, bad_classes)

    X = X[mask]
    y_raw = y_raw[mask]
    # ── Class weights (handles imbalanced data) ───────────────────────────────
    unique_classes = np.unique(y_tr)
    cw_values = compute_class_weight("balanced", classes=unique_classes, y=y_tr)
    class_weights = {i: 1.0 for i in range(n_cls)}
    for i, cls in enumerate(unique_classes):
       class_weights[int(cls)] = cw_values[i]
    print(f"\n[INFO] Split sizes  →  train: {len(X_tr)}  val: {len(X_val)}  test: {len(X_te)}")
    print(f"[INFO] Class weights: { {le.classes_[k]: round(v, 2) for k, v in class_weights.items()} }")

    splits = dict(
        X_train=X_tr,  y_train=to_categorical(y_tr,  n_cls),
        X_val=X_val,   y_val=to_categorical(y_val, n_cls),
        X_test=X_te,   y_test=to_categorical(y_te,  n_cls),
        y_test_int=y_te,
    )
    meta = dict(
        label_encoder=le, n_classes=n_cls,
        timesteps=T, features=F,
        feat_min=feat_min, feat_rng=feat_rng,
        class_weights=class_weights,
    )
    return splits, meta


# =============================================================================
# 3.  MODEL
# =============================================================================
def build_model(timesteps, features, n_classes, cfg):
    model = Sequential([
        Input(shape=(timesteps, features)),

        # BiLSTM block 1
        Bidirectional(LSTM(cfg["lstm_units_1"],
                           return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.2),
                      name="bilstm_1"),
        BatchNormalization(),
        Dropout(cfg["dropout_rate"]),

        # BiLSTM block 2
        Bidirectional(LSTM(cfg["lstm_units_2"],
                           return_sequences=False,
                           dropout=0.2, recurrent_dropout=0.2),
                      name="bilstm_2"),
        BatchNormalization(),
        Dropout(cfg["dropout_rate"]),

        # Classifier head
        Dense(cfg["dense_units"], activation="relu"),
        Dropout(cfg["dropout_rate"] / 2),
        Dense(n_classes, activation="softmax"),
    ], name="BiLSTM_SignLanguage")

    model.compile(
        optimizer = Adam(learning_rate=cfg["learning_rate"]),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy",
                     tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model


# =============================================================================
# 4.  TRAIN
# =============================================================================
def train(model, splits, meta, cfg):
    ckpt = os.path.join(SAVE_DIR, "best_bilstm.keras")

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(ckpt, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    print("\n[INFO] Training …\n")
    t0 = time.time()
    history = model.fit(
        splits["X_train"], splits["y_train"],
        validation_data = (splits["X_val"], splits["y_val"]),
        epochs          = cfg["epochs"],
        batch_size      = cfg["batch_size"],
        class_weight    = meta["class_weights"],
        callbacks       = callbacks,
        verbose         = 1,
    )
    print(f"\n[INFO] Training done in {time.time()-t0:.1f}s")
    return history


# =============================================================================
# 5.  EVALUATE
# =============================================================================
def evaluate(model, splits, meta):
    le     = meta["label_encoder"]
    X_te   = splits["X_test"]
    y_true = splits["y_test_int"]

    loss, acc, top3 = model.evaluate(X_te, splits["y_test"], verbose=0)
    print(f"\n  Test Loss      : {loss:.4f}")
    print(f"  Test Accuracy  : {acc * 100:.2f}%")
    print(f"  Top-3 Accuracy : {top3 * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

    print("\n── Classification Report ─────────────────────────────────────")
    print(classification_report(y_true, y_pred, labels=np.arange(len(le.classes_)), target_names=le.classes_, zero_division=0))
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[INFO] Confusion matrix → {path}")


# =============================================================================
# 6.  PLOT TRAINING CURVES
# =============================================================================
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("BiLSTM Training — Sign Language", y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[INFO] Training curves → {path}")


# =============================================================================
# 7.  SAVE
# =============================================================================
def save(model, meta):
    path = os.path.join(SAVE_DIR, "bilstm_sign_lang.keras")
    model.save(path)
    np.save(os.path.join(SAVE_DIR, "feat_min.npy"), meta["feat_min"])
    np.save(os.path.join(SAVE_DIR, "feat_rng.npy"), meta["feat_rng"])
    np.save(os.path.join(SAVE_DIR, "classes.npy"),  meta["label_encoder"].classes_)
    print(f"[INFO] Model saved      → {path}")
    print(f"[INFO] Norm stats saved → {SAVE_DIR}/feat_min.npy, feat_rng.npy")
    print(f"[INFO] Classes saved    → {SAVE_DIR}/classes.npy")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  Sign Language BiLSTM — Training Pipeline")
    print("=" * 60)

    X, y_raw       = load_data(DATA_CONFIG)
    splits, meta   = preprocess(X, y_raw, MODEL_CONFIG)
    model          = build_model(meta["timesteps"], meta["features"],
                                 meta["n_classes"], MODEL_CONFIG)
    model.summary()
    history        = train(model, splits, meta, MODEL_CONFIG)
    plot_history(history)
    evaluate(model, splits, meta)
    save(model, meta)

    print("\n✅ Done! Next step → run step3_realtime.py to see sign-to-text live.")


if __name__ == "__main__":
    main()
