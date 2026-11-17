import os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

def main():

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    DATA_PATH   = os.path.join(BASE_DIR, "dataset", "features_with_domain.csv")
    MODEL_PATH  = os.path.join(BASE_DIR, "model", "typosquat_hybrid.keras")
    SCALER_PATH = os.path.join(BASE_DIR, "model", "hybrid_scaler.pkl")
    META_PATH   = os.path.join(BASE_DIR, "model", "hybrid_meta.json")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Loaded shape:", df.shape)

    if "domain" not in df.columns or "label" not in df.columns:
        raise ValueError("features_with_domain.csv must contain 'domain' and 'label'")

    y_raw = df["label"].astype(str).str.lower().str.strip()
    y = (y_raw == "fake").astype(int).values

    domains = df["domain"].astype(str).str.lower().values

    X_num_df = df.drop(columns=["domain", "label"])
    X_num_df = X_num_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_num = X_num_df.values.astype(np.float32)

    print("Numeric features:", X_num.shape[1])

    Xn_train, Xn_test, d_train, d_test, y_train, y_test = train_test_split(
        X_num, domains, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    Xn_test_scaled = scaler.transform(Xn_test)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    threshold = 0.5
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            if "best_threshold" in meta:
                threshold = float(meta["best_threshold"])
        except:
            pass

    print("Using threshold:", threshold)

    probs = model.predict({"domain": d_test, "num": Xn_test_scaled}, batch_size=1024).ravel()
    preds = (probs >= threshold).astype(int)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)

    try:
        auc = roc_auc_score(y_test, probs)

        fpr, tpr, thresholds = roc_curve(y_test, probs)

        plt.figure()
        plt.plot(fpr, tpr, label=f'Hybrid CNN+NN (AUC = {auc:.3f})')
        plt.plot([0,1], [0,1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Typosquatting Detection Model")
        plt.legend()
        plt.grid()

        plt.savefig("roc_curve.png")
        plt.show()

    except:
        auc = None

    print("\nFINAL METRICS")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    if auc is not None:
        print("ROC-AUC  :", auc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()