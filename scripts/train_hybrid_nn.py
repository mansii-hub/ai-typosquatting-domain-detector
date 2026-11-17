import os, json, joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve

from tensorflow.keras import layers, models, callbacks

# ================= PATHS =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "features_with_domain.csv")

OUT_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(OUT_DIR, "typosquat_hybrid.keras")
SCALER_PATH = os.path.join(OUT_DIR, "hybrid_scaler.pkl")
META_PATH   = os.path.join(OUT_DIR, "hybrid_meta.json")

# ================= SETTINGS =================
SEED = 42

# For speed on CPU, train on a sample first.
# After you confirm it works, increase to 500000 or 1000000.
MAX_ROWS = 300000

EPOCHS = 15
BATCH_SIZE = 512
LR = 1e-3

MAX_TOKENS = 6000
SEQ_LEN = 80

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}

def best_threshold_f1(y_true, y_prob):
    p, r, th = precision_recall_curve(y_true, y_prob)
    f1 = (2 * p * r) / (p + r + 1e-9)
    idx = int(np.nanargmax(f1))
    thr = 0.5 if idx == 0 else float(th[idx - 1])
    return thr, float(f1[idx])

def build_model(num_features, text_vec):
    # inputs
    in_domain = layers.Input(shape=(), dtype=tf.string, name="domain")
    in_num    = layers.Input(shape=(num_features,), dtype=tf.float32, name="num")

    # ----- TEXT BRANCH (CNN) -----
    x_txt = text_vec(in_domain)
    x_txt = layers.Embedding(input_dim=MAX_TOKENS, output_dim=64)(x_txt)

    x_txt = layers.Conv1D(128, 3, padding="same", activation="relu")(x_txt)
    x_txt = layers.BatchNormalization()(x_txt)
    x_txt = layers.Dropout(0.20)(x_txt)

    x_txt = layers.Conv1D(128, 5, padding="same", activation="relu")(x_txt)
    x_txt = layers.BatchNormalization()(x_txt)
    x_txt = layers.Dropout(0.20)(x_txt)

    x_txt = layers.GlobalMaxPooling1D()(x_txt)
    x_txt = layers.Dense(128, activation="relu")(x_txt)
    x_txt = layers.Dropout(0.20)(x_txt)

    # ----- NUMERIC BRANCH (NN) -----
    x_num = layers.Dense(256, activation="relu")(in_num)
    x_num = layers.BatchNormalization()(x_num)
    x_num = layers.Dropout(0.30)(x_num)

    x_num = layers.Dense(128, activation="relu")(x_num)
    x_num = layers.BatchNormalization()(x_num)
    x_num = layers.Dropout(0.25)(x_num)

    # merge
    x = layers.Concatenate()([x_txt, x_num])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(1, activation="sigmoid", name="prob_fake")(x)

    model = models.Model(inputs={"domain": in_domain, "num": in_num}, outputs=out, name="Typosquat_Hybrid_CNN_NN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")]
    )
    return model

def main():
    set_seed(SEED)

    print("🚀 Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("✅ Loaded:", df.shape)

    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)
        print("⚡ Sampled:", df.shape)

    # domain + label + 35 numeric
    domains = df["domain"].astype(str).str.lower().values

    y_raw = df["label"].astype(str).str.lower().str.strip()
    y = (y_raw == "fake").astype(int).values

    X_num_df = df.drop(columns=["domain", "label"])
    X_num_df = X_num_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_num = X_num_df.values.astype(np.float32)
    num_features = X_num.shape[1]

    print("✅ Numeric features:", num_features)
    print("✅ Fake ratio:", round(y.mean(), 4))

    # split
    Xn_train, Xn_temp, d_train, d_temp, y_train, y_temp = train_test_split(
        X_num, domains, y, test_size=0.25, random_state=SEED, stratify=y
    )
    Xn_val, Xn_test, d_val, d_test, y_val, y_test = train_test_split(
        Xn_temp, d_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    # scale numeric
    scaler = StandardScaler()
    Xn_train = scaler.fit_transform(Xn_train)
    Xn_val   = scaler.transform(Xn_val)
    Xn_test  = scaler.transform(Xn_test)

    # char-level vectorizer
    text_vec = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        split="character"
    )
    text_vec.adapt(d_train)

    cw = class_weights(y_train)
    print("⚖️ Class weights:", cw)

    model = build_model(num_features=num_features, text_vec=text_vec)
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    ]

    print("\n🧠 Training hybrid CNN+NN...")
    model.fit(
        {"domain": d_train, "num": Xn_train}, y_train,
        validation_data=({"domain": d_val, "num": Xn_val}, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=cb,
        verbose=1
    )

    best = tf.keras.models.load_model(MODEL_PATH)

    val_prob = best.predict({"domain": d_val, "num": Xn_val}, batch_size=1024).ravel()
    test_prob = best.predict({"domain": d_test, "num": Xn_test}, batch_size=1024).ravel()

    thr, best_f1 = best_threshold_f1(y_val, val_prob)

    print("\n📊 RESULTS")
    print("VAL AUC:", round(roc_auc_score(y_val, val_prob), 4))
    print("TEST AUC:", round(roc_auc_score(y_test, test_prob), 4))
    print("Best threshold:", round(thr, 4), "| Best F1:", round(best_f1, 4))

    test_pred = (test_prob >= thr).astype(int)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))
    print("\nReport:\n", classification_report(y_test, test_pred, digits=4))

    # Save scaler + meta (threshold)
    joblib.dump(scaler, SCALER_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"best_threshold": float(thr)}, f, indent=2)

    print("\n💾 Saved:")
    print(" - Model:", MODEL_PATH)
    print(" - Scaler:", SCALER_PATH)
    print(" - Meta:", META_PATH)

if __name__ == "__main__":
    main()
    