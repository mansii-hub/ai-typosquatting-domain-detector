# import os
# import pandas as pd
# import joblib

# from scipy.sparse import hstack, csr_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

# from nlp_features import get_tfidf_features
# from xgboost import XGBClassifier

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# data_path = os.path.join(BASE_DIR, "dataset", "features.csv")
# model_path = os.path.join(BASE_DIR, "model", "domain_detector_xgb.pkl")

# os.makedirs(os.path.dirname(model_path), exist_ok=True)

# print("🚀 Loading dataset...")
# df = pd.read_csv(data_path)

# # LIMIT SIZE
# df = df.sample(50000, random_state=42)

# df["label"] = df["label"].map({"safe":0,"fake":1})

# domains = df["domain"].astype(str)

# # NUMERIC FEATURES ONLY
# numeric_features = df[
#     ["length","entropy","num_digits","num_special_chars","num_dots"]
# ]

# numeric_sparse = csr_matrix(numeric_features.values)

# print("🧠 TF-IDF...")
# tfidf, vectorizer = get_tfidf_features(domains.tolist())

# X = hstack([numeric_sparse, tfidf])
# y = df["label"]

# X_train,X_test,y_train,y_test = train_test_split(
#     X,y,test_size=0.2,random_state=42,stratify=y
# )

# scale = len(y[y==0]) / len(y[y==1])

# print("⚡ Training XGBoost...")

# model = XGBClassifier(
#     n_estimators=300,
#     max_depth=10,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     scale_pos_weight=scale,
#     eval_metric="logloss",
#     tree_method="hist",
#     random_state=42
# )

# model.fit(X_train,y_train)

# pred = model.predict(X_test)

# print(confusion_matrix(y_test,pred))
# print(classification_report(y_test,pred))

# joblib.dump((model,vectorizer),model_path)


# print("✅ Model saved:",model_path)
import os
import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Import nlp_features from same scripts/ folder ──────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from nlp_features import get_tfidf_features

from xgboost import XGBClassifier

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path   = os.path.join(BASE_DIR, "dataset", "features.csv")
model_dir   = os.path.join(BASE_DIR, "model")
model_path  = os.path.join(model_dir, "domain_detector_xgb.pkl")

os.makedirs(model_dir, exist_ok=True)

# ── Load dataset ────────────────────────────────────────────────────────────
print("🚀 Loading dataset...")

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"\n❌  features.csv not found at:\n    {data_path}\n"
        "    Please run feature_extraction.py first to generate it."
    )

df = pd.read_csv(data_path)
print(f"   ✅ Loaded {len(df)} rows from features.csv")

# ── FIX: sample only if we have more rows than the target ───────────────────
SAMPLE_TARGET = 50_000
if len(df) >= SAMPLE_TARGET:
    df = df.sample(SAMPLE_TARGET, random_state=42)
    print(f"   🔀 Sampled {SAMPLE_TARGET} rows for training")
else:
    print(f"   ⚠️  Dataset has only {len(df)} rows — using all of them.")
    print("       Model will train but may not generalise well on unseen domains.")
    print("       Consider adding more labelled data to features.csv.\n")

# ── Label encode ────────────────────────────────────────────────────────────
if df["label"].dtype == object:
    df["label"] = df["label"].map({"safe": 0, "fake": 1})

# Drop any rows where label mapping failed
before = len(df)
df = df.dropna(subset=["label"])
if len(df) < before:
    print(f"   ⚠️  Dropped {before - len(df)} rows with unrecognised labels.")

df["label"] = df["label"].astype(int)
domains = df["domain"].astype(str)

# ── Numeric features ────────────────────────────────────────────────────────
numeric_cols = ["length", "entropy", "num_digits", "num_special_chars", "num_dots"]
for col in numeric_cols:
    if col not in df.columns:
        raise ValueError(
            f"❌  Column '{col}' missing from features.csv.\n"
            "    Re-run feature_extraction.py to regenerate it."
        )

numeric_sparse = csr_matrix(df[numeric_cols].values)

# ── TF-IDF features ─────────────────────────────────────────────────────────
print("🧠 Building TF-IDF features...")
tfidf, vectorizer = get_tfidf_features(domains.tolist())

# ── Combine ─────────────────────────────────────────────────────────────────
X = hstack([numeric_sparse, tfidf])
y = df["label"]

# ── Train / test split ──────────────────────────────────────────────────────
# Use stratify only when both classes are present
unique_classes = y.nunique()
if unique_classes < 2:
    raise ValueError(
        "❌  Dataset must contain both 'safe' and 'fake' labelled domains.\n"
        "    Add more labelled data and re-run feature_extraction.py."
    )

# For very small datasets fall back to a smaller test size
test_size = 0.2 if len(df) >= 20 else 0.1

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42,
    stratify=y,
)

# ── Class balance weight ─────────────────────────────────────────────────────
n_safe = len(y[y == 0])
n_fake = len(y[y == 1])
scale  = n_safe / max(n_fake, 1)          # avoid zero-division

print(f"   📊 Class balance  →  safe: {n_safe}  |  fake: {n_fake}  "
      f"|  scale_pos_weight: {scale:.2f}")

# ── Train XGBoost ────────────────────────────────────────────────────────────
print("\n⚡ Training XGBoost...")

model = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 10,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = scale,
    eval_metric      = "logloss",
    tree_method      = "hist",
    random_state     = 42,
)

model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
pred = model.predict(X_test)
print("\n📋 Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\n📋 Classification Report:")
print(classification_report(y_test, pred))

# ── Save model + vectorizer ──────────────────────────────────────────────────
joblib.dump((model, vectorizer), model_path)
print(f"\n✅  Model saved  →  {model_path}")