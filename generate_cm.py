import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import math
from collections import Counter
import os

# ------------------ FUNCTIONS ------------------

def entropy(domain):
    counts = Counter(domain)
    probs = [c/len(domain) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def extract_numeric_features(domain):
    return [
        len(domain),
        entropy(domain),
        sum(c.isdigit() for c in domain),
        sum(not c.isalnum() for c in domain),
        domain.count('.')
    ]

# ------------------ LOAD MODEL ------------------

model, vectorizer = joblib.load("model/domain_detector_xgb.pkl")

# ------------------ LOAD DATASET ------------------

df = pd.read_csv("dataset/balanced_domains.csv")

# ------------------ FIX LABELS ------------------

# Convert "safe"/"fake" → 0/1
df["label"] = df["label"].str.lower().map({"safe": 0, "fake": 1})

# ------------------ PREPARE FEATURES ------------------

X_text = vectorizer.transform(df["domain"])
X_num = csr_matrix([extract_numeric_features(d) for d in df["domain"]])
X = hstack([X_num, X_text])

y_true = df["label"]
y_pred = model.predict(X)

# ------------------ CONFUSION MATRIX ------------------

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Fake"])
disp.plot()

# ------------------ SAVE IMAGE ------------------

os.makedirs("scripts", exist_ok=True)
plt.savefig("scripts/confusion_matrix.png")
plt.close()

print("✅ confusion_matrix.png successfully created in 'scripts' folder")