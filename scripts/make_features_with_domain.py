import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FEATURES_PATH = os.path.join(BASE_DIR, "dataset", "features.csv")
BALANCED_PATH = os.path.join(BASE_DIR, "dataset", "balanced_domains.csv")
OUT_PATH      = os.path.join(BASE_DIR, "dataset", "features_with_domain.csv")

df_feat = pd.read_csv(FEATURES_PATH)
df_bal  = pd.read_csv(BALANCED_PATH)

# Detect columns
if "domain" not in df_bal.columns:
    raise ValueError("❌ balanced_domains.csv must contain a 'domain' column.")
if "label" not in df_feat.columns:
    raise ValueError("❌ features.csv must contain a 'label' column.")

# Must match because features.csv was created from balanced_domains.csv
if len(df_feat) != len(df_bal):
    raise ValueError(
        f"❌ Row mismatch!\n"
        f"balanced_domains.csv rows = {len(df_bal)}\n"
        f"features.csv rows         = {len(df_feat)}\n\n"
        f"This means your features.csv was NOT generated from balanced_domains.csv."
    )

df_out = df_feat.copy()
df_out.insert(0, "domain", df_bal["domain"].astype(str).values)

df_out.to_csv(OUT_PATH, index=False)
print("✅ Saved:", OUT_PATH)
print("✅ Shape:", df_out.shape)
print("✅ First columns:", df_out.columns[:6].tolist())
