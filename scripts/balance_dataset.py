import pandas as pd
import os

# === File Paths ===
base_path = "E:/Project/dataset"
merged_path = os.path.join(base_path, "merged_domains.csv")
balanced_path = os.path.join(base_path, "balanced_domains.csv")

# === Load dataset ===
df = pd.read_csv(merged_path)

print("✅ Loaded merged dataset:", df.shape, "rows")

# === Check label column values ===
print("\n🔍 Unique label values:", df["label"].unique())

# === Map numeric to readable labels if needed ===
if df["label"].dtype != 'object':
    df["label"] = df["label"].map({0: "safe", 1: "fake"})

# === Check class distribution ===
print("\n🔍 Label distribution before balancing:")
print(df["label"].value_counts())

# === Separate safe and fake ===
safe = df[df["label"] == "safe"]
fake = df[df["label"] == "fake"]

# === Find smaller size ===
min_count = min(len(safe), len(fake))

# === Randomly sample both equally ===
safe_sampled = safe.sample(min_count, random_state=42)
fake_sampled = fake.sample(min_count, random_state=42)

# === Merge & shuffle ===
balanced_df = pd.concat([safe_sampled, fake_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# === Save new balanced dataset ===
balanced_df.to_csv(balanced_path, index=False)

print(f"\n✅ Balanced dataset created with {len(balanced_df)} total rows")
print("✅ Each class has:", min_count, "rows")
print("📁 Saved to:", balanced_path)

print("\n🔍 Sample rows:")
print(balanced_df.head(10))
