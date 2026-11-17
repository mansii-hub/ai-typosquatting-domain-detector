import pandas as pd
import os

# === File Paths ===
base_path = "E:/Project/dataset"
safe_path = os.path.join(base_path, "clean_safe_domains.csv")
fake_path = os.path.join(base_path, "fake_domains.csv")
output_path = os.path.join(base_path, "merged_domains.csv")

print("🚀 Loading datasets...")

# === Read CSVs ===
safe_df = pd.read_csv(safe_path)
fake_df = pd.read_csv(fake_path)

print(f"✅ Safe file shape: {safe_df.shape}")
print(f"✅ Fake file shape: {fake_df.shape}")

# === Ensure Same Columns ===
required_columns = ["domain", "label", "type_of_typosquatting", "target_domain"]
safe_df = safe_df[required_columns]
fake_df = fake_df[required_columns]

# === Merge + Clean ===
merged_df = pd.concat([safe_df, fake_df], ignore_index=True)
merged_df.drop_duplicates(subset=["domain"], inplace=True)

# === Shuffle the dataset (important!) ===
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Save Final Dataset ===
merged_df.to_csv(output_path, index=False)

print(f"\n✅ Merged & shuffled dataset saved at: {output_path}")
print("\nRows per label:")
print(merged_df['label'].value_counts())

# === Show few random samples ===
print("\n🔍 Sample records:")
print(merged_df.sample(5))
