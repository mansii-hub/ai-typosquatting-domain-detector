import pandas as pd
import os

# File paths
base_path = "E:/Project/dataset"
safe_path = os.path.join(base_path, "safe_domains.csv")

# Read the safe CSV
df = pd.read_csv(safe_path, header=None, low_memory=False)

print("✅ Original safe file shape:", df.shape)
print("First few rows before cleaning:")
print(df.head())

# Drop numeric ID columns and rows with missing domains
df = df.dropna()

# Case 1: if CSV looks like [1, google.com, safe]
if df.shape[1] >= 2:
    df = df.iloc[:, 1]  # keep only domain column

# Create clean structure
clean_df = pd.DataFrame()
clean_df["domain"] = df.astype(str).str.strip()
clean_df["label"] = 0  # 0 = legitimate
clean_df["type_of_typosquatting"] = "legitimate"
clean_df["target_domain"] = clean_df["domain"]

# Save cleaned file
output_path = os.path.join(base_path, "clean_safe_domains.csv")
clean_df.to_csv(output_path, index=False)

print(f"\n✅ Cleaned safe dataset saved to: {output_path}")
print(clean_df.head())
