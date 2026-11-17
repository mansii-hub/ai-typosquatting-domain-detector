import pandas as pd
import numpy as np
import math
import re

print("Loading dataset...")

df = pd.read_csv("../dataset/balanced_domains.csv")

# -----------------------------
# Helper functions
# -----------------------------

def entropy(s):
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    return -sum([p * math.log(p) / math.log(2.0) for p in prob]) if len(s) > 0 else 0

def extract_features(domain):
    d = str(domain).lower()
    length = len(d)

    vowels = "aeiou"
    digits = sum(c.isdigit() for c in d)
    hyphens = d.count('-')
    dots = d.count('.')
    special = sum(not c.isalnum() for c in d)

    words = d.split('.')
    tokens = re.split(r'[-.]', d)

    features = {

        # Basic
        "length": length,
        "num_digits": digits,
        "num_hyphens": hyphens,
        "num_dots": dots,
        "num_special": special,

        # Ratios
        "digit_ratio": digits/length if length else 0,
        "hyphen_ratio": hyphens/length if length else 0,

        # Letters
        "num_vowels": sum(c in vowels for c in d),
        "num_consonants": sum(c.isalpha() and c not in vowels for c in d),

        # Structure
        "num_tokens": len(tokens),
        "num_subdomains": len(words)-2 if len(words) > 2 else 0,
        "longest_token": max([len(t) for t in tokens]) if tokens else 0,

        # Patterns
        "starts_digit": int(d[0].isdigit()) if length else 0,
        "ends_digit": int(d[-1].isdigit()) if length else 0,
        "double_hyphen": int("--" in d),

        # Entropy
        "entropy": entropy(d),

        # Suspicious words
        "has_login": int("login" in d),
        "has_verify": int("verify" in d),
        "has_bank": int("bank" in d),
        "has_secure": int("secure" in d),
        "has_account": int("account" in d),
        "has_update": int("update" in d),
        "has_free": int("free" in d),
        "has_bonus": int("bonus" in d),
        "has_offer": int("offer" in d),
        "has_pay": int("pay" in d),

        # Brand bait words
        "has_google": int("google" in d),
        "has_facebook": int("facebook" in d),
        "has_amazon": int("amazon" in d),
        "has_paypal": int("paypal" in d),

        # TLD risk
        "has_xyz": int(".xyz" in d),
        "has_top": int(".top" in d),
        "has_ru": int(".ru" in d),
        "has_cn": int(".cn" in d),

        # Repetition
        "max_repeat_char": max([d.count(c) for c in set(d)]) if length else 0,
    }

    return list(features.values())

# -----------------------------
# Extract features
# -----------------------------

X = []
for domain in df["domain"]:
    X.append(extract_features(domain))

X = np.array(X)
y = df["label"]

features_df = pd.DataFrame(X)
features_df["label"] = y

features_df.to_csv("../dataset/features.csv", index=False)

print("PRO feature extraction done!")
print("Total features:", X.shape[1])
print("Saved to dataset/features.csv")
