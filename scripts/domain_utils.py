    
import re
import difflib
import tldextract
from typing import Tuple, List, Dict

# --------- LEET / SIMPLE MAP ----------
LEET_MAP = {
    '0': 'o',
    '1': 'l',
    '3': 'e',
    '4': 'a',
    '5': 's',
    '@': 'a'
}

# --------- MULTI-CHAR VISUAL ATTACKS ----------
MULTI_CHAR_MAP = {
    'rn': 'm',
    'nn': 'm',
    'vv': 'w'
}

# --------- HOMOGLYPHS ----------
HOMOGLYPHS = {
    '\u0131': 'i',
    '\u0430': 'a',
    '\u0435': 'e',
    '\u0456': 'i',
    '\u03BF': 'o',
    '\u03B1': 'a',
}

# ----------------- Helpers -----------------
def strip_tld(domain: str) -> str:
    ext = tldextract.extract(domain)
    return ext.domain or domain


def normalize_domain(domain: str) -> str:
    domain = domain.strip().lower()

    # Remove protocol and paths
    domain = re.sub(r'https?://', '', domain)
    domain = domain.split('/')[0]
    domain = domain.split(':')[0]
    domain = re.sub(r'^www\.', '', domain)

    main = strip_tld(domain)

    # 🔥 1. Apply homoglyph replacements
    for k, v in HOMOGLYPHS.items():
        main = main.replace(k, v)

    # 🔥 2. Apply multi-character substitutions FIRST
    for k, v in MULTI_CHAR_MAP.items():
        main = main.replace(k, v)

    # 🔥 3. Apply single-character substitutions
    for k, v in LEET_MAP.items():
        main = main.replace(k, v)

    # 🔥 4. Reduce repeated characters
    main = re.sub(r'(.)\1{2,}', r'\1\1', main)

    # 🔥 5. Remove unwanted characters
    main = re.sub(r'[^a-z0-9\-]', '', main)

    return main


def extract_tokens(domain: str):
    domain = domain.lower()
    domain = re.sub(r'https?://', '', domain)
    domain = domain.split('/')[0]
    domain = domain.split(':')[0]

    base = strip_tld(domain)

    # split on dash, underscore
    tokens = re.split(r'[-_]', base)

    return tokens

def brand_similarity_score(domain: str, brand_list: List[str]) -> Tuple[float, str]:
    norm = normalize_domain(domain)
    tokens = extract_tokens(domain)

    best_brand = None
    best_score = 0.0

    for b in brand_list:
        b = b.lower()

        # 🔥 1. DIRECT TOKEN MATCH
        if b in tokens:
            return 1.0, b

        # 🔥 2. LENGTH FILTER
        if abs(len(norm) - len(b)) > 3:
            continue

        # 🔥 3. SIMILARITY
        score = difflib.SequenceMatcher(None, norm, b).ratio()

        if score > best_score:
            best_score = score
            best_brand = b

    # 🔥 4. THRESHOLD
    if best_score < 0.75:
        return 0.0, None

    return best_score, best_brand

# ----------------- Visual substitution -----------------
def detect_visual_substitutions(domain: str, brand: str) -> List[str]:
    if not brand:
        return []

    raw = strip_tld(domain.lower())
    brand = brand.lower()
    detected = []

    # 🔥 Multi-character substitutions
    for fake, real in MULTI_CHAR_MAP.items():
        if fake in raw:
            detected.append(f"Visual substitution detected: '{fake}' → '{real}'")

    # 🔥 Single-character substitutions
    for fake, real in LEET_MAP.items():
        if fake in raw:
            detected.append(f"Visual substitution detected: '{fake}' → '{real}'")

    # 🔥 Homoglyphs
    for fake, real in HOMOGLYPHS.items():
        if fake in raw:
            detected.append(f"Visual homoglyph detected: '{fake}' → '{real}'")

    # 🔥 Visual impersonation check
    modified = raw
    for fake, real in {**MULTI_CHAR_MAP, **LEET_MAP}.items():
        modified = modified.replace(fake, real)

    similarity = difflib.SequenceMatcher(None, modified, brand).ratio()

    if similarity > 0.85 and raw != brand:
        detected.append(f"Visual impersonation: '{raw}' looks like '{brand}'")

    return list(set(detected))


# ----------------- Edit distance / typos -----------------
def explain_char_difference(fake: str, real: str) -> str:
    matcher = difflib.SequenceMatcher(None, fake, real)

    inserts = []
    deletes = []
    replaces = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            inserts.append(real[j1:j2])
        elif tag == "delete":
            deletes.append(fake[i1:i2])
        elif tag == "replace":
            replaces.append((fake[i1:i2], real[j1:j2]))

    # 🔥 PRIORITY FIX (GLOBAL SOLUTION)

    # Case 1: Proper replacement
    if replaces:
        f, r = replaces[0]
        return f"Character substitution: '{f}' → '{r}'"

    # Case 2: BOTH insert + delete → treat as substitution
    if inserts and deletes:
        return f"Character substitution: '{deletes[0]}' → '{inserts[0]}'"

    # Case 3: Only insert → missing character
    if inserts:
        return f"Missing character(s): '{inserts[0]}'"

    # Case 4: Only delete → extra character
    if deletes:
        return f"Extra character(s): '{deletes[0]}'"

    return "Minor character manipulation detected"


def detect_swaps(fake: str, real: str) -> List[str]:
    swaps = []

    # Try swapping every adjacent pair in fake
    for i in range(len(fake) - 1):
        swapped = list(fake)
        swapped[i], swapped[i+1] = swapped[i+1], swapped[i]

        if "".join(swapped) == real:
            swaps.append(
                f"Swapped characters: '{fake[i]}' ↔ '{fake[i+1]}' at position {i}"
            )

    return swaps

def detect_edit_distance_typos(domain: str, brand: str) -> List[str]:
    if not brand:
        return []

    base = strip_tld(domain.lower())
    brand = brand.lower()

    ratio = difflib.SequenceMatcher(None, base, brand).ratio()

    alerts = []

    # 🔥 1. Exact swap detection (your existing function)
    swap_alerts = detect_swaps(base, brand)
    alerts.extend(swap_alerts)

    # 🔥 2. Partial swap detection (NEW ADDITION)
    if not swap_alerts:
        for i in range(min(len(base), len(brand)) - 1):
            if base[i] == brand[i+1] and base[i+1] == brand[i]:
                alerts.append(
                    f"Swapped characters: '{base[i]}' ↔ '{base[i+1]}' near position {i}"
                )
                break  # avoid multiple noisy messages

    # 🔥 3. Typo detection (only if no swap found)
    if 0.75 < ratio < 1.0 and not alerts:
        explanation = explain_char_difference(base, brand)
        alerts.append(f"Typo detected: {explanation} ('{base}' vs '{brand}')")

    return alerts

# ----------------- TLD typos -----------------
def detect_tld_typos(domain: str) -> List[str]:
    ext = tldextract.extract(domain.lower())
    tld = ext.suffix

    alerts = []
    common_tlds = ["com", "net", "org"]

    for ctld in common_tlds:
        sim = difflib.SequenceMatcher(None, tld, ctld).ratio()

        if tld != ctld and sim > 0.75:
            alerts.append(f"TLD typo detected: '{tld}' looks like '{ctld}'")

    return alerts


# ----------------- Suspicious features -----------------
def extract_suspicious_features(domain: str) -> Dict[str, float]:
    ext = tldextract.extract(domain.lower())
    name = ext.domain or domain

    return {
        "unusual_length": 1.0 if len(name) > 12 else 0.0,
        "many_symbols": 1.0 if re.search(r'[^a-z0-9\-]', name) else 0.0,
        "repeated_chars": 1.0 if re.search(r'(.)\1\1', name) else 0.0,
        "multiple_subdomains": 1.0 if ext.subdomain and '.' in ext.subdomain else 0.0,
        "has_dash": 1.0 if '-' in name else 0.0
    }


# ----------------- Risk scoring -----------------
def compute_risk_score(
    domain: str,
    model_prob_fake: float,
    brand_list: List[str],
    suspicious_features: Dict[str, float]
) -> Tuple[float, Dict]:

    sim, brand = brand_similarity_score(domain, brand_list)

    if not brand:
        sim = 0

    edit_alerts = detect_edit_distance_typos(domain, brand)
    visual_alerts = detect_visual_substitutions(domain, brand)
    tld_alerts = detect_tld_typos(domain)

    ext = tldextract.extract(domain.lower())
    tld = ext.suffix

    brand_tld_abuse = brand and ext.domain == brand and tld not in ["com", "net", "org"]

    score = (
        0.5 * model_prob_fake +
        0.3 * sim +
        0.2 * (sum(suspicious_features.values()) / max(1, len(suspicious_features)))
    )

    score = round(min(100, score * 100), 1)

    reasons = []

    if sim > 0.75 and brand:
        reasons.append(f"Domain resembles brand '{brand}'")
        reasons.extend(visual_alerts)
        reasons.extend(edit_alerts)

    if brand_tld_abuse:
        reasons.append(f"Brand abuse with suspicious TLD '.{tld}'")

    reasons.extend(tld_alerts)

    if model_prob_fake > 0.7:
        reasons.append(f"ML probability high: {model_prob_fake:.2f}")

    for k, v in suspicious_features.items():
        if v == 1.0:
            reasons.append(f"Suspicious pattern: {k}")

    if not reasons:
        reasons.append("No strong phishing indicators detected")

    return score, {
        "risk_score": score,
        "brand_similarity": sim,
        "matched_brand": brand,
        "model_probability": model_prob_fake,
        "reasons": reasons,
        "is_fake": (
            score >= 60 or
            model_prob_fake > 0.7 or
            (sim > 0.75 and brand is not None) or
            brand_tld_abuse or
            len(visual_alerts) > 0
        )
    }