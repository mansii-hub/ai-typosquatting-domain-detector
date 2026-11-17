
import os
import socket
import math
import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import re
import ipaddress
from difflib import SequenceMatcher
from urllib.parse import urlparse

try:
    from . import domain_utils as domain_utils
    from .domain_utils import (
        normalize_domain,
        strip_tld,
        brand_similarity_score,
        detect_visual_substitutions,
        detect_edit_distance_typos,
        detect_tld_typos,
        extract_suspicious_features,
    )
except ImportError:
    import domain_utils as domain_utils
    from domain_utils import (
        normalize_domain,
        strip_tld,
        brand_similarity_score,
        detect_visual_substitutions,
        detect_edit_distance_typos,
        detect_tld_typos,
        extract_suspicious_features,
    )

tldextract = domain_utils.tldextract

# ======================================================
# DNS EXISTENCE CHECK
# ======================================================
def domain_exists(domain: str) -> bool:
    try:
        socket.gethostbyname(domain)
        return True
    except socket.gaierror:
        return False

# ======================================================
# DOMAIN VALIDATION
# ======================================================
DOMAIN_REGEX_STRICT = re.compile(
    r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)\.(com|net|org)$"
)
DOMAIN_REGEX_GENERAL = re.compile(
    r"^(?=.{1,253}$)([A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,63}$"
)

SUSPICIOUS_PATH_WORDS = {
    "login", "verify", "secure", "update", "account", "password",
    "signin", "banking", "confirm", "validate", "wallet", "payment",
    "reset", "support", "auth", "recover",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rebrand.ly", "shorturl.at", "s.id",
    "rb.gy", "lnkd.in", "bitly.com", "tiny.cc",
}

SAFE_PORTS = {80, 443}

def is_valid_domain(domain: str) -> bool:
    """Returns True only for clean alpha-numeric domains."""
    return bool(DOMAIN_REGEX_GENERAL.match(domain))


def normalize_url_input(raw_input: str) -> dict:
    """Parse plain domains and full URLs into a consistent structure."""
    original_input = (raw_input or "").strip()
    candidate = original_input
    if candidate and "://" not in candidate and not candidate.startswith("//"):
        candidate = f"//{candidate}"

    parsed = urlparse(candidate)
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        scheme = ""

    port = parsed.port
    path = parsed.path or ""
    query = parsed.query or ""

    host_ascii = host
    is_unicode_spoof = any(ord(ch) > 127 for ch in host)
    is_punycode = "xn--" in host
    try:
        host_ascii = host.encode("idna").decode("ascii")
        is_punycode = is_punycode or "xn--" in host_ascii
    except UnicodeError:
        pass

    ext = tldextract.extract(host_ascii)
    registered_domain = ext.registered_domain or host_ascii
    subdomain = ext.subdomain or ""

    return {
        "original_input": original_input,
        "normalized_url": parsed.geturl() if "://" in original_input else original_input,
        "scheme": scheme,
        "host": host_ascii,
        "display_host": host,
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "path": path,
        "query": query,
        "port": port,
        "has_full_url": bool(urlparse(original_input).scheme),
        "is_punycode": is_punycode,
        "is_unicode_spoof": is_unicode_spoof,
    }


def is_ip_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def build_url_analysis(parsed_url: dict) -> dict:
    host = parsed_url["host"]
    subdomain = parsed_url["subdomain"]
    path = parsed_url["path"].lower()
    query = parsed_url["query"]
    scheme = parsed_url["scheme"]
    port = parsed_url["port"]
    registered_domain = parsed_url["registered_domain"]

    subdomain_count = len([part for part in subdomain.split(".") if part])
    suspicious_words = sorted(
        word for word in SUSPICIOUS_PATH_WORDS if re.search(rf"\b{re.escape(word)}\b", path)
    )
    query_params = [part for part in query.split("&") if part]

    checks = {
        "url_length": len(parsed_url["original_input"]),
        "suspicious_path_words": suspicious_words,
        "query_parameter_count": len(query_params),
        "too_many_subdomains": subdomain_count >= 3,
        "subdomain_count": subdomain_count,
        "is_ip_address": is_ip_address(host),
        "is_url_shortener": registered_domain in URL_SHORTENERS,
        "is_punycode": parsed_url["is_punycode"],
        "is_unicode_spoof": parsed_url["is_unicode_spoof"],
        "uses_http": scheme == "http",
        "uses_https": scheme == "https",
        "missing_scheme": not scheme,
        "suspicious_port": port is not None and port not in SAFE_PORTS,
        "port": port,
    }

    reasons = []
    score = 0

    if checks["url_length"] > 120:
        score += 8
        reasons.append(f"Very long URL ({checks['url_length']} characters)")
    elif checks["url_length"] > 75:
        score += 4
        reasons.append(f"Long URL ({checks['url_length']} characters)")

    if suspicious_words:
        score += min(18, 6 * len(suspicious_words))
        reasons.append("Suspicious path words found: " + ", ".join(suspicious_words))

    if checks["query_parameter_count"] >= 4:
        score += 8
        reasons.append(f"Many query parameters ({checks['query_parameter_count']})")
    elif checks["query_parameter_count"]:
        score += 3
        reasons.append(f"URL contains query parameters ({checks['query_parameter_count']})")

    if checks["too_many_subdomains"]:
        score += 10
        reasons.append(f"Too many subdomains ({subdomain_count})")

    if checks["is_ip_address"]:
        score += 20
        reasons.append("URL uses an IP address instead of a domain name")

    if checks["is_url_shortener"]:
        score += 12
        reasons.append("URL uses a known URL shortener")

    if checks["is_punycode"] or checks["is_unicode_spoof"]:
        score += 18
        reasons.append("Punycode or Unicode hostname may be hiding a spoofed domain")

    if checks["uses_http"]:
        score += 6
        reasons.append("URL uses HTTP instead of HTTPS")

    if checks["suspicious_port"]:
        score += 10
        reasons.append(f"URL uses a non-standard port ({port})")

    return {
        "score": min(score, 40),
        "reasons": reasons,
        "checks": checks,
    }

# ======================================================
# BASE DIRECTORY
# ======================================================
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _cand in (
    os.path.join(BASE_DIR, "model", "domain_detector_xgb.pkl"),
    os.path.join(BASE_DIR, "dataset", "model", "domain_detector_xgb.pkl"),
):
    if os.path.isfile(_cand):
        MODEL_PATH = _cand
        break
else:
    raise FileNotFoundError(
        "domain_detector_xgb.pkl not found. Expected under model/ or dataset/model/."
    )
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "balanced_domains.csv")

# ======================================================
# LOAD MODEL + VECTORIZER
# ======================================================
print("🚀 Loading trained model...")
model, vectorizer = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# ======================================================
# LOAD TRUSTED BRANDS FROM DATASET
# ======================================================
import pickle

TRUSTED_BRANDS_PATH = os.path.join(BASE_DIR, "dataset", "model", "trusted_brands.pkl")

with open(TRUSTED_BRANDS_PATH, "rb") as f:
    TRUSTED_BRANDS = pickle.load(f)

# ======================================================
# CANONICAL BRAND WHITELIST
# ======================================================
CANONICAL_BRANDS = {
    "google", "youtube", "facebook", "instagram", "twitter", "x",
    "amazon", "apple", "microsoft", "netflix", "linkedin", "reddit",
    "wikipedia", "yahoo", "bing", "ebay", "paypal", "github",
    "stackoverflow", "twitch", "discord", "spotify", "zoom",
    "dropbox", "adobe", "salesforce", "shopify", "wordpress",
    "cloudflare", "godaddy", "namecheap", "digitalocean", "aws",
    "azure", "gmail", "outlook", "hotmail", "live", "icloud",
    "whatsapp", "telegram", "snapchat", "tiktok", "pinterest",
    "tumblr", "flickr", "vimeo", "dailymotion", "quora",
}

print(f"✅ Loaded {len(TRUSTED_BRANDS)} trusted brands\n")
print("🌐 Major-Level Domain Risk Detector (AI Explainability Enabled)\n")

# ======================================================
# HOMOGLYPH MAP  (fake char → real char)
# ======================================================
HOMOGLYPH_MAP = {
    '@': 'a',
    '0': 'o',
    '1': 'l',
    '3': 'e',
    '4': 'a',
    '5': 's',
    '6': 'b',
    '8': 'b',
    '$': 's',
    '!': 'i',
}

HOMOGLYPH_EXPLAIN = {
    '@': "'@' looks like the letter 'a'",
    '0': "'0' (zero) looks like the letter 'o'",
    '1': "'1' (one) looks like the letter 'l'",
    '3': "'3' looks like the letter 'e'",
    '4': "'4' looks like the letter 'a'",
    '5': "'5' looks like the letter 's'",
    '6': "'6' looks like the letter 'b'",
    '8': "'8' looks like the letter 'b'",
    '$': "'$' looks like the letter 's'",
    '!': "'!' looks like the letter 'i'",
}

def decode_homoglyphs(text: str) -> str:
    """Replace lookalike characters with their real letter equivalents."""
    result = ""
    for ch in text:
        result += HOMOGLYPH_MAP.get(ch, ch)
    result = result.replace("rn", "m").replace("vv", "w").replace("cl", "d")
    return result


# ======================================================
# FIND BEST CANONICAL BRAND MATCH
# ======================================================
def find_best_canonical_match(raw_base: str):
    decoded_base  = decode_homoglyphs(raw_base)
    stripped_base = re.sub(r'[^a-z0-9]', '', raw_base)

    best_score        = 0.0
    best_brand        = ""
    best_decoded_base = decoded_base

    for brand in CANONICAL_BRANDS:
        score_raw      = SequenceMatcher(None, raw_base,      brand).ratio()
        score_decoded  = SequenceMatcher(None, decoded_base,  brand).ratio()
        score_stripped = SequenceMatcher(None, stripped_base, brand).ratio()

        score, winning_base = max(
            (score_raw,      raw_base),
            (score_decoded,  decoded_base),
            (score_stripped, stripped_base),
            key=lambda x: x[0],
        )

        if score > best_score:
            best_score        = score
            best_brand        = brand
            best_decoded_base = winning_base

    return round(best_score, 4), best_brand, best_decoded_base


# ======================================================
# ADVANCED DETECTION FUNCTIONS
# ======================================================

def detect_illegal_char_trick(raw_base: str, brand: str) -> list:
    """
    Detect illegal/special characters used as letter disguises.
    e.g. payp@l → '@' is used to look like 'a' → resembles 'paypal'
    """
    alerts = []
    if not brand:
        return alerts

    tricks_found = []
    for ch in raw_base:
        if ch in HOMOGLYPH_MAP:
            tricks_found.append(HOMOGLYPH_EXPLAIN[ch])

    if not tricks_found:
        return alerts

    decoded      = decode_homoglyphs(raw_base)
    sim_decoded  = SequenceMatcher(None, decoded, brand).ratio()
    sim_raw      = SequenceMatcher(None, raw_base, brand).ratio()

    if (
        sim_decoded > sim_raw + 0.05
        or sim_decoded >= 0.75
        or decoded == brand
    ):
        trick_str = " | ".join(dict.fromkeys(tricks_found))
        alerts.append(
            f"Special character disguise: {trick_str} — "
            f"'{raw_base}' is disguised to look like '{brand}'"
        )

    return alerts


def detect_character_swap(base: str, brand: str) -> list:
    """Two neighbouring letters swapped — e.g. googel vs google."""
    alerts = []
    if not brand or len(base) != len(brand):
        return alerts
    diffs = [(i, base[i], brand[i]) for i in range(len(base)) if base[i] != brand[i]]
    if len(diffs) == 2:
        i1, b1, br1 = diffs[0]
        i2, b2, br2 = diffs[1]
        if b1 == br2 and b2 == br1 and i2 == i1 + 1:
            alerts.append(
                f"Letter-swap trick: '{base[i1]}' and '{base[i2]}' are swapped — "
                f"'{base}' should read '{brand}' "
                f"(swapped letters are very easy to miss when reading quickly)"
            )
    return alerts


def detect_repeated_chars(base: str, brand: str) -> list:
    """Extra duplicate letter — e.g. gooogle, amazzon."""
    alerts = []
    if not brand:
        return alerts

    def collapse(s):
        if not s:
            return s
        r = s[0]
        for c in s[1:]:
            if c != r[-1]:
                r += c
        return r

    if collapse(base) == collapse(brand) and base != brand:
        extra = ""
        for i in range(1, len(base)):
            if base[i] == base[i - 1]:
                extra = base[i]
                break
        if extra:
            alerts.append(
                f"Extra duplicate letter: the letter '{extra}' appears twice in "
                f"'{base}' but only once in '{brand}' "
                f"— the extra copy is easy to overlook"
            )
        else:
            alerts.append(
                f"Repeated character inserted: '{base}' is '{brand}' "
                f"with an extra duplicate letter added"
            )
    return alerts


def detect_missing_char(base: str, brand: str) -> list:
    """One letter quietly removed — e.g. amzon vs amazon."""
    alerts = []
    if not brand or abs(len(base) - len(brand)) != 1:
        return alerts
    if len(base) >= len(brand):
        return alerts

    shorter, longer = base, brand
    for i in range(len(longer)):
        candidate = longer[:i] + longer[i + 1:]
        if candidate == shorter:
            missing_letter = longer[i]
            alerts.append(
                f"Missing letter: the letter '{missing_letter}' has been "
                f"removed from '{brand}' to make '{base}' "
                f"— looks almost identical when scanned quickly"
            )
            break
    return alerts


def detect_extra_char(base: str, brand: str) -> list:
    """One extra letter slipped in — e.g. amazoon, paypaal."""
    alerts = []
    if not brand or abs(len(base) - len(brand)) != 1:
        return alerts
    if len(base) <= len(brand):
        return alerts

    longer, shorter = base, brand
    for i in range(len(longer)):
        if longer[:i] + longer[i + 1:] == shorter:
            extra_letter = longer[i]
            alerts.append(
                f"Extra letter inserted: '{extra_letter}' has been slipped into "
                f"'{brand}' to make '{base}' "
                f"— the extra letter is easy to miss at a glance"
            )
            break
    return alerts


def detect_homoglyph_advanced(base: str, brand: str) -> list:
    """Multi-character visual lookalikes: rn→m, vv→w, cl→d."""
    alerts = []
    if not brand:
        return alerts

    decoded = decode_homoglyphs(base)

    if decoded == brand and base != brand:
        subs = [
            HOMOGLYPH_EXPLAIN.get(o, f"'{o}' → '{d}'")
            for o, d in zip(base, decoded)
            if o != d
        ]
        if subs:
            alerts.append(
                f"Visual character disguise: {' | '.join(subs)} — "
                f"'{base}' is made to look like '{brand}'"
            )

    for fake_seq, real_seq, explanation in [
        ('rn', 'm', "the two letters 'rn' placed together look like 'm'"),
        ('vv', 'w', "two 'v' letters placed together look like 'w'"),
        ('cl', 'd', "the letters 'cl' placed together look like 'd'"),
    ]:
        if fake_seq in base and base.replace(fake_seq, real_seq) == brand:
            alerts.append(
                f"Two-letter lookalike trick: {explanation} — "
                f"'{base}' uses '{fake_seq}' to impersonate '{brand}'"
            )

    return alerts


def detect_subdomain_abuse(full_domain: str) -> list:
    """Brand name used as subdomain — e.g. paypal.evil.com."""
    alerts = []
    ext = tldextract.extract(full_domain)
    if ext.subdomain:
        prefix = ext.subdomain.split(".")[0]
        if prefix in CANONICAL_BRANDS and ext.domain != prefix:
            alerts.append(
                f"Subdomain trick: '{prefix}' is a real brand placed as a subdomain — "
                f"'{full_domain}' is NOT the official '{prefix}' website. "
                f"The real site is www.{prefix}.com"
            )
    return alerts


def detect_keyword_combo(base: str) -> list:
    """Brand + phishing keyword — e.g. paypal-secure, amazon-login."""
    alerts = []
    phishing_keywords = [
        "secure", "login", "signin", "account", "update", "verify",
        "support", "help", "service", "online", "web", "mail",
        "bank", "pay", "wallet", "shop", "store", "official",
        "confirm", "billing", "password", "reset", "auth", "portal",
        "alert", "notice", "access", "manage", "customer",
    ]
    for brand in CANONICAL_BRANDS:
        if brand in base and base != brand:
            remainder = base.replace(brand, "").strip("-_")
            for kw in phishing_keywords:
                if kw in remainder:
                    alerts.append(
                        f"Phishing keyword combo: brand '{brand}' + urgency word '{kw}' — "
                        f"scammers add words like '{kw}' next to a brand name to make "
                        f"fake sites look like official login or support pages"
                    )
                    return alerts
    return alerts


def run_all_advanced_checks(raw_base: str, brand: str, full_domain: str) -> list:
    """Run every check and return all plain-English alerts."""
    decoded_base = decode_homoglyphs(raw_base)
    alerts = []
    if brand:
        alerts += detect_illegal_char_trick(raw_base, brand)
        alerts += detect_character_swap(decoded_base, brand)
        alerts += detect_repeated_chars(decoded_base, brand)
        alerts += detect_missing_char(decoded_base, brand)
        alerts += detect_extra_char(decoded_base, brand)
        alerts += detect_homoglyph_advanced(raw_base, brand)
    alerts += detect_subdomain_abuse(full_domain)
    alerts += detect_keyword_combo(raw_base)
    return alerts


# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_numeric_features(domain: str):
    base = strip_tld(domain)
    if base:
        entropy = -sum(
            (base.count(c) / len(base)) * math.log2(base.count(c) / len(base))
            for c in set(base)
        )
    else:
        entropy = 0.0

    features = {
        "length":            len(base),
        "entropy":           entropy,
        "num_digits":        sum(c.isdigit() for c in base),
        "num_special_chars": sum(not c.isalnum() for c in base),
        "num_dots":          domain.count("."),
    }
    return pd.DataFrame([features])[[
        "length", "entropy", "num_digits", "num_special_chars", "num_dots"
    ]]


def compute_final_risk(
    prob_fake, sim_score, suspicious_count, exists, advanced_alert_count
):
    risk = (
          prob_fake            * 50
        + sim_score            * 25
        + suspicious_count     *  4
        + advanced_alert_count *  6
    )
    if not exists:
        risk += 20
    return min(round(risk, 2), 100)


def risk_label(score: float) -> str:
    if score >= 80:
        return "🔴 CRITICAL"
    if score >= 60:
        return "🟠 HIGH"
    if score >= 40:
        return "🟡 MEDIUM"
    if score >= 20:
        return "🟡 LOW"
    return "🟢 NONE"


def analyze_domain(full_domain: str) -> dict:
    parsed_url = normalize_url_input(full_domain)
    host = parsed_url["host"]
    analysis_domain = parsed_url["registered_domain"] or host
    url_analysis = build_url_analysis(parsed_url)

    if not host:
        return {
            "input": parsed_url["original_input"],
            "domain": "",
            "host": "",
            "risk_score": 100.0,
            "risk_level": risk_label(100),
            "exists": False,
            "is_fake": True,
            "ml_prob": "0.0%",
            "ml_probability": 0.0,
            "brand": "",
            "reasons": ["Could not extract a valid hostname from the URL"],
            "url_analysis": url_analysis,
        }

    host_is_ip = is_ip_address(host)
    invalid_format = not host_is_ip and not is_valid_domain(analysis_domain)
    ext_parts = tldextract.extract(analysis_domain)
    raw_base = ext_parts.domain or analysis_domain.rsplit(".", 1)[0]
    ext = ext_parts.suffix or (analysis_domain.split(".")[-1] if "." in analysis_domain else "")
    normalized_base = normalize_domain(analysis_domain)

    exists = domain_exists(host)
    if host_is_ip:
        sim_score, matched_brand = 0.0, ""
        advanced_pre = []
    else:
        sim_score, matched_brand, _effective_base = find_best_canonical_match(raw_base)
        advanced_pre = run_all_advanced_checks(raw_base, matched_brand, host)

    if (
        raw_base in CANONICAL_BRANDS
        and normalized_base == raw_base
        and ext in {"com", "net", "org"}
        and exists
        and not advanced_pre
        and not invalid_format
        and not url_analysis["reasons"]
    ):
        return {
            "input": parsed_url["original_input"],
            "domain": analysis_domain,
            "host": host,
            "risk_score": 0.0,
            "risk_level": risk_label(0),
            "exists": exists,
            "is_fake": False,
            "ml_prob": "0.0%",
            "ml_probability": 0.0,
            "brand": matched_brand,
            "reasons": ["Exact trusted brand + valid TLD + DNS verified"],
            "url_analysis": url_analysis,
        }

    visual_subs = [] if raw_base == matched_brand else detect_visual_substitutions(analysis_domain, matched_brand)
    edit_alerts = [] if raw_base == matched_brand else detect_edit_distance_typos(analysis_domain, matched_brand)
    tld_alerts = detect_tld_typos(analysis_domain)
    advanced_alerts = [] if host_is_ip else run_all_advanced_checks(raw_base, matched_brand, host)

    if host_is_ip:
        prob_fake = 0.0
    else:
        numeric_df = extract_numeric_features(analysis_domain)
        numeric_sparse = csr_matrix(numeric_df.values)
        tfidf_sparse = vectorizer.transform([analysis_domain])
        X = hstack([numeric_sparse, tfidf_sparse])
        prob_fake = float(model.predict_proba(X)[0][1])

    if raw_base in CANONICAL_BRANDS and exists and not invalid_format and not advanced_alerts:
        prob_fake = min(prob_fake, 0.10)

    suspicious_feats = extract_suspicious_features(analysis_domain)
    suspicious_count = sum(suspicious_feats.values())
    risk_score = compute_final_risk(
        prob_fake, sim_score, suspicious_count, exists, len(advanced_alerts)
    )
    risk_score = min(round(risk_score + url_analysis["score"], 2), 100)

    is_fake = False
    fake_reasons = []

    if prob_fake >= 0.75:
        is_fake = True
        fake_reasons.append(
            f"AI model is {prob_fake * 100:.1f}% confident this is a fake domain"
        )

    if sim_score > 0.75 and (visual_subs or edit_alerts):
        is_fake = True
        fake_reasons.append(
            f"Very high resemblance to '{matched_brand}' combined with character manipulation"
        )

    if advanced_alerts:
        is_fake = True
        fake_reasons.append(
            "Character-level manipulation detected (trick characters / letter swap / insertion / homoglyph)"
        )
        fake_reasons.extend(advanced_alerts)

    if sim_score > 0.60 and (
        visual_subs or edit_alerts or advanced_alerts or suspicious_count >= 1
    ):
        is_fake = True
        fake_reasons.append(
            f"Looks similar to '{matched_brand}' ({sim_score * 100:.0f}% match) with additional suspicious patterns"
        )

    if suspicious_count >= 2 and (risk_score >= 55 or sim_score > 0.60):
        is_fake = True
        fake_reasons.append(f"{suspicious_count} independent suspicious features found")

    if not exists and risk_score >= 55:
        is_fake = True
        fake_reasons.append("Domain does not exist in DNS and overall risk is elevated")

    if invalid_format and sim_score > 0.5:
        is_fake = True
        fake_reasons.append(
            "Domain uses illegal/special characters to impersonate a real brand"
        )

    if tld_alerts and sim_score > 0.60:
        is_fake = True
        fake_reasons.append("Suspicious / unusual TLD combined with brand similarity")

    if url_analysis["score"] >= 12:
        is_fake = True

    fake_reasons.extend(url_analysis["reasons"])

    return {
        "input": parsed_url["original_input"],
        "domain": analysis_domain,
        "host": host,
        "display_host": parsed_url["display_host"],
        "scheme": parsed_url["scheme"],
        "path": parsed_url["path"],
        "query": parsed_url["query"],
        "risk_score": risk_score,
        "risk_level": risk_label(risk_score),
        "exists": exists,
        "is_fake": is_fake,
        "ml_prob": f"{prob_fake * 100:.1f}%",
        "ml_probability": prob_fake,
        "brand": matched_brand if sim_score >= 0.5 else "",
        "reasons": list(dict.fromkeys(fake_reasons)),
        "url_analysis": url_analysis,
    }


# ======================================================
# MAIN LOOP
# ======================================================
if __name__ == "__main__":
    while True:
        try:
            full_domain = input("🌐 Enter a domain (or 'exit'): ").strip().lower()

            if full_domain == "exit":
                print("👋 Exiting...")
                break

            print()

            # ── FORMAT VALIDATION ──────────────────────────────
            invalid_format = not is_valid_domain(full_domain)

            # ── FIX: derive raw_base directly via rsplit on the last dot.
            # strip_tld() may treat '@' as an email separator and return only
            # the part AFTER '@' — e.g. "payp@l.com" → "l" instead of "payp@l".
            # Using rsplit(".", 1) always gives us the full base before the TLD.
            _tld_parts      = full_domain.rsplit(".", 1)
            raw_base        = _tld_parts[0] if len(_tld_parts) == 2 else full_domain
            ext             = full_domain.split(".")[-1] if "." in full_domain else ""
            normalized_base = normalize_domain(full_domain)

            # ── DNS CHECK ──────────────────────────────────────
            exists = domain_exists(full_domain)

            # ── BRAND SIMILARITY ───────────────────────────────
            sim_score, matched_brand, effective_base = find_best_canonical_match(raw_base)

            # ── ADVANCED CHECKS (needed for safe-exit guard) ───
            advanced_pre = run_all_advanced_checks(raw_base, matched_brand, full_domain)

            # ── STRICT SAFE CHECK ──────────────────────────────
            if (
                raw_base in CANONICAL_BRANDS
                and normalized_base == raw_base
                and ext in {"com", "net", "org"}
                and exists
                and not advanced_pre
                and not invalid_format
            ):
                print(f"✅  SAFE DOMAIN: {full_domain}")
                print("🟢 Reason : Exact trusted brand + valid TLD + DNS verified")
                print("💯 Risk Score: 0 / 100  (🟢 NONE)\n")
                continue

            # ── RULE-BASED DETECTION ───────────────────────────
            visual_subs     = detect_visual_substitutions(full_domain, matched_brand)
            edit_alerts     = detect_edit_distance_typos(full_domain, matched_brand)
            tld_alerts      = detect_tld_typos(full_domain)
            advanced_alerts = run_all_advanced_checks(raw_base, matched_brand, full_domain)

            # ── ML PREDICTION ──────────────────────────────────
            numeric_df     = extract_numeric_features(full_domain)
            numeric_sparse = csr_matrix(numeric_df.values)
            tfidf_sparse   = vectorizer.transform([full_domain])
            X = hstack([numeric_sparse, tfidf_sparse])
            prob_fake = float(model.predict_proba(X)[0][1])

            # ── SUSPICIOUS FEATURES ────────────────────────────
            suspicious_feats = extract_suspicious_features(full_domain)
            suspicious_count = sum(suspicious_feats.values())

            # ── RISK SCORE ─────────────────────────────────────
            risk_score = compute_final_risk(
                prob_fake, sim_score, suspicious_count, exists, len(advanced_alerts)
            )

            # ── FINAL DECISION ─────────────────────────────────
            is_fake      = False
            fake_reasons = []

            if prob_fake >= 0.75:
                is_fake = True
                fake_reasons.append(
                    f"AI model is {prob_fake * 100:.1f}% confident this is a fake domain"
                )

            if sim_score > 0.75 and (visual_subs or edit_alerts):
                is_fake = True
                fake_reasons.append(
                    f"Very high resemblance to '{matched_brand}' "
                    f"combined with character manipulation"
                )

            if advanced_alerts:
                is_fake = True
                fake_reasons.append(
                    "Character-level manipulation detected "
                    "(trick characters / letter swap / insertion / homoglyph)"
                )

            if sim_score > 0.60 and (
                visual_subs or edit_alerts or advanced_alerts or suspicious_count >= 1
            ):
                is_fake = True
                fake_reasons.append(
                    f"Looks similar to '{matched_brand}' ({sim_score * 100:.0f}% match) "
                    f"with additional suspicious patterns"
                )

            if suspicious_count >= 2:
                is_fake = True
                fake_reasons.append(
                    f"{suspicious_count} independent suspicious features found"
                )

            if not exists and risk_score >= 55:
                is_fake = True
                fake_reasons.append(
                    "Domain does not exist in DNS and overall risk is elevated"
                )

            if invalid_format and sim_score > 0.5:
                is_fake = True
                fake_reasons.append(
                    "Domain uses illegal/special characters to impersonate a real brand"
                )

            if tld_alerts and sim_score > 0.60:
                is_fake = True
                fake_reasons.append(
                    "Suspicious / unusual TLD combined with brand similarity"
                )

            fake_reasons = list(dict.fromkeys(fake_reasons))

            # ── PRINT OUTPUT ───────────────────────────────────
            sep = "=" * 60
            print(sep)
            print(f"  Domain Analysed : {full_domain}")
            print(f"  Risk Score      : {risk_score} / 100  ({risk_label(risk_score)})")
            print(f"  DNS Status      : {'✅ Resolves OK' if exists else '❌ Does NOT exist'}")
            print(f"  Format Valid    : {'✅ Yes' if not invalid_format else '⚠️  No — illegal characters present'}")
            print(sep)

            if is_fake:
                print("  ⛔  VERDICT: FAKE / PHISHING DOMAIN\n")

                print("  📋 WHY THIS DOMAIN IS SUSPICIOUS:")
                for i, reason in enumerate(fake_reasons, 1):
                    print(f"     {i}. {reason}")

                # ── Brand impersonation block ────────────────────
                # FIX: use a lower threshold (0.3) for domains with illegal chars
                # because even after decoding, the raw similarity was artificially
                # depressed by the special character before the fix. At sim=1.0
                # (payp@l decoded = paypal) this now always fires correctly.
                _sim_threshold = 0.3 if invalid_format else 0.5
                if matched_brand and sim_score >= _sim_threshold:
                    print()
                    print("  🎯 BRAND IMPERSONATION EXPLAINED:")
                    print(f"     ├─ Real website  : www.{matched_brand}.com")
                    print(f"     ├─ Fake domain   : {full_domain}")
                    print(f"     ├─ Similarity    : {sim_score * 100:.0f}%")
                    print(f"     └─ What this means: This domain is crafted to look")
                    print(f"        like '{matched_brand}' so victims visit it by")
                    print(f"        mistake instead of the real site.")

                # ── Trick explanation — covers ALL manipulation types ──
                decoded      = decode_homoglyphs(raw_base)
                decoded_base = decode_homoglyphs(raw_base)   # for structural checks
                trick_lines  = []

                # 1. Two-letter visual lookalikes FIRST (rn→m, vv→w, cl→d)
                # Must run before char-by-char diff so we can suppress the diff
                # when the lookalike already fully explains the domain
                # (e.g. rnicrosoft → 'rn' looks like 'm' → microsoft).
                _multichar_explains_all = False
                for fake_seq, real_seq in [('rn', 'm'), ('vv', 'w'), ('cl', 'd')]:
                    if fake_seq in raw_base and raw_base.replace(fake_seq, real_seq) == matched_brand:
                        trick_lines.append(
                            f"     • Two letters '{fake_seq}' placed together look "
                            f"like one letter '{real_seq}'\n"
                            f"       → The attacker used '{fake_seq}' to replace "
                            f"'{real_seq}' in '{matched_brand}' — this is the only "
                            f"change that makes '{raw_base}' look like '{matched_brand}'"
                        )
                        _multichar_explains_all = True

                # 2. Special / homoglyph single characters (e.g. @ → a)
                # Skip this when a multi-char lookalike already explains the domain,
                # because zip() would pair wrong chars and produce nonsense output.
                if not _multichar_explains_all:
                    for orig_ch, real_ch in zip(raw_base, decoded):
                        if orig_ch != real_ch:
                            explain = HOMOGLYPH_EXPLAIN.get(
                                orig_ch, f"'{orig_ch}' looks like '{real_ch}'"
                            )
                            trick_lines.append(
                                f"     • {explain}\n"
                                f"       → The attacker wrote '{orig_ch}' instead of "
                                f"'{real_ch}' in the domain name"
                            )

                # 3. Letter swap  (e.g. googel → google)
                # Skip structural checks entirely when rn/vv/cl already explains all
                if not _multichar_explains_all and matched_brand:
                    # 3. Letter swap  (e.g. googel → google)
                    if len(decoded_base) == len(matched_brand):
                        diffs = [
                            (i, decoded_base[i], matched_brand[i])
                            for i in range(len(decoded_base))
                            if decoded_base[i] != matched_brand[i]
                        ]
                        if len(diffs) == 2:
                            i1, b1, br1 = diffs[0]
                            i2, b2, br2 = diffs[1]
                            if b1 == br2 and b2 == br1 and i2 == i1 + 1:
                                trick_lines.append(
                                    f"     • Letters '{b1}' and '{b2}' are swapped at "
                                    f"position {i1+1}-{i2+1}\n"
                                    f"       → '{decoded_base}' should read '{matched_brand}' "
                                    f"— swapped letters are very easy to miss when reading quickly"
                                )

                    # 4. Missing letter  (e.g. amzon → amazon)
                    if len(decoded_base) == len(matched_brand) - 1:
                        for i in range(len(matched_brand)):
                            if matched_brand[:i] + matched_brand[i+1:] == decoded_base:
                                trick_lines.append(
                                    f"     • The letter '{matched_brand[i]}' at position "
                                    f"{i+1} has been quietly removed\n"
                                    f"       → '{decoded_base}' is '{matched_brand}' with "
                                    f"one letter dropped — looks almost identical at a glance"
                                )
                                break

                    # 5. Extra letter inserted  (e.g. amazoon → amazon)
                    if len(decoded_base) == len(matched_brand) + 1:
                        for i in range(len(decoded_base)):
                            if decoded_base[:i] + decoded_base[i+1:] == matched_brand:
                                trick_lines.append(
                                    f"     • An extra letter '{decoded_base[i]}' has been "
                                    f"slipped in at position {i+1}\n"
                                    f"       → '{decoded_base}' is '{matched_brand}' with "
                                    f"one letter added — easy to miss at a glance"
                                )
                                break

                    # 6. Repeated / doubled letter  (e.g. gooogle → google)
                    # Only fire if extra-letter check didn't already explain this.
                    _extra_already_found = any("slipped in" in l for l in trick_lines)
                    if not _extra_already_found:
                        def _collapse(s):
                            if not s: return s
                            r = s[0]
                            for c in s[1:]:
                                if c != r[-1]: r += c
                            return r
                        if _collapse(decoded_base) == _collapse(matched_brand) and decoded_base != matched_brand:
                            for i in range(1, len(decoded_base)):
                                if decoded_base[i] == decoded_base[i-1]:
                                    trick_lines.append(
                                        f"     • The letter '{decoded_base[i]}' is doubled "
                                        f"at position {i}-{i+1}\n"
                                        f"       → '{decoded_base}' is '{matched_brand}' with "
                                        f"an extra duplicate letter — easy to overlook"
                                    )
                                    break

                if trick_lines:
                    print()
                    print("  ℹ️  TRICK USED — How This Domain Fools You:")
                    for line in trick_lines:
                        print(line)
                    if decoded != raw_base:
                        print(f"     → Final result: '{raw_base}' visually looks like "
                              f"'{decoded}' (= {matched_brand})")



                # ── Other suspicious signals ─────────────────────
                if suspicious_feats:
                    triggered = {k: v for k, v in suspicious_feats.items() if v > 0}
                    if triggered:
                        print()
                        print("  🚩 OTHER SUSPICIOUS SIGNALS:")
                        for k in triggered:
                            print(f"     • {k}")

                # ── Scores ───────────────────────────────────────
                print()
                print(f"  📊 AI Fake Probability : {prob_fake * 100:.1f}%")
                print(f"  📐 Brand Similarity    : {sim_score * 100:.1f}%")

            else:
                print("  ✅  VERDICT: DOMAIN APPEARS SAFE\n")
                print(f"  📐 Brand Similarity : {sim_score * 100:.1f}%")
                print("  🟢 No impersonation or manipulation patterns detected")

            print(sep + "\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")