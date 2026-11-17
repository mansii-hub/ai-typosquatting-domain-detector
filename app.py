import streamlit as st
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter
import math

from scripts.check_domain import analyze_domain

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
/* ── base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #080e1a;
    color: #e2e8f0;
}

/* ── tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0f172a;
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 22px;
    color: #94a3b8;
    font-weight: 500;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #38bdf8 !important;
}

/* ── input ── */
.stTextInput > div > div > input {
    background: #0f172a !important;
    border: 1.5px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
    padding: 14px 16px !important;
    transition: border 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}

/* ── analyze button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 40px !important;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── verdict banner ── */
.verdict-safe {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border-left: 5px solid #10b981;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 18px;
}
.verdict-moderate {
    background: linear-gradient(135deg, #451a03, #78350f);
    border-left: 5px solid #f59e0b;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 18px;
}
.verdict-danger {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border-left: 5px solid #ef4444;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 18px;
}
.verdict-title { font-size: 1.8rem; font-weight: 700; margin: 0 0 4px 0; }
.verdict-sub   { font-size: 1rem; opacity: 0.75; margin: 0; }

/* ── info cards ── */
.card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.card-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 4px;
}
.card-value { font-size: 1.05rem; font-weight: 600; color: #f1f5f9; }

/* ── check-row badges ── */
.check-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 11px 16px;
    margin-bottom: 8px;
}
.check-label { font-size: 0.9rem; color: #cbd5e1; }
.badge-ok   { background:#064e3b; color:#34d399; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
.badge-warn { background:#451a03; color:#fbbf24; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:600; }

/* ── reason pills ── */
.reason-pill {
    background: #1e1025;
    border: 1px solid #7c3aed55;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #c4b5fd;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}

/* ── stat card ── */
.stat-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.stat-card .num { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.stat-card .lbl { font-size: 0.8rem; color: #64748b; margin-top: 4px; }

/* ── dashboard metric ── */
.dash-metric {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
}
.dash-metric .dnum { font-size: 2.6rem; font-weight: 700; margin: 8px 0 4px; }
.dash-metric .dlbl { font-size: 0.85rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase; }
.dash-metric .dico { font-size: 2rem; }

/* ── section header ── */
.sec-header {
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 24px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def extract_numeric_features(domain):
    if not domain:
        return [0, 0, 0, 0, 0]
    return [
        len(domain),
        round(entropy(domain), 2),
        sum(c.isdigit() for c in domain),
        sum(not c.isalnum() for c in domain),
        domain.count("."),
    ]


def score_color(score):
    if score < 30:
        return "#10b981"
    if score < 60:
        return "#f59e0b"
    return "#ef4444"


def risk_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    theta = np.linspace(0, np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#1e293b", linewidth=14, solid_capstyle="round")
    fill_end = int(score / 100 * 300)
    color = score_color(score)
    ax.plot(np.cos(theta[:fill_end]), np.sin(theta[:fill_end]), color=color, linewidth=14, solid_capstyle="round")
    ax.text(0, -0.12, f"{score:.0f}", ha="center", va="center", fontsize=28, fontweight="bold",
            color=color, transform=ax.transData)
    ax.text(0, -0.52, "/ 100 Risk Score", ha="center", va="center", fontsize=10,
            color="#64748b", transform=ax.transData)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.8, 1.1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def domain_feature_chart(domain):
    names = ["Length", "Entropy", "Digits", "Special\nChars", "Dots"]
    vals = extract_numeric_features(domain)
    colors = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171"]
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    bars = ax.bar(names, vals, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(round(val, 1)), ha="center", va="bottom", fontsize=9, color="#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title("Domain Feature Breakdown", color="#94a3b8", fontsize=9, pad=8)
    fig.tight_layout(pad=0.5)
    return fig


def check_badge(label, triggered, triggered_text="Detected", ok_text="Clear"):
    val_html = (
        f"<span class='badge-warn'>⚠ {triggered_text}</span>"
        if triggered
        else f"<span class='badge-ok'>✓ {ok_text}</span>"
    )
    st.markdown(
        f"<div class='check-row'><span class='check-label'>{label}</span>{val_html}</div>",
        unsafe_allow_html=True,
    )


tabs = st.tabs(["  🏠  Dashboard  ", "  🔍  Scanner  ", "  📊  Performance  "])

# ════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin-bottom:32px;'>
        <div style='font-size:3rem;'>🛡️</div>
        <h1 style='font-size:2.2rem; font-weight:700; margin:8px 0 6px;'>PhishGuard AI</h1>
        <p style='color:#64748b; font-size:1.05rem;'>
            Real-time phishing, typosquatting &amp; URL threat detection powered by machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("🎯", "96%",  "Model Accuracy",  "#10b981"),
        ("🔬", "95%",  "Precision",        "#38bdf8"),
        ("📡", "94%",  "Recall",           "#818cf8"),
        ("⚡", "95%",  "F1 Score",         "#f59e0b"),
    ]
    for col, (icon, num, lbl, clr) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div class='dash-metric'>
            <div class='dico'>{icon}</div>
            <div class='dnum' style='color:{clr}'>{num}</div>
            <div class='dlbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-header'>What This System Detects</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    features = [
        ("🔤", "Typosquatting", "Detects domains with swapped, missing, or doubled letters designed to look like trusted brands."),
        ("👁️", "Visual Spoofing", "Catches homoglyphs and special characters like '0→o', '1→l', '@→a' used to fool the eye."),
        ("🌐", "URL Analysis", "Inspects full URLs for suspicious paths, query params, subdomains, HTTP vs HTTPS, and more."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3], features):
        col.markdown(f"""
        <div class='card' style='height:160px;'>
            <div style='font-size:1.6rem; margin-bottom:8px;'>{icon}</div>
            <div style='font-weight:600; font-size:0.95rem; margin-bottom:6px; color:#f1f5f9;'>{title}</div>
            <div style='font-size:0.82rem; color:#94a3b8; line-height:1.5;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    f4, f5, f6 = st.columns(3)
    features2 = [
        ("🤖", "AI Model (XGBoost)", "ML model trained on 800K+ domains gives a probability score for each domain being fake."),
        ("🏷️", "Brand Impersonation", "Compares domains against 50+ canonical brands and flags similarity-based attacks."),
        ("📡", "DNS Verification", "Checks whether the domain actually resolves in DNS — a common signal of phishing sites."),
    ]
    for col, (icon, title, desc) in zip([f4, f5, f6], features2):
        col.markdown(f"""
        <div class='card' style='height:160px;'>
            <div style='font-size:1.6rem; margin-bottom:8px;'>{icon}</div>
            <div style='font-weight:600; font-size:0.95rem; margin-bottom:6px; color:#f1f5f9;'>{title}</div>
            <div style='font-size:0.82rem; color:#94a3b8; line-height:1.5;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# SCANNER
# ════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin-bottom:28px;'>
        <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:6px;'>🔍 URL & Domain Scanner</h2>
        <p style='color:#64748b;'>Paste any domain or full URL — the AI engine will analyse it instantly</p>
    </div>
    """, unsafe_allow_html=True)

    input_col, _ = st.columns([3, 1])
    with input_col:
        domain_input = st.text_input(
            "",
            placeholder="https://www.example.com/login?next=account",
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("⚡ Analyze Now", use_container_width=True)

    if analyze_clicked:
        if not domain_input.strip():
            st.warning("Please enter a domain or URL to scan.")
        else:
            with st.spinner("Scanning with AI engine …"):
                time.sleep(0.6)
                result = analyze_domain(domain_input)

            score      = result["risk_score"]
            exists     = result["exists"]
            is_fake    = result["is_fake"]
            ml_prob    = result["ml_prob"]
            ml_pct     = result["ml_probability"] * 100
            reasons    = result["reasons"]
            brand      = result["brand"]
            url_info   = result.get("url_analysis", {})
            checks     = url_info.get("checks", {})

            # ── VERDICT BANNER ───────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if is_fake:
                verdict_cls   = "verdict-danger"
                verdict_icon  = "🔴"
                verdict_title = "Fake / Malicious"
                verdict_sub   = "This URL shows strong signs of phishing or brand impersonation."
            elif score < 30:
                verdict_cls   = "verdict-safe"
                verdict_icon  = "🟢"
                verdict_title = "Safe"
                verdict_sub   = "No significant threat signals detected."
            else:
                verdict_cls   = "verdict-moderate"
                verdict_icon  = "🟡"
                verdict_title = "Moderate Risk"
                verdict_sub   = "Some suspicious signals found — proceed with caution."

            st.markdown(f"""
            <div class='{verdict_cls}'>
                <p class='verdict-title'>{verdict_icon} {verdict_title}</p>
                <p class='verdict-sub'>{verdict_sub}</p>
            </div>""", unsafe_allow_html=True)

            # ── TOP ROW: gauge + meta ─────────────────────────────
            g_col, m_col = st.columns([1, 2])

            with g_col:
                st.pyplot(risk_gauge(score), use_container_width=True)

            with m_col:
                st.markdown("<div class='sec-header'>Target Information</div>", unsafe_allow_html=True)
                r1, r2, r3 = st.columns(3)
                r1.markdown(f"<div class='card'><div class='card-label'>Registered Domain</div><div class='card-value'>{result.get('domain', '—')}</div></div>", unsafe_allow_html=True)
                r2.markdown(f"<div class='card'><div class='card-label'>Full Host</div><div class='card-value'>{result.get('host', '—')}</div></div>", unsafe_allow_html=True)
                r3.markdown(f"<div class='card'><div class='card-label'>Scheme</div><div class='card-value'>{(result.get('scheme') or 'not provided').upper()}</div></div>", unsafe_allow_html=True)

                d1, d2, d3 = st.columns(3)
                dns_val = "✅ Resolves" if exists else "❌ No DNS"
                dns_clr = "#10b981" if exists else "#ef4444"
                d1.markdown(f"<div class='card'><div class='card-label'>DNS Status</div><div class='card-value' style='color:{dns_clr}'>{dns_val}</div></div>", unsafe_allow_html=True)
                d2.markdown(f"<div class='card'><div class='card-label'>ML Fake Probability</div><div class='card-value' style='color:{score_color(ml_pct)}'>{ml_prob}</div></div>", unsafe_allow_html=True)
                brand_val = f"⚠ {brand.upper()}" if brand else "✓ None"
                brand_clr = "#f59e0b" if brand else "#10b981"
                d3.markdown(f"<div class='card'><div class='card-label'>Brand Impersonation</div><div class='card-value' style='color:{brand_clr}'>{brand_val}</div></div>", unsafe_allow_html=True)

            # ── MIDDLE ROW: URL checks + reasons ─────────────────
            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.markdown("<div class='sec-header'>URL Security Checks</div>", unsafe_allow_html=True)

                u1, u2, u3 = st.columns(3)
                u1.markdown(f"<div class='stat-card'><div class='num'>{checks.get('url_length', 0)}</div><div class='lbl'>URL Length</div></div>", unsafe_allow_html=True)
                u2.markdown(f"<div class='stat-card'><div class='num'>{checks.get('query_parameter_count', 0)}</div><div class='lbl'>Query Params</div></div>", unsafe_allow_html=True)
                u3.markdown(f"<div class='stat-card'><div class='num'>{checks.get('subdomain_count', 0)}</div><div class='lbl'>Subdomains</div></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                path_words = checks.get("suspicious_path_words", [])
                check_badge("Suspicious path words", bool(path_words),
                            triggered_text=", ".join(path_words))
                check_badge("Too many subdomains",   checks.get("too_many_subdomains", False))
                check_badge("IP address URL",         checks.get("is_ip_address", False))
                check_badge("URL shortener",          checks.get("is_url_shortener", False))
                check_badge("Punycode / Unicode spoof", checks.get("is_punycode") or checks.get("is_unicode_spoof", False))
                check_badge("HTTP (no encryption)",   checks.get("uses_http", False),
                            triggered_text="Yes — insecure")
                port = checks.get("port")
                check_badge("Suspicious port",        checks.get("suspicious_port", False),
                            triggered_text=f"Port {port}")

            with right_col:
                st.markdown("<div class='sec-header'>Detection Reasons</div>", unsafe_allow_html=True)
                if reasons:
                    for r in reasons:
                        st.markdown(f"<div class='reason-pill'>⚠️ &nbsp;{r}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='reason-pill' style='color:#34d399; border-color:#065f4655;'>✅ &nbsp;No suspicious signals detected.</div>", unsafe_allow_html=True)

            # ── BOTTOM ROW: domain feature chart ─────────────────
            st.markdown("<div class='sec-header'>Domain Feature Analysis</div>", unsafe_allow_html=True)
            ch_col, _ = st.columns([2, 1])
            with ch_col:
                st.pyplot(domain_feature_chart(result.get("domain", "")), use_container_width=True)

# ════════════════════════════════════════════════════
# PERFORMANCE
# ════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin-bottom:32px;'>
        <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:6px;'>📊 Model Performance</h2>
        <p style='color:#64748b;'>Trained on 800,000+ labelled domains</p>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    perf = [
        ("🎯", "96%", "Accuracy",  "#10b981"),
        ("🔬", "95%", "Precision", "#38bdf8"),
        ("📡", "94%", "Recall",    "#818cf8"),
        ("⚡", "95%", "F1 Score",  "#f59e0b"),
    ]
    for col, (icon, num, lbl, clr) in zip([p1, p2, p3, p4], perf):
        col.markdown(f"""
        <div class='dash-metric'>
            <div class='dico'>{icon}</div>
            <div class='dnum' style='color:{clr}'>{num}</div>
            <div class='dlbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    try:
        st.image("scripts/roc_curve.png", caption="ROC Curve", use_container_width=True)
    except Exception:
        st.info("ROC curve image not found at scripts/roc_curve.png")
