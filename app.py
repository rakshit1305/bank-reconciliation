"""
app.py — Bank Reconciliation Agent Dashboard

A Streamlit front-end over the reconcile/ engine. Upload files (or use the
bundled demo dataset), tune thresholds, run the pipeline, and explore
results through KPIs, charts, filterable tables, and per-department views.
"""

import io
import os
import sys
import tempfile
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv


def get_file_hash(file):
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reconcile.loader import load_source_auto, list_sheets, BANK_COLUMN_MAP, LEDGER_COLUMN_MAP
from reconcile.cleaner import normalize, deduplicate
from reconcile import matcher_exact
from reconcile import matcher_fuzzy
from reconcile.llm_match import llm_available, run_llm_disambiguation, default_provider
from reconcile.categorizer import categorize, CATEGORIES
from reconcile.writer import write_output

load_dotenv()

# ---------------------------------------------------------------------------
# Palette / design tokens
# ---------------------------------------------------------------------------
INK = "#0F1F38"
PAPER = "#F6F5F1"
PAPER_RAISED = "#FFFFFF"
SLATE = "#5B6472"
SLATE_LIGHT = "#8A93A3"
LEDGER_GREEN = "#2F6F5E"
AMBER = "#C98A2C"
CORAL = "#B23A48"
BORDER = "#E2DFD6"

METHOD_COLORS = {
    "exact_reference": LEDGER_GREEN,
    "exact_amount_date": "#4C8C7A",
    "fuzzy_auto": AMBER,
    "llm_match": "#7A5FB5",
}
METHOD_LABELS = {
    "exact_reference": "Exact — Reference",
    "exact_amount_date": "Exact — Amount + Date",
    "fuzzy_auto": "Fuzzy — Auto-accepted",
    "llm_match": "LLM — Disambiguated",
}

st.set_page_config(
    page_title="Ledger & Line — Reconciliation",
    page_icon="📒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {INK};
}}
.stApp {{ background-color: {PAPER}; }}
p, span, label, div {{ color: inherit; }}

.masthead {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 22px 28px; background: {INK}; border-radius: 14px; margin-bottom: 22px;
}}
.masthead-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 26px; color: {PAPER}; margin: 0; }}
.masthead-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: {SLATE_LIGHT}; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }}

.stamp {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 15px;
    letter-spacing: 0.12em; text-transform: uppercase; padding: 10px 22px;
    border: 3px solid currentColor; border-radius: 8px; transform: rotate(-4deg); display: inline-block;
}}
.stamp-pass {{ color: #6FCF97; }}
.stamp-fail {{ color: #EB8794; }}
.stamp-idle {{ color: {SLATE_LIGHT}; }}

.kpi-card {{ background: {PAPER_RAISED}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px; height: 100%; }}
.kpi-label {{ font-size: 12px; font-weight: 600; color: {SLATE}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
.kpi-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 600; color: {INK}; line-height: 1.1; }}

.section-label {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 15px; color: {INK}; margin: 6px 0 10px 0; padding-bottom: 8px; border-bottom: 2px solid {BORDER}; }}

.detect-card {{
    background: {PAPER_RAISED}; border: 1px solid {BORDER}; border-left: 4px solid {LEDGER_GREEN};
    border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; font-size: 13px; color: {INK};
}}
.detect-card b {{ color: {INK}; }}
.detect-card.ledger {{ border-left-color: #4C8C7A; }}

section[data-testid="stSidebar"] {{ background-color: {INK}; }}
section[data-testid="stSidebar"] * {{ color: {PAPER} !important; }}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 4px; background: transparent; }}
[data-testid="stTab"],
[data-testid="stTab"] p,
[data-testid="stTab"] span,
[data-testid="stTab"] div {{
    color: {INK} !important;
    -webkit-text-fill-color: {INK} !important;
    opacity: 1 !important;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600 !important;
}}
[data-testid="stTab"][aria-selected="true"],
[data-testid="stTab"][aria-selected="true"] p,
[data-testid="stTab"][aria-selected="true"] span {{
    color: {INK} !important;
    -webkit-text-fill-color: {INK} !important;
}}
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📒 Ledger & Line")
    st.caption("Bank Reconciliation Agent")
    st.markdown("---")

    st.markdown("**1 · Data source**")
    data_source = st.radio(
        "Choose input",
        ["Use small demo dataset", "Upload my own files"],
        label_visibility="collapsed",
    )

    bank_files, ledger_files = [], []
    bank_file_depts, ledger_file_depts = {}, {}

    if data_source == "Upload my own files":
        st.caption("You can upload multiple files per side — e.g. a separate bank export per department.")

        bank_files = st.file_uploader(
            "Bank statement(s) (.csv or .xlsx)", type=["csv", "xlsx"],
            accept_multiple_files=True, key="bank_upl",
        )
        for f in bank_files or []:
            sheets = list_sheets(f)
            if len(sheets) <= 1:
                bank_file_depts[f.name] = st.text_input(
                    f"Department for '{f.name}' (optional)",
                    placeholder="e.g. Sales",
                    key=f"bank_dept_tag_{f.name}",
                )
            else:
                st.caption(f"📁 '{f.name}': {len(sheets)} sheets detected — each treated as its own department: {', '.join(sheets)}")
                bank_file_depts[f.name] = None

        ledger_files = st.file_uploader(
            "Ledger extract(s) (.csv or .xlsx)", type=["csv", "xlsx"],
            accept_multiple_files=True, key="ledger_upl",
        )
        for f in ledger_files or []:
            sheets = list_sheets(f)
            if len(sheets) <= 1:
                ledger_file_depts[f.name] = st.text_input(
                    f"Department for '{f.name}' (optional)",
                    placeholder="e.g. Sales",
                    key=f"ledger_dept_tag_{f.name}",
                )
            else:
                st.caption(f"📁 '{f.name}': {len(sheets)} sheets detected — each treated as its own department: {', '.join(sheets)}")
                ledger_file_depts[f.name] = None
    else:
        st.info("Using bundled sample_data/ — small demo files.", icon="ℹ️")

    st.markdown("---")
    st.markdown("---")
    st.markdown("**1a · Ledger credit/debit convention**")
    sign_convention_choice = st.radio(
        "Sign convention",
        ["Auto-detect (recommended)", "Force flip", "Force no flip"],
        label_visibility="collapsed",
        help="Some ledgers record their own bank account from a standard accounting "
             "perspective (money in = debit) — the opposite of how the bank statement "
             "itself labels the same transaction (money in = credit). Auto-detect tries "
             "both orientations and keeps whichever produces more matches. Use the "
             "manual options only if auto-detect gets it wrong for a specific file.",
    )

    st.markdown("---")
    st.markdown("**2 · Matching thresholds**")
    date_window_exact = st.slider(
        "Exact match — date window (days)",
        0, 10, 3,
        help="What it does: When a bank transaction and a ledger entry have the exact same amount, the system still checks that their dates are close enough to be the same real-world event — not just a coincidence of two unrelated transactions happening to be the same amount.\n\nExample: A ₹10,000 salary credit hits the bank account on 5 June. The ledger recorded it on 7 June (2 days later, common when payroll processing lags). With the date window set to 3 days, the system treats these as an automatic, high-confidence match."
    )
    amount_tol_pct = st.slider("Fuzzy match — amount tolerance (%)", 0.0, 10.0, 2.0, 0.5,
        help="What it does: Sometimes the amount on the bank statement and the amount in the ledger don't match to the exact paisa — because of bank fees, rounding, or a small deduction. This slider allows a small percentage difference and still considers the two a possible match.\n\nExample: With a 2% tolerance, a bank withdrawal of ₹10,000 would be compared against ledger entries anywhere between ₹9,800 and ₹10,200 — covering, for example, a ₹150 bank service charge that made the ledger's recorded amount slightly different from what actually left the account.\n\nWhen to change it: Increase it if your bank routinely deducts fees/charges that create small but consistent amount gaps.\n\n Decrease it (even to 0%) if you need strict amount-for-amount matching and don't want any tolerance for discrepancies."
    )
    date_window_fuzzy = st.slider("Fuzzy match — date window (days)", 1, 30, 10,
        help="Default: 10 days.\n\n What it does:If we can't find an exact match, the system looks a bit wider to find possible matches. This setting controls how many days before or after the transaction we should search.\n\nExample: A ₹5,000 payment on 1 June can be matched with entries up to 10 days later (e.g., 11 June), in case it was recorded late or described differently.\n\n Why this is wider: Exact matching is strict. This step is more flexible and helps catch real-world cases where dates or descriptions don't line up perfectly.")
    min_desc_score = st.slider(
        "Fuzzy — minimum score to shortlist",
        0, 100, 45,
        help="Default: 45\n\nWhat it does:\nThis controls how similar the transaction descriptions need to be before they are considered a possible match.\n\nExample:\n\"ATM WDL 4521\" and \"ATM Cash Withdrawal\" are similar → included.\n\"ATM WDL 4521\" and \"Office Rent Payment\" are very different → ignored.\n\nWhen to change it:\nIncrease it if you see too many irrelevant matches.\nLower it if real matches are being missed due to different wording."
    )

    auto_accept_threshold = st.slider("Fuzzy — auto-accept threshold", 50, 100, 90,
        help="Default: 90 (out of 100).\n\nWhat it does: Once a bank transaction has exactly one shortlisted candidate, this slider decides whether that candidate is good enough to be automatically accepted as a match — with no human or LLM review needed — based on how similar its description is.\n\nExample:\n\"NEFT TRANSFER FROM ABC CORP\" vs. \"NEFT ABC CORP\" → scores around 92 → automatically accepted as a match (above 90).\n\"NEFT TRANSFER FROM ABC CORP\" vs. \"NEFT XYZ Ltd\" → scores around 40 → not auto-accepted; goes to the LLM step (if enabled) or the \"Needs review\" tab for a human to decide.\n\nWhen to change it:\nRaise it (e.g. to 95+) to be more cautious — fewer transactions get auto-accepted, and more go to human/LLM review instead.\nLower it (e.g. to 80) to let the system resolve more matches on its own automatically — faster, but with slightly more risk of an incorrect auto-match slipping through."
    )

    st.markdown("---")
    st.markdown("**3 · LLM provider**")
    provider_options = []
    if llm_available("groq"):
        provider_options.append("Groq (free)")
    if llm_available("openai"):
        provider_options.append("OpenAI")
    if llm_available("anthropic"):
        provider_options.append("Anthropic")
    if not provider_options:
        provider_options = ["None configured"]
        st.caption("Set GROQ_API_KEY (free), OPENAI_API_KEY, or ANTHROPIC_API_KEY in your .env to enable LLM steps.")

    llm_provider_label = st.radio("Provider", provider_options, horizontal=True, disabled=(provider_options == ["None configured"]))
    if "Groq" in llm_provider_label:
        llm_provider = "groq"
    elif "OpenAI" in llm_provider_label:
        llm_provider = "openai"
    elif "Anthropic" in llm_provider_label:
        llm_provider = "anthropic"
    else:
        llm_provider = None

    st.markdown("**3a · Match disambiguation**")
    use_llm = st.checkbox(
        "Enable LLM step for leftover ambiguous rows", value=False,
        help="Only calls the API for rows exact+fuzzy matching couldn't resolve.",
    )
    llm_confidence_threshold = st.slider("LLM auto-accept confidence", 0.5, 1.0, 0.85, 0.05, disabled=not use_llm)

    st.markdown("**3b · Description categorization**")
    use_llm_categorize = st.checkbox(
        "Enable LLM categorization for uncategorized rows", value=False,
        help="Free keyword rules run first automatically. This only sends the leftovers, batched.",
    )

    st.markdown("---")
    run_clicked = st.button("▶  Run reconciliation", width='stretch', type="primary")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _detect_and_fix_ledger_sign_convention(bank_raw, ledger_raw, override: str = "auto"):
    """Some ledgers record their OWN bank account from a standard
    double-entry accounting perspective, where the bank account is an
    asset -- money coming IN is a DEBIT to that asset, money going OUT
    is a CREDIT. This is the exact opposite of how the bank's own
    statement labels the same transaction (money in = credit, from the
    bank's perspective, since the bank owes the customer more).

    Left unhandled, this makes every genuinely matching pair fail the
    is_credit equality check in the matchers, even though amount, date,
    and description all agree -- producing near-zero matches on an
    otherwise perfectly reconcilable file.

    override: "auto" (default) runs the empirical trial below. "flip" or
    "no_flip" skips detection entirely and does what it says -- for the
    rare case where auto-detect gets it wrong on a specific file and the
    user needs a manual escape hatch.

    Detected empirically rather than assumed: runs a cheap exact-match
    trial under both the as-loaded orientation and a sign-flipped
    orientation, and keeps whichever produces meaningfully more
    matches. Returns (ledger_raw_to_use, was_flipped: bool).
    """
    if override == "flip":
        ledger_flipped = ledger_raw.copy()
        ledger_flipped["amount"] = -ledger_flipped["amount"]
        return ledger_flipped, True
    if override == "no_flip":
        return ledger_raw, False

    def trial_count(ledger_df):
        b = deduplicate(normalize(bank_raw.copy()), "sign-detect-trial")
        l = deduplicate(normalize(ledger_df.copy()), "sign-detect-trial")
        return len(matcher_exact.run_exact_matching(b, l))

    count_normal = trial_count(ledger_raw)

    ledger_flipped = ledger_raw.copy()
    ledger_flipped["amount"] = -ledger_flipped["amount"]
    count_flipped = trial_count(ledger_flipped)

    # Only flip on a clear, meaningful improvement -- avoids flip-
    # flopping on noise for files where neither orientation matches well.
    if count_flipped > max(count_normal * 1.5, 5) and count_flipped > count_normal:
        return ledger_flipped, True
    return ledger_raw, False


def run_pipeline():
    if data_source == "Upload my own files":
        if not bank_files or not ledger_files:
            st.error("Please upload at least one bank statement file and one ledger file.")
            return None

        # --- Duplicate-file safety checks (scoped to the upload path only) ---
        bank_hashes = [get_file_hash(f) for f in bank_files]
        ledger_hashes = [get_file_hash(f) for f in ledger_files]

        if any(bh in ledger_hashes for bh in bank_hashes):
            st.error("⚠️ Same file detected in Bank and Ledger uploads. Please upload different files.")
            return None

        if len(set(bank_hashes)) != len(bank_hashes):
            st.error("⚠️ Duplicate files uploaded in Bank section.")
            return None

        if len(set(ledger_hashes)) != len(ledger_hashes):
            st.error("⚠️ Duplicate files uploaded in Ledger section.")
            return None

        bank_parts, bank_styles = [], []
        for f in bank_files:
            part = load_source_auto(f, "Bank statement", BANK_COLUMN_MAP)
            dept = bank_file_depts.get(f.name)
            if dept:
                part["department"] = dept
            bank_styles.append(part.attrs.get("detected_amount_style", "unknown"))
            bank_parts.append(part)
        bank_raw = pd.concat(bank_parts, ignore_index=True)
        bank_style = bank_styles[0] if len(set(bank_styles)) == 1 else "mixed"

        ledger_parts, ledger_styles = [], []
        for f in ledger_files:
            part = load_source_auto(f, "Ledger", LEDGER_COLUMN_MAP)
            dept = ledger_file_depts.get(f.name)
            if dept:
                part["department"] = dept
            ledger_styles.append(part.attrs.get("detected_amount_style", "unknown"))
            ledger_parts.append(part)
        ledger_raw = pd.concat(ledger_parts, ignore_index=True)
        ledger_style = ledger_styles[0] if len(set(ledger_styles)) == 1 else "mixed"
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        bank_src = os.path.join(base, "sample_data", "bank_statement.csv")
        ledger_src = os.path.join(base, "sample_data", "ledger.xlsx")
        bank_raw = load_source_auto(bank_src, "Bank statement", BANK_COLUMN_MAP)
        ledger_raw = load_source_auto(ledger_src, "Ledger", LEDGER_COLUMN_MAP)
        bank_style = bank_raw.attrs.get("detected_amount_style", "unknown")
        ledger_style = ledger_raw.attrs.get("detected_amount_style", "unknown")

    override_map = {
        "Auto-detect (recommended)": "auto",
        "Force flip": "flip",
        "Force no flip": "no_flip",
    }
    ledger_raw, sign_convention_flipped = _detect_and_fix_ledger_sign_convention(
        bank_raw, ledger_raw, override=override_map.get(sign_convention_choice, "auto")
    )

    bank = normalize(bank_raw)
    bank = deduplicate(bank, "Bank statement")
    ledger = normalize(ledger_raw)
    ledger = deduplicate(ledger, "Ledger")

    matcher_exact.DATE_WINDOW_DAYS = date_window_exact
    matcher_fuzzy.AMOUNT_TOLERANCE_PCT = amount_tol_pct / 100.0
    matcher_fuzzy.DATE_WINDOW_DAYS = date_window_fuzzy
    matcher_fuzzy.MIN_DESC_SCORE = min_desc_score
    matcher_fuzzy.TOP_N_CANDIDATES = 3

    matches = matcher_exact.run_exact_matching(bank, ledger)

    shortlist = matcher_fuzzy.shortlist_candidates(bank, ledger)
    fuzzy_matches, remaining = matcher_fuzzy.auto_accept_high_confidence(
        shortlist, bank, ledger, threshold=auto_accept_threshold
    )
    matches += fuzzy_matches

    llm_log = []
    llm_provider_warning = None
    if use_llm and llm_provider and llm_available(llm_provider):
        import reconcile.llm_match as llm_mod
        llm_mod.CONFIDENCE_AUTO_ACCEPT = llm_confidence_threshold
        llm_matches = run_llm_disambiguation(remaining, bank, ledger, llm_log, provider=llm_provider)
        matches += llm_matches
        skipped = [r for r in llm_log if r["result"].get("rationale", "").startswith("Skipped —")]
        if skipped:
            llm_provider_warning = (
                f"{skipped[0]['result']['rationale'].replace('Skipped — ', '')} "
                f"— {len(skipped)} row(s) fell back to manual review instead of being auto-matched."
            )

    bank = categorize(bank, use_llm=use_llm_categorize and llm_provider and llm_available(llm_provider), provider=llm_provider)

    unmatched_reasons = matcher_exact.diagnose_unmatched(bank, ledger, date_window_days=date_window_exact)

    return {
        "bank": bank, "ledger": ledger, "matches": matches, "llm_log": llm_log,
        "llm_provider_warning": llm_provider_warning,
        "sign_convention_flipped": sign_convention_flipped,
        "unmatched_reasons": unmatched_reasons,
        "run_at": datetime.now(), "bank_style": bank_style, "ledger_style": ledger_style,
    }


if run_clicked:
    with st.spinner("Running exact match → fuzzy match → LLM disambiguation..."):
        result = run_pipeline()
        if result:
            st.session_state.results = result

# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
res = st.session_state.results

if res:
    bank, ledger, matches = res["bank"], res["ledger"], res["matches"]
    total_bank_amt = bank["amount"].sum()
    matched_bank_amt = bank[bank["matched"]]["amount"].sum()
    unmatched_bank_amt = bank[~bank["matched"]]["amount"].sum()
    control_pass = abs((matched_bank_amt + unmatched_bank_amt) - total_bank_amt) < 0.01
    stamp_class = "stamp-pass" if control_pass else "stamp-fail"
    stamp_text = "Reconciled ✓" if control_pass else "Control Check Failed"
    run_label = res["run_at"].strftime("%d %b %Y, %H:%M")
else:
    stamp_class, stamp_text, run_label = "stamp-idle", "Awaiting Run", "—"

st.markdown(f"""
<div class="masthead">
    <div>
        <p class="masthead-title">📒 Ledger &amp; Line</p>
        <p class="masthead-sub">Bank Reconciliation Agent · Last run: {run_label}</p>
    </div>
    <div class="stamp {stamp_class}">{stamp_text}</div>
</div>
""", unsafe_allow_html=True)

if not res:
    st.info("Set your data source and thresholds in the sidebar, then click **Run reconciliation**.", icon="👈")
    st.stop()

if res.get("llm_provider_warning"):
    st.warning(res["llm_provider_warning"], icon="⚠️")

if res.get("sign_convention_flipped"):
    st.info(
        "Detected that your ledger records this bank account from a standard accounting "
        "perspective (money in = debit, money out = credit) — the opposite of how the bank "
        "statement itself labels the same transactions. Automatically corrected before matching, "
        "based on which orientation actually produced more matches.",
        icon="🔁",
    )

# ---------------------------------------------------------------------------
# Detection confirmation — "is this actually the bank statement or ledger?"
# ---------------------------------------------------------------------------
d1, d2 = st.columns(2)
with d1:
    st.markdown(f"""
    <div class="detect-card">
        🏦 <b>Bank statement detected:</b> {len(bank)} rows across {bank['department'].nunique()} department(s)<br>
        Format: {res['bank_style']}
    </div>
    """, unsafe_allow_html=True)
with d2:
    st.markdown(f"""
    <div class="detect-card ledger">
        📗 <b>Ledger detected:</b> {len(ledger)} rows across {ledger['department'].nunique()} department(s)<br>
        Format: {res['ledger_style']}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Date range + department filter
# ---------------------------------------------------------------------------
min_date, max_date = bank["date"].min().date(), bank["date"].max().date()
departments = sorted(d for d in bank["department"].unique() if d != "N/A")
has_departments = len(departments) > 0

fc1, fc2 = st.columns([2, 1]) if has_departments else (st.container(), None)

with fc1:
    st.markdown('<p class="section-label">Filter by transaction date</p>', unsafe_allow_html=True)
    date_range = st.slider(
        "Date range", min_value=min_date, max_value=max_date,
        value=(min_date, max_date), label_visibility="collapsed", format="DD MMM",
    )

if has_departments:
    with fc2:
        st.markdown('<p class="section-label">Department</p>', unsafe_allow_html=True)
        selected_departments = st.multiselect("Department", departments, default=departments, label_visibility="collapsed")
else:
    selected_departments = ["N/A"]

mask_bank = (
    (bank["date"].dt.date >= date_range[0]) & (bank["date"].dt.date <= date_range[1])
    & (bank["department"].isin(selected_departments))
)
bank_f = bank[mask_bank]
matched_bank_rows = set(bank_f[bank_f["matched"]]["source_row"])
matches_f = [m for m in matches if m["bank_row"] in set(bank_f["source_row"])]


def render_dashboard(bank_f, matches_f, ledger, key_prefix, unmatched_reasons=None):
    n_bank = len(bank_f)
    n_matched = len(matches_f)
    matched_rows = set(bank_f[bank_f["matched"]]["source_row"])
    n_unmatched_bank = n_bank - len(matched_rows)
    match_rate = (len(matched_rows) / n_bank * 100) if n_bank else 0
    unmatched_value = bank_f[~bank_f["matched"]]["amount"].abs().sum()

    kpis = [
        ("Bank transactions", f"{n_bank}"),
        ("Matched pairs", f"{n_matched}"),
        ("Match rate", f"{match_rate:.1f}%"),
        ("Unmatched (bank)", f"{n_unmatched_bank}"),
        ("Unmatched value", f"₹{unmatched_value:,.0f}"),
    ]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, kpis):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.markdown('<p class="section-label">Matched vs unmatched</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Matched", "Unmatched"], values=[len(matched_rows), n_unmatched_bank], hole=0.62,
            marker=dict(colors=[LEDGER_GREEN, CORAL]), textinfo="value+percent",
            textfont=dict(family="IBM Plex Mono", size=13),
        ))
        fig.update_layout(showlegend=True, height=290, margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter", color=INK, size=13),
                           legend=dict(orientation="h", yanchor="bottom", y=-0.1, font=dict(color=INK)))
        st.plotly_chart(fig, width='stretch', theme=None, key=f"{key_prefix}_donut")

    with c2:
        st.markdown('<p class="section-label">Match method breakdown</p>', unsafe_allow_html=True)
        if matches_f:
            method_counts = pd.Series([m["method"] for m in matches_f]).value_counts()
            fig2 = go.Figure(go.Bar(
                x=method_counts.values, y=[METHOD_LABELS.get(m, m) for m in method_counts.index], orientation="h",
                marker_color=[METHOD_COLORS.get(m, SLATE) for m in method_counts.index],
                text=[f"  {v}" for v in method_counts.values], textposition="outside", textfont=dict(family="IBM Plex Mono"),
            ))
            fig2.update_layout(height=290, margin=dict(t=36, b=40, l=10, r=40),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Inter", color=INK, size=13),
                               xaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(color=INK), title_font=dict(color=INK), automargin=True),
                               yaxis=dict(autorange="reversed", tickfont=dict(color=INK), title_font=dict(color=INK), automargin=True))
            st.plotly_chart(fig2, width='stretch', theme=None, key=f"{key_prefix}_method")
        else:
            st.caption("No matches yet in this selection.")

    c3, c4 = st.columns([1.3, 1])
    with c3:
        st.markdown('<p class="section-label">Transactions over time</p>', unsafe_allow_html=True)
        timeline = bank_f.copy()
        timeline["status"] = timeline["matched"].map({True: "Matched", False: "Unmatched"})
        tg = timeline.groupby([timeline["date"].dt.date, "status"]).size().reset_index(name="count")
        tg.columns = ["date", "status", "count"]
        fig3 = px.bar(tg, x="date", y="count", color="status",
                      color_discrete_map={"Matched": LEDGER_GREEN, "Unmatched": CORAL}, barmode="stack")
        fig3.update_layout(height=320, margin=dict(t=36, b=50, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter", color=INK, size=13),
                           xaxis=dict(showgrid=False, tickfont=dict(color=INK),
                                      title=dict(text="Date", font=dict(color=INK), standoff=15), automargin=True),
                           yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(color=INK),
                                      title=dict(text="Count", font=dict(color=INK), standoff=10), automargin=True),
                           legend=dict(font=dict(color=INK)), legend_title_text="")
        st.plotly_chart(fig3, width='stretch', theme=None, key=f"{key_prefix}_timeline")

    with c4:
        st.markdown('<p class="section-label">Match confidence distribution</p>', unsafe_allow_html=True)
        if matches_f:
            conf_df = pd.DataFrame(matches_f)
            fig4 = go.Figure(go.Histogram(x=conf_df["confidence"], nbinsx=10, marker_color=LEDGER_GREEN, opacity=0.85))
            fig4.update_layout(height=320, margin=dict(t=36, b=50, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Inter", color=INK, size=13),
                               xaxis=dict(title=dict(text="Confidence", font=dict(color=INK), standoff=15),
                                          range=[0, 1.05], showgrid=False, tickfont=dict(color=INK), automargin=True),
                               yaxis=dict(title=dict(text="Count", font=dict(color=INK), standoff=10),
                                          showgrid=True, gridcolor=BORDER, tickfont=dict(color=INK), automargin=True))
            st.plotly_chart(fig4, width='stretch', theme=None, key=f"{key_prefix}_conf")
        else:
            st.caption("No matches yet in this selection.")

    st.markdown('<p class="section-label">Spend by category</p>', unsafe_allow_html=True)
    if "category" in bank_f.columns and len(bank_f):
        cat_grp = bank_f.assign(abs_amount=bank_f["amount"].abs()).groupby("category")["abs_amount"].sum().sort_values(ascending=True)
        fig5 = go.Figure(go.Bar(x=cat_grp.values, y=cat_grp.index, orientation="h", marker_color=LEDGER_GREEN,
                                text=[f"  ₹{v:,.0f}" for v in cat_grp.values], textposition="outside", textfont=dict(family="IBM Plex Mono")))
        fig5.update_layout(height=max(260, 32 * len(cat_grp)), margin=dict(t=36, b=50, l=10, r=70),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter", color=INK, size=13),
                           xaxis=dict(showgrid=True, gridcolor=BORDER,
                                      title=dict(text="Amount (₹, absolute)", font=dict(color=INK), standoff=15),
                                      tickfont=dict(color=INK), automargin=True),
                           yaxis=dict(tickfont=dict(color=INK), automargin=True))
        st.plotly_chart(fig5, width='stretch', theme=None, key=f"{key_prefix}_cat")

    st.markdown('<p class="section-label">Transaction detail</p>', unsafe_allow_html=True)
    tab_m, tab_bo, tab_lo, tab_r = st.tabs(["✅ Matched", "🏦 Bank only", "📗 Ledger only", "🕵️ Needs review"])

    with tab_m:
        if matches_f:
            mdf = pd.DataFrame(matches_f)
            mdf["date"] = pd.to_datetime(mdf["date"]).dt.strftime("%d %b %Y")
            mdf["method"] = mdf["method"].map(lambda m: METHOD_LABELS.get(m, m))
            cols_show = [c for c in ["bank_row", "ledger_row", "date", "amount", "description", "method", "confidence", "rationale"] if c in mdf.columns]
            st.dataframe(mdf[cols_show], width='stretch', hide_index=True)
        else:
            st.caption("No matched rows in this selection.")

    with tab_bo:
        cols_show = [c for c in ["source_row", "date", "description", "amount", "reference", "category", "department"] if c in bank_f.columns]
        bo = bank_f[~bank_f["matched"]][cols_show].copy()
        bo["date"] = bo["date"].dt.strftime("%d %b %Y")
        if unmatched_reasons:
            bo["likely_reason"] = bank_f[~bank_f["matched"]].index.map(lambda i: unmatched_reasons.get(i, ""))
        st.dataframe(bo.sort_values("amount", key=abs, ascending=False), width='stretch', hide_index=True)

    with tab_lo:
        lo = ledger[~ledger["matched"]][["source_row", "date", "description", "amount", "reference", "department"]].copy()
        lo["date"] = lo["date"].dt.strftime("%d %b %Y")
        st.dataframe(lo.sort_values("amount", key=abs, ascending=False), width='stretch', hide_index=True)

    with tab_r:
        llm_log = res["llm_log"]
        if llm_log:
            review_rows = [r for r in llm_log if r["result"].get("confidence", 0) < llm_confidence_threshold or r["result"].get("match_ledger_row") is None]
            if review_rows:
                rdf = pd.DataFrame([{
                    "bank_row": r["bank_row"]["row"], "description": r["bank_row"]["description"],
                    "amount": r["bank_row"]["amount"], "candidates_considered": len(r["candidates"]),
                    "llm_suggestion": r["result"].get("match_ledger_row"), "llm_confidence": r["result"].get("confidence"),
                    "llm_rationale": r["result"].get("rationale"),
                } for r in review_rows])
                st.dataframe(rdf, width='stretch', hide_index=True)
            else:
                st.caption("No rows required manual review.")
        else:
            st.caption("LLM step was not run.")


# ---------------------------------------------------------------------------
# Render: "All departments" + one tab per department (if any detected)
# ---------------------------------------------------------------------------
if has_departments and len(selected_departments) > 1:
    dept_tabs = st.tabs(["🌐 All departments"] + [f"🏷️ {d}" for d in selected_departments])
    with dept_tabs[0]:
        render_dashboard(bank_f, matches_f, ledger, "all", unmatched_reasons=res.get("unmatched_reasons"))
    for i, dept in enumerate(selected_departments, start=1):
        with dept_tabs[i]:
            bank_dept = bank_f[bank_f["department"] == dept]
            matches_dept = [m for m in matches_f if m["bank_row"] in set(bank_dept["source_row"])]
            render_dashboard(bank_dept, matches_dept, ledger, f"dept_{dept}", unmatched_reasons=res.get("unmatched_reasons"))
else:
    render_dashboard(bank_f, matches_f, ledger, "single", unmatched_reasons=res.get("unmatched_reasons"))

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
tmp_path = os.path.join(tempfile.gettempdir(), "_recon_download.xlsx")
write_output(tmp_path, matches, bank, ledger, res["llm_log"], llm_confidence_threshold=llm_confidence_threshold)
with open(tmp_path, "rb") as f:
    st.download_button(
        "⬇ Download full reconciliation workbook (.xlsx)", data=f.read(),
        file_name=f"reconciled_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width='stretch',
    )