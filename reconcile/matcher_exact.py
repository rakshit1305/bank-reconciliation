"""
matcher_exact.py
Deterministic, auditable matching — no LLM involved. Two passes:
  1. Reference match: same reference number + same amount (strongest signal).
  2. Amount + date-window match: same absolute amount, matching sign,
     within DATE_WINDOW_DAYS of each other.
Matched pairs are removed from the pool before fuzzy matching runs,
so the LLM only ever sees genuinely ambiguous leftovers.

Both passes require:
  - matching sign (is_credit) — a credit can never match a debit, even if
    the magnitude happens to coincide. Because bank statements and
    company ledgers don't always agree on what "credit" means (see
    CreditConventionResolver below), the ledger's is_credit column is
    normalized to the bank's convention BEFORE either pass runs, so this
    check is always a same-convention, apples-to-apples comparison.
  - matching department, when the LEDGER carries real department scoping
    (see _same_department below) — an untagged bank row is NOT treated
    as a wildcard against a department-scoped ledger, since that was
    proven to cause cross-department false positives (e.g. a Sales-only
    ledger silently "matching" HR/Finance transactions).
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DATE_WINDOW_DAYS = 3

_UTR_PATTERN = re.compile(r"(?<!\d)\d{10,16}(?!\d)")
_CHEQUE_PATTERN = re.compile(r"(?<!\d)\d{6}(?!\d)")


def normalize_reference(ref) -> str:
    """Strip spaces/dashes and leading zeros so that visually-equivalent
    references compare equal (' UTR-004521 ' == '4521' == '004521').
    Returns '' for anything empty/NaN/None."""
    if ref is None or (isinstance(ref, float) and pd.isna(ref)):
        return ""
    s = str(ref).strip()
    if s.lower() in ("", "nan", "none", "n/a"):
        return ""
    s = s.replace(" ", "").replace("-", "")
    s = s.lstrip("0")
    return s


def extract_reference(*fields) -> str:
    """Pulls a UTR/NEFT/RTGS-style reference (10-16 digits) or, failing
    that, a cheque number (exactly 6 digits) out of any of the given
    free-text fields. Returns '' if nothing reference-like is found."""
    parts = []
    for f in fields:
        if f is None:
            continue
        if isinstance(f, float) and pd.isna(f):
            continue
        s = str(f).strip()
        if s == "" or s.lower() in ("n/a", "nan", "none"):
            continue
        parts.append(s)
    joined = " ".join(parts)
    if not joined:
        return ""

    utr_matches = _UTR_PATTERN.findall(joined)
    if utr_matches:
        return normalize_reference(max(utr_matches, key=len))

    cheque_matches = _CHEQUE_PATTERN.findall(joined)
    if cheque_matches:
        return normalize_reference(cheque_matches[0])

    return ""


def _best_reference(row: pd.Series, extra_fields: tuple = ()) -> str:
    direct = normalize_reference(row.get("reference", ""))
    if direct:
        return direct
    fields = [row.get("description", "")] + [row.get(f, "") for f in extra_fields]
    return extract_reference(*fields)


def _prepare_references(bank: pd.DataFrame, ledger: pd.DataFrame) -> None:
    """Populates a normalized '_ref_norm' column on both dataframes.
    Idempotent / cheap to call multiple times — recomputes in place."""
    bank["_ref_norm"] = bank.apply(lambda r: _best_reference(r), axis=1)
    ledger["_ref_norm"] = ledger.apply(
        lambda r: _best_reference(r, extra_fields=("narration", "voucher_no", "bank_ref_no")),
        axis=1,
    )


class CreditConventionResolver:
    """Bank statements and company ledgers don't always agree on what
    "credit" means. A bank statement's is_credit is always from the
    bank's own point of view. A ledger might record the same event the
    same way (bank convention) or the opposite way (accounting
    convention). Blindly requiring is_credit to match, without checking
    which convention the ledger uses, silently zeroes out every match
    for any ledger using the opposite convention.

    Primary strategy: sample sign correlation from reference-matched
    pairs (fast, precise when it works). This has a real blind spot,
    though — if the bank and ledger use genuinely unrelated reference
    ID schemes (e.g. a bank's UTR/transaction ID vs. a company's
    internal voucher numbering, common in real files, with literally no
    shared digits to extract), there's nothing to sample from at all.

    Fallback strategy (see resolve_with_fallback): when reference
    sampling has no usable signal, fall back to a trial-based check —
    run exact matching under both orientations on disposable copies and
    keep whichever produces meaningfully more matches. This has no
    dependency on references working at all, so it covers the case the
    primary strategy structurally cannot.
    """

    def __init__(self, min_samples: int = 3, min_margin: float = 0.6):
        self.min_samples = min_samples
        self.min_margin = min_margin

    def _reference_only_pairs(self, bank: pd.DataFrame, ledger: pd.DataFrame) -> list:
        pairs = []
        ref_bank_rows = bank[bank["_ref_norm"] != ""]
        for _, b in ref_bank_rows.iterrows():
            candidates = ledger[
                (ledger["_ref_norm"] == b["_ref_norm"]) & (ledger["amount_abs"] == b["amount_abs"])
            ]
            for _, l in candidates.iterrows():
                pairs.append((bool(b["is_credit"]), bool(l["is_credit"])))
        return pairs

    def resolve_verbose(self, bank: pd.DataFrame, ledger: pd.DataFrame):
        """Returns (should_invert: bool, confident: bool). confident=False
        means there wasn't enough (or was too mixed) reference-sampled
        signal to trust this method's answer -- the caller should fall
        back to a different detection strategy rather than trusting the
        default False here."""
        pairs = self._reference_only_pairs(bank, ledger)
        total = len(pairs)

        if total < self.min_samples:
            logger.info(
                "CreditConventionResolver: only %d reference-matched sample(s) found "
                "(need >= %d) — inconclusive, no confident answer.",
                total, self.min_samples,
            )
            return False, False

        same_sign = sum(1 for bc, lc in pairs if bc == lc)
        opposite_sign = total - same_sign

        if opposite_sign / total >= self.min_margin:
            logger.info(
                "CreditConventionResolver: %d/%d sampled pairs are opposite-sign -> "
                "inverting is_credit (confident).", opposite_sign, total,
            )
            return True, True
        elif same_sign / total >= self.min_margin:
            logger.info(
                "CreditConventionResolver: %d/%d sampled pairs are same-sign -> "
                "no inversion needed (confident).", same_sign, total,
            )
            return False, True
        else:
            logger.info(
                "CreditConventionResolver: ambiguous signal (%d same-sign / %d "
                "opposite-sign out of %d) — inconclusive.", same_sign, opposite_sign, total,
            )
            return False, False

    def resolve(self, bank: pd.DataFrame, ledger: pd.DataFrame) -> bool:
        """Reference-only decision (no fallback). Prefer
        resolve_with_fallback for actual use — kept for backward
        compatibility with anything calling resolve() directly."""
        decision, _ = self.resolve_verbose(bank, ledger)
        return decision


def _trial_based_sign_detection(bank: pd.DataFrame, ledger: pd.DataFrame) -> bool:
    """Fallback for when reference-based detection has no usable signal
    at all -- e.g. bank and ledger use completely unrelated reference
    schemes (a bank's UTR/transaction ID vs. a company's own internal
    voucher numbering), which is common and NOT fixable by better
    reference parsing, since there's genuinely nothing shared to extract.

    Runs a cheap trial on DISPOSABLE COPIES (never mutates the real
    bank/ledger the caller passed in): counts exact matches under the
    as-loaded orientation vs. a sign-flipped orientation, and returns
    True (invert) only on a clear, meaningful improvement -- avoiding
    flip-flopping on noise for files where neither orientation matches
    well anyway."""

    def trial_count(flip: bool) -> int:
        b = bank.copy()
        l = ledger.copy()
        b["matched"] = False
        l["matched"] = False
        if flip:
            l["is_credit"] = ~l["is_credit"]
        return len(match_by_reference(b, l)) + len(match_by_amount_and_date(b, l))

    count_normal = trial_count(flip=False)
    count_flipped = trial_count(flip=True)

    logger.info(
        "Trial-based sign detection fallback: normal=%d, flipped=%d",
        count_normal, count_flipped,
    )

    if count_flipped > max(count_normal * 1.5, 5) and count_flipped > count_normal:
        logger.info("Trial-based fallback: inverting is_credit.")
        return True
    return False


def resolve_credit_convention(bank: pd.DataFrame, ledger: pd.DataFrame) -> bool:
    """The actual entry point used by run_exact_matching's 'auto' mode:
    try reference-based detection first (fast, precise when references
    genuinely overlap between bank and ledger); if that's inconclusive,
    fall back to trial-based detection (works regardless of whether
    references overlap at all). Returns True if ledger.is_credit should
    be inverted."""
    should_invert, confident = CreditConventionResolver().resolve_verbose(bank, ledger)
    if confident:
        return should_invert

    logger.info("resolve_credit_convention: reference-based detection inconclusive, falling back to trial-based detection.")
    return _trial_based_sign_detection(bank, ledger)


def _normalize_dept(series: pd.Series) -> pd.Series:
    """Strip + lowercase for comparison purposes only. Sheet/tag names
    that are visually identical (e.g. 'Finance_Dept' vs 'Finance_Dept '
    with a trailing space, or a stray case difference) must still be
    treated as the same department — a single invisible whitespace
    difference otherwise silently zeroes out EVERY match across EVERY
    department, since the strict equality check fails uniformly."""
    return series.astype(str).str.strip().str.lower()


def _same_department(bank_dept, ledger_df: pd.DataFrame) -> pd.Series:
    """Returns a boolean mask of ledger rows that are a valid department
    match for this bank row.

    1. The LEDGER has no 'department' column at all, or has one but no
       row in it carries a real department (every value 'N/A') — there's
       nothing to scope against, so every ledger row is a candidate.

    2. The LEDGER *is* department-scoped — requires a department match
       (whitespace/case-insensitive). An untagged ('N/A') bank row is
       NOT treated as a wildcard here — assuming an unlabeled
       transaction belongs to whichever single department the ledger
       happens to cover is exactly what caused cross-department false
       positives (a Sales-only ledger silently absorbing HR/Finance
       rows just because they weren't tagged).
    """
    if "department" not in ledger_df.columns:
        return pd.Series(True, index=ledger_df.index)

    ledger_dept_norm = _normalize_dept(ledger_df["department"])
    ledger_has_real_depts = (ledger_dept_norm != "n/a").any()
    if not ledger_has_real_depts:
        return pd.Series(True, index=ledger_df.index)

    bank_dept_norm = str(bank_dept).strip().lower() if bank_dept not in (None, "") else "n/a"
    return ledger_dept_norm == bank_dept_norm


def match_by_reference(bank: pd.DataFrame, ledger: pd.DataFrame) -> list:
    matches = []
    ref_bank_rows = bank[bank["_ref_norm"] != ""].index.tolist()

    for idx in ref_bank_rows:
        if bank.loc[idx, "matched"]:
            continue
        b = bank.loc[idx]
        dept_mask = _same_department(b.get("department", "N/A"), ledger)

        candidates = ledger[
            (~ledger["matched"])
            & dept_mask
            & (ledger["_ref_norm"] != "")
            & (ledger["_ref_norm"] == b["_ref_norm"])
            & (ledger["amount_abs"] == b["amount_abs"])
            & (ledger["is_credit"] == b["is_credit"])
        ]
        if len(candidates) >= 1:
            l = candidates.iloc[0]
            bank.loc[idx, "matched"] = True
            ledger.loc[l.name, "matched"] = True
            logger.debug(
                "match_by_reference: bank row %s <-> ledger row %s (ref=%s, amount=%s)",
                b["source_row"], l["source_row"], b["_ref_norm"], b["amount"],
            )
            matches.append(
                {
                    "bank_row": b["source_row"],
                    "ledger_row": l["source_row"],
                    "date": b["date"],
                    "amount": b["amount"],
                    "description": b["description"],
                    "bank_date": b["date"],
                    "ledger_date": l["date"],
                    "bank_amount": b["amount"],
                    "ledger_amount": l["amount"],
                    "bank_description": b["description"],
                    "ledger_description": l["description"],
                    "method": "exact_reference",
                    "confidence": 1.0,
                }
            )
    return matches


def match_by_amount_and_date(bank: pd.DataFrame, ledger: pd.DataFrame) -> list:
    matches = []
    for _, b in bank[~bank["matched"]].iterrows():
        dept_mask = _same_department(b.get("department", "N/A"), ledger)
        window = ledger[
            (~ledger["matched"])
            & dept_mask
            & (ledger["amount_abs"] == b["amount_abs"])
            & (ledger["is_credit"] == b["is_credit"])
            & (ledger["date"].sub(b["date"]).abs().dt.days <= DATE_WINDOW_DAYS)
        ]
        if len(window) == 1:
            l = window.iloc[0]
            bank.loc[b.name, "matched"] = True
            ledger.loc[l.name, "matched"] = True
            logger.debug(
                "match_by_amount_and_date: bank row %s <-> ledger row %s (amount=%s, date_diff=%sd)",
                b["source_row"], l["source_row"], b["amount"], abs((l["date"] - b["date"]).days),
            )
            matches.append(
                {
                    "bank_row": b["source_row"],
                    "ledger_row": l["source_row"],
                    "date": b["date"],
                    "amount": b["amount"],
                    "description": b["description"],
                    "bank_date": b["date"],
                    "ledger_date": l["date"],
                    "bank_amount": b["amount"],
                    "ledger_amount": l["amount"],
                    "bank_description": b["description"],
                    "ledger_description": l["description"],
                    "method": "exact_amount_date",
                    "confidence": 0.98,
                }
            )
        elif len(window) > 1:
            logger.debug(
                "match_by_amount_and_date: %d ambiguous candidates for bank row %s "
                "(amount=%s) — leaving for fuzzy/LLM review.",
                len(window), b["source_row"], b["amount_abs"],
            )
    return matches


def run_exact_matching(
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    invert_credit_convention="auto",
) -> list:
    """
    invert_credit_convention:
      - "auto" (default): reference-correlation first, trial-based
        fallback if that's inconclusive (see resolve_credit_convention).
      - True / False: explicit override, skips detection entirely.
    """
    _prepare_references(bank, ledger)

    if ledger.attrs.get("_credit_convention_resolved"):
        logger.info("run_exact_matching: ledger credit convention already resolved upstream; skipping re-detection.")
    else:
        if invert_credit_convention == "auto":
            should_invert = resolve_credit_convention(bank, ledger)
        else:
            should_invert = bool(invert_credit_convention)

        if should_invert:
            ledger["is_credit"] = ~ledger["is_credit"]
            logger.info("run_exact_matching: inverted ledger.is_credit to align with bank convention.")

        ledger.attrs["_credit_convention_resolved"] = True

    ref_matches = match_by_reference(bank, ledger)
    amount_date_matches = match_by_amount_and_date(bank, ledger)
    logger.info(
        "run_exact_matching: %d exact matches total (%d by reference, %d by amount+date).",
        len(ref_matches) + len(amount_date_matches), len(ref_matches), len(amount_date_matches),
    )
    return ref_matches + amount_date_matches


def diagnose_unmatched(bank: pd.DataFrame, ledger: pd.DataFrame, date_window_days: int = None) -> dict:
    """For each still-unmatched bank row, returns {bank_index: short
    human-readable reason} for why no exact candidate was found."""
    window_days = date_window_days if date_window_days is not None else DATE_WINDOW_DAYS
    reasons = {}

    for idx, b in bank[~bank["matched"]].iterrows():
        dept_mask = _same_department(b.get("department", "N/A"), ledger)
        same_dept = ledger[dept_mask]
        if same_dept.empty and "department" in ledger.columns:
            reasons[idx] = "No ledger rows in this department at all"
            continue

        same_amount = same_dept[same_dept["amount_abs"] == b["amount_abs"]]
        if same_amount.empty:
            reasons[idx] = "No ledger row anywhere with this exact amount"
            continue

        same_sign = same_amount[same_amount["is_credit"] == b["is_credit"]]
        if same_sign.empty:
            reasons[idx] = "Amount matches, but credit/debit direction is opposite on every candidate — check sign convention"
            continue

        within_window = same_sign[same_sign["date"].sub(b["date"]).abs().dt.days <= window_days]
        if within_window.empty:
            closest = int(same_sign["date"].sub(b["date"]).abs().dt.days.min())
            reasons[idx] = f"Amount + direction match, but closest date is {closest} day(s) away (window is {window_days})"
            continue

        reasons[idx] = "Multiple equally-plausible candidates by amount/date — needs description-based (fuzzy/LLM) review"

    return reasons