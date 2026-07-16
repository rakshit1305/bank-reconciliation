"""
matcher_exact.py
Deterministic, auditable matching — no LLM involved. Two passes:
  1. Reference match: same reference number + same amount (strongest signal).
  2. Amount + date-window match: same absolute amount, opposite sign
     (bank credit <-> ledger debit or vice versa, depending on your
     ledger's sign convention), within DATE_WINDOW_DAYS of each other.
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
    # Library-friendly default: only configures handlers if nothing else
    # (e.g. the calling application) already has.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DATE_WINDOW_DAYS = 3

# 10-16 digit sequences catch UTR / NEFT / RTGS / IMPS reference numbers
# (bank-issued transaction IDs); a bare 6-digit sequence catches cheque
# numbers. Negative lookaround on both sides keeps us from matching a
# 6-digit substring out of the middle of a longer digit run (e.g. we
# don't want the "123456" inside a 12-digit UTR to be misread as a
# cheque number).
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
    free-text fields (description, narration, voucher_no, bank_ref_no,
    ...). UTRs are preferred over cheque numbers since they're a much
    stronger unique-transaction signal. Returns a normalized string, or
    '' if nothing reference-like is found anywhere in the fields."""
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
        # Longest match wins in the (rare) case of multiple candidates —
        # UTRs are usually the longest digit run in a narration string.
        return normalize_reference(max(utr_matches, key=len))

    cheque_matches = _CHEQUE_PATTERN.findall(joined)
    if cheque_matches:
        return normalize_reference(cheque_matches[0])

    return ""


def _best_reference(row: pd.Series, extra_fields: tuple[str, ...] = ()) -> str:
    """A row's reference is whatever's in its own 'reference' column if
    that's usable; otherwise we fall back to regex-extracting one out of
    free-text fields (description plus whatever extra_fields apply,
    e.g. narration/voucher_no/bank_ref_no for ledger rows)."""
    direct = normalize_reference(row.get("reference", ""))
    if direct:
        return direct
    fields = [row.get("description", "")] + [row.get(f, "") for f in extra_fields]
    return extract_reference(*fields)


def _prepare_references(bank: pd.DataFrame, ledger: pd.DataFrame) -> None:
    """Populates a normalized '_ref_norm' column on both dataframes.
    Reused by exact reference matching AND by CreditConventionResolver
    (which needs reference-only pairs to sample sign correlation from).
    Idempotent / cheap to call multiple times — recomputes in place."""
    bank["_ref_norm"] = bank.apply(lambda r: _best_reference(r), axis=1)
    ledger["_ref_norm"] = ledger.apply(
        lambda r: _best_reference(r, extra_fields=("narration", "voucher_no", "bank_ref_no")),
        axis=1,
    )


class CreditConventionResolver:
    """Bank statements and company ledgers don't always agree on what
    "credit" means. A bank statement's is_credit is always from the
    bank's own point of view (money in = credit). A ledger, however,
    might record things the same way (bank convention) OR the opposite
    way (accounting convention — e.g. a customer receipt is a credit to
    Cash but the matching AR entry is posted as a debit, or vice versa
    for payments). Blindly requiring is_credit to match, without first
    checking which convention the ledger uses, silently zeroes out every
    match for any ledger that happens to use the opposite convention.

    Detection strategy: find a handful of high-confidence pairs using
    reference + amount ONLY (sign intentionally ignored), then check
    whether those pairs agree in sign (same-sign -> bank convention) or
    disagree (opposite-sign -> accounting convention, needs inverting).
    If there isn't enough signal, or the signal is genuinely mixed, we
    default to NOT inverting (i.e. keep the historical, pre-existing
    behavior) rather than guess.
    """

    def __init__(self, min_samples: int = 3, min_margin: float = 0.6):
        self.min_samples = min_samples
        self.min_margin = min_margin  # fraction of samples required to agree before we trust it

    def _reference_only_pairs(self, bank: pd.DataFrame, ledger: pd.DataFrame) -> list[tuple[bool, bool]]:
        pairs = []
        ref_bank_rows = bank[bank["_ref_norm"] != ""]
        for _, b in ref_bank_rows.iterrows():
            candidates = ledger[
                (ledger["_ref_norm"] == b["_ref_norm"]) & (ledger["amount_abs"] == b["amount_abs"])
            ]
            for _, l in candidates.iterrows():
                pairs.append((bool(b["is_credit"]), bool(l["is_credit"])))
        return pairs

    def resolve(self, bank: pd.DataFrame, ledger: pd.DataFrame) -> bool:
        """Returns True if the ledger's is_credit column should be
        inverted before matching against the bank's is_credit column."""
        pairs = self._reference_only_pairs(bank, ledger)
        total = len(pairs)

        if total < self.min_samples:
            logger.info(
                "CreditConventionResolver: only %d reference-matched sample(s) found "
                "(need >= %d) — defaulting to no inversion.",
                total, self.min_samples,
            )
            return False

        same_sign = sum(1 for bc, lc in pairs if bc == lc)
        opposite_sign = total - same_sign

        if opposite_sign / total >= self.min_margin:
            logger.info(
                "CreditConventionResolver: %d/%d sampled pairs are opposite-sign -> "
                "ledger appears to use accounting convention; inverting is_credit.",
                opposite_sign, total,
            )
            return True
        elif same_sign / total >= self.min_margin:
            logger.info(
                "CreditConventionResolver: %d/%d sampled pairs are same-sign -> "
                "ledger appears to use bank convention; no inversion needed.",
                same_sign, total,
            )
            return False
        else:
            logger.info(
                "CreditConventionResolver: ambiguous signal (%d same-sign / %d "
                "opposite-sign out of %d) — defaulting to no inversion.",
                same_sign, opposite_sign, total,
            )
            return False


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

    Three situations, handled distinctly:

    1. The LEDGER has no 'department' column at all — nothing to scope
       against, so every ledger row is a candidate.

    2. The LEDGER has a 'department' column, but no row in it carries a
       real department (every value is 'N/A' / blank — e.g. one unified
       ledger file with no per-sheet/per-tag breakdown). Same as (1):
       fall back to comparing the bank row against the whole ledger.

    3. The LEDGER *is* department-scoped (e.g. it's a Sales-only extract,
       or has real per-sheet department names). In that case we require
       a department match (whitespace/case-insensitive — see
       _normalize_dept). Critically, an untagged ('N/A') bank row is NOT
       treated as a wildcard here — assuming an unlabeled transaction
       belongs to whichever single department the ledger happens to
       cover is exactly what caused cross-department false positives
       (a Sales-only ledger silently absorbing HR/Finance rows just
       because they weren't tagged).
    """
    if "department" not in ledger_df.columns:
        return pd.Series(True, index=ledger_df.index)

    ledger_dept_norm = _normalize_dept(ledger_df["department"])
    ledger_has_real_depts = (ledger_dept_norm != "n/a").any()
    if not ledger_has_real_depts:
        return pd.Series(True, index=ledger_df.index)

    bank_dept_norm = str(bank_dept).strip().lower() if bank_dept not in (None, "") else "n/a"
    return ledger_dept_norm == bank_dept_norm


def match_by_reference(bank: pd.DataFrame, ledger: pd.DataFrame) -> list[dict]:
    matches = []
    ref_bank_rows = bank[bank["_ref_norm"] != ""].index.tolist()

    for idx in ref_bank_rows:
        if bank.loc[idx, "matched"]:
            continue
        b = bank.loc[idx]
        dept_mask = _same_department(b.get("department", "N/A"), ledger)

        # Filter the LIVE ledger dataframe fresh on every iteration — not a
        # pre-sliced copy — so a ledger row claimed by an earlier bank row
        # in this same loop (e.g. two duplicate-looking bank transactions
        # sharing a reference) can never be matched a second time.
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
            logger.info(
                "match_by_reference: bank row %s <-> ledger row %s (ref=%s, amount=%s)",
                b["source_row"], l["source_row"], b["_ref_norm"], b["amount"],
            )
            matches.append(
                {
                    "bank_row": b["source_row"],
                    "ledger_row": l["source_row"],
                    "date": b["date"],
                    "amount": b["amount"],
                    "bank_description": b["description"],
                    "ledger_description": l["description"],
                    "method": "exact_reference",
                    "confidence": 1.0,
                }
            )
        else:
            logger.debug(
                "match_by_reference: no candidate for bank row %s (ref=%s, amount=%s)",
                b["source_row"], b["_ref_norm"], b["amount_abs"],
            )
    return matches


def match_by_amount_and_date(bank: pd.DataFrame, ledger: pd.DataFrame) -> list[dict]:
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
            logger.info(
                "match_by_amount_and_date: bank row %s <-> ledger row %s (amount=%s, "
                "date_diff=%sd)",
                b["source_row"], l["source_row"], b["amount"],
                abs((l["date"] - b["date"]).days),
            )
            matches.append(
                {
                    "bank_row": b["source_row"],
                    "ledger_row": l["source_row"],
                    "date": b["date"],
                    "amount": b["amount"],
                    "bank_description": b["description"],
                    "ledger_description": l["description"],
                    "method": "exact_amount_date",
                    "confidence": 0.98,
                }
            )
        elif len(window) > 1:
            # Genuinely ambiguous (e.g. two identical payments same week) —
            # deliberately left for fuzzy/LLM/human review rather than
            # guessed here.
            logger.debug(
                "match_by_amount_and_date: %d ambiguous candidates for bank row %s "
                "(amount=%s) — leaving for fuzzy/LLM review.",
                len(window), b["source_row"], b["amount_abs"],
            )
    return matches


def run_exact_matching(
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    invert_credit_convention: bool | str = "auto",
) -> list[dict]:
    """
    invert_credit_convention:
      - "auto" (default): use CreditConventionResolver to detect whether
        the ledger's is_credit needs flipping to align with the bank's
        convention, based on sign-correlation of reference-matched pairs.
      - True / False: explicit override, skips detection entirely.
    """
    _prepare_references(bank, ledger)

    if ledger.attrs.get("_credit_convention_resolved"):
        # Already normalized by a previous call (e.g. shortlist_candidates
        # was run first) — do NOT invert again, that would flip it back.
        logger.info("run_exact_matching: ledger credit convention already resolved upstream; skipping re-detection.")
    else:
        if invert_credit_convention == "auto":
            should_invert = CreditConventionResolver().resolve(bank, ledger)
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