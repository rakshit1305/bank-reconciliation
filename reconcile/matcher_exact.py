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
    the magnitude happens to coincide.
  - matching department, when the LEDGER carries real department scoping
    (see _same_department below) — an untagged bank row is NOT treated
    as a wildcard against a department-scoped ledger, since that was
    proven to cause cross-department false positives (e.g. a Sales-only
    ledger silently "matching" HR/Finance transactions).
"""

import pandas as pd

DATE_WINDOW_DAYS = 3


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

    Two genuinely different situations, handled differently:

    1. The LEDGER itself carries no real department split (every row is
       'N/A' — e.g. one unified ledger file with no per-sheet/per-tag
       breakdown). There's nothing to scope against, so we fall back to
       comparing the bank row against the whole ledger.

    2. The LEDGER *is* department-scoped (e.g. it's a Sales-only extract,
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

    bank_dept_norm = str(bank_dept).strip().lower()
    return ledger_dept_norm == bank_dept_norm


def match_by_reference(bank: pd.DataFrame, ledger: pd.DataFrame) -> list[dict]:
    matches = []
    ref_bank_rows = bank[bank["reference"] != ""].index.tolist()

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
            & (ledger["reference"] != "")
            & (ledger["reference"] == b["reference"])
            & (ledger["amount_abs"] == b["amount_abs"])
            & (ledger["is_credit"] == b["is_credit"])
        ]
        if len(candidates) >= 1:
            l = candidates.iloc[0]
            bank.loc[idx, "matched"] = True
            ledger.loc[l.name, "matched"] = True
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
        # if len(window) > 1, it's genuinely ambiguous (e.g. two identical
        # payments same week) — deliberately left for fuzzy/LLM/human review
        # rather than guessed here.
    return matches


def run_exact_matching(bank: pd.DataFrame, ledger: pd.DataFrame) -> list[dict]:
    matches = match_by_reference(bank, ledger)
    matches += match_by_amount_and_date(bank, ledger)
    return matches
