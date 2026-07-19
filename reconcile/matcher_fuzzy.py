"""
matcher_fuzzy.py
For rows the exact matcher couldn't resolve, shortlist plausible
counterpart candidates using description similarity + amount proximity.
This keeps the LLM call cheap and focused: it only ever sees a handful
of pre-filtered candidates per unmatched row, never the whole ledger.
"""

import logging

import pandas as pd
from rapidfuzz import fuzz

from .matcher_exact import (
    resolve_credit_convention,
    _prepare_references,
    extract_reference,
    normalize_reference,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

AMOUNT_TOLERANCE_PCT = 0.02   # allow 2% variance (bank fees, rounding)
DATE_WINDOW_DAYS = 10         # wider than exact match window
MIN_DESC_SCORE = 45           # below this, don't even bother shortlisting
TOP_N_CANDIDATES = 3


def _normalize_dept(series: pd.Series) -> pd.Series:
    """Strip + lowercase for comparison purposes only — see the matching
    helper in matcher_exact.py for why this matters (an invisible
    trailing space or case difference between bank/ledger department
    tags otherwise zeroes out every match, not just some)."""
    return series.astype(str).str.strip().str.lower()


def _same_department(bank_dept, ledger_has_real_depts: bool, ledger_pool: pd.DataFrame) -> pd.Series:
    """Mirrors matcher_exact._same_department's logic (kept in sync
    deliberately). If the ledger has no real department split at all,
    compare against the whole pool. If it IS department-scoped, require
    a department match (whitespace/case-insensitive); an untagged bank
    row does NOT get treated as a wildcard."""
    if "department" not in ledger_pool.columns or not ledger_has_real_depts:
        return pd.Series(True, index=ledger_pool.index)
    bank_dept_norm = str(bank_dept).strip().lower() if bank_dept not in (None, "") else "n/a"
    return _normalize_dept(ledger_pool["department"]) == bank_dept_norm


def _ensure_credit_convention_resolved(
    bank: pd.DataFrame, ledger: pd.DataFrame, invert_credit_convention="auto"
) -> None:
    """Normalizes ledger.is_credit to the bank's convention, exactly like
    run_exact_matching does — but only if that hasn't already happened.
    Uses the same hybrid detection as matcher_exact.py (reference-
    correlation first, trial-based fallback when reference sampling is
    inconclusive — e.g. bank UTR numbers vs. a ledger's own internal
    voucher numbering, which share no digits at all and give the
    reference method literally nothing to sample). This lets
    shortlist_candidates() be called standalone while still being a
    no-op (not a double-invert) when it runs after run_exact_matching
    on the same dataframes."""
    if ledger.attrs.get("_credit_convention_resolved"):
        return

    if "_ref_norm" not in bank.columns or "_ref_norm" not in ledger.columns:
        _prepare_references(bank, ledger)

    if invert_credit_convention == "auto":
        should_invert = resolve_credit_convention(bank, ledger)
    else:
        should_invert = bool(invert_credit_convention)

    if should_invert:
        ledger["is_credit"] = ~ledger["is_credit"]
        logger.info("shortlist_candidates: inverted ledger.is_credit to align with bank convention.")

    ledger.attrs["_credit_convention_resolved"] = True


def _safe_desc_score(a, b) -> int:
    """rapidfuzz chokes on None/NaN and returns misleading scores on
    empty strings (an empty string can spuriously score high against
    another near-empty string). Treat any missing/blank description as
    zero similarity rather than letting it blow up or silently inflate
    a shortlist with junk matches."""
    if a is None or b is None:
        return 0
    if isinstance(a, float) and pd.isna(a):
        return 0
    if isinstance(b, float) and pd.isna(b):
        return 0
    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return 0
    return fuzz.token_sort_ratio(a, b)


def shortlist_candidates(
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    invert_credit_convention="auto",
) -> list:
    """Returns one entry per unmatched bank row, each with its top-N
    unmatched ledger candidates and a similarity score.

    Candidates are ranked primarily by description similarity, but when
    a candidate shares a reference number (UTR/cheque, extracted the same
    way as in matcher_exact) with the bank row, that's a much stronger
    signal than fuzzy text overlap and is used as a tie-breaker ahead of
    the description score."""
    _ensure_credit_convention_resolved(bank, ledger, invert_credit_convention)

    shortlist = []
    unmatched_ledger = ledger[~ledger["matched"]]

    ledger_has_real_depts = (
        "department" in ledger.columns and (_normalize_dept(ledger["department"]) != "n/a").any()
    )

    for _, b in bank[~bank["matched"]].iterrows():
        b_ref_norm = b.get("_ref_norm", "") or normalize_reference(b.get("reference", "")) or extract_reference(
            b.get("description", "")
        )

        lo = b["amount_abs"] * (1 - AMOUNT_TOLERANCE_PCT)
        hi = b["amount_abs"] * (1 + AMOUNT_TOLERANCE_PCT)

        pool = unmatched_ledger[
            (unmatched_ledger["amount_abs"] >= lo)
            & (unmatched_ledger["amount_abs"] <= hi)
            & (unmatched_ledger["is_credit"] == b["is_credit"])
            & (unmatched_ledger["date"].sub(b["date"]).abs().dt.days <= DATE_WINDOW_DAYS)
        ].copy()

        bank_dept = b.get("department", "N/A")
        pool = pool[_same_department(bank_dept, ledger_has_real_depts, pool)]

        if pool.empty:
            shortlist.append({"bank_row": b, "candidates": []})
            continue

        pool["desc_score"] = pool["description_clean"].apply(
            lambda d: _safe_desc_score(b["description_clean"], d)
        )
        pool = pool[pool["desc_score"] >= MIN_DESC_SCORE]

        if pool.empty:
            shortlist.append({"bank_row": b, "candidates": []})
            continue

        if "_ref_norm" in pool.columns and b_ref_norm:
            pool["_ref_match"] = pool["_ref_norm"] == b_ref_norm
        else:
            pool["_ref_match"] = False

        pool = pool.sort_values(["_ref_match", "desc_score"], ascending=[False, False]).head(TOP_N_CANDIDATES)

        candidates = [
            {
                "ledger_row": row["source_row"],
                "ledger_index": idx,
                "date": row["date"],
                "amount": row["amount"],
                "description": row["description"],
                "desc_score": row["desc_score"],
                "ref_match": bool(row.get("_ref_match", False)),
            }
            for idx, row in pool.iterrows()
        ]
        shortlist.append({"bank_row": b, "candidates": candidates})

    logger.info(
        "shortlist_candidates: shortlisted %d unmatched bank row(s), %d with at least one candidate.",
        len(shortlist), sum(1 for s in shortlist if s["candidates"]),
    )
    return shortlist


def auto_accept_high_confidence(shortlist: list, bank, ledger, threshold=90) -> tuple:
    """Optional pre-LLM pass: if the top candidate's description score is
    very high and it's the only candidate, accept it automatically instead
    of spending an LLM call on it. Everything else goes to the LLM/review.
    A reference match on the sole candidate also qualifies for auto-accept
    even if its desc_score happens to sit under threshold — a matching
    UTR/cheque number is a stronger signal than fuzzy text similarity."""
    matches = []
    remaining = []

    for item in shortlist:
        b = item["bank_row"]
        cands = item["candidates"]
        if len(cands) == 1 and (cands[0]["desc_score"] >= threshold or cands[0]["ref_match"]):
            c = cands[0]
            bank.loc[b.name, "matched"] = True
            ledger.loc[c["ledger_index"], "matched"] = True
            method = "fuzzy_auto_ref" if c["ref_match"] else "fuzzy_auto"
            confidence = 0.97 if c["ref_match"] else round(c["desc_score"] / 100, 2)
            logger.info(
                "auto_accept_high_confidence: bank row %s <-> ledger row %s (method=%s, "
                "desc_score=%s, ref_match=%s)",
                b["source_row"], c["ledger_row"], method, c["desc_score"], c["ref_match"],
            )
            matches.append(
                {
                    "bank_row": b["source_row"],
                    "ledger_row": c["ledger_row"],
                    "date": b["date"],
                    "amount": b["amount"],
                    "description": b["description"],
                    "bank_date": b["date"],
                    "ledger_date": c["date"],
                    "bank_amount": b["amount"],
                    "ledger_amount": c["amount"],
                    "bank_description": b["description"],
                    "ledger_description": c["description"],
                    "method": method,
                    "confidence": confidence,
                }
            )
        elif cands:
            remaining.append(item)
        else:
            remaining.append(item)

    logger.info(
        "auto_accept_high_confidence: %d auto-accepted, %d sent onward for fuzzy/LLM review.",
        len(matches), len(remaining),
    )
    return matches, remaining


def diagnose_shortlist(shortlist: list) -> dict:
    """For rows that made it to fuzzy matching but got zero or multiple
    candidates (rather than a clean auto-accept), returns
    {bank_index: short human-readable reason} explaining the fuzzy-stage
    outcome. Complements matcher_exact.diagnose_unmatched, which only
    covers rows that never made it this far at all."""
    reasons = {}
    for item in shortlist:
        b = item["bank_row"]
        cands = item["candidates"]
        idx = b.name

        if not cands:
            reasons[idx] = (
                "No ledger row within amount tolerance / date window / department "
                "scope — or none cleared the minimum description similarity score"
            )
        elif len(cands) > 1:
            top = cands[0]
            reasons[idx] = (
                f"{len(cands)} candidates matched amount/date, best description "
                f"similarity {top['desc_score']:.0f} — not a single clean best match, "
                f"needs LLM/manual review"
            )
        else:
            reasons[idx] = (
                f"Single candidate found, but description similarity "
                f"({cands[0]['desc_score']:.0f}) was below the auto-accept threshold"
            )
    return reasons