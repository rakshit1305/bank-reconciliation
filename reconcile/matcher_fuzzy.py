"""
matcher_fuzzy.py
For rows the exact matcher couldn't resolve, shortlist plausible
counterpart candidates using description similarity + amount proximity.
This keeps the LLM call cheap and focused: it only ever sees a handful
of pre-filtered candidates per unmatched row, never the whole ledger.
"""

import pandas as pd
from rapidfuzz import fuzz

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
    bank_dept_norm = str(bank_dept).strip().lower()
    return _normalize_dept(ledger_pool["department"]) == bank_dept_norm


def shortlist_candidates(bank: pd.DataFrame, ledger: pd.DataFrame) -> list[dict]:
    """Returns one entry per unmatched bank row, each with its top-N
    unmatched ledger candidates and a similarity score."""
    shortlist = []
    unmatched_ledger = ledger[~ledger["matched"]]

    ledger_has_real_depts = (
        "department" in ledger.columns and (_normalize_dept(ledger["department"]) != "n/a").any()
    )

    for _, b in bank[~bank["matched"]].iterrows():
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
            lambda d: fuzz.token_sort_ratio(b["description_clean"], d)
        )
        pool = pool[pool["desc_score"] >= MIN_DESC_SCORE]
        pool = pool.sort_values("desc_score", ascending=False).head(TOP_N_CANDIDATES)

        candidates = [
            {
                "ledger_row": row["source_row"],
                "ledger_index": idx,
                "date": row["date"],
                "amount": row["amount"],
                "description": row["description"],
                "desc_score": row["desc_score"],
            }
            for idx, row in pool.iterrows()
        ]
        shortlist.append({"bank_row": b, "candidates": candidates})

    return shortlist


def auto_accept_high_confidence(shortlist: list[dict], bank, ledger, threshold=90) -> list[dict]:
    """Optional pre-LLM pass: if the top candidate's description score is
    very high and it's the only candidate, accept it automatically instead
    of spending an LLM call on it. Everything else goes to the LLM/review."""
    matches = []
    remaining = []

    for item in shortlist:
        b = item["bank_row"]
        cands = item["candidates"]
        if len(cands) == 1 and cands[0]["desc_score"] >= threshold:
            c = cands[0]
            bank.loc[b.name, "matched"] = True
            ledger.loc[c["ledger_index"], "matched"] = True
            matches.append(
                {
                    "bank_row": b["source_row"],
                    "ledger_row": c["ledger_row"],
                    "date": b["date"],
                    "amount": b["amount"],
                    "bank_description": b["description"],
                    "ledger_description": c["description"],
                    "method": "fuzzy_auto",
                    "confidence": round(c["desc_score"] / 100, 2),
                }
            )
        elif cands:
            remaining.append(item)
        else:
            remaining.append(item)

    return matches, remaining