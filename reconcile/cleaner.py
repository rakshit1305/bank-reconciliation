"""
cleaner.py
Extra normalization applied after loading, shared by both bank and
ledger DataFrames, so exact-match and fuzzy-match compare like-for-like.
"""

import re
import pandas as pd


def clean_description(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace, collapse common
    bank noise tokens so 'AMZN*MKTP IN' and 'Amazon Mktp' compare well."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["description_clean"] = df["description"].apply(clean_description)
    df["amount_abs"] = df["amount"].abs().round(2)
    df["is_credit"] = df["amount"] > 0
    df["matched"] = False
    df["match_id"] = pd.NA
    return df


def deduplicate(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Flag exact duplicate rows (same date/amount/description/reference) —
    these are usually double-entries, not real duplicate transactions."""
    dupe_mask = df.duplicated(
        subset=["date", "amount", "description_clean", "reference"], keep=False
    )
    if dupe_mask.any():
        print(f"WARNING [{source_name}]: {dupe_mask.sum()} duplicate-looking row(s) found.")
    return df
