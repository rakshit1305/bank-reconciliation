"""
loader.py
Reads the bank statement (CSV) and the ledger extract (Excel) into
normalized pandas DataFrames with a consistent schema:
    date (datetime64), description (str), amount (float, +ve=credit/-ve=debit),
    reference (str), source_row (int, original row index for traceability)

Handles three common amount formats automatically:
    1. Single "Amount" column with +/- sign
    2. Separate "Credit" / "Debit" columns
    3. Separate "Deposit" / "Withdrawal" columns
"""

import io
import warnings
import pandas as pd

# Map your real column names here once you see the actual files.
# Keeping this as a single config block means format changes are a
# one-line fix, not a rewrite. Only used for the strict/fast path —
# load_flexible() below falls back to auto-detection if these don't match.
BANK_COLUMN_MAP = {
    "Date": "date",
    "Description": "description",
    "Amount": "amount",
    "Reference": "reference",
}

LEDGER_COLUMN_MAP = {
    "TxnDate": "date",
    "Narration": "description",
    "Amount": "amount",
    "RefNo": "reference",
}


def _parse_dates(series: pd.Series) -> pd.Series:
    """Tries both dayfirst=True (DD-MM-YYYY, common in India) and
    dayfirst=False (YYYY-MM-DD / MM-DD-YYYY) and keeps whichever parses
    more rows successfully. A single hardcoded dayfirst assumption silently
    corrupts whichever format doesn't match it — this avoids that."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
        parsed_yearfirst = pd.to_datetime(series, errors="coerce", dayfirst=False)

    if parsed_yearfirst.notna().sum() > parsed_dayfirst.notna().sum():
        return parsed_yearfirst
    return parsed_dayfirst


def _standardize(df: pd.DataFrame, column_map: dict, source_name: str) -> pd.DataFrame:
    missing = [c for c in column_map if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name}: expected column(s) {missing} not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.rename(columns=column_map)
    df = df[list(column_map.values())].copy()

    df["date"] = _parse_dates(df["date"])
    df["amount"] = _clean_numeric(df["amount"])
    df["description"] = df["description"].astype(str).str.strip()
    df["reference"] = df["reference"].fillna("").astype(str).str.strip().str.upper()
    df["reference"] = df["reference"].replace({"NAN": "", "NONE": "", "<NA>": ""})

    bad_rows = df[df["date"].isna() | df["amount"].isna()]
    if len(bad_rows):
        print(
            f"WARNING [{source_name}]: dropping {len(bad_rows)} row(s) with "
            f"unparseable date/amount."
        )
        df = df.dropna(subset=["date", "amount"])

    df = df.reset_index(drop=True)
    df["source_row"] = df.index + 2  # +2 to align with 1-indexed sheet row incl. header
    return df


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strips commas/currency symbols/whitespace and converts to float.
    Treats blank/'-'/'nan' cells as 0, not a parse failure — this matters
    for Credit/Debit and Deposit/Withdrawal columns, where one side is
    legitimately empty for every single row."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.strip()
        .replace({"": "0", "-": "0", "nan": "0", "NaN": "0", "None": "0"})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def load_bank_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return _standardize(raw, BANK_COLUMN_MAP, "Bank statement")


def load_ledger_excel(path: str, sheet_name=0) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name)
    return _standardize(raw, LEDGER_COLUMN_MAP, "Ledger")


# ---------------------------------------------------------------------------
# Flexible loading: accepts CSV *or* Excel for either file, handles all three
# amount-column styles, and falls back to keyword-based auto-detection if the
# incoming headers don't match the hardcoded BANK_COLUMN_MAP / LEDGER_COLUMN_MAP.
# Used by the Streamlit app so uploads aren't locked to one fixed format.
# ---------------------------------------------------------------------------

_AUTO_KEYWORDS = {
    "date": ["date", "txn date", "value date", "posting date", "transaction date"],
    "description": ["description", "narration", "particulars", "details", "remarks"],
    "amount": ["amount", "amt", "transaction amount"],
    "reference": ["reference", "ref", "ref no", "refno", "chq", "utr", "cheque", "transaction id", "txn id"],
    # Deposit/withdrawal and credit/debit are checked as PAIRS, not single columns
    "credit": ["credit", "cr amount", "cr amt", " cr"],
    "debit": ["debit", "dr amount", "dr amt", " dr"],
    "deposit": ["deposit", "paid in", "inflow"],
    "withdrawal": ["withdrawal", "withdrawl", "paid out", "outflow"],
    # 4th format: single unsigned Amount column + a separate DR/CR indicator flag
    "dr_cr_flag": ["dr/cr", "cr/dr", "dr cr", "debit/credit indicator"],
}


def list_sheets(file_obj_or_path) -> list:
    """Returns the sheet names of an Excel file — used to let the user pick
    a specific month/department tab, or combine all of them. Returns a
    single-item list (['Sheet1']) for CSVs, which have no concept of sheets."""
    name = getattr(file_obj_or_path, "name", None) or str(file_obj_or_path)
    if name.lower().endswith(".csv"):
        return ["(CSV — single sheet)"]
    xls = pd.ExcelFile(file_obj_or_path)
    return xls.sheet_names


def _decode_bytes(raw_bytes: bytes) -> str:
    """Tries utf-8 first (the common case), falls back to latin-1 and
    cp1252 for files exported by older/regional banking systems that
    don't use UTF-8. latin-1 never actually fails to decode (it maps
    every byte to a character), so it's used as a safety-net last resort."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode("latin-1", errors="replace")


def _detect_delimiter(sample_lines: list) -> str:
    """Picks comma or semicolon based on whichever appears more
    consistently across the first few non-empty lines — handles
    European-style semicolon-delimited exports."""
    comma_counts = [line.count(",") for line in sample_lines if line.strip()]
    semi_counts = [line.count(";") for line in sample_lines if line.strip()]
    if sum(semi_counts) > sum(comma_counts):
        return ";"
    return ","


def _find_header_row_index(lines: list, delimiter: str) -> int:
    """Scans the first ~20 lines for the row that actually looks like a
    column header (contains at least 2 recognizable field-name keywords),
    skipping past title/disclaimer/account-info rows that some banks put
    at the top of their exports."""
    all_keywords = [kw for kws in _AUTO_KEYWORDS.values() for kw in kws]
    for i, line in enumerate(lines[:20]):
        fields = [f.strip().lower() for f in line.split(delimiter)]
        hits = sum(1 for f in fields for kw in all_keywords if kw in f)
        if hits >= 2:
            return i
    return 0  # fall back to assuming the first row is the header


def _read_any(file_obj_or_path, sheet_name=0):
    """Reads a CSV or Excel file, from a path string or a file-like object
    (e.g. Streamlit's UploadedFile), based on its extension. Robust to:
    non-UTF8 encoding, semicolon delimiters, and junk title/disclaimer
    rows above the real header (common in real bank exports)."""
    name = getattr(file_obj_or_path, "name", None) or str(file_obj_or_path)

    if name.lower().endswith(".csv"):
        if hasattr(file_obj_or_path, "seek"):
            file_obj_or_path.seek(0)
            raw_bytes = file_obj_or_path.read()
            if isinstance(raw_bytes, str):
                raw_bytes = raw_bytes.encode("utf-8")
        else:
            with open(file_obj_or_path, "rb") as fh:
                raw_bytes = fh.read()

        if not raw_bytes.strip():
            raise ValueError("This file appears to be completely empty (0 bytes).")

        text = _decode_bytes(raw_bytes)
        lines = text.splitlines()
        non_empty_lines = [l for l in lines if l.strip()]
        if not non_empty_lines:
            raise ValueError("This file has no non-blank lines to read.")

        delimiter = _detect_delimiter(non_empty_lines[:10])
        header_idx = _find_header_row_index(non_empty_lines, delimiter)

        try:
            df = pd.read_csv(
                io.StringIO(text), sep=delimiter, skiprows=header_idx,
                engine="python", on_bad_lines="skip",
            )
        except Exception as e:
            raise ValueError(f"Could not parse this CSV file even after cleanup attempts: {e}")

        df = df.dropna(how="all")  # drop fully-blank rows (mid-file gaps, trailing blank lines)
        if df.empty:
            raise ValueError("No data rows found — the file only contains headers (or blank rows).")
        return df

    # Excel path
    if hasattr(file_obj_or_path, "seek"):
        file_obj_or_path.seek(0)
    raw_preview = pd.read_excel(file_obj_or_path, sheet_name=sheet_name, header=None, nrows=20)
    preview_lines = [",".join(str(v) for v in row if pd.notna(v)) for row in raw_preview.values]
    header_idx = _find_header_row_index(preview_lines, ",")

    if hasattr(file_obj_or_path, "seek"):
        file_obj_or_path.seek(0)
    df = pd.read_excel(file_obj_or_path, sheet_name=sheet_name, header=header_idx)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No data rows found — the sheet only contains headers (or blank rows).")
    return df


def _find_column(lower_cols: dict, keywords: list):
    for original_col, lower_col in lower_cols.items():
        if any(kw in lower_col for kw in keywords):
            return original_col
    return None


DR_VALUES = {"dr", "d", "debit", "withdrawal"}
CR_VALUES = {"cr", "c", "credit", "deposit"}


def _find_dr_cr_column(df: pd.DataFrame, search_cols: dict):
    """Finds a DR/CR indicator column two ways: by header keyword match,
    or (fallback) by scanning for any low-cardinality column whose actual
    values are drawn from {DR, CR, Debit, Credit, D, C} — catches files
    where the column is named something we didn't anticipate, e.g. just
    'Type'."""
    col = _find_column(search_cols, _AUTO_KEYWORDS["dr_cr_flag"])
    if col:
        return col

    for original_col in search_cols:
        try:
            unique_vals = set(df[original_col].dropna().astype(str).str.strip().str.lower().unique())
        except Exception:
            continue
        if 0 < len(unique_vals) <= 4 and unique_vals.issubset(DR_VALUES | CR_VALUES):
            return original_col
    return None


def _combine_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detects which of the 4 amount styles this file uses and produces a
    single unified 'amount' column (+ve = money in, -ve = money out).
    Priority: Credit/Debit pair > Deposit/Withdrawal pair >
    Amount+DR/CR flag > single signed Amount.
    The date column is identified and excluded first, so a column like
    'Value Date' can never be mistaken for an amount column."""
    df = df.copy()
    lower_cols = {c: str(c).strip().lower() for c in df.columns}

    date_col = _find_column(lower_cols, _AUTO_KEYWORDS["date"])
    search_cols = {c: v for c, v in lower_cols.items() if c != date_col}

    credit_col = _find_column(search_cols, _AUTO_KEYWORDS["credit"])
    debit_col = _find_column(search_cols, _AUTO_KEYWORDS["debit"])
    deposit_col = _find_column(search_cols, _AUTO_KEYWORDS["deposit"])
    withdrawal_col = _find_column(search_cols, _AUTO_KEYWORDS["withdrawal"])
    amount_col = _find_column(search_cols, _AUTO_KEYWORDS["amount"])

    if credit_col and debit_col:
        df["__amount__"] = _clean_numeric(df[credit_col]) - _clean_numeric(df[debit_col])
        detected_style = f"Credit/Debit columns ('{credit_col}' / '{debit_col}')"
    elif deposit_col and withdrawal_col:
        df["__amount__"] = _clean_numeric(df[deposit_col]) - _clean_numeric(df[withdrawal_col])
        detected_style = f"Deposit/Withdrawal columns ('{deposit_col}' / '{withdrawal_col}')"
    elif amount_col:
        amount_search_cols = {c: v for c, v in search_cols.items() if c != amount_col}
        flag_col = _find_dr_cr_column(df, amount_search_cols)
        magnitude = _clean_numeric(df[amount_col])

        if flag_col and (magnitude < 0).sum() == 0:
            # Unsigned Amount + separate DR/CR indicator — very easy to get
            # wrong (an unsigned amount looks like "single Amount column"
            # on its own), so this is only trusted when the amount column
            # has NO negative values at all — a genuinely signed Amount
            # column would already encode direction and shouldn't be
            # reinterpreted through a coincidental flag-like column.
            flag_vals = df[flag_col].astype(str).str.strip().str.lower()
            sign = flag_vals.map(lambda v: -1 if v in DR_VALUES else (1 if v in CR_VALUES else 0))
            df["__amount__"] = magnitude * sign
            detected_style = f"Amount + DR/CR indicator ('{amount_col}' / '{flag_col}')"
        else:
            df["__amount__"] = magnitude
            detected_style = f"Single Amount column ('{amount_col}', +/- sign)"
    else:
        raise ValueError(
            "Could not detect an amount format. Expected either: a single "
            "'Amount' column (+/-), a 'Credit'+'Debit' pair, a "
            "'Deposit'+'Withdrawal' pair, or an Amount column with a "
            "separate DR/CR indicator column. "
            f"Available columns: {list(df.columns)}"
        )

    df.attrs["detected_amount_style"] = detected_style
    return df


def _auto_map_columns(df: pd.DataFrame) -> dict:
    """Best-effort guess of which column is date/description/reference,
    based on header keywords. Amount is handled separately by
    _combine_amount_columns since it may be 1 or 2 source columns."""
    found = {}
    lower_cols = {c: str(c).strip().lower() for c in df.columns}

    for target in ("date", "description", "reference"):
        col = _find_column(lower_cols, _AUTO_KEYWORDS[target])
        if col:
            found[target] = col

    missing = [k for k in ("date", "description") if k not in found]
    if missing:
        raise ValueError(
            f"Could not auto-detect column(s) for: {missing}. "
            f"Available columns: {list(df.columns)}. "
            f"Please rename headers or use the manual column mapping."
        )
    return found


def load_flexible(file_obj_or_path, source_name: str, preferred_map: dict = None, sheet_name=0) -> pd.DataFrame:
    """Loads bank or ledger data from CSV/Excel, in any of the 3 supported
    amount formats, trying the preferred strict column map first, then
    falling back to keyword auto-detection. Returns a DataFrame with
    df.attrs['detected_amount_style'] set, so the UI can show the user what
    format was detected."""
    raw = _read_any(file_obj_or_path, sheet_name=sheet_name)

    # Fast path: exact expected single-Amount format
    if preferred_map and all(c in raw.columns for c in preferred_map):
        result = _standardize(raw, preferred_map, source_name)
        result.attrs["detected_amount_style"] = f"Single Amount column ('{preferred_map.get('Amount', 'Amount')}', matched expected schema)"
        return result

    # Combine whichever amount format is present into one 'amount' column
    raw = _combine_amount_columns(raw)
    style = raw.attrs.get("detected_amount_style", "unknown")

    # Auto-detect date/description/reference columns
    field_map = _auto_map_columns(raw)
    rename_map = {v: k for k, v in field_map.items()}
    rename_map["__amount__"] = "amount"

    if "reference" not in field_map:
        raw["__no_reference__"] = ""
        rename_map["__no_reference__"] = "reference"

    result = _standardize(raw, rename_map, source_name)
    result.attrs["detected_amount_style"] = style
    return result


def load_source_auto(file_obj_or_path, source_name: str, preferred_map: dict = None) -> pd.DataFrame:
    """Top-level loader used by the dashboard. If the file is a CSV, or a
    single-sheet Excel file, this behaves exactly like load_flexible() and
    tags every row with department='N/A'.

    If the file is a MULTI-SHEET Excel workbook (e.g. one tab per
    department — Sales_Dept, HR_Dept, Finance_Dept...), every sheet is
    loaded independently (each may be in a *different* amount format —
    Credit/Debit vs Deposit/Withdrawal vs single Amount — that's fine,
    each sheet is auto-detected on its own) and stacked into one
    DataFrame with a 'department' column set to the sheet name. This is
    what powers the per-department tabs in the dashboard."""
    sheets = list_sheets(file_obj_or_path)

    if len(sheets) <= 1:
        df = load_flexible(file_obj_or_path, source_name, preferred_map)
        df["department"] = "N/A"
        return df

    frames = []
    detected_styles = {}
    for sheet in sheets:
        try:
            df_sheet = load_flexible(file_obj_or_path, f"{source_name} [{sheet}]", None, sheet_name=sheet)
        except ValueError as e:
            print(f"WARNING: skipping sheet '{sheet}' in {source_name} — {e}")
            continue
        df_sheet["department"] = sheet
        detected_styles[sheet] = df_sheet.attrs.get("detected_amount_style", "unknown")
        frames.append(df_sheet)

    if not frames:
        raise ValueError(f"{source_name}: none of the {len(sheets)} sheet(s) could be parsed.")

    combined = pd.concat(frames, ignore_index=True)
    combined["source_row"] = combined.index + 2
    combined.attrs["detected_amount_style"] = "; ".join(f"{k}: {v}" for k, v in detected_styles.items())
    combined.attrs["sheet_styles"] = detected_styles
    return combined


def check_same_file(bank_file_obj_or_path, ledger_file_obj_or_path) -> dict:
    """Catches the common accidental-upload mistake: the same file (or a
    byte-identical copy) uploaded as both bank statement and ledger.
    Returns {'identical': bool, 'reason': str}. Uses a content hash, not
    filename, so it catches renamed duplicates too."""
    import hashlib

    def _get_bytes(f):
        if hasattr(f, "seek"):
            f.seek(0)
            data = f.read()
            f.seek(0)
            return data if isinstance(data, bytes) else data.encode("utf-8")
        with open(f, "rb") as fh:
            return fh.read()

    try:
        bank_hash = hashlib.md5(_get_bytes(bank_file_obj_or_path)).hexdigest()
        ledger_hash = hashlib.md5(_get_bytes(ledger_file_obj_or_path)).hexdigest()
    except Exception:
        return {"identical": False, "reason": ""}

    if bank_hash == ledger_hash:
        return {
            "identical": True,
            "reason": "The bank statement and ledger files are byte-for-byte identical. "
                      "This is almost always an accidental upload of the same file twice "
                      "rather than a genuine bank + ledger pair.",
        }
    return {"identical": False, "reason": ""}
