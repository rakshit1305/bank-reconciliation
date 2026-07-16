"""
categorizer.py
Two-pass description categorization, run independently of matching (it
reads bank['description_clean'], never touches matched/matched_id).

Pass 1 (free, instant): keyword rules against description_clean. Catches
the majority of real-world statements — salary, EMI, rent, POS, ATM, etc.

Pass 2 (LLM, batched): only the rows Pass 1 couldn't classify are sent to
an LLM, in batches (default 50/call) rather than one call per row, to keep
cost and latency down even on thousands of rows. Works with Groq, OpenAI,
or Anthropic via the same provider switch as llm_match.py.

RESILIENCE: rate-limit and auth error handling is shared with
llm_match.py (imported, not re-implemented) so the two modules can't
silently drift out of sync the way they did before. If the provider
becomes unavailable partway through Pass 2 (e.g. a daily token cap is
hit), the remaining unresolved rows fall back to 'OTHER' instead of
raising and crashing the whole reconciliation pipeline -- categorization
is a nice-to-have on top of matching, so a categorization failure should
never take matching down with it.
"""

import json
import re
import pandas as pd

from reconcile.llm_match import (
    LLMProviderUnavailable,
    _is_rate_limit_error,
    _is_auth_error,
    _parse_retry_after_seconds,
    MAX_RATE_LIMIT_RETRIES,
    MAX_RATE_LIMIT_WAIT_SECONDS,
)
import time

CATEGORIES = [
    "SALARY", "VENDOR_PAYMENT", "RENT", "UTILITIES", "LOAN_EMI",
    "INTEREST", "ATM_WITHDRAWAL", "POS_PURCHASE", "REFUND",
    "TRANSFER", "BANK_CHARGES", "GST_TAX", "OTHER",
]

# Pass 1 rules: (category, [regex patterns]). Order matters — first match
# wins, so more specific patterns should sit above generic ones.
_RULES = [
    ("SALARY", [r"\bsalary\b", r"\bpayroll\b"]),
    ("LOAN_EMI", [r"\bemi\b", r"\bloan repay", r"\becs\b"]),
    ("RENT", [r"\brent\b"]),
    ("UTILITIES", [r"electricity", r"water bill", r"internet bill", r"\bbroadband\b"]),
    ("INTEREST", [r"\binterest\b", r"\bint pd\b", r"\bint cr\b"]),
    ("ATM_WITHDRAWAL", [r"\batm\b", r"cash withdrawal"]),
    ("POS_PURCHASE", [r"\bpos\b", r"debit card"]),
    ("REFUND", [r"\brefund\b", r"\breversal\b"]),
    ("BANK_CHARGES", [r"service charge", r"sms alert", r"maint(enance)? charge", r"\bamc\b"]),
    ("GST_TAX", [r"\bgst\b", r"\btds\b", r"\bchallan\b"]),
    ("VENDOR_PAYMENT", [r"vendor", r"supplier", r"\binvoice\b", r"\binv\d"]),
    ("TRANSFER", [r"\bupi\b", r"\bimps\b", r"\bneft\b", r"\brtgs\b", r"fund transfer"]),
]
_COMPILED_RULES = [(cat, [re.compile(p) for p in pats]) for cat, pats in _RULES]


def categorize_rule_based(description_clean: str) -> str | None:
    """Returns a category string, or None if no rule matched (goes to Pass 2)."""
    for category, patterns in _COMPILED_RULES:
        if any(p.search(description_clean) for p in patterns):
            return category
    return None


def apply_pass1(df: pd.DataFrame, desc_col: str = "description_clean") -> pd.DataFrame:
    """Adds a 'category' column. Rows Pass 1 can't classify get 'UNRESOLVED'
    (a sentinel, not shown to the user) so Pass 2 knows what to pick up."""
    df = df.copy()
    df["category"] = df[desc_col].apply(lambda d: categorize_rule_based(d) or "UNRESOLVED")
    return df


# ---------------------------------------------------------------------------
# Pass 2 — LLM batches for leftovers
# ---------------------------------------------------------------------------

def _build_batch_prompt(descriptions: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
    cats = ", ".join(CATEGORIES)
    return f"""Classify each numbered bank transaction description into exactly
one of these categories: {cats}.

Respond with ONLY a JSON object mapping the number (as a string) to its
category, nothing else. Example: {{"1": "SALARY", "2": "OTHER"}}

Descriptions:
{numbered}"""


def _call_anthropic(prompt: str, model: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic to use the Anthropic provider")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _call_groq(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("pip install groq to use the Groq provider")
    client = Groq()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _call_openai(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai to use the OpenAI provider")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


_PROVIDER_CALLERS = {
    "groq": _call_groq,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}
_PROVIDER_DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}


def _call_provider_with_retry(prompt: str, provider: str, model: str, max_tokens: int) -> str:
    """Same retry-on-rate-limit / fail-fast-on-auth policy as
    llm_match.py, reusing its error-classification helpers directly so
    the two modules can't quietly diverge on what counts as
    retryable."""
    caller = _PROVIDER_CALLERS.get(provider)
    if caller is None:
        raise LLMProviderUnavailable(f"Unknown provider '{provider}'")

    attempt = 0
    while True:
        try:
            return caller(prompt, model, max_tokens)
        except Exception as e:
            if _is_auth_error(e):
                raise LLMProviderUnavailable(f"{provider} auth failed: {e}") from e

            if _is_rate_limit_error(e):
                attempt += 1
                wait = _parse_retry_after_seconds(e, default=5.0)
                if attempt > MAX_RATE_LIMIT_RETRIES or wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                    raise LLMProviderUnavailable(
                        f"{provider} rate limit hit and not recovering "
                        f"(after {attempt} attempt(s)): {e}"
                    ) from e
                time.sleep(wait)
                continue

            raise


def categorize_batch_llm(descriptions: list[str], provider: str = "groq", model: str = None) -> dict:
    """Returns {index(0-based): category}. Falls back to 'OTHER' for any
    row the LLM's response doesn't cover or that fails to parse.

    Raises LLMProviderUnavailable if the provider is down for this run
    (rate limit exhausted its retries, or auth failed) -- the caller
    (apply_pass2) is responsible for catching that and falling back
    gracefully rather than letting it propagate and crash the pipeline.
    """
    if not descriptions:
        return {}
    provider = provider or "groq"
    model = model or _PROVIDER_DEFAULT_MODELS.get(provider, "llama-3.1-8b-instant")
    prompt = _build_batch_prompt(descriptions)
    max_tokens = min(2000, 50 + 20 * len(descriptions))

    raw_text = _call_provider_with_retry(prompt, provider, model, max_tokens)
    raw_text = raw_text.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {i: "OTHER" for i in range(len(descriptions))}

    result = {}
    for i in range(len(descriptions)):
        cat = parsed.get(str(i + 1), "OTHER")
        result[i] = cat if cat in CATEGORIES else "OTHER"
    return result


def apply_pass2(df: pd.DataFrame, provider: str = "groq", batch_size: int = 50,
                 desc_col: str = "description") -> pd.DataFrame:
    """Sends only 'UNRESOLVED' rows to the LLM, in batches. Mutates and
    returns df with 'category' filled in for those rows too.

    Never raises: if the provider becomes unavailable partway through
    (e.g. a rate limit that didn't recover), every row from that point
    on falls back to 'OTHER' instead of being left 'UNRESOLVED' or
    crashing the pipeline. Rows already categorized by earlier batches
    keep their LLM-assigned category.
    """
    df = df.copy()
    unresolved_idx = df.index[df["category"] == "UNRESOLVED"].tolist()
    provider_down = False

    for start in range(0, len(unresolved_idx), batch_size):
        batch_idx = unresolved_idx[start:start + batch_size]

        if provider_down:
            for idx in batch_idx:
                df.loc[idx, "category"] = "OTHER"
            continue

        descs = df.loc[batch_idx, desc_col].tolist()
        try:
            results = categorize_batch_llm(descs, provider=provider)
        except LLMProviderUnavailable:
            provider_down = True
            for idx in batch_idx:
                df.loc[idx, "category"] = "OTHER"
            continue

        for pos, idx in enumerate(batch_idx):
            df.loc[idx, "category"] = results.get(pos, "OTHER")

    return df


def categorize(df: pd.DataFrame, use_llm: bool = False, provider: str = "groq",
                batch_size: int = 50) -> pd.DataFrame:
    """Full two-pass entry point. If use_llm is False, unresolved rows are
    simply labeled 'OTHER' instead of calling out to an LLM."""
    df = apply_pass1(df)
    if use_llm:
        df = apply_pass2(df, provider=provider, batch_size=batch_size)
    else:
        df.loc[df["category"] == "UNRESOLVED", "category"] = "OTHER"
    return df