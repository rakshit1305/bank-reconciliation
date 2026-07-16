"""
llm_match.py
Optional judgment layer. Only called for rows the exact matcher and the
high-confidence fuzzy pass could NOT resolve. Sends a small, structured
prompt (never the raw files) and requires structured JSON back, so the
result can be parsed deterministically and is always human-reviewable.

Supports three providers, selectable independently of each other (or via
LLM_PROVIDER env var):
  - "groq"       — needs GROQ_API_KEY, free tier, fast Llama models.
  - "openai"     — needs OPENAI_API_KEY, GPT-4o-mini, very cheap.
  - "anthropic"  — needs ANTHROPIC_API_KEY, Claude, paid (a few cents
                    per run typically).
The prompt contract (input shape -> required JSON output shape) is
identical across all three, so nothing downstream needs to know which
was used.

If no key is set for any of them, this module is skipped entirely and
every remaining row simply falls through to "Needs Review" for a human.

PERFORMANCE + RESILIENCE NOTES:
  - Rows are sent to the LLM in BATCHES (BATCH_SIZE per call, not one row
    per call). This cuts the number of API requests roughly BATCH_SIZE-x,
    which both speeds up a run and burns far fewer requests/tokens
    against provider rate limits for the same amount of work.
  - Rate-limit (429) errors get a real retry: the provider's own
    "try again in Xs" message is parsed and waited out (bounded so this
    can't hang the app for minutes), then the SAME batch is retried a
    couple of times before giving up on it.
  - Auth errors (bad/missing key) are NOT retried — retrying can't fix
    those, so they fail fast.
  - If a batch still can't get through after retries, the whole run
    doesn't crash: that batch's rows are marked as "skipped" and, since
    further calls would almost certainly hit the same wall, remaining
    batches are skipped too rather than repeating the same wait. Batches
    already completed keep their results.
"""

import os
import re
import json
import time

from dotenv import load_dotenv
# Loads .env first (if present), then groq.env as a fallback/override —
# so GROQ_API_KEY works whichever file you keep it in.
load_dotenv()
load_dotenv("groq.env", override=False)

CONFIDENCE_AUTO_ACCEPT = 0.85

GROQ_MODEL = "llama-3.1-8b-instant"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o-mini"

BATCH_SIZE = 3                     # rows sent per API call
MAX_RATE_LIMIT_RETRIES = 2         # retries per batch on a 429 before giving up
MAX_RATE_LIMIT_WAIT_SECONDS = 90   # never block the app longer than this per wait


class LLMProviderUnavailable(Exception):
    """Raised when the provider itself is failing in a way where retrying
    is pointless (bad auth) or has already been retried without success
    (rate limit exhausted its retry budget) -- the caller should stop
    sending further batches for the rest of this run."""
    pass


def llm_available(provider: str = None) -> bool:
    """With no provider given, returns True if ANY provider's key is
    present — used by the sidebar to decide whether to show the LLM
    option at all."""
    provider = provider or os.environ.get("LLM_PROVIDER")
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    return (
        bool(os.environ.get("GROQ_API_KEY"))
        or bool(os.environ.get("OPENAI_API_KEY"))
        or bool(os.environ.get("ANTHROPIC_API_KEY"))
    )


def default_provider() -> str:
    """Picks Groq first (free tier) if available, then OpenAI (very
    cheap), then Anthropic (paid) -- cost-driven ordering, free first
    then cheapest paid option. Override by passing provider= explicitly
    or setting LLM_PROVIDER."""
    env_choice = os.environ.get("LLM_PROVIDER")
    if env_choice in ("anthropic", "groq", "openai"):
        return env_choice
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "groq"


def _build_batch_prompt(items: list[dict]) -> str:
    """items: list of {'id': int, 'bank_transaction': {...}, 'candidates': [...]}"""
    return f"""You are assisting a bank reconciliation review. Below is a list of items.
Each item has ONE bank transaction that could not be matched automatically,
and a short list of candidate ledger entries. For EACH item, decide which
candidate (if any) is the true match, based on description, amount, and date.

Items:
{json.dumps(items, default=str)}

Respond with ONLY valid JSON, no other text, in exactly this shape:
{{"results": [
  {{"id": <the item's id>, "match_ledger_row": <row number or null>, "confidence": <0.0-1.0>, "rationale": "<one short sentence>"}},
  ...
]}}
Include exactly one result per item, in any order, using each item's "id" to
identify it. If none of an item's candidates are a plausible match, set that
item's match_ledger_row to null."""


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("rate limit", "rate_limit", "quota", "insufficient_quota"))


def _is_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("unauthorized", "invalid api key", "authentication"))


def _parse_retry_after_seconds(exc: Exception, default: float = 5.0) -> float:
    """Prefers a Retry-After response header if the SDK exposes one;
    falls back to parsing text like 'try again in 1m18.624s' out of the
    error message, since that's what Groq's 429 body includes."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            header_val = resp.headers.get("retry-after")
            if header_val:
                return float(header_val)
        except Exception:
            pass

    msg = str(exc)
    m = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", msg, re.IGNORECASE)
    if m:
        minutes = float(m.group(1)) if m.group(1) else 0.0
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    return default


def _call_anthropic(prompt: str, model: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic to use the Anthropic provider")
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_groq(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("pip install groq to use the Groq provider")
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _call_openai(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai to use the OpenAI provider")
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


_PROVIDER_CALLERS = {
    "groq": _call_groq,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}

_PROVIDER_DEFAULT_MODELS = {
    "groq": GROQ_MODEL,
    "openai": OPENAI_MODEL,
    "anthropic": ANTHROPIC_MODEL,
}


def _call_provider_with_retry(prompt: str, provider: str, model: str, max_tokens: int) -> str:
    """Runs the underlying API call, retrying on rate limits (bounded,
    using the provider's own suggested wait time) and failing fast on
    auth errors. Raises LLMProviderUnavailable once retries are
    exhausted or the error isn't retryable."""
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

            # Not a rate limit or auth issue — likely a one-off / transient
            # failure. Don't retry indefinitely; surface it to the caller
            # to handle at the batch level instead.
            raise


def _ask_llm_for_batch(items: list[dict], provider: str = None, model: str = None) -> dict:
    """items: list of {'id', 'bank_transaction', 'candidates'}.
    Returns {id: {"match_ledger_row", "confidence", "rationale"}, ...}.
    Raises LLMProviderUnavailable if the provider is down for this run.
    A malformed/unparseable response (not an account-level issue) is
    caught here and turned into "no match" results for every item in
    the batch, rather than propagating and aborting the whole run.
    """
    provider = provider or default_provider()
    model = model or _PROVIDER_DEFAULT_MODELS.get(provider, GROQ_MODEL)
    prompt = _build_batch_prompt(items)
    max_tokens = min(4000, 200 + 150 * len(items))

    text = _call_provider_with_retry(prompt, provider, model, max_tokens)

    text = text.strip().replace("```json", "").replace("```", "").strip()
    fallback = {
        it["id"]: {"match_ledger_row": None, "confidence": 0.0, "rationale": "LLM returned unparseable output"}
        for it in items
    }
    try:
        parsed = json.loads(text)
        results_list = parsed.get("results", parsed if isinstance(parsed, list) else [])
    except (json.JSONDecodeError, AttributeError):
        return fallback

    out = dict(fallback)  # start from "no match" defaults, overwrite with what we got
    for r in results_list:
        try:
            out[r["id"]] = {
                "match_ledger_row": r.get("match_ledger_row"),
                "confidence": r.get("confidence", 0.0),
                "rationale": r.get("rationale", ""),
            }
        except (KeyError, TypeError):
            continue
    return out


def ask_llm_for_match(bank_row: dict, candidates: list[dict], provider: str = None, model: str = None) -> dict:
    """Single-row convenience wrapper around the batch call, kept for
    backward compatibility / direct use. Prefer run_llm_disambiguation
    for bulk work, since it batches automatically."""
    batch_result = _ask_llm_for_batch(
        [{"id": 0, "bank_transaction": bank_row, "candidates": candidates}],
        provider=provider, model=model,
    )
    return batch_result[0]


def run_llm_disambiguation(remaining_shortlist: list[dict], bank, ledger, llm_log: list, provider=None) -> list[dict]:
    """Runs the LLM over each remaining ambiguous row, in batches of
    BATCH_SIZE. Every row is logged (input + output) to llm_log for
    audit, regardless of the outcome.

    `provider` accepts three shapes:
      - None            -> uses [default_provider()]
      - a single string -> uses [that provider] (old behavior, unchanged)
      - a list/tuple    -> a FALLBACK CHAIN, tried in order. If the
        current provider becomes unavailable mid-run (rate limit
        exhausted its retries, or auth failed), the SAME batch is
        immediately retried on the next provider in the chain instead
        of being skipped -- and every subsequent batch goes straight to
        that provider too (no wasted retries on a provider already
        known to be dead). Only once every provider in the chain has
        failed do the remaining rows fall back to "needs review".

    Never raises: even if every provider in the chain is unavailable,
    the remaining rows are marked as skipped in llm_log and the
    function returns normally with whatever matches it found up to
    that point.
    """
    if provider is None:
        providers = [default_provider()]
    elif isinstance(provider, str):
        providers = [provider]
    else:
        providers = list(provider)

    matches = []
    active_idx = 0          # index into `providers` currently in use
    dead_providers = {}     # {provider_name: reason} -- confirmed unavailable this run

    # Only keep rows that actually have candidates -- nothing to ask the
    # LLM about otherwise.
    workable = [item for item in remaining_shortlist if item["candidates"]]

    for batch_start in range(0, len(workable), BATCH_SIZE):
        batch_items = workable[batch_start: batch_start + BATCH_SIZE]

        payloads = []
        for i, item in enumerate(batch_items):
            b = item["bank_row"]
            cands = item["candidates"]
            bank_row_payload = {
                "row": int(b["source_row"]),
                "date": str(b["date"].date()),
                "description": b["description"],
                "amount": float(b["amount"]),
            }
            cand_payload = [
                {
                    "row": int(c["ledger_row"]),
                    "date": str(c["date"].date()),
                    "description": c["description"],
                    "amount": float(c["amount"]),
                }
                for c in cands
            ]
            payloads.append({
                "id": i, "item": item,
                "bank_row_payload": bank_row_payload, "cand_payload": cand_payload,
            })

        batch_result = None
        used_provider = None

        # Walk down the fallback chain for THIS batch, starting from
        # whichever provider is currently active (skips providers
        # already confirmed dead from earlier batches).
        while active_idx < len(providers):
            current_provider = providers[active_idx]
            try:
                batch_result = _ask_llm_for_batch(
                    [{"id": p["id"], "bank_transaction": p["bank_row_payload"], "candidates": p["cand_payload"]} for p in payloads],
                    provider=current_provider,
                )
                used_provider = current_provider
                break
            except LLMProviderUnavailable as e:
                dead_providers[current_provider] = str(e)
                active_idx += 1  # permanently move past this provider for all future batches too

        if batch_result is None:
            # Every provider in the chain is now dead -- this batch,
            # and every batch after it, gets skipped without further
            # API calls.
            reason = "; ".join(f"{p}: {r}" for p, r in dead_providers.items())
            for p in payloads:
                llm_log.append({
                    "bank_row": p["bank_row_payload"], "candidates": p["cand_payload"],
                    "result": {"match_ledger_row": None, "confidence": 0.0,
                               "rationale": f"Skipped — all providers unavailable ({reason})"},
                    "provider": None,
                })
            continue

        for p in payloads:
            b = p["item"]["bank_row"]
            cands = p["item"]["candidates"]
            result = batch_result.get(p["id"], {"match_ledger_row": None, "confidence": 0.0, "rationale": "No result returned for this row"})
            llm_log.append({"bank_row": p["bank_row_payload"], "candidates": p["cand_payload"], "result": result, "provider": used_provider})

            if result.get("match_ledger_row") is not None and result.get("confidence", 0) >= CONFIDENCE_AUTO_ACCEPT:
                matched_cand = next(
                    (c for c in cands if c["ledger_row"] == result["match_ledger_row"]), None
                )
                if matched_cand:
                    bank.loc[b.name, "matched"] = True
                    ledger.loc[matched_cand["ledger_index"], "matched"] = True
                    matches.append(
                        {
                            "bank_row": b["source_row"],
                            "ledger_row": matched_cand["ledger_row"],
                            # Kept for backward compatibility with any code
                            # still reading the old shared keys (defaults
                            # to the bank side).
                            "date": b["date"],
                            "amount": b["amount"],
                            "description": b["description"],
                            # Both-sides columns.
                            "bank_date": b["date"],
                            "ledger_date": matched_cand["date"],
                            "bank_amount": b["amount"],
                            "ledger_amount": matched_cand["amount"],
                            "bank_description": b["description"],
                            "ledger_description": matched_cand["description"],
                            "method": "llm_match",
                            "confidence": result["confidence"],
                            "rationale": result.get("rationale", ""),
                        }
                    )

    return matches