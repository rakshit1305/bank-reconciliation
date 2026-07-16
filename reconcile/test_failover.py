import sys, json
sys.path.insert(0, "/home/claude")
import pandas as pd
from reconcile import llm_match as lm
lm.time.sleep = lambda s: None

def make_shortlist(n):
    rows = []
    for i in range(n):
        bank_row = pd.Series({
            "source_row": i + 2, "date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=i),
            "description": f"Payment {i}", "amount": 100.0 + i,
        }, name=i)
        candidates = [{
            "ledger_row": i + 2, "ledger_index": i,
            "date": bank_row["date"], "amount": bank_row["amount"],
            "description": f"Ledger Payment {i}",
        }]
        rows.append({"bank_row": bank_row, "candidates": candidates})
    return rows

def make_bank_ledger(n):
    return (pd.DataFrame({"matched": [False]*n}, index=range(n)),
            pd.DataFrame({"matched": [False]*n}, index=range(n)))


print("=== Test: groq dies after batch 1, should walk down to openai for the rest ===")
call_log = []

class FakeRateLimitError(Exception):
    status_code = 429
    def __str__(self): return "Rate limit reached. Please try again in 999s"  # exceeds cap -> dies immediately

def fake_groq(prompt, model, max_tokens):
    call_log.append("groq")
    raise FakeRateLimitError()

def fake_openai(prompt, model, max_tokens):
    call_log.append("openai")
    items = json.loads(prompt.split("Items:\n")[1].split("\n\nRespond")[0])
    results = [{"id": it["id"], "match_ledger_row": it["candidates"][0]["row"],
                "confidence": 0.9, "rationale": "matched via openai fallback"} for it in items]
    return json.dumps({"results": results})

lm._PROVIDER_CALLERS["groq"] = fake_groq
lm._PROVIDER_CALLERS["openai"] = fake_openai

# 9 rows, BATCH_SIZE=3 -> 3 batches
shortlist = make_shortlist(9)
bank, ledger = make_bank_ledger(9)
llm_log = []
matches = lm.run_llm_disambiguation(shortlist, bank, ledger, llm_log, provider=["groq", "openai"])

print("Call sequence:", call_log)
print(f"Matches found: {len(matches)} (expected 9 -- ALL rows recovered via fallback, none lost)")
print(f"llm_log entries: {len(llm_log)} (expected 9)")

# groq should be tried exactly once (dies immediately, permanently marked dead),
# then openai handles all 3 batches.
assert call_log[0] == "groq", "should try groq first"
assert call_log.count("groq") == 1, f"groq should only be tried once (then marked dead), got {call_log.count('groq')} attempts"
assert call_log.count("openai") == 3, f"openai should handle all 3 batches, got {call_log.count('openai')}"
assert len(matches) == 9, f"expected all 9 rows to be recovered via fallback, got {len(matches)}"

# Confirm the log correctly attributes which provider actually resolved each row
providers_used = set(r["provider"] for r in llm_log)
print("Providers recorded in llm_log:", providers_used)
assert providers_used == {"openai"}, f"expected all entries attributed to openai, got {providers_used}"

print("PASS -- zero rows lost, groq tried once then automatically failed over to openai\n")


print("=== Test: BOTH providers die -> graceful skip, no crash ===")
def fake_both_dead(prompt, model, max_tokens):
    raise FakeRateLimitError()

lm._PROVIDER_CALLERS["groq"] = fake_both_dead
lm._PROVIDER_CALLERS["openai"] = fake_both_dead

shortlist2 = make_shortlist(6)
bank2, ledger2 = make_bank_ledger(6)
llm_log2 = []
matches2 = lm.run_llm_disambiguation(shortlist2, bank2, ledger2, llm_log2, provider=["groq", "openai"])
print(f"Matches: {len(matches2)} (expected 0)")
print(f"llm_log entries: {len(llm_log2)} (expected 6, all skipped)")
skipped = [r for r in llm_log2 if "Skipped" in r["result"]["rationale"]]
print(f"Skipped: {len(skipped)}")
print("Sample rationale:", skipped[0]["result"]["rationale"])
assert len(matches2) == 0
assert len(skipped) == 6
print("PASS -- no exception raised even with a full fallback chain exhausted\n")


print("=== Test: backward compatibility -- single string provider still works ===")
def fake_groq_ok(prompt, model, max_tokens):
    items = json.loads(prompt.split("Items:\n")[1].split("\n\nRespond")[0])
    results = [{"id": it["id"], "match_ledger_row": it["candidates"][0]["row"],
                "confidence": 0.9, "rationale": "ok"} for it in items]
    return json.dumps({"results": results})
lm._PROVIDER_CALLERS["groq"] = fake_groq_ok

shortlist3 = make_shortlist(3)
bank3, ledger3 = make_bank_ledger(3)
llm_log3 = []
matches3 = lm.run_llm_disambiguation(shortlist3, bank3, ledger3, llm_log3, provider="groq")  # plain string, not a list
assert len(matches3) == 3
print("PASS -- single-string provider (old call signature) still works unchanged\n")

print("ALL TESTS PASSED")
