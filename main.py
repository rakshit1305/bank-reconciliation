"""
main.py
Entry point for the Bank Reconciliation Agent.

Usage:
    python main.py --bank sample_data/bank_statement.csv \
                    --ledger sample_data/ledger.xlsx \
                    --output output/reconciled.xlsx

Pipeline:
    1. Load + normalize both sources
    2. Exact matching (reference, then amount+date window)
    3. Fuzzy shortlist + high-confidence auto-accept
    4. LLM disambiguation for whatever's still unresolved (optional)
    5. Write formatted Excel output with full audit trail
"""

import argparse
import os
from dotenv import load_dotenv

from reconcile.loader import load_bank_csv, load_ledger_excel
from reconcile.cleaner import normalize, deduplicate
from reconcile.matcher_exact import run_exact_matching
from reconcile.matcher_fuzzy import shortlist_candidates, auto_accept_high_confidence
from reconcile.llm_match import llm_available, run_llm_disambiguation
from reconcile.writer import write_output

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Bank Reconciliation Agent")
    parser.add_argument("--bank", required=True, help="Path to bank statement CSV")
    parser.add_argument("--ledger", required=True, help="Path to ledger Excel file")
    parser.add_argument("--output", required=True, help="Path to write reconciled Excel output")
    parser.add_argument("--no-llm", action="store_true", help="Skip the LLM disambiguation step")
    args = parser.parse_args()

    print(f"Loading bank statement: {args.bank}")
    bank = normalize(load_bank_csv(args.bank))
    bank = deduplicate(bank, "Bank statement")

    print(f"Loading ledger: {args.ledger}")
    ledger = normalize(load_ledger_excel(args.ledger))
    ledger = deduplicate(ledger, "Ledger")

    print(f"Bank rows: {len(bank)} | Ledger rows: {len(ledger)}")

    print("Running exact matching...")
    matches = run_exact_matching(bank, ledger)
    print(f"  -> {len(matches)} exact match(es)")

    print("Running fuzzy shortlisting...")
    shortlist = shortlist_candidates(bank, ledger)
    fuzzy_matches, remaining = auto_accept_high_confidence(shortlist, bank, ledger)
    matches += fuzzy_matches
    print(f"  -> {len(fuzzy_matches)} high-confidence fuzzy match(es), "
          f"{len(remaining)} row(s) still ambiguous")

    llm_log = []
    if not args.no_llm and llm_available():
        print("Running LLM disambiguation on remaining ambiguous rows...")
        llm_matches = run_llm_disambiguation(remaining, bank, ledger, llm_log)
        matches += llm_matches
        print(f"  -> {len(llm_matches)} match(es) confirmed by LLM "
              f"(logged {len(llm_log)} LLM call(s))")
    elif not args.no_llm:
        print("  (ANTHROPIC_API_KEY not set — skipping LLM step; "
              "remaining rows go to Needs_Review)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_output(args.output, matches, bank, ledger, llm_log)

    unmatched_bank = (~bank["matched"]).sum()
    unmatched_ledger = (~ledger["matched"]).sum()
    print(f"\nDone. {len(matches)} total matched pair(s). "
          f"{unmatched_bank} bank row(s) and {unmatched_ledger} ledger row(s) still unmatched.")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
