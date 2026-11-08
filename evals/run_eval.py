"""Run the labeled eval set through the pipeline and print a scorecard.

    python -m evals.run_eval                 # offline baseline, no API key needed
    python -m evals.run_eval --mode live     # Gemini extraction + hybrid RAG + LLM auditor
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from invoice_intelligence.config import get_api_key
from invoice_intelligence.evaluation import load_dataset, score_case, summarize
from invoice_intelligence.pipeline import InvoicePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    parser.add_argument("--output", type=Path, help="Also write a full per-case JSON report to this path.")
    parser.add_argument("--min-tier-accuracy", type=float, default=0.0,
                        help="Exit non-zero if tier accuracy falls below this (for CI regression gates).")
    args = parser.parse_args()

    client = None
    if args.mode == "live":
        load_dotenv()
        from google import genai

        key = get_api_key()
        if not key:
            print("GOOGLE_API_KEY is not set.", file=sys.stderr)
            return 2
        client = genai.Client(api_key=key)

    pipeline = InvoicePipeline(client=client, top_k=args.top_k)
    scores = []
    for case in load_dataset():
        result = pipeline.run(case["document"], mode=args.mode)
        scores.append(score_case(case, result))

    summary = summarize(scores)
    if args.output:
        report = {"mode": args.mode, "top_k": args.top_k, "summary": summary,
                  "cases": [{**asdict(s), "tier_correct": s.tier_correct} for s in scores]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\nEval mode: {args.mode}\n")
        print(f"{'case':<28}{'tier (exp→pred)':<26}{'fields':<10}{'policy TP/FP/FN':<18}misses")
        for s in scores:
            mark = "✓" if s.tier_correct else "✗"
            tier = f"{mark} {s.expected_tier}→{s.predicted_tier}"
            print(f"{s.id:<28}{tier:<26}{s.field_hits}/{s.field_total:<8}"
                  f"{s.policy_tp}/{s.policy_fp}/{s.policy_fn:<14}{', '.join(s.field_misses)}")
        print()
        for key, value in summary.items():
            print(f"{key:<18}{value:.3f}" if isinstance(value, float) else f"{key:<18}{value}")

    return 0 if summary["tier_accuracy"] >= args.min_tier_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
