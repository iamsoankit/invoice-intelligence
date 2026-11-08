"""Command-line entry point for the invoice audit pipeline.

    python main.py path/to/invoice.txt            # offline (no API key)
    python main.py path/to/invoice.pdf --live     # Gemini multimodal + hybrid RAG
    python main.py --sample bec-routing-sms       # run a labeled sample
"""

import argparse
import json
import mimetypes
import sys
from pathlib import Path

from dotenv import load_dotenv

from invoice_intelligence.config import get_api_key
from invoice_intelligence.evaluation import load_dataset
from invoice_intelligence.pipeline import InvoicePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("path", nargs="?", type=Path, help="Invoice file (.txt, .pdf, .png, .jpg)")
    source.add_argument("--sample", choices=[c["id"] for c in load_dataset()])
    parser.add_argument("--live", action="store_true", help="Use Gemini instead of the offline engine.")
    args = parser.parse_args()

    client = None
    if args.live:
        load_dotenv()
        from google import genai

        if not get_api_key():
            print("GOOGLE_API_KEY is not set.", file=sys.stderr)
            return 2
        client = genai.Client(api_key=get_api_key())

    text, file_bytes, mime = "", None, None
    if args.sample:
        text = next(c["document"] for c in load_dataset() if c["id"] == args.sample)
    elif args.path.suffix.lower() in {".txt", ".md", ""}:
        text = args.path.read_text(encoding="utf-8")
    else:
        file_bytes = args.path.read_bytes()
        mime = mimetypes.guess_type(args.path.name)[0] or "application/octet-stream"

    result = InvoicePipeline(client=client).run(text, file_bytes, mime, mode="live" if args.live else "offline")
    print(json.dumps({
        "extraction": result.extraction.model_dump(),
        "audit": result.audit.model_dump(mode="json"),
        "retrieved_policies": [r.chunk.id for r in result.retrieved],
        "trace": [s.model_dump() for s in result.trace],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
