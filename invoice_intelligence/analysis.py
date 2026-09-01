"""Core invoice analysis helpers."""

import csv
import io
import json


def extract_risk_tier(analysis_text: str) -> str:
    """Extract the risk tier from an analyst response.

    Defaults to MODERATE when no recognized risk tag is present.
    """
    for tier in ("CRITICAL", "SECURE", "MODERATE"):
        if f"[{tier}]" in analysis_text:
            return tier

    return "MODERATE"


def json_to_csv(json_payload: str) -> str:
    """Convert a JSON object payload into a single-row CSV string.

    Raises:
        ValueError: If the payload is not a JSON object.
    """
    parsed = json.loads(json_payload)

    if not isinstance(parsed, dict):
        raise ValueError("JSON payload must contain an object.")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=parsed.keys())
    writer.writeheader()
    writer.writerow(parsed)

    return output.getvalue()