"""Shared healthcare claims tool used by all demo modes.

The dataset is intentionally expanded so the "traditional" mode sends a larger
payload to the model and the token-usage difference is easier to observe.
"""

from __future__ import annotations

import os


_DENIALS_BY_DATE = {
    "2026-02-18": [
        {
            "claim_id": "CLM-1001",
            "patient_id": "P-001",
            "payer": "Aetna",
            "denial_reason": "Eligibility",
            "denied_amount_NEW": 250,
        },
        {
            "claim_id": "CLM-1002",
            "patient_id": "P-002",
            "payer": "BCBS",
            "denial_reason": "Prior Authorization",
            "denied_amount_NEW": 430,
        },
        {
            "claim_id": "CLM-1003",
            "patient_id": "P-003",
            "payer": "Aetna",
            "denial_reason": "Eligibility",
            "denied_amount_NEW": 180,
        },
    ],
    "2026-02-19": [
        {
            "claim_id": "CLM-1004",
            "patient_id": "P-004",
            "payer": "United",
            "denial_reason": "Coding Error",
            "denied_amount_NEW": 520,
        },
        {
            "claim_id": "CLM-1005",
            "patient_id": "P-005",
            "payer": "Aetna",
            "denial_reason": "Eligibility",
            "denied_amount_NEW": 300,
        },
    ],
    "2026-02-20": [
        {
            "claim_id": "CLM-1006",
            "patient_id": "P-006",
            "payer": "BCBS",
            "denial_reason": "Prior Authorization",
            "denied_amount_NEW": 610,
        },
        {
            "claim_id": "CLM-1007",
            "patient_id": "P-007",
            "payer": "United",
            "denial_reason": "Eligibility",
            "denied_amount_NEW": 275,
        },
        {
            "claim_id": "CLM-1008",
            "patient_id": "P-008",
            "payer": "Cigna",
            "denial_reason": "Timely Filing",
            "denied_amount_NEW": 190,
        },
    ],
}


def _expanded_denials() -> dict[str, list[dict]]:
    expanded: dict[str, list[dict]] = {}
    for service_date, base_rows in _DENIALS_BY_DATE.items():
        rows: list[dict] = []
        for idx, row in enumerate(base_rows):
            rows.append(dict(row))
            # Create multiple realistic variations per base claim to increase dataset size.
            for n in range(1, 28):
                clone = dict(row)
                clone["claim_id"] = f"{row['claim_id']}-{n:02d}"
                clone["patient_id"] = f"{row['patient_id']}-{n:02d}"
                clone["denied_amount_NEW"] = int(row["denied_amount_NEW"] + (n * 7) + (idx * 3))
                if n % 5 == 0 and clone["denial_reason"] == "Eligibility":
                    clone["denial_reason"] = "Registration Error"
                elif n % 7 == 0 and clone["denial_reason"] == "Prior Authorization":
                    clone["denial_reason"] = "Medical Necessity"
                elif n % 6 == 0 and clone["payer"] == "Aetna":
                    clone["payer"] = "Humana"
                rows.append(clone)
        expanded[service_date] = rows
    return expanded


_EXPANDED_DENIALS_BY_DATE = _expanded_denials()


def healthcare_claims_get_denials(service_date: str) -> list[dict]:
    """Return demo denied claims for a service date."""
    mode = os.environ.get("DEMO_MODE", "UNKNOWN")
    print("🔥 TOOL CALLED: healthcare_claims_get_denials")
    print(f"[{mode}] TOOL INPUT service_date={service_date!r}")
    rows = list(_EXPANDED_DENIALS_BY_DATE.get(service_date, []))
    preview = rows[:2]
    print(f"[{mode}] TOOL OUTPUT row_count={len(rows)} preview={preview}")
    return rows
