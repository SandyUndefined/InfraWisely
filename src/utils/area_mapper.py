"""Area inference helpers for BESCOM EV charger station text."""

from __future__ import annotations

import re


AREA_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("Indiranagar", ("Indiranagar",)),
    ("HSR Layout", ("HSR",)),
    ("BTM Layout", ("BTM",)),
    ("Yelahanka", ("Yelahanka", "Yalhanka")),
    ("Peenya", ("Peenya",)),
    ("Mahadevpura", ("Mahadevpura", "Mahadevapura")),
    ("KR Puram", ("KR Puram", "K R Puram", "Krishnarajapura")),
    ("Whitefield", ("Whitefield",)),
    ("Jayanagar", ("Jayanagar", "JAYANAGARA")),
    ("Malleshwaram", ("Malleshwaram", "Malleswaram")),
    ("Hebbal", ("Hebbala", "Hebbal")),
    ("Sarjapur Road", ("Sarjapura", "Sarjapur")),
    ("Kengeri", ("Kengeri",)),
    ("Banashankari", ("Banashankari",)),
    ("Koramangala", ("Koramangala",)),
    ("Electronic City", ("Electronic City",)),
    ("MG Road", ("MG Road", "M G Road")),
    ("Rajajinagar", ("Rajajinagar",)),
    ("Marathahalli", ("Marathahalli",)),
    ("Bellandur", ("Bellandur",)),
    ("Majestic", ("Majestic", "Kempegowda")),
]


def infer_area(station_name: str) -> str:
    """Infer a normalized Bengaluru area from charger station/location text."""
    if not isinstance(station_name, str) or not station_name.strip():
        return "Unknown"

    normalized = re.sub(r"\s+", " ", station_name).strip()
    for area, patterns in AREA_PATTERNS:
        for pattern in patterns:
            if re.search(re.escape(pattern), normalized, flags=re.IGNORECASE):
                return area

    return "Unknown"
