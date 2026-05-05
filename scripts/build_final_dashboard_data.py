"""Build cleaned final-prototype dashboard JSON without changing old outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DASHBOARD = ROOT / "data" / "dashboard"
TARGET_DASHBOARD = ROOT / "final_prototype" / "data" / "dashboard"
TARGET_REPORT = ROOT / "final_prototype" / "reports" / "final_prototype_data_notes.md"


FILES = [
    "summary_metrics.json",
    "demand_forecast.json",
    "grid_stress_alerts.json",
    "scheduling_recommendations.json",
    "load_curve_before_after.json",
    "station_recommendations.json",
    "map_zones.json",
    "demo_storyline.json",
    "model_explainability_summary.json",
]


def read_json(name: str) -> Any:
    path = SOURCE_DASHBOARD / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required dashboard JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: Any) -> None:
    TARGET_DASHBOARD.mkdir(parents=True, exist_ok=True)
    (TARGET_DASHBOARD / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def hour_label(hour: int | float | str) -> str:
    value = int(hour)
    suffix = "AM" if value < 12 else "PM"
    display_hour = value % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:00 {suffix}"


def convert_time_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return hour_label(int(match.group(1)))

    return re.sub(r"\b([01]?\d|2[0-3]):00\b", replace, text)


def add_hour_labels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in records:
        item = dict(row)
        if "hour" in item:
            item["hour_label"] = hour_label(item["hour"])
        if "peak_hour" in item:
            item["peak_hour_label"] = hour_label(item["peak_hour"])
        if "recommended_offpeak_hours" in item:
            item["recommended_offpeak_hours_label"] = convert_time_text(
                str(item["recommended_offpeak_hours"])
            )
        for field in ("risk_reason", "recommendation_text", "explanation"):
            if field in item and isinstance(item[field], str):
                item[field] = convert_time_text(item[field])
        output.append(item)
    return output


def sort_alerts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort stress alerts by actual load ratio, then risk score."""
    return sorted(
        records,
        key=lambda row: (
            float(row.get("model_load_ratio", 0)),
            float(row.get("risk_score", 0)),
        ),
        reverse=True,
    )


def build_summary(summary: dict[str, Any]) -> dict[str, Any]:
    final = dict(summary)
    final["critical_risk_before"] = summary["critical_hours_before"]
    final["critical_risk_after"] = summary["critical_hours_after"]
    final["overload_risk_before"] = summary["overloaded_hours_before"]
    final["overload_risk_after"] = summary["overloaded_hours_after"]
    return final


def build_storyline(storyline: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for item in storyline:
        updated = dict(item)
        if updated["title"] == "Smart Charging Optimization":
            updated["key_metric"] = (
                "Critical risk reduced from "
                f"{summary['critical_risk_before']} to {summary['critical_risk_after']}"
            )
        output.append(updated)
    return output


def build_explainability(explainability: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    output = dict(explainability)
    examples = []
    for item in output.get("explainability_examples", []):
        updated = dict(item)
        updated["message"] = updated["message"].replace("Critical hours reduce", "Critical risk reduces")
        updated["message"] = updated["message"].replace("Critical zone-hours reduce", "Critical risk reduces")
        updated["message"] = updated["message"].replace("Critical alerts reduce", "Critical risk reduces")
        updated["message"] = convert_time_text(updated["message"])
        examples.append(updated)
    output["explainability_examples"] = examples
    return output


def main() -> None:
    for name in FILES:
        read_json(name)

    summary = build_summary(read_json("summary_metrics.json"))
    write_json("summary_metrics.json", summary)
    write_json("demand_forecast.json", add_hour_labels(read_json("demand_forecast.json")))
    write_json(
        "grid_stress_alerts.json",
        add_hour_labels(sort_alerts(read_json("grid_stress_alerts.json"))),
    )
    write_json(
        "scheduling_recommendations.json",
        add_hour_labels(read_json("scheduling_recommendations.json")),
    )
    write_json(
        "load_curve_before_after.json",
        add_hour_labels(read_json("load_curve_before_after.json")),
    )
    write_json("station_recommendations.json", read_json("station_recommendations.json"))
    write_json("map_zones.json", read_json("map_zones.json"))
    write_json("demo_storyline.json", build_storyline(read_json("demo_storyline.json"), summary))
    write_json(
        "model_explainability_summary.json",
        build_explainability(read_json("model_explainability_summary.json"), summary),
    )

    TARGET_REPORT.parent.mkdir(parents=True, exist_ok=True)
    TARGET_REPORT.write_text(
        "# Final Prototype Data Notes\n\n"
        "The final prototype preserves all original AI/ML outputs and creates cleaned copies under "
        "`final_prototype/data/dashboard`.\n\n"
        "- Clock hours are displayed in 12-hour format.\n"
        "- Confusing alert wording is removed from the visible final dashboard.\n"
        "- The frontend focuses on selected-zone demand, scheduling, and station planning.\n"
        "- Original model files and CSV outputs are unchanged.\n",
        encoding="utf-8",
    )
    print(f"Built cleaned dashboard JSON in {TARGET_DASHBOARD}")


if __name__ == "__main__":
    main()
