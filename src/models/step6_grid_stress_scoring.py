"""Create grid stress scores from model-predicted EV charging demand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_INPUT_PATH = Path("data/processed/hourly_demand_with_predictions.csv")
GRID_STRESS_PATH = Path("data/processed/grid_stress_predictions.csv")
ZONE_SUMMARY_PATH = Path("data/processed/grid_stress_summary_by_zone.csv")
REPORT_PATH = Path("reports/step6_grid_stress_report.md")

REQUIRED_COLUMNS = {
    "zone_id",
    "zone_name",
    "hour",
    "time_period",
    "is_peak_hour",
    "zone_type",
    "ev_count_estimate",
    "traffic_score",
    "existing_chargers",
    "demand_growth_rate",
    "transformer_capacity_kw",
    "base_grid_load_kw",
    "model_predicted_ev_load_kw",
}

DETAIL_COLUMNS = [
    "zone_id",
    "zone_name",
    "hour",
    "time_period",
    "is_peak_hour",
    "zone_type",
    "transformer_capacity_kw",
    "base_grid_load_kw",
    "model_predicted_ev_load_kw",
    "model_total_load_kw",
    "model_load_ratio",
    "spare_capacity_after_ev_kw",
    "model_risk_level",
    "risk_score",
    "is_overloaded",
    "action_priority",
    "recommended_action",
    "risk_reason",
]


def ensure_directories() -> None:
    """Create folders used by Step 6."""
    Path("src/models").mkdir(parents=True, exist_ok=True)
    GRID_STRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_predictions(path: Path = PREDICTIONS_INPUT_PATH) -> pd.DataFrame:
    """Read and validate Step 5 prediction output."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}. Run Step 5 first.")

    predictions = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(predictions.columns)
    if missing_columns:
        raise ValueError(f"{path} missing required columns: {sorted(missing_columns)}")

    predictions = predictions.copy()
    predictions["is_peak_hour"] = predictions["is_peak_hour"].astype(bool)
    print(f"Loaded model prediction data from: {path}")
    print(f"Input shape: {predictions.shape}")
    return predictions


def classify_risk(load_ratio: float) -> str:
    """Classify model-based grid risk level."""
    if load_ratio < 0.75:
        return "Low"
    if load_ratio < 0.90:
        return "Medium"
    if load_ratio <= 1.00:
        return "High"
    return "Critical"


def action_priority(risk_level: str) -> str:
    """Map risk level to operator priority."""
    priorities = {
        "Critical": "Immediate Action",
        "High": "Schedule Shift Recommended",
        "Medium": "Monitor",
        "Low": "Normal",
    }
    return priorities[risk_level]


def recommended_action(risk_level: str) -> str:
    """Map risk level to recommended BESCOM operator action."""
    actions = {
        "Critical": "Shift EV charging immediately and restrict fast charging during this hour.",
        "High": "Shift part of EV charging demand to off-peak hours.",
        "Medium": "Monitor load and encourage off-peak charging.",
        "Low": "No immediate action required.",
    }
    return actions[risk_level]


def generate_risk_reason(row: pd.Series) -> str:
    """Generate an explainable grid stress reason for one zone-hour row."""
    hour_label = f"{int(row['hour']):02d}:00"
    load_percent = row["model_load_ratio"] * 100

    if row["model_risk_level"] == "Critical":
        return (
            f"{row['zone_name']} at {hour_label} is Critical because predicted total "
            f"load reaches {load_percent:.1f}% of transformer capacity."
        )
    if row["model_risk_level"] == "High":
        return (
            f"{row['zone_name']} at {hour_label} is High risk because predicted total "
            f"load reaches {load_percent:.1f}% of transformer capacity during "
            f"{row['time_period']}."
        )
    if row["model_risk_level"] == "Medium":
        return (
            f"{row['zone_name']} at {hour_label} needs monitoring because load reaches "
            f"{load_percent:.1f}% of transformer capacity."
        )
    return (
        f"{row['zone_name']} at {hour_label} is Low risk because load remains within "
        "safe capacity limits."
    )


def create_grid_stress_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate model-based grid stress alerts for every zone-hour."""
    grid_stress = predictions.copy()
    grid_stress["model_total_load_kw"] = (
        grid_stress["base_grid_load_kw"] + grid_stress["model_predicted_ev_load_kw"]
    ).round(2)
    grid_stress["model_load_ratio"] = (
        grid_stress["model_total_load_kw"] / grid_stress["transformer_capacity_kw"]
    ).round(3)
    grid_stress["spare_capacity_after_ev_kw"] = (
        grid_stress["transformer_capacity_kw"] - grid_stress["model_total_load_kw"]
    ).round(2)
    grid_stress["model_risk_level"] = grid_stress["model_load_ratio"].apply(classify_risk)
    grid_stress["risk_score"] = np.minimum(
        100, grid_stress["model_load_ratio"] * 100
    ).round(2)
    grid_stress["is_overloaded"] = grid_stress["model_load_ratio"] > 1.0
    grid_stress["action_priority"] = grid_stress["model_risk_level"].apply(action_priority)
    grid_stress["recommended_action"] = grid_stress["model_risk_level"].apply(
        recommended_action
    )
    grid_stress["risk_reason"] = grid_stress.apply(generate_risk_reason, axis=1)

    return grid_stress[DETAIL_COLUMNS]


def zone_priority(
    critical_hours_count: int, high_hours_count: int, overloaded_hours_count: int
) -> str:
    """Classify zone-level action priority."""
    if critical_hours_count >= 3 or overloaded_hours_count >= 3:
        return "Very High"
    if critical_hours_count >= 1 or high_hours_count >= 4:
        return "High"
    if high_hours_count >= 1:
        return "Medium"
    return "Low"


def create_zone_summary(grid_stress: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detailed stress predictions into zone-level priorities."""
    summary_records = []

    for (zone_id, zone_name), group in grid_stress.groupby(["zone_id", "zone_name"]):
        worst_row = group.sort_values(
            ["risk_score", "model_load_ratio"], ascending=False
        ).iloc[0]
        peak_group = group[group["is_peak_hour"]]
        critical_hours_count = int((group["model_risk_level"] == "Critical").sum())
        high_hours_count = int((group["model_risk_level"] == "High").sum())
        overloaded_hours_count = int(group["is_overloaded"].sum())

        summary_records.append(
            {
                "zone_id": zone_id,
                "zone_name": zone_name,
                "max_risk_score": round(float(group["risk_score"].max()), 2),
                "avg_risk_score": round(float(group["risk_score"].mean()), 2),
                "max_model_load_ratio": round(float(group["model_load_ratio"].max()), 3),
                "critical_hours_count": critical_hours_count,
                "high_hours_count": high_hours_count,
                "overloaded_hours_count": overloaded_hours_count,
                "peak_hour_avg_risk_score": round(
                    float(peak_group["risk_score"].mean()), 2
                ),
                "worst_hour": int(worst_row["hour"]),
                "worst_time_period": worst_row["time_period"],
                "worst_risk_level": worst_row["model_risk_level"],
                "zone_priority": zone_priority(
                    critical_hours_count, high_hours_count, overloaded_hours_count
                ),
            }
        )

    priority_order = {"Very High": 0, "High": 1, "Medium": 2, "Low": 3}
    summary = pd.DataFrame(summary_records)
    summary["_priority_order"] = summary["zone_priority"].map(priority_order)
    summary = summary.sort_values(
        ["_priority_order", "max_risk_score", "avg_risk_score"],
        ascending=[True, False, False],
    ).drop(columns=["_priority_order"])

    return summary.reset_index(drop=True)


def create_report(grid_stress: pd.DataFrame, zone_summary: pd.DataFrame) -> None:
    """Create Step 6 Markdown report."""
    risk_counts = grid_stress["model_risk_level"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"], fill_value=0
    )
    overloaded_count = int(grid_stress["is_overloaded"].sum())
    top_risk = grid_stress.nlargest(10, ["risk_score", "model_load_ratio"])[
        [
            "zone_name",
            "hour",
            "time_period",
            "model_total_load_kw",
            "model_load_ratio",
            "model_risk_level",
            "action_priority",
        ]
    ]
    top_zones = zone_summary.head(10)
    peak_critical_high_count = int(
        (
            grid_stress["is_peak_hour"]
            & grid_stress["model_risk_level"].isin(["Critical", "High"])
        ).sum()
    )

    risk_lines = [f"| {risk} | {count} |" for risk, count in risk_counts.items()]
    top_risk_lines = [
        (
            f"| {row.zone_name} | {row.hour} | {row.time_period} | "
            f"{row.model_total_load_kw:.2f} | {row.model_load_ratio:.3f} | "
            f"{row.model_risk_level} | {row.action_priority} |"
        )
        for row in top_risk.itertuples(index=False)
    ]
    top_zone_lines = [
        (
            f"| {row.zone_name} | {row.zone_priority} | {row.max_risk_score:.2f} | "
            f"{row.critical_hours_count} | {row.high_hours_count} | "
            f"{row.overloaded_hours_count} | {row.worst_hour}:00 |"
        )
        for row in top_zones.itertuples(index=False)
    ]

    report = f"""# Step 6 Grid Stress Report

## Step Name

Create grid stress scoring from model-predicted EV charging demand.

## Inputs And Outputs

- Input used: `data/processed/hourly_demand_with_predictions.csv`
- Outputs generated:
  - `data/processed/grid_stress_predictions.csv`
  - `data/processed/grid_stress_summary_by_zone.csv`

## Summary

- Total zone-hour records: {len(grid_stress)}
- Number of overloaded records: {overloaded_count}
- Peak-hour Critical/High count: {peak_critical_high_count}

## Risk Level Counts

| Risk level | Count |
|---|---:|
{chr(10).join(risk_lines)}

## Top 10 Highest-Risk Zone-Hour Combinations

| Zone | Hour | Period | Model total load kW | Model load ratio | Risk | Action priority |
|---|---:|---|---:|---:|---|---|
{chr(10).join(top_risk_lines)}

## Top 10 Priority Zones

| Zone | Zone priority | Max risk score | Critical hours | High hours | Overloaded hours | Worst hour |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(top_zone_lines)}

## Notes

- Turns ML demand prediction into grid-risk alerts.
- Uses transformer capacity constraints.
- Provides explainable BESCOM operator actions.
- Supports scheduling optimization in the next step.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved Step 6 report to: {REPORT_PATH}")


def print_summary(grid_stress: pd.DataFrame, zone_summary: pd.DataFrame) -> None:
    """Print useful Step 6 inspection outputs."""
    top_risk = grid_stress.nlargest(10, ["risk_score", "model_load_ratio"])[
        [
            "zone_name",
            "hour",
            "time_period",
            "model_total_load_kw",
            "model_load_ratio",
            "model_risk_level",
            "action_priority",
        ]
    ]
    top_zones = zone_summary.head(10)[
        [
            "zone_name",
            "zone_priority",
            "max_risk_score",
            "critical_hours_count",
            "high_hours_count",
            "overloaded_hours_count",
            "worst_hour",
        ]
    ]

    print(f"\ngrid_stress_predictions shape: {grid_stress.shape}")
    print("\nmodel_risk_level value counts:")
    print(grid_stress["model_risk_level"].value_counts().to_string())
    print(f"\nOverloaded record count: {int(grid_stress['is_overloaded'].sum())}")
    print("\nTop 10 highest risk rows:")
    print(top_risk.to_string(index=False))
    print("\nTop 10 priority zones:")
    print(top_zones.to_string(index=False))


def main() -> None:
    """Run Step 6 grid stress scoring."""
    try:
        ensure_directories()
        predictions = read_predictions()
        grid_stress = create_grid_stress_predictions(predictions)
        zone_summary = create_zone_summary(grid_stress)

        grid_stress.to_csv(GRID_STRESS_PATH, index=False)
        print(f"Saved detailed grid stress predictions to: {GRID_STRESS_PATH}")
        zone_summary.to_csv(ZONE_SUMMARY_PATH, index=False)
        print(f"Saved grid stress zone summary to: {ZONE_SUMMARY_PATH}")
        create_report(grid_stress, zone_summary)
        print_summary(grid_stress, zone_summary)
        print("\nStep 6 grid stress scoring complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 6 failed: {exc}") from exc


if __name__ == "__main__":
    main()
