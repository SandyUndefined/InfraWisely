"""Export GridCharge AI outputs as dashboard/API-ready JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DASHBOARD_DIR = Path("data/dashboard")
REPORT_PATH = Path("reports/step9_dashboard_export_report.md")

INPUT_PATHS = {
    "zones_master": Path("data/processed/zones_master.csv"),
    "hourly_predictions": Path("data/processed/hourly_demand_with_predictions.csv"),
    "grid_stress": Path("data/processed/grid_stress_predictions.csv"),
    "grid_stress_summary": Path("data/processed/grid_stress_summary_by_zone.csv"),
    "schedule_recommendations": Path("data/processed/charging_schedule_recommendations.csv"),
    "load_curve": Path("data/processed/load_curve_before_after.csv"),
    "scheduling_impact": Path("data/processed/scheduling_impact_summary.csv"),
    "station_recommendations": Path("data/processed/station_location_recommendations.csv"),
    "station_planning": Path("data/processed/station_planning_summary.csv"),
    "model_metrics": Path("models/ev_demand_model_metrics.json"),
}

OUTPUT_PATHS = {
    "summary_metrics": DASHBOARD_DIR / "summary_metrics.json",
    "demand_forecast": DASHBOARD_DIR / "demand_forecast.json",
    "grid_stress_alerts": DASHBOARD_DIR / "grid_stress_alerts.json",
    "scheduling_recommendations": DASHBOARD_DIR / "scheduling_recommendations.json",
    "load_curve_before_after": DASHBOARD_DIR / "load_curve_before_after.json",
    "station_recommendations": DASHBOARD_DIR / "station_recommendations.json",
    "map_zones": DASHBOARD_DIR / "map_zones.json",
}

REQUIRED_COLUMNS = {
    "zones_master": {
        "zone_id",
        "zone_name",
        "latitude",
        "longitude",
        "zone_type",
        "existing_chargers",
        "ev_count_estimate",
    },
    "hourly_predictions": {
        "zone_id",
        "zone_name",
        "hour",
        "time_period",
        "zone_type",
        "ev_count_estimate",
        "existing_chargers",
        "base_grid_load_kw",
        "model_predicted_ev_load_kw",
    },
    "grid_stress": {
        "zone_id",
        "zone_name",
        "hour",
        "time_period",
        "transformer_capacity_kw",
        "model_total_load_kw",
        "model_load_ratio",
        "model_risk_level",
        "risk_score",
        "is_overloaded",
        "action_priority",
        "recommended_action",
        "risk_reason",
    },
    "grid_stress_summary": {
        "zone_id",
        "zone_name",
        "max_risk_score",
        "avg_risk_score",
        "critical_hours_count",
        "overloaded_hours_count",
        "zone_priority",
    },
    "schedule_recommendations": {
        "zone_id",
        "zone_name",
        "peak_hour",
        "original_risk_level",
        "original_load_ratio",
        "recommended_shift_percent",
        "shifted_load_kw",
        "allocated_shift_kw",
        "unallocated_shift_kw",
        "recommended_offpeak_hours",
        "feasibility_status",
        "expected_peak_load_reduction_kw",
        "expected_peak_load_reduction_percent",
        "recommendation_text",
        "explanation",
    },
    "load_curve": {
        "zone_id",
        "zone_name",
        "hour",
        "time_period",
        "transformer_capacity_kw",
        "before_ev_load_kw",
        "after_ev_load_kw",
        "before_total_load_kw",
        "after_total_load_kw",
        "before_load_ratio",
        "after_load_ratio",
        "before_risk_level",
        "after_risk_level",
    },
    "scheduling_impact": {
        "peak_load_reduction_percent",
        "before_critical_hours",
        "after_critical_hours",
        "before_overloaded_hours",
        "after_overloaded_hours",
    },
    "station_recommendations": {
        "rank",
        "zone_id",
        "zone_name",
        "latitude",
        "longitude",
        "zone_type",
        "ev_count_estimate",
        "demand_growth_rate",
        "existing_chargers",
        "charger_gap_score",
        "spare_capacity_kw",
        "grid_health_score",
        "avg_risk_score",
        "overloaded_hours_count",
        "station_priority_score",
        "grid_feasibility_label",
        "recommended_station_type",
        "recommended_chargers",
        "planning_priority",
        "capital_planning_flag",
        "recommendation_reason",
    },
    "station_planning": {
        "total_recommended_chargers",
        "invest_now_zones",
        "fast_charging_hubs",
        "top_priority_zone",
    },
}


def ensure_directories() -> None:
    """Create dashboard and report directories."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def validate_file_exists(path: Path) -> None:
    """Raise a clear error if an input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")


def validate_columns(df: pd.DataFrame, required_columns: set[str], name: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{INPUT_PATHS[name]} missing columns: {sorted(missing_columns)}")


def load_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load and validate all dashboard export inputs."""
    for path in INPUT_PATHS.values():
        validate_file_exists(path)

    frames: dict[str, pd.DataFrame] = {}
    for name, path in INPUT_PATHS.items():
        if name == "model_metrics":
            continue
        frames[name] = pd.read_csv(path)
        validate_columns(frames[name], REQUIRED_COLUMNS[name], name)
        print(f"Loaded {name}: {path} ({frames[name].shape[0]} rows)")

    metrics = json.loads(INPUT_PATHS["model_metrics"].read_text(encoding="utf-8"))
    print(f"Loaded model metrics: {INPUT_PATHS['model_metrics']}")
    return frames, metrics


def to_python_value(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 2)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return round(value, 2)
    return value


def records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to clean frontend-friendly JSON records."""
    clean_df = df.replace({np.nan: None})
    records = clean_df.to_dict(orient="records")
    return [
        {key: to_python_value(value) for key, value in record.items()}
        for record in records
    ]


def write_json(path: Path, payload: Any) -> int:
    """Write JSON payload and return record count."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        return 1
    return 0


def rounded_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select columns and round numeric values to 2 decimals."""
    result = df[columns].copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(2)
    return result


def create_summary_metrics(frames: dict[str, pd.DataFrame], metrics: dict[str, Any]) -> dict[str, Any]:
    """Create dashboard KPI summary metrics."""
    zones = frames["zones_master"]
    station_planning = frames["station_planning"].iloc[0]
    impact = frames["scheduling_impact"].iloc[0]
    station_recs = frames["station_recommendations"]
    grid_summary = frames["grid_stress_summary"]
    schedule_recs = frames["schedule_recommendations"]

    highest_risk = grid_summary.sort_values(
        ["max_risk_score", "avg_risk_score"], ascending=False
    ).iloc[0]

    summary = {
        "total_zones": int(zones["zone_id"].nunique()),
        "total_existing_chargers": int(zones["existing_chargers"].sum()),
        "total_recommended_chargers": int(station_planning["total_recommended_chargers"]),
        "model_r2_score": round(float(metrics.get("r2_score", 0)), 2),
        "model_mae": round(float(metrics.get("mae", 0)), 2),
        "model_rmse": round(float(metrics.get("rmse", 0)), 2),
        "peak_load_reduction_percent": round(float(impact["peak_load_reduction_percent"]), 2),
        "critical_hours_before": int(impact["before_critical_hours"]),
        "critical_hours_after": int(impact["after_critical_hours"]),
        "overloaded_hours_before": int(impact["before_overloaded_hours"]),
        "overloaded_hours_after": int(impact["after_overloaded_hours"]),
        "total_scheduling_recommendations": int(len(schedule_recs)),
        "invest_now_zones": int(station_planning["invest_now_zones"]),
        "fast_charging_hubs": int(station_planning["fast_charging_hubs"]),
        "top_priority_station_zone": station_planning["top_priority_zone"],
        "highest_risk_zone": highest_risk["zone_name"],
    }
    return summary


def create_demand_forecast(hourly: pd.DataFrame) -> list[dict[str, Any]]:
    """Create demand forecast JSON records."""
    demand = hourly.copy()
    if "model_total_load_kw" not in demand.columns:
        demand["model_total_load_kw"] = (
            demand["base_grid_load_kw"] + demand["model_predicted_ev_load_kw"]
        )
    columns = [
        "zone_id",
        "zone_name",
        "hour",
        "time_period",
        "zone_type",
        "ev_count_estimate",
        "existing_chargers",
        "base_grid_load_kw",
        "model_predicted_ev_load_kw",
        "model_total_load_kw",
    ]
    return records_for_json(rounded_frame(demand, columns))


def create_grid_stress_alerts(grid_stress: pd.DataFrame) -> list[dict[str, Any]]:
    """Create top High/Critical grid stress alert records."""
    alerts = grid_stress[
        grid_stress["model_risk_level"].isin(["High", "Critical"])
    ].sort_values("risk_score", ascending=False)
    columns = [
        "zone_id",
        "zone_name",
        "hour",
        "time_period",
        "transformer_capacity_kw",
        "model_total_load_kw",
        "model_load_ratio",
        "model_risk_level",
        "risk_score",
        "is_overloaded",
        "action_priority",
        "recommended_action",
        "risk_reason",
    ]
    return records_for_json(rounded_frame(alerts[columns].head(50), columns))


def create_scheduling_recommendations(schedule_recs: pd.DataFrame) -> list[dict[str, Any]]:
    """Create scheduling recommendation JSON records."""
    sorted_recs = schedule_recs.sort_values("original_load_ratio", ascending=False)
    columns = [
        "zone_id",
        "zone_name",
        "peak_hour",
        "original_risk_level",
        "recommended_shift_percent",
        "shifted_load_kw",
        "allocated_shift_kw",
        "unallocated_shift_kw",
        "recommended_offpeak_hours",
        "feasibility_status",
        "expected_peak_load_reduction_kw",
        "expected_peak_load_reduction_percent",
        "recommendation_text",
        "explanation",
    ]
    return records_for_json(rounded_frame(sorted_recs, columns))


def create_station_recommendations(station_recs: pd.DataFrame) -> list[dict[str, Any]]:
    """Create station recommendation JSON records."""
    sorted_recs = station_recs.sort_values("rank")
    columns = [
        "rank",
        "zone_id",
        "zone_name",
        "latitude",
        "longitude",
        "zone_type",
        "ev_count_estimate",
        "demand_growth_rate",
        "existing_chargers",
        "charger_gap_score",
        "spare_capacity_kw",
        "grid_health_score",
        "avg_risk_score",
        "overloaded_hours_count",
        "station_priority_score",
        "grid_feasibility_label",
        "recommended_station_type",
        "recommended_chargers",
        "planning_priority",
        "capital_planning_flag",
        "recommendation_reason",
    ]
    return records_for_json(rounded_frame(sorted_recs, columns))


def create_map_zones(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Create map marker/heatmap zone records."""
    zones = frames["zones_master"]
    stress = frames["grid_stress_summary"]
    stations = frames["station_recommendations"]
    station_cols = [
        "zone_id",
        "zone_name",
        "station_priority_score",
        "recommended_station_type",
        "recommended_chargers",
        "capital_planning_flag",
    ]
    map_df = zones.merge(
        stress[
            [
                "zone_id",
                "zone_name",
                "max_risk_score",
                "avg_risk_score",
                "critical_hours_count",
                "overloaded_hours_count",
                "zone_priority",
            ]
        ],
        on=["zone_id", "zone_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        stations[station_cols],
        on=["zone_id", "zone_name"],
        how="inner",
        validate="one_to_one",
    )
    columns = [
        "zone_id",
        "zone_name",
        "latitude",
        "longitude",
        "zone_type",
        "existing_chargers",
        "ev_count_estimate",
        "max_risk_score",
        "avg_risk_score",
        "critical_hours_count",
        "overloaded_hours_count",
        "zone_priority",
        "station_priority_score",
        "recommended_station_type",
        "recommended_chargers",
        "capital_planning_flag",
    ]
    return records_for_json(rounded_frame(map_df, columns))


def export_dashboard_json(frames: dict[str, pd.DataFrame], metrics: dict[str, Any]) -> dict[str, int]:
    """Create all dashboard JSON files and return record counts."""
    summary_metrics = create_summary_metrics(frames, metrics)
    demand_forecast = create_demand_forecast(frames["hourly_predictions"])
    grid_stress_alerts = create_grid_stress_alerts(frames["grid_stress"])
    scheduling_recommendations = create_scheduling_recommendations(
        frames["schedule_recommendations"]
    )
    load_curve = records_for_json(
        rounded_frame(
            frames["load_curve"],
            [
                "zone_id",
                "zone_name",
                "hour",
                "time_period",
                "transformer_capacity_kw",
                "before_ev_load_kw",
                "after_ev_load_kw",
                "before_total_load_kw",
                "after_total_load_kw",
                "before_load_ratio",
                "after_load_ratio",
                "before_risk_level",
                "after_risk_level",
            ],
        )
    )
    station_recommendations = create_station_recommendations(
        frames["station_recommendations"]
    )
    map_zones = create_map_zones(frames)

    payloads = {
        "summary_metrics": summary_metrics,
        "demand_forecast": demand_forecast,
        "grid_stress_alerts": grid_stress_alerts,
        "scheduling_recommendations": scheduling_recommendations,
        "load_curve_before_after": load_curve,
        "station_recommendations": station_recommendations,
        "map_zones": map_zones,
    }

    record_counts: dict[str, int] = {}
    for name, payload in payloads.items():
        record_counts[name] = write_json(OUTPUT_PATHS[name], payload)
        print(f"Generated {OUTPUT_PATHS[name]} ({record_counts[name]} records)")

    return record_counts


def create_report(
    record_counts: dict[str, int],
    summary_metrics: dict[str, Any],
) -> None:
    """Create the Step 9 dashboard export report."""
    input_lines = [f"- `{path}`" for path in INPUT_PATHS.values()]
    output_lines = [f"- `{path}`" for path in OUTPUT_PATHS.values()]
    count_lines = [
        f"| {OUTPUT_PATHS[name].name} | {count} |"
        for name, count in record_counts.items()
    ]
    summary_lines = [f"| {key} | {value} |" for key, value in summary_metrics.items()]

    report = f"""# Step 9 Dashboard Export Report

## Step Name

Export AI/ML outputs into dashboard/API-ready JSON files.

## Inputs Used

{chr(10).join(input_lines)}

## Outputs Generated

{chr(10).join(output_lines)}

## JSON Record Counts

| JSON file | Records |
|---|---:|
{chr(10).join(count_lines)}

## Summary Metrics

| Metric | Value |
|---|---|
{chr(10).join(summary_lines)}

## Notes For Frontend/Backend Teammate

- `summary_metrics.json` powers KPI cards.
- `demand_forecast.json` powers demand charts.
- `grid_stress_alerts.json` powers risk alert table.
- `scheduling_recommendations.json` powers charging shift recommendations.
- `load_curve_before_after.json` powers before-vs-after graph.
- `station_recommendations.json` powers infrastructure planning table.
- `map_zones.json` powers map markers/heatmap.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved Step 9 report to: {REPORT_PATH}")


def print_summary(
    record_counts: dict[str, int],
    summary_metrics: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> None:
    """Print dashboard export summary."""
    top_stations = frames["station_recommendations"].sort_values("rank").head(5)
    top_alerts = frames["grid_stress"].sort_values("risk_score", ascending=False).head(5)

    print("\nGenerated JSON files:")
    for name, count in record_counts.items():
        print(f"- {OUTPUT_PATHS[name]}: {count} records")

    print("\nSummary metrics:")
    print(json.dumps(summary_metrics, indent=2))

    print("\nTop 5 station recommendation zones:")
    print(
        top_stations[
            [
                "rank",
                "zone_name",
                "station_priority_score",
                "recommended_station_type",
                "recommended_chargers",
            ]
        ].to_string(index=False)
    )

    print("\nTop 5 grid stress alerts:")
    print(
        top_alerts[
            [
                "zone_name",
                "hour",
                "model_risk_level",
                "risk_score",
                "action_priority",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    """Run Step 9 dashboard JSON export."""
    try:
        ensure_directories()
        frames, metrics = load_inputs()
        record_counts = export_dashboard_json(frames, metrics)
        summary_metrics = create_summary_metrics(frames, metrics)
        create_report(record_counts, summary_metrics)
        print_summary(record_counts, summary_metrics, frames)
        print("\nStep 9 dashboard JSON export complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 9 failed: {exc}") from exc


if __name__ == "__main__":
    main()
