"""Rank Bengaluru zones for new EV charging station infrastructure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ZONES_MASTER_PATH = Path("data/processed/zones_master.csv")
GRID_CAPACITY_PATH = Path("data/processed/grid_capacity.csv")
GRID_STRESS_SUMMARY_PATH = Path("data/processed/grid_stress_summary_by_zone.csv")
RECOMMENDATIONS_PATH = Path("data/processed/station_location_recommendations.csv")
PLANNING_SUMMARY_PATH = Path("data/processed/station_planning_summary.csv")
REPORT_PATH = Path("reports/step8_station_location_report.md")

ZONES_REQUIRED_COLUMNS = {
    "zone_id",
    "zone_name",
    "latitude",
    "longitude",
    "zone_type",
    "population_density",
    "traffic_score",
    "ev_count_estimate",
    "demand_growth_rate",
    "existing_chargers",
    "charger_gap_score",
    "charger_coverage_level",
}

GRID_REQUIRED_COLUMNS = {
    "zone_id",
    "zone_name",
    "transformer_capacity_kw",
    "current_peak_load_kw",
    "spare_capacity_kw",
    "load_ratio",
    "grid_health_score",
    "grid_capacity_status",
}

STRESS_REQUIRED_COLUMNS = {
    "zone_id",
    "zone_name",
    "max_risk_score",
    "avg_risk_score",
    "max_model_load_ratio",
    "critical_hours_count",
    "high_hours_count",
    "overloaded_hours_count",
    "peak_hour_avg_risk_score",
    "zone_priority",
}

FINAL_COLUMNS = [
    "rank",
    "zone_id",
    "zone_name",
    "latitude",
    "longitude",
    "zone_type",
    "ev_count_estimate",
    "demand_growth_rate",
    "traffic_score",
    "existing_chargers",
    "charger_gap_score",
    "charger_coverage_level",
    "spare_capacity_kw",
    "grid_health_score",
    "grid_capacity_status",
    "avg_risk_score",
    "overloaded_hours_count",
    "infrastructure_need_score",
    "grid_feasibility_score",
    "station_priority_score",
    "grid_feasibility_label",
    "recommended_station_type",
    "recommended_chargers",
    "planning_priority",
    "capital_planning_flag",
    "recommendation_reason",
]


def ensure_directories() -> None:
    """Create folders used by Step 8."""
    Path("src/models").mkdir(parents=True, exist_ok=True)
    RECOMMENDATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    """Validate required columns in an input dataset."""
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{path} missing required columns: {sorted(missing_columns)}")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and validate Step 8 input files."""
    for path in (ZONES_MASTER_PATH, GRID_CAPACITY_PATH, GRID_STRESS_SUMMARY_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    zones = pd.read_csv(ZONES_MASTER_PATH)
    grid = pd.read_csv(GRID_CAPACITY_PATH)
    stress = pd.read_csv(GRID_STRESS_SUMMARY_PATH)

    validate_columns(zones, ZONES_REQUIRED_COLUMNS, ZONES_MASTER_PATH)
    validate_columns(grid, GRID_REQUIRED_COLUMNS, GRID_CAPACITY_PATH)
    validate_columns(stress, STRESS_REQUIRED_COLUMNS, GRID_STRESS_SUMMARY_PATH)

    print(f"Loaded zones master from: {ZONES_MASTER_PATH}")
    print(f"Loaded grid capacity from: {GRID_CAPACITY_PATH}")
    print(f"Loaded grid stress summary from: {GRID_STRESS_SUMMARY_PATH}")
    return zones, grid, stress


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize a numeric series between 0 and 1."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    min_value = numeric.min()
    max_value = numeric.max()
    if max_value == min_value:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - min_value) / (max_value - min_value)


def merge_inputs(zones: pd.DataFrame, grid: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    """Merge zone, grid, and stress features."""
    grid_cols = [
        "zone_id",
        "zone_name",
        "transformer_capacity_kw",
        "current_peak_load_kw",
        "spare_capacity_kw",
        "load_ratio",
        "grid_health_score",
        "grid_capacity_status",
    ]
    stress_cols = [
        "zone_id",
        "zone_name",
        "max_risk_score",
        "avg_risk_score",
        "max_model_load_ratio",
        "critical_hours_count",
        "high_hours_count",
        "overloaded_hours_count",
        "peak_hour_avg_risk_score",
        "zone_priority",
    ]

    merged = zones.merge(
        grid[grid_cols],
        on=["zone_id", "zone_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        stress[stress_cols],
        on=["zone_id", "zone_name"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(zones):
        raise ValueError(
            f"Merged row count {len(merged)} does not match zones row count {len(zones)}."
        )

    print(f"Merged planning rows: {len(merged)}")
    return merged


def grid_feasibility_label(row: pd.Series) -> str:
    """Assign a grid feasibility label."""
    if row["grid_health_score"] >= 22 and row["spare_capacity_kw"] >= 350:
        return "High Feasibility"
    if row["grid_health_score"] >= 14 and row["spare_capacity_kw"] >= 200:
        return "Moderate Feasibility"
    return "Low Feasibility"


def recommended_station_type(row: pd.Series) -> str:
    """Assign recommended station type with grid-aware fast charger guardrail."""
    score = row["station_priority_score"]
    feasibility = row["grid_feasibility_label"]

    if feasibility == "Low Feasibility" and score >= 70:
        return "AC Charging Only / Grid Upgrade Needed"
    if feasibility == "High Feasibility" and score >= 75:
        return "Fast Charging Hub"
    if feasibility in {"High Feasibility", "Moderate Feasibility"} and score >= 60:
        return "Mixed AC/DC Station"
    if score >= 45:
        return "Slow/AC Charging Station"
    return "Monitor Only"


def recommended_chargers(station_type: str) -> int:
    """Map station type to recommended charger count."""
    chargers = {
        "Fast Charging Hub": 3,
        "Mixed AC/DC Station": 2,
        "Slow/AC Charging Station": 1,
        "AC Charging Only / Grid Upgrade Needed": 1,
        "Monitor Only": 0,
    }
    return chargers[station_type]


def planning_priority(score: float) -> str:
    """Assign planning priority label from station priority score."""
    if score >= 80:
        return "Very High"
    if score >= 65:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def capital_planning_flag(row: pd.Series) -> str:
    """Assign capital planning action."""
    if row["planning_priority"] in {"Very High", "High"}:
        if row["grid_feasibility_label"] == "Low Feasibility":
            return "Invest with Grid Upgrade"
        return "Invest Now"
    return "Monitor Demand"


def generate_station_reason(row: pd.Series) -> str:
    """Generate an explainable recommendation reason."""
    growth_percent = row["demand_growth_rate"] * 100
    common = (
        f"{row['zone_name']} has an EV count estimate of {int(row['ev_count_estimate'])}, "
        f"{growth_percent:.1f}% demand growth, {int(row['existing_chargers'])} existing "
        f"chargers, {row['grid_feasibility_label'].lower()}, and "
        f"{int(row['overloaded_hours_count'])} overloaded hours. "
    )

    if row["recommended_station_type"] == "Fast Charging Hub":
        return (
            common
            + f"It is recommended as a Fast Charging Hub because demand, charger gap, "
            f"and spare grid capacity are all strong."
        )
    if row["recommended_station_type"] == "AC Charging Only / Grid Upgrade Needed":
        return (
            common
            + "Demand is strong but grid feasibility is low, so AC charging or a grid "
            "upgrade is recommended before fast charging."
        )
    if row["recommended_station_type"] == "Mixed AC/DC Station":
        return (
            common
            + "A Mixed AC/DC Station balances demand growth with available grid capacity."
        )
    if row["recommended_station_type"] == "Slow/AC Charging Station":
        return (
            common
            + "A Slow/AC Charging Station is recommended to add coverage without stressing "
            "the local transformer."
        )
    return (
        common
        + "The zone is marked for monitoring because current priority or charger gap does "
        "not justify immediate infrastructure investment."
    )


def create_recommendations(merged: pd.DataFrame) -> pd.DataFrame:
    """Score and rank zones for charging station placement."""
    recommendations = merged.copy()

    recommendations["ev_demand_score"] = min_max_normalize(
        recommendations["ev_count_estimate"]
    )
    recommendations["growth_score"] = min_max_normalize(
        recommendations["demand_growth_rate"]
    )
    recommendations["traffic_score_norm"] = min_max_normalize(
        recommendations["traffic_score"]
    )
    recommendations["charger_gap_score_norm"] = min_max_normalize(
        recommendations["charger_gap_score"]
    )
    recommendations["grid_spare_capacity_score"] = min_max_normalize(
        recommendations["spare_capacity_kw"]
    )
    recommendations["grid_health_score_norm"] = min_max_normalize(
        recommendations["grid_health_score"]
    )
    recommendations["stress_score"] = min_max_normalize(recommendations["avg_risk_score"])
    recommendations["overload_score"] = min_max_normalize(
        recommendations["overloaded_hours_count"]
    )

    recommendations["infrastructure_need_score"] = (
        0.25 * recommendations["ev_demand_score"]
        + 0.20 * recommendations["growth_score"]
        + 0.20 * recommendations["charger_gap_score_norm"]
        + 0.15 * recommendations["traffic_score_norm"]
        + 0.10 * recommendations["stress_score"]
        + 0.10 * recommendations["overload_score"]
    ).round(4)
    recommendations["grid_feasibility_score"] = (
        0.55 * recommendations["grid_spare_capacity_score"]
        + 0.45 * recommendations["grid_health_score_norm"]
    ).round(4)
    recommendations["station_priority_score"] = (
        (
            0.70 * recommendations["infrastructure_need_score"]
            + 0.30 * recommendations["grid_feasibility_score"]
        )
        * 100
    ).round(2)

    recommendations["grid_feasibility_label"] = recommendations.apply(
        grid_feasibility_label, axis=1
    )
    recommendations["recommended_station_type"] = recommendations.apply(
        recommended_station_type, axis=1
    )
    recommendations["recommended_chargers"] = recommendations[
        "recommended_station_type"
    ].apply(recommended_chargers)
    recommendations["planning_priority"] = recommendations["station_priority_score"].apply(
        planning_priority
    )
    recommendations["capital_planning_flag"] = recommendations.apply(
        capital_planning_flag, axis=1
    )
    recommendations["recommendation_reason"] = recommendations.apply(
        generate_station_reason, axis=1
    )

    recommendations = recommendations.sort_values(
        ["station_priority_score", "infrastructure_need_score", "grid_feasibility_score"],
        ascending=False,
    ).reset_index(drop=True)
    recommendations.insert(0, "rank", recommendations.index + 1)

    return recommendations[FINAL_COLUMNS]


def create_planning_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Create one-row station planning summary."""
    station_counts = recommendations["recommended_station_type"].value_counts()
    flag_counts = recommendations["capital_planning_flag"].value_counts()

    summary = {
        "total_zones_evaluated": int(len(recommendations)),
        "invest_now_zones": int(flag_counts.get("Invest Now", 0)),
        "invest_with_grid_upgrade_zones": int(
            flag_counts.get("Invest with Grid Upgrade", 0)
        ),
        "monitor_zones": int(flag_counts.get("Monitor Demand", 0)),
        "total_recommended_chargers": int(recommendations["recommended_chargers"].sum()),
        "fast_charging_hubs": int(station_counts.get("Fast Charging Hub", 0)),
        "mixed_ac_dc_stations": int(station_counts.get("Mixed AC/DC Station", 0)),
        "slow_ac_stations": int(station_counts.get("Slow/AC Charging Station", 0)),
        "ac_only_grid_upgrade_needed": int(
            station_counts.get("AC Charging Only / Grid Upgrade Needed", 0)
        ),
        "top_priority_zone": recommendations.iloc[0]["zone_name"],
        "average_station_priority_score": round(
            float(recommendations["station_priority_score"].mean()), 2
        ),
    }
    return pd.DataFrame([summary])


def create_report(
    recommendations: pd.DataFrame, planning_summary: pd.DataFrame
) -> None:
    """Create Step 8 Markdown report."""
    summary = planning_summary.iloc[0]
    station_type_counts = recommendations["recommended_station_type"].value_counts()
    flag_counts = recommendations["capital_planning_flag"].value_counts()
    top_zones = recommendations.head(10)
    upgrade_zones = recommendations[
        recommendations["capital_planning_flag"] == "Invest with Grid Upgrade"
    ].head(5)

    station_type_lines = [
        f"| {station_type} | {count} |"
        for station_type, count in station_type_counts.items()
    ]
    flag_lines = [f"| {flag} | {count} |" for flag, count in flag_counts.items()]
    top_zone_lines = [
        (
            f"| {row.rank} | {row.zone_name} | {row.station_priority_score:.2f} | "
            f"{row.recommended_station_type} | {row.recommended_chargers} | "
            f"{row.grid_feasibility_label} | {row.capital_planning_flag} |"
        )
        for row in top_zones.itertuples(index=False)
    ]
    if upgrade_zones.empty:
        upgrade_lines = ["| None | - | - | - |"]
    else:
        upgrade_lines = [
            (
                f"| {row.zone_name} | {row.station_priority_score:.2f} | "
                f"{row.grid_feasibility_label} | {row.recommended_station_type} |"
            )
            for row in upgrade_zones.itertuples(index=False)
        ]

    report = f"""# Step 8 Station Location Report

## Step Name

Charging station location recommender.

## Inputs And Outputs

- Inputs used:
  - `data/processed/zones_master.csv`
  - `data/processed/grid_capacity.csv`
  - `data/processed/grid_stress_summary_by_zone.csv`
- Outputs generated:
  - `data/processed/station_location_recommendations.csv`
  - `data/processed/station_planning_summary.csv`

## Summary

- Total zones evaluated: {int(summary.total_zones_evaluated)}
- Total recommended chargers: {int(summary.total_recommended_chargers)}
- Top priority zone: {summary.top_priority_zone}
- Average station priority score: {summary.average_station_priority_score:.2f}

## Count By Recommended Station Type

| Recommended station type | Count |
|---|---:|
{chr(10).join(station_type_lines)}

## Count By Capital Planning Flag

| Capital planning flag | Count |
|---|---:|
{chr(10).join(flag_lines)}

## Top 10 Recommended Zones

| Rank | Zone | Priority score | Recommendation | Chargers | Grid feasibility | Capital flag |
|---:|---|---:|---|---:|---|---|
{chr(10).join(top_zone_lines)}

## Top 5 Zones Requiring Grid Upgrade Before Investment

| Zone | Priority score | Grid feasibility | Recommendation |
|---|---:|---|---|
{chr(10).join(upgrade_lines)}

## Notes

- Combines demand growth, charger gap, and grid feasibility.
- Prevents blindly placing chargers in overloaded areas.
- Provides explainable infrastructure planning recommendations.
- Supports BESCOM planners with actionable zone ranking.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved Step 8 report to: {REPORT_PATH}")


def print_summary(
    recommendations: pd.DataFrame, planning_summary: pd.DataFrame
) -> None:
    """Print Step 8 inspection output."""
    print(f"\nRecommendations shape: {recommendations.shape}")
    print("\nTop 10 recommended zones:")
    print(
        recommendations[
            [
                "rank",
                "zone_name",
                "station_priority_score",
                "recommended_station_type",
                "recommended_chargers",
                "grid_feasibility_label",
                "capital_planning_flag",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nTotal recommended chargers: {int(recommendations['recommended_chargers'].sum())}")
    print("\nCount by station type:")
    print(recommendations["recommended_station_type"].value_counts().to_string())
    print("\nCount by capital planning flag:")
    print(recommendations["capital_planning_flag"].value_counts().to_string())
    print("\nPlanning summary:")
    print(planning_summary.to_string(index=False))


def main() -> None:
    """Run Step 8 station location recommendation."""
    try:
        ensure_directories()
        zones, grid, stress = read_inputs()
        merged = merge_inputs(zones, grid, stress)
        recommendations = create_recommendations(merged)
        planning_summary = create_planning_summary(recommendations)

        recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
        print(f"Saved station location recommendations to: {RECOMMENDATIONS_PATH}")
        planning_summary.to_csv(PLANNING_SUMMARY_PATH, index=False)
        print(f"Saved station planning summary to: {PLANNING_SUMMARY_PATH}")
        create_report(recommendations, planning_summary)
        print_summary(recommendations, planning_summary)
        print("\nStep 8 station location recommendation complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 8 failed: {exc}") from exc


if __name__ == "__main__":
    main()
