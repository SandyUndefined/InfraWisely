"""Create final AI/ML demo summary artifacts for GridCharge AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_METRICS_PATH = Path("data/dashboard/summary_metrics.json")
STATION_RECOMMENDATIONS_PATH = Path("data/dashboard/station_recommendations.json")
GRID_STRESS_ALERTS_PATH = Path("data/dashboard/grid_stress_alerts.json")
SCHEDULING_RECOMMENDATIONS_PATH = Path("data/dashboard/scheduling_recommendations.json")
MODEL_METRICS_PATH = Path("models/ev_demand_model_metrics.json")
SCHEDULING_IMPACT_PATH = Path("data/processed/scheduling_impact_summary.csv")
STATION_PLANNING_PATH = Path("data/processed/station_planning_summary.csv")
STEP5_REPORT_PATH = Path("reports/step5_model_training_report.md")

FINAL_SUMMARY_PATH = Path("reports/final_ai_ml_summary.md")
DEMO_STORYLINE_PATH = Path("data/dashboard/demo_storyline.json")
MODEL_EXPLAINABILITY_PATH = Path("data/dashboard/model_explainability_summary.json")

REQUIRED_FILES = [
    SUMMARY_METRICS_PATH,
    STATION_RECOMMENDATIONS_PATH,
    GRID_STRESS_ALERTS_PATH,
    SCHEDULING_RECOMMENDATIONS_PATH,
    MODEL_METRICS_PATH,
    SCHEDULING_IMPACT_PATH,
    STATION_PLANNING_PATH,
    STEP5_REPORT_PATH,
]


def ensure_directories() -> None:
    """Create output folders."""
    FINAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_STORYLINE_PATH.parent.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    """Raise clear errors for missing required files."""
    missing_files = [path for path in REQUIRED_FILES if not path.exists()]
    if missing_files:
        missing = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing required Step 10 input files:\n{missing}")


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write indented JSON."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_demo_storyline(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Create ordered demo storyline sections."""
    return [
        {
            "step": 1,
            "title": "EV Demand Forecasting",
            "message": (
                "GridCharge AI predicts hourly EV charging demand for each Bengaluru "
                "zone using traffic, EV count, charger availability, time period, and "
                "grid context."
            ),
            "key_metric": f"Model R2: {summary['model_r2_score']}",
        },
        {
            "step": 2,
            "title": "Grid Stress Detection",
            "message": (
                "Predicted EV demand is combined with transformer capacity to identify "
                "high-risk and critical grid stress hours."
            ),
            "key_metric": f"Highest risk zone: {summary['highest_risk_zone']}",
        },
        {
            "step": 3,
            "title": "Smart Charging Optimization",
            "message": (
                "The optimizer shifts charging from overloaded evening peak hours to "
                "safer off-peak windows while respecting transformer capacity limits."
            ),
            "key_metric": (
                "Critical hours reduced from "
                f"{summary['critical_hours_before']} to {summary['critical_hours_after']}"
            ),
        },
        {
            "step": 4,
            "title": "Infrastructure Planning",
            "message": (
                "The recommender ranks zones for new charging stations using demand, "
                "growth, charger gap, and grid feasibility."
            ),
            "key_metric": (
                f"{summary['total_recommended_chargers']} chargers recommended across "
                "priority zones"
            ),
        },
    ]


def create_model_explainability_summary(
    summary: dict[str, Any],
    model_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Create compact model explainability JSON."""
    return {
        "model_name": model_metrics["model_name"],
        "target_variable": model_metrics["target_column"],
        "input_features": model_metrics["feature_columns"],
        "model_r2_score": round(float(model_metrics["r2_score"]), 2),
        "mae": round(float(model_metrics["mae"]), 2),
        "rmse": round(float(model_metrics["rmse"]), 2),
        "baseline_comparison": {
            "baseline_mae": round(float(model_metrics["baseline_mae"]), 2),
            "baseline_rmse": round(float(model_metrics["baseline_rmse"]), 2),
            "mae_improvement_percent": model_metrics[
                "improvement_over_baseline_mae_percent"
            ],
            "rmse_improvement_percent": model_metrics[
                "improvement_over_baseline_rmse_percent"
            ],
        },
        "why_model_is_useful": (
            "The model predicts hourly EV charging demand by zone, creating the demand "
            "signal needed for grid stress detection and scheduling decisions."
        ),
        "grid_constraints_used": [
            "transformer_capacity_kw",
            "base_grid_load_kw",
            "current_peak_load_kw",
            "spare_capacity_kw",
            "grid_health_score",
        ],
        "recommendation_logic": {
            "grid_stress": (
                "Model-predicted EV load is added to base grid load and compared with "
                "transformer capacity."
            ),
            "scheduling": (
                "High and Critical evening load is shifted to off-peak hours only where "
                "the post-shift load stays below 90% of transformer capacity."
            ),
            "station_planning": (
                "Zones are ranked using EV demand, growth, charger gap, traffic, grid "
                "spare capacity, grid health, stress, and overload counts."
            ),
        },
        "explainability_examples": [
            {
                "type": "Demand prediction explanation",
                "message": (
                    "Evening peak, zone type, EV count estimate, traffic score, and base "
                    "grid load explain why predicted demand rises in residential and "
                    "mixed zones."
                ),
            },
            {
                "type": "Grid risk explanation",
                "message": (
                    f"{summary['highest_risk_zone']} is highlighted because predicted "
                    "total load reaches a high share of transformer capacity during "
                    "critical hours."
                ),
            },
            {
                "type": "Scheduling explanation",
                "message": (
                    f"Critical hours reduce from {summary['critical_hours_before']} to "
                    f"{summary['critical_hours_after']} after shifting flexible charging "
                    "to feasible off-peak windows."
                ),
            },
            {
                "type": "Station planning explanation",
                "message": (
                    f"{summary['top_priority_station_zone']} is prioritized because the "
                    "zone combines demand need with grid-aware feasibility for new "
                    "charging infrastructure."
                ),
            },
        ],
    }


def create_final_summary_markdown(
    summary: dict[str, Any],
    model_metrics: dict[str, Any],
    scheduling_impact: pd.Series,
    station_planning: pd.Series,
) -> str:
    """Create polished final AI/ML summary markdown."""
    return f"""# GridCharge AI - Final AI/ML Summary

## 1. AI/ML Role in the System

The AI/ML module provides the intelligence layer for GridCharge AI. It handles data preparation, EV demand forecasting, grid stress scoring, smart charging optimization, station location planning, and dashboard-ready JSON exports for integration by the frontend/backend team.

## 2. Data Sources and Data Strategy

- The official BESCOM charger PDF was used to extract existing EV charger infrastructure.
- Public/synthetic zone-level features were created for 20 Bengaluru zones.
- Transformer and feeder constraints were synthetically masked because real utility grid data is sensitive.
- Hourly EV demand was generated using realistic charging behavior by zone type, including residential evening peaks, commercial daytime peaks, mixed-use dual peaks, and industrial shift-based patterns.

## 3. Feature Engineering

The pipeline creates ML-ready features including:

- `hour`
- `time_period`
- `is_peak_hour`
- `zone_type`
- EV count estimate
- traffic score
- existing chargers
- demand growth rate
- transformer capacity
- base grid load
- charger gap score
- spare grid capacity

## 4. Demand Prediction Model

- Model type: Random Forest Regressor
- Target: `predicted_ev_load_kw`
- R2 score: {float(model_metrics["r2_score"]):.4f}
- MAE: {float(model_metrics["mae"]):.2f}
- RMSE: {float(model_metrics["rmse"]):.2f}
- Baseline MAE: {float(model_metrics["baseline_mae"]):.2f}
- Baseline RMSE: {float(model_metrics["baseline_rmse"]):.2f}
- MAE improvement over baseline: {float(model_metrics["improvement_over_baseline_mae_percent"]):.2f}%
- RMSE improvement over baseline: {float(model_metrics["improvement_over_baseline_rmse_percent"]):.2f}%

## 5. Grid Stress Scoring

Model-predicted EV demand is added to base grid load, and the resulting total load is compared against transformer capacity. Each zone-hour is classified as Low, Medium, High, or Critical risk. The output becomes operator-facing alerts with recommended actions and clear reasons.

- Highest risk zone: {summary["highest_risk_zone"]}
- Critical hours before optimization: {summary["critical_hours_before"]}
- Overloaded hours before optimization: {summary["overloaded_hours_before"]}

## 6. Smart Charging Optimization

The optimizer shifts High and Critical evening peak loads to off-peak hours. Off-peak allocation respects a 90% transformer safe capacity limit, and each recommendation is marked as feasible, partially feasible, or limited feasibility.

- Peak load reduction: {float(scheduling_impact["peak_load_reduction_percent"]):.2f}%
- Critical hours before/after: {int(scheduling_impact["before_critical_hours"])} -> {int(scheduling_impact["after_critical_hours"])}
- Overloaded hours before/after: {int(scheduling_impact["before_overloaded_hours"])} -> {int(scheduling_impact["after_overloaded_hours"])}
- Total scheduling recommendations: {summary["total_scheduling_recommendations"]}

## 7. Station Location Recommendation

The station recommender ranks zones using EV demand, demand growth, charger gap, traffic, grid stress, overload hours, spare capacity, and grid health. Fast chargers are recommended only where grid feasibility is sufficient. Constrained zones are marked for AC charging, monitoring, or future grid upgrade.

- Total recommended chargers: {int(station_planning["total_recommended_chargers"])}
- Fast charging hubs: {int(station_planning["fast_charging_hubs"])}
- Invest now zones: {int(station_planning["invest_now_zones"])}
- Top priority station zone: {station_planning["top_priority_zone"]}

## 8. Dashboard/API Outputs

- `summary_metrics.json` powers KPI cards.
- `demand_forecast.json` powers demand charts.
- `grid_stress_alerts.json` powers risk alert tables.
- `scheduling_recommendations.json` powers charging shift recommendations.
- `load_curve_before_after.json` powers before-vs-after graphs.
- `station_recommendations.json` powers infrastructure planning tables.
- `map_zones.json` powers map markers and heatmaps.

## 9. Explainability and Actionability

Every risk alert includes a reason. Every schedule recommendation includes a shift percentage and target off-peak window. Every station recommendation includes grid feasibility, recommended charger type, charger count, and a capital planning flag.

## 10. Limitations and Future Improvements

- Real BESCOM feeder and transformer data can replace synthetic grid constraints.
- Real charging session logs can improve demand forecasting.
- User adoption of off-peak charging can be modeled later.
- Tariff and dynamic pricing can be added later.
- Renewable availability and battery storage can be integrated.

## 11. Final Prototype Impact

GridCharge AI demonstrates how BESCOM can move from reactive EV infrastructure planning to predictive, grid-aware EV charging optimization without modifying existing distribution systems.
"""


def print_summary(
    summary: dict[str, Any],
    storyline: list[dict[str, Any]],
) -> None:
    """Print created files and key final metrics."""
    print("\nCreated files:")
    print(f"- {FINAL_SUMMARY_PATH}")
    print(f"- {DEMO_STORYLINE_PATH}")
    print(f"- {MODEL_EXPLAINABILITY_PATH}")

    print("\nKey summary metrics:")
    for key in [
        "model_r2_score",
        "peak_load_reduction_percent",
        "critical_hours_before",
        "critical_hours_after",
        "overloaded_hours_before",
        "overloaded_hours_after",
        "total_recommended_chargers",
        "top_priority_station_zone",
        "highest_risk_zone",
    ]:
        print(f"- {key}: {summary[key]}")

    print("\nDemo storyline sections:")
    for section in storyline:
        print(f"{section['step']}. {section['title']} - {section['key_metric']}")


def main() -> None:
    """Run Step 10 final demo summary creation."""
    try:
        ensure_directories()
        validate_inputs()

        summary = read_json(SUMMARY_METRICS_PATH)
        model_metrics = read_json(MODEL_METRICS_PATH)
        scheduling_impact = pd.read_csv(SCHEDULING_IMPACT_PATH).iloc[0]
        station_planning = pd.read_csv(STATION_PLANNING_PATH).iloc[0]
        STEP5_REPORT_PATH.read_text(encoding="utf-8")

        storyline = create_demo_storyline(summary)
        explainability = create_model_explainability_summary(summary, model_metrics)
        final_summary = create_final_summary_markdown(
            summary, model_metrics, scheduling_impact, station_planning
        )

        write_json(DEMO_STORYLINE_PATH, storyline)
        write_json(MODEL_EXPLAINABILITY_PATH, explainability)
        FINAL_SUMMARY_PATH.write_text(final_summary, encoding="utf-8")

        print_summary(summary, storyline)
        print("\nStep 10 final demo summary creation complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 10 failed: {exc}") from exc


if __name__ == "__main__":
    main()
