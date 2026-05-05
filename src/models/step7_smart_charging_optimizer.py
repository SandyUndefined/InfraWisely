"""Create smart charging schedule recommendations from grid stress alerts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GRID_STRESS_PATH = Path("data/processed/grid_stress_predictions.csv")
RECOMMENDATIONS_PATH = Path("data/processed/charging_schedule_recommendations.csv")
LOAD_CURVE_PATH = Path("data/processed/load_curve_before_after.csv")
IMPACT_SUMMARY_PATH = Path("data/processed/scheduling_impact_summary.csv")
REPORT_PATH = Path("reports/step7_scheduling_optimizer_report.md")

REQUIRED_COLUMNS = {
    "zone_id",
    "zone_name",
    "hour",
    "time_period",
    "is_peak_hour",
    "transformer_capacity_kw",
    "base_grid_load_kw",
    "model_predicted_ev_load_kw",
    "model_total_load_kw",
    "model_load_ratio",
    "model_risk_level",
    "risk_score",
    "is_overloaded",
}

OFFPEAK_HOURS = [23, 0, 1, 2, 3, 4, 5]
PEAK_RISK_LEVELS = {"High", "Critical"}
OFFPEAK_ALLOWED_RISK_LEVELS = {"Low", "Medium"}


def ensure_directories() -> None:
    """Create Step 7 output directories."""
    Path("src/models").mkdir(parents=True, exist_ok=True)
    RECOMMENDATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_grid_stress(path: Path = GRID_STRESS_PATH) -> pd.DataFrame:
    """Read and validate Step 6 grid stress predictions."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}. Run Step 6 first.")

    grid_stress = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(grid_stress.columns)
    if missing_columns:
        raise ValueError(f"{path} missing required columns: {sorted(missing_columns)}")

    grid_stress = grid_stress.copy()
    grid_stress["is_peak_hour"] = grid_stress["is_peak_hour"].astype(bool)
    grid_stress["is_overloaded"] = grid_stress["is_overloaded"].astype(bool)
    print(f"Loaded grid stress predictions from: {path}")
    print(f"Input shape: {grid_stress.shape}")
    return grid_stress


def classify_risk(load_ratio: float) -> str:
    """Classify risk after optimization."""
    if load_ratio < 0.75:
        return "Low"
    if load_ratio < 0.90:
        return "Medium"
    if load_ratio <= 1.00:
        return "High"
    return "Critical"


def format_hours(hours: list[int]) -> str:
    """Format hour integers as HH:00 text."""
    return ", ".join(f"{hour:02d}:00" for hour in hours)


def initialize_load_curve(grid_stress: pd.DataFrame) -> pd.DataFrame:
    """Create the before-after load curve table before optimization."""
    load_curve = grid_stress[
        [
            "zone_id",
            "zone_name",
            "hour",
            "time_period",
            "transformer_capacity_kw",
            "base_grid_load_kw",
            "model_predicted_ev_load_kw",
            "model_total_load_kw",
            "model_load_ratio",
            "model_risk_level",
        ]
    ].copy()
    load_curve = load_curve.rename(
        columns={
            "model_predicted_ev_load_kw": "before_ev_load_kw",
            "model_total_load_kw": "before_total_load_kw",
            "model_load_ratio": "before_load_ratio",
            "model_risk_level": "before_risk_level",
        }
    )
    load_curve["after_ev_load_kw"] = load_curve["before_ev_load_kw"]
    load_curve["after_total_load_kw"] = load_curve["before_total_load_kw"]
    load_curve["after_load_ratio"] = load_curve["before_load_ratio"]
    load_curve["after_risk_level"] = load_curve["before_risk_level"]
    return load_curve


def update_after_metrics(load_curve: pd.DataFrame) -> pd.DataFrame:
    """Recalculate after-optimization totals, ratios, and risk labels."""
    load_curve["after_ev_load_kw"] = load_curve["after_ev_load_kw"].clip(lower=0).round(2)
    load_curve["after_total_load_kw"] = (
        load_curve["base_grid_load_kw"] + load_curve["after_ev_load_kw"]
    ).round(2)
    load_curve["after_load_ratio"] = (
        load_curve["after_total_load_kw"] / load_curve["transformer_capacity_kw"]
    ).round(3)
    load_curve["after_risk_level"] = load_curve["after_load_ratio"].apply(classify_risk)
    return load_curve


def allocate_shift_to_offpeak(
    load_curve: pd.DataFrame,
    zone_id: str,
    shift_load_kw: float,
) -> tuple[float, float, list[int]]:
    """Greedily allocate shifted load to off-peak hours with spare capacity."""
    zone_mask = load_curve["zone_id"] == zone_id
    offpeak_mask = (
        zone_mask
        & load_curve["hour"].isin(OFFPEAK_HOURS)
        & load_curve["after_risk_level"].isin(OFFPEAK_ALLOWED_RISK_LEVELS)
    )
    candidates = load_curve.loc[offpeak_mask].copy()
    if candidates.empty:
        return 0.0, shift_load_kw, []

    candidates["safe_capacity_limit_kw"] = candidates["transformer_capacity_kw"] * 0.90
    candidates["available_shift_capacity_kw"] = (
        candidates["safe_capacity_limit_kw"] - candidates["after_total_load_kw"]
    )
    candidates = candidates[candidates["available_shift_capacity_kw"] > 0].sort_values(
        "available_shift_capacity_kw", ascending=False
    )
    if candidates.empty:
        return 0.0, shift_load_kw, []

    remaining = shift_load_kw
    allocated_total = 0.0
    allocated_hours: list[int] = []

    for idx, candidate in candidates.iterrows():
        if remaining <= 0:
            break
        allocation = min(float(candidate["available_shift_capacity_kw"]), remaining)
        load_curve.loc[idx, "after_ev_load_kw"] += allocation
        allocated_total += allocation
        remaining -= allocation
        allocated_hours.append(int(candidate["hour"]))
        load_curve.loc[[idx]] = update_after_metrics(load_curve.loc[[idx]].copy())

    return round(allocated_total, 2), round(max(0.0, remaining), 2), allocated_hours


def create_recommendations_and_curve(
    grid_stress: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create schedule recommendations and before-after load curves."""
    load_curve = initialize_load_curve(grid_stress)
    recommendation_rows: list[dict[str, object]] = []

    for zone_id, zone_group in grid_stress.groupby("zone_id", sort=False):
        peak_rows = zone_group[
            zone_group["is_peak_hour"]
            & zone_group["model_risk_level"].isin(PEAK_RISK_LEVELS)
        ].sort_values(["model_risk_level", "risk_score"], ascending=[True, False])
        if peak_rows.empty:
            continue

        for peak_row in peak_rows.itertuples(index=False):
            shift_percent = 0.35 if peak_row.model_risk_level == "Critical" else 0.25
            shift_percent = min(shift_percent, 0.40)
            intended_shift_kw = round(float(peak_row.model_predicted_ev_load_kw) * shift_percent, 2)

            peak_mask = (load_curve["zone_id"] == zone_id) & (
                load_curve["hour"] == int(peak_row.hour)
            )
            peak_idx = load_curve.index[peak_mask][0]
            actual_shift_kw = min(
                intended_shift_kw,
                round(float(load_curve.loc[peak_idx, "after_ev_load_kw"]) * 0.40, 2),
            )

            allocated_kw, unallocated_kw, allocated_hours = allocate_shift_to_offpeak(
                load_curve, zone_id, actual_shift_kw
            )

            if allocated_kw > 0:
                load_curve.loc[peak_idx, "after_ev_load_kw"] -= allocated_kw
                load_curve.loc[[peak_idx]] = update_after_metrics(
                    load_curve.loc[[peak_idx]].copy()
                )

            if allocated_kw == 0:
                feasibility_status = "Limited Feasibility"
            elif unallocated_kw > 0:
                feasibility_status = "Partially Feasible"
            else:
                feasibility_status = "Feasible"

            offpeak_hours_text = format_hours(allocated_hours)
            if not offpeak_hours_text:
                offpeak_hours_text = "No feasible off-peak hour"

            recommendation_text = (
                f"Shift {int(shift_percent * 100)}% of EV charging from "
                f"{int(peak_row.hour):02d}:00 to off-peak hours {offpeak_hours_text}."
            )
            load_percent = float(peak_row.model_load_ratio) * 100
            explanation = (
                f"{peak_row.zone_name} at {int(peak_row.hour):02d}:00 is "
                f"{peak_row.model_risk_level} because load reaches {load_percent:.1f}% "
                "of transformer capacity. Shifting charging demand reduces peak stress "
                "while keeping off-peak load below 90% capacity where feasible."
            )

            recommendation_rows.append(
                {
                    "zone_id": peak_row.zone_id,
                    "zone_name": peak_row.zone_name,
                    "peak_hour": int(peak_row.hour),
                    "original_risk_level": peak_row.model_risk_level,
                    "original_ev_load_kw": round(float(peak_row.model_predicted_ev_load_kw), 2),
                    "original_total_load_kw": round(float(peak_row.model_total_load_kw), 2),
                    "original_load_ratio": round(float(peak_row.model_load_ratio), 3),
                    "recommended_shift_percent": round(shift_percent, 2),
                    "shifted_load_kw": round(actual_shift_kw, 2),
                    "allocated_shift_kw": round(allocated_kw, 2),
                    "unallocated_shift_kw": round(unallocated_kw, 2),
                    "recommended_offpeak_hours": offpeak_hours_text,
                    "feasibility_status": feasibility_status,
                    "expected_peak_load_reduction_kw": round(allocated_kw, 2),
                    "expected_peak_load_reduction_percent": round(
                        (allocated_kw / peak_row.model_total_load_kw) * 100, 2
                    ),
                    "recommendation_text": recommendation_text,
                    "explanation": explanation,
                }
            )

    load_curve = update_after_metrics(load_curve)
    recommendations = pd.DataFrame(
        recommendation_rows,
        columns=[
            "zone_id",
            "zone_name",
            "peak_hour",
            "original_risk_level",
            "original_ev_load_kw",
            "original_total_load_kw",
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
        ],
    )

    return recommendations, load_curve[
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
        ]
    ]


def create_impact_summary(
    recommendations: pd.DataFrame, load_curve: pd.DataFrame
) -> pd.DataFrame:
    """Calculate overall scheduling impact metrics."""
    peak_mask = load_curve["hour"].between(17, 22)
    before_peak_total = float(load_curve.loc[peak_mask, "before_total_load_kw"].sum())
    after_peak_total = float(load_curve.loc[peak_mask, "after_total_load_kw"].sum())
    peak_reduction = before_peak_total - after_peak_total

    if recommendations.empty:
        total_shifted = total_allocated = total_unallocated = 0.0
        feasible = partial = limited = 0
    else:
        total_shifted = float(recommendations["shifted_load_kw"].sum())
        total_allocated = float(recommendations["allocated_shift_kw"].sum())
        total_unallocated = float(recommendations["unallocated_shift_kw"].sum())
        feasible = int((recommendations["feasibility_status"] == "Feasible").sum())
        partial = int((recommendations["feasibility_status"] == "Partially Feasible").sum())
        limited = int((recommendations["feasibility_status"] == "Limited Feasibility").sum())

    summary = {
        "total_shifted_load_kw": round(total_shifted, 2),
        "total_allocated_shift_kw": round(total_allocated, 2),
        "total_unallocated_shift_kw": round(total_unallocated, 2),
        "before_peak_total_load_kw": round(before_peak_total, 2),
        "after_peak_total_load_kw": round(after_peak_total, 2),
        "peak_load_reduction_kw": round(peak_reduction, 2),
        "peak_load_reduction_percent": round(
            (peak_reduction / before_peak_total) * 100 if before_peak_total else 0.0, 2
        ),
        "before_critical_hours": int((load_curve["before_risk_level"] == "Critical").sum()),
        "after_critical_hours": int((load_curve["after_risk_level"] == "Critical").sum()),
        "before_high_hours": int((load_curve["before_risk_level"] == "High").sum()),
        "after_high_hours": int((load_curve["after_risk_level"] == "High").sum()),
        "before_overloaded_hours": int((load_curve["before_load_ratio"] > 1.0).sum()),
        "after_overloaded_hours": int((load_curve["after_load_ratio"] > 1.0).sum()),
        "feasible_recommendations": feasible,
        "partial_recommendations": partial,
        "limited_recommendations": limited,
    }
    return pd.DataFrame([summary])


def create_report(
    recommendations: pd.DataFrame,
    load_curve: pd.DataFrame,
    impact_summary: pd.DataFrame,
) -> None:
    """Create Step 7 optimizer report."""
    impact = impact_summary.iloc[0]
    top_recs = recommendations.nlargest(10, "allocated_shift_kw") if not recommendations.empty else recommendations
    zone_reductions = (
        load_curve[load_curve["hour"].between(17, 22)]
        .groupby("zone_name")
        .agg(
            before_peak=("before_total_load_kw", "sum"),
            after_peak=("after_total_load_kw", "sum"),
        )
        .reset_index()
    )
    zone_reductions["peak_reduction_kw"] = (
        zone_reductions["before_peak"] - zone_reductions["after_peak"]
    ).round(2)
    zone_reductions["peak_reduction_percent"] = (
        zone_reductions["peak_reduction_kw"] / zone_reductions["before_peak"] * 100
    ).round(2)
    top_zones = zone_reductions.nlargest(10, "peak_reduction_kw")

    rec_lines = [
        (
            f"| {row.zone_name} | {row.peak_hour}:00 | {row.original_risk_level} | "
            f"{row.allocated_shift_kw:.2f} | {row.unallocated_shift_kw:.2f} | "
            f"{row.feasibility_status} |"
        )
        for row in top_recs.itertuples(index=False)
    ]
    zone_lines = [
        (
            f"| {row.zone_name} | {row.peak_reduction_kw:.2f} | "
            f"{row.peak_reduction_percent:.2f}% |"
        )
        for row in top_zones.itertuples(index=False)
    ]

    report = f"""# Step 7 Scheduling Optimizer Report

## Step Name

Smart charging schedule optimizer.

## Inputs And Outputs

- Input used: `data/processed/grid_stress_predictions.csv`
- Outputs generated:
  - `data/processed/charging_schedule_recommendations.csv`
  - `data/processed/load_curve_before_after.csv`
  - `data/processed/scheduling_impact_summary.csv`

## Summary

- Number of recommendations generated: {len(recommendations)}
- Total shifted load: {impact.total_shifted_load_kw:.2f} kW
- Total allocated shifted load: {impact.total_allocated_shift_kw:.2f} kW
- Peak load reduction percentage: {impact.peak_load_reduction_percent:.2f}%
- Critical hours before vs after: {int(impact.before_critical_hours)} -> {int(impact.after_critical_hours)}
- High hours before vs after: {int(impact.before_high_hours)} -> {int(impact.after_high_hours)}
- Overloaded hours before vs after: {int(impact.before_overloaded_hours)} -> {int(impact.after_overloaded_hours)}

## Top 10 Recommendations By Shifted Load

| Zone | Peak hour | Original risk | Allocated shift kW | Unallocated shift kW | Feasibility |
|---|---:|---|---:|---:|---|
{chr(10).join(rec_lines)}

## Top 10 Zones By Peak Reduction

| Zone | Peak reduction kW | Peak reduction percent |
|---|---:|---:|
{chr(10).join(zone_lines)}

## Notes

- Translates grid risk into actionable charging schedules.
- Reduces evening peak stress without changing BESCOM systems.
- Respects transformer capacity limits.
- Provides explainable operator recommendations.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved Step 7 report to: {REPORT_PATH}")


def print_summary(
    recommendations: pd.DataFrame,
    impact_summary: pd.DataFrame,
) -> None:
    """Print useful Step 7 inspection output."""
    impact = impact_summary.iloc[0]
    print(f"\nNumber of recommendations: {len(recommendations)}")
    print(f"Total shifted load: {impact.total_shifted_load_kw:.2f} kW")
    print(f"Total allocated shift: {impact.total_allocated_shift_kw:.2f} kW")
    print(f"Total unallocated shift: {impact.total_unallocated_shift_kw:.2f} kW")
    print(f"Peak load reduction percent: {impact.peak_load_reduction_percent:.2f}%")
    print(
        "Critical hours before vs after: "
        f"{int(impact.before_critical_hours)} -> {int(impact.after_critical_hours)}"
    )
    print(
        "Overloaded hours before vs after: "
        f"{int(impact.before_overloaded_hours)} -> {int(impact.after_overloaded_hours)}"
    )
    print("\nFirst 10 recommendations:")
    if recommendations.empty:
        print("No recommendations generated.")
    else:
        print(recommendations.head(10).to_string(index=False))
    print("\nImpact summary:")
    print(impact_summary.to_string(index=False))


def main() -> None:
    """Run Step 7 smart charging optimization."""
    try:
        ensure_directories()
        grid_stress = read_grid_stress()
        recommendations, load_curve = create_recommendations_and_curve(grid_stress)
        impact_summary = create_impact_summary(recommendations, load_curve)

        recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
        print(f"Saved charging schedule recommendations to: {RECOMMENDATIONS_PATH}")
        load_curve.to_csv(LOAD_CURVE_PATH, index=False)
        print(f"Saved before-after load curve to: {LOAD_CURVE_PATH}")
        impact_summary.to_csv(IMPACT_SUMMARY_PATH, index=False)
        print(f"Saved scheduling impact summary to: {IMPACT_SUMMARY_PATH}")
        create_report(recommendations, load_curve, impact_summary)
        print_summary(recommendations, impact_summary)
        print("\nStep 7 smart charging optimization complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 7 failed: {exc}") from exc


if __name__ == "__main__":
    main()
