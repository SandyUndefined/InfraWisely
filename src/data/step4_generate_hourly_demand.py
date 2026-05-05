"""Generate hourly synthetic EV charging demand for GridCharge AI zones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ZONES_MASTER_PATH = Path("data/processed/zones_master.csv")
GRID_CAPACITY_PATH = Path("data/processed/grid_capacity.csv")
HOURLY_DEMAND_PATH = Path("data/processed/hourly_demand.csv")
REPORT_PATH = Path("reports/step4_hourly_demand_report.md")
RANDOM_SEED = 42


def ensure_directories() -> None:
    """Create required output directories."""
    HOURLY_DEMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Step 2 and Step 3 processed datasets."""
    if not ZONES_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {ZONES_MASTER_PATH}. Run Step 2 first."
        )
    if not GRID_CAPACITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {GRID_CAPACITY_PATH}. Run Step 3 first."
        )

    zones = pd.read_csv(ZONES_MASTER_PATH)
    grid = pd.read_csv(GRID_CAPACITY_PATH)

    zone_required = {
        "zone_id",
        "zone_name",
        "zone_type",
        "ev_count_estimate",
        "traffic_score",
        "existing_chargers",
        "demand_growth_rate",
    }
    grid_required = {
        "zone_id",
        "zone_name",
        "base_load_kw",
        "transformer_capacity_kw",
    }
    missing_zone = zone_required.difference(zones.columns)
    missing_grid = grid_required.difference(grid.columns)
    if missing_zone:
        raise ValueError(f"{ZONES_MASTER_PATH} missing columns: {sorted(missing_zone)}")
    if missing_grid:
        raise ValueError(f"{GRID_CAPACITY_PATH} missing columns: {sorted(missing_grid)}")

    print(f"Loaded zones master from: {ZONES_MASTER_PATH}")
    print(f"Loaded grid capacity from: {GRID_CAPACITY_PATH}")
    return zones, grid


def merge_inputs(zones: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Merge zone features with grid capacity constraints."""
    grid_fields = [
        "zone_id",
        "zone_name",
        "base_load_kw",
        "transformer_capacity_kw",
    ]
    merged = zones.merge(
        grid[grid_fields],
        on=["zone_id", "zone_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(zones):
        raise ValueError(
            f"Merged row count {len(merged)} does not match zones row count {len(zones)}."
        )

    print(f"Merged zone-grid rows: {len(merged)}")
    return merged


def time_period_for_hour(hour: int) -> str:
    """Return a readable time-period label for an hour of day."""
    if 0 <= hour <= 5:
        return "Night"
    if 6 <= hour <= 9:
        return "Morning"
    if 10 <= hour <= 16:
        return "Day"
    if 17 <= hour <= 22:
        return "Evening Peak"
    return "Late Night"


def base_grid_multiplier_range(hour: int) -> tuple[float, float]:
    """Return base grid hourly load multiplier range."""
    if 0 <= hour <= 5:
        return 0.65, 0.85
    if 6 <= hour <= 9:
        return 0.85, 1.05
    if 10 <= hour <= 16:
        return 0.90, 1.10
    if 17 <= hour <= 22:
        return 1.05, 1.25
    return 0.75, 0.90


def ev_multiplier_range(zone_type: str, hour: int) -> tuple[float, float]:
    """Return zone-type-specific EV charging behavior multiplier range."""
    if zone_type == "residential":
        if 0 <= hour <= 5:
            return 0.55, 0.85
        if 6 <= hour <= 9:
            return 0.75, 1.05
        if 10 <= hour <= 16:
            return 0.65, 0.95
        if 17 <= hour <= 22:
            return 1.65, 2.40
        return 1.05, 1.35

    if zone_type == "commercial":
        if 0 <= hour <= 5:
            return 0.30, 0.55
        if 6 <= hour <= 9:
            return 0.75, 1.05
        if 10 <= hour <= 17:
            return 1.45, 2.10
        if 18 <= hour <= 21:
            return 0.95, 1.35
        return 0.45, 0.75

    if zone_type == "mixed":
        if 0 <= hour <= 5:
            return 0.45, 0.75
        if 6 <= hour <= 9:
            return 0.85, 1.15
        if 10 <= hour <= 16:
            return 1.10, 1.55
        if 17 <= hour <= 22:
            return 1.45, 2.15
        return 0.80, 1.10

    if zone_type == "industrial":
        if 0 <= hour <= 5:
            return 0.45, 0.75
        if 6 <= hour <= 11:
            return 1.35, 1.90
        if 12 <= hour <= 15:
            return 0.90, 1.20
        if 16 <= hour <= 20:
            return 1.25, 1.75
        return 0.55, 0.85

    raise ValueError(f"Unsupported zone_type: {zone_type}")


def classify_risk(load_ratio_after_ev: float) -> str:
    """Classify grid stress risk from total load ratio after EV demand."""
    if load_ratio_after_ev < 0.75:
        return "Low"
    if load_ratio_after_ev < 0.90:
        return "Medium"
    if load_ratio_after_ev <= 1.00:
        return "High"
    return "Critical"


def generate_hourly_demand(zone_grid: pd.DataFrame) -> pd.DataFrame:
    """Generate 24 hourly demand rows for every zone."""
    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for zone in zone_grid.itertuples(index=False):
        ev_base_load_kw = (
            (zone.ev_count_estimate * 0.035)
            + (zone.traffic_score * 1.15)
            + (zone.existing_chargers * 6)
            + (zone.demand_growth_rate * 250)
        )

        for hour in range(24):
            base_low, base_high = base_grid_multiplier_range(hour)
            ev_low, ev_high = ev_multiplier_range(zone.zone_type, hour)

            base_grid_load_kw = zone.base_load_kw * rng.uniform(base_low, base_high)
            predicted_ev_load_kw = ev_base_load_kw * rng.uniform(ev_low, ev_high)
            predicted_ev_load_kw *= rng.uniform(0.95, 1.08)
            predicted_ev_load_kw = max(0.0, predicted_ev_load_kw)

            total_load_kw = base_grid_load_kw + predicted_ev_load_kw
            load_ratio_after_ev = total_load_kw / zone.transformer_capacity_kw

            records.append(
                {
                    "zone_id": zone.zone_id,
                    "zone_name": zone.zone_name,
                    "hour": hour,
                    "time_period": time_period_for_hour(hour),
                    "is_peak_hour": 17 <= hour <= 22,
                    "zone_type": zone.zone_type,
                    "ev_count_estimate": int(zone.ev_count_estimate),
                    "traffic_score": int(zone.traffic_score),
                    "existing_chargers": int(zone.existing_chargers),
                    "demand_growth_rate": round(float(zone.demand_growth_rate), 3),
                    "transformer_capacity_kw": int(zone.transformer_capacity_kw),
                    "base_grid_load_kw": round(float(base_grid_load_kw), 2),
                    "predicted_ev_load_kw": round(float(predicted_ev_load_kw), 2),
                    "total_load_kw": round(float(total_load_kw), 2),
                    "load_ratio_after_ev": round(float(load_ratio_after_ev), 3),
                    "risk_level": classify_risk(load_ratio_after_ev),
                }
            )

    return pd.DataFrame(
        records,
        columns=[
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
            "predicted_ev_load_kw",
            "total_load_kw",
            "load_ratio_after_ev",
            "risk_level",
        ],
    )


def create_report(hourly_demand: pd.DataFrame) -> None:
    """Create Step 4 hourly demand report."""
    risk_counts = hourly_demand["risk_level"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"], fill_value=0
    )
    top_risk = hourly_demand.nlargest(10, "load_ratio_after_ev")[
        [
            "zone_name",
            "hour",
            "time_period",
            "predicted_ev_load_kw",
            "total_load_kw",
            "load_ratio_after_ev",
            "risk_level",
        ]
    ]
    peak_avg = hourly_demand.loc[
        hourly_demand["is_peak_hour"], "predicted_ev_load_kw"
    ].mean()
    off_peak_avg = hourly_demand.loc[
        ~hourly_demand["is_peak_hour"], "predicted_ev_load_kw"
    ].mean()

    risk_lines = [f"| {risk} | {count} |" for risk, count in risk_counts.items()]
    top_risk_lines = [
        (
            f"| {row.zone_name} | {row.hour} | {row.time_period} | "
            f"{row.predicted_ev_load_kw:.2f} | {row.total_load_kw:.2f} | "
            f"{row.load_ratio_after_ev:.3f} | {row.risk_level} |"
        )
        for row in top_risk.itertuples(index=False)
    ]

    report = f"""# Step 4 Hourly Demand Report

## Step Name

Generate synthetic hourly EV charging demand and grid stress labels.

## Inputs And Outputs

- Inputs used:
  - `data/processed/zones_master.csv`
  - `data/processed/grid_capacity.csv`
- Output generated:
  - `data/processed/hourly_demand.csv`

## Summary

- Number of zones: {hourly_demand["zone_id"].nunique()}
- Number of hourly rows: {len(hourly_demand)}
- Average predicted EV load: {hourly_demand["predicted_ev_load_kw"].mean():.2f} kW
- Maximum predicted EV load: {hourly_demand["predicted_ev_load_kw"].max():.2f} kW
- Average total load: {hourly_demand["total_load_kw"].mean():.2f} kW
- Peak hour average EV load: {peak_avg:.2f} kW
- Off-peak average EV load: {off_peak_avg:.2f} kW

## Risk Level Counts

| Risk level | Count |
|---|---:|
{chr(10).join(risk_lines)}

## Top 10 Highest-Risk Zone-Hour Combinations

| Zone | Hour | Period | EV load kW | Total load kW | Load ratio | Risk |
|---|---:|---|---:|---:|---:|---|
{chr(10).join(top_risk_lines)}

## Notes For ML

- `hourly_demand.csv` becomes the training table for demand forecasting.
- `risk_level` becomes a target/label for grid stress classification.
- `hour`, `zone_type`, `traffic_score`, EV count, and existing chargers become demand prediction features.
- `load_ratio_after_ev` enables explainable grid stress scoring.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved Step 4 report to: {REPORT_PATH}")


def print_summary(hourly_demand: pd.DataFrame) -> None:
    """Print useful Step 4 inspection outputs."""
    top_risk = hourly_demand.nlargest(10, "load_ratio_after_ev")[
        [
            "zone_name",
            "hour",
            "time_period",
            "total_load_kw",
            "load_ratio_after_ev",
            "risk_level",
        ]
    ]
    peak_avg = hourly_demand.loc[
        hourly_demand["is_peak_hour"], "predicted_ev_load_kw"
    ].mean()
    off_peak_avg = hourly_demand.loc[
        ~hourly_demand["is_peak_hour"], "predicted_ev_load_kw"
    ].mean()

    print(f"\nhourly_demand shape: {hourly_demand.shape}")
    print("\nFirst 10 rows:")
    print(hourly_demand.head(10).to_string(index=False))
    print("\nrisk_level value counts:")
    print(hourly_demand["risk_level"].value_counts().to_string())
    print("\nTop 10 highest load_ratio_after_ev rows:")
    print(top_risk.to_string(index=False))
    print(f"\nPeak hour average EV load: {peak_avg:.2f} kW")
    print(f"Off-peak average EV load: {off_peak_avg:.2f} kW")


def main() -> None:
    """Run Step 4 hourly demand generation."""
    try:
        ensure_directories()
        zones, grid = read_inputs()
        zone_grid = merge_inputs(zones, grid)
        hourly_demand = generate_hourly_demand(zone_grid)
        hourly_demand.to_csv(HOURLY_DEMAND_PATH, index=False)
        print(f"Saved hourly demand dataset to: {HOURLY_DEMAND_PATH}")
        create_report(hourly_demand)
        print_summary(hourly_demand)
        print("\nStep 4 hourly demand pipeline complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 4 failed: {exc}") from exc


if __name__ == "__main__":
    main()
